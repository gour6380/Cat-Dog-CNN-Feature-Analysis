"""Official Oxford-IIIT Pet ingestion and deterministic sample handling."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from src.config import ExperimentConfig
from src.io_utils import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file, utc_now


class DataError(RuntimeError):
    """Dataset provenance or split invariants failed."""


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    image_path: str
    label: int
    species: int
    breed_name: str
    official_split: Literal["trainval", "test"]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _annotation_root(data_root: Path) -> Path:
    return data_root / "oxford-iiit-pet" / "annotations"


def _image_root(data_root: Path) -> Path:
    return data_root / "oxford-iiit-pet" / "images"


def download_official_dataset(data_root: Path) -> None:
    """Use Torchvision's official URLs and integrity-aware extraction."""

    data_root.mkdir(parents=True, exist_ok=True)
    OxfordIIITPet(root=data_root, split="trainval", target_types="category", download=True)
    OxfordIIITPet(root=data_root, split="test", target_types="category", download=True)


def _breed_name(sample_id: str) -> str:
    return re.sub(r"_\d+$", "", sample_id)


def parse_official_split(data_root: Path, split: Literal["trainval", "test"]) -> list[SampleRecord]:
    annotation = _annotation_root(data_root) / f"{split}.txt"
    if not annotation.is_file():
        raise DataError(f"missing official annotation: {annotation}")
    images = _image_root(data_root)
    records: list[SampleRecord] = []
    for line_number, line in enumerate(annotation.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) != 4:
            raise DataError(f"invalid {annotation.name}:{line_number}: expected four fields")
        sample_id, class_text, species_text, _breed_text = fields
        image_path = images / f"{sample_id}.jpg"
        if not image_path.is_file():
            raise DataError(f"annotation references missing image: {image_path}")
        records.append(
            SampleRecord(
                sample_id=sample_id,
                image_path=str(image_path.resolve()),
                label=int(class_text) - 1,
                species=int(species_text) - 1,
                breed_name=_breed_name(sample_id),
                official_split=split,
            )
        )
    return records


def _rank(seed: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def stratified_train_calibration_split(
    records: Sequence[SampleRecord], train_fraction: float, seed: str
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Split each breed by a filename-derived stable hash."""

    grouped: dict[int, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)
    training: list[SampleRecord] = []
    calibration: list[SampleRecord] = []
    for label in sorted(grouped):
        ranked = sorted(
            grouped[label], key=lambda item: (_rank(seed, item.sample_id), item.sample_id)
        )
        calibration_count = max(1, int(round(len(ranked) * (1.0 - train_fraction))))
        calibration.extend(ranked[:calibration_count])
        training.extend(ranked[calibration_count:])
    return sorted(training, key=lambda item: item.sample_id), sorted(
        calibration, key=lambda item: item.sample_id
    )


def stratified_hash_selection(
    records: Sequence[SampleRecord], per_class: int, seed: str
) -> list[SampleRecord]:
    grouped: dict[int, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)
    selected: list[SampleRecord] = []
    for label in sorted(grouped):
        candidates = sorted(
            grouped[label],
            key=lambda item: (_rank(f"{seed}:{label}", item.sample_id), item.sample_id),
        )
        if len(candidates) < per_class:
            raise DataError(
                f"class {label} has only {len(candidates)} samples; requires {per_class}"
            )
        selected.extend(candidates[:per_class])
    return sorted(selected, key=lambda item: (item.label, item.sample_id))


def validate_records(
    trainval: Sequence[SampleRecord], test: Sequence[SampleRecord], config: ExperimentConfig
) -> dict[str, Any]:
    expected_trainval = config.integer("dataset", "expected_trainval")
    expected_test = config.integer("dataset", "expected_test")
    expected_classes = config.integer("dataset", "classes")
    if len(trainval) != expected_trainval or len(test) != expected_test:
        raise DataError(
            f"official split counts changed: trainval={len(trainval)}, test={len(test)}; "
            f"expected {expected_trainval}/{expected_test}"
        )
    train_ids = {record.sample_id for record in trainval}
    test_ids = {record.sample_id for record in test}
    if len(train_ids) != len(trainval) or len(test_ids) != len(test):
        raise DataError("duplicate sample identifiers detected")
    overlap = train_ids & test_ids
    if overlap:
        raise DataError(f"official trainval/test overlap: {sorted(overlap)[:3]}")
    expected_labels = set(range(expected_classes))
    train_labels = {record.label for record in trainval}
    test_labels = {record.label for record in test}
    if train_labels != expected_labels or test_labels != expected_labels:
        raise DataError("not all 37 class identifiers occur in both official partitions")
    label_to_name: dict[int, str] = {}
    for record in [*trainval, *test]:
        previous = label_to_name.setdefault(record.label, record.breed_name)
        if previous != record.breed_name:
            raise DataError(f"label {record.label} maps to multiple breed names")
    return {
        "trainval_count": len(trainval),
        "test_count": len(test),
        "class_count": len(expected_labels),
        "trainval_per_class": dict(sorted(Counter(item.label for item in trainval).items())),
        "test_per_class": dict(sorted(Counter(item.label for item in test).items())),
        "label_to_name": {str(key): value for key, value in sorted(label_to_name.items())},
    }


def dataset_content_hash(records: Sequence[SampleRecord]) -> str:
    """Hash every official image in deterministic split/id order."""

    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: (item.official_split, item.sample_id)):
        path = Path(record.image_path)
        digest.update(record.sample_id.encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def split_payload_hash(payload: dict[str, Any]) -> str:
    """Hash scientific split content while excluding mutable recording metadata."""

    stable = {
        key: value for key, value in payload.items() if key not in {"created_at", "split_sha256"}
    }
    return sha256_bytes(canonical_json_bytes(stable))


def create_data_manifests(config: ExperimentConfig) -> dict[str, Any]:
    data_root = config.project_path("data")
    trainval = parse_official_split(data_root, "trainval")
    test = parse_official_split(data_root, "test")
    integrity = validate_records(trainval, test, config)
    training, calibration = stratified_train_calibration_split(
        trainval,
        config.number("dataset", "train_fraction"),
        config.value("dataset", "split_seed", str),
    )
    train_ids = {item.sample_id for item in training}
    calibration_ids = {item.sample_id for item in calibration}
    if train_ids & calibration_ids or train_ids | calibration_ids != {
        item.sample_id for item in trainval
    }:
        raise DataError("derived training/calibration partitions are not disjoint and exhaustive")
    class_count = config.integer("dataset", "classes")
    if {item.label for item in training} != set(range(class_count)):
        raise DataError("training split lost class coverage")
    if {item.label for item in calibration} != set(range(class_count)):
        raise DataError("calibration split lost class coverage")
    attack = stratified_hash_selection(
        test,
        config.integer("dataset", "attack_per_class"),
        config.value("dataset", "selection_seed", str) + ":attack",
    )
    projection = stratified_hash_selection(
        attack,
        config.integer("dataset", "projection_per_class"),
        config.value("dataset", "selection_seed", str) + ":projection",
    )
    split_payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "config_sha256": config.sha256,
        "split_seed": config.value("dataset", "split_seed", str),
        "selection_seed": config.value("dataset", "selection_seed", str),
        "training": [item.to_json() for item in training],
        "calibration": [item.to_json() for item in calibration],
        "test": [item.to_json() for item in sorted(test, key=lambda item: item.sample_id)],
        "attack_subset_ids": [item.sample_id for item in attack],
        "projection_subset_ids": [item.sample_id for item in projection],
    }
    split_hash = split_payload_hash(split_payload)
    split_payload["split_sha256"] = split_hash
    artifacts = config.project_path("artifacts")
    atomic_write_json(artifacts / "data" / "splits.json", split_payload)
    annotation_root = _annotation_root(data_root)
    dataset_hash = dataset_content_hash([*trainval, *test])
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source": "Oxford-IIIT Pet official images and annotations via Torchvision",
        "license": "CC BY-SA 4.0 dataset; original image copyrights retained",
        "integrity": integrity,
        "training_count": len(training),
        "calibration_count": len(calibration),
        "attack_subset_count": len(attack),
        "projection_subset_count": len(projection),
        "annotation_sha256": {
            name: sha256_file(annotation_root / name) for name in ("trainval.txt", "test.txt")
        },
        "dataset_content_sha256": dataset_hash,
        "split_sha256": split_hash,
    }
    atomic_write_json(artifacts / "data" / "manifest.json", manifest)
    return manifest


def _records_from_json(values: object) -> list[SampleRecord]:
    if not isinstance(values, list):
        raise DataError("split record list is invalid")
    records: list[SampleRecord] = []
    for value in values:
        if not isinstance(value, dict):
            raise DataError("split record entry is invalid")
        records.append(SampleRecord(**cast(dict[str, Any], value)))
    return records


def load_registered_splits(config: ExperimentConfig) -> dict[str, list[SampleRecord]]:
    import json

    path = config.project_path("artifacts") / "data" / "splits.json"
    if not path.is_file():
        raise DataError("registered split is missing; run setup first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataError("registered split manifest is invalid")
    expected = payload.get("split_sha256")
    if split_payload_hash(payload) != expected:
        raise DataError("registered split manifest hash is invalid")
    training = _records_from_json(payload.get("training"))
    calibration = _records_from_json(payload.get("calibration"))
    test = _records_from_json(payload.get("test"))
    attack = _select_ids(payload, "attack_subset_ids")
    projection = _select_ids(payload, "projection_subset_ids")

    # Only data-dependent settings invalidate a registered split. Training epochs,
    # learning rates, attack steps, and other downstream parameters remain editable.
    trainval = [*training, *calibration]
    if len(trainval) != config.integer("dataset", "expected_trainval") or len(
        test
    ) != config.integer("dataset", "expected_test"):
        raise DataError("configured dataset counts changed; run setup again")
    classes = config.integer("dataset", "classes")
    if {record.label for record in trainval} != set(range(classes)) or {
        record.label for record in test
    } != set(range(classes)):
        raise DataError("configured dataset classes changed; run setup again")
    expected_training, expected_calibration = stratified_train_calibration_split(
        trainval,
        config.number("dataset", "train_fraction"),
        config.value("dataset", "split_seed", str),
    )
    expected_attack = stratified_hash_selection(
        test,
        config.integer("dataset", "attack_per_class"),
        config.value("dataset", "selection_seed", str) + ":attack",
    )
    expected_projection = stratified_hash_selection(
        expected_attack,
        config.integer("dataset", "projection_per_class"),
        config.value("dataset", "selection_seed", str) + ":projection",
    )

    def ids(records: list[SampleRecord]) -> list[str]:
        return [record.sample_id for record in records]

    if any(
        ids(actual) != ids(expected_records)
        for actual, expected_records in (
            (training, expected_training),
            (calibration, expected_calibration),
            (attack, expected_attack),
            (projection, expected_projection),
        )
    ):
        raise DataError("configured split or sample-selection settings changed; run setup again")
    return {
        "training": training,
        "calibration": calibration,
        "test": test,
        "attack": attack,
        "projection": projection,
    }


def _select_ids(payload: dict[str, Any], key: str) -> list[SampleRecord]:
    test = _records_from_json(payload.get("test"))
    by_id = {item.sample_id: item for item in test}
    ids = payload.get(key)
    if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
        raise DataError(f"invalid {key}")
    try:
        return [by_id[value] for value in cast(list[str], ids)]
    except KeyError as error:
        raise DataError(f"{key} references an unknown test sample") from error


def epoch_order(records: Sequence[SampleRecord], seed: int, epoch: int) -> list[SampleRecord]:
    return sorted(
        records,
        key=lambda item: (_rank(f"order:{seed}:{epoch}", item.sample_id), item.sample_id),
    )


def _local_generator(seed: str) -> torch.Generator:
    value = int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16) % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(value)


def deterministic_train_transform(
    image: Image.Image,
    sample_id: str,
    epoch: int,
    size: int,
    crop_scale: tuple[float, float],
    flip_probability: float,
    seed: int,
) -> torch.Tensor:
    """Random-resized crop and flip driven only by sample, epoch, and locked seed."""

    generator = _local_generator(f"augmentation:{seed}:{epoch}:{sample_id}")
    width, height = image.size
    area = float(width * height)
    log_min = math.log(3.0 / 4.0)
    log_max = math.log(4.0 / 3.0)
    top = left = 0
    crop_height, crop_width = height, width
    found = False
    for _ in range(10):
        target = area * float(
            torch.empty((), dtype=torch.float64).uniform_(
                crop_scale[0], crop_scale[1], generator=generator
            )
        )
        ratio = math.exp(
            float(
                torch.empty((), dtype=torch.float64).uniform_(log_min, log_max, generator=generator)
            )
        )
        candidate_width = int(round(math.sqrt(target * ratio)))
        candidate_height = int(round(math.sqrt(target / ratio)))
        if 0 < candidate_width <= width and 0 < candidate_height <= height:
            crop_width, crop_height = candidate_width, candidate_height
            top = int(torch.randint(0, height - crop_height + 1, (), generator=generator))
            left = int(torch.randint(0, width - crop_width + 1, (), generator=generator))
            found = True
            break
    if not found:
        input_ratio = width / height
        if input_ratio < 3.0 / 4.0:
            crop_width = width
            crop_height = int(round(width / (3.0 / 4.0)))
        elif input_ratio > 4.0 / 3.0:
            crop_height = height
            crop_width = int(round(height * (4.0 / 3.0)))
        else:
            crop_width, crop_height = width, height
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
    output = tvf.resized_crop(
        image,
        top,
        left,
        crop_height,
        crop_width,
        [size, size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    if float(torch.rand((), generator=generator)) < flip_probability:
        output = tvf.hflip(output)
    return cast(torch.Tensor, tvf.pil_to_tensor(output).to(dtype=torch.float32).div_(255.0))


def deterministic_eval_transform(image: Image.Image, resize: int, size: int) -> torch.Tensor:
    output = tvf.resize(
        image,
        resize,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    output = tvf.center_crop(output, [size, size])
    return cast(torch.Tensor, tvf.pil_to_tensor(output).to(dtype=torch.float32).div_(255.0))


class PetRecordDataset(Dataset[tuple[torch.Tensor, int, str, int]]):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        config: ExperimentConfig,
        *,
        training: bool,
        epoch: int = 0,
    ) -> None:
        self.records = list(records)
        self.training = training
        self.epoch = epoch
        self.size = config.integer("input", "size")
        self.resize = config.integer("input", "eval_resize")
        scale = config.section("input").get("train_crop_scale")
        if not isinstance(scale, list) or len(scale) != 2:
            raise DataError("input.train_crop_scale must have two entries")
        self.crop_scale = (float(scale[0]), float(scale[1]))
        self.flip_probability = config.number("input", "horizontal_flip_probability")
        self.seed = config.integer("training", "data_seed")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str, int]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            rgb = image.convert("RGB")
            if self.training:
                tensor = deterministic_train_transform(
                    rgb,
                    record.sample_id,
                    self.epoch,
                    self.size,
                    self.crop_scale,
                    self.flip_probability,
                    self.seed,
                )
            else:
                tensor = deterministic_eval_transform(rgb, self.resize, self.size)
        return tensor, record.label, record.sample_id, record.species


def build_loader(
    records: Sequence[SampleRecord],
    config: ExperimentConfig,
    *,
    training: bool,
    epoch: int = 0,
    batch_size: int | None = None,
) -> DataLoader[Any]:
    ordered = (
        epoch_order(records, config.integer("training", "data_seed"), epoch)
        if training
        else sorted(records, key=lambda item: item.sample_id)
    )
    dataset = PetRecordDataset(ordered, config, training=training, epoch=epoch)
    selected_batch = batch_size or (
        config.integer("training", "micro_batch_size")
        if training
        else config.integer("training", "evaluation_batch_size")
    )
    return DataLoader(
        dataset,
        batch_size=selected_batch,
        shuffle=False,
        num_workers=config.integer("training", "num_workers"),
        persistent_workers=False,
        pin_memory=False,
        drop_last=False,
    )
