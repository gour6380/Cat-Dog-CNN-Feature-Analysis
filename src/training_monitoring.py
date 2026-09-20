"""Training-only validation, clean monitoring, and actual objective-input receipts.

These opt-in helpers belong to the revised matched experiment, not the frozen
class-weighted pilot. Evaluation observes the configured final checkpoint only;
validation never selects a checkpoint or uses calibration/test examples.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from src.config import ExperimentConfig
from src.data import (
    SampleRecord,
    build_loader,
    load_registered_splits,
    stratified_hash_selection,
    stratified_train_calibration_split,
)
from src.io_utils import (
    atomic_save_npz,
    atomic_write_json,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    tree_hash,
    utc_now,
)
from src.metrics import accuracy_metrics
from src.progress import tqdm
from src.runtime import ensure_finite


class MonitoringError(RuntimeError):
    """The registered training-only split or monitoring evidence is inconsistent."""


@dataclass(frozen=True)
class TrainingMonitoringProtocol:
    fit_records: list[SampleRecord]
    validation_records: list[SampleRecord]
    gallery_records: list[SampleRecord]
    sha256: str
    manifest: dict[str, Any]


TRAINING_SCIENTIFIC_SOURCE_FILES = (
    "src/training.py",
    "src/training_monitoring.py",
    "src/attacks.py",
    "src/data.py",
    "src/model.py",
    "src/config.py",
    "src/runtime.py",
    "src/io_utils.py",
    "src/metrics.py",
    "src/progress.py",
    "requirements.txt",
)


def training_monitoring_enabled(config: ExperimentConfig) -> bool:
    section = config.raw.get("training_monitoring")
    return isinstance(section, dict) and section.get("enabled") is True


def _ids(records: list[SampleRecord]) -> list[str]:
    return [record.sample_id for record in records]


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"created_at", "training_monitoring_protocol_sha256"}
            }
        )
    )


def prepare_training_monitoring(
    config: ExperimentConfig,
    splits: dict[str, list[SampleRecord]] | None = None,
) -> TrainingMonitoringProtocol:
    """Register identical deterministic breed-stratified fitting/validation IDs."""

    if not training_monitoring_enabled(config):
        raise MonitoringError("training monitoring requires training_monitoring.enabled: true")
    if config.label_mode != "species":
        raise MonitoringError("the revised notebook uses cat/dog targets")
    if isinstance(config.raw.get("pilot"), dict) and config.raw["pilot"].get("enabled") is True:
        raise MonitoringError("main monitoring and the frozen exploratory pilot are separate")
    fraction = config.number("training_monitoring", "validation_fraction")
    seed = config.value("training_monitoring", "validation_seed", str)
    if not 0 < fraction < 1 or not seed:
        raise MonitoringError("validation needs a fraction between zero and one and a seed")
    registered = load_registered_splits(config) if splits is None else splits
    original = sorted(registered["training"], key=lambda record: record.sample_id)
    original_ids = set(_ids(original))
    if len(original_ids) != len(original):
        raise MonitoringError("registered training contains duplicate IDs")
    if any(
        record.official_split != "trainval" or record.label != record.species for record in original
    ):
        raise MonitoringError("fitting records must have official trainval species targets")
    if original_ids & {*_ids(registered["calibration"]), *_ids(registered["test"])}:
        raise MonitoringError("registered training overlaps calibration or test")
    fit, validation = stratified_train_calibration_split(original, 1 - fraction, seed)
    fit_ids, validation_ids = set(_ids(fit)), set(_ids(validation))
    if fit_ids & validation_ids or fit_ids | validation_ids != original_ids:
        raise MonitoringError("fitting/validation must partition registered training")
    if {record.breed_name for record in fit} != {record.breed_name for record in validation}:
        raise MonitoringError("each original breed must remain in fitting and validation")
    if {record.label for record in fit} != {0, 1} or {record.label for record in validation} != {
        0,
        1,
    }:
        raise MonitoringError("both species must occur in fitting and validation")
    gallery = stratified_hash_selection(fit, 2, seed + ":training-gallery")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "config_sha256": config.sha256,
        "status": "registered_before_fitting",
        "validation_source": "original_registered_training_only",
        "validation_seed": seed,
        "validation_fraction": fraction,
        "stratification": "breed",
        "checkpoint_selection": "none; fixed configured final epoch",
        "fixed_final_epoch": config.integer("training", "epochs"),
        "training_objectives": {
            "standard": "unweighted clean augmented cross entropy",
            "adversarial": "unweighted PGD augmented cross entropy",
        },
        "counts": {
            "original_training": len(original),
            "fit": len(fit),
            "validation": len(validation),
            "gallery": len(gallery),
        },
        "per_species_counts": {
            name: {str(label): sum(record.label == label for record in records) for label in (0, 1)}
            for name, records in (("fit", fit), ("validation", validation))
        },
        "original_training_ids": _ids(original),
        "fit_ids": _ids(fit),
        "validation_ids": _ids(validation),
        "gallery_ids": _ids(gallery),
        "fit_records": [record.to_json() for record in fit],
        "validation_records": [record.to_json() for record in validation],
        "untouched_calibration_ids_sha256": sha256_bytes(
            canonical_json_bytes(sorted(_ids(registered["calibration"])))
        ),
        "untouched_test_ids_sha256": sha256_bytes(
            canonical_json_bytes(sorted(_ids(registered["test"])))
        ),
    }
    fingerprint = _manifest_hash(manifest)
    manifest["training_monitoring_protocol_sha256"] = fingerprint
    path = config.project_path("artifacts") / "training" / "monitoring-protocol.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or (
            _manifest_hash(existing) != fingerprint
            or existing.get("training_monitoring_protocol_sha256") != fingerprint
        ):
            raise MonitoringError("registered training monitoring protocol differs; do not resume")
        manifest = existing
    else:
        atomic_write_json(path, manifest)
    return TrainingMonitoringProtocol(fit, validation, gallery, fingerprint, manifest)


def register_training_monitoring_source(config: ExperimentConfig) -> str:
    """Freeze scientific training code so resumed measurements share an identity."""

    paths = [config.root / name for name in TRAINING_SCIENTIFIC_SOURCE_FILES]
    if any(not path.is_file() for path in paths):
        raise MonitoringError("scientific training source or dependency lock is missing")
    fingerprint = tree_hash(config.root, paths)
    hashes = {name: sha256_file(config.root / name) for name in TRAINING_SCIENTIFIC_SOURCE_FILES}
    path = config.project_path("artifacts") / "training" / "monitoring-source.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or (
            existing.get("training_monitoring_source_sha256") != fingerprint
            or existing.get("files") != hashes
            or existing.get("config_sha256") != config.sha256
        ):
            raise MonitoringError("scientific training source changed; refusing to resume")
    else:
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "config_sha256": config.sha256,
                "training_monitoring_source_sha256": fingerprint,
                "files": hashes,
            },
        )
    return fingerprint


@contextmanager
def observational_evaluation(model: nn.Module, device: torch.device) -> Iterator[None]:
    """Leave mixed module modes and caller RNG streams exactly as they were."""

    modes = [(module, module.training) for module in model.modules()]
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def collect_clean_monitoring(
    model: nn.Module,
    records: list[SampleRecord],
    config: ExperimentConfig,
    device: torch.device,
    *,
    progress: bool = True,
    description: str = "clean monitoring",
) -> dict[str, Any]:
    """Measure all fixed records with clean eval crops, without features or updates."""

    collected_logits: list[np.ndarray[Any, Any]] = []
    collected_labels: list[np.ndarray[Any, Any]] = []
    collected_ids: list[str] = []
    total_loss = 0.0
    with observational_evaluation(model, device):
        loader = build_loader(records, config, training=False)
        # DataLoader normally draws its base seed from the caller's global CPU
        # generator even when shuffle=False. Isolate it as well as restoring RNG.
        loader.generator = torch.Generator(device="cpu").manual_seed(
            config.integer("training", "data_seed")
        )
        with tqdm(
            loader,
            total=len(loader),
            desc=description,
            unit="batch",
            leave=False,
            disable=not progress,
        ) as batches:
            for pixels, labels, sample_ids, _species in batches:
                pixels = pixels.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)
                logits = model(pixels)
                ensure_finite("clean monitoring logits", logits)
                loss = nnf.cross_entropy(logits, labels, reduction="sum")
                ensure_finite("clean monitoring loss", loss)
                total_loss += float(loss.item())
                collected_logits.append(logits.detach().cpu().numpy().astype(np.float64))
                collected_labels.append(labels.cpu().numpy().astype(np.int64))
                collected_ids.extend(str(value) for value in sample_ids)
    if not collected_ids:
        raise MonitoringError("clean monitoring requires nonempty records")
    if collected_ids != sorted(_ids(records)):
        raise MonitoringError("clean monitoring changed fixed sample order or coverage")
    logits_array = np.concatenate(collected_logits)
    labels_array = np.concatenate(collected_labels)
    predictions = logits_array.argmax(axis=1)
    classes = config.integer("model", "classes")
    return {
        **accuracy_metrics(logits_array, labels_array, classes),
        "loss": total_loss / len(collected_ids),
        "mean_loss": total_loss / len(collected_ids),
        "count": len(collected_ids),
        "correct_count": int((predictions == labels_array).sum()),
        "prediction_counts": {
            str(label): int((predictions == label).sum()) for label in range(classes)
        },
        "sample_ids_sha256": sha256_bytes(canonical_json_bytes(collected_ids)),
        "input_policy": "clean deterministic resize and center crop",
        "model_mode": "evaluation; no gradients or updates",
    }


def training_objective_metrics(
    confusion: torch.Tensor, total_loss: float, count: int, arm: str
) -> dict[str, Any]:
    """Aggregate the existing training forwards, never add a second forward."""

    matrix = confusion.detach().cpu().numpy().astype(np.int64)
    counts = matrix.sum(axis=1)
    recalls = np.divide(
        matrix.diagonal(), counts, out=np.full(len(counts), np.nan), where=counts > 0
    )
    return {
        "objective": "clean augmented CE" if arm == "standard" else "PGD augmented CE",
        "loss": total_loss / count,
        "mean_loss": total_loss / count,
        "count": count,
        "correct_count": int(matrix.diagonal().sum()),
        "accuracy": float(matrix.diagonal().sum() / count),
        "macro_accuracy": float(np.nanmean(recalls)),
        "per_class_accuracy": {str(label): float(value) for label, value in enumerate(recalls)},
        "confusion_matrix": matrix.tolist(),
        "prediction_counts": {
            str(label): int(value) for label, value in enumerate(matrix.sum(axis=0))
        },
        "model_mode": "training; measured from actual objective forward passes",
    }


def training_examples_path(config: ExperimentConfig, arm: str) -> Path:
    return config.project_path("artifacts") / "training" / f"{arm}-examples.npz"


def save_training_examples(
    config: ExperimentConfig,
    arm: Literal["standard", "adversarial"],
    epoch: int,
    rows: list[dict[str, Any]],
    provenance: dict[str, str],
    protocol: TrainingMonitoringProtocol,
) -> dict[str, Any]:
    """Persist first/final-epoch pixels captured in actual optimization batches."""

    if {row["sample_id"] for row in rows} != set(_ids(protocol.gallery_records)):
        raise MonitoringError("training example capture must cover every fixed fitting anchor")
    rows = sorted(rows, key=lambda row: str(row["sample_id"]))
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "sample_ids": np.asarray([row["sample_id"] for row in rows]),
        "clean_pixels": np.stack([row["clean_pixels"] for row in rows]),
        "objective_pixels": np.stack([row["objective_pixels"] for row in rows]),
        "logits": np.stack([row["logits"] for row in rows]),
        "labels": np.asarray([row["label"] for row in rows], dtype=np.int64),
        "epochs": np.full(len(rows), epoch, dtype=np.int64),
        "batch_indices": np.asarray([row["batch_index"] for row in rows], dtype=np.int64),
        "update_indices": np.asarray([row["update_index"] for row in rows], dtype=np.int64),
        "original_image_paths": np.asarray([row["original_image_path"] for row in rows]),
    }
    path = training_examples_path(config, arm)
    sidecar = path.with_suffix(".json")
    if path.is_file() != sidecar.is_file():
        raise MonitoringError("training example arrays/receipt pair is incomplete")
    if path.is_file():
        old_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            old_metadata.get("provenance") != provenance
            or old_metadata.get("sha256") != sha256_file(path)
            or old_metadata.get("protocol_sha256") != protocol.sha256
        ):
            raise MonitoringError("existing training examples have stale provenance or bytes")
        with np.load(path, allow_pickle=False) as saved:
            # A retry can replace pixels for the same epoch, not duplicate them.
            keep = saved["epochs"] != epoch
            arrays = {
                key: np.concatenate((saved[key][keep], value)) for key, value in arrays.items()
            }
    atomic_save_npz(path, arrays)
    metadata = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "arm": arm,
        "path": str(path.relative_to(config.root)),
        "sha256": sha256_file(path),
        "npz_sha256": sha256_file(path),
        "provenance": provenance,
        "protocol_sha256": protocol.sha256,
        "sample_ids": arrays["sample_ids"].tolist(),
        "epochs": arrays["epochs"].tolist(),
        "count": len(arrays["labels"]),
        "pixel_range": "[0,1]; before ImageNet normalization",
        "capture": "actual optimization input and logits, before its optimizer update",
        "update_index_semantics": "one-based optimizer update containing the micro-batch",
        "photographs": "ignored local artifact; do not publish",
    }
    atomic_write_json(sidecar, metadata)
    return metadata
