from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from src.config import ExperimentConfig
from src.data import (
    DataError,
    SampleRecord,
    create_data_manifests,
    deterministic_eval_transform,
    deterministic_train_transform,
    epoch_order,
    load_registered_splits,
    parse_official_split,
    split_payload_hash,
    stratified_hash_selection,
    stratified_train_calibration_split,
    validate_records,
)


def _records(classes: int = 3, per_class: int = 10) -> list[SampleRecord]:
    return [
        SampleRecord(
            sample_id=f"breed_{label}_{index:03d}",
            image_path="/unused.jpg",
            label=label,
            species=label % 2,
            breed_name=f"breed_{label}",
            official_split="test",
        )
        for label in range(classes)
        for index in range(per_class)
    ]


def test_stratified_split_is_disjoint_exhaustive_and_80_20() -> None:
    records = _records()
    training, calibration = stratified_train_calibration_split(records, 0.8, "seed")
    assert len(training) == 24
    assert len(calibration) == 6
    assert {item.sample_id for item in training}.isdisjoint(
        {item.sample_id for item in calibration}
    )
    assert {item.sample_id for item in training + calibration} == {
        item.sample_id for item in records
    }


def test_stratified_split_repeats_exactly() -> None:
    first = stratified_train_calibration_split(_records(), 0.8, "locked")
    second = stratified_train_calibration_split(list(reversed(_records())), 0.8, "locked")
    assert [[item.sample_id for item in side] for side in first] == [
        [item.sample_id for item in side] for side in second
    ]


def test_fixed_selection_has_equal_class_counts() -> None:
    selected = stratified_hash_selection(_records(), 4, "selection")
    assert len(selected) == 12
    assert [sum(item.label == label for item in selected) for label in range(3)] == [4, 4, 4]


def test_epoch_order_is_arm_independent_and_epoch_specific() -> None:
    records = _records()
    first = [item.sample_id for item in epoch_order(records, 11, 1)]
    repeat = [item.sample_id for item in epoch_order(list(reversed(records)), 11, 1)]
    second_epoch = [item.sample_id for item in epoch_order(records, 11, 2)]
    assert first == repeat
    assert first != second_epoch


def test_deterministic_augmentation_repeats() -> None:
    image = Image.new("RGB", (300, 260), (80, 120, 160))
    first = deterministic_train_transform(image, "sample", 3, 224, (0.8, 1.0), 0.5, 7)
    repeat = deterministic_train_transform(image, "sample", 3, 224, (0.8, 1.0), 0.5, 7)
    assert torch.equal(first, repeat)
    assert first.shape == (3, 224, 224)
    assert 0 <= float(first.min()) <= float(first.max()) <= 1


def test_evaluation_transform_is_fixed() -> None:
    image = Image.new("RGB", (300, 260), (20, 40, 60))
    first = deterministic_eval_transform(image, 256, 224)
    second = deterministic_eval_transform(image, 256, 224)
    assert torch.equal(first, second)
    assert first.shape == (3, 224, 224)


def test_split_hash_excludes_recording_timestamp() -> None:
    first = {"created_at": "one", "training": ["a"], "test": ["b"]}
    second = {"created_at": "two", "training": ["a"], "test": ["b"]}
    assert split_payload_hash(first) == split_payload_hash(second)
    second["test"] = ["c"]
    assert split_payload_hash(first) != split_payload_hash(second)


def test_registered_split_is_scoped_to_data_settings_not_full_config(tmp_path: Path) -> None:
    trainval = _records()
    test = [
        SampleRecord(
            sample_id=f"test_{record.sample_id}",
            image_path=record.image_path,
            label=record.label,
            species=record.species,
            breed_name=record.breed_name,
            official_split="test",
        )
        for record in _records()
    ]
    training, calibration = stratified_train_calibration_split(trainval, 0.8, "split")
    attack = stratified_hash_selection(test, 4, "selection:attack")
    projection = stratified_hash_selection(attack, 2, "selection:projection")
    payload = {
        "schema_version": 1,
        "created_at": "recorded",
        "config_sha256": "old-full-config",
        "split_seed": "split",
        "selection_seed": "selection",
        "training": [record.to_json() for record in training],
        "calibration": [record.to_json() for record in calibration],
        "test": [record.to_json() for record in test],
        "attack_subset_ids": [record.sample_id for record in attack],
        "projection_subset_ids": [record.sample_id for record in projection],
    }
    payload["split_sha256"] = split_payload_hash(payload)
    destination = tmp_path / "artifacts" / "data" / "splits.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(payload), encoding="utf-8")
    raw: dict[str, Any] = {
        "paths": {"artifacts": "artifacts"},
        "dataset": {
            "expected_trainval": 30,
            "expected_test": 30,
            "classes": 3,
            "train_fraction": 0.8,
            "split_seed": "split",
            "attack_per_class": 4,
            "projection_per_class": 2,
            "selection_seed": "selection",
        },
        "training": {"epochs": 1},
    }
    changed = ExperimentConfig(
        tmp_path / "configs" / "experiment.yaml", tmp_path, raw, "new-full-config"
    )
    assert len(load_registered_splits(changed)["training"]) == 24

    stale_raw = deepcopy(raw)
    stale_raw["dataset"]["selection_seed"] = "different"
    stale = ExperimentConfig(changed.path, changed.root, stale_raw, "another-full-config")
    with pytest.raises(DataError, match="run setup again"):
        load_registered_splits(stale)


def test_species_targets_preserve_per_breed_train_calibration_split() -> None:
    breeds = _records(classes=4)
    binary = [replace(record, label=record.species, breed_label=record.label) for record in breeds]
    training, calibration = stratified_train_calibration_split(binary, 0.8, "seed")
    assert {record.label for record in training} == {0, 1}
    for breed_label in range(4):
        assert sum(record.breed_label == breed_label for record in training) == 8
        assert sum(record.breed_label == breed_label for record in calibration) == 2
    original = stratified_train_calibration_split(breeds, 0.8, "seed")
    assert [[record.sample_id for record in part] for part in original] == [
        [record.sample_id for record in part] for part in (training, calibration)
    ]


def test_species_fixed_subsets_are_target_class_balanced_and_deterministic() -> None:
    records = [
        replace(record, label=record.species, breed_label=record.label)
        for record in _records(classes=4)
    ]
    selected = stratified_hash_selection(records, 7, "binary-selection")
    repeat = stratified_hash_selection(list(reversed(records)), 7, "binary-selection")
    assert selected == repeat
    assert [sum(record.label == target for record in selected) for target in range(2)] == [7, 7]
    projection = stratified_hash_selection(selected, 3, "binary-projection")
    assert [sum(record.label == target for record in projection) for target in range(2)] == [3, 3]
    assert {record.sample_id for record in projection} <= {record.sample_id for record in selected}


def _tiny_official_data(tmp_path: Path) -> ExperimentConfig:
    data_root = tmp_path / "data" / "oxford-iiit-pet"
    annotations = data_root / "annotations"
    images = data_root / "images"
    annotations.mkdir(parents=True)
    images.mkdir()
    for split in ("trainval", "test"):
        lines = []
        for breed in range(4):
            for index in range(10):
                image_number = index + (10 if split == "test" else 0)
                sample_id = f"breed{breed}_{image_number}"
                Image.new("RGB", (8, 8), (breed * 40, index * 20, 80)).save(
                    images / f"{sample_id}.jpg"
                )
                # Official annotations are 1-based for both breed and species.
                lines.append(f"{sample_id} {breed + 1} {breed % 2 + 1} {breed + 1}")
        (annotations / f"{split}.txt").write_text("\n".join(lines), encoding="utf-8")
    raw: dict[str, Any] = {
        "paths": {"data": "data", "artifacts": "artifacts"},
        "dataset": {
            "label_mode": "species",
            "expected_trainval": 40,
            "expected_test": 40,
            "classes": 2,
            "train_fraction": 0.8,
            "split_seed": "split",
            "attack_per_class": 7,
            "projection_per_class": 3,
            "selection_seed": "selection",
        },
    }
    return ExperimentConfig(tmp_path / "configs" / "experiment.yaml", tmp_path, raw, "binary")


def test_official_species_mapping_retains_original_breed_identity(tmp_path: Path) -> None:
    config = _tiny_official_data(tmp_path)
    breed_records = parse_official_split(config.project_path("data"), "trainval")
    species_records = parse_official_split(
        config.project_path("data"), "trainval", label_mode="species"
    )
    assert {record.label for record in breed_records} == {0, 1, 2, 3}
    assert {record.label for record in species_records} == {0, 1}
    for breed, species in zip(breed_records, species_records, strict=True):
        assert species.label == species.species
        assert species.breed_label == breed.label == breed.breed_label
        assert species.breed_name == breed.breed_name
        assert species.image_path == breed.image_path


def test_species_setup_and_round_trip_record_target_identity(tmp_path: Path) -> None:
    config = _tiny_official_data(tmp_path)
    manifest = create_data_manifests(config)
    splits = load_registered_splits(config)
    assert manifest["label_mode"] == "species"
    assert manifest["integrity"]["label_to_name"] == {"0": "cat", "1": "dog"}
    assert manifest["integrity"]["breed_count"] == 4
    assert manifest["training_count"] == 32
    assert manifest["calibration_count"] == 8
    assert len(splits["attack"]) == 14
    assert len(splits["projection"]) == 6
    path = config.project_path("artifacts") / "data" / "splits.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["label_mode"] == "species"
    assert payload["train_calibration_stratification"] == "breed"
    assert payload["fixed_subset_stratification"] == "target_class"

    changed_raw = deepcopy(config.raw)
    changed_raw["dataset"]["label_mode"] = "breed"
    changed = ExperimentConfig(config.path, config.root, changed_raw, "breed")
    with pytest.raises(DataError, match="target label mode changed; run setup again"):
        load_registered_splits(changed)


def test_species_integrity_rejects_wrong_target_mapping(tmp_path: Path) -> None:
    config = _tiny_official_data(tmp_path)
    trainval = parse_official_split(config.project_path("data"), "trainval", label_mode="species")
    test = parse_official_split(config.project_path("data"), "test", label_mode="species")
    broken = [replace(trainval[0], label=1 - trainval[0].species), *trainval[1:]]
    with pytest.raises(DataError, match="species target labels must equal species metadata"):
        validate_records(broken, test, config)
