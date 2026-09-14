from __future__ import annotations

import torch
from PIL import Image

from src.data import (
    SampleRecord,
    deterministic_eval_transform,
    deterministic_train_transform,
    epoch_order,
    split_payload_hash,
    stratified_hash_selection,
    stratified_train_calibration_split,
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
