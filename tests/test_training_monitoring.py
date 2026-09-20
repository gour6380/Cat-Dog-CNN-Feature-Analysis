from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.data import SampleRecord, epoch_order
from src.io_utils import atomic_write_json, atomic_write_text
from src.model import atomic_torch_save, state_dict_hash
from src.training import ResumeError, checkpoint_path, experiment_provenance, train_arm
from src.training_monitoring import (
    TRAINING_SCIENTIFIC_SOURCE_FILES,
    MonitoringError,
    collect_clean_monitoring,
    observational_evaluation,
    prepare_training_monitoring,
    register_training_monitoring_source,
    save_training_examples,
    training_examples_path,
    training_monitoring_enabled,
    training_objective_metrics,
)


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        tmp_path / "configs" / "experiment.yaml",
        tmp_path,
        {
            "paths": {"artifacts": "artifacts", "checkpoints": "checkpoints"},
            "dataset": {"label_mode": "species"},
            "model": {"classes": 2},
            "training": {
                "epochs": 2,
                "micro_batch_size": 8,
                "evaluation_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "data_seed": 42,
                "attack_seed": 43,
                "backbone_learning_rate": 0.001,
                "head_learning_rate": 0.01,
                "weight_decay": 0.0001,
                "warmup_epochs": 1,
            },
            "attack": {
                "epsilon": 4 / 255,
                "step_size": 1 / 255,
                "train_steps": 5,
                "random_start": True,
            },
            "training_monitoring": {
                "enabled": True,
                "validation_fraction": 0.1,
                "validation_seed": "oxford-pets-validation-20260918",
            },
        },
        "a" * 64,
    )


def _record(sample_id: str, species: int, breed: int, split: str = "trainval") -> SampleRecord:
    return SampleRecord(
        sample_id,
        "/synthetic/not-a-real-photograph.jpg",
        species,
        species,
        f"breed_{breed}",
        cast(Any, split),
        breed,
    )


def _splits() -> dict[str, list[SampleRecord]]:
    return {
        "training": [
            _record(f"fit-{breed}-{index:02d}", breed, breed)
            for breed in range(2)
            for index in range(20)
        ],
        "calibration": [_record(f"cal-{breed}", breed, breed) for breed in range(2)],
        "test": [_record(f"test-{breed}", breed, breed, "test") for breed in range(2)],
    }


def _loader(
    records: list[SampleRecord],
    config: ExperimentConfig,
    *,
    training: bool,
    epoch: int = 0,
) -> DataLoader[Any]:
    ordered = (
        epoch_order(records, config.integer("training", "data_seed"), epoch)
        if training
        else sorted(records, key=lambda record: record.sample_id)
    )
    # These tensors are synthetic fixtures, not downloaded images or model data.
    values = [
        (
            torch.full((3, 2, 2), 0.2 + 0.6 * record.species + 0.01 * epoch),
            record.label,
            record.sample_id,
            record.species,
        )
        for record in ordered
    ]
    return DataLoader(values, batch_size=8, shuffle=False, num_workers=0)


class TinyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(3, 2)
        with torch.no_grad():
            self.classifier.weight.fill_(0.1)
            self.classifier.bias.zero_()

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.bn(pixels.mean(dim=(2, 3)))))


def test_opt_in_leaves_historical_and_pilot_configs_unaffected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    del config.raw["training_monitoring"]
    assert not training_monitoring_enabled(config)
    config.raw["pilot"] = {"enabled": True}
    assert not training_monitoring_enabled(config)


def test_validation_is_deterministic_train_only_and_breed_stratified(tmp_path: Path) -> None:
    config = _config(tmp_path)
    splits = _splits()
    untouched = deepcopy(splits)
    protocol = prepare_training_monitoring(config, splits)
    repeated = prepare_training_monitoring(
        config, {key: list(reversed(value)) for key, value in splits.items()}
    )
    fit_ids = {record.sample_id for record in protocol.fit_records}
    validation_ids = {record.sample_id for record in protocol.validation_records}
    assert not fit_ids & validation_ids
    assert fit_ids | validation_ids == {record.sample_id for record in splits["training"]}
    assert len(protocol.fit_records) == 36
    assert len(protocol.validation_records) == 4
    assert len(protocol.gallery_records) == 4
    assert {record.species for record in protocol.gallery_records} == {0, 1}
    assert protocol.sha256 == repeated.sha256
    assert splits == untouched
    assert {record.breed_name for record in protocol.fit_records} == {
        record.breed_name for record in protocol.validation_records
    }


def test_synthetic_official_sized_train_reserve_has_expected_counts(tmp_path: Path) -> None:
    # Breed sizes reproduce the required registered-training total without
    # reading, downloading, or evaluating the real Oxford dataset.
    sizes = [72, 76, 76, *([80] * 34)]
    splits = _splits()
    splits["training"] = [
        _record(f"fit-{breed}-{index:03d}", int(breed >= 12), breed)
        for breed, count in enumerate(sizes)
        for index in range(count)
    ]
    protocol = prepare_training_monitoring(_config(tmp_path), splits)
    assert protocol.manifest["counts"] == {
        "original_training": 2944,
        "fit": 2649,
        "validation": 295,
        "gallery": 4,
    }


@pytest.mark.parametrize("problem", ["overlap", "duplicate", "wrong_partition"])
def test_rejects_invalid_training_registration(tmp_path: Path, problem: str) -> None:
    splits = _splits()
    if problem == "overlap":
        splits["calibration"].append(splits["training"][0])
    elif problem == "duplicate":
        splits["training"].append(splits["training"][0])
    else:
        splits["training"][0] = _record("bad", 0, 0, "test")
    with pytest.raises(MonitoringError):
        prepare_training_monitoring(_config(tmp_path), splits)


def test_tampered_registered_protocol_is_not_overwritten(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prepare_training_monitoring(config, _splits())
    path = config.project_path("artifacts") / "training" / "monitoring-protocol.json"
    payload = json.loads(path.read_text())
    payload["fit_ids"] = ["foreign-id"]
    atomic_write_json(path, payload)
    with pytest.raises(MonitoringError, match="differs"):
        prepare_training_monitoring(config, _splits())
    assert json.loads(path.read_text())["fit_ids"] == ["foreign-id"]


def test_scientific_source_is_frozen_and_part_of_main_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    for name in TRAINING_SCIENTIFIC_SOURCE_FILES:
        atomic_write_text(tmp_path / name, f"synthetic source fixture {name}\n")
    atomic_write_json(
        tmp_path / "artifacts" / "data" / "manifest.json",
        {"dataset_content_sha256": "data", "split_sha256": "split"},
    )
    atomic_write_json(
        tmp_path / "artifacts" / "model" / "initialization.json", {"state_sha256": "init"}
    )
    monkeypatch.setattr("src.training_monitoring.load_registered_splits", lambda _: _splits())
    fingerprint = register_training_monitoring_source(config)
    provenance = experiment_provenance(config)
    assert provenance["training_monitoring_source_sha256"] == fingerprint
    assert provenance["training_monitoring_protocol_sha256"]
    atomic_write_text(tmp_path / "src" / "training.py", "changed scientific source\n")
    with pytest.raises(MonitoringError, match="source changed"):
        register_training_monitoring_source(config)


def test_clean_monitoring_preserves_modes_bn_gradients_weights_and_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.training_monitoring.build_loader", _loader)
    model = TinyNetwork().train()
    model.dropout.eval()  # Preserve an intentionally mixed mode, not just root mode.
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in model.modules()]
    states = {name: value.clone() for name, value in model.state_dict().items()}
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    random.seed(5)
    np.random.seed(6)
    torch.manual_seed(7)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    metrics = collect_clean_monitoring(
        model, _splits()["training"], _config(tmp_path), torch.device("cpu"), progress=False
    )
    assert metrics["count"] == 40
    assert np.isfinite(metrics["loss"])
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_accuracy"] == 0.5
    assert metrics["prediction_counts"] == {"0": 40, "1": 0}
    assert metrics["confusion_matrix"] == [[20, 0], [20, 0]]
    assert [module.training for module in model.modules()] == modes
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in states.items())
    assert all(
        torch.equal(old, parameter.grad)
        for old, parameter in zip(gradients, model.parameters(), strict=True)
    )
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert np.random.get_state()[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_observational_modes_and_rng_restore_after_failure() -> None:
    model = TinyNetwork().train()
    model.bn.eval()
    modes = [module.training for module in model.modules()]
    state = torch.get_rng_state().clone()
    with (
        pytest.raises(RuntimeError, match="synthetic failure"),
        observational_evaluation(model, torch.device("cpu")),
    ):
        torch.rand(3)
        raise RuntimeError("synthetic failure")
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(state, torch.get_rng_state())


def test_objective_metrics_expose_single_class_collapse() -> None:
    metrics = training_objective_metrics(torch.tensor([[0, 3], [0, 6]]), 9.0, 9, "adversarial")
    assert metrics["loss"] == 1.0
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["macro_accuracy"] == 0.5
    assert metrics["per_class_accuracy"] == {"0": 0.0, "1": 1.0}
    assert metrics["prediction_counts"] == {"0": 0, "1": 9}


def _rows(protocol: Any, epoch: int) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": record.sample_id,
            "label": record.label,
            "clean_pixels": np.full((3, 2, 2), 0.5, dtype=np.float32),
            "objective_pixels": np.full((3, 2, 2), 0.51, dtype=np.float32),
            "logits": np.array([0.1, 0.2], dtype=np.float32),
            "batch_index": epoch,
            "update_index": epoch + 1,
            "original_image_path": record.image_path,
        }
        for record in protocol.gallery_records
    ]


def test_training_examples_first_final_and_idempotent_retry(tmp_path: Path) -> None:
    config = _config(tmp_path)
    protocol = prepare_training_monitoring(config, _splits())
    provenance = {"config_sha256": config.sha256}
    save_training_examples(config, "adversarial", 1, _rows(protocol, 1), provenance, protocol)
    save_training_examples(config, "adversarial", 2, _rows(protocol, 2), provenance, protocol)
    save_training_examples(config, "adversarial", 2, _rows(protocol, 2), provenance, protocol)
    path = training_examples_path(config, "adversarial")
    with np.load(path, allow_pickle=False) as arrays:
        assert len(arrays["labels"]) == 8
        assert arrays["epochs"].tolist() == [1] * 4 + [2] * 4
        assert arrays["objective_pixels"].max() <= 1
        assert arrays["clean_pixels"].min() >= 0
        assert set(arrays["sample_ids"]) == set(protocol.manifest["gallery_ids"])
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["provenance"] == provenance
    assert metadata["protocol_sha256"] == protocol.sha256


def test_training_examples_reject_stale_bytes_and_missing_anchors(tmp_path: Path) -> None:
    config = _config(tmp_path)
    protocol = prepare_training_monitoring(config, _splits())
    with pytest.raises(MonitoringError, match="every fixed"):
        save_training_examples(config, "standard", 1, [], {}, protocol)
    save_training_examples(config, "standard", 1, _rows(protocol, 1), {}, protocol)
    path = training_examples_path(config, "standard")
    atomic_write_text(path, "tampered arrays\n")
    with pytest.raises(MonitoringError, match="stale"):
        save_training_examples(config, "standard", 2, _rows(protocol, 2), {}, protocol)


@pytest.mark.parametrize("arm", ["standard", "adversarial"])
def test_tiny_matched_training_observer_gallery_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    config = _config(tmp_path)
    config.raw["training"]["epochs"] = 1
    splits = _splits()
    protocol = prepare_training_monitoring(config, splits)
    model = TinyNetwork()
    initial_state = deepcopy(model.state_dict())
    provenance = {
        "initialization_sha256": state_dict_hash(initial_state),
        "config_sha256": config.sha256,
        "training_monitoring_protocol_sha256": protocol.sha256,
    }
    monkeypatch.setattr("src.training.load_registered_splits", lambda _: splits)
    monkeypatch.setattr("src.training_monitoring.load_registered_splits", lambda _: splits)
    monkeypatch.setattr("src.training.build_loader", _loader)
    monkeypatch.setattr("src.training_monitoring.build_loader", _loader)
    monkeypatch.setattr("src.training.experiment_provenance", lambda _: provenance)

    def build(_: Any) -> TinyNetwork:
        instance = TinyNetwork()
        instance.load_state_dict(initial_state)
        return instance

    monkeypatch.setattr("src.training.build_model", build)
    attack_calls: list[int] = []

    def fake_attack(network: Any, pixels: torch.Tensor, labels: Any, ids: Any, spec: Any) -> Any:
        attack_calls.append(len(ids))
        return SimpleNamespace(adversarial=(pixels + 0.01).clamp(0, 1))

    monkeypatch.setattr("src.training.pgd_attack", fake_attack)
    events: list[list[dict[str, Any]]] = []

    def observe(history: list[dict[str, Any]], observed_arm: Any) -> None:
        assert observed_arm == arm
        events.append(deepcopy(history))
        history[0]["epoch"] = 999  # Observer cannot corrupt persisted history.

    result = train_arm(
        config,
        cast(Any, arm),
        torch.device("cpu"),
        progress=False,
        epoch_observer=observe,
    )
    assert result["epochs"] == 1
    assert result["updates"] == 3
    assert len(events) == 1
    record = events[0][0]
    assert record["epoch"] == 1
    assert record["sample_count"] == 36
    assert record["training_objective"]["loss"] == record["mean_training_loss"]
    assert record["monitoring"]["train_clean"]["count"] == 36
    assert record["monitoring"]["validation_clean"]["count"] == 4
    checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    # Exactly the five optimization forwards updated BatchNorm; clean fitting
    # and validation monitoring never add training-mode forwards.
    assert int(checkpoint["model_state"]["bn.num_batches_tracked"]) == 5
    path = training_examples_path(config, arm)
    with np.load(path, allow_pickle=False) as arrays:
        assert len(arrays["labels"]) == 4  # First=final one-epoch run: no duplicate rows.
        difference = arrays["objective_pixels"] - arrays["clean_pixels"]
        assert np.allclose(difference, 0.01 if arm == "adversarial" else 0.0)
    assert len(attack_calls) == (5 if arm == "adversarial" else 0)
    resumed = train_arm(
        config,
        cast(Any, arm),
        torch.device("cpu"),
        progress=False,
        epoch_observer=observe,
    )
    assert resumed["resumed_without_work"] is True
    assert len(events) == 2
    assert events[1][0]["epoch"] == 1
    assert len(attack_calls) == (5 if arm == "adversarial" else 0)
    assert (
        json.loads((tmp_path / "artifacts/training" / f"{arm}.json").read_text())["history"][0][
            "epoch"
        ]
        == 1
    )


def test_main_source_mismatch_is_also_a_checkpoint_resume_error(tmp_path: Path) -> None:
    # The checkpoint's normal exact-provenance check includes the new fields.
    from src.training import _validate_checkpoint

    expected = {"training_monitoring_source_sha256": "new", "combined_sha256": "new"}
    payload = {
        "arm": "standard",
        "epoch": 1,
        "update_count": 3,
        "provenance": {"training_monitoring_source_sha256": "old", "combined_sha256": "old"},
    }
    with pytest.raises(ResumeError, match="provenance"):
        _validate_checkpoint(payload, "standard", expected)


def test_mid_run_resume_replays_then_extends_history_and_retains_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    splits = _splits()
    protocol = prepare_training_monitoring(config, splits)
    initial_state = deepcopy(TinyNetwork().state_dict())
    provenance = {
        "initialization_sha256": state_dict_hash(initial_state),
        "config_sha256": config.sha256,
        "training_monitoring_protocol_sha256": protocol.sha256,
    }
    monkeypatch.setattr("src.training.load_registered_splits", lambda _: splits)
    monkeypatch.setattr("src.training_monitoring.load_registered_splits", lambda _: splits)
    monkeypatch.setattr("src.training.build_loader", _loader)
    monkeypatch.setattr("src.training_monitoring.build_loader", _loader)
    monkeypatch.setattr("src.training.experiment_provenance", lambda _: provenance)

    def build(_: Any) -> TinyNetwork:
        instance = TinyNetwork()
        instance.load_state_dict(initial_state)
        return instance

    monkeypatch.setattr("src.training.build_model", build)

    def interrupt_after_commit(history: list[dict[str, Any]], arm: Any) -> None:
        assert len(history) == 1
        checkpoint = tmp_path / "checkpoints" / config.sha256[:16] / arm / "epoch-001.pt"
        assert checkpoint.is_file()
        manifest = json.loads((tmp_path / "artifacts/training" / f"{arm}.json").read_text())
        assert manifest["completed_epochs"] == 1
        raise RuntimeError("synthetic notebook interruption after committed epoch")

    with pytest.raises(RuntimeError, match="notebook interruption"):
        train_arm(
            config,
            "standard",
            torch.device("cpu"),
            progress=False,
            epoch_observer=interrupt_after_commit,
        )
    replayed_epochs: list[list[int]] = []

    def observe(history: list[dict[str, Any]], arm: Any) -> None:
        replayed_epochs.append([item["epoch"] for item in history])

    result = train_arm(
        config,
        "standard",
        torch.device("cpu"),
        progress=False,
        epoch_observer=observe,
    )
    assert result["resumed_from_epoch"] == 1
    assert result["updates"] == 6
    assert replayed_epochs == [[1], [1, 2]]
    with np.load(training_examples_path(config, "standard"), allow_pickle=False) as arrays:
        assert arrays["epochs"].tolist() == [1] * 4 + [2] * 4
    checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
    assert [item["update_count"] for item in checkpoint["history"]] == [3, 6]
    assert int(checkpoint["model_state"]["bn.num_batches_tracked"]) == 10


@pytest.mark.parametrize("journal_state", ["missing", "stale"])
def test_completed_resume_repairs_journal_before_single_observer_replay_without_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, journal_state: str
) -> None:
    config = _config(tmp_path)
    splits = _splits()
    protocol = prepare_training_monitoring(config, splits)
    model = TinyNetwork()
    initial_state = deepcopy(model.state_dict())
    provenance = {
        "initialization_sha256": state_dict_hash(initial_state),
        "config_sha256": config.sha256,
        "training_monitoring_protocol_sha256": protocol.sha256,
    }
    history = [
        {"epoch": 1, "mean_training_loss": 1.0, "update_count": 3},
        {"epoch": 2, "mean_training_loss": 0.5, "update_count": 6},
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.bn.parameters()), "lr": 0.001},
            {"params": list(model.classifier.parameters()), "lr": 0.01},
        ]
    )
    path = checkpoint_path(config, "standard", 2)
    atomic_torch_save(
        path,
        {
            "arm": "standard",
            "provenance": provenance,
            "epoch": 2,
            "update_count": 6,
            "history": history,
            "model_state": initial_state,
            "optimizer_state": optimizer.state_dict(),
        },
    )
    journal_path = tmp_path / "artifacts/training/standard.json"
    if journal_state == "stale":
        atomic_write_json(
            journal_path,
            {
                "status": "running",
                "arm": "standard",
                "provenance": provenance,
                "history": history[:1],
                "completed_epochs": 1,
                "completed_updates": 3,
                "elapsed_this_invocation_seconds": 12.5,
                "start_memory": {"system_available_gib": 5.0},
            },
        )
    monkeypatch.setattr("src.training.load_registered_splits", lambda _: splits)
    monkeypatch.setattr("src.training_monitoring.load_registered_splits", lambda _: splits)
    monkeypatch.setattr("src.training.experiment_provenance", lambda _: provenance)

    def build(_: Any) -> TinyNetwork:
        instance = TinyNetwork()
        instance.load_state_dict(initial_state)
        return instance

    def forbid(*_: Any, **__: Any) -> Any:
        raise AssertionError("completed resume must not rerun optimization or monitoring")

    monkeypatch.setattr("src.training.build_model", build)
    monkeypatch.setattr("src.training.build_loader", forbid)
    monkeypatch.setattr("src.training.collect_clean_monitoring", forbid)
    monkeypatch.setattr("src.training.pgd_attack", forbid)
    observed: list[list[dict[str, Any]]] = []

    def observe(replayed: list[dict[str, Any]], arm: Any) -> None:
        saved = json.loads(journal_path.read_text())
        assert saved["history"] == replayed == history
        assert saved["completed_epochs"] == 2
        assert saved["completed_updates"] == 6
        observed.append(deepcopy(replayed))

    result = train_arm(
        config,
        "standard",
        torch.device("cpu"),
        progress=False,
        epoch_observer=observe,
    )
    assert result["resumed_without_work"] is True
    assert result["journal_repaired_from_checkpoint"] is True
    assert observed == [history]
    saved = json.loads(journal_path.read_text())
    assert saved["checkpoint"] == str(path)
    assert saved["status"] == "complete"
    assert saved["effective_fit_count"] == 36
    assert saved["validation_count"] == 4
    if journal_state == "stale":
        assert saved["elapsed_this_invocation_seconds"] == 12.5
        assert saved["start_memory"] == {"system_available_gib": 5.0}
    else:
        assert "elapsed_this_invocation_seconds" not in saved
        assert "start_memory" not in saved
    before = journal_path.read_bytes()
    second = train_arm(config, "standard", torch.device("cpu"), progress=False)
    assert second["journal_repaired_from_checkpoint"] is False
    assert journal_path.read_bytes() == before
