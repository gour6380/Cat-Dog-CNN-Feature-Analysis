from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as nnf

from src.attacks import AttackSpec
from src.config import ExperimentConfig
from src.data import SampleRecord
from src.io_utils import atomic_write_json, atomic_write_text, canonical_json_bytes, sha256_bytes
from src.model import atomic_torch_save
from src.pilot_protocol import (
    PILOT_SCIENTIFIC_SOURCE_FILES,
    PilotError,
    collect_pilot_validation,
    pilot_enabled,
    pilot_phase,
    pilot_training_loss,
    prepare_pilot_protocol,
    register_pilot_training_source,
)
from src.training import (
    ResumeError,
    _load_resume,
    _validate_checkpoint,
    checkpoint_path,
    experiment_provenance,
    train_arm,
)


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        tmp_path / "configs" / "pilot.yaml",
        tmp_path,
        {
            "paths": {"artifacts": "artifacts/pilot", "results": "results/pilot"},
            "dataset": {"label_mode": "species"},
            "training": {"epochs": 15},
            "attack": {
                "epsilon": 4 / 255,
                "step_size": 1 / 255,
                "evaluation_steps": 20,
                "evaluation_restarts": 5,
                "random_start": True,
                "evaluation_seed": 42,
            },
            "pilot": {
                "enabled": True,
                "clean_warmup_epochs": 3,
                "clean_loss_weight": 0.5,
                "validation_fraction": 0.1,
                "validation_seed": "pilot-test",
                "validation_attack_per_species": 2,
            },
        },
        "a" * 64,
    )


def _record(sample_id: str, species: int, breed: int, split: str = "trainval") -> SampleRecord:
    return SampleRecord(
        sample_id,
        "/unused/local-image.jpg",
        species,
        species,
        f"breed_{breed}",
        cast(Any, split),
        breed,
    )


def _splits() -> dict[str, list[SampleRecord]]:
    training = [
        _record(f"train-{breed}-{index:02d}", int(breed >= 3), breed)
        for breed in range(7)
        for index in range(20)
    ]
    calibration = [_record(f"cal-{breed}", int(breed >= 3), breed) for breed in range(7)]
    test = [_record(f"test-{breed}", int(breed >= 3), breed, "test") for breed in range(7)]
    return {"training": training, "calibration": calibration, "test": test}


def test_pilot_split_is_deterministic_train_only_and_breed_stratified(tmp_path: Path) -> None:
    config = _config(tmp_path)
    splits = _splits()
    protocol = prepare_pilot_protocol(config, splits)
    repeated = prepare_pilot_protocol(
        config, {key: list(reversed(value)) for key, value in splits.items()}
    )
    fit_ids = {record.sample_id for record in protocol.fit_records}
    validation_ids = {record.sample_id for record in protocol.validation_records}
    excluded = {record.sample_id for key in ("calibration", "test") for record in splits[key]}
    assert not fit_ids & validation_ids
    assert fit_ids | validation_ids == {record.sample_id for record in splits["training"]}
    assert not (fit_ids | validation_ids) & excluded
    assert {record.breed_name for record in protocol.fit_records} == {
        record.breed_name for record in protocol.validation_records
    }
    assert len(protocol.fit_records) == 126
    assert len(protocol.validation_records) == 14
    assert protocol.manifest["per_species_counts"]["validation_attack"] == {"0": 2, "1": 2}
    assert protocol.sha256 == repeated.sha256
    assert protocol.weights == pytest.approx((126 / (2 * 54), 126 / (2 * 72)))
    assert protocol.manifest["checkpoint_selection"] == "none; fixed configured final epoch"


def test_pilot_weights_use_effective_fit_only(tmp_path: Path) -> None:
    first = prepare_pilot_protocol(_config(tmp_path / "first"), _splits())
    changed = _splits()
    changed["calibration"].extend(_record(f"extra-cal-{index}", 0, 0) for index in range(50))
    changed["test"].extend(_record(f"extra-test-{index}", 1, 6, "test") for index in range(50))
    second = prepare_pilot_protocol(_config(tmp_path / "second"), changed)
    assert first.weights == second.weights
    assert first.manifest["fit_ids"] == second.manifest["fit_ids"]


def test_pilot_rejects_training_overlap_and_duplicate_ids(tmp_path: Path) -> None:
    splits = _splits()
    splits["calibration"].append(splits["training"][0])
    with pytest.raises(PilotError, match="overlaps"):
        prepare_pilot_protocol(_config(tmp_path), splits)
    splits = _splits()
    splits["training"].append(splits["training"][0])
    with pytest.raises(PilotError, match="duplicate"):
        prepare_pilot_protocol(_config(tmp_path), splits)


def test_pilot_manifest_tampering_cannot_resume_or_overwrite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prepare_pilot_protocol(config, _splits())
    path = config.project_path("artifacts") / "training" / "validation-split.json"
    payload = json.loads(path.read_text())
    payload["species_weights"]["0"] = 999
    atomic_write_json(path, payload)
    with pytest.raises(PilotError, match="refusing to overwrite or resume"):
        prepare_pilot_protocol(config, _splits())
    assert json.loads(path.read_text())["species_weights"]["0"] == 999


def test_pilot_phase_and_weighted_warmup_have_no_attack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    protocol = prepare_pilot_protocol(config, _splits())
    assert [pilot_phase(config, epoch) for epoch in (1, 3, 4, 15)] == [
        "clean_warmup",
        "clean_warmup",
        "mixed_clean_pgd",
        "mixed_clean_pgd",
    ]
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
    pixels = torch.linspace(0, 1, 24).reshape(2, 3, 2, 2)
    labels = torch.tensor([0, 1])

    def forbidden_attack(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("clean warmup generated an attack")

    monkeypatch.setattr("src.pilot_protocol.pgd_attack", forbidden_attack)
    loss, clean, adversarial = pilot_training_loss(
        model,
        pixels,
        labels,
        ["cat", "dog"],
        3,
        AttackSpec(4 / 255, 1 / 255, 5, True, 1, 42),
        protocol,
        config,
    )
    expected = nnf.cross_entropy(model(pixels), labels, weight=torch.tensor(protocol.weights))
    assert torch.equal(loss, expected)
    assert clean is loss and adversarial is None
    loss.backward()  # type: ignore[no-untyped-call]
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_pilot_mixed_objective_is_equal_weighted_clean_and_pgd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    protocol = prepare_pilot_protocol(config, _splits())
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
    pixels = torch.linspace(0, 1, 24).reshape(2, 3, 2, 2)
    labels = torch.tensor([0, 1])
    calls: list[Any] = []

    def fake_attack(
        model: nn.Module,
        pixels: torch.Tensor,
        labels: torch.Tensor,
        ids: list[str],
        spec: AttackSpec,
    ) -> Any:
        calls.append((ids, spec))
        return SimpleNamespace(adversarial=pixels * 0.5)

    monkeypatch.setattr("src.pilot_protocol.pgd_attack", fake_attack)
    spec = AttackSpec(4 / 255, 1 / 255, 5, True, 1, 42)
    loss, clean, adversarial = pilot_training_loss(
        model, pixels, labels, ["cat", "dog"], 4, spec, protocol, config
    )
    weights = torch.tensor(protocol.weights)
    expected_clean = nnf.cross_entropy(model(pixels), labels, weight=weights)
    expected_adversarial = nnf.cross_entropy(model(pixels * 0.5), labels, weight=weights)
    assert torch.equal(clean, expected_clean)
    assert adversarial is not None and torch.equal(adversarial, expected_adversarial)
    assert torch.equal(loss, 0.5 * expected_clean + 0.5 * expected_adversarial)
    assert calls == [(["cat:epoch:4", "dog:epoch:4"], spec)]


def test_pilot_provenance_is_optional_and_rejects_resume_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    artifacts = config.project_path("artifacts")
    atomic_write_json(
        artifacts / "data" / "manifest.json",
        {"dataset_content_sha256": "dataset", "split_sha256": "split"},
    )
    atomic_write_json(
        artifacts / "model" / "initialization.json", {"state_sha256": "initialization"}
    )
    monkeypatch.setattr("src.pilot_protocol.load_registered_splits", lambda _: _splits())
    monkeypatch.setattr(
        "src.training.register_pilot_training_source", lambda _: "scientific-source"
    )
    provenance = experiment_provenance(config)
    assert provenance["pilot_protocol_sha256"] == prepare_pilot_protocol(config, _splits()).sha256
    assert provenance["pilot_training_source_sha256"] == "scientific-source"
    payload = {
        "arm": "adversarial",
        "epoch": 3,
        "update_count": 12,
        "provenance": {**provenance, "pilot_protocol_sha256": "changed"},
    }
    with pytest.raises(ResumeError, match="provenance"):
        _validate_checkpoint(payload, "adversarial", provenance)
    config.raw.pop("pilot")
    assert not pilot_enabled(config)
    original = experiment_provenance(config)
    assert set(original) == {
        "config_sha256",
        "dataset_sha256",
        "split_sha256",
        "initialization_sha256",
        "combined_sha256",
    }
    assert original["combined_sha256"] == sha256_bytes(
        canonical_json_bytes(
            {
                "config": config.sha256,
                "dataset": "dataset",
                "split": "split",
                "initialization": "initialization",
            }
        )
    )


def test_validation_pairs_collected_filename_order_and_restores_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    protocol = prepare_pilot_protocol(config, _splits())
    dog, cat = _record("a-dog", 1, 6), _record("z-cat", 0, 0)
    protocol.validation_records[:] = [dog, cat]
    protocol.attack_records[:] = [
        cat,
        dog,
    ]  # label order differs from the eval loader's filename order.
    arrays = {
        "sample_ids": np.asarray(["a-dog", "z-cat"]),
        "labels": np.asarray([1, 0]),
        "species": np.asarray([1, 0]),
        "logits": np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        "features": np.ones((2, 3)),
    }
    model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2)).train()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    old_state = {key: value.clone() for key, value in model.state_dict().items()}
    old_gradients = [cast(torch.Tensor, parameter.grad).clone() for parameter in model.parameters()]

    def fake_clean(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert not model.training
        return arrays

    def fake_attack(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert not model.training
        return {**arrays, "cumulative_predictions": np.tile(np.asarray([1, 0]), (5, 1))}

    monkeypatch.setattr("src.evaluation._collect_clean_or_shift", fake_clean)
    monkeypatch.setattr("src.evaluation._collect_attack", fake_attack)
    metrics = collect_pilot_validation(
        cast(Any, model), protocol, config, torch.device("cpu"), 1, progress=False
    )
    assert metrics["pgd"]["accuracy"] == 1.0
    assert metrics["attack"]["clean_accuracy_on_subset"] == 1.0
    assert metrics["pgd"]["prediction_counts"] == {"0": 1, "1": 1}
    assert metrics["clean"]["count"] == 2
    assert model.training
    assert all(torch.equal(value, old_state[key]) for key, value in model.state_dict().items())
    assert all(
        torch.equal(cast(torch.Tensor, parameter.grad), old)
        for parameter, old in zip(model.parameters(), old_gradients, strict=True)
    )
    assert all((config.root / item["path"]).is_file() for item in metrics["per_sample"].values())


def test_validation_failure_restores_model_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    protocol = prepare_pilot_protocol(config, _splits())
    model = nn.Linear(3, 2).train()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("validation failed")

    monkeypatch.setattr("src.evaluation._collect_clean_or_shift", fail)
    with pytest.raises(RuntimeError, match="validation failed"):
        collect_pilot_validation(
            cast(Any, model), protocol, config, torch.device("cpu"), 1, progress=False
        )
    assert model.training


def test_pilot_resume_preserves_phase_and_validation_history(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.raw["paths"]["checkpoints"] = "checkpoints"
    protocol = prepare_pilot_protocol(config, _splits())
    provenance = {"pilot_protocol_sha256": protocol.sha256}
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    history = [
        {
            "epoch": 3,
            "pilot_phase": "clean_warmup",
            "validation": {
                "protocol_sha256": protocol.sha256,
                "clean": {"accuracy": 0.8},
                "pgd": {"accuracy": 0.6},
            },
        }
    ]
    atomic_torch_save(
        checkpoint_path(config, "adversarial", 3),
        {
            "arm": "adversarial",
            "epoch": 3,
            "update_count": 12,
            "provenance": provenance,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
        },
    )
    restored = nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    epoch, updates, restored_history = _load_resume(
        config, "adversarial", cast(Any, restored), restored_optimizer, provenance
    )
    assert (epoch, updates, restored_history) == (3, 12, history)
    assert pilot_phase(config, epoch + 1) == "mixed_clean_pgd"
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in model.state_dict().items()
    )


def test_pilot_rejects_standard_arm_without_constructing_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr("src.training.load_registered_splits", lambda _: _splits())

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pilot standard arm constructed a model")

    monkeypatch.setattr("src.training.build_model", forbidden)
    with pytest.raises(ValueError, match="only the adversarial arm"):
        train_arm(config, "standard", torch.device("cpu"), progress=False)


def test_actual_strong_validation_attack_preserves_batchnorm_and_parameter_gradients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    protocol = prepare_pilot_protocol(config, _splits())
    dog, cat = _record("a-dog", 1, 6), _record("z-cat", 0, 0)
    protocol.validation_records[:] = [dog, cat]
    protocol.attack_records[:] = [cat, dog]

    class TinyFeatureModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm2d(3)
            self.classifier = nn.Linear(3, 2)

        def forward_with_features(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            features = self.bn(pixels).mean(dim=(2, 3))
            return self.classifier(features), features

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return self.forward_with_features(pixels)[0]

    model = TinyFeatureModel().train()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    old_state = {key: value.clone() for key, value in model.state_dict().items()}
    old_gradients = [cast(torch.Tensor, parameter.grad).clone() for parameter in model.parameters()]

    def fake_loader(records: list[SampleRecord], *args: Any, **kwargs: Any) -> list[Any]:
        ordered = sorted(records, key=lambda record: record.sample_id)
        labels = torch.tensor([record.label for record in ordered])
        pixels = torch.stack(
            [torch.full((3, 2, 2), 0.3 + 0.4 * record.label) for record in ordered]
        )
        return [(pixels, labels, [record.sample_id for record in ordered], labels.clone())]

    monkeypatch.setattr("src.evaluation.build_loader", fake_loader)
    metrics = collect_pilot_validation(
        cast(Any, model), protocol, config, torch.device("cpu"), 1, progress=False
    )
    assert metrics["attack"]["steps"] == 20
    assert metrics["attack"]["restarts"] == 5
    assert len(metrics["attack"]["cumulative_restart_robust_accuracy"]) == 5
    assert metrics["pgd"]["count"] == 2
    assert model.training
    assert all(torch.equal(value, old_state[key]) for key, value in model.state_dict().items())
    assert all(
        torch.equal(cast(torch.Tensor, parameter.grad), old)
        for parameter, old in zip(model.parameters(), old_gradients, strict=True)
    )


def test_scientific_source_freeze_rejects_edits_but_ignores_reporting(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for name in PILOT_SCIENTIFIC_SOURCE_FILES:
        atomic_write_text(config.root / name, f"fixture for {name}\n")
    first = register_pilot_training_source(config)
    source_path = config.project_path("artifacts") / "training" / "source.json"
    registered_bytes = source_path.read_bytes()
    payload = json.loads(registered_bytes)
    assert payload["pilot_training_source_sha256"] == first
    assert set(payload["files"]) == set(PILOT_SCIENTIFIC_SOURCE_FILES)
    assert "environment" in payload and "git" in payload
    atomic_write_text(config.root / "src/pilot_reporting.py", "a new reporting chart\n")
    atomic_write_text(config.root / "src/cli.py", "a new CLI help message\n")
    assert register_pilot_training_source(config) == first
    assert source_path.read_bytes() == registered_bytes
    atomic_write_text(config.root / "src/attacks.py", "scientific attack code changed\n")
    with pytest.raises(PilotError, match="scientific source changed"):
        register_pilot_training_source(config)
    assert source_path.read_bytes() == registered_bytes
