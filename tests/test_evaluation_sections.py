"""Bounded synthetic section evaluation; no dataset, weights, or training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from src import evaluation
from src.attacks import AttackResult
from src.config import ExperimentConfig
from src.data import SampleRecord
from src.io_utils import atomic_save_npz, atomic_write_json, sha256_file
from src.training import checkpoint_path


def _record(name: str, species: int) -> SampleRecord:
    return SampleRecord(name, f"/{name}.jpg", species, species, name, "test", species)


def _arrays(records: list[SampleRecord], condition: str = "clean") -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: item.sample_id)
    labels = np.asarray([item.label for item in ordered], dtype=np.int64)
    logits = np.full((len(labels), 2), -0.5, dtype=np.float32)
    logits[np.arange(len(labels)), labels] = 1.0
    if condition == "fgsm":
        logits[:] = np.asarray([1.0, -0.5])
    elif condition == "pgd":
        logits = logits[:, ::-1].copy()
    gallery = ordered[:4]
    arrays = {
        "sample_ids": np.asarray([item.sample_id for item in ordered]),
        "labels": labels,
        "species": np.asarray([item.species for item in ordered]),
        "logits": logits,
        "features": np.ones((len(labels), 3), dtype=np.float32),
        "gallery_sample_ids": np.asarray([item.sample_id for item in gallery]),
        "gallery_labels": np.asarray([item.label for item in gallery]),
        "gallery_species": np.asarray([item.species for item in gallery]),
    }
    pixels = np.full((len(gallery), 3, 4, 4), 0.5, dtype=np.float32)
    if condition in {"fgsm", "pgd"}:
        arrays.update(
            gallery_clean_pixels=pixels,
            gallery_attack_pixels=pixels + 1.0 / 255.0,
            cumulative_predictions=np.tile(logits.argmax(axis=1), (5, 1)),
        )
    else:
        arrays["gallery_pixels"] = pixels
    return arrays


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    raw = {
        "paths": {
            "artifacts": "artifacts",
            "results": "results/generated",
            "checkpoints": "checkpoints",
        },
        "training": {"epochs": 15},
        "dataset": {"classes": 2, "attack_per_class": 1},
        "attack": {
            "norm": "linf",
            "epsilon": 4.0 / 255.0,
            "step_size": 1.0 / 255.0,
            "evaluation_steps": 20,
            "evaluation_restarts": 5,
            "evaluation_seed": 42,
            "random_start": True,
        },
        "calibration": {
            "target_coverage": 0.9,
            "temperature_bounds": [0.05, 10.0],
            "ece_bins": 15,
        },
        "corruptions": {
            "gaussian_noise": [0.02, 0.05],
            "gaussian_blur": [0.75, 1.5],
            "brightness": [0.8, 0.6],
            "contrast": [0.75, 0.5],
        },
    }
    config = ExperimentConfig(tmp_path / "configs/experiment.yaml", tmp_path, raw, "fresh")
    test = [_record("cat-a", 0), _record("cat-b", 0), _record("dog-a", 1), _record("dog-b", 1)]
    splits = {
        "test": test,
        "attack": [test[1], test[2]],
        "calibration": [_record("cal-cat", 0), _record("cal-dog", 1)],
    }
    state: dict[str, Any] = {
        "config": config,
        "splits": splits,
        "calls": [],
        "model_loads": [],
        "policy_fits": [],
        "method_sha256": "synthetic-evaluation-method-v1",
        "real_clean": evaluation._collect_clean_or_shift,
        "real_attack": evaluation._collect_attack,
        "real_method_sha256": evaluation._evaluation_method_sha256,
    }
    for arm in ("standard", "adversarial"):
        path = checkpoint_path(config, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(arm.encode())
    monkeypatch.setattr(evaluation, "load_registered_splits", lambda config: splits)
    monkeypatch.setattr(
        evaluation, "experiment_provenance", lambda config: {"config_sha256": config.sha256}
    )
    monkeypatch.setattr(
        evaluation, "_evaluation_method_sha256", lambda config: state["method_sha256"]
    )
    monkeypatch.setattr(evaluation, "memory_snapshot", lambda device: {"synthetic": True})
    monkeypatch.setattr(evaluation, "synchronize", lambda device: None)

    def load_model(config: Any, checkpoint: Path, device: torch.device) -> Any:
        state["model_loads"].append(checkpoint.parent.name)
        return nn.Linear(3, 2)

    def clean(
        model: Any,
        records: list[SampleRecord],
        config: Any,
        device: Any,
        transform: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        state["calls"].append(kwargs["description"])
        return _arrays(records)

    def attack(
        model: Any,
        records: list[SampleRecord],
        config: Any,
        device: Any,
        kind: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        state["calls"].append(kwargs["description"])
        return _arrays(records, kind)

    original_fit = evaluation._policy_from_calibration

    def fit(config: ExperimentConfig, arrays: Any) -> Any:
        state["policy_fits"].append(config.sha256)
        return original_fit(config, arrays)

    monkeypatch.setattr(evaluation, "load_checkpoint_model", load_model)
    monkeypatch.setattr(evaluation, "_collect_clean_or_shift", clean)
    monkeypatch.setattr(evaluation, "_collect_attack", attack)
    monkeypatch.setattr(evaluation, "_policy_from_calibration", fit)
    return state


def test_clean_section_only_and_repeated_cache_has_zero_model_work(harness: dict[str, Any]) -> None:
    config = harness["config"]
    first = evaluation.evaluate(config, torch.device("cpu"), section="clean", progress=False)
    assert first["status"] == "partial"
    assert len(harness["calls"]) == 4  # calibration and full-test clean for each arm.
    assert len(harness["policy_fits"]) == 2
    assert len(harness["model_loads"]) == 2
    for arm in first["arms"].values():
        assert set(arm["full_test"]) == {"clean"}
        assert not arm["attack_subset"]
    assert not (config.project_path("results") / "evaluation.json").exists()
    harness["calls"].clear()
    harness["policy_fits"].clear()
    harness["model_loads"].clear()
    repeated = evaluation.evaluate(config, torch.device("cpu"), section="clean", progress=False)
    assert repeated["arms"] == first["arms"]
    assert not harness["calls"] and not harness["model_loads"] and not harness["policy_fits"]


@pytest.mark.parametrize("section", ["gaussian_noise", "gaussian_blur", "brightness", "contrast"])
def test_corruption_section_only_runs_selected_family(
    harness: dict[str, Any], section: evaluation.EvaluationSection
) -> None:
    config = harness["config"]
    result = evaluation.evaluate(config, torch.device("cpu"), section=section, progress=False)
    for arm in result["arms"].values():
        assert len(arm["full_test"]) == 3
        assert all(name == "clean" or name.startswith(section) for name in arm["full_test"])
        assert not arm["attack_subset"]
    assert len(harness["calls"]) == 8
    assert all("attack" not in call for call in harness["calls"])


@pytest.mark.parametrize("section", ["fgsm", "pgd"])
def test_attack_section_only_runs_selected_attack(
    harness: dict[str, Any], section: evaluation.EvaluationSection
) -> None:
    result = evaluation.evaluate(
        harness["config"], torch.device("cpu"), section=section, progress=False
    )
    for arm in result["arms"].values():
        assert set(arm["full_test"]) == {"clean"}
        assert set(arm["attack_subset"]) == {section}
        assert "attack_strength_check" not in arm
    assert len(harness["calls"]) == 8
    assert not any("gaussian" in call for call in harness["calls"])


def test_all_sections_publish_complete_manifest_with_full_call_metric_parity(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = harness["config"]
    device = torch.device("cpu")
    for section in evaluation.EVALUATION_SECTIONS:
        staged = evaluation.evaluate(config, device, section=section, progress=False)
    assert staged["status"] == "complete"
    evaluation.assert_evaluation_complete(config, staged)
    saved = json.loads((config.project_path("results") / "evaluation.json").read_text())
    assert saved["status"] == "complete" and saved["arms"] == staged["arms"]
    harness["calls"].clear()
    harness["model_loads"].clear()
    harness["policy_fits"].clear()
    full = evaluation.evaluate(config, device, progress=False)
    assert full["arms"] == staged["arms"]
    assert full["secondary_policy_shift"] == staged["secondary_policy_shift"]
    assert not harness["calls"] and not harness["model_loads"] and not harness["policy_fits"]
    # Also compare to a fresh complete call rather than just loading staged arrays.
    separate = ExperimentConfig(config.path, config.root / "fresh-full", config.raw, config.sha256)
    for arm in ("standard", "adversarial"):
        path = checkpoint_path(separate, arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(arm.encode())
    fresh = evaluation.evaluate(separate, device, progress=False)
    for arm in ("standard", "adversarial"):
        assert fresh["arms"][arm]["full_test"] == staged["arms"][arm]["full_test"]
        assert fresh["arms"][arm]["attack_subset"] == staged["arms"][arm]["attack_subset"]


def test_partial_progress_survives_interruption_and_does_not_repeat_finished_condition(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = evaluation._collect_clean_or_shift

    def stop_second_level(*args: Any, **kwargs: Any) -> Any:
        if kwargs["description"] == "standard gaussian_noise-0.05":
            raise RuntimeError("synthetic interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluation, "_collect_clean_or_shift", stop_second_level)
    config = harness["config"]
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        evaluation.evaluate(config, torch.device("cpu"), section="gaussian_noise", progress=False)
    partial = json.loads((config.project_path("results") / "evaluation.partial.json").read_text())
    assert "gaussian_noise-0.02" in partial["arms"]["standard"]["full_test"]
    assert "gaussian_noise-0.05" not in partial["arms"]["standard"]["full_test"]
    assert not (config.project_path("results") / "evaluation.json").exists()
    monkeypatch.setattr(evaluation, "_collect_clean_or_shift", original)
    harness["calls"].clear()
    resumed = evaluation.evaluate(
        config, torch.device("cpu"), section="gaussian_noise", progress=False
    )
    assert "standard gaussian_noise-0.02" not in harness["calls"]
    assert "standard gaussian_noise-0.05" in harness["calls"]
    assert len(resumed["arms"]["standard"]["full_test"]) == 3


@pytest.mark.parametrize("damage", ["hash", "ids", "labels", "species", "nonfinite"])
def test_invalid_cached_evidence_is_preserved_and_rejected(
    harness: dict[str, Any], damage: str
) -> None:
    config = harness["config"]
    evaluation.evaluate(config, torch.device("cpu"), section="clean", progress=False)
    path = evaluation._arrays_path(config, "full-test/standard", "clean")
    arrays = evaluation.load_arrays(path)
    if damage == "ids":
        arrays["sample_ids"] = arrays["sample_ids"][::-1]
    elif damage == "labels":
        arrays["labels"] = 1 - arrays["labels"]
    elif damage == "species":
        arrays["species"] = 1 - arrays["species"]
    elif damage == "nonfinite":
        arrays["logits"][0, 0] = np.nan
    else:
        arrays["logits"][0, 0] += 0.125
    atomic_save_npz(path, arrays)
    if damage != "hash":
        metadata = json.loads(path.with_suffix(".json").read_text())
        metadata["npz_sha256"] = sha256_file(path)
        atomic_write_json(path.with_suffix(".json"), metadata)
    damaged_hash = sha256_file(path)
    harness["calls"].clear()
    with pytest.raises(evaluation.EvaluationError, match="mismatch|aligned|non-finite"):
        evaluation.evaluate(config, torch.device("cpu"), section="clean", progress=False)
    assert not harness["calls"] and sha256_file(path) == damaged_hash


def test_stale_provenance_recomputes_only_selected_prerequisite(harness: dict[str, Any]) -> None:
    config = harness["config"]
    evaluation.evaluate(config, torch.device("cpu"), section="clean", progress=False)
    path = evaluation._arrays_path(config, "full-test/standard", "clean").with_suffix(".json")
    metadata = json.loads(path.read_text())
    metadata["provenance"]["checkpoint_sha256"] = "old-checkpoint"
    atomic_write_json(path, metadata)
    harness["calls"].clear()
    harness["policy_fits"].clear()
    evaluation.evaluate(config, torch.device("cpu"), section="clean", progress=False)
    assert harness["calls"] == ["standard clean test"]
    assert not harness["policy_fits"]


def test_evaluation_method_change_refits_policy_and_only_recomputes_requested_family(
    harness: dict[str, Any],
) -> None:
    config = harness["config"]
    old = evaluation.evaluate(config, torch.device("cpu"), progress=False)
    unchanged = {
        path: sha256_file(path)
        for arm in ("standard", "adversarial")
        for path in (
            evaluation._arrays_path(config, f"attack/{arm}", "fgsm"),
            evaluation._arrays_path(config, f"attack/{arm}", "pgd"),
            evaluation._arrays_path(config, f"full-test/{arm}", "brightness-0.6"),
        )
    }
    policy_path = evaluation._policy_path(config, "standard")
    old_policy_hash = sha256_file(policy_path)
    harness["method_sha256"] = "synthetic-evaluation-method-v2"
    # Report rejection is based on method identity, even when all old file bytes
    # and their registered receipts remain untouched.
    with pytest.raises(evaluation.EvaluationError, match="method/provenance"):
        evaluation.assert_evaluation_complete(config, old)
    harness["calls"].clear()
    harness["model_loads"].clear()
    harness["policy_fits"].clear()
    current = evaluation.evaluate(
        config, torch.device("cpu"), section="gaussian_noise", progress=False
    )
    assert current["status"] == "partial"
    assert len(harness["calls"]) == 8  # calibration, clean and two noise levels per arm.
    assert len(harness["policy_fits"]) == 2 and len(harness["model_loads"]) == 2
    assert all(
        "calibration" in call or "clean test" in call or "gaussian_noise" in call
        for call in harness["calls"]
    )
    for arm in current["arms"].values():
        assert arm["provenance"]["evaluation_method_sha256"] == harness["method_sha256"]
        assert len(arm["full_test"]) == 3 and not arm["attack_subset"]
    assert sha256_file(policy_path) != old_policy_hash
    assert all(sha256_file(path) == old_hash for path, old_hash in unchanged.items())
    # Complete old evidence is not silently republished as a complete new run.
    recorded = json.loads((config.project_path("results") / "evaluation.json").read_text())
    assert (
        recorded["arms"]["standard"]["provenance"]["evaluation_method_sha256"]
        != (harness["method_sha256"])
    )


def test_evaluation_method_fingerprint_requires_sources_and_covers_calibration(
    harness: dict[str, Any],
) -> None:
    config = harness["config"]
    fingerprint = harness["real_method_sha256"]
    with pytest.raises(evaluation.EvaluationError, match="missing evaluation scientific source"):
        fingerprint(config)
    for relative in evaluation.EVALUATION_METHOD_FILES:
        path = config.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic source for {relative}\n", encoding="utf-8")
    before = fingerprint(config)
    path = config.root / "src/calibration.py"
    path.write_text("# synthetic changed calibration method\n", encoding="utf-8")
    assert fingerprint(config) != before


def test_complete_gate_rejects_partial_and_changed_checkpoint(harness: dict[str, Any]) -> None:
    config = harness["config"]
    result = evaluation.evaluate(config, torch.device("cpu"), section="clean", progress=False)
    with pytest.raises(evaluation.EvaluationError, match="incomplete"):
        evaluation.assert_evaluation_complete(config, result)
    result = evaluation.evaluate(config, torch.device("cpu"), progress=False)
    checkpoint_path(config, "adversarial").write_bytes(b"changed checkpoint")
    with pytest.raises(evaluation.EvaluationError, match="checkpoint"):
        evaluation.assert_evaluation_complete(config, result)


def test_single_arm_selection_preserves_pilot_full_schema(harness: dict[str, Any]) -> None:
    result = evaluation.evaluate(
        harness["config"], torch.device("cpu"), arms=("adversarial",), progress=False
    )
    assert result["status"] == "complete"
    assert set(result["arms"]) == {"adversarial"}
    assert len(result["arms"]["adversarial"]["full_test"]) == 9
    assert set(result["arms"]["adversarial"]["attack_subset"]) == {"fgsm", "pgd"}
    evaluation.assert_evaluation_complete(harness["config"], result, arms=("adversarial",))


@pytest.mark.parametrize("damage", ["missing_receipt", "policy_hash"])
def test_reporting_gate_rejects_incomplete_or_modified_receipts(
    harness: dict[str, Any], damage: str
) -> None:
    config = harness["config"]
    result = evaluation.evaluate(config, torch.device("cpu"), progress=False)
    if damage == "missing_receipt":
        del result["arms"]["standard"]["per_sample"]["full_test/clean"]
    else:
        path = Path(result["arms"]["standard"]["policy_cache"]["path"])
        value = json.loads(path.read_text())
        value["policy"]["threshold"] = 0.999999
        atomic_write_json(path, value)
    with pytest.raises(evaluation.EvaluationError, match="receipt"):
        evaluation.assert_evaluation_complete(config, result)


def test_actual_pixels_are_captured_during_collection_without_extra_forwards(
    harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    records = harness["splits"]["test"]
    pixels = torch.full((4, 3, 4, 4), 0.5)
    labels = torch.tensor([item.label for item in records])
    species = torch.tensor([item.species for item in records])
    ids = [item.sample_id for item in records]
    monkeypatch.setattr(
        evaluation, "build_loader", lambda *args, **kwargs: [(pixels, labels, ids, species)]
    )

    class Model(nn.Module):
        calls = 0

        def forward_with_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls += 1
            return torch.zeros(len(inputs), 2), inputs.mean(dim=(2, 3))

    def attack(*args: Any, **kwargs: Any) -> Any:
        return AttackResult(
            adversarial=args[1] + 0.01,
            best_loss=torch.ones(4),
            cumulative_predictions=(torch.zeros(4, dtype=torch.long),),
        )

    model = Model()
    monkeypatch.setattr(evaluation, "fgsm_attack", attack)
    attacked = harness["real_attack"](
        model,
        records,
        harness["config"],
        torch.device("cpu"),
        "fgsm",  # type: ignore[arg-type]
    )
    assert model.calls == 1
    np.testing.assert_array_equal(attacked["gallery_clean_pixels"], pixels.numpy())
    np.testing.assert_array_equal(attacked["gallery_attack_pixels"], (pixels + 0.01).numpy())
    model.calls = 0
    shifted = harness["real_clean"](
        model,
        records,
        harness["config"],
        torch.device("cpu"),  # type: ignore[arg-type]
        lambda values, sample_ids: values * 0.6,
    )
    assert model.calls == 1
    np.testing.assert_array_equal(shifted["gallery_pixels"], (pixels * 0.6).numpy())
