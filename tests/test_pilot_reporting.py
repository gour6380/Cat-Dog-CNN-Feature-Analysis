from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.config import ExperimentConfig, load_config
from src.io_utils import atomic_save_npz, atomic_write_bytes, atomic_write_json, sha256_file
from src.metrics import accuracy_metrics
from src.pilot_reporting import PilotReportError, _interpretation, report_pilot
from src.training import checkpoint_path

ROOT = Path(__file__).parents[1]


def _arrays(predictions: list[int]) -> dict[str, np.ndarray[Any, Any]]:
    return {
        "sample_ids": np.asarray(["cat-a", "cat-b", "dog-a", "dog-b"]),
        "labels": np.asarray([0, 0, 1, 1]),
        "logits": np.eye(2)[predictions] * 2,
    }


def _arm(clean_predictions: list[int], pgd_predictions: list[int]) -> dict[str, Any]:
    clean, pgd = _arrays(clean_predictions), _arrays(pgd_predictions)
    clean_correct = clean["logits"].argmax(axis=1) == clean["labels"]
    robust_correct = pgd["logits"].argmax(axis=1) == pgd["labels"]
    success = clean_correct & ~robust_correct
    attacks = {
        "clean_accuracy_on_subset": float(clean_correct.mean()),
        "robust_accuracy": float(robust_correct.mean()),
        "clean_correct_count": int(clean_correct.sum()),
        "attack_success_count": int(success.sum()),
        "attack_success_clean_correct": float(success.sum() / clean_correct.sum()),
        "per_class": {
            str(label): {
                "robust_accuracy": float(robust_correct[clean["labels"] == label].mean()),
                "count": 2,
            }
            for label in (0, 1)
        },
    }
    return {
        "full_test": {"clean": accuracy_metrics(clean["logits"], clean["labels"], 2)},
        "attack_subset": {"fgsm": deepcopy(attacks), "pgd": attacks},
        "checkpoint_sha256": "original-checkpoint",
        "attack_strength_check": {"pgd_not_weaker_by_robust_accuracy": True},
    }


def _fixture(tmp_path: Path) -> ExperimentConfig:
    base = load_config(ROOT / "configs/experiment.yaml")
    raw = deepcopy(base.raw)
    for key, value in {
        "artifacts": "artifacts/pilots/test",
        "results": "results/generated/pilots/test",
        "figures": "figures/generated/pilots/test",
        "reports": "reports/generated/pilots/test",
    }.items():
        raw["paths"][key] = value
    raw["pilot"] = {"enabled": True, "original_config_sha256": "original-config"}
    config_path = tmp_path / "configs/pgd_pilot.yaml"
    atomic_write_bytes(config_path, b"pilot fixture config")
    config = ExperimentConfig(config_path, tmp_path, raw, "pilot-config")
    checkpoint = checkpoint_path(config, "adversarial")
    atomic_write_bytes(checkpoint, b"immutable pilot checkpoint")
    scope = {
        "norm": "linf",
        "epsilon": 4 / 255,
        "step_size": 1 / 255,
        "pgd_steps": 20,
        "pgd_restarts": 5,
        "samples_per_class": 2,
    }
    original = {
        "config_sha256": "original-config",
        "finite_attack_scope": scope,
        "arms": {
            "standard": _arm([0, 0, 1, 1], [1, 1, 0, 0]),
            "adversarial": _arm([1, 1, 1, 1], [1, 1, 1, 1]),
        },
    }
    original_path = tmp_path / "results/generated/evaluation.json"
    atomic_write_json(original_path, original)
    raw["pilot"]["original_evaluation_sha256"] = sha256_file(original_path)
    pilot_arm = _arm([0, 0, 1, 1], [0, 1, 1, 1])
    pilot_arm["checkpoint_sha256"] = sha256_file(checkpoint)
    atomic_write_json(
        config.project_path("results") / "evaluation.json",
        {
            "config_sha256": config.sha256,
            "finite_attack_scope": scope,
            "arms": {"adversarial": pilot_arm},
        },
    )
    for directory, arm, clean, pgd in (
        (original_path.parent, "standard", [0, 0, 1, 1], [1, 1, 0, 0]),
        (original_path.parent, "adversarial", [1, 1, 1, 1], [1, 1, 1, 1]),
        (config.project_path("results"), "adversarial", [0, 0, 1, 1], [0, 1, 1, 1]),
    ):
        atomic_save_npz(directory / "per_sample/attack" / arm / "clean.npz", _arrays(clean))
        atomic_save_npz(directory / "per_sample/attack" / arm / "pgd.npz", _arrays(pgd))
    protocol = {
        "pilot_protocol_sha256": "registered-protocol",
        "counts": {"fit": 40, "validation": 4, "validation_attack": 4},
        "species_weights": {"0": 1.5, "1": 0.75},
    }
    atomic_write_json(config.project_path("artifacts") / "training/validation-split.json", protocol)
    validation = accuracy_metrics(
        _arrays([0, 0, 1, 1])["logits"], _arrays([0, 0, 1, 1])["labels"], 2
    )
    validation["count"] = 4
    history = [
        {
            "epoch": epoch,
            "pilot_phase": "clean_warmup" if epoch <= 3 else "mixed_clean_pgd",
            "mean_training_loss": 0.5,
            "mean_clean_training_loss": 0.4,
            "mean_adversarial_training_loss": None if epoch <= 3 else 0.6,
            "validation": {
                "clean": validation,
                "pgd": validation,
                "protocol_sha256": "registered-protocol",
            },
        }
        for epoch in range(1, 16)
    ]
    atomic_write_json(
        config.project_path("artifacts") / "training/adversarial.json",
        {
            "status": "complete",
            "completed_epochs": 15,
            "completed_updates": 30,
            "pilot_protocol_sha256": "registered-protocol",
            "history": history,
            "provenance": {
                "config_sha256": config.sha256,
                "dataset_sha256": "dataset",
                "initialization_sha256": "initial",
            },
        },
    )
    atomic_write_json(
        config.project_path("artifacts") / "data/manifest.json",
        {"dataset_content_sha256": "dataset"},
    )
    atomic_write_json(
        config.project_path("artifacts") / "model/initialization.json",
        {"initialization_sha256": "initial"},
    )
    return config


def test_pilot_reports_saved_evidence_without_modifying_original(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    original = config.root / "results/generated/evaluation.json"
    baseline_bytes = original.read_bytes()
    result = report_pilot(config, progress=False)
    assert original.read_bytes() == baseline_bytes
    comparison = json.loads(Path(result["comparison"]).read_text())
    assert comparison["paired_attack_evidence"]["same_ids_and_labels"]
    assert comparison["observed_changes"]["pgd_accuracy_vs_original_pgd"] == 0.25
    assert comparison["rows"][-1]["metrics"]["pgd"]["confusion_matrix"] == [[1, 1], [0, 2]]
    text = Path(result["report"]).read_text()
    assert "75.00% PGD accuracy" in text
    assert "fixed epoch 15" in text
    assert "exploratory, not a fresh confirmatory finding" in text
    assert "not a matched reproduction" in text
    assert "Original feature/localization panels" in text
    notebook = json.loads(Path(result["notebook"]).read_text())
    assert all(cell["cell_type"] == "markdown" for cell in notebook["cells"])
    assert sum(bool(cell.get("attachments")) for cell in notebook["cells"]) == 4
    receipt = json.loads(Path(result["release_manifest"]).read_text())
    assert receipt["original_evaluation_sha256"] == sha256_file(original)
    assert receipt["checkpoint_sha256"] == sha256_file(checkpoint_path(config, "adversarial"))
    for item in receipt["files"]:
        assert sha256_file(config.root / item["path"]) == item["sha256"]
    assert all((config.root / figure["path"]).is_file() for figure in result["figures"])


@pytest.mark.parametrize("changed", ["original", "epochs", "checkpoint", "protocol", "scope"])
def test_pilot_rejects_stale_or_incomplete_evidence(tmp_path: Path, changed: str) -> None:
    config = _fixture(tmp_path)
    if changed == "original":
        path = config.root / "results/generated/evaluation.json"
        atomic_write_bytes(path, path.read_bytes() + b" ")
    elif changed == "checkpoint":
        atomic_write_bytes(checkpoint_path(config, "adversarial"), b"changed checkpoint")
    else:
        name = "evaluation.json" if changed == "scope" else "training/adversarial.json"
        path = (
            config.project_path("results")
            if changed == "scope"
            else config.project_path("artifacts")
        ) / name
        value = json.loads(path.read_text())
        if changed == "epochs":
            value["completed_epochs"] = 14
        elif changed == "protocol":
            value["pilot_protocol_sha256"] = "wrong"
        else:
            value["finite_attack_scope"]["pgd_restarts"] = 1
        atomic_write_json(path, value)
    with pytest.raises(PilotReportError):
        report_pilot(config, progress=False)
    assert not (config.project_path("reports") / "pgd-pilot-report.md").exists()


def test_pilot_rejects_different_test_sample_ids(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    changed = _arrays([0, 1, 1, 1])
    changed["sample_ids"][0] = "other"
    atomic_save_npz(
        config.project_path("results") / "per_sample/attack/adversarial/pgd.npz", changed
    )
    with pytest.raises(PilotReportError, match="not aligned"):
        report_pilot(config, progress=False)


def test_pilot_reporting_refuses_original_output_paths(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config.raw["paths"]["reports"] = "reports/generated"
    with pytest.raises(PilotReportError, match="must not overwrite"):
        report_pilot(config, progress=False)


def test_zero_species_recall_keeps_failure_warning(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    result = report_pilot(config, progress=False)
    comparison = json.loads(Path(result["comparison"]).read_text())
    comparison["observed_changes"]["pilot_predicts_both_species_on_clean_test"] = False
    comparison["observed_changes"]["pilot_recognizes_both_species_under_pgd"] = False
    text = _interpretation(comparison)
    assert "remains a failed classifier comparison" in text
    assert "must not be presented as useful robust cat/dog recognition" in text
