from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from src.cli import _parser
from src.config import ConfigError, ExperimentConfig, load_config
from src.evaluation import EvaluationError, evaluate
from src.io_utils import atomic_write_json, sha256_file
from src.pilot import assert_pilot_isolated, pilot_summary, verify_preserved_baseline

ROOT = Path(__file__).parents[1]


@pytest.fixture
def current_baseline_pilot() -> ExperimentConfig:
    """Synthetic pilot identity, without changing the archived pilot YAML."""

    archived = load_config(ROOT / "configs/pgd_pilot.yaml")
    original = load_config(ROOT / archived.value("pilot", "original_config", str))
    raw = copy.deepcopy(archived.raw)
    raw["pilot"]["original_config_sha256"] = original.sha256
    return ExperimentConfig(archived.path, archived.root, raw, "synthetic-current-baseline-pilot")


def test_pilot_parser_and_original_configuration_are_separate(
    current_baseline_pilot: ExperimentConfig,
) -> None:
    args = _parser().parse_args(
        ["pilot", "--config", "configs/pgd_pilot.yaml", "--device", "mps", "--no-progress"]
    )
    assert args.command == "pilot" and args.no_progress
    assert_pilot_isolated(current_baseline_pilot)
    original = load_config(ROOT / current_baseline_pilot.value("pilot", "original_config", str))
    assert original.path != current_baseline_pilot.path
    assert original.sha256 == current_baseline_pilot.value("pilot", "original_config_sha256", str)


def test_archived_pilot_rejects_revised_original_configuration_identity() -> None:
    archived = load_config(ROOT / "configs/pgd_pilot.yaml")
    original = load_config(ROOT / archived.value("pilot", "original_config", str))
    assert archived.value("pilot", "original_config_sha256", str) == (
        "3b96c26d328cffc308cb6de70e070748bba3c9fe7a7b229ccba6749f8aec2a91"
    )
    assert original.sha256 != archived.value("pilot", "original_config_sha256", str)
    with pytest.raises(ConfigError, match="original configuration identity changed"):
        assert_pilot_isolated(archived)


def test_pilot_rejects_original_output_collision(current_baseline_pilot: ExperimentConfig) -> None:
    config = current_baseline_pilot
    raw = copy.deepcopy(config.raw)
    raw["paths"]["results"] = "results/generated"
    changed = ExperimentConfig(config.path, config.root, raw, "changed")
    with pytest.raises(ConfigError, match="overwrite"):
        assert_pilot_isolated(changed)


def test_baseline_receipt_detects_changed_or_missing_evidence(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/pgd_pilot.yaml")
    isolated = ExperimentConfig(config.path, tmp_path, config.raw, config.sha256)
    evidence = tmp_path / "original.json"
    atomic_write_json(evidence, {"result": "failed comparison retained"})
    receipt = isolated.project_path("artifacts") / "baseline-preservation.json"
    atomic_write_json(receipt, {"sha256_by_path": {"original.json": sha256_file(evidence)}})
    assert verify_preserved_baseline(isolated)["original.json"] == sha256_file(evidence)
    atomic_write_json(evidence, {"result": "changed"})
    with pytest.raises(ConfigError, match="changed"):
        verify_preserved_baseline(isolated)
    evidence.unlink()
    with pytest.raises(ConfigError, match="missing"):
        verify_preserved_baseline(isolated)


def test_evaluation_rejects_empty_or_duplicate_arm_selection(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/pgd_pilot.yaml")
    isolated = ExperimentConfig(config.path, tmp_path, config.raw, config.sha256)
    for arms in ((), ("adversarial", "adversarial")):
        with pytest.raises(EvaluationError, match="unique"):
            evaluate(isolated, torch.device("cpu"), progress=False, arms=arms)


def test_pilot_terminal_summary_omits_sample_manifest_and_full_shift_matrix() -> None:
    result = {
        "status": "complete",
        "config_sha256": "pilot",
        "baseline_preserved": True,
        "stages": {
            "training_protocol": {"manifest": {"fit_ids": ["private-sample-id"]}},
            "train_adversarial": {"epochs": 15},
            "report": {"notebook": "saved-results.ipynb"},
            "evaluate": {
                "arms": {
                    "adversarial": {
                        "full_test": {
                            "clean": {
                                "accuracy": 0.85,
                                "per_class_accuracy": {"0": 0.54, "1": 1.0},
                            },
                            "blur": {"accuracy": 0.5},
                        },
                        "attack_subset": {"pgd": {"robust_accuracy": 0.025}},
                    }
                }
            },
        },
    }
    summary = pilot_summary(result, Path("pilot-run.json"))
    assert summary["clean_accuracy"] == 0.85
    assert summary["pgd"]["robust_accuracy"] == 0.025
    assert "private-sample-id" not in str(summary)
    assert "blur" not in str(summary)
    assert summary["run_manifest"] == "pilot-run.json"
