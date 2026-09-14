from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import load_config
from src.reporting import _attack_label, _attack_table, _geometry_table

ROOT = Path(__file__).parents[1]


def test_report_labels_follow_configured_attack_and_neighbours() -> None:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    config.raw["attack"]["evaluation_steps"] = 10
    config.raw["attack"]["evaluation_restarts"] = 2
    config.raw["representations"]["neighbours"] = 3
    attack_metrics: dict[str, Any] = {
        "clean_accuracy_on_subset": 0.8,
        "robust_accuracy": 0.5,
        "attack_success_clean_correct": 0.25,
    }
    evaluation: dict[str, Any] = {
        "arms": {
            arm: {
                "attack_subset": {
                    "fgsm": attack_metrics,
                    "pgd": attack_metrics,
                }
            }
            for arm in ("standard", "adversarial")
        }
    }
    geometry: dict[str, Any] = {
        "cosine_drift": {"median": 0.1},
        "relative_l2_drift": {"median": 0.2},
        "knn": {"pgd_accuracy": 0.4, "pgd_retention": 0.3},
        "linear_cka_clean_to_pgd": 0.7,
    }
    representations: dict[str, Any] = {
        "geometry": {arm: geometry for arm in ("standard", "adversarial")}
    }

    assert _attack_label(config) == "PGD-10×2"
    assert "PGD-10×2 robust" in _attack_table(config, evaluation)
    assert "PGD 3-NN retention" in _geometry_table(config, representations)
