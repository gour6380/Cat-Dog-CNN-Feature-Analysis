from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml  # type: ignore[import-untyped]

from src.config import ConfigError, load_config

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def test_registered_config_loads() -> None:
    config = load_config(CONFIG)
    assert config.value("experiment", "id", str) == "oxford-pets-adversarial-representations-v1"
    assert config.integer("dataset", "classes") == 37


def test_default_attack_values() -> None:
    config = load_config(CONFIG)
    assert config.number("attack", "epsilon") == 4 / 255
    assert config.number("attack", "step_size") == 1 / 255
    assert config.integer("attack", "evaluation_restarts") == 5


def test_project_paths_cannot_escape() -> None:
    config = load_config(CONFIG)
    config.raw["paths"]["data"] = "../../outside"
    with pytest.raises(ConfigError, match="escapes"):
        config.project_path("data")


def test_public_actions_are_disabled() -> None:
    config = load_config(CONFIG)
    assert config.section("experiment")["public_actions_authorized"] is False


def test_default_projection_values() -> None:
    config = load_config(CONFIG)
    assert config.integer("dataset", "projection_per_class") == 10
    assert config.integer("representations", "tsne_iterations") == 1500
    assert config.value("representations", "umap_metric", str) == "cosine"


def _write_config(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "configs" / "experiment.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_tunable_values_are_not_hard_coded(tmp_path: Path) -> None:
    original = load_config(CONFIG)
    raw = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    cast(dict[str, Any], raw["training"])["epochs"] = 1
    cast(dict[str, Any], raw["training"])["warmup_epochs"] = 0
    cast(dict[str, Any], raw["training"])["micro_batch_size"] = 8
    cast(dict[str, Any], raw["attack"])["epsilon"] = 8 / 255
    cast(dict[str, Any], raw["attack"])["train_steps"] = 3
    cast(dict[str, Any], raw["attack"])["evaluation_steps"] = 10
    cast(dict[str, Any], raw["attack"])["evaluation_restarts"] = 2
    cast(dict[str, Any], raw["calibration"])["target_coverage"] = 0.8
    cast(dict[str, Any], raw["representations"])["tsne_iterations"] = 500

    changed = load_config(_write_config(tmp_path, raw))
    assert changed.integer("training", "epochs") == 1
    assert changed.integer("training", "micro_batch_size") == 8
    assert changed.number("attack", "epsilon") == 8 / 255
    assert changed.integer("attack", "evaluation_steps") == 10
    assert changed.number("calibration", "target_coverage") == 0.8
    assert changed.integer("representations", "tsne_iterations") == 500
    assert changed.sha256 != original.sha256


def test_invalid_ranges_still_fail_early(tmp_path: Path) -> None:
    raw = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    cast(dict[str, Any], raw["training"])["epochs"] = 0
    with pytest.raises(ConfigError, match="training.epochs must be positive"):
        load_config(_write_config(tmp_path, raw))
