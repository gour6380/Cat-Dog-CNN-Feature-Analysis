from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigError, load_config

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def test_registered_config_loads() -> None:
    config = load_config(CONFIG)
    assert config.value("experiment", "id", str) == "oxford-pets-adversarial-representations-v1"
    assert config.integer("dataset", "classes") == 37


def test_locked_attack_values() -> None:
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


def test_projection_protocol_is_locked() -> None:
    config = load_config(CONFIG)
    assert config.integer("dataset", "projection_per_class") == 10
    assert config.integer("representations", "tsne_iterations") == 1500
    assert config.value("representations", "umap_metric", str) == "cosine"
