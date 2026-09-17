from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml  # type: ignore[import-untyped]
from torchvision.models import resnet18

from src.config import ConfigError, load_config
from src.model import build_model, initialization_manifest, state_dict_hash

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


def test_registered_config_loads() -> None:
    config = load_config(CONFIG)
    assert config.value("experiment", "id", str)
    assert config.label_mode == "species"
    assert config.integer("dataset", "classes") == 2
    assert config.integer("model", "classes") == 2


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


def test_memory_telemetry_has_no_configured_stop_thresholds() -> None:
    preflight = load_config(CONFIG).section("preflight")
    assert "minimum_available_memory_gib" not in preflight
    assert "maximum_memory_growth_mib" not in preflight


def _write_config(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "configs" / "experiment.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
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


def test_species_label_mode_uses_binary_targets(tmp_path: Path) -> None:
    raw = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    raw["dataset"]["label_mode"] = "species"
    raw["dataset"]["classes"] = 2
    raw["model"]["classes"] = 2
    config = load_config(_write_config(tmp_path, raw))
    assert config.label_mode == "species"
    assert config.integer("model", "classes") == 2


def test_breed_label_mode_remains_supported(tmp_path: Path) -> None:
    raw = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    raw["dataset"].pop("label_mode", None)
    raw["dataset"]["classes"] = 37
    raw["model"]["classes"] = 37
    assert load_config(_write_config(tmp_path, raw)).label_mode == "breed"


@pytest.mark.parametrize("mode", ["other", None, ["species"]])
def test_invalid_target_label_mode_is_actionable(tmp_path: Path, mode: object) -> None:
    raw = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    raw["dataset"]["label_mode"] = mode
    with pytest.raises(ConfigError, match="dataset.label_mode must be breed or species"):
        load_config(_write_config(tmp_path, raw))


def test_species_mode_cannot_keep_a_breed_classifier(tmp_path: Path) -> None:
    raw = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    raw["dataset"]["label_mode"] = "species"
    raw["dataset"]["classes"] = 37
    raw["model"]["classes"] = 37
    with pytest.raises(ConfigError, match="species targets use two classes"):
        load_config(_write_config(tmp_path, raw))


def test_binary_head_initialization_is_matched_and_distinct_from_breeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = cast(dict[str, Any], yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    raw["dataset"]["label_mode"] = "species"
    raw["dataset"]["classes"] = 2
    raw["model"]["classes"] = 2
    config = load_config(_write_config(tmp_path, raw))
    # A local untrained base stands in for the official weights so this unit
    # test performs no download, training, or data access.
    base = resnet18(weights=None).state_dict()
    monkeypatch.setattr("src.model.load_pretrained_state", lambda _config: base)
    before_rng = torch.get_rng_state().clone()
    first = build_model(config)
    second = build_model(config)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert first.classifier.out_features == 2
    assert first.classifier.in_features == 512
    first_hash = state_dict_hash(first.state_dict())
    assert state_dict_hash(second.state_dict()) == first_hash
    manifest = initialization_manifest(config)
    assert manifest["state_sha256"] == first_hash
    assert manifest["label_mode"] == "species"
    assert manifest["target_names"] == ["cat", "dog"]
    assert manifest["classes"] == 2
    config.raw["dataset"]["label_mode"] = "breed"
    config.raw["dataset"]["classes"] = 37
    config.raw["model"]["classes"] = 37
    assert state_dict_hash(build_model(config).state_dict()) != first_hash
