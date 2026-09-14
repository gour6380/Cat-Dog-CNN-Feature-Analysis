"""Configuration loading and structural validation."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml  # type: ignore[import-untyped]

T = TypeVar("T")


class ConfigError(RuntimeError):
    """Raised when the experiment configuration is missing or invalid."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed wrapper around the registered YAML configuration."""

    path: Path
    root: Path
    raw: dict[str, Any]
    sha256: str

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"configuration section {name!r} is missing or invalid")
        return cast(dict[str, Any], value)

    def value(self, section: str, key: str, expected: type[T]) -> T:
        value = self.section(section).get(key)
        if not isinstance(value, expected):
            raise ConfigError(f"{section}.{key} must be {expected.__name__}")
        return value

    def number(self, section: str, key: str) -> float:
        value = self.section(section).get(key)
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ConfigError(f"{section}.{key} must be numeric")
        return float(value)

    def integer(self, section: str, key: str) -> int:
        value = self.section(section).get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"{section}.{key} must be an integer")
        return value

    def project_path(self, key: str) -> Path:
        relative = self.value("paths", key, str)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ConfigError(f"paths.{key} escapes the standalone project")
        return path


def _canonical_hash(raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_config(config: ExperimentConfig) -> None:
    """Reject malformed or unsupported settings without freezing tunable values."""

    for section in (
        "experiment",
        "paths",
        "dataset",
        "model",
        "input",
        "training",
        "attack",
        "corruptions",
        "calibration",
        "representations",
        "preflight",
    ):
        config.section(section)

    if config.section("experiment").get("public_actions_authorized") is not False:
        raise ConfigError("public actions must remain disabled for this local experiment")

    if config.value("model", "architecture", str) != "resnet18":
        raise ConfigError("model.architecture is unsupported; available: resnet18")
    if config.value("model", "weights", str) != "IMAGENET1K_V1":
        raise ConfigError("model.weights is unsupported; available: IMAGENET1K_V1")
    if config.value("training", "dtype", str) != "float32":
        raise ConfigError("training.dtype is unsupported; available: float32")
    if config.value("training", "device", str) not in {"cpu", "mps"}:
        raise ConfigError("training.device must be cpu or mps")
    if config.value("attack", "norm", str) != "linf":
        raise ConfigError("attack.norm is unsupported; available: linf")

    dataset_classes = config.integer("dataset", "classes")
    model_classes = config.integer("model", "classes")
    if dataset_classes <= 1 or model_classes != dataset_classes:
        raise ConfigError("dataset.classes and model.classes must match and exceed one")

    positive_integers = (
        ("dataset", "expected_trainval"),
        ("dataset", "expected_test"),
        ("dataset", "attack_per_class"),
        ("dataset", "projection_per_class"),
        ("model", "feature_dim"),
        ("input", "size"),
        ("input", "eval_resize"),
        ("training", "epochs"),
        ("training", "micro_batch_size"),
        ("training", "gradient_accumulation_steps"),
        ("training", "evaluation_batch_size"),
        ("attack", "evaluation_restarts"),
        ("calibration", "ece_bins"),
        ("representations", "neighbours"),
        ("representations", "bootstrap_replicates"),
        ("representations", "tsne_iterations"),
        ("representations", "umap_neighbours"),
    )
    for section, key in positive_integers:
        if config.integer(section, key) <= 0:
            raise ConfigError(f"{section}.{key} must be positive")

    if config.integer("dataset", "projection_per_class") > config.integer(
        "dataset", "attack_per_class"
    ):
        raise ConfigError("dataset.projection_per_class cannot exceed attack_per_class")
    if config.integer("training", "num_workers") < 0:
        raise ConfigError("training.num_workers cannot be negative")
    warmup_epochs = config.integer("training", "warmup_epochs")
    if not 0 <= warmup_epochs <= config.integer("training", "epochs"):
        raise ConfigError("training.warmup_epochs must be between zero and training.epochs")
    for key in ("train_steps", "evaluation_steps"):
        if config.integer("attack", key) < 0:
            raise ConfigError(f"attack.{key} cannot be negative")

    train_fraction = config.number("dataset", "train_fraction")
    target_coverage = config.number("calibration", "target_coverage")
    flip_probability = config.number("input", "horizontal_flip_probability")
    if not 0.0 < train_fraction < 1.0:
        raise ConfigError("dataset.train_fraction must be between zero and one")
    if not 0.0 < target_coverage <= 1.0:
        raise ConfigError("calibration.target_coverage must be in (0, 1]")
    if not 0.0 <= flip_probability <= 1.0:
        raise ConfigError("input.horizontal_flip_probability must be in [0, 1]")

    epsilon = config.number("attack", "epsilon")
    step_size = config.number("attack", "step_size")
    if not 0.0 <= epsilon <= 1.0:
        raise ConfigError("attack.epsilon must be in [0, 1]")
    if step_size < 0.0:
        raise ConfigError("attack.step_size cannot be negative")


def load_config(path: str | Path, *, enforce_python: bool = True) -> ExperimentConfig:
    """Load a YAML config, establish its project root, and validate its structure."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConfigError(f"configuration does not exist: {resolved}")
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be a mapping")
    raw = cast(dict[str, Any], loaded)
    root = resolved.parent.parent.resolve()
    config = ExperimentConfig(resolved, root, raw, _canonical_hash(raw))
    validate_config(config)
    expected_python = config.value("experiment", "expected_python", str)
    if enforce_python and platform.python_version() != expected_python:
        raise ConfigError(
            f"configured runtime is CPython {expected_python}; running {platform.python_version()}"
        )
    return config
