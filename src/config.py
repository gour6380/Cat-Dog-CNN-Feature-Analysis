"""Configuration loading and locked-protocol validation."""

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
    """Raised when the registered protocol is missing or has been changed."""


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


LOCKED_VALUES: dict[tuple[str, str], object] = {
    ("experiment", "id"): "oxford-pets-adversarial-representations-v1",
    ("experiment", "expected_python"): "3.13.15",
    ("dataset", "classes"): 37,
    ("dataset", "attack_per_class"): 20,
    ("dataset", "projection_per_class"): 10,
    ("model", "architecture"): "resnet18",
    ("model", "weights"): "IMAGENET1K_V1",
    ("model", "feature_dim"): 512,
    ("input", "size"): 224,
    ("training", "epochs"): 15,
    ("training", "micro_batch_size"): 16,
    ("training", "gradient_accumulation_steps"): 2,
    ("training", "dtype"): "float32",
    ("attack", "norm"): "linf",
    ("attack", "train_steps"): 5,
    ("attack", "evaluation_steps"): 20,
    ("attack", "evaluation_restarts"): 5,
    ("representations", "tsne_iterations"): 1500,
}


def _canonical_hash(raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_locked_protocol(config: ExperimentConfig) -> None:
    """Reject changes that would silently weaken or redirect the experiment."""

    for (section, key), expected in LOCKED_VALUES.items():
        actual = config.section(section).get(key)
        if actual != expected:
            raise ConfigError(
                f"locked field {section}.{key}: expected {expected!r}, got {actual!r}"
            )

    expected_epsilon = 4.0 / 255.0
    expected_step = 1.0 / 255.0
    if abs(config.number("attack", "epsilon") - expected_epsilon) > 1e-15:
        raise ConfigError("attack.epsilon must remain exactly 4/255")
    if abs(config.number("attack", "step_size") - expected_step) > 1e-15:
        raise ConfigError("attack.step_size must remain exactly 1/255")
    if config.number("dataset", "train_fraction") != 0.8:
        raise ConfigError("dataset.train_fraction must remain 0.8")
    if config.number("calibration", "target_coverage") != 0.9:
        raise ConfigError("calibration.target_coverage must remain 0.9")
    if config.section("experiment").get("public_actions_authorized") is not False:
        raise ConfigError("public actions must remain disabled for this local experiment")


def load_config(path: str | Path, *, enforce_python: bool = True) -> ExperimentConfig:
    """Load a YAML config, establish its project root, and validate locked fields."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConfigError(f"configuration does not exist: {resolved}")
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError("configuration root must be a mapping")
    raw = cast(dict[str, Any], loaded)
    root = resolved.parent.parent.resolve()
    config = ExperimentConfig(resolved, root, raw, _canonical_hash(raw))
    validate_locked_protocol(config)
    expected_python = config.value("experiment", "expected_python", str)
    if enforce_python and platform.python_version() != expected_python:
        raise ConfigError(
            f"locked runtime is CPython {expected_python}; running {platform.python_version()}"
        )
    return config
