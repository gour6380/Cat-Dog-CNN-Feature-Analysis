"""Explicit, isolated exploratory pilot orchestration; never rerun the original arms."""

from __future__ import annotations

import json
from typing import Any

import torch

from src.config import ConfigError, ExperimentConfig, load_config
from src.io_utils import atomic_write_json, sha256_file, utc_now
from src.progress import status, tqdm


def assert_pilot_isolated(config: ExperimentConfig) -> None:
    """Protect completed evidence, without freezing editable training parameters."""

    from src.pilot_protocol import pilot_enabled

    if not pilot_enabled(config):
        raise ConfigError("pilot command requires an explicitly enabled pilot configuration")
    original_path = (config.root / config.value("pilot", "original_config", str)).resolve()
    if not original_path.is_relative_to(config.root) or original_path == config.path:
        raise ConfigError("pilot original_config must name a separate in-project configuration")
    original = load_config(original_path)
    if original.sha256 != config.value("pilot", "original_config_sha256", str):
        raise ConfigError("original configuration identity changed")
    for key in ("artifacts", "results", "figures", "reports"):
        if config.project_path(key) == original.project_path(key):
            raise ConfigError(f"pilot paths.{key} must not overwrite the original output directory")
    if config.sha256[:16] == original.sha256[:16]:
        raise ConfigError("pilot checkpoint namespace collides with the original")


def verify_preserved_baseline(config: ExperimentConfig) -> dict[str, str]:
    """Check the frozen original result files, not subsequently edited source files."""

    receipt = config.project_path("artifacts") / "baseline-preservation.json"
    if not receipt.is_file():
        raise ConfigError("baseline-preservation.json is missing; preserve original evidence first")
    raw = json.loads(receipt.read_text(encoding="utf-8"))
    hashes = raw.get("sha256_by_path") if isinstance(raw, dict) else None
    if not isinstance(hashes, dict) or not hashes:
        raise ConfigError("invalid baseline preservation receipt")
    verified: dict[str, str] = {}
    for relative, expected in hashes.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ConfigError("invalid baseline preservation hash entry")
        path = (config.root / relative).resolve()
        if not path.is_relative_to(config.root) or not path.is_file():
            raise ConfigError(f"preserved baseline file missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ConfigError(f"preserved baseline file changed: {relative}")
        verified[relative] = actual
    return verified


def run_pilot(
    config: ExperimentConfig, device: torch.device, *, progress: bool = True
) -> dict[str, Any]:
    """Run one fixed final-epoch pilot, with no selection, retries or weak attacks."""

    from src.evaluation import evaluate
    from src.pilot_protocol import prepare_pilot_protocol
    from src.pilot_reporting import report_pilot
    from src.preflight import run_preflight
    from src.setup_stage import setup_experiment
    from src.training import train_arm

    assert_pilot_isolated(config)
    preserved_before = verify_preserved_baseline(config)
    stages: dict[str, Any] = {}
    status("Pilot: separate clean-to-mixed PGD run; original evidence is frozen.", enabled=progress)
    with tqdm(total=6, desc="exploratory pilot", unit="stage", disable=not progress) as bar:
        stages["setup"] = setup_experiment(config, progress=progress)
        bar.update(1)
        protocol = prepare_pilot_protocol(config)
        stages["training_protocol"] = {"sha256": protocol.sha256, "manifest": protocol.manifest}
        bar.update(1)
        stages["preflight"] = run_preflight(config, device, progress=progress)
        bar.update(1)
        stages["train_adversarial"] = train_arm(config, "adversarial", device, progress=progress)
        bar.update(1)
        stages["evaluate"] = evaluate(config, device, progress=progress, arms=("adversarial",))
        bar.update(1)
        stages["report"] = report_pilot(config, progress=progress)
        bar.update(1)
    preserved_after = verify_preserved_baseline(config)
    result = {
        "status": "complete",
        "created_at": utc_now(),
        "exploratory": True,
        "config_sha256": config.sha256,
        "stages": stages,
        "baseline_preserved": preserved_before == preserved_after,
        "baseline_hashes": preserved_after,
        "public_actions": "none",
    }
    atomic_write_json(config.project_path("artifacts") / "pilot-run.json", result)
    return result
