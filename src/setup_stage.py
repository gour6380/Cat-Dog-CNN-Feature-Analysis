"""One-time official input and provenance setup."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from src.config import ExperimentConfig
from src.data import create_data_manifests, download_official_dataset
from src.io_utils import (
    atomic_write_json,
    environment_snapshot,
    git_state,
    sha256_file,
    source_hash,
    utc_now,
)
from src.model import download_pretrained_state, initialization_manifest
from src.progress import status, tqdm


class SetupError(RuntimeError):
    """Standalone local setup invariant failed."""


def _remote_names(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "remote"], check=False, capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def setup_experiment(config: ExperimentConfig, *, progress: bool = True) -> dict[str, Any]:
    if _remote_names(config.root):
        raise SetupError("this local-only Week 3 project must not have a Git remote")
    for key in ("data", "weights", "checkpoints", "artifacts", "results", "figures", "reports"):
        config.project_path(key).mkdir(parents=True, exist_ok=True)
    status("Setup: verifying the official Oxford-IIIT Pet files...", enabled=progress)
    with tqdm(total=4, desc="setup stages", unit="stage", disable=not progress) as stages:
        download_official_dataset(config.project_path("data"))
        stages.update(1)
        data_manifest = create_data_manifests(config)
        stages.update(1)
        pretrained_manifest = download_pretrained_state(config)
        stages.update(1)
        initialization = initialization_manifest(config)
        stages.update(1)
    requirements = config.root / "requirements.txt"
    if not requirements.is_file():
        raise SetupError("canonical requirements.txt is missing")
    result = {
        "schema_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "config_sha256": config.sha256,
        "label_mode": config.label_mode,
        "target_classes": config.integer("dataset", "classes"),
        "source_sha256": source_hash(config.root),
        "requirements_sha256": sha256_file(requirements),
        "data": data_manifest,
        "pretrained": pretrained_manifest,
        "initialization": initialization,
        "environment": environment_snapshot(),
        "git": git_state(config.root),
        "remote_names": [],
        "public_actions": "none",
    }
    atomic_write_json(config.project_path("artifacts") / "setup.json", result)
    status(
        "Setup complete: data, model initialization, and provenance are registered.",
        enabled=progress,
    )
    return result
