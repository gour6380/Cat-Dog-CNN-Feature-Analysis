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


class SetupError(RuntimeError):
    """Standalone local setup invariant failed."""


def _remote_names(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "remote"], check=False, capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def setup_experiment(config: ExperimentConfig) -> dict[str, Any]:
    if _remote_names(config.root):
        raise SetupError("this local-only Week 3 project must not have a Git remote")
    for key in ("data", "weights", "checkpoints", "artifacts", "results", "figures", "reports"):
        config.project_path(key).mkdir(parents=True, exist_ok=True)
    download_official_dataset(config.project_path("data"))
    data_manifest = create_data_manifests(config)
    pretrained_manifest = download_pretrained_state(config)
    initialization = initialization_manifest(config)
    lock = config.root / "requirements-macos-arm64-py313.lock.txt"
    result = {
        "schema_version": 1,
        "status": "complete",
        "created_at": utc_now(),
        "config_sha256": config.sha256,
        "source_sha256": source_hash(config.root),
        "lock_sha256": sha256_file(lock) if lock.is_file() else None,
        "data": data_manifest,
        "pretrained": pretrained_manifest,
        "initialization": initialization,
        "environment": environment_snapshot(),
        "git": git_state(config.root),
        "remote_names": [],
        "public_actions": "none",
    }
    atomic_write_json(config.project_path("artifacts") / "setup.json", result)
    return result
