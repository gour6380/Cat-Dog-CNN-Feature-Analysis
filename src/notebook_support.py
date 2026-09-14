"""Safe notebook facade over the same implementation used by the CLI."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal, cast

import torch

from src.config import ExperimentConfig, load_config
from src.data import load_registered_splits
from src.io_utils import environment_snapshot, sha256_file
from src.model import build_model
from src.progress import status
from src.runtime import require_device

ROOT = Path(__file__).resolve().parents[1]
NotebookStage = Literal["preflight", "train", "evaluate", "represent", "report"]
Arm = Literal["standard", "adversarial"]


def require_project_environment() -> None:
    expected = (ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError(
            "Wrong notebook interpreter. Select Python (Oxford Pets Adversarial "
            f"Representations), or {expected / 'bin/python'}, restart the kernel, "
            f"and rerun the setup cell. Current interpreter: {sys.executable}"
        )


def notebook_context(*, full: bool = False) -> tuple[ExperimentConfig, bool, str]:
    """Resolve the local configuration and enforce an explicit long-run opt-in."""

    require_project_environment()
    config_path = os.environ.get(
        "OXFORD_PETS_NOTEBOOK_CONFIG", str(ROOT / "configs" / "experiment.yaml")
    )
    config = load_config(config_path)
    override = os.environ.get("OXFORD_PETS_NOTEBOOK_RUN_FULL")
    if override not in {None, "0", "1"}:
        raise ValueError("OXFORD_PETS_NOTEBOOK_RUN_FULL must be 0 or 1")
    selected_full = full if override is None else override == "1"
    device = os.environ.get("OXFORD_PETS_NOTEBOOK_DEVICE", "mps")
    if device not in {"mps", "cpu"}:
        raise ValueError("OXFORD_PETS_NOTEBOOK_DEVICE must be mps or cpu")
    return config, selected_full, device


def inspect_environment(config: ExperimentConfig, requested: str) -> dict[str, Any]:
    requirements = ROOT / "requirements.txt"
    return {
        "environment": environment_snapshot(),
        "requested_device": requested,
        "training_device_locked_by_config": config.value("training", "device", str),
        "silent_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "requirements": str(requirements.relative_to(ROOT)),
        "requirements_sha256": sha256_file(requirements),
        "scientific_training": "not started by this inspection cell",
    }


def _per_class(records: list[Any], classes: int) -> list[int]:
    counts = [0] * classes
    for record in records:
        label = getattr(record, "label", None)
        if not isinstance(label, int) or not 0 <= label < classes:
            raise RuntimeError("registered split contains an invalid class label")
        counts[label] += 1
    return counts


def inspect_data(config: ExperimentConfig) -> dict[str, Any]:
    status("Notebook: checking the registered split and fixed sample sets...")
    splits = load_registered_splits(config)
    classes = config.integer("dataset", "classes")
    manifest_path = config.project_path("artifacts") / "data" / "manifest.json"
    manifest_raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_raw, dict):
        raise RuntimeError("data manifest is invalid")
    manifest = cast(dict[str, Any], manifest_raw)
    return {
        "training": len(splits["training"]),
        "calibration": len(splits["calibration"]),
        "official_test": len(splits["test"]),
        "attack_subset": len(splits["attack"]),
        "projection_subset": len(splits["projection"]),
        "classes": classes,
        "training_per_class": _per_class(splits["training"], classes),
        "calibration_per_class": _per_class(splits["calibration"], classes),
        "test_per_class": _per_class(splits["test"], classes),
        "dataset_sha256": manifest.get("dataset_content_sha256"),
        "split_sha256": manifest.get("split_sha256"),
        "official_test_preserved": True,
    }


def inspect_model(config: ExperimentConfig) -> dict[str, Any]:
    status("Notebook: constructing the registered ResNet-18 on CPU...")
    model = build_model(config).eval()
    size = config.integer("input", "size")
    with torch.inference_mode():
        logits, features = model.forward_with_features(torch.zeros(1, 3, size, size))
    result = {
        "architecture": config.value("model", "architecture", str),
        "weights": config.value("model", "weights", str),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": [1, 3, size, size],
        "logit_shape": list(logits.shape),
        "feature_shape": list(features.shape),
        "normalization": "inside the model; attacks operate on raw [0,1] pixels",
        "device": "cpu inspection only",
    }
    del model, logits, features
    return result


def run_stage(
    config: ExperimentConfig,
    stage: NotebookStage,
    *,
    full: bool,
    device: str,
    arm: Arm | None = None,
    progress: bool = True,
) -> object:
    label = f"{arm} training" if stage == "train" else stage
    if not full:
        status(
            f"{label}: skipped in safe mode. Set RUN_FULL_EXPERIMENT = True in the "
            "setup cell and rerun that cell to enable scientific work.",
            enabled=progress,
        )
        return {
            "stage": stage,
            "status": "skipped in safe mode",
            "results_available": False,
        }
    require_project_environment()
    status(f"Starting {label} (requested device: {device})...", enabled=progress)
    try:
        if stage == "preflight":
            from src.preflight import run_preflight

            result = run_preflight(config, require_device(device), progress=progress)
        elif stage == "train":
            from src.training import train_arm

            if arm is None:
                raise ValueError("notebook train stage requires an arm")
            locked = config.value("training", "device", str)
            if device != locked:
                raise ValueError(f"training is locked to {locked}, not {device}")
            result = train_arm(config, arm, require_device(locked), progress=progress)
        elif stage == "evaluate":
            from src.evaluation import evaluate

            result = evaluate(config, require_device(device), progress=progress)
        elif stage == "represent":
            from src.representations import represent

            result = represent(config, progress=progress)
        elif stage == "report":
            from src.reporting import report

            result = report(config, progress=progress)
        else:
            raise AssertionError(f"unsupported notebook stage: {stage}")
    except BaseException:
        status(f"{label} stopped before completion. See the error below.", enabled=progress)
        raise
    status(f"{label} completed.", enabled=progress)
    return result


def inventory(config: ExperimentConfig) -> dict[str, list[str]]:
    return {
        folder: [
            str(path.relative_to(config.root))
            for path in sorted((config.root / folder).rglob("*"))
            if path.is_file()
        ][:100]
        for folder in ("artifacts", "checkpoints", "results", "figures", "reports")
    }


def show(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str), flush=True)
