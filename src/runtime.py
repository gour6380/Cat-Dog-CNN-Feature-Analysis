"""Runtime gates for deterministic Apple-silicon execution."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from src.io_utils import atomic_write_json, utc_now


class SafetyStop(RuntimeError):
    """A registered safety or numerical gate stopped the experiment."""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def require_device(name: str) -> torch.device:
    """Select the requested device without any scientific fallback."""

    if name == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise SafetyStop(
                "MPS was requested but is not built and available; fallback is forbidden"
            )
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise SafetyStop(f"unsupported device {name!r}; only cpu and mps are explicit")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device: torch.device) -> dict[str, float]:
    virtual = psutil.virtual_memory()
    snapshot = {
        "system_available_gib": virtual.available / 2**30,
        "system_used_percent": float(virtual.percent),
        "process_rss_mib": psutil.Process().memory_info().rss / 2**20,
    }
    if device.type == "mps":
        snapshot["mps_allocated_mib"] = torch.mps.current_allocated_memory() / 2**20
        snapshot["mps_driver_mib"] = torch.mps.driver_allocated_memory() / 2**20
    return snapshot


def ensure_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise SafetyStop(f"non-finite {name} detected")


def ensure_memory(device: torch.device, minimum_available_gib: float) -> dict[str, float]:
    snapshot = memory_snapshot(device)
    if snapshot["system_available_gib"] < minimum_available_gib:
        raise SafetyStop(
            "unsafe memory pressure: "
            f"{snapshot['system_available_gib']:.2f} GiB available, "
            f"requires {minimum_available_gib:.2f} GiB"
        )
    return snapshot


def record_failure(
    destination: Path,
    stage: str,
    error: BaseException,
    context: dict[str, Any] | None = None,
) -> None:
    atomic_write_json(
        destination,
        {
            "status": "failed",
            "stage": stage,
            "recorded_at": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "context": context or {},
        },
    )
