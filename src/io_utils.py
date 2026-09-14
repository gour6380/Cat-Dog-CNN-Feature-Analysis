"""Atomic artifacts, hashes, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write a file durably enough for resumable local experiments."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, value: object) -> None:
    safe = _json_safe(value)
    atomic_write_bytes(
        path,
        json.dumps(safe, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n",
    )


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray[Any, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tree_hash(root: Path, include: Iterable[Path]) -> str:
    """Hash relative names and bytes for an explicit source-file set."""

    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in include), key=lambda item: str(item)):
        relative = path.relative_to(root.resolve())
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def source_hash(root: Path) -> str:
    suffixes = {".py", ".yaml", ".toml", ".md", ".in", ".txt"}
    excluded = {".venv", ".git", "data", "weights", "checkpoints", "artifacts", "results"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and not any(part in excluded for part in path.relative_to(root).parts)
    ]
    return tree_hash(root, files)


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--short"),
    }


def environment_snapshot() -> dict[str, Any]:
    try:
        import torch
        import tqdm

        torch_version: str | None = torch.__version__
        torchvision_version: str | None
        try:
            import torchvision

            torchvision_version = torchvision.__version__
        except ImportError:
            torchvision_version = None
        mps_available = bool(torch.backends.mps.is_available())
        mps_built = bool(torch.backends.mps.is_built())
        tqdm_version: str | None = tqdm.__version__
    except ImportError:
        torch_version = None
        torchvision_version = None
        tqdm_version = None
        mps_available = False
        mps_built = False
    return {
        "captured_at": utc_now(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "torch": torch_version,
        "torchvision": torchvision_version,
        "tqdm": tqdm_version,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "pid": os.getpid(),
    }
