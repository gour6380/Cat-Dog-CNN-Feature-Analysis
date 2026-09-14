"""Shared tqdm rendering and durable stage-status messages."""

from __future__ import annotations

import sys
from typing import Any

from tqdm.auto import tqdm as AutoProgress


def tqdm(*args: Any, **kwargs: Any) -> Any:
    """Create a consistent progress bar for training, evaluation, and analysis."""

    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.5)
    return AutoProgress(*args, **kwargs)


def status(message: str, *, enabled: bool = True) -> None:
    """Print a durable status line independently of machine-readable artifacts."""

    if enabled:
        print(message, file=sys.stderr, flush=True)
