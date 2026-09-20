"""Typed contract for independently computed feature families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import torch
from torch import nn

from src.config import ExperimentConfig
from src.data import SampleRecord
from src.training import Arm

FeatureSection = Literal[
    "stages",
    "kernels",
    "synthetic",
    "real_patches",
    "activations",
    "gradcam",
    "occlusion",
    "diagnostics",
]
FEATURE_SECTIONS: tuple[FeatureSection, ...] = (
    "stages",
    "kernels",
    "synthetic",
    "real_patches",
    "activations",
    "gradcam",
    "occlusion",
    "diagnostics",
)


@dataclass
class SectionData:
    payload: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray[Any, Any]] = field(default_factory=dict)


class FeatureContext(Protocol):
    config: ExperimentConfig
    device: torch.device
    arm: Arm
    model: nn.Module
    identity: str
    references: list[SampleRecord]
    anchors: list[SampleRecord]
    progress: bool

    def reference_pixels(self) -> torch.Tensor: ...
    def reference_probe(self, *, initial: bool = False) -> dict[str, dict[str, torch.Tensor]]: ...
    def selected_channels(self) -> dict[str, list[int]]: ...
    def stage_channels(self) -> dict[str, list[int]]: ...
    def initial_kernels(self) -> torch.Tensor: ...
    def anchor_pixels(self, index: int) -> torch.Tensor: ...
    def dependency(self, section: FeatureSection) -> SectionData: ...
    def register(
        self,
        figure: Any,
        name: str,
        caption: str,
        kind: str,
        *,
        shareable: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
