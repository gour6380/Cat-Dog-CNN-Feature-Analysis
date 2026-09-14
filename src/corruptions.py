"""Registered deterministic post-transform pixel-space shifts."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from torchvision.transforms import functional as tvf

CorruptionName = Literal["gaussian_noise", "gaussian_blur", "brightness", "contrast"]


@dataclass(frozen=True)
class Corruption:
    name: CorruptionName
    value: float

    @property
    def identifier(self) -> str:
        return f"{self.name}-{self.value:g}"


def registered_corruptions(raw: dict[str, object]) -> list[Corruption]:
    output: list[Corruption] = []
    names: tuple[CorruptionName, ...] = (
        "gaussian_noise",
        "gaussian_blur",
        "brightness",
        "contrast",
    )
    for name in names:
        values = raw.get(name)
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"corruptions.{name} must contain exactly two levels")
        output.extend(Corruption(name, float(value)) for value in values)
    return output


def _noise_for_sample(shape: torch.Size, seed_text: str) -> torch.Tensor:
    value = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(value)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def apply_corruption(
    inputs: torch.Tensor,
    sample_ids: list[str],
    corruption: Corruption,
    seed: int,
) -> torch.Tensor:
    if inputs.shape[0] != len(sample_ids):
        raise ValueError("sample IDs and inputs are not aligned")
    if corruption.name == "gaussian_noise":
        noise = torch.stack(
            [
                _noise_for_sample(inputs[index].shape, f"noise:{seed}:{sample_id}")
                for index, sample_id in enumerate(sample_ids)
            ]
        ).to(device=inputs.device, dtype=inputs.dtype)
        return (inputs + corruption.value * noise).clamp(0.0, 1.0)
    if corruption.name == "gaussian_blur":
        radius = max(1, math.ceil(3.0 * corruption.value))
        kernel = 2 * radius + 1
        return cast(
            torch.Tensor,
            tvf.gaussian_blur(inputs, [kernel, kernel], [corruption.value, corruption.value]).clamp(
                0.0, 1.0
            ),
        )
    if corruption.name == "brightness":
        return (inputs * corruption.value).clamp(0.0, 1.0)
    if corruption.name == "contrast":
        mean = inputs.mean(dim=(-2, -1), keepdim=True)
        return (mean + corruption.value * (inputs - mean)).clamp(0.0, 1.0)
    raise ValueError(f"unknown corruption: {corruption.name}")
