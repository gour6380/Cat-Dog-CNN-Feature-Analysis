from __future__ import annotations

import torch

from src.corruptions import Corruption, apply_corruption


def test_noise_is_deterministic_and_batch_order_independent() -> None:
    inputs = torch.full((2, 3, 16, 16), 0.5)
    first = apply_corruption(inputs, ["a", "b"], Corruption("gaussian_noise", 0.05), 4)
    repeat = apply_corruption(inputs, ["a", "b"], Corruption("gaussian_noise", 0.05), 4)
    reversed_result = apply_corruption(
        inputs.flip(0), ["b", "a"], Corruption("gaussian_noise", 0.05), 4
    ).flip(0)
    assert torch.equal(first, repeat)
    assert torch.equal(first, reversed_result)


def test_all_corruptions_stay_in_pixel_bounds() -> None:
    inputs = torch.rand(2, 3, 16, 16)
    for corruption in (
        Corruption("gaussian_noise", 0.05),
        Corruption("gaussian_blur", 1.5),
        Corruption("brightness", 0.6),
        Corruption("contrast", 0.5),
    ):
        shifted = apply_corruption(inputs, ["a", "b"], corruption, 2)
        assert shifted.shape == inputs.shape
        assert 0 <= float(shifted.min()) <= float(shifted.max()) <= 1


def test_nested_noise_severity_uses_same_field() -> None:
    inputs = torch.full((1, 3, 8, 8), 0.5)
    low = apply_corruption(inputs, ["sample"], Corruption("gaussian_noise", 0.02), 5)
    high = apply_corruption(inputs, ["sample"], Corruption("gaussian_noise", 0.05), 5)
    assert torch.allclose((low - inputs) / 0.02, (high - inputs) / 0.05, atol=1e-5)
