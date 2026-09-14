from __future__ import annotations

import torch
from torch import nn

from src.attacks import AttackSpec, batchnorm_state, fgsm_attack, pgd_attack


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.linear = nn.Linear(16, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.bn(inputs).flatten(1))


def _batch() -> tuple[TinyClassifier, torch.Tensor, torch.Tensor, list[str]]:
    torch.manual_seed(3)
    return (
        TinyClassifier(),
        torch.rand(4, 1, 4, 4),
        torch.tensor([0, 1, 0, 1]),
        ["a", "b", "c", "d"],
    )


def test_epsilon_zero_is_exact_identity() -> None:
    model, inputs, labels, ids = _batch()
    result = pgd_attack(model, inputs, labels, ids, AttackSpec(0, 1, 0, True, 1, 4))
    assert torch.equal(result.adversarial, inputs)


def test_pgd_respects_linf_and_pixel_bounds() -> None:
    model, inputs, labels, ids = _batch()
    epsilon = 4 / 255
    result = pgd_attack(model, inputs, labels, ids, AttackSpec(epsilon, 1 / 255, 5, True, 2, 4))
    assert float((result.adversarial - inputs).abs().max()) <= epsilon + 1e-6
    assert float(result.adversarial.min()) >= 0
    assert float(result.adversarial.max()) <= 1


def test_random_start_is_repeatable_and_seed_isolated() -> None:
    model, inputs, labels, ids = _batch()
    spec = AttackSpec(4 / 255, 1 / 255, 2, True, 1, 10)
    first = pgd_attack(model, inputs, labels, ids, spec).adversarial
    repeat = pgd_attack(model, inputs, labels, ids, spec).adversarial
    other = pgd_attack(
        model, inputs, labels, ids, AttackSpec(4 / 255, 1 / 255, 2, True, 1, 11)
    ).adversarial
    assert torch.equal(first, repeat)
    assert not torch.equal(first, other)


def test_attack_does_not_update_batchnorm_or_parameter_gradients() -> None:
    model, inputs, labels, ids = _batch()
    model.train()
    before = batchnorm_state(model)
    pgd_attack(model, inputs, labels, ids, AttackSpec(4 / 255, 1 / 255, 2, True, 1, 9))
    after = batchnorm_state(model)
    assert model.training
    assert all(
        torch.equal(left, right)
        for name in before
        for left, right in zip(before[name], after[name], strict=True)
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_fgsm_is_bounded() -> None:
    model, inputs, labels, ids = _batch()
    epsilon = 4 / 255
    result = fgsm_attack(model, inputs, labels, ids, epsilon, 7)
    assert float((result.adversarial - inputs).abs().max()) <= epsilon + 1e-6


def test_restart_predictions_accumulate() -> None:
    model, inputs, labels, ids = _batch()
    result = pgd_attack(model, inputs, labels, ids, AttackSpec(4 / 255, 1 / 255, 2, True, 3, 9))
    assert len(result.cumulative_predictions) == 3
    assert all(value.shape == labels.shape for value in result.cumulative_predictions)
