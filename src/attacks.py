"""Bounded pixel-space attacks with model-state and determinism guards."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as nnf

from src.runtime import SafetyStop, ensure_finite


@dataclass(frozen=True)
class AttackSpec:
    epsilon: float
    step_size: float
    steps: int
    random_start: bool
    restarts: int
    seed: int


@dataclass(frozen=True)
class AttackResult:
    adversarial: torch.Tensor
    best_loss: torch.Tensor
    cumulative_predictions: tuple[torch.Tensor, ...]


def _seed_value(seed: int, sample_id: str, restart: int) -> int:
    digest = hashlib.sha256(f"attack:{seed}:{restart}:{sample_id}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def _random_delta(
    inputs: torch.Tensor,
    sample_ids: list[str],
    epsilon: float,
    seed: int,
    restart: int,
) -> torch.Tensor:
    if len(sample_ids) != inputs.shape[0]:
        raise ValueError("sample IDs and input batch are not aligned")
    deltas = []
    for index, sample_id in enumerate(sample_ids):
        generator = torch.Generator(device="cpu").manual_seed(_seed_value(seed, sample_id, restart))
        delta = torch.empty(inputs[index].shape, dtype=torch.float32, device="cpu").uniform_(
            -epsilon, epsilon, generator=generator
        )
        deltas.append(delta)
    return torch.stack(deltas).to(device=inputs.device, dtype=inputs.dtype)


def batchnorm_state(model: nn.Module) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    state: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.running_mean is None or module.running_var is None:
                continue
            batches = (
                module.num_batches_tracked.detach().clone()
                if module.num_batches_tracked is not None
                else torch.tensor(0)
            )
            state[name] = (
                module.running_mean.detach().clone(),
                module.running_var.detach().clone(),
                batches,
            )
    return state


def assert_batchnorm_unchanged(
    before: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    after: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    if before.keys() != after.keys():
        raise SafetyStop("BatchNorm module set changed during attack generation")
    for name in before:
        for earlier, later in zip(before[name], after[name], strict=True):
            if not torch.equal(earlier, later):
                raise SafetyStop(f"BatchNorm state changed during attack generation: {name}")


def _single_restart(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    spec: AttackSpec,
    restart: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.epsilon == 0.0 or spec.steps == 0:
        with torch.no_grad():
            loss = nnf.cross_entropy(model(inputs), labels, reduction="none")
        return inputs.detach().clone(), loss.detach()
    if spec.epsilon < 0.0 or spec.step_size <= 0.0 or spec.steps < 1:
        raise ValueError("attack epsilon, step size, and steps are invalid")
    original = inputs.detach()
    adversarial = original.clone()
    if spec.random_start:
        adversarial = (
            adversarial + _random_delta(original, sample_ids, spec.epsilon, spec.seed, restart)
        ).clamp(0.0, 1.0)
    for _ in range(spec.steps):
        adversarial = adversarial.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model(adversarial)
            ensure_finite("attack logits", logits)
            loss = nnf.cross_entropy(logits, labels, reduction="sum")
            gradient = torch.autograd.grad(loss, adversarial, only_inputs=True)[0]
        ensure_finite("attack gradient", gradient)
        adversarial = adversarial.detach() + spec.step_size * gradient.sign()
        delta = (adversarial - original).clamp(-spec.epsilon, spec.epsilon)
        adversarial = (original + delta).clamp(0.0, 1.0).detach()
    with torch.no_grad():
        losses = nnf.cross_entropy(model(adversarial), labels, reduction="none")
    return adversarial, losses.detach()


def pgd_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    spec: AttackSpec,
) -> AttackResult:
    """Return per-example worst loss across deterministic untargeted restarts."""

    if not bool(((inputs >= 0.0) & (inputs <= 1.0)).all().item()):
        raise SafetyStop("attacks require input pixels in [0,1]")
    if spec.restarts < 1:
        raise ValueError("attack requires at least one restart")
    was_training = model.training
    before = batchnorm_state(model)
    model.eval()
    best_adversarial = inputs.detach().clone()
    best_loss = torch.full((inputs.shape[0],), -torch.inf, device=inputs.device)
    cumulative: list[torch.Tensor] = []
    try:
        for restart in range(spec.restarts):
            candidate, losses = _single_restart(model, inputs, labels, sample_ids, spec, restart)
            replace = losses > best_loss
            view_shape = (replace.shape[0],) + (1,) * (candidate.ndim - 1)
            best_adversarial = torch.where(replace.view(view_shape), candidate, best_adversarial)
            best_loss = torch.maximum(best_loss, losses)
            with torch.no_grad():
                cumulative.append(model(best_adversarial).argmax(dim=1).detach().cpu())
    finally:
        model.train(was_training)
    after = batchnorm_state(model)
    assert_batchnorm_unchanged(before, after)
    perturbation = (best_adversarial - inputs).detach()
    maximum = float(perturbation.abs().amax().item())
    if maximum > spec.epsilon + 1e-6:
        raise SafetyStop(f"L-infinity bound violated: {maximum} > {spec.epsilon}")
    if not bool(((best_adversarial >= 0.0) & (best_adversarial <= 1.0)).all().item()):
        raise SafetyStop("pixel clipping failed")
    return AttackResult(best_adversarial.detach(), best_loss.detach(), tuple(cumulative))


def fgsm_attack(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    epsilon: float,
    seed: int,
) -> AttackResult:
    return pgd_attack(
        model,
        inputs,
        labels,
        sample_ids,
        AttackSpec(
            epsilon=epsilon,
            step_size=epsilon if epsilon > 0 else 1.0,
            steps=1 if epsilon > 0 else 0,
            random_start=False,
            restarts=1,
            seed=seed,
        ),
    )
