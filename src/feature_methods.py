"""Input-level probes of learned CNN channels, without changing model state.

Activation maps and input gradients describe responses to particular inputs.
Activation maximization produces a synthetic preference, not a recovered training
image or proof that a channel corresponds to one semantic object part.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from typing import TypedDict

import torch
from torch import nn


class FeatureMethodError(RuntimeError):
    """A visualization input or numerical result is invalid."""


class _LayerCaptured(RuntimeError):
    """Internal early-forward exit: no downstream blocks are needed."""


class OcclusionResult(TypedDict):
    score_drop: torch.Tensor
    positions: torch.Tensor
    base_margin: float


@contextmanager
def _evaluation_state(model: nn.Module, *, freeze: bool = False) -> Iterator[None]:
    """Preserve mixed module modes, requires-grad flags, and existing gradients."""

    modes = [(module, module.training) for module in model.modules()]
    parameters = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    model.eval()
    if freeze:
        for parameter, _requires_grad in parameters:
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
        # Calling train() recursively would destroy deliberately mixed modes.
        for module, training in modes:
            module.training = training


def _validate_pixels(pixels: torch.Tensor, *, single: bool = False) -> None:
    if pixels.ndim != 4 or pixels.shape[1] != 3 or min(pixels.shape) < 1:
        raise FeatureMethodError("pixels must have non-empty shape N×3×H×W")
    if single and pixels.shape[0] != 1:
        raise FeatureMethodError("this method requires exactly one image")
    if not pixels.is_floating_point() or not bool(torch.isfinite(pixels).all()):
        raise FeatureMethodError("pixels must contain finite floating-point values")
    if float(pixels.detach().min()) < 0 or float(pixels.detach().max()) > 1:
        raise FeatureMethodError("pixels must be in [0, 1] before model normalization")


def _module(model: nn.Module, layer: str) -> nn.Module:
    try:
        return model.get_submodule(layer)
    except AttributeError as error:
        raise FeatureMethodError(f"unknown model layer: {layer}") from error


@contextmanager
def _differentiable_activation(
    model: nn.Module, layer: str, *, stop_at_layer: bool = False
) -> Iterator[dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> torch.Tensor:
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise FeatureMethodError("selected layer must return N×C×H×W activations")
        anchor = output.clone()
        captured["activation"] = anchor
        if stop_at_layer:
            raise _LayerCaptured
        # Keep the differentiation anchor separate from downstream in-place ReLUs.
        return anchor.clone()

    handle = _module(model, layer).register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def _forward_to_activation(model: nn.Module, inputs: torch.Tensor) -> None:
    with suppress(_LayerCaptured):
        model(inputs)


def _activation(captured: dict[str, torch.Tensor], channel: int | None = None) -> torch.Tensor:
    if "activation" not in captured:
        raise FeatureMethodError("selected layer did not execute")
    activation = captured["activation"]
    if channel is not None and not 0 <= channel < activation.shape[1]:
        raise FeatureMethodError(f"channel {channel} is outside selected layer")
    return activation


def capture_activations(
    model: nn.Module, pixels: torch.Tensor, layers: list[str]
) -> dict[str, torch.Tensor]:
    """Return cloned N×C×H×W responses in evaluation mode, with hooks removed."""

    _validate_pixels(pixels)
    if len(set(layers)) != len(layers):
        raise FeatureMethodError("layer names must be unique")
    captures: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(
        name: str,
    ) -> Callable[[nn.Module, tuple[object, ...], object], None]:
        def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise FeatureMethodError("selected layer must return N×C×H×W activations")
            captures[name] = output.detach().clone()

        return hook

    try:
        for name in layers:
            handles.append(_module(model, name).register_forward_hook(make_hook(name)))
        with _evaluation_state(model), torch.no_grad():
            model(pixels)
        if set(captures) != set(layers):
            raise FeatureMethodError("one or more selected layers did not execute")
    finally:
        for handle in handles:
            handle.remove()
    return captures


def grad_cam(
    model: nn.Module, pixels: torch.Tensor, target: int, layer: str = "network.layer4"
) -> torch.Tensor:
    """Return an unnormalized nonnegative 2-D Grad-CAM for the target logit.

    This is class attribution, not the receptive field or a channel visualization.
    The returned map retains the selected layer's native spatial resolution.
    """

    _validate_pixels(pixels, single=True)
    inputs = pixels.detach().clone().requires_grad_(True)
    with (
        _evaluation_state(model, freeze=True),
        torch.enable_grad(),
        _differentiable_activation(model, layer) as captured,
    ):
        logits = model(inputs)
        if not isinstance(logits, torch.Tensor) or not 0 <= target < logits.shape[1]:
            raise FeatureMethodError("target class is outside model logits")
        activation = _activation(captured)
        gradient = torch.autograd.grad(logits[0, target], activation)[0]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        result = (weights * activation).sum(dim=1).relu()[0]
    if not bool(torch.isfinite(result).all()):
        raise FeatureMethodError("nonfinite Grad-CAM")
    return result.detach()


def channel_saliency(
    model: nn.Module, pixels: torch.Tensor, layer: str, channel: int
) -> torch.Tensor:
    """Return absolute RGB-mean input gradients of one channel's peak response.

    A large value means local infinitesimal sensitivity at this input. It does
    not prove that the region caused learning or that removing it changes class.
    """

    _validate_pixels(pixels, single=True)
    inputs = pixels.detach().clone().requires_grad_(True)
    with (
        _evaluation_state(model, freeze=True),
        torch.enable_grad(),
        _differentiable_activation(model, layer, stop_at_layer=True) as captured,
    ):
        _forward_to_activation(model, inputs)
        activation = _activation(captured, channel)
        spatial = activation[0, channel].flatten()
        # Match the first peak site used by the displayed receptive-field box,
        # rather than distributing gradients across every tied maximum.
        objective = spatial[spatial.argmax()]
        gradient = torch.autograd.grad(objective, inputs)[0]
    result = gradient[0].abs().mean(dim=0)
    if not bool(torch.isfinite(result).all()):
        raise FeatureMethodError("nonfinite channel saliency")
    return result.detach()


def activation_maximize(
    model: nn.Module,
    layer: str,
    channel: int,
    device: torch.device,
    seed: int,
    steps: int = 60,
    lr: float = 0.05,
    tv_weight: float = 0.0005,
    l2_weight: float = 0.001,
    size: int = 224,
) -> torch.Tensor:
    """Optimize bounded synthetic CHW pixels for a channel's spatial mean.

    Deterministic private CPU randomness initializes and jitters the input. Adam
    only owns the pixels; model parameters, gradients, modes and BN stay intact.
    TV and distance-to-gray penalties favor smoother, less extreme images.
    """

    if steps < 0 or lr <= 0 or size < 2 or tv_weight < 0 or l2_weight < 0:
        raise FeatureMethodError("invalid activation-maximization settings")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = (0.5 + 0.1 * torch.randn(1, 3, size, size, generator=generator)).clamp(0, 1)
    pixels = initial.to(device).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([pixels], lr=lr)
    with (
        _evaluation_state(model, freeze=True),
        torch.enable_grad(),
        _differentiable_activation(model, layer, stop_at_layer=True) as captured,
    ):
        for _step in range(steps):
            shifts = torch.randint(-8, 9, (2,), generator=generator).tolist()
            jittered = torch.roll(pixels, shifts=(shifts[0], shifts[1]), dims=(2, 3))
            _forward_to_activation(model, jittered)
            activation = _activation(captured, channel)
            objective = activation[0, channel].mean()
            tv = (pixels[:, :, 1:, :] - pixels[:, :, :-1, :]).abs().mean()
            tv = tv + (pixels[:, :, :, 1:] - pixels[:, :, :, :-1]).abs().mean()
            loss = -objective + tv_weight * tv + l2_weight * (pixels - 0.5).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FeatureMethodError("nonfinite activation-maximization objective")
            gradient = torch.autograd.grad(loss, pixels)[0]
            if not bool(torch.isfinite(gradient).all()):
                raise FeatureMethodError("nonfinite activation-maximization gradient")
            optimizer.zero_grad(set_to_none=True)
            pixels.grad = gradient
            optimizer.step()
            with torch.no_grad():
                pixels.clamp_(0, 1)
            captured.clear()
    return pixels.detach()[0]


def _positions(length: int, patch_size: int, stride: int) -> list[int]:
    last = max(0, length - patch_size)
    positions = list(range(0, last + 1, stride))
    if positions[-1] != last:
        positions.append(last)
    return positions


def occlusion_map(
    model: nn.Module,
    pixels: torch.Tensor,
    target: int,
    patch_size: int = 32,
    stride: int = 16,
    batch_size: int = 16,
) -> OcclusionResult:
    """Measure target-versus-other logit-margin drops after mean-color occlusion.

    Positive drops identify patches whose replacement reduced the chosen class
    margin; negative values helped that margin. Overlapping patches are not
    independent and the filled images are an artificial distribution shift.
    """

    _validate_pixels(pixels, single=True)
    if min(patch_size, stride, batch_size) < 1 or target not in {0, 1}:
        raise FeatureMethodError("occlusion requires positive sizes and a binary target")
    height, width = pixels.shape[-2:]
    xs = _positions(width, patch_size, stride)
    ys = _positions(height, patch_size, stride)
    boxes = [
        (x, y, min(width, x + patch_size), min(height, y + patch_size)) for y in ys for x in xs
    ]
    fill = pixels.mean(dim=(2, 3), keepdim=True)
    drops: list[torch.Tensor] = []
    with _evaluation_state(model), torch.no_grad():
        logits = model(pixels)
        if not isinstance(logits, torch.Tensor) or logits.shape != (1, 2):
            raise FeatureMethodError("occlusion expects exactly two class logits")
        base = logits[0, target] - logits[0, 1 - target]
        for start in range(0, len(boxes), batch_size):
            chunk = boxes[start : start + batch_size]
            occluded = pixels.repeat(len(chunk), 1, 1, 1)
            for index, (x0, y0, x1, y1) in enumerate(chunk):
                occluded[index, :, y0:y1, x0:x1] = fill[0]
            scores = model(occluded)
            margins = scores[:, target] - scores[:, 1 - target]
            drops.append(base - margins)
    score_drop = torch.cat(drops).reshape(len(ys), len(xs)).detach()
    if not bool(torch.isfinite(score_drop).all()) or not bool(torch.isfinite(base)):
        raise FeatureMethodError("nonfinite occlusion margin")
    return {
        "score_drop": score_drop,
        "positions": torch.tensor(boxes, dtype=torch.int64),
        "base_margin": float(base),
    }


def receptive_field_box(layer: str, y: int, x: int, size: int = 224) -> tuple[int, int, int, int]:
    """Return the clipped theoretical maximum ResNet-18 receptive-field box.

    Coordinates are pixel edges (x0, y0, x1, y1). Residual paths have smaller
    fields; this longest convolutional path is not measured effective evidence.
    Odd, symmetric padding leaves the first response center at pixel center .5.
    Central layer4 positions already span a 224px image; edge sites retain some
    clipped boundary asymmetry despite their 435px theoretical field.
    """

    specifications = {
        "network.relu": (7, 2),
        "network.layer2": (99, 8),
        "network.layer4": (435, 32),
    }
    if layer not in specifications or min(y, x) < 0 or size < 1:
        raise FeatureMethodError("unknown receptive-field layer or invalid coordinate")
    field, jump = specifications[layer]
    dimension = (size + jump - 1) // jump
    if y >= dimension or x >= dimension:
        raise FeatureMethodError("receptive-field coordinate outside layer response")
    x0 = x * jump - (field - 1) // 2
    y0 = y * jump - (field - 1) // 2
    return max(0, x0), max(0, y0), min(size, x0 + field), min(size, y0 + field)
