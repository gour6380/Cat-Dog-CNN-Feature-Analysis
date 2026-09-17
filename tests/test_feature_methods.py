from __future__ import annotations

import pytest
import torch
from torch import nn

from src.feature_methods import (
    FeatureMethodError,
    activation_maximize,
    capture_activations,
    channel_saliency,
    grad_cam,
    occlusion_map,
    receptive_field_box,
)


class TinyNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 6, 3, stride=2, padding=1), nn.BatchNorm2d(6), nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(6, 8, 3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(8, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.relu(self.bn1(self.conv1(inputs)))
        value = self.layer4(self.layer2(value))
        return self.fc(value.mean(dim=(2, 3)))


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = TinyNetwork()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _model() -> tuple[TinyClassifier, torch.Tensor]:
    with torch.random.fork_rng():
        torch.manual_seed(321)
        return TinyClassifier(), torch.rand(1, 3, 12, 12)


def _hook_count(model: nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


def test_capture_returns_detached_native_responses_and_removes_hooks() -> None:
    model, pixels = _model()
    layers = ["network.relu", "network.layer2", "network.layer4"]
    captured = capture_activations(model, pixels.requires_grad_(True), layers)
    assert set(captured) == set(layers)
    assert captured[layers[0]].shape == (1, 4, 12, 12)
    assert captured[layers[1]].shape == (1, 6, 6, 6)
    assert captured[layers[2]].shape == (1, 8, 3, 3)
    assert all(not value.requires_grad for value in captured.values())
    assert _hook_count(model) == 0
    frozen = captured[layers[0]].clone()
    capture_activations(model, torch.zeros_like(pixels), layers)
    assert torch.equal(captured[layers[0]], frozen)


def test_gradient_probes_have_correct_shapes_and_are_finite() -> None:
    model, pixels = _model()
    cam = grad_cam(model, pixels, target=0)
    saliency = channel_saliency(model, pixels, "network.layer2", channel=0)
    assert cam.shape == (3, 3)
    assert saliency.shape == (12, 12)
    assert bool(torch.isfinite(cam).all()) and bool(torch.isfinite(saliency).all())
    assert float(cam.min()) >= 0 and float(saliency.min()) >= 0
    assert not cam.requires_grad and not saliency.requires_grad
    assert _hook_count(model) == 0


def test_tied_channel_peaks_probe_only_the_first_boxed_location() -> None:
    model = nn.Sequential(nn.Identity())
    pixels = torch.zeros(1, 3, 4, 4)
    pixels[0, 0, 1, 1] = pixels[0, 0, 2, 2] = 0.8
    sensitivity = channel_saliency(model, pixels, "0", 0)
    assert sensitivity[1, 1].item() == pytest.approx(1 / 3)
    assert sensitivity[2, 2].item() == 0
    assert torch.count_nonzero(sensitivity) == 1


def test_all_probes_preserve_weights_bn_mixed_modes_flags_and_parameter_gradients() -> None:
    model, pixels = _model()
    model.train()
    model.network.bn1.eval()
    parameters = list(model.parameters())
    parameters[0].requires_grad_(False)
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.full_like(parameter, index + 1.0)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    requires = [parameter.requires_grad for parameter in parameters]
    gradients = [parameter.grad.clone() for parameter in parameters if parameter.grad is not None]
    capture_activations(model, pixels, ["network.relu"])
    grad_cam(model, pixels, target=1)
    channel_saliency(model, pixels, "network.layer4", 1)
    activation_maximize(model, "network.layer2", 0, torch.device("cpu"), seed=7, steps=2, size=12)
    occlusion_map(model, pixels, 0, patch_size=4, stride=4, batch_size=4)
    assert modes == [module.training for module in model.modules()]
    assert requires == [parameter.requires_grad for parameter in parameters]
    assert all(torch.equal(state[name], value) for name, value in model.state_dict().items())
    assert all(
        torch.equal(gradient, parameter.grad)
        for gradient, parameter in zip(gradients, parameters, strict=True)
    )
    assert _hook_count(model) == 0


def test_maximization_is_seeded_bounded_and_global_rng_isolated() -> None:
    model, _pixels = _model()
    rng = torch.get_rng_state().clone()
    first = activation_maximize(
        model, "network.layer2", 0, torch.device("cpu"), 21, steps=2, size=12
    )
    repeat = activation_maximize(
        model, "network.layer2", 0, torch.device("cpu"), 21, steps=2, size=12
    )
    other = activation_maximize(
        model, "network.layer2", 0, torch.device("cpu"), 22, steps=2, size=12
    )
    assert first.shape == (3, 12, 12)
    assert float(first.min()) >= 0 and float(first.max()) <= 1
    assert torch.equal(first, repeat)
    assert not torch.equal(first, other)
    assert torch.equal(rng, torch.get_rng_state())


class PatchClassifier(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs[:, :, :2, :2].sum(dim=(1, 2, 3))
        return torch.stack((score, torch.zeros_like(score)), dim=1)


def test_occlusion_measures_binary_margin_drop_and_reverses_target_sign() -> None:
    model = PatchClassifier()
    pixels = torch.zeros(1, 3, 4, 4)
    pixels[:, :, :2, :2] = 1
    result = occlusion_map(model, pixels, target=0, patch_size=2, stride=2, batch_size=3)
    assert result["base_margin"] == 12
    assert torch.equal(result["score_drop"], torch.tensor([[9.0, 0.0], [0.0, 0.0]]))
    assert result["positions"].tolist() == [[0, 0, 2, 2], [2, 0, 4, 2], [0, 2, 2, 4], [2, 2, 4, 4]]
    opposite = occlusion_map(model, pixels, target=1, patch_size=2, stride=2)
    assert torch.equal(opposite["score_drop"], -result["score_drop"])


def test_occlusion_includes_final_edge_and_handles_large_patch() -> None:
    model = PatchClassifier()
    pixels = torch.rand(1, 3, 7, 9)
    result = occlusion_map(model, pixels, 0, patch_size=3, stride=4)
    assert result["positions"][-1].tolist() == [6, 4, 9, 7]
    assert result["score_drop"].shape == (2, 3)
    large = occlusion_map(model, pixels, 0, patch_size=32)
    assert large["positions"].tolist() == [[0, 0, 9, 7]]
    assert large["score_drop"].shape == (1, 1)


def test_hook_and_state_cleanup_after_failure() -> None:
    model, pixels = _model()
    modes = [module.training for module in model.modules()]
    with pytest.raises(FeatureMethodError, match="channel"):
        channel_saliency(model, pixels, "network.layer2", 999)
    with pytest.raises(FeatureMethodError, match="unknown model layer"):
        capture_activations(model, pixels, ["network.relu", "missing"])
    assert _hook_count(model) == 0
    assert modes == [module.training for module in model.modules()]


def test_nonfinite_optimization_stops_and_restores_state() -> None:
    model, _pixels = _model()
    with torch.no_grad():
        model.network.conv1.weight.fill_(float("nan"))
    with pytest.raises(FeatureMethodError, match="nonfinite"):
        activation_maximize(model, "network.layer2", 0, torch.device("cpu"), 1, steps=1, size=12)
    assert model.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert _hook_count(model) == 0


def test_resnet18_receptive_fields_follow_longest_convolutional_path() -> None:
    assert receptive_field_box("network.relu", 10, 20) == (37, 17, 44, 24)
    assert receptive_field_box("network.relu", 0, 0) == (0, 0, 4, 4)
    assert receptive_field_box("network.layer2", 10, 10) == (31, 31, 130, 130)
    assert receptive_field_box("network.layer4", 3, 3) == (0, 0, 224, 224)
    assert receptive_field_box("network.layer4", 0, 0) == (0, 0, 218, 218)
    with pytest.raises(FeatureMethodError, match="outside"):
        receptive_field_box("network.layer4", 7, 0)
