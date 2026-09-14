"""Matched ResNet-18 initialization and feature interface."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import ResNet

from src.config import ExperimentConfig
from src.io_utils import atomic_write_json, sha256_file, utc_now


class ModelError(RuntimeError):
    """Model provenance or architecture invariant failed."""


class NormalizedResNet18(nn.Module):
    """ResNet-18 with normalization inside the model and exposed 512-D features."""

    def __init__(self, network: ResNet, mean: list[float], std: list[float]) -> None:
        super().__init__()
        self.network = network
        self.register_buffer("input_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(std).view(1, 3, 1, 1))

    @property
    def classifier(self) -> nn.Linear:
        if not isinstance(self.network.fc, nn.Linear):
            raise ModelError("ResNet classifier is not linear")
        return self.network.fc

    def normalize(self, pixels: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.input_mean)
        std = cast(torch.Tensor, self.input_std)
        return (pixels - mean) / std

    def forward_features(self, pixels: torch.Tensor) -> torch.Tensor:
        x = self.normalize(pixels)
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)
        x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)
        x = self.network.avgpool(x)
        return torch.flatten(x, 1)

    def forward_with_features(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(pixels)
        return self.classifier(features), features

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        logits, _features = self.forward_with_features(pixels)
        return logits


def atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def pretrained_state_path(config: ExperimentConfig) -> Path:
    return config.project_path("weights") / "resnet18-imagenet1k-v1-state.pt"


def download_pretrained_state(config: ExperimentConfig) -> dict[str, Any]:
    """Cache the pinned official state inside the standalone project."""

    destination = pretrained_state_path(config)
    config.project_path("weights").mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(config.project_path("weights") / "torch-hub"))
    if not destination.is_file():
        state = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)
        atomic_torch_save(destination, state)
    state = torch.load(destination, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "conv1.weight" not in state or "fc.weight" not in state:
        raise ModelError("local ImageNet state is invalid")
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "architecture": "resnet18",
        "weights_enum": "IMAGENET1K_V1",
        "file": str(destination.relative_to(config.root)),
        "sha256": sha256_file(destination),
        "tensor_count": len(state),
    }
    atomic_write_json(config.project_path("artifacts") / "model" / "pretrained.json", manifest)
    return manifest


def load_pretrained_state(config: ExperimentConfig) -> dict[str, torch.Tensor]:
    path = pretrained_state_path(config)
    if not path.is_file():
        raise ModelError("pinned pretrained state is missing; run setup first")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ModelError("pinned pretrained state is not a state dictionary")
    return cast(dict[str, torch.Tensor], value)


def build_model(config: ExperimentConfig) -> NormalizedResNet18:
    """Construct either arm from byte-identical base and new-head initialization."""

    prior_rng = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        network = resnet18(weights=None)
        network.load_state_dict(load_pretrained_state(config), strict=True)
        torch.manual_seed(config.integer("model", "initialization_seed"))
        network.fc = nn.Linear(network.fc.in_features, config.integer("model", "classes"))
    finally:
        torch.set_rng_state(prior_rng)
    mean_raw = config.section("input").get("imagenet_mean")
    std_raw = config.section("input").get("imagenet_std")
    if not isinstance(mean_raw, list) or not isinstance(std_raw, list):
        raise ModelError("ImageNet normalization lists are missing")
    model = NormalizedResNet18(network, [float(x) for x in mean_raw], [float(x) for x in std_raw])
    if model.classifier.in_features != config.integer("model", "feature_dim"):
        raise ModelError("registered penultimate feature dimension does not match model")
    return model


def state_dict_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def initialization_manifest(config: ExperimentConfig) -> dict[str, Any]:
    first = build_model(config)
    second = build_model(config)
    first_hash = state_dict_hash(first.state_dict())
    second_hash = state_dict_hash(second.state_dict())
    if first_hash != second_hash:
        raise ModelError("matched initialization is not byte-identical")
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "seed": config.integer("model", "initialization_seed"),
        "state_sha256": first_hash,
        "matched_rebuild_sha256": second_hash,
        "feature_dim": first.classifier.in_features,
        "classes": first.classifier.out_features,
    }
    atomic_write_json(config.project_path("artifacts") / "model" / "initialization.json", manifest)
    return manifest


def load_checkpoint_model(
    config: ExperimentConfig, checkpoint: Path, device: torch.device
) -> NormalizedResNet18:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ModelError(f"invalid checkpoint: {checkpoint}")
    model = build_model(config)
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model_state"]), strict=True)
    model.to(device)
    model.eval()
    return model
