"""Measured major-stage CNN walkthroughs; never reconstructed photographs.

Channels are chosen on clean calibration images, then frozen for test anchors.
Every view uses clean inputs; PGD-trained is a training-arm label, not an input
attack. All photographic figures produced here are local-only artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import nn
from torch.nn import functional as F

from src.feature_methods import capture_activations
from src.io_utils import sha256_file

STAGE_LAYERS = [
    "network.relu",
    "network.maxpool",
    "network.layer1",
    "network.layer2",
    "network.layer3",
    "network.layer4",
]
STAGE_LABELS = [
    "Stem ReLU · early responses",
    "Max-pool · downsampled responses",
    "Residual layer1",
    "Residual layer2",
    "Residual layer3",
    "Residual layer4 · late responses",
]
POOL_LAYER = "network.avgpool"
CHANNEL_RULE = "descending mean spatial peak on balanced clean calibration; tie by channel ID"


def rank_stage_channels(
    model: nn.Module, pixels: torch.Tensor, batch_size: int, count: int = 3
) -> dict[str, list[int]]:
    """Keep only accumulated CPU channel scores, not all calibration maps."""
    if batch_size < 1 or count < 1 or len(pixels) < 1:
        raise ValueError("stage selection needs positive batch/count and calibration samples")
    sums: dict[str, torch.Tensor] = {}
    for start in range(0, len(pixels), batch_size):
        captures = capture_activations(model, pixels[start : start + batch_size], STAGE_LAYERS)
        for layer, activation in captures.items():
            # MPS cannot cast to float64, even in a combined CPU transfer/cast.
            # Finish the device transfer first; high-precision accumulation is CPU-only.
            peaks = activation.flatten(2).amax(dim=2).detach().cpu().to(dtype=torch.float64)
            if not bool(torch.isfinite(peaks).all()):
                raise ValueError("nonfinite calibration activation in stage selection")
            score = peaks.sum(dim=0)
            sums[layer] = sums[layer] + score if layer in sums else score
        del captures
    return {
        layer: np.argsort(-sums[layer].numpy() / len(pixels), kind="stable")[
            : min(count, sums[layer].numel())
        ].tolist()
        for layer in STAGE_LAYERS
    }


def _image_panel(axis: Any, image: np.ndarray[Any, Any], title: str) -> None:
    axis.imshow(image)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def create_stage_walkthrough(
    model: nn.Module,
    pixels: torch.Tensor,
    original_image_path: Path,
    selected_channels: dict[str, list[int]],
    *,
    sample_id: str,
    arm: str,
    identity: str,
    config_sha256: str,
    checkpoint_sha256: str,
    true_label: int | None = None,
) -> tuple[Figure, dict[str, Any]]:
    """Capture one real inference and plot its stages without altering parameters.

    Softmax is explicitly raw/un-calibrated; it is not a calibrated confidence
    policy. An overlay only locates a response, not semantic or causal evidence.
    """
    if pixels.ndim != 4 or pixels.shape[0] != 1:
        raise ValueError("stage walkthrough needs one NCHW input")
    captures = capture_activations(model, pixels, [*STAGE_LAYERS, POOL_LAYER])
    pooled = captures[POOL_LAYER].flatten(1)
    classifier = model.get_submodule("network.fc")
    if not isinstance(classifier, nn.Linear):
        raise ValueError("stage walkthrough requires the linear species classifier")
    with torch.no_grad():
        logits = classifier(pooled)[0].detach().cpu()
        probabilities = logits.softmax(dim=0)
    if logits.numel() != 2 or not bool(torch.isfinite(logits).all()):
        raise ValueError("stage walkthrough requires two finite species logits")
    vector = pooled[0].detach().cpu().numpy()
    if not np.isfinite(vector).all() or len(vector) < 4:
        raise ValueError("stage walkthrough needs finite pooled features")
    image = pixels[0].detach().cpu().numpy().transpose(1, 2, 0)
    with Image.open(original_image_path) as original:
        original_pixels = np.asarray(original.convert("RGB")).copy()
    size = tuple(int(value) for value in pixels.shape[-2:])
    figure, axes = plt.subplots(9, 4, figsize=(18, 24), squeeze=False)
    _image_panel(
        axes[0, 0],
        original_pixels,
        f"Original dataset photograph\n{original_pixels.shape[1]}×{original_pixels.shape[0]} RGB",
    )
    _image_panel(
        axes[0, 1], image, f"Actual evaluated input\n{size[1]}×{size[0]} RGB, pixels [0,1]"
    )
    for axis in axes[0, 2:]:
        axis.axis("off")
    axes[0, 2].text(
        0.05,
        0.7,
        "Resize + center crop\nthen ImageNet normalization\ninside the model.\n\n"
        "Maps are measured responses,\nnot reconstructed photographs.",
        transform=axes[0, 2].transAxes,
        fontsize=11,
        va="top",
    )
    axes[0, 3].text(
        0.05,
        0.7,
        "Shown channels were ranked\non balanced clean calibration\n"
        "images, not this test image.\n\nAn overlay is not proof of\na named body-part detector.",
        transform=axes[0, 3].transAxes,
        fontsize=11,
        va="top",
    )
    stages: list[dict[str, Any]] = []
    for row, (layer, label) in enumerate(zip(STAGE_LAYERS, STAGE_LABELS, strict=True), 1):
        activation = captures[layer][0]
        channels = selected_channels.get(layer, [])[:3]
        if not channels or any(not 0 <= channel < activation.shape[0] for channel in channels):
            plt.close(figure)
            raise ValueError(f"invalid calibration-selected channels for {layer}")
        maps = activation[channels].detach().cpu()
        if not bool(torch.isfinite(maps).all()):
            plt.close(figure)
            raise ValueError("nonfinite stage activation")
        maximum = max(float(maps.max()), 1e-6)
        for column in range(3):
            axis = axes[row, column]
            if column >= len(channels):
                axis.axis("off")
                continue
            artist = axis.imshow(maps[column].numpy(), cmap="inferno", vmin=0, vmax=maximum)
            axis.set_title(
                f"{label}\nC×H×W={tuple(activation.shape)} · ch {channels[column]}\n"
                f"native activation grid · common row scale [0,{maximum:.3g}]",
                fontsize=9,
            )
            axis.axis("off")
            figure.colorbar(artist, ax=axis, shrink=0.6)
        strongest = maps[0]
        low, high = float(strongest.min()), float(strongest.max())
        # Normalize at native resolution first: interpolation rounding on a
        # constant response must not turn into a fictitious localization map.
        tolerance = max(1e-12, max(abs(low), abs(high)) * 1e-6)
        normalized = (strongest - low) / (high - low) if high - low > tolerance else strongest * 0
        display_heat = F.interpolate(
            normalized[None, None], size=size, mode="bilinear", align_corners=False
        )[0, 0]
        _image_panel(
            axes[row, 3],
            image,
            f"Channel {channels[0]} response location\nupsampled, display-normalized overlay",
        )
        axes[row, 3].imshow(display_heat.numpy(), cmap="inferno", alpha=0.48, vmin=0, vmax=1)
        stages.append(
            {
                "layer": layer,
                "activation_shape_chw": list(activation.shape),
                "selected_channels": channels,
                "overlay_channel": channels[0],
                "display_scale_max": maximum,
                "selection_rule": CHANNEL_RULE,
                "selection_split": "calibration",
            }
        )
    for column, indices in enumerate(np.array_split(np.arange(len(vector)), 4)):
        axis = axes[7, column]
        axis.plot(indices, vector[indices], linewidth=0.8, color="#2563eb")
        axis.set(
            title=f"Global-average-pooled {len(vector)}-D vector\n"
            f"indices {indices[0]}–{indices[-1]}",
            xlabel="feature dimension",
            ylabel="mean layer4 activation",
        )
        axis.grid(alpha=0.2)
    species = ["cat", "dog"]
    axes[8, 0].bar(species, logits.numpy(), color=["#7c3aed", "#ea580c"])
    axes[8, 0].set(title="Linear classifier · two raw logits", ylabel="logit score")
    axes[8, 1].bar(species, probabilities.numpy(), color=["#7c3aed", "#ea580c"])
    axes[8, 1].set(
        title="Raw softmax · not temperature calibrated", ylabel="probability", ylim=(0, 1)
    )
    for axis in axes[8, 2:]:
        axis.axis("off")
    prediction = int(logits.argmax())
    axes[8, 2].text(
        0.05,
        0.75,
        f"Predicted species: {species[prediction]}\n\n"
        f"cat logit: {float(logits[0]):.4f}\ndog logit: {float(logits[1]):.4f}\n\n"
        "Pooling summarizes spatial maps;\nthe linear head combines those values.",
        transform=axes[8, 2].transAxes,
        fontsize=11,
        va="top",
    )
    axes[8, 3].text(
        0.05,
        0.75,
        "Independent overlays are display-scaled.\nBrightness cannot compare models.\n\n"
        "ImageNet pretraining supplies\nexisting feature detectors.\n\n"
        "One image is explanatory evidence,\nnot a robustness or safety guarantee.",
        transform=axes[8, 3].transAxes,
        fontsize=10,
        va="top",
    )
    arm_label = {"standard": "Standard", "adversarial": "PGD-trained"}.get(arm, arm)
    figure.suptitle(
        f"{arm_label} · {sample_id}"
        + (f" · true {species[true_label]}" if true_label in (0, 1) else "")
        + "\nClean input → every major CNN stage → "
        "pooled features → species scores",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
    metadata = {
        "sample_id": sample_id,
        "arm": arm,
        "arm_label": arm_label,
        "input_policy": "clean_only",
        "identity": identity,
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "input_shape_chw": list(pixels.shape[1:]),
        "original_shape_hwc": list(original_pixels.shape),
        "original_image_sha256": sha256_file(original_image_path),
        "stages": stages,
        "pooled_layer": POOL_LAYER,
        "pooled_shape": list(pooled.shape),
        "pooled_features": vector.tolist(),
        "logits": logits.tolist(),
        "raw_softmax_probabilities": probabilities.tolist(),
        "prediction": prediction,
        "true_label": true_label,
        "shareable": False,
        "interpretation": "Measured activations and raw class scores, not reconstructions "
        "or verified semantic detectors.",
    }
    return figure, metadata
