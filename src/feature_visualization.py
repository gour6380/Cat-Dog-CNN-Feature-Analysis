"""Layer-wise CNN patterns, real activating regions, and prediction attribution.

All photographic panels are ignored local artifacts. Optimized stimuli and
aggregate charts are separately marked shareable; no paper figure is reused.
"""

from __future__ import annotations

import copy
import io
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch.nn import functional as F

from src.classifier_diagnostics import species_diagnostic_warning, species_prediction_diagnostic
from src.config import ExperimentConfig
from src.feature_methods import (
    activation_maximize,
    capture_activations,
)
from src.feature_section_types import FEATURE_SECTIONS, FeatureSection
from src.io_utils import (
    atomic_save_npz,
    atomic_write_bytes,
    atomic_write_json,
    environment_snapshot,
    sha256_file,
)
from src.runtime import record_failure, synchronize
from src.training import Arm

LAYERS = ["network.relu", "network.layer2", "network.layer4"]
LEVELS = ["Low · stem ReLU", "Mid · layer2", "High · layer4"]
FIELDS = [7, 99, 435]
SPECIES = ["cat", "dog"]
ARMS: tuple[Arm, Arm] = ("standard", "adversarial")
LIMITATIONS = [
    "Responses do not identify which training photograph caused a filter to be learned.",
    "Synthetic patterns are regularized optimized stimuli, not reconstructed pet photographs.",
    "ImageNet pretraining already supplies feature detectors; "
    "changes from initialization are reported.",
    "No channel is claimed to be a verified eye, ear, fur, or other named semantic detector.",
    "Grad-CAM is coarse class attribution; channel sensitivity is a local input gradient.",
    "Theoretical high-layer receptive fields exceed the crop; "
    "an activation cell is not a tiny isolated part.",
    "Occlusion introduces an artificial shift; four fixed anchors support "
    "descriptive, not population, conclusions.",
    "Channels are ranked independently within each arm; "
    "equal channel IDs do not guarantee equal semantics.",
    "One matched model pair is interpreted on clean inputs; this walkthrough "
    "does not measure robust accuracy or establish physical robustness or safety.",
]


def _single_image_pixels(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Collation-free image probes need contiguous NCHW for native MPS backward."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("single image must have shape 3×H×W")
    return image.unsqueeze(0).contiguous().to(device)


def normalize_map(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Display-only scaling, safely handling constant maps."""
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        raise ValueError("empty display map")
    if not np.isfinite(array).all():
        raise ValueError("nonfinite display map")
    span = float(np.ptp(array))
    return (array - array.min()) / span if span > 1e-12 else np.zeros_like(array)


def map_correlation(first: np.ndarray[Any, Any], second: np.ndarray[Any, Any]) -> float | None:
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape:
        raise ValueError("map alignment mismatch")
    a, b = first.ravel(), second.ravel()
    if not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("correlation requires nonempty finite maps")
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def localization_comparison(
    cam: np.ndarray[Any, Any],
    boxes: np.ndarray[Any, Any],
    drops: np.ndarray[Any, Any],
    top_count: int,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    """Compare individual equal-area tiles, not additive joint-mask effects.

    This diagnostic is descriptive for the selected anchors. Random repeats are
    a spatial control, not independent subjects or a population significance test.
    """
    cam, boxes, drops = np.asarray(cam), np.asarray(boxes), np.asarray(drops)
    if cam.ndim != 2 or not cam.size or not np.isfinite(cam).all():
        raise ValueError("localization requires a finite two-dimensional input-space map")
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes):
        raise ValueError("localization requires nonempty N×4 boxes")
    if not np.issubdtype(boxes.dtype, np.integer):
        raise ValueError("localization boxes must use integer pixel-edge coordinates")
    if not drops.size or not np.isfinite(drops).all():
        raise ValueError("localization requires finite occlusion effects")
    height, width = cam.shape
    if (
        np.any(boxes[:, :2] < 0)
        or np.any(boxes[:, 2] > width)
        or np.any(boxes[:, 3] > height)
        or np.any(boxes[:, 2] <= boxes[:, 0])
        or np.any(boxes[:, 3] <= boxes[:, 1])
    ):
        raise ValueError("localization boxes must lie within the input-space map")
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if len(boxes) != drops.size or not np.all(areas == areas[0]):
        raise ValueError("localization requires aligned equal-area occlusion tiles")
    if not 1 <= top_count <= len(boxes) or repeats < 1:
        raise ValueError("invalid localization control count")
    scores = np.array([cam[y0:y1, x0:x1].mean() for x0, y0, x1, y1 in boxes])
    ranked = np.argsort(-scores, kind="stable")[:top_count]
    rng = np.random.default_rng(seed)
    flat = drops.ravel()
    random_means = [
        flat[rng.choice(len(boxes), top_count, replace=False)].mean() for _ in range(repeats)
    ]
    return {
        "top_gradcam_occlusion_mean_drop": float(flat[ranked].mean()),
        "random_occlusion_mean_drop": float(np.mean(random_means)),
    }


def _np(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    return tensor.detach().cpu().numpy()


def _rgb(tensor: torch.Tensor) -> np.ndarray[Any, Any]:
    return _np(tensor).transpose(1, 2, 0)


def _resize_map(tensor: torch.Tensor, size: int, *, mode: str = "bilinear") -> np.ndarray[Any, Any]:
    kwargs: dict[str, Any] = {"size": (size, size), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return _np(F.interpolate(tensor[None, None].float(), **kwargs)[0, 0])


def _save(figure: Any, path: Path) -> None:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    atomic_write_bytes(path, buffer.getvalue())


def _save_figure_record(
    figure: Any,
    path: Path,
    root: Path,
    *,
    arm: str,
    caption: str,
    kind: str,
    shareable: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist an image and its measured identity, with explicit sharing status."""
    _save(figure, path)
    record: dict[str, Any] = {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "caption": caption,
        "kind": kind,
        "arm": arm,
        "shareable": shareable,
    }
    if metadata is not None:
        record["metadata"] = metadata
        for key in ("sample_id", "identity", "config_sha256", "checkpoint_sha256"):
            record[key] = metadata[key]
    return record


def _panel(axis: Any, image: np.ndarray[Any, Any], title: str) -> None:
    axis.imshow(image)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _overlay(
    axis: Any, image: np.ndarray[Any, Any], heat: np.ndarray[Any, Any], title: str
) -> None:
    _panel(axis, image, title)
    axis.imshow(normalize_map(heat), cmap="inferno", alpha=0.48, vmin=0, vmax=1)


def _probe(model: Any, pixels: torch.Tensor, batch: int) -> dict[str, dict[str, torch.Tensor]]:
    means: dict[str, list[torch.Tensor]] = {layer: [] for layer in LAYERS}
    peaks: dict[str, list[torch.Tensor]] = {layer: [] for layer in LAYERS}
    positions: dict[str, list[torch.Tensor]] = {layer: [] for layer in LAYERS}
    for start in range(0, len(pixels), batch):
        for layer, activation in capture_activations(
            model, pixels[start : start + batch], LAYERS
        ).items():
            flattened = activation.flatten(2)
            peak, indices = flattened.max(dim=2)
            means[layer].append(activation.mean(dim=(2, 3)).cpu())
            peaks[layer].append(peak.cpu())
            positions[layer].append(
                torch.stack(
                    (indices // activation.shape[3], indices % activation.shape[3]), dim=-1
                ).cpu()
            )
    return {
        layer: {
            "means": torch.cat(means[layer]),
            "peaks": torch.cat(peaks[layer]),
            "positions": torch.cat(positions[layer]),
        }
        for layer in LAYERS
    }


def _randomized(model: Any, seed: int) -> Any:
    """Full parameter-randomization control, without altering the trained model."""
    result = copy.deepcopy(model)
    with torch.random.fork_rng(devices=[]):
        # manual_seed() also changes device RNGs; only the CPU RNG is needed.
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        # Reset on CPU so the private CPU seed controls all random draws.
        device = next(result.parameters()).device
        result.cpu()
        for module in result.network.modules():
            if isinstance(module, torch.nn.Conv2d | torch.nn.Linear | torch.nn.BatchNorm2d):
                module.reset_parameters()
        result.to(device).eval()
    return result


def _pattern(
    config: ExperimentConfig,
    model: Any,
    arm: str,
    layer: str,
    channel: int,
    device: torch.device,
    identity: str,
    seed: int,
) -> torch.Tensor:
    name = f"{arm}-{layer.split('.')[-1]}-{channel}"
    path = config.project_path("artifacts") / "features" / "synthetic" / f"{name}.npz"
    manifest = path.with_suffix(".json")
    if path.is_file() and manifest.is_file():
        saved = json.loads(manifest.read_text())
        if saved.get("identity") == identity and saved.get("sha256") == sha256_file(path):
            with np.load(path, allow_pickle=False) as arrays:
                return torch.from_numpy(arrays["pixels"].copy()).to(device)
    values = config.section("feature_visualization")
    pixels = activation_maximize(
        model,
        layer,
        channel,
        device,
        seed,
        steps=int(values["activation_maximization_steps"]),
        lr=float(values["activation_maximization_learning_rate"]),
        tv_weight=float(values["total_variation_weight"]),
        l2_weight=float(values["pixel_l2_weight"]),
        size=config.integer("input", "size"),
    )
    size = config.integer("input", "size")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_pixels = (
        (0.5 + 0.1 * torch.randn(1, 3, size, size, generator=generator)).clamp(0, 1).to(device)
    )
    initial_response = float(
        capture_activations(model, initial_pixels, [layer])[layer][0, channel].mean()
    )
    final_response = float(
        capture_activations(model, pixels[None], [layer])[layer][0, channel].mean()
    )
    atomic_save_npz(path, {"pixels": _np(pixels)})
    atomic_write_json(
        manifest,
        {
            "identity": identity,
            "sha256": sha256_file(path),
            "layer": layer,
            "channel": channel,
            "seed": seed,
            "objective": "spatial mean channel activation minus TV and pixel-L2 penalties",
            "initial_unregularized_mean_response": initial_response,
            "final_unregularized_mean_response": final_response,
            "unregularized_response_gain": final_response - initial_response,
            "response_measurement": "Unjittered initial/final bounded input; "
            "raw channel spatial mean. "
            "This is not the regularized, jittered optimization loss.",
        },
    )
    return pixels


def _workflow_figure(size: int) -> Any:
    """Show the actual probed layers and distinguish three visualization questions."""
    figure, axis = plt.subplots(figsize=(19, 5.7))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    texts = [
        f"{size}×{size} RGB\npixels in [0,1]",
        f"Stem conv/BN/ReLU\n64×{(size + 1) // 2}×{(size + 1) // 2}\nRF 7 px",
        f"Residual layer2\n128×{(size + 7) // 8}×{(size + 7) // 8}\nRF 99 px",
        f"Residual layer4\n512×{(size + 31) // 32}×{(size + 31) // 32}\nRF 435 px",
        "Global average pool\n512-D vector",
        "Linear species head\n2 logits\ncat versus dog",
    ]
    width, step, y = 0.142, 0.163, 0.52
    colors = ["#f1f5f9", "#e0f2fe", "#dbeafe", "#ddd6fe", "#e2e8f0", "#dcfce7"]
    for index, (text, color) in enumerate(zip(texts, colors, strict=True)):
        x = 0.02 + index * step
        axis.add_patch(
            Rectangle((x, y), width, 0.29, facecolor=color, edgecolor="#334155", linewidth=1.5)
        )
        axis.text(x + width / 2, y + 0.145, text, ha="center", va="center", fontsize=10)
        if index < len(texts) - 1:
            axis.annotate(
                "",
                xy=(x + step, y + 0.145),
                xytext=(x + width, y + 0.145),
                arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.8},
            )
    for index, text in enumerate(("ImageNet\nnormalize", "pool + layer1", "layer3")):
        center = 0.02 + index * step + (width + step) / 2
        axis.text(center, 0.86, text, ha="center", va="bottom", fontsize=8)
    questions = [
        "CHANNEL PREFERENCES\nSynthetic maximization + top responding real regions\n"
        "What inputs excite a channel?",
        "SPATIAL RESPONSES\nLow/mid/high maps + peak-channel input gradients\n"
        "Where does it respond on this image?",
        "CLASS ATTRIBUTION\nTrue-species Grad-CAM + signed patch occlusion\n"
        "Which regions influence the class score?",
    ]
    for index, question in enumerate(questions):
        x = 0.02 + index * 0.326
        axis.add_patch(Rectangle((x, 0.10), 0.30, 0.26, facecolor="#f8fafc", edgecolor="#94a3b8"))
        axis.text(x + 0.15, 0.23, question, ha="center", va="center", fontsize=9)
    axis.text(
        0.5,
        0.025,
        "RF = longest-path theoretical receptive field, not measured anatomical evidence. "
        "ImageNet pretraining supplies existing detectors.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.suptitle(
        "Cat/dog ResNet-18: forward path and feature-visualization questions", fontsize=16
    )
    return figure


def _context_sources(config: ExperimentConfig) -> dict[str, str]:
    """Hash optional already measured evidence only when its configuration matches."""
    candidates = [config.project_path("results") / "evaluation.json"]
    candidates.extend(config.project_path("artifacts") / "training" / f"{arm}.json" for arm in ARMS)
    sources: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"optional evidence is not a JSON object: {path.name}")
        config_hash = value.get("config_sha256", value.get("provenance", {}).get("config_sha256"))
        complete = path.name == "evaluation.json" or value.get("status") == "complete"
        if config_hash == config.sha256 and complete:
            sources[str(path.relative_to(config.root))] = sha256_file(path)
    return sources


def _context_figures(
    config: ExperimentConfig, sources: dict[str, str]
) -> list[tuple[Any, str, str]]:
    """Use saved aggregate results; this function never evaluates a model."""
    figures: list[tuple[Any, str, str]] = []
    evaluation_path = config.project_path("results") / "evaluation.json"
    if str(evaluation_path.relative_to(config.root)) in sources:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        colors = ["#2563eb", "#ea580c"]
        test_count = config.integer("dataset", "expected_test")
        attack_count = config.integer("dataset", "attack_per_class") * config.integer(
            "model", "classes"
        )
        clean = [evaluation["arms"][arm]["full_test"]["clean"]["accuracy"] * 100 for arm in ARMS]
        axes[0].bar(np.arange(2), clean, color=colors)
        axes[0].set(
            xticks=np.arange(2),
            xticklabels=ARMS,
            ylim=(0, 100),
            ylabel="accuracy (%)",
            title=f"Clean · full official test\nn={test_count}",
        )
        attack_label = (
            f"PGD-{config.integer('attack', 'evaluation_steps')}"
            f"×{config.integer('attack', 'evaluation_restarts')}"
        )
        for index, arm in enumerate(ARMS):
            attacks = evaluation["arms"][arm]["attack_subset"]
            if (
                attacks["fgsm"]["clean_accuracy_on_subset"]
                != attacks["pgd"]["clean_accuracy_on_subset"]
            ):
                raise ValueError("saved attacks do not have the same clean subset accuracy")
            scores = [
                attacks["fgsm"]["clean_accuracy_on_subset"],
                attacks["fgsm"]["robust_accuracy"],
                attacks["pgd"]["robust_accuracy"],
            ]
            axes[1].bar(
                np.arange(3) + (index - 0.5) * 0.35,
                np.array(scores) * 100,
                0.35,
                label=arm,
                color=colors[index],
            )
        axes[1].set(
            xticks=np.arange(3),
            xticklabels=["clean subset", "FGSM", attack_label],
            ylim=(0, 100),
            ylabel="accuracy (%)",
            title=f"Matched attack subset\nn={attack_count}",
        )
        axes[1].legend()
        figure.suptitle("Binary classification context: denominators are shown separately")
        warnings = [
            species_diagnostic_warning(
                species_prediction_diagnostic(evaluation["arms"][arm]["full_test"]["clean"])
            )
            for arm in ARMS
        ]
        if any(warnings):
            figure.suptitle(
                "Binary classification: PGD arm has single-species clean-test predictions"
            )
        figures.append(
            (
                figure,
                "accuracy-context",
                "Saved classification evidence: clean official-test accuracy is separate "
                "from matched-subset clean/FGSM/PGD accuracy. Finite attacks do not "
                "certify robustness, and these values do not validate a named feature. "
                + " ".join(warning for warning in warnings if warning),
            )
        )
    training_paths = [config.project_path("artifacts") / "training" / f"{arm}.json" for arm in ARMS]
    if all(str(path.relative_to(config.root)) in sources for path in training_paths):
        figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        for index, (arm, path) in enumerate(zip(ARMS, training_paths, strict=True)):
            history = json.loads(path.read_text(encoding="utf-8"))["history"]
            epochs = [row["epoch"] for row in history]
            losses = [row["mean_training_loss"] for row in history]
            objective = (
                "clean augmented CE"
                if arm == "standard"
                else f"PGD-{config.integer('attack', 'train_steps')} input CE"
            )
            axes[index].plot(epochs, losses, marker="o", color=["#2563eb", "#ea580c"][index])
            axes[index].set(
                xlabel="epoch",
                ylabel="mean training cross-entropy",
                xticks=epochs,
                title=f"{arm}: {objective}",
            )
            axes[index].grid(alpha=0.2)
        figure.suptitle("Saved training objectives: separate inputs, not one interchangeable loss")
        figures.append(
            (
                figure,
                "training-objectives",
                "Per-epoch saved objectives shown separately: standard CE uses clean "
                "augmented inputs; adversarial CE uses PGD training inputs. Lower loss "
                "across arms is not a matched-objective quality comparison.",
            )
        )
    return figures


def visualize_features(
    config: ExperimentConfig,
    device: torch.device,
    *,
    progress: bool = True,
    section: FeatureSection | None = None,
) -> dict[str, Any]:
    """Protect every stage, and persist synchronized timing/environment for new evidence."""
    try:
        synchronize(device)
        started = time.perf_counter()
        if section is not None and section not in FEATURE_SECTIONS:
            raise ValueError(f"unknown feature section: {section}")
        options: dict[str, Any] = {} if section is None else {"section": section}
        result = _visualize_features(config, device, progress=progress, **options)
        synchronize(device)
        if "elapsed_seconds" not in result:
            result["elapsed_seconds"] = time.perf_counter() - started
            result["environment"] = environment_snapshot()
            atomic_write_json(
                config.project_path("results")
                / (
                    "feature_visualizations.partial.json"
                    if result.get("status") == "partial"
                    else "feature_visualizations.json"
                ),
                result,
            )
        return result
    except BaseException as error:
        record_failure(
            config.project_path("artifacts") / "failures" / "features.json",
            "features",
            error,
            {
                "config_sha256": config.sha256,
                "device": str(device),
                **({"section": section} if section is not None else {}),
            },
        )
        raise


def _visualize_features(
    config: ExperimentConfig,
    device: torch.device,
    *,
    progress: bool = True,
    section: FeatureSection | None = None,
) -> dict[str, Any]:
    """Run one independent family, or assemble the full registered walkthrough."""
    from src.feature_sections import compute_feature_sections

    return compute_feature_sections(config, device, progress=progress, section=section)
