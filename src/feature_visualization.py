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
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from torch.nn import functional as F

from src.attacks import AttackSpec, pgd_attack
from src.classifier_diagnostics import species_diagnostic_warning, species_prediction_diagnostic
from src.config import ExperimentConfig
from src.data import PetRecordDataset, load_registered_splits, stratified_hash_selection
from src.eda import generate_eda
from src.feature_methods import (
    activation_maximize,
    capture_activations,
    channel_saliency,
    grad_cam,
    occlusion_map,
    receptive_field_box,
)
from src.io_utils import (
    atomic_save_npz,
    atomic_write_bytes,
    atomic_write_json,
    environment_snapshot,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from src.model import build_model, load_checkpoint_model, state_dict_hash
from src.progress import status, tqdm
from src.runtime import memory_snapshot, record_failure, synchronize
from src.training import Arm, checkpoint_path, experiment_provenance

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
    "Only one initialization and finite digital attack family are evaluated; "
    "no physical or safety claim is made.",
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
    config: ExperimentConfig, device: torch.device, *, progress: bool = True
) -> dict[str, Any]:
    """Protect every stage, and persist synchronized timing/environment for new evidence."""
    try:
        synchronize(device)
        started = time.perf_counter()
        result = _visualize_features(config, device, progress=progress)
        synchronize(device)
        if "elapsed_seconds" not in result:
            result["elapsed_seconds"] = time.perf_counter() - started
            result["environment"] = environment_snapshot()
            atomic_write_json(
                config.project_path("results") / "feature_visualizations.json", result
            )
        return result
    except BaseException as error:
        record_failure(
            config.project_path("artifacts") / "failures" / "features.json",
            "features",
            error,
            {"config_sha256": config.sha256, "device": str(device)},
        )
        raise


def _visualize_features(
    config: ExperimentConfig, device: torch.device, *, progress: bool = True
) -> dict[str, Any]:
    """Generate measured feature images from genuine binary checkpoints."""
    if config.label_mode != "species":
        raise ValueError("feature walkthrough requires a genuine cat/dog species head")
    values = config.section("feature_visualization")
    if values.get("layers") != LAYERS:
        raise ValueError("feature methods currently support stem ReLU, layer2 and layer4")
    size = config.integer("input", "size")
    if size % int(values["occlusion_patch_size"]) or int(values["occlusion_stride"]) != int(
        values["occlusion_patch_size"]
    ):
        raise ValueError(
            "equal-area localization uses a nonoverlapping tile grid dividing the input"
        )
    splits = load_registered_splits(config)
    seed = int(values["seed"])
    references = stratified_hash_selection(
        splits["calibration"], int(values["calibration_per_class"]), f"features:{seed}:calibration"
    )
    anchors = stratified_hash_selection(
        splits["test"], int(values["anchors_per_class"]), f"features:{seed}:anchors"
    )
    checkpoints = {arm: sha256_file(checkpoint_path(config, arm)) for arm in ARMS}
    provenance = experiment_provenance(config)
    context_sources = _context_sources(config)
    evaluation_path = config.project_path("results") / "evaluation.json"
    diagnostics: dict[str, dict[str, Any]] = {}
    if str(evaluation_path.relative_to(config.root)) in context_sources:
        evaluation = json.loads(evaluation_path.read_text())
        diagnostics = {
            arm: species_prediction_diagnostic(evaluation["arms"][arm]["full_test"]["clean"])
            for arm in ARMS
        }
    source_files = [
        "src/feature_visualization.py",
        "src/feature_methods.py",
        "src/classifier_diagnostics.py",
        "src/config.py",
        "src/data.py",
        "src/model.py",
        "src/attacks.py",
        "src/eda.py",
        "src/io_utils.py",
        "src/runtime.py",
        "src/training.py",
        "requirements.txt",
    ]
    method_sources = {name: sha256_file(config.root / name) for name in source_files}
    method_hash = sha256_bytes(json.dumps(method_sources, sort_keys=True).encode())
    identity = sha256_bytes(
        json.dumps(
            [config.sha256, checkpoints, method_hash, provenance, context_sources], sort_keys=True
        ).encode()
    )
    destination = config.project_path("results") / "feature_visualizations.json"
    if destination.is_file():
        cached_raw = json.loads(destination.read_text())
        if not isinstance(cached_raw, dict):
            raise ValueError("feature visualization manifest must be a JSON object")
        cached = cast(dict[str, Any], cached_raw)
        if cached.get("identity") == identity and all(
            (config.root / f["path"]).is_file()
            and sha256_file(config.root / f["path"]) == f["sha256"]
            for f in cached["figures"]
        ):
            status("Features: reusing matching feature images and diagnostics.", enabled=progress)
            return cached
    reference_data = PetRecordDataset(references, config, training=False)
    reference_pixels = torch.stack([reference_data[i][0] for i in range(len(references))]).to(
        device
    )
    labels = np.array([record.label for record in references])
    anchor_data = PetRecordDataset(anchors, config, training=False)
    initial = build_model(config).to(device).eval()
    batch = config.integer("training", "evaluation_batch_size")
    baseline = _probe(initial, reference_pixels, batch)
    initial_kernels = initial.network.conv1.weight.detach().cpu().clone()
    initial.cpu()
    del initial
    result: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "identity": identity,
        "config_sha256": config.sha256,
        "method_sha256": method_hash,
        "method_source_files_sha256": method_sources,
        "device": str(device),
        "provenance": provenance,
        "context_evidence_sha256": context_sources,
        "selection": {
            "calibration_sample_ids": [r.sample_id for r in references],
            "test_anchor_ids": [r.sample_id for r in anchors],
            "channel_rule": "six largest balanced-calibration mean peak activations, "
            "tie by channel ID",
        },
        "arms": {},
        "figures": [],
        "limitations": LIMITATIONS,
    }
    root = config.project_path("figures") / "features"

    def register(
        figure: Any, arm: str, name: str, caption: str, kind: str, shareable: bool = False
    ) -> None:
        warning = species_diagnostic_warning(diagnostics.get(arm, {}))
        if warning:
            caption += " " + warning
            title = getattr(figure, "_suptitle", None)
            if title is not None:
                title.set_text(
                    title.get_text().replace(arm, f"{arm} [single-species clean predictions]", 1)
                )
        path = root / f"{arm}-{name}.png"
        _save(figure, path)
        result["figures"].append(
            {
                "path": str(path.relative_to(config.root)),
                "sha256": sha256_file(path),
                "caption": caption,
                "kind": kind,
                "arm": arm,
                "shareable": shareable,
            }
        )

    register(
        _workflow_figure(size),
        "experiment",
        "workflow",
        "Measured-layer architecture and distinct probe questions: synthetic channel preferences, "
        "spatial responses/local sensitivity, and class attribution. Intermediate layer1/layer3 "
        "are included in the forward path, although not selected for image grids.",
        "species_response",
        True,
    )
    for figure, name, caption in _context_figures(config, context_sources):
        register(figure, "experiment", name, caption, "species_response", True)
    eda = generate_eda(config, progress=progress)
    for name, relative in eda["figures"].items():
        result["figures"].append(
            {
                "path": relative,
                "sha256": sha256_file(config.root / relative),
                "caption": f"Dataset EDA: {name.replace('_', ' ')}.",
                "kind": "species_response",
                "arm": "dataset",
                "shareable": True,
            }
        )
    try:
        for arm in ARMS:
            status(
                f"Features: {arm} low/mid/high channel selection and patterns...", enabled=progress
            )
            model = load_checkpoint_model(config, checkpoint_path(config, arm), device)
            before = state_dict_hash(model.state_dict())
            probe = _probe(model, reference_pixels, batch)
            arm_result: dict[str, Any] = {
                "checkpoint_sha256": checkpoints[arm],
                "layers": {},
                "anchors": [],
                "clean_classifier_diagnostic": diagnostics.get(arm, {"available": False}),
            }
            count = int(values["channels_per_layer"])
            synthetic, response_axes = plt.subplots(
                3, count, figsize=(count * 2.4, 8.3), squeeze=False
            )
            response_figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
            change_figure, change_axes = plt.subplots(
                1, 3, figsize=(16, 4.5), constrained_layout=True
            )
            selected_by_layer: dict[str, list[int]] = {}
            for level, layer in enumerate(LAYERS):
                channels = np.argsort(-_np(probe[layer]["peaks"]).mean(axis=0), kind="stable")[
                    :count
                ].tolist()
                selected_by_layer[layer] = channels
                means = _np(probe[layer]["means"])
                old = _np(baseline[layer]["means"])
                class_means = [
                    means[labels == target][:, channels].mean(axis=0).tolist() for target in (0, 1)
                ]
                relative = (
                    (means[:, channels].mean(axis=0) - old[:, channels].mean(axis=0))
                    / np.maximum(np.abs(old[:, channels].mean(axis=0)), 1e-6)
                ).tolist()
                spatial = [
                    size // divisor
                    for divisor in (2 if level == 0 else 8 if level == 1 else 32,) * 2
                ]
                arm_result["layers"][layer] = {
                    "selected_channels": channels,
                    "calibration_mean_by_species": class_means,
                    "relative_change_from_initial": relative,
                    "spatial_shape": spatial,
                    "receptive_field_pixels": FIELDS[level],
                    "synthetic_responses": [],
                }
                for target, color in ((0, "#7c3aed"), (1, "#ea580c")):
                    axes[level].bar(
                        np.arange(count) + (target - 0.5) * 0.35,
                        class_means[target],
                        0.35,
                        label=SPECIES[target],
                        color=color,
                    )
                axes[level].set(
                    xticks=np.arange(count),
                    xticklabels=channels,
                    title=LEVELS[level],
                    xlabel="channel ID",
                    ylabel="mean activation · raw units",
                )
                axes[level].legend()
                change_axes[level].bar(np.arange(count), relative, color="#0f766e")
                change_axes[level].set(
                    xticks=np.arange(count),
                    xticklabels=channels,
                    title=LEVELS[level],
                    xlabel="channel ID",
                    ylabel="relative mean-response change",
                )
                change_axes[level].axhline(0, color="black", linewidth=0.8)
                patch_fig, patch_axes = plt.subplots(
                    count, 4, figsize=(12, count * 2.5), squeeze=False
                )
                for col, channel in enumerate(
                    tqdm(channels, desc=f"{arm} {layer} stimuli", disable=not progress)
                ):
                    pattern = _pattern(
                        config,
                        model,
                        arm,
                        layer,
                        channel,
                        device,
                        identity,
                        seed + level * 10000 + channel,
                    )
                    pattern_manifest = (
                        config.project_path("artifacts")
                        / "features"
                        / "synthetic"
                        / (f"{arm}-{layer.split('.')[-1]}-{channel}.json")
                    )
                    pattern_result = json.loads(pattern_manifest.read_text())
                    arm_result["layers"][layer]["synthetic_responses"].append(pattern_result)
                    _panel(
                        response_axes[level, col],
                        _rgb(pattern),
                        f"{LEVELS[level]}\nchannel {channel} · synthetic\n"
                        f"mean response {pattern_result['initial_unregularized_mean_response']:.2f}"
                        f" → {pattern_result['final_unregularized_mean_response']:.2f}",
                    )
                    top = np.argsort(-_np(probe[layer]["peaks"][:, channel]), kind="stable")[:2]
                    for rank, index in enumerate(top):
                        pixels = reference_pixels[int(index)]
                        y, x = probe[layer]["positions"][int(index), channel].tolist()
                        box = receptive_field_box(layer, int(y), int(x), size)
                        image = _rgb(pixels)
                        x0, y0, x1, y1 = box
                        _panel(
                            patch_axes[col, rank * 2],
                            image,
                            f"ch {channel} · rank {rank + 1}\n"
                            f"{references[int(index)].sample_id}\n"
                            f"peak={probe[layer]['peaks'][int(index), channel]:.3f}",
                        )
                        patch_axes[col, rank * 2].add_patch(
                            Rectangle(
                                (x0, y0),
                                x1 - x0,
                                y1 - y0,
                                fill=False,
                                edgecolor="#00ff88",
                                linewidth=2,
                            )
                        )
                        _panel(
                            patch_axes[col, rank * 2 + 1],
                            image[y0:y1, x0:x1],
                            f"theoretical RF crop\n{FIELDS[level]}px before clipping",
                        )
                patch_fig.suptitle(
                    f"{arm} · {LEVELS[level]} · strongest calibration responses\n"
                    "Green boxes are theoretical receptive fields, "
                    "not proven anatomical concepts",
                    fontsize=12,
                )
                register(
                    patch_fig,
                    arm,
                    f"top-patches-{level}",
                    f"Real calibration examples maximally activating the selected "
                    f"{LEVELS[level]} channels; RF extents clipped to the evaluation crop.",
                    "real_patches",
                )
            synthetic.suptitle(
                f"{arm} · regularized activation-maximization patterns\n"
                "Optimized stimuli, not training photographs or generative reconstructions",
                fontsize=13,
            )
            register(
                synthetic,
                arm,
                "synthetic-atlas",
                "Layer-wise optimized channel preferences. Every tile is independently "
                "optimized from private seeded noise; visual appearance is not evidence "
                "of a named detector.",
                "activation_maximization",
                True,
            )
            response_figure.suptitle(
                f"{arm} · measured responses on 32 cats + 32 dogs from calibration"
            )
            register(
                response_figure,
                arm,
                "species-response",
                "Raw channel response means on balanced calibration samples; activation "
                "scales differ by layer/arm, and brightness in overlays is not an arm comparison.",
                "species_response",
                True,
            )
            change_figure.suptitle(
                f"{arm} · response changes from the identical ImageNet-initialized encoder"
            )
            register(
                change_figure,
                arm,
                "initial-response-change",
                "Relative change on identical calibration images from ImageNet initialization. "
                "This demonstrates change, not that all visible patterns were newly learned here.",
                "initial_response_change",
                True,
            )
            kernel_fig, kernel_axes = plt.subplots(
                3, count, figsize=(count * 2.1, 6.8), squeeze=False
            )
            for column, channel in enumerate(selected_by_layer[LAYERS[0]]):
                old_kernel = _np(initial_kernels[channel]).transpose(1, 2, 0)
                current = _np(model.network.conv1.weight[channel]).transpose(1, 2, 0)
                for row, pattern in enumerate((old_kernel, current, current - old_kernel)):
                    _panel(
                        kernel_axes[row, column],
                        normalize_map(pattern),
                        f"{['ImageNet initial', 'fine-tuned', 'weight difference'][row]}\n"
                        f"conv1 channel {channel}",
                    )
            kernel_fig.suptitle(
                f"{arm} · actual 7×7 RGB convolution kernels\n"
                "Each tile independently scaled; difference colors are not RGB semantics"
            )
            register(
                kernel_fig,
                arm,
                "kernels",
                "Actual first-layer weights, initialized weights, and their differences. "
                "Deeper channel filters have many input channels and are instead probed "
                "with optimization and real examples.",
                "kernels",
                True,
            )
            randomized = _randomized(model, seed)
            attribution_fig, attr_axes = plt.subplots(
                len(anchors),
                4,
                figsize=(14, len(anchors) * 4.0),
                squeeze=False,
                constrained_layout=True,
            )
            for index, record in enumerate(anchors):
                pixels = _single_image_pixels(anchor_data[index][0], device)
                image = _rgb(pixels[0])
                target = record.label
                with torch.no_grad():
                    logits = model(pixels)[0]
                clean = capture_activations(model, pixels, LAYERS)
                spec = AttackSpec(
                    config.number("attack", "epsilon"),
                    config.number("attack", "step_size"),
                    config.integer("attack", "evaluation_steps"),
                    True,
                    config.integer("attack", "evaluation_restarts"),
                    config.integer("attack", "evaluation_seed"),
                )
                attacked = pgd_attack(
                    model, pixels, torch.tensor([target], device=device), [record.sample_id], spec
                ).adversarial
                shifted = capture_activations(model, attacked, LAYERS)
                with torch.no_grad():
                    attack_logits = model(attacked)[0]
                cam = _resize_map(grad_cam(model, pixels, target), size)
                random_cam = _resize_map(grad_cam(randomized, pixels, target), size)
                occlusion = occlusion_map(
                    model,
                    pixels,
                    target,
                    patch_size=int(values["occlusion_patch_size"]),
                    stride=int(values["occlusion_stride"]),
                    batch_size=batch,
                )
                drops = _np(occlusion["score_drop"])
                controls = localization_comparison(
                    cam,
                    _np(occlusion["positions"]),
                    drops,
                    int(values["top_occlusion_tiles"]),
                    int(values["random_occlusion_repeats"]),
                    seed + index,
                )
                anchor_result: dict[str, Any] = {
                    "sample_id": record.sample_id,
                    "label": target,
                    "prediction": int(logits.argmax()),
                    "clean_margin": float(logits[target] - logits[1 - target]),
                    "pgd_prediction": int(attack_logits.argmax()),
                    "pgd_margin": float(attack_logits[target] - attack_logits[1 - target]),
                    **controls,
                    "randomization_cam_correlation": map_correlation(cam, random_cam),
                    "channel_peaks": {},
                }
                _panel(
                    attr_axes[index, 0],
                    image,
                    f"{record.sample_id}\ntrue={SPECIES[target]} · "
                    f"predicted={SPECIES[int(logits.argmax())]}\n"
                    f"margin={anchor_result['clean_margin']:.3f}",
                )
                _overlay(
                    attr_axes[index, 1],
                    image,
                    cam,
                    f"True-{SPECIES[target]} Grad-CAM\ncoarse 7×7 attribution",
                )
                _panel(
                    attr_axes[index, 2],
                    image,
                    "Occlusion target-margin decrease\nred supports · blue suppresses",
                )
                up = _resize_map(occlusion["score_drop"], size, mode="nearest")
                vmax = max(float(np.abs(up).max()), 1e-6)
                artist = attr_axes[index, 2].imshow(
                    up, cmap="RdBu_r", alpha=0.65, vmin=-vmax, vmax=vmax
                )
                attribution_fig.colorbar(artist, ax=attr_axes[index, 2], shrink=0.7)
                _overlay(
                    attr_axes[index, 3],
                    image,
                    random_cam,
                    "Fully randomized model Grad-CAM\nqualitative sanity control",
                )
                walk_fig, walk_axes = plt.subplots(3, 5, figsize=(17, 10), squeeze=False)
                for level, layer in enumerate(LAYERS):
                    channels = selected_by_layer[layer]
                    channel = max(channels, key=lambda c: float(clean[layer][0, c].max()))
                    raw = clean[layer][0, channel]
                    y, x = divmod(int(raw.argmax()), raw.shape[1])
                    x0, y0, x1, y1 = receptive_field_box(layer, y, x, size)
                    saliency = _resize_map(channel_saliency(model, pixels, layer, channel), size)
                    _panel(
                        walk_axes[level, 0],
                        image,
                        f"{LEVELS[level]} · channel {channel}\npeak unit RF={FIELDS[level]}px",
                    )
                    walk_axes[level, 0].add_patch(
                        Rectangle(
                            (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#00ff88", linewidth=2
                        )
                    )
                    clean_map, pgd_map = _np(raw), _np(shifted[layer][0, channel])
                    limit = max(float(clean_map.max()), float(pgd_map.max()), 1e-6)
                    for col, heat, title in (
                        (1, clean_map, "clean spatial response"),
                        (3, pgd_map, "PGD spatial response"),
                    ):
                        artist = walk_axes[level, col].imshow(
                            heat, cmap="inferno", vmin=0, vmax=limit
                        )
                        walk_axes[level, col].set_title(
                            f"{title}\nshared raw-unit scale={limit:.3f}", fontsize=9
                        )
                        walk_axes[level, col].axis("off")
                        walk_fig.colorbar(artist, ax=walk_axes[level, col], shrink=0.65)
                    _overlay(
                        walk_axes[level, 2],
                        image,
                        saliency,
                        "Input-gradient sensitivity\nof this channel's peak",
                    )
                    _panel(
                        walk_axes[level, 4], np.abs(clean_map - pgd_map), "absolute response change"
                    )
                    anchor_result["channel_peaks"][layer] = {
                        "channel": channel,
                        "clean_peak": float(raw.max()),
                        "pgd_peak": float(shifted[layer][0, channel].max()),
                        "peak_grid_yx": [y, x],
                        "receptive_field_xyxy": [x0, y0, x1, y1],
                    }
                walk_fig.suptitle(
                    f"{arm} · {record.sample_id} · true {SPECIES[target]}\n"
                    "Filter response, theoretical extent, and local sensitivity "
                    "are different quantities",
                    fontsize=13,
                )
                register(
                    walk_fig,
                    arm,
                    f"walkthrough-{record.sample_id}",
                    "A fixed test anchor through three levels. Same channel and scale within "
                    "clean/PGD maps; high-layer RF is broad. Input gradients indicate local "
                    "sensitivity, not training origin.",
                    "activation_walkthrough",
                )
                arm_result["anchors"].append(anchor_result)
                atomic_save_npz(
                    config.project_path("artifacts") / "features" / arm / f"{record.sample_id}.npz",
                    {
                        "clean_pixels": _np(pixels[0]),
                        "pgd_pixels": _np(attacked[0]),
                        "gradcam": cam,
                        "randomized_gradcam": random_cam,
                        "occlusion_drop": drops,
                    },
                )
            attribution_fig.suptitle(
                f"{arm} · which regions influence the binary prediction?\n"
                "Target is always the true species, including errors; "
                "overlays independently normalized",
                fontsize=13,
            )
            register(
                attribution_fig,
                arm,
                "class-attribution",
                "True-species Grad-CAM, signed equal-area occlusion margin drops, and fully "
                "randomized-model sanity control on all four hash-selected anchors.",
                "class_attribution",
            )
            local_fig, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
            rows = arm_result["anchors"]
            xaxis = np.arange(len(rows))
            axis.bar(
                xaxis - 0.18,
                [r["top_gradcam_occlusion_mean_drop"] for r in rows],
                0.36,
                label="top 4 Grad-CAM tiles",
                color="#0f766e",
            )
            axis.bar(
                xaxis + 0.18,
                [r["random_occlusion_mean_drop"] for r in rows],
                0.36,
                label="mean of 100 random 4-tile sets",
                color="#64748b",
            )
            axis.set(
                xticks=xaxis,
                xticklabels=[r["sample_id"] for r in rows],
                ylabel="mean true-class logit-margin decrease",
                title=f"{arm} · attribution-to-occlusion diagnostic",
            )
            axis.tick_params(axis="x", labelsize=8)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.legend()
            register(
                local_fig,
                arm,
                "localization-summary",
                "Average single-tile occlusion effect for Grad-CAM-ranked versus random "
                "equal-area tiles. Not an additive whole-mask effect; four anchors "
                "do not support a population claim.",
                "localization_summary",
                True,
            )
            arm_result["state_unchanged"] = before == state_dict_hash(model.state_dict())
            if not arm_result["state_unchanged"]:
                raise RuntimeError("visualization modified trained model state")
            result["arms"][arm] = arm_result
            atomic_write_json(
                config.project_path("artifacts") / "features" / f"{arm}.json", arm_result
            )
            model.cpu()
            randomized.cpu()
            del model, randomized
            if device.type == "mps":
                torch.mps.empty_cache()
        result["final_memory"] = memory_snapshot(device)
        atomic_write_json(destination, result)
        status(
            "Features complete: patterns, real patches, responses, attribution, "
            "controls and chart evidence saved.",
            enabled=progress,
        )
        return result
    except BaseException as error:
        record_failure(
            config.project_path("artifacts") / "failures" / "features.json", "features", error
        )
        raise
