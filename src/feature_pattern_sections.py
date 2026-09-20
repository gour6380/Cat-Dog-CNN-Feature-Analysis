"""Independent filter, synthetic-preference and real-calibration-patch views.

These sections reuse the registered calibration selection and existing methods.
They never choose test examples, attack images or run randomized/initial encoders.
Photographic arrays and figures remain ignored local evidence.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from torch import nn

from src import feature_visualization as fv
from src.feature_methods import receptive_field_box
from src.feature_section_types import FeatureContext, FeatureSection, SectionData
from src.progress import tqdm


def _layer_metadata(ctx: FeatureContext, channels: dict[str, list[int]]) -> dict[str, Any]:
    size = ctx.config.integer("input", "size")
    return {
        layer: {
            "selected_channels": selected,
            "spatial_shape": [(size + jump - 1) // jump] * 2,
            "receptive_field_pixels": field,
        }
        for layer, selected, jump, field in zip(
            fv.LAYERS,
            (channels[layer] for layer in fv.LAYERS),
            (2, 8, 32),
            fv.FIELDS,
            strict=True,
        )
    }


def _kernels(ctx: FeatureContext) -> SectionData:
    selected = ctx.selected_channels()
    channels = selected[fv.LAYERS[0]]
    convolution = ctx.model.get_submodule("network.conv1")
    if not isinstance(convolution, nn.Conv2d):
        raise TypeError("first-layer kernel view requires network.conv1 Conv2d")
    initial = ctx.initial_kernels().detach().cpu()[channels].contiguous()
    current = convolution.weight.detach().cpu()[channels].contiguous()
    difference = current - initial
    figure, axes = plt.subplots(3, len(channels), figsize=(len(channels) * 2.1, 6.8), squeeze=False)
    for column, channel in enumerate(channels):
        for row, values in enumerate((initial[column], current[column], difference[column])):
            fv._panel(
                axes[row, column],
                fv.normalize_map(fv._rgb(values)),
                f"{['ImageNet initial', 'fine-tuned', 'weight difference'][row]}\n"
                f"conv1 channel {channel}",
            )
    figure.suptitle(
        f"{ctx.arm} · actual 7×7 RGB convolution kernels\n"
        "Each tile independently scaled; difference colors are not RGB semantics"
    )
    ctx.register(
        figure,
        "kernels",
        "Actual first-layer weights, initialized weights, and their differences. "
        "Deeper channel filters have many input channels and are instead probed "
        "with optimization and real examples.",
        "kernels",
        shareable=True,
    )
    return SectionData(
        payload={"layers": _layer_metadata(ctx, selected)},
        arrays={
            "kernel_channels": np.asarray(channels, dtype=np.int64),
            "initial_kernels": fv._np(initial),
            "current_kernels": fv._np(current),
            "kernel_differences": fv._np(difference),
        },
    )


def _synthetic(ctx: FeatureContext) -> SectionData:
    selected = ctx.selected_channels()
    count = max(len(channels) for channels in selected.values())
    seed = ctx.config.integer("feature_visualization", "seed")
    figure, axes = plt.subplots(3, count, figsize=(count * 2.4, 8.3), squeeze=False)
    layers = _layer_metadata(ctx, selected)
    layer_ids: list[str] = []
    channel_ids: list[int] = []
    seeds: list[int] = []
    pixels: list[np.ndarray[Any, Any]] = []
    initial_responses: list[float] = []
    final_responses: list[float] = []
    gains: list[float] = []
    for level, layer in enumerate(fv.LAYERS):
        layers[layer]["synthetic_responses"] = []
        for column, channel in enumerate(
            tqdm(selected[layer], desc=f"{ctx.arm} {layer} stimuli", disable=not ctx.progress)
        ):
            probe_seed = seed + level * 10000 + channel
            pattern = fv._pattern(
                ctx.config, ctx.model, ctx.arm, layer, channel, ctx.device, ctx.identity, probe_seed
            )
            pattern_array = fv._np(pattern.contiguous())
            size = ctx.config.integer("input", "size")
            if (
                pattern_array.shape != (3, size, size)
                or not np.isfinite(pattern_array).all()
                or pattern_array.min() < 0.0
                or pattern_array.max() > 1.0
            ):
                raise ValueError("synthetic probe pixels must be finite bounded RGB inputs")
            path = (
                ctx.config.project_path("artifacts")
                / "features"
                / "synthetic"
                / f"{ctx.arm}-{layer.split('.')[-1]}-{channel}.json"
            )
            response = json.loads(path.read_text(encoding="utf-8"))
            if (
                response.get("identity") != ctx.identity
                or response.get("layer") != layer
                or response.get("channel") != channel
                or response.get("seed") != probe_seed
            ):
                raise ValueError("synthetic response record does not match the registered probe")
            initial = float(response["initial_unregularized_mean_response"])
            final = float(response["final_unregularized_mean_response"])
            gain = float(response["unregularized_response_gain"])
            if not np.isfinite([initial, final, gain]).all() or not np.isclose(
                gain, final - initial, atol=1e-12, rtol=1e-12
            ):
                raise ValueError("synthetic response gain is non-finite or inconsistent")
            layers[layer]["synthetic_responses"].append(response)
            failure_note = (
                "\nzero-response trial · not a dead channel" if initial == final == 0 else ""
            )
            fv._panel(
                axes[level, column],
                fv._rgb(pattern),
                f"{fv.LEVELS[level]}\nchannel {channel} · synthetic\n"
                f"mean response {initial:.2f} → {final:.2f}{failure_note}",
            )
            layer_ids.append(layer)
            channel_ids.append(channel)
            seeds.append(probe_seed)
            pixels.append(pattern_array)
            initial_responses.append(initial)
            final_responses.append(final)
            gains.append(gain)
        for axis in axes[level, len(selected[layer]) :]:
            axis.axis("off")
    figure.suptitle(
        f"{ctx.arm} · regularized activation-maximization patterns\n"
        "Optimized stimuli, not training photographs or generative reconstructions",
        fontsize=13,
    )
    ctx.register(
        figure,
        "synthetic-atlas",
        "Layer-wise optimized channel preferences. Every tile is independently "
        "optimized from private seeded noise; visual appearance is not evidence "
        "of a named detector. Fixed-start zero-response trials are retained, not replaced.",
        "activation_maximization",
        shareable=True,
    )
    return SectionData(
        payload={"layers": layers},
        arrays={
            "synthetic_layers": np.asarray(layer_ids, dtype=np.str_),
            "synthetic_channels": np.asarray(channel_ids, dtype=np.int64),
            "synthetic_seeds": np.asarray(seeds, dtype=np.int64),
            "synthetic_pixels": np.stack(pixels),
            "initial_unregularized_mean_responses": np.asarray(initial_responses, dtype=np.float64),
            "final_unregularized_mean_responses": np.asarray(final_responses, dtype=np.float64),
            "unregularized_response_gains": np.asarray(gains, dtype=np.float64),
        },
    )


def _real_patches(ctx: FeatureContext) -> SectionData:
    selected = ctx.selected_channels()
    probe = ctx.reference_probe()
    pixels = ctx.reference_pixels()
    size = ctx.config.integer("input", "size")
    layers = _layer_metadata(ctx, selected)
    patches: list[dict[str, Any]] = []
    image_indices: list[int] = []
    for level, layer in enumerate(fv.LAYERS):
        channels = selected[layer]
        figure, axes = plt.subplots(
            len(channels), 4, figsize=(12, len(channels) * 2.5), squeeze=False
        )
        layers[layer]["real_patches"] = []
        for row, channel in enumerate(channels):
            top = np.argsort(-fv._np(probe[layer]["peaks"][:, channel]), kind="stable")[:2]
            for rank, index_raw in enumerate(top):
                index = int(index_raw)
                record = ctx.references[index]
                y, x = probe[layer]["positions"][index, channel].tolist()
                box = receptive_field_box(layer, int(y), int(x), size)
                peak = float(probe[layer]["peaks"][index, channel])
                image = fv._rgb(pixels[index])
                x0, y0, x1, y1 = box
                fv._panel(
                    axes[row, rank * 2],
                    image,
                    f"ch {channel} · rank {rank + 1}\n{record.sample_id}\npeak={peak:.3f}",
                )
                axes[row, rank * 2].add_patch(
                    Rectangle(
                        (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#00ff88", linewidth=2
                    )
                )
                fv._panel(
                    axes[row, rank * 2 + 1],
                    image[y0:y1, x0:x1],
                    f"theoretical RF crop\n{fv.FIELDS[level]}px before clipping",
                )
                patch = {
                    "layer": layer,
                    "channel": channel,
                    "rank": rank + 1,
                    "sample_id": record.sample_id,
                    "label": record.label,
                    "species": record.species,
                    "reference_index": index,
                    "peak_response": peak,
                    "activation_position": [int(y), int(x)],
                    "receptive_field_box": list(box),
                    "receptive_field_pixels": fv.FIELDS[level],
                }
                patches.append(patch)
                layers[layer]["real_patches"].append(patch)
                image_indices.append(index)
        figure.suptitle(
            f"{ctx.arm} · {fv.LEVELS[level]} · strongest calibration responses\n"
            "Green boxes are theoretical receptive fields, not proven anatomical concepts",
            fontsize=12,
        )
        ctx.register(
            figure,
            f"top-patches-{level}",
            f"Real calibration examples maximally activating the selected {fv.LEVELS[level]} "
            "channels; RF extents clipped to the evaluation crop.",
            "real_patches",
            shareable=False,
        )
    unique_images = sorted(set(image_indices))
    image_lookup = {index: unique_index for unique_index, index in enumerate(unique_images)}
    return SectionData(
        payload={"layers": layers},
        arrays={
            "patch_layers": np.asarray([patch["layer"] for patch in patches], dtype=np.str_),
            "patch_channels": np.asarray([patch["channel"] for patch in patches], dtype=np.int64),
            "patch_ranks": np.asarray([patch["rank"] for patch in patches], dtype=np.int64),
            "patch_sample_ids": np.asarray(
                [patch["sample_id"] for patch in patches], dtype=np.str_
            ),
            "patch_labels": np.asarray([patch["label"] for patch in patches], dtype=np.int64),
            "patch_species": np.asarray([patch["species"] for patch in patches], dtype=np.int64),
            "patch_peak_responses": np.asarray(
                [patch["peak_response"] for patch in patches], dtype=np.float64
            ),
            "patch_positions": np.asarray(
                [patch["activation_position"] for patch in patches], dtype=np.int64
            ),
            "patch_rf_boxes": np.asarray(
                [patch["receptive_field_box"] for patch in patches], dtype=np.int64
            ),
            "patch_image_indices": np.asarray(
                [image_lookup[index] for index in image_indices], dtype=np.int64
            ),
            "reference_image_ids": np.asarray(
                [ctx.references[index].sample_id for index in unique_images], dtype=np.str_
            ),
            "reference_pixels": fv._np(pixels[unique_images].contiguous()),
        },
    )


def compute_patterns(section: FeatureSection, ctx: FeatureContext) -> SectionData:
    """Compute only the requested registered pattern family."""

    before = set(plt.get_fignums())
    try:
        if section == "kernels":
            return _kernels(ctx)
        if section == "synthetic":
            return _synthetic(ctx)
        if section == "real_patches":
            return _real_patches(ctx)
        raise ValueError(f"not a pattern feature section: {section}")
    except BaseException:
        # Do not leak incomplete panels or close another notebook's live chart.
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
        raise
