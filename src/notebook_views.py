"""Small, read-only presentation helpers for recorded notebook measurements.

These helpers never load a model or change the scientific evidence. In particular,
failed probes and undefined controls remain visible rather than being replaced.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.config import ExperimentConfig
from src.io_utils import sha256_file

MODEL_NAMES = {"standard": "Standard", "adversarial": "PGD-trained"}


def validation_note(history: list[dict[str, Any]], arm: str) -> str:
    """Explain the final clean validation check, not test or attack accuracy."""
    if not history:
        return "Clean validation details unavailable; no measurements are inferred."
    metrics = history[-1].get("monitoring", {}).get("validation_clean", {})
    matrix = np.asarray(metrics.get("confusion_matrix", []))
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all() or matrix.sum() <= 0:
        return "Clean validation details unavailable in this saved history; none are inferred."
    counts = matrix.sum(axis=1)
    predictions = matrix.sum(axis=0)
    total = int(matrix.sum())
    correct = int(matrix.trace())
    lines = [
        f"**{MODEL_NAMES[arm]} · epoch {history[-1]['epoch']} · clean validation**",
        "",
        "| Check | Recorded result |",
        "|---|---:|",
        f"| Correct overall | {correct}/{total} ({100 * correct / total:.2f}%) |",
    ]
    for label, species in enumerate(("Cats", "Dogs")):
        count = int(counts[label])
        found = int(matrix[label, label])
        value = f"{found}/{count} ({100 * found / count:.2f}%)" if count else "Unavailable"
        lines.append(f"| {species} found correctly | {value} |")
    lines += [
        f"| Labels predicted | {int(predictions[0])} cat / {int(predictions[1])} dog |",
        "",
        "Validation means pictures held aside from fitting. It is not an attack test.",
    ]
    if np.count_nonzero(predictions) == 1:
        name = ("cat", "dog")[int(predictions.argmax())]
        lines += [
            "",
            f"**Watch out: this model calls all {total} validation pictures '{name}'.** "
            "A majority-class guess can look moderately accurate while completely missing "
            "the other animal. Falling loss does not fix that failure.",
        ]
    return "\n".join(lines)


def compact_records(evidence: dict[str, Any], section: str) -> list[dict[str, Any]]:
    """Select by registered IDs/species only, never by prediction or visual appeal."""
    records = [item for item in evidence.get("figures", []) if item.get("section") == section]
    ids = evidence.get("selection", {}).get("test_anchor_ids", [])
    labels = {
        row["sample_id"]: row["label"]
        for model in evidence.get("arms", {}).values()
        for row in model.get("anchors", [])
        if "sample_id" in row and "label" in row
    }
    chosen = [next((name for name in ids if labels.get(name) == label), None) for label in (0, 1)]
    if section in {"stages", "activations"} and all(name is not None for name in chosen):
        records = [item for item in records if item.get("sample_id") in chosen]
        order = {name: index for index, name in enumerate(chosen)}
        records.sort(key=lambda item: (order[item["sample_id"]], item.get("arm") != "standard"))
    elif section == "diagnostics":
        kinds = {"randomization_control": 0, "localization_summary": 1}
        records = [item for item in records if item.get("kind") in kinds]
        records.sort(key=lambda item: (kinds[item["kind"]], item.get("arm") != "standard"))
    return records


def _registered_path(config: ExperimentConfig, relative: str, expected_hash: str) -> Path:
    path = (config.root / relative).resolve()
    if not path.is_relative_to(config.root) or not path.is_file():
        raise ValueError("saved view measurement is missing or outside this project")
    if sha256_file(path) != expected_hash:
        raise ValueError("saved view measurement hash mismatch")
    return path


def section_measurements(
    config: ExperimentConfig, evidence: dict[str, Any], arm: str, section: str
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    """Read verified arrays after the caller validates the complete/partial registry."""
    record = evidence["section_receipts"][arm][section]
    receipt = json.loads(_registered_path(config, record["path"], record["sha256"]).read_text())
    if any(
        receipt.get(key) != value
        for key, value in {
            "arm": arm,
            "section": section,
            "identity": evidence["identity"],
            "config_sha256": config.sha256,
            "method_sha256": evidence["method_sha256"],
        }.items()
    ):
        raise ValueError("saved view measurement identity mismatch")
    path = _registered_path(config, receipt["arrays_path"], receipt["arrays_sha256"])
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    return receipt, arrays


def synthetic_preview(arrays: dict[str, np.ndarray[Any, Any]], arm: str) -> bytes:
    """Re-layout recorded stimuli only; do not optimize or substitute any tile."""
    layers = arrays["synthetic_layers"]
    channels = arrays["synthetic_channels"]
    pixels = arrays["synthetic_pixels"]
    initial = arrays["initial_unregularized_mean_responses"]
    final = arrays["final_unregularized_mean_responses"]
    gains = arrays["unregularized_response_gains"]
    count = len(channels)
    if (
        pixels.ndim != 4
        or pixels.shape[0] != count
        or pixels.shape[1] != 3
        or any(len(values) != count for values in (layers, initial, final, gains))
        or not np.isfinite(pixels).all()
        or pixels.min() < 0
        or pixels.max() > 1
        or not np.isfinite(np.concatenate((initial, final, gains))).all()
        or not np.allclose(gains, final - initial, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("saved synthetic tiles or responses are invalid")
    level_names = ("Early", "Middle", "Late")
    layer_names = list(dict.fromkeys(layers.tolist()))
    if len(layer_names) != 3:
        raise ValueError("synthetic walkthrough needs three registered layer groups")
    columns = max(int((layers == layer).sum()) for layer in layer_names)
    figure, axes = plt.subplots(
        3, columns, figsize=(columns * 2.15, 8), squeeze=False, constrained_layout=True
    )
    try:
        for row, layer in enumerate(layer_names):
            for axis in axes[row]:
                axis.axis("off")
            for column, index in enumerate(np.flatnonzero(layers == layer)):
                axis = axes[row, column]
                axis.imshow(pixels[index].transpose(1, 2, 0))
                note = "\nNo response increase" if gains[index] <= 0 else ""
                axis.set_title(
                    f"{level_names[row]} · channel {int(channels[index])}\n"
                    f"Response {initial[index]:.2f} → {final[index]:.2f}{note}",
                    fontsize=8,
                )
        figure.suptitle(f"{MODEL_NAMES[arm]} · synthetic preferred patterns", fontsize=13)
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=105, bbox_inches="tight")
        return output.getvalue()
    finally:
        plt.close(figure)


def real_patch_preview(
    receipt: dict[str, Any], arrays: dict[str, np.ndarray[Any, Any]], arm: str
) -> bytes:
    """Show two calibration-ranked channels per level, without picking new examples."""
    figure, axes = plt.subplots(3, 4, figsize=(11, 8), constrained_layout=True)
    try:
        for row, layer in enumerate(("network.relu", "network.layer2", "network.layer4")):
            channels = receipt["payload"]["layers"][layer]["selected_channels"][:2]
            for axis in axes[row]:
                axis.axis("off")
            for column, channel in enumerate(channels):
                matches = np.flatnonzero(
                    (arrays["patch_layers"] == layer)
                    & (arrays["patch_channels"] == channel)
                    & (arrays["patch_ranks"] == 1)
                )
                if len(matches) != 1:
                    raise ValueError(
                        "real-patch preview needs one registered top image per channel"
                    )
                index = int(matches[0])
                image_index = int(arrays["patch_image_indices"][index])
                name = str(arrays["patch_sample_ids"][index])
                if name != arrays["reference_image_ids"][image_index]:
                    raise ValueError("real-patch sample and image are not aligned")
                image = arrays["reference_pixels"][image_index].transpose(1, 2, 0)
                x0, y0, x1, y1 = arrays["patch_rf_boxes"][index].astype(int).tolist()
                if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
                    raise ValueError("real-patch receptive-field box is outside its input")
                axis, crop_axis = axes[row, column * 2 : column * 2 + 2]
                axis.imshow(image)
                axis.add_patch(
                    Rectangle(
                        (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#00ff88", linewidth=2
                    )
                )
                axis.set_title(
                    f"{('Early', 'Middle', 'Late')[row]} · channel {channel}\n{name}", fontsize=8
                )
                crop_axis.imshow(image[y0:y1, x0:x1])
                crop_axis.set_title(
                    "Possible receptive field\nnot a proven body-part detector", fontsize=8
                )
        figure.suptitle(f"{MODEL_NAMES[arm]} · strongest recorded real-image patches", fontsize=12)
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=105, bbox_inches="tight")
        return output.getvalue()
    finally:
        plt.close(figure)


def feature_note(
    section: str, receipt: dict[str, Any], arrays: dict[str, np.ndarray[Any, Any]]
) -> str:
    """Highlight failures in measured probes without diagnosing their cause."""
    if section == "synthetic":
        gains = arrays["unregularized_response_gains"]
        failed = int((gains <= 0).sum())
        return (
            f"{failed}/{len(gains)} trials showed no response increase. Gray tiles are kept: "
            "this optimizer trial did not find a preferred pattern from its fixed starting "
            "noise. That does not mean the channel is dead. These are invented probes, "
            "not photographs the model remembers."
        )
    if section == "gradcam":
        empty = [
            row["sample_id"]
            for index, row in enumerate(receipt["payload"].get("anchors", []))
            if not np.any(arrays[f"anchor_{index}_gradcam"] > 0)
        ]
        return (
            "No positive Grad-CAM map for the true-species score: "
            + ", ".join(empty)
            + ". This method found no positive highlight here; it does not mean there "
            "are no learned features."
            if empty
            else "Grad-CAM highlights are for the true animal's score, "
            "even if the model guessed wrong."
        )
    if section == "diagnostics":
        rows = receipt["payload"].get("anchors", [])
        undefined = sum(row.get("randomization_cam_correlation") is None for row in rows)
        return (
            f"Map similarity is undefined for {undefined}/{len(rows)} anchors "
            "(a flat map cannot support a correlation); undefined is not zero or a passed check. "
            "Changing the weights should be treated as a sanity check, not proof that "
            "a highlight is the correct explanation."
        )
    return ""
