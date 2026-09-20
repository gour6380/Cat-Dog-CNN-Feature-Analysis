"""Export current photograph-free feature evidence, or an honest fresh-results-pending page.

Presentation only: no model loading, inference, attacks, training or downloads.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig, load_config  # noqa: E402
from src.io_utils import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402
from src.reporting import ReportError, load_current_feature_evidence  # noqa: E402

SAFE_FIGURE_KINDS = {
    "kernels",
    "species_response",
    "initial_response_change",
    "localization_summary",
}
EXCLUDED_PUBLIC_FIGURE_KINDS = {
    # The complete 18-tile atlases are scientifically retained, but failed-probe
    # captions are too dense for a polished public page. The curated comparison
    # below uses the same receipt-verified arrays.
    "activation_maximization",
}
GENERATED_PUBLIC_FIGURE_KINDS = {
    "activation_maximization_comparison",
    "validation_monitoring",
}
ANCHOR_FIELDS = (
    "sample_id",
    "label",
    "prediction",
    "clean_margin",
    "top_gradcam_occlusion_mean_drop",
    "random_occlusion_mean_drop",
    "randomization_cam_correlation",
)
LIMITATIONS = [
    "One ImageNet-initialized ResNet-18 pair and one split/seed family limit scope.",
    "Synthetic stimuli are optimized inputs, not recovered training photographs.",
    "Channel responses do not prove anatomical concepts or causal training-source attribution.",
    "Receptive-field boxes are theoretical support, not exact contributing pixels.",
    "Grad-CAM, occlusion and randomized-weight controls are descriptive diagnostics.",
    "Four fixed anchors are not a population accuracy or robustness evaluation.",
    "PGD is a training objective here; no attack accuracy, calibration or safe-use claim is made.",
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected an object in {path.name}")
    return cast(dict[str, Any], value)


def _protocol(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "dataset": config.value("dataset", "name", str),
        "label_mode": config.label_mode,
        "classes": config.integer("dataset", "classes"),
        "target_names": {"0": "cat", "1": "dog"},
        "architecture": config.value("model", "architecture", str),
        "pretrained_weights": config.value("model", "weights", str),
        "epochs": config.integer("training", "epochs"),
        "dtype": config.value("training", "dtype", str),
        "training_objectives": {
            "standard": "clean cross-entropy",
            "adversarial": f"pure PGD-{config.integer('attack', 'train_steps')} cross-entropy",
        },
        "feature_input_policy": "clean_only",
        "official_test_accuracy_measured": False,
        "post_training_attack_accuracy_measured": False,
    }


def _write_pending(
    config: ExperimentConfig, reason: str = "Owner-run feature evidence not yet available."
) -> tuple[Path, Path]:
    markdown = config.root / "docs/results.md"
    aggregate = config.root / "docs/results.json"
    atomic_write_json(
        aggregate,
        {
            "schema_version": 3,
            "project": config.value("experiment", "title", str),
            "status": "fresh results pending",
            "protocol": _protocol(config),
            "arms": {},
            "figures": [],
            "provenance": {
                "configuration_sha256": config.sha256,
                "public_assets": {},
                "public_reports": {},
            },
            "limitations": LIMITATIONS,
        },
    )
    atomic_write_text(
        markdown,
        f"""# Cat/dog CNN feature walkthrough — fresh results pending

The current configuration uses {config.integer("training", "epochs")} epochs and two models:
standard clean training and pure PGD-5 training. No fresh result has been substituted
from an older run. {reason}

The owner-run notebook will create eight clean-input feature families: full-stage views,
kernels, synthetic preferred stimuli, strongest real patches, activations, Grad-CAM,
occlusion and randomized-weight diagnostics. Both models remain visible.

Photographs and derived photo panels stay ignored locally. Public assets are empty until
complete current evidence exists; only reviewed kernel/synthetic/aggregate figures qualify.
This feature-only run does not measure full-test or attack accuracy, calibration or risk.
Four fixed cat/dog anchors are illustrative examples, not a population evaluation.

[Configuration](../configs/experiment.yaml), [protocol](PROTOCOL.md),
[guided notebook](../notebooks/cat_dog_cnn_features.ipynb),
and [machine-readable status](results.json).
""",
    )
    return markdown, aggregate


def _shareable_figures(config: ExperimentConfig, features: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for figure in features["figures"]:
        if figure.get("shareable") is not True:
            continue
        if figure.get("kind") in EXCLUDED_PUBLIC_FIGURE_KINDS:
            continue
        if figure.get("kind") not in SAFE_FIGURE_KINDS:
            raise RuntimeError("unapproved shareable figure kind")
        relative = Path(figure["path"])
        source = (config.root / relative).resolve()
        if (
            relative.is_absolute()
            or not source.is_relative_to(config.project_path("figures"))
            or source.suffix.lower() != ".png"
            or not source.is_file()
            or sha256_file(source) != figure.get("sha256")
        ):
            raise RuntimeError("shareable figure must be a hash-matched local generated PNG")
        selected.append({**figure, "source": source})
    return selected


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _synthetic_response_summary(layer: dict[str, Any]) -> dict[str, Any]:
    responses = []
    for record in layer.get("synthetic_responses", []):
        initial = record.get("initial_unregularized_mean_response")
        final = record.get("final_unregularized_mean_response")
        if initial is None or final is None:
            continue
        status = (
            "unsuccessful_zero_response_single_start"
            if initial == final == 0
            else "measured_response_increased"
            if final > initial
            else "no_measured_response_increase_single_start"
        )
        response = {
            key: record[key]
            for key in (
                "channel",
                "seed",
                "objective",
                "response_measurement",
                "initial_unregularized_mean_response",
                "final_unregularized_mean_response",
                "unregularized_response_gain",
            )
            if key in record
        }
        response["single_start_status"] = status
        channels = layer.get("selected_channels", [])
        if record.get("channel") in channels and "calibration_mean_by_species" in layer:
            index = channels.index(record["channel"])
            response["calibration_mean_response_cat"] = layer["calibration_mean_by_species"][0][
                index
            ]
            response["calibration_mean_response_dog"] = layer["calibration_mean_by_species"][1][
                index
            ]
        responses.append(response)
    return {
        "available": bool(responses),
        "responses": responses,
        "zero_response_single_start_count": sum(
            response["single_start_status"] == "unsuccessful_zero_response_single_start"
            for response in responses
        ),
        "interpretation": "One recorded optimization start; a failed gray stimulus "
        "is not a dead-channel test.",
    }


def _public_feature_summary(arm: dict[str, Any]) -> dict[str, Any]:
    layers = {}
    for name, layer in arm.get("layers", {}).items():
        responses = layer.get("calibration_mean_by_species", [[], []])
        layers[name] = {
            "selected_channels": layer.get("selected_channels", []),
            "spatial_shape": layer.get("spatial_shape"),
            "receptive_field_pixels": layer.get("receptive_field_pixels"),
            "calibration_mean_response_cat": _mean(responses[0]),
            "calibration_mean_response_dog": _mean(responses[1]),
            "mean_relative_response_change_from_initial": _mean(
                layer.get("relative_change_from_initial", [])
            ),
            "activation_maximization": _synthetic_response_summary(layer),
        }
    anchors = [
        {key: anchor[key] for key in ANCHOR_FIELDS if key in anchor}
        for anchor in arm.get("anchors", [])
    ]
    for anchor in anchors:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", str(anchor.get("sample_id", ""))):
            raise RuntimeError("anchor identifiers must not contain paths")
    return {"layers": layers, "anchors": anchors, "state_unchanged": arm["state_unchanged"]}


def _checkpoint_from_manifest(config: ExperimentConfig, value: object) -> Path:
    if not isinstance(value, str):
        raise RuntimeError("training manifest has no checkpoint path")
    path = Path(value)
    path = path if path.is_absolute() else config.root / path
    path = path.resolve()
    if not path.is_relative_to(config.project_path("checkpoints")) or not path.is_file():
        raise RuntimeError("training checkpoint is outside the registered checkpoint root")
    return path


def _training_monitoring(
    config: ExperimentConfig, features: dict[str, Any]
) -> dict[str, Any] | None:
    paths = {
        arm: config.project_path("artifacts") / "training" / f"{arm}.json"
        for arm in ("standard", "adversarial")
    }
    available = {arm: path.is_file() for arm, path in paths.items()}
    if not any(available.values()):
        return None
    if not all(available.values()):
        raise RuntimeError("both completed training histories are required for comparison")

    histories: dict[str, list[dict[str, Any]]] = {}
    public_arms: dict[str, Any] = {}
    order_hashes: dict[str, list[str]] = {}
    update_counts: dict[str, list[int]] = {}
    for arm, path in paths.items():
        record = _read_json(path)
        history = record.get("history")
        if (
            record.get("status") != "complete"
            or record.get("completed_epochs") != config.integer("training", "epochs")
            or not isinstance(history, list)
            or len(history) != config.integer("training", "epochs")
            or record.get("provenance", {}).get("config_sha256") != config.sha256
        ):
            raise RuntimeError(f"{arm} training history is incomplete or stale")
        checkpoint = _checkpoint_from_manifest(config, record.get("checkpoint"))
        if sha256_file(checkpoint) != features["arms"][arm]["checkpoint_sha256"]:
            raise RuntimeError(f"{arm} training checkpoint differs from feature evidence")
        final = history[-1]
        validation = final.get("monitoring", {}).get("validation_clean")
        if not isinstance(validation, dict) or final.get("epoch") != len(history):
            raise RuntimeError(f"{arm} final validation monitoring is unavailable")
        histories[arm] = cast(list[dict[str, Any]], history)
        order_hashes[arm] = [str(row.get("order_sha256")) for row in history]
        update_counts[arm] = [int(row.get("update_count", -1)) for row in history]
        public_arms[arm] = {
            "final_epoch": final["epoch"],
            "objective": final.get("training_objective", {}).get("objective"),
            "validation": {
                key: validation[key]
                for key in (
                    "accuracy",
                    "macro_accuracy",
                    "per_class_accuracy",
                    "prediction_counts",
                    "confusion_matrix",
                    "correct_count",
                    "count",
                    "mean_loss",
                )
            },
            "training_seconds": sum(float(row.get("seconds", 0.0)) for row in history),
            "updates": record.get("completed_updates"),
        }
    if order_hashes["standard"] != order_hashes["adversarial"]:
        raise RuntimeError("training histories do not use matched sample order")
    if update_counts["standard"] != update_counts["adversarial"]:
        raise RuntimeError("training histories do not use matched update counts")

    count = public_arms["standard"]["validation"]["count"]
    cat_count = sum(public_arms["standard"]["validation"]["confusion_matrix"][0])
    dog_count = sum(public_arms["standard"]["validation"]["confusion_matrix"][1])
    return {
        "histories": histories,
        "public": {
            "purpose": "held-out clean monitoring; no checkpoint selection",
            "official_test_accuracy_measured": False,
            "post_training_attack_accuracy_measured": False,
            "validation_count": count,
            "validation_species_counts": {"cat": cat_count, "dog": dog_count},
            "majority_dog_baseline_accuracy": dog_count / count,
            "arms": public_arms,
        },
    }


def _write_validation_figure(root: Path, monitoring: dict[str, Any]) -> dict[str, Any]:
    histories = monitoring["histories"]
    epochs = [int(row["epoch"]) for row in histories["standard"]]
    colors = {
        "standard_overall": "#2563EB",
        "standard_macro": "#7C3AED",
        "adversarial_overall": "#F97316",
        "adversarial_macro": "#DC2626",
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6))
    for arm, label in (("standard", "Standard"), ("adversarial", "PGD-trained")):
        validations = [row["monitoring"]["validation_clean"] for row in histories[arm]]
        axes[0].plot(
            epochs,
            [100 * float(row["accuracy"]) for row in validations],
            marker="o",
            linewidth=2.5,
            label=f"{label}: overall",
            color=colors[f"{arm}_overall"],
        )
        axes[0].plot(
            epochs,
            [100 * float(row["macro_accuracy"]) for row in validations],
            marker="s",
            linewidth=2.5,
            linestyle="--",
            label=f"{label}: macro",
            color=colors[f"{arm}_macro"],
        )
    majority = 100 * float(monitoring["public"]["majority_dog_baseline_accuracy"])
    axes[0].axhline(majority, color="#64748B", linestyle=":", linewidth=2.2)
    axes[0].text(15.15, majority, f" majority dog baseline {majority:.1f}%", va="center")
    axes[0].set(title="Validation history", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[0].set(xlim=(1, 15.9), ylim=(0, 105), xticks=range(1, 16, 2))
    axes[0].grid(alpha=0.22)
    axes[0].legend(loc="lower left", frameon=False)

    categories = ["Overall", "Macro", "Cat recall", "Dog recall"]
    public_arms = monitoring["public"]["arms"]
    values = {}
    for arm in ("standard", "adversarial"):
        validation = public_arms[arm]["validation"]
        values[arm] = [
            100 * float(validation["accuracy"]),
            100 * float(validation["macro_accuracy"]),
            100 * float(validation["per_class_accuracy"]["0"]),
            100 * float(validation["per_class_accuracy"]["1"]),
        ]
    x = np.arange(len(categories))
    width = 0.36
    bars = [
        axes[1].bar(x - width / 2, values["standard"], width, label="Standard", color="#2563EB"),
        axes[1].bar(
            x + width / 2,
            values["adversarial"],
            width,
            label="PGD-trained",
            color="#F97316",
        ),
    ]
    for group in bars:
        axes[1].bar_label(group, fmt="%.1f%%", padding=3, fontsize=10)
    axes[1].axhline(majority, color="#64748B", linestyle=":", linewidth=2.2)
    axes[1].set(
        title="Fixed epoch-15 checkpoint",
        ylabel="Held-out clean validation (%)",
        xticks=x,
        xticklabels=categories,
        ylim=(0, 112),
    )
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(loc="upper center", frameon=False, ncols=2)
    fig.suptitle(
        "Overall accuracy hid complete majority-class collapse",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "Held-out clean validation: 295 images (95 cats, 200 dogs). "
        "Fixed final epoch; validation did not select checkpoints. "
        "Robust accuracy was not measured.",
        ha="center",
        fontsize=11,
        color="#334155",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    destination = root / "docs/assets/validation-monitoring-comparison.png"
    fig.savefig(destination, dpi=140, facecolor="white")
    plt.close(fig)
    return {
        "arm": "comparison",
        "caption": "Matched epoch-15 training. The pure-PGD arm's 67.8% overall validation "
        "accuracy equals the dog-majority baseline, while macro accuracy is 50% and cat "
        "recall is 0%. This is a failed classifier comparison, not a robustness result.",
        "kind": "validation_monitoring",
        "path": destination.relative_to(root).as_posix(),
        "shareable": True,
    }


def _write_synthetic_comparison(root: Path, features: dict[str, Any]) -> dict[str, Any] | None:
    receipts = []
    for arm in ("standard", "adversarial"):
        receipt = features.get("section_receipts", {}).get(arm, {}).get("synthetic", {})
        if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
            return None
        receipts.append((arm, root / receipt["path"]))

    layer_order = ("network.relu", "network.layer2", "network.layer4")
    layer_titles = ("Early · stem ReLU", "Middle · layer2", "Late · layer4")
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2))
    for row, (arm, receipt_path) in enumerate(receipts):
        receipt = _read_json(receipt_path)
        arrays_path = root / receipt["arrays_path"]
        if sha256_file(arrays_path) != receipt["arrays_sha256"]:
            raise RuntimeError(f"{arm} synthetic arrays differ from their receipt")
        with np.load(arrays_path) as arrays:
            layers = arrays["synthetic_layers"].astype(str)
            channels = arrays["synthetic_channels"]
            pixels = arrays["synthetic_pixels"]
            gains = arrays["unregularized_response_gains"]
            for column, (layer, title) in enumerate(zip(layer_order, layer_titles, strict=True)):
                matches = np.flatnonzero(layers == layer)
                if not len(matches):
                    raise RuntimeError(f"{arm} synthetic arrays omit {layer}")
                index = int(matches[0])
                axes[row, column].imshow(np.moveaxis(pixels[index], 0, -1))
                axes[row, column].set_title(
                    f"{title}\nchannel {int(channels[index])} · gain {float(gains[index]):.2f}",
                    fontsize=12,
                )
                axes[row, column].axis("off")
        axes[row, 0].text(
            -0.16,
            0.5,
            "Standard" if arm == "standard" else "PGD-trained",
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=15,
            fontweight="bold",
        )
    fig.suptitle(
        "Registered channel preferences from early to late layers",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "First calibration-ranked channel at each layer. Optimized stimuli—not pet images, "
        "training memories, or verified semantic detectors. Channels are selected independently.",
        ha="center",
        fontsize=11,
        color="#334155",
    )
    fig.tight_layout(rect=(0.04, 0.055, 1, 0.93))
    destination = root / "docs/assets/feature-preference-comparison.png"
    fig.savefig(destination, dpi=140, facecolor="white")
    plt.close(fig)
    return {
        "arm": "comparison",
        "caption": "Deterministic calibration-ranked early, middle and late channel probes for "
        "both models. These optimized inputs show response preference, not anatomy or robustness.",
        "kind": "activation_maximization_comparison",
        "path": destination.relative_to(root).as_posix(),
        "shareable": True,
    }


def build(root: Path = ROOT) -> tuple[Path, Path]:
    config = load_config(root / "configs/experiment.yaml")
    feature_path = config.project_path("results") / "feature_visualizations.json"
    if not feature_path.is_file():
        return _write_pending(config)
    try:
        features = load_current_feature_evidence(config)
    except ReportError:
        return _write_pending(config, "Current feature evidence is incomplete or stale.")
    figures = _shareable_figures(config, features)
    assets = root / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    public_figures, hashes = [], {}
    for figure in figures:
        stem = re.sub(r"[^A-Za-z0-9_-]", "-", figure["source"].stem)
        destination = assets / f"{figure['arm']}-{figure['kind']}-{stem}.png"
        shutil.copyfile(figure["source"], destination)
        relative = destination.relative_to(root).as_posix()
        hashes[relative] = sha256_file(destination)
        public_figures.append(
            {key: figure[key] for key in ("caption", "kind", "arm")} | {"path": relative}
        )
    monitoring = _training_monitoring(config, features)
    generated_figures: list[dict[str, Any]] = []
    if monitoring is not None:
        generated_figures.append(_write_validation_figure(root, monitoring))
    synthetic_comparison = _write_synthetic_comparison(root, features)
    if synthetic_comparison is not None:
        generated_figures.append(synthetic_comparison)
    for figure in generated_figures:
        if figure["kind"] not in GENERATED_PUBLIC_FIGURE_KINDS:
            raise RuntimeError("unregistered generated public figure kind")
        path = root / figure["path"]
        hashes[figure["path"]] = sha256_file(path)
        public_figures.append(figure)
    public = {
        "schema_version": 3,
        "project": config.value("experiment", "title", str),
        "status": "complete current feature walkthrough",
        "observed_at": features["created_at"],
        "protocol": _protocol(config),
        "arms": {
            arm: {"feature_diagnostics": _public_feature_summary(features["arms"][arm])}
            for arm in ("standard", "adversarial")
        },
        "training_monitoring": monitoring["public"] if monitoring is not None else None,
        "figures": public_figures,
        "provenance": {
            "configuration_sha256": config.sha256,
            "feature_method_sha256": features["method_sha256"],
            "feature_visualizations_sha256": sha256_file(feature_path),
            "checkpoints": {
                arm: {"sha256": features["arms"][arm]["checkpoint_sha256"]}
                for arm in ("standard", "adversarial")
            },
            "public_assets": hashes,
            "public_reports": {},
        },
        "limitations": list(dict.fromkeys([*LIMITATIONS, *features.get("limitations", [])])),
    }
    if any(
        marker in json.dumps(public) for marker in ("/Users/", "/private/", "/tmp/", "image_path")
    ):
        raise RuntimeError("public aggregate contains a private path")
    aggregate, markdown = root / "docs/results.json", root / "docs/results.md"
    atomic_write_json(aggregate, public)
    gallery = "\n\n".join(
        f"![{item['arm']} {item['kind']}]({Path(item['path']).relative_to('docs')})\n\n"
        f"Figure caption: {item['caption']}"
        for item in public_figures
    )
    failed = []
    for arm, result in public["arms"].items():
        for name, layer in result["feature_diagnostics"]["layers"].items():
            for response in layer["activation_maximization"]["responses"]:
                if response["single_start_status"] != "measured_response_increased":
                    failed.append(
                        f"- {arm} · {name} channel {response.get('channel')}: "
                        f"{response['single_start_status']}; not evidence of a dead channel."
                    )
    failed_note = "\n".join(failed) or "No unsuccessful recorded single-start probe."
    limits = "\n".join(f"- {item}" for item in public["limitations"])
    if monitoring is None:
        training_note = "Held-out training-monitoring evidence is unavailable."
    else:
        standard = monitoring["public"]["arms"]["standard"]["validation"]
        adversarial = monitoring["public"]["arms"]["adversarial"]["validation"]
        training_note = f"""The fixed epoch-15 standard checkpoint reached
**{100 * standard["accuracy"]:.2f}%** clean validation accuracy and
**{100 * standard["macro_accuracy"]:.2f}%** macro accuracy. The pure-PGD checkpoint reached
**{100 * adversarial["accuracy"]:.2f}%** overall accuracy, but only
**{100 * adversarial["macro_accuracy"]:.2f}%** macro accuracy: it predicted every validation
image as dog, giving **0% cat recall** and **100% dog recall**. Its overall number equals the
dog-majority baseline. This is a failed classifier comparison, not evidence that PGD training
or adversarial training generally fails. Robust accuracy was not measured in this workflow."""
    atomic_write_text(
        markdown,
        f"""# Cat/dog CNN — current feature walkthrough

Complete current clean-input feature evidence for both standard and PGD-trained models.
Configured training length: {config.integer("training", "epochs")} epochs. No full-test accuracy,
attack accuracy, confidence policy or risk evaluation is claimed. The four fixed anchors
are descriptive examples, not a population experiment.

## Training outcome

{training_note}

{gallery}

## Synthetic optimization outcomes

{failed_note}

## Local photo evidence

The full-stage walkthrough, real patches, activation overlays, Grad-CAM and occlusion
images remain ignored local artifacts, in the owner-generated read-only feature notebook.
Only kernel, synthetic-stimulus and aggregate figures appear above.

## Limitations

{limits}

[Machine-readable feature summary](results.json), [protocol](PROTOCOL.md),
[guided notebook](../notebooks/cat_dog_cnn_features.ipynb).
""",
    )
    return markdown, aggregate


if __name__ == "__main__":
    for output in build():
        print(output.relative_to(ROOT))
