"""Current-run charts and verified galleries; never loads historical public exports."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

from src.calibration import softmax, tie_aware_risk_coverage
from src.config import ExperimentConfig
from src.corruptions import registered_corruptions
from src.evaluation import EvaluationSection, _artifact_provenance, load_arrays
from src.feature_section_types import FEATURE_SECTIONS, FeatureSection
from src.io_utils import atomic_write_bytes, atomic_write_json, sha256_file, utc_now
from src.training import Arm, checkpoint_path, experiment_provenance

ARMS: tuple[Arm, ...] = ("standard", "adversarial")
COLORS = {"standard": "#2563eb", "adversarial": "#ea580c"}
LABELS = {"standard": "Standard", "adversarial": "PGD-trained"}
SPECIES = ("cat", "dog")


class VisualEvidenceError(RuntimeError):
    """A picture cannot be tied to the current scientific evidence."""


def _ipython_display() -> tuple[Any, Any, Any]:
    from IPython.display import Image, Markdown, display

    return Image, Markdown, display


def _number(value: Any) -> float:
    return float(value) if isinstance(value, int | float) else float("nan")


def _finish(axis: Any, title: str, ylabel: str, *, percent: bool = False) -> None:
    axis.set(title=title, ylabel=ylabel)
    axis.grid(axis="y", alpha=0.18)
    if percent:
        axis.set_ylim(0, 105)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize=8, loc="best")


def _curve(axis: Any, x: Sequence[Any], values: Sequence[float], label: str, **kw: Any) -> None:
    axis.plot(x, values, marker="o", markersize=3, linewidth=1.5, label=label, **kw)


def _save(
    config: ExperimentConfig,
    group: str,
    name: str,
    figure: Any,
    caption: str,
    *,
    shareable: bool = True,
) -> dict[str, Any]:
    destination = config.project_path("figures") / group / f"{name}.png"
    payload = io.BytesIO()
    try:
        figure.savefig(payload, format="png", dpi=130, bbox_inches="tight")
        atomic_write_bytes(destination, payload.getvalue())
    finally:
        plt.close(figure)
    return {
        "path": str(destination.relative_to(config.root)),
        "sha256": sha256_file(destination),
        "caption": caption,
        "shareable": shareable,
        "kind": name,
    }


def _manifest(
    config: ExperimentConfig,
    group: str,
    figures: list[dict[str, Any]],
    checkpoints: Mapping[str, str],
    **extra: Any,
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "group": group,
        "created_at": utc_now(),
        "config_sha256": config.sha256,
        "plot_source_sha256": sha256_file(Path(__file__)),
        "checkpoint_sha256": checkpoints,
        "figures": figures,
        **extra,
    }
    if group.startswith("evaluation-"):
        result["evaluation_provenance"] = {
            arm: _artifact_provenance(config, arm, sha256_file(checkpoint_path(config, arm)))
            for arm in ARMS
        }
    atomic_write_json(config.project_path("results") / "visualizations" / f"{group}.json", result)
    return result


def _verified_path(config: ExperimentConfig, record: dict[str, Any]) -> Path:
    relative = record.get("path")
    if not isinstance(relative, str):
        raise VisualEvidenceError("figure path is missing")
    path = (config.root / relative).resolve()
    if not path.is_relative_to(config.root) or not path.is_file():
        raise VisualEvidenceError(f"missing current-run image: {relative}")
    if sha256_file(path) != record.get("sha256"):
        raise VisualEvidenceError(f"image hash mismatch: {relative}")
    return path


def display_figures(config: ExperimentConfig, manifest: dict[str, Any]) -> None:
    Image, Markdown, display = _ipython_display()

    if manifest.get("config_sha256") != config.sha256:
        raise VisualEvidenceError("figures belong to another configuration")
    for record in manifest.get("figures", []):
        path = _verified_path(config, record)
        display(
            Markdown(
                f"### {record['kind'].replace('_', ' ').replace('-', ' ')}\n\n{record['caption']}"
            )
        )
        display(Image(filename=str(path), width=1200))


def plot_learning_curves(history: list[dict[str, Any]], arm: Arm, *, compact: bool = False) -> Any:
    """Pure plotting: absent historical measurements remain gaps, never invented zeros."""
    if compact:
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
        epochs = [row["epoch"] for row in history]
        _curve(
            axes[0],
            epochs,
            [_number(row.get("mean_training_loss")) for row in history],
            "PGD augmented CE" if arm == "adversarial" else "Clean augmented CE",
            color=COLORS[arm],
        )
        _finish(axes[0], "Training objective", "Cross-entropy")
        _curve(
            axes[1],
            epochs,
            [
                100 * _number(row.get("monitoring", {}).get("validation_clean", {}).get("accuracy"))
                for row in history
            ],
            "Overall correct",
            color=COLORS[arm],
        )
        _curve(
            axes[1],
            epochs,
            [
                100
                * _number(
                    row.get("monitoring", {}).get("validation_clean", {}).get("macro_accuracy")
                )
                for row in history
            ],
            "Cat/dog equally weighted",
            color="#7c3aed",
            linestyle="--",
        )
        majority = []
        for row in history:
            matrix = np.asarray(
                row.get("monitoring", {}).get("validation_clean", {}).get("confusion_matrix", [])
            )
            majority.append(
                100 * float(matrix.sum(axis=1).max() / matrix.sum())
                if matrix.shape == (2, 2) and matrix.sum() > 0
                else float("nan")
            )
        _curve(
            axes[1],
            epochs,
            majority,
            "Always guess the majority animal",
            color="#64748b",
            linestyle=":",
        )
        _finish(axes[1], "Validation accuracy", "Accuracy (%)", percent=True)
        for axis in axes:
            axis.set_xlabel("Epoch")
            if len(epochs) <= 10:
                axis.set_xticks(epochs)
            if len(epochs) == 1:
                axis.set_xticks(epochs)
                axis.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
        figure.suptitle(f"{LABELS[arm]} · saved learning history")
        return figure
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    objective = "PGD augmented CE" if arm == "adversarial" else "Clean augmented CE"
    _curve(
        axes[0, 0],
        epochs,
        [_number(row.get("mean_training_loss")) for row in history],
        objective,
        color=COLORS[arm],
    )
    _finish(axes[0, 0], "Training objective (train mode)", "Cross-entropy")
    for split, name, color in (
        ("train_clean", "Fitting · clean eval", "#2563eb"),
        ("validation_clean", "Validation · clean eval", "#ea580c"),
    ):
        metrics = [row.get("monitoring", {}).get(split, {}) for row in history]
        _curve(
            axes[0, 1],
            epochs,
            [_number(m.get("loss", m.get("cross_entropy"))) for m in metrics],
            name,
            color=color,
        )
        for axis, metric in ((axes[0, 2], "accuracy"), (axes[1, 0], "macro_accuracy")):
            _curve(axis, epochs, [100 * _number(m.get(metric)) for m in metrics], name, color=color)
    _finish(axes[0, 1], "Clean fitting vs held-out validation", "Cross-entropy")
    _finish(axes[0, 2], "Clean overall accuracy", "Accuracy (%)", percent=True)
    _finish(axes[1, 0], "Clean macro accuracy (equal species weight)", "Accuracy (%)", percent=True)
    for label, color in ((0, "#7c3aed"), (1, "#0f766e")):
        values = [
            100
            * _number(
                row.get("monitoring", {})
                .get("validation_clean", {})
                .get("per_class_accuracy", {})
                .get(str(label))
            )
            for row in history
        ]
        _curve(axes[1, 1], epochs, values, SPECIES[label], color=color)
    _finish(axes[1, 1], "Held-out validation species recall", "Recall (%)", percent=True)
    for index, name, style in ((0, "Backbone", "-"), (1, "Classifier head", "--")):
        rates = [row.get("last_learning_rates", [None, None]) for row in history]
        _curve(axes[1, 2], epochs, [_number(r[index]) for r in rates], name, linestyle=style)
    _finish(axes[1, 2], "Learning rates at epoch end", "Learning rate")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
    figure.suptitle(f"{LABELS[arm]} · measured learning history", fontsize=15)
    if len(epochs) == 1:
        for axis in axes.flat:
            axis.set_xticks(epochs)
            axis.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
    if not any(row.get("monitoring") for row in history):
        figure.supxlabel(
            "Clean monitoring unavailable in this historical history; no curves inferred."
        )
    return figure


def _confusion(axis: Any, metrics: dict[str, Any], title: str) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    axis.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(len(matrix)):
        for column in range(len(matrix)):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=SPECIES,
        yticklabels=SPECIES,
        xlabel="Predicted species",
        ylabel="True species",
        title=title,
    )


def _training_examples(
    config: ExperimentConfig, arm: Arm, *, completed_epoch: int | None = None
) -> Any | None:
    path = config.project_path("artifacts") / "training" / f"{arm}-examples.npz"
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        return None
    receipt = json.loads(metadata.read_text())
    if receipt.get("npz_sha256") != sha256_file(path):
        raise VisualEvidenceError("training example hash mismatch")
    if receipt.get("provenance") != experiment_provenance(config):
        raise VisualEvidenceError("training examples belong to another run")
    arrays = load_arrays(path)
    from src.training_monitoring import prepare_training_monitoring

    protocol = prepare_training_monitoring(config)
    if receipt.get("protocol_sha256") != protocol.sha256:
        raise VisualEvidenceError("training example protocol identity mismatch")
    size = config.integer("input", "size")
    count = len(arrays.get("sample_ids", []))
    if not count:
        raise VisualEvidenceError("training example gallery is empty")
    for key in ("clean_pixels", "objective_pixels"):
        pixels = arrays.get(key)
        if (
            pixels is None
            or pixels.shape != (count, 3, size, size)
            or not np.isfinite(pixels).all()
            or pixels.min() < 0
            or pixels.max() > 1
        ):
            raise VisualEvidenceError("invalid captured training pixels")
    for key in ("labels", "epochs", "original_image_paths"):
        if key not in arrays or len(arrays[key]) != count:
            raise VisualEvidenceError("unaligned training example metadata")
    if np.any(arrays["epochs"] < 1) or np.any(
        arrays["epochs"] > config.integer("training", "epochs")
    ):
        raise VisualEvidenceError("training example epochs are invalid")
    logits = arrays.get("logits")
    if (
        logits is None
        or logits.shape != (count, config.integer("model", "classes"))
        or not np.isfinite(logits).all()
    ):
        raise VisualEvidenceError("invalid captured training logits")
    completed = completed_epoch or config.integer("training", "epochs")
    eligible = arrays["epochs"][arrays["epochs"] <= completed]
    if not len(eligible):
        return None
    last_epoch = eligible.max()
    indices = np.flatnonzero(arrays["epochs"] == last_epoch)
    expected = {record.sample_id: record for record in protocol.gallery_records}
    if len(indices) != len(expected) or set(arrays["sample_ids"][indices]) != set(expected):
        raise VisualEvidenceError(
            "captured training gallery IDs differ from fixed fitting selection"
        )
    for index in indices:
        record = expected[str(arrays["sample_ids"][index])]
        if (
            int(arrays["labels"][index]) != record.label
            or str(arrays["original_image_paths"][index]) != record.image_path
        ):
            raise VisualEvidenceError("training gallery labels/photographs do not align")
    difference = np.abs(arrays["objective_pixels"] - arrays["clean_pixels"])
    if (arm == "standard" and np.any(difference != 0)) or (
        arm == "adversarial" and difference.max() > config.number("attack", "epsilon") + 1e-6
    ):
        raise VisualEvidenceError(
            "captured training inputs violate the configured objective/attack bounds"
        )
    figure, axes = plt.subplots(
        len(indices), 3, figsize=(11, 3 * len(indices)), squeeze=False, constrained_layout=True
    )
    for row, index in enumerate(indices):
        clean = arrays["clean_pixels"][index].transpose(1, 2, 0)
        objective = arrays["objective_pixels"][index].transpose(1, 2, 0)
        if arm == "standard":
            with PILImage.open(str(arrays["original_image_paths"][index])) as original:
                axes[row, 0].imshow(original.convert("RGB"))
            axes[row, 0].set_title("Original fitting photograph")
            axes[row, 1].imshow(clean)
            axes[row, 1].set_title("Actual augmented training input")
            axes[row, 2].imshow(objective)
            axes[row, 2].set_title("Input used for clean CE")
        else:
            for axis, pixels, title in zip(
                axes[row],
                (clean, objective, np.clip(np.abs(objective - clean) * 32, 0, 1)),
                (
                    "Actual clean augmented input",
                    f"Actual PGD-{config.integer('attack', 'train_steps')} training input",
                    "Absolute perturbation ×32 · display only",
                ),
                strict=True,
            ):
                axis.imshow(pixels)
                axis.set_title(title, fontsize=9)
        axes[row, 0].set_ylabel(
            f"{arrays['sample_ids'][index]}\n"
            f"true: {SPECIES[int(arrays['labels'][index])]}\n"
            f"epoch {int(last_epoch)}",
            fontsize=9,
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(f"{LABELS[arm]} · images captured from the actual training passes")
    return figure


def training_visuals(
    config: ExperimentConfig,
    history: list[dict[str, Any]],
    arm: Arm,
    *,
    final: bool = False,
    compact: bool = False,
) -> dict[str, Any]:
    if not history:
        raise VisualEvidenceError("training history is unavailable; run this arm first")
    checkpoint = checkpoint_path(config, arm, int(history[-1]["epoch"]))
    figures = [
        _save(
            config,
            "training",
            f"{arm}-learning-curves{'-compact' if compact else ''}",
            plot_learning_curves(history, arm, compact=compact),
            "Actual objective loss is separate from clean eval-mode fitting/validation "
            "metrics. Validation is held out from fitting; no checkpoint selection.",
        )
    ]
    last = history[-1].get("monitoring", {}).get("validation_clean", {})
    if final and not compact and "confusion_matrix" in last:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        _confusion(axes[0], last, "Final held-out validation confusion")
        predictions = np.asarray(last["confusion_matrix"]).sum(axis=0)
        axes[1].bar(SPECIES, predictions, color=COLORS[arm])
        _finish(axes[1], "Validation predicted-class counts", "Images")
        figures.append(
            _save(
                config,
                "training",
                f"{arm}-validation",
                figure,
                "Inspect species recall and predicted counts, not only overall accuracy.",
            )
        )
    if final and not compact:
        examples = _training_examples(config, arm, completed_epoch=int(history[-1]["epoch"]))
        if examples is not None:
            figures.append(
                _save(
                    config,
                    "training",
                    f"{arm}-examples",
                    examples,
                    "Fixed fitting IDs, actual final-epoch augmented/attacked inputs; "
                    "amplified perturbations are not the model inputs.",
                    shareable=False,
                )
            )
    return _manifest(
        config,
        f"training-{arm}",
        figures,
        {arm: sha256_file(checkpoint)},
        completed_epoch=history[-1]["epoch"],
        final=final,
        compact=compact,
    )


def training_observer(
    config: ExperimentConfig, *, compact: bool = False
) -> Callable[[list[dict[str, Any]], Arm], None]:
    """Refresh one display handle after each durable epoch; chart errors never lose checkpoints."""
    Image, _, display = _ipython_display()

    handles: dict[str, Any] = {}

    def update(history: list[dict[str, Any]], arm: Arm) -> None:
        if not history:
            return
        try:
            evidence = training_visuals(config, history, arm, compact=compact)
            path = _verified_path(config, evidence["figures"][0])
            picture = Image(filename=str(path), width=1200)
            if arm not in handles:
                handles[arm] = display(picture, display_id=True)
            elif handles[arm] is not None:
                handles[arm].update(picture)
        except Exception as error:
            print(f"Chart refresh unavailable ({error}); saved scientific history is preserved.")

    return update


def display_training(
    config: ExperimentConfig,
    arm: Arm,
    *,
    compact: bool = False,
    summary_only: bool = False,
) -> None:
    Image, Markdown, display = _ipython_display()

    path = config.project_path("artifacts") / "training" / f"{arm}.json"
    if not path.is_file():
        display(
            Markdown("Training evidence unavailable. Enable the full run and execute this arm.")
        )
        return
    saved = json.loads(path.read_text())
    if saved.get("provenance") != experiment_provenance(config):
        raise VisualEvidenceError("training history belongs to another protocol")
    if compact:
        from src.notebook_views import validation_note

        if not summary_only:
            visuals = training_visuals(config, saved["history"], arm, final=True, compact=True)
            display(Image(filename=str(_verified_path(config, visuals["figures"][0])), width=1200))
        display(Markdown(validation_note(saved["history"], arm)))
    else:
        display_figures(
            config, training_visuals(config, saved["history"], arm, final=True, compact=False)
        )


def _evaluation(config: ExperimentConfig) -> dict[str, Any]:
    for name in ("evaluation.partial.json", "evaluation.json"):
        path = config.project_path("results") / name
        if path.is_file():
            result = json.loads(path.read_text())
            if result.get("config_sha256") == config.sha256:
                for arm, saved in result.get("arms", {}).items():
                    if arm in ARMS and saved.get("checkpoint_sha256") != sha256_file(
                        checkpoint_path(config, cast(Arm, arm))
                    ):
                        raise VisualEvidenceError("saved evaluation checkpoint identity mismatch")
                return cast(dict[str, Any], result)
    raise VisualEvidenceError(
        "current evaluation unavailable; execute the evaluation sections first"
    )


def _array(
    config: ExperimentConfig, arm: Arm, partition: str, condition: str = "clean"
) -> dict[str, np.ndarray[Any, Any]]:
    group = "calibration" if partition == "calibration" else f"{partition}/{arm}"
    name = arm if partition == "calibration" else condition
    path = config.project_path("results") / "per_sample" / group / f"{name}.npz"
    metadata = path.with_suffix(".json")
    if not path.is_file() or not metadata.is_file():
        raise VisualEvidenceError(f"missing current arrays: {partition}/{arm}/{condition}")
    receipt = json.loads(metadata.read_text())
    expected = _artifact_provenance(config, arm, sha256_file(checkpoint_path(config, arm)))
    if receipt.get("provenance") != expected or receipt.get("npz_sha256") != sha256_file(path):
        raise VisualEvidenceError(f"stale/corrupt arrays: {partition}/{arm}/{condition}")
    from src.data import load_registered_splits
    from src.evaluation import _validate_arrays

    arrays = load_arrays(path)
    records = load_registered_splits(config)["test" if partition == "full-test" else partition]
    _validate_arrays(arrays, records, config.integer("dataset", "classes"))
    return arrays


def _accuracy_bars(axis: Any, metrics: dict[str, dict[str, Any]], title: str) -> None:
    names = ("Overall", "Macro", "Cat recall", "Dog recall")
    x = np.arange(len(names))
    for index, (arm, row) in enumerate(metrics.items()):
        values = [
            row.get("accuracy"),
            row.get("macro_accuracy"),
            row.get("per_class_accuracy", {}).get("0"),
            row.get("per_class_accuracy", {}).get("1"),
        ]
        axis.bar(
            x + (index - 0.5) * 0.36,
            [100 * _number(v) for v in values],
            0.34,
            label=LABELS[arm],
            color=COLORS[arm],
        )
    axis.set_xticks(x, names)
    _finish(axis, title, "Accuracy / recall (%)", percent=True)


def _reliability(
    axis: Any,
    logits: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    temperature: float,
    bins: int,
    name: str,
) -> None:
    probabilities = softmax(logits.astype(np.float64) / temperature)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    bin_ids = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    means, accuracies = [], []
    for index in range(bins):
        mask = bin_ids == index
        if mask.any():
            means.append(100 * float(confidence[mask].mean()))
            accuracies.append(100 * float(correct[mask].mean()))
    _curve(axis, means, accuracies, name)


def _image_caption(
    arrays: dict[str, np.ndarray[Any, Any]], sample_id: str, temperature: float
) -> str:
    matches = np.flatnonzero(arrays["sample_ids"] == sample_id)
    if len(matches) != 1:
        raise VisualEvidenceError(
            "gallery image does not align with exactly one measured prediction"
        )
    probabilities = softmax(arrays["logits"][matches].astype(np.float64) / temperature)[0]
    prediction = int(probabilities.argmax())
    return f"{SPECIES[prediction]} · confidence {probabilities[prediction]:.1%}"


def _input_gallery(
    config: ExperimentConfig,
    arm: Arm,
    conditions: list[str],
    temperature: float,
    *,
    attack: bool = False,
) -> Any:
    partition = "attack" if attack else "full-test"
    arrays = [_array(config, arm, partition, condition) for condition in conditions]
    ids = arrays[-1].get("gallery_sample_ids")
    if ids is None or not len(ids):
        raise VisualEvidenceError("current input gallery unavailable; no historical substitute")
    columns = len(conditions) + (1 if attack else 0)
    figure, axes = plt.subplots(
        len(ids),
        columns,
        figsize=(4 * columns, 3 * len(ids)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, raw_id in enumerate(ids):
        sample_id = str(raw_id)
        clean_pixels: np.ndarray[Any, Any] | None = None
        attacked_pixels: np.ndarray[Any, Any] | None = None
        for column, (condition, saved) in enumerate(zip(conditions, arrays, strict=True)):
            if attack:
                gallery = arrays[-1]
                pixels = gallery[
                    "gallery_clean_pixels" if condition == "clean" else "gallery_attack_pixels"
                ][row]
                if condition == "clean":
                    clean_pixels = pixels
                else:
                    attacked_pixels = pixels
            else:
                matching = np.flatnonzero(saved["gallery_sample_ids"] == sample_id)
                if len(matching) != 1:
                    raise VisualEvidenceError("corruption gallery IDs do not align")
                pixels = saved["gallery_pixels"][matching[0]]
            axes[row, column].imshow(pixels.transpose(1, 2, 0))
            axes[row, column].set_title(
                f"{condition.replace('_', ' ')}\n{_image_caption(saved, sample_id, temperature)}",
                fontsize=9,
            )
        if attack and clean_pixels is not None and attacked_pixels is not None:
            difference = np.abs(attacked_pixels - clean_pixels)
            axes[row, -1].imshow(np.clip(difference.transpose(1, 2, 0) * 32, 0, 1))
            axes[row, -1].set_title(
                f"|Δ| ×32 · display only\nactual L∞={difference.max():.5f}", fontsize=9
            )
        label = int(arrays[-1]["gallery_labels"][row])
        axes[row, 0].set_ylabel(f"{sample_id}\ntrue: {SPECIES[label]}", fontsize=9)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(f"{LABELS[arm]} · actual inputs captured during evaluation")
    return figure


def evaluation_visuals(
    config: ExperimentConfig, section: EvaluationSection | str
) -> dict[str, Any]:
    """Render only recorded arrays/metrics; never constructs or evaluates a model."""
    result = _evaluation(config)
    models = {arm: result["arms"][arm] for arm in ARMS}
    checkpoints: dict[str, str] = {arm: sha256_file(checkpoint_path(config, arm)) for arm in ARMS}
    for arm, saved in models.items():
        if saved.get("checkpoint_sha256") != checkpoints[arm]:
            raise VisualEvidenceError("evaluation checkpoint identity mismatch")
    figures: list[dict[str, Any]] = []
    group = f"evaluation/{section}"
    if section == "clean":
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        _accuracy_bars(
            axes[0],
            {a: m["full_test"]["clean"] for a, m in models.items()},
            "Full official test · clean",
        )
        for axis, (arm, saved) in zip(axes[1:], models.items(), strict=True):
            _confusion(axis, saved["full_test"]["clean"], f"{LABELS[arm]} · clean test")
        figures.append(
            _save(
                config,
                group,
                "clean-recognition",
                figure,
                "Full official test. Macro accuracy gives cats and dogs equal weight.",
            )
        )
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for axis, (arm, saved) in zip(axes, models.items(), strict=True):
            arrays = _array(config, arm, "full-test")
            axis.plot([0, 100], [0, 100], "--", color="gray", label="Perfect reliability")
            for temperature, name in (
                (1.0, "Before scaling"),
                (float(saved["policy"]["temperature"]), "After scaling"),
            ):
                _reliability(
                    axis,
                    arrays["logits"],
                    arrays["labels"],
                    temperature,
                    config.integer("calibration", "ece_bins"),
                    name,
                )
            _finish(
                axis, f"{LABELS[arm]} · clean test reliability", "Bin accuracy (%)", percent=True
            )
            axis.set(xlabel="Mean confidence (%)", xlim=(0, 100))
        figures.append(
            _save(
                config,
                group,
                "clean-reliability",
                figure,
                "Temperature fitted on calibration only; diagrams measured on test. "
                "Empty confidence bins are omitted, not assigned zero accuracy.",
            )
        )
        conditions, attack = ["clean"], False
    elif section in {"fgsm", "pgd"}:
        figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for index, (arm, saved) in enumerate(models.items()):
            metrics = saved["attack_subset"][section]
            offset = (index - 0.5) * 0.36
            axes[0].bar(
                np.arange(2) + offset,
                [100 * metrics["clean_accuracy_on_subset"], 100 * metrics["robust_accuracy"]],
                0.34,
                color=COLORS[arm],
                label=LABELS[arm],
            )
            axes[1].bar(
                np.arange(2) + offset,
                [100 * metrics["per_class"][str(i)]["robust_accuracy"] for i in range(2)],
                0.34,
                color=COLORS[arm],
                label=LABELS[arm],
            )
            success = 100 * _number(metrics.get("attack_success_clean_correct"))
            if np.isfinite(success):
                axes[2].bar(LABELS[arm], success, color=COLORS[arm])
            else:
                axes[2].text(index, 5, "Undefined: no clean-correct samples", rotation=90)
        axes[0].set_xticks([0, 1], ["Clean subset", section.upper()])
        axes[1].set_xticks([0, 1], SPECIES)
        _finish(axes[0], "Balanced paired attack subset", "Accuracy (%)", percent=True)
        _finish(axes[1], "Attacked species recall", "Recall (%)", percent=True)
        _finish(axes[2], "Attack success among clean-correct", "Success (%)", percent=True)
        figures.append(
            _save(
                config,
                group,
                "attack-recognition",
                figure,
                "Balanced fixed attack subset, not full-test accuracy. "
                "Attack success denominator contains clean-correct samples only.",
            )
        )
        if section == "pgd":
            figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
            for arm, saved in models.items():
                values = saved["attack_subset"]["pgd"]["cumulative_restart_robust_accuracy"]
                _curve(
                    axis,
                    list(range(1, len(values) + 1)),
                    [100 * v for v in values],
                    LABELS[arm],
                    color=COLORS[arm],
                )
            _finish(
                axis, "Cumulative PGD survival across restarts", "Robust accuracy (%)", percent=True
            )
            axis.set_xlabel("Completed restart")
            figures.append(
                _save(
                    config,
                    group,
                    "restart-survival",
                    figure,
                    "Cumulative search failures across restarts; finite attacks are "
                    "not a robustness certificate.",
                )
            )
        conditions, attack = ["clean", section], True
    elif section == "confidence":
        return confidence_visuals(config)
    else:
        levels = [
            c.identifier
            for c in registered_corruptions(config.section("corruptions"))
            if c.name == section
        ]
        if not levels:
            raise ValueError(f"unsupported evaluation view: {section}")
        conditions, attack = ["clean", *levels], False
        figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        metrics = (
            ("accuracy", "Overall accuracy"),
            ("macro_accuracy", "Macro accuracy"),
            ("cat", "Cat recall"),
            ("dog", "Dog recall"),
            ("coverage", "Accepted coverage"),
            ("selective_risk", "Selective risk"),
        )
        for axis, (metric, title) in zip(axes.flat, metrics, strict=True):
            for arm, saved in models.items():
                rows = [saved["full_test"][c] for c in conditions]
                if metric in {"cat", "dog"}:
                    values = [r["per_class_accuracy"][str(SPECIES.index(metric))] for r in rows]
                elif metric in {"coverage", "selective_risk"}:
                    values = [r["risk"][metric] for r in rows]
                else:
                    values = [r[metric] for r in rows]
                _curve(
                    axis,
                    list(range(len(conditions))),
                    [100 * _number(v) for v in values],
                    LABELS[arm],
                    color=COLORS[arm],
                )
            axis.set_xticks(
                range(len(conditions)), ["clean", *[c.rsplit("-", 1)[-1] for c in levels]]
            )
            axis.set_xlabel("Transform parameter (family-specific)")
            _finish(axis, title, "Percent (%)", percent=True)
        figure.suptitle(f"{section.replace('_', ' ').title()} · full official test")
        figures.append(
            _save(
                config,
                group,
                "condition-dashboard",
                figure,
                "Same full-test images at each configured level. Temperature and "
                "confidence threshold remain clean-calibration fitted; no refitting. "
                "Parameters across transform families are not interchangeable severity units.",
            )
        )
    for arm, saved in models.items():
        gallery = _input_gallery(
            config, arm, conditions, float(saved["policy"]["temperature"]), attack=attack
        )
        figures.append(
            _save(
                config,
                group,
                f"{arm}-inputs",
                gallery,
                "Fixed sample IDs and exact model inputs saved during this evaluation. "
                "Predictions/confidence align with recorded logits.",
                shareable=False,
            )
        )
    return _manifest(
        config,
        f"evaluation-{section}",
        figures,
        checkpoints,
        population="balanced attack subset" if attack else "full official test",
    )


def confidence_visuals(config: ExperimentConfig) -> dict[str, Any]:
    result = _evaluation(config)
    conditions = [
        "clean",
        *[c.identifier for c in registered_corruptions(config.section("corruptions"))],
    ]
    models = {arm: result["arms"][arm] for arm in ARMS}
    missing = [
        f"{arm}/{condition}"
        for arm, saved in models.items()
        for condition in conditions
        if condition not in saved["full_test"]
    ]
    if missing:
        raise VisualEvidenceError(
            "confidence summary needs completed conditions: " + ", ".join(missing)
        )
    figure, axes = plt.subplots(1, 2, figsize=(16, 4.5), constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("coverage", "selective_risk"),
        ("Accepted coverage (%)", "Selective risk (%)"),
        strict=True,
    ):
        values = np.asarray(
            [
                [100 * _number(saved["full_test"][c]["risk"][metric]) for c in conditions]
                for saved in models.values()
            ]
        )
        artist = axis.imshow(values, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        axis.set(
            xticks=range(len(conditions)),
            xticklabels=conditions,
            yticks=[0, 1],
            yticklabels=[LABELS[a] for a in ARMS],
            title=title,
        )
        axis.tick_params(axis="x", labelrotation=55)
        for row in range(2):
            for column in range(len(conditions)):
                text = f"{values[row, column]:.1f}" if np.isfinite(values[row, column]) else "—"
                axis.text(column, row, text, ha="center", va="center", color="white", fontsize=8)
        figure.colorbar(artist, ax=axis, shrink=0.65)
    figures = [
        _save(
            config,
            "evaluation/confidence",
            "coverage-risk",
            figure,
            "Clean-fitted operating points under all full-test shifts. Undefined risk "
            "when no predictions are accepted is shown as unavailable, never zero.",
        )
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    severe = [
        "clean",
        *[c.identifier for c in registered_corruptions(config.section("corruptions"))][1::2],
    ]
    for axis, (arm, saved) in zip(axes, models.items(), strict=True):
        for condition in severe:
            arrays = _array(config, arm, "full-test", condition)
            probabilities = softmax(
                arrays["logits"].astype(np.float64) / float(saved["policy"]["temperature"])
            )
            coverage, risk, _aurc = tie_aware_risk_coverage(
                probabilities.max(axis=1), probabilities.argmax(axis=1) == arrays["labels"]
            )
            axis.step(
                100 * coverage,
                100 * risk,
                where="pre",
                label=condition,
                linewidth=1.5,
                marker="o",
                markersize=3,
            )
        axis.axvline(
            100 * config.number("calibration", "target_coverage"),
            color="gray",
            linestyle="--",
            linewidth=1,
        )
        _finish(
            axis, f"{LABELS[arm]} · tie-aware risk–coverage", "Selective risk (%)", percent=True
        )
        axis.set(xlabel="Accepted coverage (%)", xlim=(0, 100))
    figures.append(
        _save(
            config,
            "evaluation/confidence",
            "risk-coverage",
            figure,
            "Clean plus the second registered level per family. Tied confidences "
            "are accepted together; no tie-breaking performance is invented.",
        )
    )
    return _manifest(
        config,
        "evaluation-confidence",
        figures,
        {arm: sha256_file(checkpoint_path(config, arm)) for arm in ARMS},
        population="full official test",
        no_additional_inference=True,
    )


def display_evaluation(config: ExperimentConfig, section: EvaluationSection | str) -> None:
    _, Markdown, display = _ipython_display()

    try:
        evidence = evaluation_visuals(config, section)
    except (VisualEvidenceError, KeyError) as error:
        display(
            Markdown(
                f"**Current-run visuals unavailable:** {error}. "
                "Run the required preceding sections; no saved historical image is substituted."
            )
        )
        return
    display_figures(config, evidence)


FEATURE_GROUPS = (
    ("Image → CNN stages → prediction", {"stage_walkthrough"}),
    ("Actual filters and synthetic channel preferences", {"kernels", "activation_maximization"}),
    ("Strong real calibration patches", {"real_patches"}),
    ("Image responses and sensitivity", {"activation_walkthrough", "anchor_walkthrough"}),
    (
        "Class influence and sanity controls",
        {"class_attribution", "attribution", "localization_summary"},
    ),
)


def _display_compact_features(
    config: ExperimentConfig, evidence: dict[str, Any], section: FeatureSection
) -> None:
    from src.notebook_views import (
        MODEL_NAMES,
        compact_records,
        feature_note,
        real_patch_preview,
        section_measurements,
        synthetic_preview,
    )

    Image, Markdown, display = _ipython_display()
    records = compact_records(evidence, section)
    if not records:
        display(
            Markdown(f"{section.replace('_', ' ').title()} pictures pending; no old substitute.")
        )
        return
    notes: dict[str, str] = {}
    receipts: dict[str, tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]] = {}
    if section in {"synthetic", "real_patches", "gradcam", "diagnostics"}:
        for arm in ARMS:
            if (
                arm in evidence.get("section_receipts", {})
                and section in evidence["section_receipts"][arm]
            ):
                receipts[arm] = section_measurements(config, evidence, arm, section)
                notes[arm] = feature_note(section, *receipts[arm])
    shown_patch_arms: set[str] = set()
    for record in records:
        path = _verified_path(config, record)
        arm = record["arm"]
        if section == "real_patches" and arm in shown_patch_arms:
            continue
        title = MODEL_NAMES[arm]
        sample = record.get("sample_id")
        if sample and sample != "fixed-test-anchor-set":
            title += f" · {sample}"
        elif section == "real_patches":
            title += " · early / middle / late real patches"
        elif section == "diagnostics":
            title += (
                " · "
                + {
                    "randomization_control": "learned vs scrambled weights",
                    "localization_summary": "highlighted vs random regions",
                }[record["kind"]]
            )
        display(Markdown(f"**{title}**"))
        metadata = {
            "source_image": record["path"],
            "source_image_sha256": record["sha256"],
            "feature_identity": evidence.get("identity"),
        }
        if section in {"synthetic", "real_patches"} and arm in receipts:
            receipt, arrays = receipts[arm]
            pixels = (
                synthetic_preview(arrays, arm)
                if section == "synthetic"
                else real_patch_preview(receipt, arrays, arm)
            )
            picture = Image(data=pixels, width=1200)
            metadata.update(
                {
                    "source_arrays": receipt["arrays_path"],
                    "source_arrays_sha256": receipt["arrays_sha256"],
                    "presentation": "saved-array relayout; no inference or optimization",
                }
            )
            if section == "real_patches":
                shown_patch_arms.add(arm)
        else:
            with PILImage.open(path) as original:
                preview = original.convert("RGB")
                preview.thumbnail((1200, 4096))
                buffer = io.BytesIO()
                preview.save(buffer, format="PNG")
            picture = Image(data=buffer.getvalue(), width=1200)
        display(picture, metadata=metadata)
        if arm in notes and notes[arm]:
            display(Markdown(notes.pop(arm)))
    all_records = [
        record for record in evidence.get("figures", []) if record.get("section") == section
    ]
    for record in all_records:
        _verified_path(config, record)
    links = " · ".join(
        f"[{MODEL_NAMES[record['arm']]} "
        f"{record.get('sample_id') or record['kind'].replace('_', ' ')}]"
        f"({config.root / record['path']})"
        for record in all_records
    )
    display(Markdown("Full-size pictures / additional details: " + links + "."))
    if section in {"stages", "activations"}:
        display(
            Markdown(
                "Preview: first registered cat and dog, the same for both models—not chosen "
                "for attractive maps. Use `compact=False` to show all four."
            )
        )
    if section == "real_patches":
        display(
            Markdown(
                "Preview: first two calibration-ranked channels at each level, using their "
                "recorded strongest image. All channels and runner-up patches remain in the "
                "full pictures above."
            )
        )
    available = {record.get("arm") for record in records}
    missing = [LABELS[arm] for arm in ARMS if arm not in available]
    if missing:
        display(Markdown("Pictures still pending for: " + ", ".join(missing) + "."))


def display_features(
    config: ExperimentConfig, section: FeatureSection | None = None, *, compact: bool = False
) -> None:
    _, Markdown, display = _ipython_display()

    if section is not None and section not in FEATURE_SECTIONS:
        raise ValueError(f"unsupported feature view: {section}")
    evidence = None
    for name in ("feature_visualizations.partial.json", "feature_visualizations.json"):
        path = config.project_path("results") / name
        if path.is_file():
            value = json.loads(path.read_text())
            if value.get("config_sha256") == config.sha256:
                evidence = value
                break
    if evidence is None:
        display(
            Markdown(
                "Current feature evidence unavailable; execute the corresponding feature "
                "cell. No historical picture is substituted."
            )
        )
        return
    from src.feature_sections import validate_feature_evidence

    validate_feature_evidence(config, evidence, require_complete=False)
    if compact:
        for selected in (section,) if section is not None else FEATURE_SECTIONS:
            _display_compact_features(config, evidence, selected)
        return
    if section is not None:
        records = [
            record for record in evidence.get("figures", []) if record.get("section") == section
        ]
        if not records:
            display(
                Markdown(
                    f"**{section.replace('_', ' ').title()} unavailable:** this family has no "
                    "matching runtime figures yet. Run its cell; no historical substitute is used."
                )
            )
            return
        display(Markdown(f"## {section.replace('_', ' ').title()} · current-run feature evidence"))
        display_figures(config, {"config_sha256": config.sha256, "figures": records})
        available_arms = {record.get("arm") for record in records}
        missing_arms = [LABELS[arm] for arm in ARMS if arm not in available_arms]
        if missing_arms:
            display(Markdown("Figures still pending for: " + ", ".join(missing_arms) + "."))
        return
    shown: set[str] = set()
    for heading, kinds in FEATURE_GROUPS:
        records = [record for record in evidence["figures"] if record["kind"] in kinds]
        if records:
            display(Markdown(f"## {heading}"))
            display_figures(config, {"config_sha256": config.sha256, "figures": records})
            shown.update(record["path"] for record in records)
    remaining = [record for record in evidence["figures"] if record["path"] not in shown]
    if remaining:
        display(Markdown("## Quantitative context and remaining diagnostics"))
        display_figures(config, {"config_sha256": config.sha256, "figures": remaining})
    display(
        Markdown(
            "### Interpretation limits\n\n"
            + "\n".join(f"- {item}" for item in evidence.get("limitations", []))
        )
    )


def current_visual_manifests(
    config: ExperimentConfig, *, feature_only: bool = False
) -> list[dict[str, Any]]:
    """Only config-, method-, checkpoint- and file-matched visual receipts qualify."""
    root = config.project_path("results") / "visualizations"
    expected = {arm: sha256_file(checkpoint_path(config, arm)) for arm in ARMS}
    output: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if feature_only and path.name != "report.json" and not path.name.startswith("training-"):
            continue
        value = json.loads(path.read_text())
        group = value.get("group", "")
        if feature_only and group != "report" and not group.startswith("training-"):
            continue
        if group.startswith("evaluation-"):
            evaluation_provenance = {
                arm: _artifact_provenance(config, arm, expected[arm]) for arm in ARMS
            }
            if value.get("evaluation_provenance") != evaluation_provenance:
                continue
        if (
            value.get("config_sha256") != config.sha256
            or value.get("plot_source_sha256") != sha256_file(Path(__file__))
            or any(
                expected.get(arm) != digest
                for arm, digest in value.get("checkpoint_sha256", {}).items()
            )
        ):
            continue
        if group == "report":
            feature_path = config.project_path("results") / "feature_visualizations.json"
            if not feature_path.is_file() or value.get("feature_evidence_sha256") != sha256_file(
                feature_path
            ):
                continue
            from src.reporting import ReportError, load_current_feature_evidence

            try:
                features = load_current_feature_evidence(config)
            except ReportError:
                continue
            if value.get("feature_method_sha256") != features["method_sha256"]:
                continue
        for record in value.get("figures", []):
            _verified_path(config, record)
        output.append(value)
    return output


def report_visuals(
    config: ExperimentConfig, *, training_history_status: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Feature completion and a two-model preview, reading only registered current images."""
    from src.reporting import ReportError, load_current_feature_evidence

    try:
        features = load_current_feature_evidence(config)
    except ReportError as error:
        raise VisualEvidenceError(
            f"report feature evidence is incomplete or stale: {error}"
        ) from error
    figure, axis = plt.subplots(figsize=(10, 2.7), constrained_layout=True)
    matrix = np.ones((2, len(FEATURE_SECTIONS)))
    axis.imshow(matrix, vmin=0, vmax=1, cmap="Greens", aspect="auto")
    axis.set(
        xticks=range(len(FEATURE_SECTIONS)),
        xticklabels=[name.replace("_", " ") for name in FEATURE_SECTIONS],
        yticks=[0, 1],
        yticklabels=[LABELS[arm] for arm in ARMS],
        title="Current feature evidence · all eight families verified",
    )
    axis.tick_params(axis="x", labelrotation=30)
    for row in range(2):
        for column in range(len(FEATURE_SECTIONS)):
            axis.text(column, row, "✓", ha="center", va="center", fontsize=12)
    figures = [
        _save(
            config,
            "report",
            "artifact-completion",
            figure,
            "Current section receipts, model state, checkpoint and image hashes verified. "
            "Completion is not a performance or robustness measurement.",
        )
    ]
    selected = []
    for arm in ARMS:
        for section in ("stages", "synthetic"):
            candidates = [
                record
                for record in features["figures"]
                if record.get("arm") == arm and record.get("section") == section
            ]
            if candidates:
                selected.append(candidates[0])
    if selected:
        figure, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for axis in axes.flat:
            axis.axis("off")
        for axis, record in zip(axes.flat, selected, strict=False):
            with PILImage.open(_verified_path(config, record)) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"{LABELS[record['arm']]} · {record['section']}", fontsize=10)
        figure.suptitle("Both models · first registered stage and preferred-pattern views")
        figures.append(
            _save(
                config,
                "report",
                "feature-gallery-preview",
                figure,
                "Fixed current-run views for both models; anchor predictions are examples, "
                "not an accuracy estimate. Full images are in the read-only companion.",
                shareable=all(record.get("shareable") for record in selected),
            )
        )
    return _manifest(
        config,
        "report",
        figures,
        {arm: sha256_file(checkpoint_path(config, arm)) for arm in ARMS},
        feature_evidence_sha256=sha256_file(
            config.project_path("results") / "feature_visualizations.json"
        ),
        feature_method_sha256=features["method_sha256"],
        training_history_status=dict(training_history_status or {}),
    )


def ensure_report_visuals(config: ExperimentConfig) -> None:
    """Create the compact feature dashboard and optional saved-history charts; no inference."""
    training_status: dict[str, str] = {}
    for arm in ARMS:
        path = config.project_path("artifacts") / "training" / f"{arm}.json"
        if not path.is_file():
            training_status[arm] = "unavailable: no saved history"
            continue
        try:
            saved = json.loads(path.read_text())
            if saved.get("provenance") != experiment_provenance(config):
                raise VisualEvidenceError("saved history belongs to another protocol")
            if not saved.get("history"):
                raise VisualEvidenceError("saved history has no measured epochs")
            training_visuals(config, saved["history"], arm, final=True, compact=True)
        except (ValueError, OSError, VisualEvidenceError) as error:
            training_status[arm] = f"unavailable: {error}"
            print(f"{LABELS[arm]} training chart unavailable: {error}; no curve inferred.")
        else:
            training_status[arm] = "verified saved history; compact chart registered"
    report_visuals(config, training_history_status=training_status)


def display_report(config: ExperimentConfig, *, compact: bool = False) -> None:
    _, Markdown, display = _ipython_display()
    if compact:
        from src.reporting import ReportError, load_current_feature_evidence

        try:
            load_current_feature_evidence(config)
        except ReportError as error:
            display(
                Markdown(
                    f"Current feature summary unavailable: {error}. No old pictures substituted."
                )
            )
            return
        display(
            Markdown(
                "**Saved walkthrough complete:** eight feature families for both model "
                "checkpoints. Completion means the pictures were generated and verified—not "
                "that both models work well."
            )
        )
        for arm in ARMS:
            path = config.project_path("artifacts") / "training" / f"{arm}.json"
            if path.is_file():
                saved = json.loads(path.read_text())
                if saved.get("provenance") != experiment_provenance(config):
                    display(
                        Markdown(
                            f"{LABELS[arm]}: saved validation details are stale; "
                            "no result inferred."
                        )
                    )
                    continue
                if saved.get("history"):
                    from src.notebook_views import validation_note

                    note = validation_note(saved["history"], arm)
                    warning = next(
                        (line for line in note.splitlines() if line.startswith("**Watch out:")),
                        None,
                    )
                    if warning:
                        display(Markdown(warning))
        display(
            Markdown(
                "Read the two models' pictures as descriptions of this short run. Earlier "
                "filters can already come from ImageNet pretraining. Bright highlights are "
                "not verified eye/ear detectors, and Grad-CAM is not causal proof. "
                "No attack accuracy was measured here."
            )
        )
        report_paths = [
            config.project_path("reports") / "technical-report.md",
            config.project_path("reports") / "cat_dog_feature_results.ipynb",
        ]
        display(
            Markdown(
                "Saved reports: "
                + " · ".join(f"[{path.name}]({path})" for path in report_paths if path.is_file())
                + "."
            )
        )
        return
    try:
        current = current_visual_manifests(config, feature_only=True)
        evidence = next((value for value in current if value["group"] == "report"), None)
        if evidence is None:
            evidence = report_visuals(config)
    except (VisualEvidenceError, FileNotFoundError) as error:
        display(
            Markdown(
                f"**Fresh feature results pending:** {error}. "
                "Finish both models' eight feature families; no historical substitute is used."
            )
        )
        return
    display_figures(config, evidence)
    history_status = evidence.get("training_history_status", {})
    if history_status:
        display(
            Markdown(
                "Saved training history: "
                + "; ".join(f"{LABELS[arm]} — {value}" for arm, value in history_status.items())
                + "."
            )
        )
    candidates = [
        config.project_path("reports") / "technical-report.md",
        config.project_path("reports") / "cat_dog_feature_results.ipynb",
        config.project_path("artifacts") / "release" / "local-release-manifest.json",
    ]
    links = [f"- [{path.name}]({path})" for path in candidates if path.is_file()]
    display(Markdown("### Current feature summary and illustrated evidence\n\n" + "\n".join(links)))
