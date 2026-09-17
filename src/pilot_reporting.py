"""Isolated, evidence-backed reporting for the exploratory PGD recovery pilot."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.config import ExperimentConfig
from src.evaluation import load_arrays
from src.io_utils import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    environment_snapshot,
    git_state,
    sha256_file,
    source_hash,
    utc_now,
)
from src.progress import status
from src.training import checkpoint_path


class PilotReportError(RuntimeError):
    """The pilot evidence is incomplete, stale, or not isolated from the original run."""


ORIGINAL_CONFIG_SHA256 = "3b96c26d328cffc308cb6de70e070748bba3c9fe7a7b229ccba6749f8aec2a91"
ORIGINAL_EVALUATION_SHA256 = "5a405fdff008137e9012062bcdbcb8c52d1caed128e84c5e3d4738fa13037b1c"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PilotReportError(f"required pilot evidence is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PilotReportError(f"expected a JSON object: {path}")
    return cast(dict[str, Any], document)


def _relative(config: ExperimentConfig, path: Path) -> str:
    return path.resolve().relative_to(config.root.resolve()).as_posix()


def _reference(config: ExperimentConfig) -> Path:
    pilot = cast(dict[str, Any], config.raw.get("pilot", {}))
    path = (
        config.root / str(pilot.get("original_evaluation", "results/generated/evaluation.json"))
    ).resolve()
    if not path.is_relative_to(config.root.resolve()):
        raise PilotReportError("original evaluation must remain inside the standalone project")
    return path


def _validate_isolation(config: ExperimentConfig, original_path: Path) -> None:
    original_outputs = {
        "artifacts": config.root / "artifacts",
        "results": original_path.parent,
        "figures": config.root / "figures/generated",
        "reports": config.root / "reports/generated",
    }
    for key, original in original_outputs.items():
        if config.project_path(key) == original.resolve():
            raise PilotReportError(f"pilot paths.{key} must not overwrite the original run")


def _species_metrics(arm_result: dict[str, Any]) -> dict[str, Any]:
    clean = arm_result["full_test"]["clean"]
    pgd = arm_result["attack_subset"]["pgd"]
    confusion = np.asarray(clean["confusion_matrix"], dtype=np.int64)
    if confusion.shape != (2, 2):
        raise PilotReportError("pilot comparison requires the two-species classifier")
    return {
        "clean": {
            "accuracy": float(clean["accuracy"]),
            "macro_accuracy": float(clean["macro_accuracy"]),
            "cat_recall": float(clean["per_class_accuracy"]["0"]),
            "dog_recall": float(clean["per_class_accuracy"]["1"]),
            "count": int(confusion.sum()),
            "confusion_matrix": confusion.tolist(),
            "prediction_counts": {
                "cat": int(confusion[:, 0].sum()),
                "dog": int(confusion[:, 1].sum()),
            },
        },
        "pgd": {
            "accuracy": float(pgd["robust_accuracy"]),
            "cat_recall": float(pgd["per_class"]["0"]["robust_accuracy"]),
            "dog_recall": float(pgd["per_class"]["1"]["robust_accuracy"]),
            "count": sum(int(value["count"]) for value in pgd["per_class"].values()),
            "clean_subset_accuracy": float(pgd["clean_accuracy_on_subset"]),
            "attack_success_clean_correct": pgd["attack_success_clean_correct"],
            "clean_correct_count": int(pgd["clean_correct_count"]),
            "attack_success_count": int(pgd["attack_success_count"]),
        },
        "fgsm_accuracy": float(arm_result["attack_subset"]["fgsm"]["robust_accuracy"]),
        "checkpoint_sha256": arm_result["checkpoint_sha256"],
        "attack_strength_check": arm_result["attack_strength_check"],
    }


def _paired_evidence(
    config: ExperimentConfig, original_path: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_ids: np.ndarray[Any, Any] | None = None
    reference_labels: np.ndarray[Any, Any] | None = None
    files: dict[str, Any] = {}
    for row in rows:
        results = config.project_path("results") if row["pilot"] else original_path.parent
        directory = results / "per_sample" / "attack" / row["arm"]
        clean_path = directory / "clean.npz"
        pgd_path = directory / "pgd.npz"
        clean = load_arrays(clean_path)
        pgd = load_arrays(pgd_path)
        if not np.array_equal(clean["sample_ids"], pgd["sample_ids"]) or not np.array_equal(
            clean["labels"], pgd["labels"]
        ):
            raise PilotReportError(f"clean/PGD samples are not aligned for {row['label']}")
        if reference_ids is None:
            reference_ids, reference_labels = clean["sample_ids"], clean["labels"]
        else:
            assert reference_labels is not None
            if not np.array_equal(reference_ids, clean["sample_ids"]) or not np.array_equal(
                reference_labels, clean["labels"]
            ):
                raise PilotReportError("pilot and original attack subsets differ")
        labels = pgd["labels"].astype(np.int64)
        predictions = pgd["logits"].argmax(axis=1)
        if not np.isfinite(pgd["logits"]).all() or not np.isfinite(clean["logits"]).all():
            raise PilotReportError("non-finite PGD logits cannot be reported")
        if set(labels.tolist()) != {0, 1} or len(set(clean["sample_ids"])) != len(labels):
            raise PilotReportError("attack subset needs distinct IDs and both species")
        if not np.isclose(float((predictions == labels).mean()), row["metrics"]["pgd"]["accuracy"]):
            raise PilotReportError("PGD metrics do not match their per-sample evidence")
        confusion = np.zeros((2, 2), dtype=np.int64)
        np.add.at(confusion, (labels, predictions), 1)
        metrics = row["metrics"]["pgd"]
        for species, key in ((0, "cat_recall"), (1, "dog_recall")):
            if not np.isclose(confusion[species, species] / confusion[species].sum(), metrics[key]):
                raise PilotReportError("PGD species recalls do not match per-sample evidence")
        clean_correct = clean["logits"].argmax(axis=1) == labels
        successes = clean_correct & (predictions != labels)
        if (
            metrics["count"] != len(labels)
            or metrics["clean_correct_count"] != int(clean_correct.sum())
            or metrics["attack_success_count"] != int(successes.sum())
            or not np.isclose(metrics["clean_subset_accuracy"], clean_correct.mean())
        ):
            raise PilotReportError("attack denominators do not match per-sample evidence")
        row["metrics"]["pgd"]["confusion_matrix"] = confusion.tolist()
        row["metrics"]["pgd"]["prediction_counts"] = {
            "cat": int(confusion[:, 0].sum()),
            "dog": int(confusion[:, 1].sum()),
        }
        files[row["id"]] = {
            "clean": {"path": _relative(config, clean_path), "sha256": sha256_file(clean_path)},
            "pgd": {"path": _relative(config, pgd_path), "sha256": sha256_file(pgd_path)},
        }
    assert reference_ids is not None and reference_labels is not None
    return {
        "same_ids_and_labels": True,
        "count": len(reference_ids),
        "per_species_counts": {
            "cat": int((reference_labels == 0).sum()),
            "dog": int((reference_labels == 1).sum()),
        },
        "files": files,
    }


def _comparison(
    config: ExperimentConfig, original: dict[str, Any], pilot: dict[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = [
        {
            "id": "original_standard",
            "label": "Original clean training",
            "arm": "standard",
            "pilot": False,
            "metrics": _species_metrics(original["arms"]["standard"]),
        },
        {
            "id": "original_pgd",
            "label": "Original pure PGD (failed)",
            "arm": "adversarial",
            "pilot": False,
            "metrics": _species_metrics(original["arms"]["adversarial"]),
        },
        {
            "id": "exploratory_pilot",
            "label": "Exploratory mixed-loss pilot",
            "arm": "adversarial",
            "pilot": True,
            "metrics": _species_metrics(pilot["arms"]["adversarial"]),
        },
    ]
    original_pgd, new_pgd = rows[1]["metrics"]["pgd"], rows[2]["metrics"]["pgd"]
    pilot_clean = rows[2]["metrics"]["clean"]
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "config_sha256": config.sha256,
        "classification": "exploratory; not a matched controlled comparison",
        "rows": rows,
        "observed_changes": {
            "clean_accuracy_vs_original_pgd": pilot_clean["accuracy"]
            - rows[1]["metrics"]["clean"]["accuracy"],
            "pgd_accuracy_vs_original_pgd": new_pgd["accuracy"] - original_pgd["accuracy"],
            "pgd_cat_recall_vs_original_pgd": new_pgd["cat_recall"] - original_pgd["cat_recall"],
            "pgd_dog_recall_vs_original_pgd": new_pgd["dog_recall"] - original_pgd["dog_recall"],
            "pilot_predicts_both_species_on_clean_test": all(
                pilot_clean["prediction_counts"].values()
            ),
            "pilot_recognizes_both_species_under_pgd": new_pgd["cat_recall"] > 0
            and new_pgd["dog_recall"] > 0,
        },
        "finite_attack_scope": pilot["finite_attack_scope"],
        "limitations": [
            "The inspected original test outcomes informed this recipe, "
            "so the pilot is exploratory.",
            "The pilot holds validation images out of the original fitting partition; "
            "training counts and update budgets differ.",
            "Warm-up, loss mixing, and class weighting change together; "
            "no individual causal effect is identified.",
            "Mixed batches update BatchNorm on clean and PGD forwards, unlike the "
            "original single-forward arms; this is another uncontrolled difference.",
            "One seed and finite digital PGD attacks do not establish general "
            "or physical robustness.",
            "Original feature/localization panels describe the original checkpoints, "
            "not the new pilot.",
        ],
    }


def _save_figure(config: ExperimentConfig, name: str, figure: Any) -> dict[str, Any]:
    stream = io.BytesIO()
    figure.savefig(
        stream,
        format="png",
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": "Oxford Pets PGD pilot"},
    )
    destination = config.project_path("figures") / f"{name}.png"
    atomic_write_bytes(destination, stream.getvalue())
    return {
        "name": name,
        "path": _relative(config, destination),
        "sha256": sha256_file(destination),
    }


def _figures(
    config: ExperimentConfig, history: list[dict[str, Any]], comparison: dict[str, Any]
) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(item["epoch"]) for item in history]
    warmup = sum(item["pilot_phase"] == "clean_warmup" for item in history)
    figures: list[dict[str, Any]] = []
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axis = plt.subplots(figsize=(9.4, 4.3))
        for key, label, style in (
            ("mean_training_loss", "Optimized weighted objective", "-"),
            ("mean_clean_training_loss", "Clean weighted CE", "--"),
            ("mean_adversarial_training_loss", "PGD weighted CE", ":"),
        ):
            axis.plot(
                epochs,
                [np.nan if item.get(key) is None else item[key] for item in history],
                style,
                marker="o",
                markersize=3,
                label=label,
            )
        axis.axvline(warmup + 0.5, color="grey", linestyle="--", linewidth=1)
        axis.set(
            xlabel="Completed epoch (fixed final checkpoint, no selection)",
            ylabel="Loss",
            title="Clean warm-up → equally mixed clean/PGD objective",
        )
        axis.set_xticks(epochs)
        axis.legend(loc="best", fontsize=9)
        axis.grid(alpha=0.2)
        figures.append(_save_figure(config, "training-loss", figure))
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for axis, condition, title in zip(
            axes,
            ("clean", "pgd"),
            ("Clean held-out training-derived validation", "PGD-20×5 balanced validation subset"),
            strict=True,
        ):
            for species, label, color in (
                (None, "Overall", "#1f77b4"),
                ("0", "Cat recall", "#d95f02"),
                ("1", "Dog recall", "#2a8c54"),
            ):
                values = [
                    float(
                        item["validation"][condition]["accuracy"]
                        if species is None
                        else item["validation"][condition]["per_class_accuracy"][species]
                    )
                    * 100
                    for item in history
                ]
                axis.plot(epochs, values, marker="o", markersize=3, label=label, color=color)
            count = history[-1]["validation"][condition]["count"]
            axis.axvline(warmup + 0.5, color="grey", linestyle="--", linewidth=1)
            axis.set(
                title=f"{title}\nn={count}; monitor only, not checkpoint selection",
                xlabel="Epoch",
                ylim=(-3, 103),
            )
            axis.set_xticks(epochs[::2])
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Accuracy / true-species recall (%)")
        axes[1].legend(loc="best")
        figures.append(_save_figure(config, "validation-recalls", figure))
        plt.close(figure)

        rows = comparison["rows"]
        figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)
        positions = np.arange(len(rows))
        for axis, condition, title in zip(
            axes,
            ("clean", "pgd"),
            ("Official full clean test", "Paired balanced PGD-20×5 test"),
            strict=True,
        ):
            for offset, metric, label, color in (
                (-0.25, "accuracy", "Overall", "#1f77b4"),
                (0.0, "cat_recall", "Cat recall", "#d95f02"),
                (0.25, "dog_recall", "Dog recall", "#2a8c54"),
            ):
                values = [100 * row["metrics"][condition][metric] for row in rows]
                bars = axis.bar(positions + offset, values, width=0.24, label=label, color=color)
                axis.bar_label(
                    bars, labels=[f"{value:.1f}" for value in values], fontsize=8, padding=3
                )
            axis.set_xticks(
                positions, ("Original\nclean", "Original\npure PGD", "Exploratory\npilot")
            )
            axis.set(title=f"{title}\nn={rows[0]['metrics'][condition]['count']}", ylim=(0, 112))
            axis.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel("Accuracy / true-species recall (%)")
        axes[1].legend(loc="lower center", ncol=3, fontsize=9)
        figure.suptitle(
            "Different fitting recipes and budgets: descriptive, not causal comparison", fontsize=11
        )
        figures.append(_save_figure(config, "original-vs-pilot", figure))
        plt.close(figure)

        figure, axes = plt.subplots(2, 3, figsize=(12, 7))
        for column, row in enumerate(rows):
            for line, condition in enumerate(("clean", "pgd")):
                axis = axes[line, column]
                confusion = np.asarray(row["metrics"][condition]["confusion_matrix"])
                normalized = confusion / confusion.sum(axis=1, keepdims=True)
                axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
                for true in range(2):
                    for predicted in range(2):
                        axis.text(
                            predicted,
                            true,
                            f"{confusion[true, predicted]}\n"
                            f"({100 * normalized[true, predicted]:.1f}%)",
                            ha="center",
                            va="center",
                            color="white" if normalized[true, predicted] > 0.55 else "black",
                        )
                axis.set_xticks((0, 1), ("cat", "dog"))
                axis.set_yticks((0, 1), ("cat", "dog"))
                axis.set_xlabel("Predicted species")
                axis.set_ylabel("True species")
                condition_label = "Full clean test" if condition == "clean" else "Balanced PGD test"
                axis.set_title(f"{row['label']}\n{condition_label}", fontsize=9)
        figure.suptitle(
            "Confusion matrices: exact sample counts; percentages within each true species"
        )
        figure.tight_layout()
        figures.append(_save_figure(config, "species-confusion", figure))
        plt.close(figure)
    return figures


def _table(comparison: dict[str, Any]) -> str:
    lines = [
        "| Model | Clean (full test) | Clean cat recall | Clean dog recall | "
        "PGD (balanced subset) | PGD cat recall | PGD dog recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["rows"]:
        clean, pgd = row["metrics"]["clean"], row["metrics"]["pgd"]
        values = (
            clean["accuracy"],
            clean["cat_recall"],
            clean["dog_recall"],
            pgd["accuracy"],
            pgd["cat_recall"],
            pgd["dog_recall"],
        )
        lines.append(
            f"| {row['label']} | " + " | ".join(f"{100 * value:.2f}%" for value in values) + " |"
        )
    return "\n".join(lines)


def _interpretation(comparison: dict[str, Any]) -> str:
    changes = comparison["observed_changes"]
    pilot = comparison["rows"][-1]["metrics"]
    clean = pilot["clean"]
    pgd = pilot["pgd"]
    collapse = (
        "Both species are predicted on the observed clean test."
        if changes["pilot_predicts_both_species_on_clean_test"]
        else "The pilot still predicts only one species on the observed clean test; "
        "it remains a failed classifier comparison."
    )
    robust = (
        "The pilot correctly recognizes at least some examples of both species "
        "under the registered finite PGD attack."
        if changes["pilot_recognizes_both_species_under_pgd"]
        else "At least one species still has zero PGD recall; the aggregate robust score "
        "must not be presented as useful robust cat/dog recognition."
    )
    return (
        f"The fixed final pilot checkpoint has **{100 * clean['accuracy']:.2f}% clean accuracy** "
        f"and **{100 * pgd['accuracy']:.2f}% PGD accuracy**. "
        f"PGD cat/dog recalls are {100 * pgd['cat_recall']:.2f}% / {100 * pgd['dog_recall']:.2f}%. "
        f"Compared descriptively with the failed original PGD arm, balanced PGD accuracy changed "
        f"by {100 * changes['pgd_accuracy_vs_original_pgd']:+.2f} percentage points. "
        f"{collapse} {robust}\n\n"
        "The original PGD arm's 50% balanced score came from recognizing dogs "
        "and missing every cat, "
        "not from useful two-species robustness. Overall scores should always be read alongside "
        "both recalls and prediction counts. The inspected original test results "
        "informed this pilot: its result is exploratory, not a fresh confirmatory finding "
        "or an isolated causal effect of class weighting."
    )


def _report_text(
    config: ExperimentConfig,
    training: dict[str, Any],
    protocol: dict[str, Any],
    comparison: dict[str, Any],
    figures: list[dict[str, Any]],
) -> str:
    epochs = config.integer("training", "epochs")
    history = training["history"]
    warmup = sum(item["pilot_phase"] == "clean_warmup" for item in history)
    final = history[-1]["validation"]
    counts = protocol["counts"]
    weights = protocol["species_weights"]
    parts = [
        "# Exploratory clean-warm-up / mixed-PGD recovery pilot",
        "## Measured result",
        _interpretation(comparison),
        _table(comparison),
        "## What changed, and what did not",
        f"One pre-registered local pilot, fixed epoch {epochs}, no checkpoint selection "
        f"or hyperparameter grid. Epochs 1–{warmup} optimize training-weighted clean "
        f"cross-entropy; the remaining {epochs - warmup} epochs optimize "
        "0.5 × weighted clean CE + 0.5 × weighted PGD CE on the same augmented inputs. "
        "Species weights are computed only from pilot fitting labels as N / (2 × class count). "
        f"Recorded weights: cat={weights['0']:.6f}, dog={weights['1']:.6f}. "
        "The ImageNet-initialized two-class ResNet-18, 224px inputs, float32 MPS, "
        "AdamW learning rates "
        "and weight decay, accumulation, overall LR schedule, and PGD attack settings "
        "are unchanged. "
        "The one-epoch learning-rate warm-up is distinct from the clean-training curriculum.",
        "Attack generation temporarily uses evaluation mode and leaves BatchNorm buffers "
        "unchanged. The actual mixed training objective then performs two training-mode "
        "forwards, clean and PGD, so BatchNorm receives both distributions. The original "
        "arms used one training-mode forward per batch. This changed BatchNorm exposure "
        "is disclosed, not isolated as the cause of any measured improvement.",
        f"The original fit partition had 2,944 images. The pilot instead fits {counts['fit']} and "
        f"holds out {counts['validation']} training-derived validation images "
        f"(breed-stratified filename hashes). The balanced attacked validation subset "
        f"has {counts['validation_attack']} images. Clean calibration (736) is not used "
        "to choose the recipe/checkpoint; the official test partition (3,669) is never fitted. "
        "Final attacks use the exact same 200 test IDs (100 cats/100 dogs) as the original study. "
        "The new held-out subset changes fitting counts and optimizer-update "
        "budget, so this is not a matched reproduction of the original comparison.",
        "## How to read the charts",
        "Training-loss curves show the optimized weighted objective, not test performance. "
        "The clean and PGD loss components are not independently optimized during the mixed phase. "
        "The vertical line separates clean warm-up from mixed training. A lower loss is not "
        "sufficient evidence of "
        "two-species recognition or robustness.",
        "Validation curves use only held-out original training images. Clean overall accuracy "
        "reflects that validation partition's species proportions; "
        "attacked validation is balanced. "
        "Cat/dog recall separates majority-species shortcuts from recognizing both species. "
        "All epochs are retained; "
        "these monitoring curves do not select a best epoch.",
        "The comparison chart places full-clean and balanced-attacked results in separate panels "
        "because their sample distributions differ. Confusion-matrix percentages are normalized "
        "within each true species; each cell also shows its exact count. Blue diagonal mass "
        "indicates "
        "correct decisions, not evidence that a particular CNN filter learned a semantic concept.",
        f"Final validation: clean={100 * final['clean']['accuracy']:.2f}%, "
        f"PGD={100 * final['pgd']['accuracy']:.2f}%; PGD cat/dog recalls="
        f"{100 * final['pgd']['per_class_accuracy']['0']:.2f}% / "
        f"{100 * final['pgd']['per_class_accuracy']['1']:.2f}%.",
    ]
    for figure in figures:
        relative = Path(figure["path"])
        link = Path(
            os.path.relpath(config.root / relative, config.project_path("reports"))
        ).as_posix()
        parts.append(f"![{figure['name'].replace('-', ' ')}]({link})")
    scope = comparison["finite_attack_scope"]
    parts.extend(
        [
            "## Attack scope and limits",
            "Final evaluation: untargeted L∞ FGSM and "
            f"PGD-{scope['pgd_steps']}×{scope['pgd_restarts']}, "
            f"epsilon={scope['epsilon']:.10f} (4/255), step={scope['step_size']:.10f} (1/255), "
            "uniform random starts, pixel clipping, projection, and cumulative restart selection. "
            "Attacks operate on [0,1] pixels before ImageNet normalization. "
            "Check the machine-readable attack-strength result; a finite attack is an estimate, "
            "not a proof of worst-case robustness. No AutoAttack, new dataset, physical attack, "
            "additional architecture, or weaker scoring "
            "protocol was added.",
            "\n".join(f"- {limitation}" for limitation in comparison["limitations"]),
            "## Evidence and reproducibility",
            f"Pilot config SHA-256: `{config.sha256}`. Fixed checkpoint SHA-256: "
            f"`{comparison['rows'][-1]['metrics']['checkpoint_sha256']}`. "
            "`comparison.json` contains aggregate measurements and attack-array identities. "
            "The pilot-only release manifest records data/split/initialization/config/source/"
            "environment identities and generated artifact hashes. "
            "Original checkpoints, charts, reports, and the "
            "original release manifest remain a separate historical study. Its feature/filter "
            "visualizations cannot be relabeled as visualizations of this pilot.",
        ]
    )
    return "\n\n".join(parts) + "\n"


def _notebook(text: str, config: ExperimentConfig, figures: list[dict[str, Any]]) -> dict[str, Any]:
    """Embed aggregate PNGs in Markdown; no executable cells or hidden rerun dependencies."""
    cells: list[dict[str, Any]] = []
    figure_by_link = {item["name"]: item for item in figures}
    for section in text.strip().split("\n\n"):
        attachments: dict[str, Any] = {}
        if section.startswith("!["):
            label = section.split("]", 1)[0][2:]
            item = figure_by_link[label.replace(" ", "-")]
            name = f"{item['name']}.png"
            attachments[name] = {
                "image/png": base64.b64encode((config.root / item["path"]).read_bytes()).decode(
                    "ascii"
                )
            }
            section = f"![{label}](attachment:{name})"
        cell: dict[str, Any] = {
            "cell_type": "markdown",
            "metadata": {},
            "source": section.splitlines(keepends=True),
        }
        if attachments:
            cell["attachments"] = attachments
        cells.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "title": "Exploratory PGD pilot: saved results explained",
            "read_only_results": True,
            "config_sha256": config.sha256,
        },
        "cells": cells,
    }


def report_pilot(config: ExperimentConfig, *, progress: bool = True) -> dict[str, Any]:
    """Build the pilot-only release from completed measured results; never rerun models."""
    original_path = _reference(config)
    _validate_isolation(config, original_path)
    pilot_section = cast(dict[str, Any], config.raw.get("pilot", {}))
    expected_original_hash = str(
        pilot_section.get("original_evaluation_sha256", ORIGINAL_EVALUATION_SHA256)
    )
    if sha256_file(original_path) != expected_original_hash:
        raise PilotReportError("preserved original evaluation identity changed")
    original = _load(original_path)
    expected_original_config = str(
        pilot_section.get("original_config_sha256", ORIGINAL_CONFIG_SHA256)
    )
    if original.get("config_sha256") != expected_original_config:
        raise PilotReportError("original evaluation has the wrong configuration identity")
    evaluation_path = config.project_path("results") / "evaluation.json"
    training_path = config.project_path("artifacts") / "training/adversarial.json"
    protocol_path = config.project_path("artifacts") / "training/validation-split.json"
    evaluation, training, protocol = (
        _load(evaluation_path),
        _load(training_path),
        _load(protocol_path),
    )
    epochs = config.integer("training", "epochs")
    if (
        evaluation.get("config_sha256") != config.sha256
        or training.get("provenance", {}).get("config_sha256") != config.sha256
    ):
        raise PilotReportError("pilot evaluation/training configuration identities differ")
    scope_keys = ("norm", "epsilon", "step_size", "pgd_steps", "pgd_restarts", "samples_per_class")
    if any(
        original["finite_attack_scope"][key] != evaluation["finite_attack_scope"][key]
        for key in scope_keys
    ):
        raise PilotReportError("pilot and original scoring attack scopes differ")
    history = training.get("history", [])
    if (
        training.get("status") != "complete"
        or training.get("completed_epochs") != epochs
        or [item["epoch"] for item in history] != list(range(1, epochs + 1))
    ):
        raise PilotReportError("fixed final-epoch training evidence is incomplete")
    if training.get("pilot_protocol_sha256") != protocol.get("pilot_protocol_sha256"):
        raise PilotReportError("pilot validation protocol identity differs")
    if any(
        item.get("validation", {}).get("protocol_sha256") != protocol["pilot_protocol_sha256"]
        for item in history
    ):
        raise PilotReportError("epoch validation evidence has the wrong protocol identity")
    selected_checkpoint = checkpoint_path(config, "adversarial")
    if evaluation["arms"]["adversarial"]["checkpoint_sha256"] != sha256_file(selected_checkpoint):
        raise PilotReportError(
            "evaluation checkpoint identity differs from the configured final checkpoint"
        )
    status(
        "Pilot report: validating paired evidence and drawing saved-result charts...",
        enabled=progress,
    )
    comparison = _comparison(config, original, evaluation)
    comparison["paired_attack_evidence"] = _paired_evidence(
        config, original_path, comparison["rows"]
    )
    comparison["pilot_protocol_sha256"] = protocol["pilot_protocol_sha256"]
    comparison["fitting_counts"] = protocol["counts"]
    comparison["species_weights"] = protocol["species_weights"]
    comparison["training_updates"] = training["completed_updates"]
    figures = _figures(config, history, comparison)
    comparison["figures"] = figures
    comparison_path = config.project_path("results") / "comparison.json"
    atomic_write_json(comparison_path, comparison)
    text = _report_text(config, training, protocol, comparison, figures)
    report_path = config.project_path("reports") / "pgd-pilot-report.md"
    notebook_path = config.project_path("reports") / "pgd_pilot_results.ipynb"
    atomic_write_text(report_path, text)
    atomic_write_json(notebook_path, _notebook(text, config, figures))
    data_manifest_path = config.project_path("artifacts") / "data/manifest.json"
    initialization_path = config.project_path("artifacts") / "model/initialization.json"
    manifest_paths = [
        config.path,
        evaluation_path,
        training_path,
        protocol_path,
        selected_checkpoint,
        data_manifest_path,
        initialization_path,
        comparison_path,
        report_path,
        notebook_path,
        original_path,
        *(config.root / item["path"] for item in figures),
    ]
    receipt = {
        "schema_version": 1,
        "created_at": utc_now(),
        "scope": "pilot only; original release manifest is not replaced",
        "classification": comparison["classification"],
        "config_sha256": config.sha256,
        "checkpoint_sha256": sha256_file(selected_checkpoint),
        "evaluation_sha256": sha256_file(evaluation_path),
        "original_evaluation_sha256": expected_original_hash,
        "original_config_sha256": expected_original_config,
        "provenance": training["provenance"],
        "pilot_protocol_sha256": protocol["pilot_protocol_sha256"],
        "data_manifest": _load(data_manifest_path),
        "initialization": _load(initialization_path),
        "source_sha256": source_hash(config.root),
        "git": git_state(config.root),
        "environment": environment_snapshot(),
        "files": [
            {
                "path": _relative(config, path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in manifest_paths
        ],
        "paired_attack_evidence": comparison["paired_attack_evidence"],
        "limitations": comparison["limitations"],
    }
    receipt_path = config.project_path("artifacts") / "release/pilot-release-manifest.json"
    atomic_write_json(receipt_path, receipt)
    status(
        "Pilot report complete: separate charts, read-only notebook, and release receipt.",
        enabled=progress,
    )
    return {
        "report": str(report_path),
        "notebook": str(notebook_path),
        "comparison": str(comparison_path),
        "release_manifest": str(receipt_path),
        "figures": figures,
        "observed_changes": comparison["observed_changes"],
    }
