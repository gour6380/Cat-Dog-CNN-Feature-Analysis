"""Read-only, local illustrated results; embeds saved images, never runs a model."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import nbformat

from src.classifier_diagnostics import species_diagnostic_warning, species_prediction_diagnostic
from src.config import ExperimentConfig
from src.io_utils import atomic_write_json, atomic_write_text, sha256_file
from src.training import checkpoint_path


def build_feature_results_notebook(config: ExperimentConfig) -> Path:
    root = config.root
    feature_path = config.project_path("results") / "feature_visualizations.json"
    evaluation_path = config.project_path("results") / "evaluation.json"
    features = json.loads(feature_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    if any(value.get("config_sha256") != config.sha256 for value in (features, evaluation)):
        raise ValueError("illustrated notebook requires current binary evidence")
    for arm in ("standard", "adversarial"):
        expected = sha256_file(checkpoint_path(config, arm))
        if features["arms"][arm]["checkpoint_sha256"] != expected:
            raise ValueError("feature notebook checkpoint hash mismatch")
        if evaluation["arms"][arm]["checkpoint_sha256"] != expected:
            raise ValueError("evaluation notebook checkpoint hash mismatch")
    classifier_diagnostics = {
        arm: species_prediction_diagnostic(evaluation["arms"][arm]["full_test"]["clean"])
        for arm in ("standard", "adversarial")
    }

    def markdown(source: str) -> Any:
        return nbformat.v4.new_markdown_cell(source)  # type: ignore[no-untyped-call]

    cells: list[Any] = [
        markdown(
            "# Cat/Dog CNN features — illustrated measured results\n\n"
            "This is a read-only local snapshot built from saved binary-model evidence. "
            "There are no executable cells and no rerun is needed. It contains licensed pet "
            "photographs and stays Git-ignored pending a separate publication/attribution gate.\n\n"
            "**Question:** what activates each low/mid/high CNN channel, where does it respond, "
            "and which regions affect the true-species prediction?\n\n"
            "Dataset → breed-stratified train/calibration → binary cat/dog ResNet-18 heads → "
            "matched clean/PGD training → fixed checkpoint evaluation → calibration-selected "
            "channels → synthetic patterns + real activating patches → fixed-anchor maps → "
            "Grad-CAM, occlusion and randomized-weight controls."
        )
    ]
    for arm, diagnostic in classifier_diagnostics.items():
        warning = species_diagnostic_warning(diagnostic)
        if warning:
            cells.append(
                markdown(
                    f"## Failed comparison arm: {arm}\n\n{warning}\n\n"
                    f"True counts: {diagnostic['true_counts']}; predicted counts: "
                    f"{diagnostic['prediction_counts']}. Cat recall is "
                    f"{100 * diagnostic['per_species_recall']['cat']:.2f}%, dog recall is "
                    f"{100 * diagnostic['per_species_recall']['dog']:.2f}%. "
                    "Use the standard model for the main learned-feature walkthrough. "
                    "The PGD arm remains visible as a failed fixed-protocol comparison; "
                    "its activation maps cannot rescue its classification result. "
                    "The cause of this failure is not established by these figures."
                )
            )
    rows = ["| Metric | Standard | PGD-trained |", "|---|---:|---:|"]
    attack_count = config.integer("dataset", "attack_per_class") * config.integer(
        "model", "classes"
    )
    pgd_label = (
        f"PGD-{config.integer('attack', 'evaluation_steps')}×"
        f"{config.integer('attack', 'evaluation_restarts')}"
    )
    metric_extractors: list[tuple[str, Callable[[dict[str, Any]], float]]] = [
        ("Full clean test accuracy", lambda a: a["full_test"]["clean"]["accuracy"]),
        (
            f"FGSM robust accuracy · {attack_count} paired images",
            lambda a: a["attack_subset"]["fgsm"]["robust_accuracy"],
        ),
        (
            f"{pgd_label} robust accuracy · {attack_count} paired images",
            lambda a: a["attack_subset"]["pgd"]["robust_accuracy"],
        ),
        ("Clean selective risk", lambda a: a["full_test"]["clean"]["risk"]["selective_risk"]),
    ]
    for title, extractor in metric_extractors:
        numbers = [100 * extractor(evaluation["arms"][arm]) for arm in ("standard", "adversarial")]
        rows.append(f"| {title} | {numbers[0]:.2f}% | {numbers[1]:.2f}% |")
    if all(diagnostic["available"] for diagnostic in classifier_diagnostics.values()):
        for title, extractor in (
            ("Clean macro accuracy", lambda d: d["macro_accuracy"]),
            ("Clean cat recall", lambda d: d["per_species_recall"]["cat"]),
            ("Clean dog recall", lambda d: d["per_species_recall"]["dog"]),
        ):
            values = [
                100 * extractor(classifier_diagnostics[arm]) for arm in ("standard", "adversarial")
            ]
            rows.append(f"| {title} | {values[0]:.2f}% | {values[1]:.2f}% |")
    cells.append(
        markdown(
            "## Measured classification context\n\n"
            + "\n".join(rows)
            + "\n\nThe species distribution is imbalanced; macro/per-species and shift metrics are "
            "in the technical report. Finite digital attack survival is not certified "
            "or physical robustness."
        )
    )
    cells.append(
        markdown(
            "## Three different visual questions\n\n"
            "1. **Pattern preference:** actual early kernels, optimized low/mid/high stimuli, "
            "and strongest real calibration patches.\n"
            "2. **Channel location:** activation maps, theoretical receptive-field boxes, "
            "and input gradients of the selected channel's peak.\n"
            "3. **Prediction influence:** true-species Grad-CAM plus measured margin changes "
            "when equal-area patches are occluded.\n\n"
            "An activation map is not a proof of an eye/ear detector. These methods do not "
            "recover the training patch that originally taught a filter. The high-layer "
            "theoretical receptive field is 435px before clipping, often almost the whole image. "
            "Overlays are independently normalized; raw response charts carry "
            "quantitative comparisons. Grad-CAM targets the true-species logit; "
            "occlusion measures the true-species-minus-other logit margin. "
            "These related but distinct objectives need not agree."
        )
    )
    nonpositive_stimuli = [
        {
            "arm": arm,
            "layer": layer,
            "channel": response["channel"],
            "gain": response["unregularized_response_gain"],
        }
        for arm, arm_result in features["arms"].items()
        for layer, layer_result in arm_result.get("layers", {}).items()
        for response in layer_result.get("synthetic_responses", [])
        if response["unregularized_response_gain"] <= 0
    ]
    if nonpositive_stimuli:
        units = "; ".join(
            f"{unit['arm']} · {unit['layer']} channel {unit['channel']} "
            f"(raw mean gain {unit['gain']:.3f})"
            for unit in nonpositive_stimuli
        )
        cells.append(
            markdown(
                "## Gray synthetic tiles are unsuccessful probes, not dead channels\n\n"
                f"The fixed single-start stimulus optimization did not increase the unjittered "
                f"mean activation for: {units}.\n\n"
                "These channels have positive responses on real calibration inputs. A gray "
                "optimized image therefore does not mean the channel learned nothing or is dead. "
                "Initialization, regularization and optimization affect the synthetic picture. "
                "The original selected units, seeds and unsuccessful tiles are retained; no "
                "more attractive test example or retry was substituted."
            )
        )
    for figure in features["figures"]:
        path = (root / figure["path"]).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != figure["sha256"]:
            raise ValueError("feature figure alignment/hash mismatch")
        name = path.name
        sharing = (
            "Synthetic/aggregate, eligible for review."
            if figure["shareable"]
            else "Photograph-containing/local interpretation panel; not exported."
        )
        source = (
            f"## {figure['arm']} · {figure['kind'].replace('_', ' ')}\n\n"
            f"{figure['caption']}\n\n![{figure['caption']}](attachment:{name})\n\n"
            f"Saved evidence: `{figure['path']}`. "
            f"{sharing}"
        )
        cell = markdown(source)
        cell["attachments"] = {name: {"image/png": base64.b64encode(path.read_bytes()).decode()}}
        cells.append(cell)
    attribution_rows = [
        "| Arm | Anchor | Top-4 CAM tile drop | Random tile drop | CAM/randomized correlation |",
        "|---|---|---:|---:|---:|",
    ]
    for arm, arm_result in features["arms"].items():
        for anchor in arm_result["anchors"]:
            correlation = anchor["randomization_cam_correlation"]
            corr_text = "undefined (constant map)" if correlation is None else f"{correlation:.3f}"
            attribution_rows.append(
                f"| {arm} | {anchor['sample_id']} | "
                f"{anchor['top_gradcam_occlusion_mean_drop']:.3f} | "
                f"{anchor['random_occlusion_mean_drop']:.3f} | {corr_text} |"
            )
    cells.append(
        markdown(
            "## Attribution diagnostic — inspect counterexamples\n\n"
            + "\n".join(attribution_rows)
            + "\n\nPositive drops mean masking reduced the true-class "
            "logit margin. Top CAM tiles need not beat random tiles on every anchor: that is an "
            "important disagreement, not a reason to choose a nicer image. Effects average "
            "individual nonoverlapping equal-area tiles; they are not additive joint-mask effects. "
            "Four hash-selected anchors are descriptive examples, not a population study."
        )
    )
    cells.append(
        markdown(
            "## Limits and next interpretation step\n\n"
            + "\n".join(f"- {limit}" for limit in features["limitations"])
            + "\n\nStart with the real top-patch atlas, then check the matching synthetic tile "
            "and anchor activation map. Use Grad-CAM/occlusion agreement or disagreement to "
            "assess class influence; keep any 'fur-like' or 'edge-like' description tentative."
        )
    )
    for index, cell in enumerate(cells):
        cell["id"] = f"feature-results-{index:03d}"
    notebook = nbformat.v4.new_notebook(cells=cells)  # type: ignore[no-untyped-call]
    notebook["metadata"]["evidence"] = {
        "config_sha256": config.sha256,
        "feature_sha256": sha256_file(feature_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "built_from_saved_evidence": True,
        "contains_pet_photographs": True,
        "clean_classifier_diagnostics": classifier_diagnostics,
        "synthetic_nonpositive_gain_units": nonpositive_stimuli,
    }
    nbformat.validate(notebook)
    destination = config.project_path("reports") / "cat_dog_feature_results.ipynb"
    atomic_write_text(destination, str(nbformat.writes(notebook)))  # type: ignore[no-untyped-call]
    atomic_write_json(
        config.project_path("artifacts") / "features" / "results-notebook.json",
        {
            "path": str(destination.relative_to(root)),
            "sha256": sha256_file(destination),
            "config_sha256": config.sha256,
            "code_cells": 0,
            "figure_count": len(features["figures"]),
            "contains_original_photographs": True,
            "public_export": False,
        },
    )
    return destination
