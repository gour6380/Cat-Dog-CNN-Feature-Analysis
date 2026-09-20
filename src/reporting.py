"""Evidence-backed reports and local release manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from src.classifier_diagnostics import species_diagnostic_warning, species_prediction_diagnostic
from src.config import ExperimentConfig
from src.io_utils import (
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


class ReportError(RuntimeError):
    """Required measured evidence is missing."""


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReportError(f"required measured result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReportError(f"invalid result document: {path}")
    return cast(dict[str, Any], value)


def _percent(value: object) -> str:
    return "n/a" if value is None else f"{100.0 * float(cast(float, value)):.2f}%"


def _number(value: object, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(cast(float, value)):.{digits}f}"


def _condition_table(arm: str, evaluation: dict[str, Any]) -> str:
    rows = [
        "| Condition | Accuracy | Macro accuracy | Cat accuracy | Dog accuracy | "
        "NLL | Brier | Coverage | "
        "Selective risk | ECE | AURC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    conditions = evaluation["arms"][arm]["full_test"]
    for condition, metrics in conditions.items():
        risk = metrics["risk"]
        rows.append(
            "| "
            + " | ".join(
                (
                    condition,
                    _percent(metrics["accuracy"]),
                    _percent(metrics["macro_accuracy"]),
                    _percent(metrics.get("per_class_accuracy", {}).get("0")),
                    _percent(metrics.get("per_class_accuracy", {}).get("1")),
                    _number(risk["nll"]),
                    _number(risk["brier"]),
                    _percent(risk["coverage"]),
                    _percent(risk["selective_risk"]),
                    _number(risk["ece_15"]),
                    _number(risk["aurc"]),
                )
            )
            + " |"
        )
    return "\n".join(rows)


def _attack_label(config: ExperimentConfig) -> str:
    return (
        f"PGD-{config.integer('attack', 'evaluation_steps')}×"
        f"{config.integer('attack', 'evaluation_restarts')}"
    )


def _attack_table(config: ExperimentConfig, evaluation: dict[str, Any]) -> str:
    attack_label = _attack_label(config)
    rows = [
        f"| Arm | Clean subset | FGSM robust | {attack_label} robust | "
        "PGD success among clean-correct |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("standard", "adversarial"):
        attacks = evaluation["arms"][arm]["attack_subset"]
        rows.append(
            f"| {arm} | {_percent(attacks['pgd']['clean_accuracy_on_subset'])} | "
            f"{_percent(attacks['fgsm']['robust_accuracy'])} | "
            f"{_percent(attacks['pgd']['robust_accuracy'])} | "
            f"{_percent(attacks['pgd']['attack_success_clean_correct'])} |"
        )
    return "\n".join(rows)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _feature_table(features: dict[str, Any]) -> str:
    rows = [
        "| Arm | Layer | Spatial grid | Receptive field (px) | Selected channels | "
        "Mean response change vs initialization |",
        "|---|---|---:|---:|---|---:|",
    ]
    for arm in ("standard", "adversarial"):
        for layer, metrics in features["arms"][arm]["layers"].items():
            spatial = "×".join(str(size) for size in metrics["spatial_shape"])
            channels = ", ".join(str(channel) for channel in metrics["selected_channels"])
            change = _mean(metrics["relative_change_from_initial"])
            rows.append(
                f"| {arm} | `{layer}` | {spatial} | "
                f"{metrics['receptive_field_pixels']} | {channels} | {_number(change)} |"
            )
    return "\n".join(rows)


def _localization_table(features: dict[str, Any]) -> str:
    rows = [
        "| Arm | Test anchors | Top-CAM occlusion mean margin drop | "
        "Equal-area random occlusion mean margin drop | Mean randomized-model CAM correlation | "
        "Defined CAM correlations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("standard", "adversarial"):
        anchors = features["arms"][arm]["anchors"]
        values = {
            key: _mean([float(anchor[key]) for anchor in anchors if anchor[key] is not None])
            for key in (
                "top_gradcam_occlusion_mean_drop",
                "random_occlusion_mean_drop",
                "randomization_cam_correlation",
            )
        }
        defined_correlations = sum(
            anchor["randomization_cam_correlation"] is not None for anchor in anchors
        )
        rows.append(
            f"| {arm} | {len(anchors)} | "
            f"{_number(values['top_gradcam_occlusion_mean_drop'])} | "
            f"{_number(values['random_occlusion_mean_drop'])} | "
            f"{_number(values['randomization_cam_correlation'])} | "
            f"{defined_correlations}/{len(anchors)} |"
        )
    return "\n".join(rows)


def _anchor_table(features: dict[str, Any]) -> str:
    rows = [
        "| Arm | Sample | True species | Clean prediction | Clean margin | "
        "PGD prediction | PGD margin |",
        "|---|---|---|---|---:|---|---:|",
    ]
    names = ("cat", "dog")
    for arm in ("standard", "adversarial"):
        for anchor in features["arms"][arm]["anchors"]:
            rows.append(
                f"| {arm} | `{anchor['sample_id']}` | {names[anchor['label']]} | "
                f"{names[anchor['prediction']]} | {_number(anchor['clean_margin'])} | "
                f"{names[anchor['pgd_prediction']]} | {_number(anchor['pgd_margin'])} |"
            )
    return "\n".join(rows)


def _species_diagnostics(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        arm: species_prediction_diagnostic(evaluation["arms"][arm]["full_test"]["clean"])
        for arm in ("standard", "adversarial")
    }


def _species_diagnostic_section(evaluation: dict[str, Any]) -> str:
    """Make observed classifier failure explicit without inferring feature collapse."""

    diagnostics = _species_diagnostics(evaluation)
    available = {
        arm: diagnostic for arm, diagnostic in diagnostics.items() if diagnostic["available"]
    }
    if not available:
        return ""
    warnings = [
        f"> {arm} arm — {warning}"
        for arm, diagnostic in available.items()
        if (warning := species_diagnostic_warning(diagnostic))
    ]
    rows = [
        "| Arm | True cats | True dogs | Predicted cats | Predicted dogs | "
        "Cat recall | Dog recall | Macro accuracy | Clean-test status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm, diagnostic in available.items():
        status = (
            "FAILED comparison: single-species prediction"
            if diagnostic["observed_single_class_prediction"]
            else "Both species predicted; robustness not inferred"
        )
        rows.append(
            f"| {arm} | {diagnostic['true_counts']['cat']} | "
            f"{diagnostic['true_counts']['dog']} | "
            f"{diagnostic['prediction_counts']['cat']} | "
            f"{diagnostic['prediction_counts']['dog']} | "
            f"{_percent(diagnostic['per_species_recall']['cat'])} | "
            f"{_percent(diagnostic['per_species_recall']['dog'])} | "
            f"{_percent(diagnostic['macro_accuracy'])} | {status} |"
        )
    interpretation = (
        "A single-species arm is retained as a failed comparison, not useful robust cat/dog "
        "recognition. Its channel and attribution pictures are failure diagnostics. Hidden "
        "features may remain variable even when all observed clean decisions select one "
        "species; the cause of the classifier failure remains unproven."
        if warnings
        else "Counts describe this observed clean test, not every possible input. Predicting "
        "both species is not by itself evidence of useful robustness or semantic understanding."
    )
    return (
        "## Clean species predictions and failure status\n\n"
        + "\n\n".join(warnings)
        + ("\n\n" if warnings else "")
        + "\n".join(rows)
        + "\n\n"
        + interpretation
        + "\n"
    )


def _synthetic_optimization_note(features: dict[str, Any]) -> str:
    """Disclose unsuccessful fixed-start stimuli without labeling channels dead."""

    trials = 0
    failed: list[tuple[str, str, dict[str, Any], float | None, float | None]] = []
    for arm, arm_result in features["arms"].items():
        for layer, metrics in arm_result["layers"].items():
            for response in metrics.get("synthetic_responses", []):
                trials += 1
                if response["unregularized_response_gain"] != 0.0:
                    continue
                channels = metrics["selected_channels"]
                index = channels.index(response["channel"])
                means = metrics.get("calibration_mean_by_species")
                cat = None if means is None else float(means[0][index])
                dog = None if means is None else float(means[1][index])
                failed.append((arm, layer, response, cat, dog))
    if not failed:
        return ""
    rows = [
        "| Arm | Layer | Channel | Seeded initial response | Final response | Gain | "
        "Real calibration mean: cat | Real calibration mean: dog |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, layer, response, cat, dog in failed:
        rows.append(
            f"| {arm} | `{layer}` | {response['channel']} | "
            f"{_number(response['initial_unregularized_mean_response'])} | "
            f"{_number(response['final_unregularized_mean_response'])} | "
            f"{_number(response['unregularized_response_gain'])} | "
            f"{_number(cat)} | {_number(dog)} |"
        )
    positive_real_response = all(
        cat is not None and dog is not None and max(cat, dog) > 0
        for _arm, _layer, _response, cat, dog in failed
    )
    verification = (
        "These channels respond positively to real calibration images, as shown above. "
        if positive_real_response
        else "Inspect real calibration responses before interpreting these channels. "
    )
    return (
        "### Gray synthetic tiles: zero-gain trials retained\n\n"
        f"{len(failed)} of {trials} recorded single-start synthetic optimization trials had "
        "zero unregularized response gain. Their gray/blank tiles are unsuccessful stimuli, "
        "not evidence of dead channels or that the model learned nothing.\n\n"
        + "\n".join(rows)
        + "\n\n"
        + verification
        + "The fixed seeds, selected channels, and optimization budget were preserved; "
        "no retry or replacement was used to make the atlas look better. Initial/final "
        "responses are measured on unjittered inputs, not the regularized optimization loss.\n"
    )


def _technical_report(
    config: ExperimentConfig, evaluation: dict[str, Any], features: dict[str, Any]
) -> str:
    limitations = "\n".join(f"- {item}" for item in features["limitations"])
    from src.training_monitoring import training_monitoring_enabled

    validation_note = (
        f"The revised protocol additionally reserves "
        f"{100 * config.number('training_monitoring', 'validation_fraction'):g}% "
        f"of registered training within each breed (seed "
        f"`{config.value('training_monitoring', 'validation_seed', str)}`) for validation, leaving "
        "calibration and test unchanged. Clean fitting/validation monitoring is evaluation-only; "
        "it never selects a checkpoint. Original results remain historical, not fresh v3 evidence."
        if training_monitoring_enabled(config)
        else "This protocol has no separate monitoring validation reserve."
    )
    target_coverage = 100.0 * config.number("calibration", "target_coverage")
    return f"""# Technical report: cat and dog CNN feature learning and localization

Fixed evidence snapshot completed on {features["created_at"]}.

{_species_diagnostic_section(evaluation)}

## Question and scope

What patterns activate early, middle, and late CNN channels, and where do the selected
channels and cat/dog classifier respond within actual images? This is a two-class species
study, not a 37-breed projection study. Breed metadata remains available and the official
trainval partition is split within each breed; the official test partition stays untouched.
{validation_note}

Both ResNet-18 arms start from identical ImageNet tensors and an identical new binary head.
The standard arm trains on clean inputs; the adversarial arm trains on bounded PGD inputs.
These are fixed epoch-{config.integer("training", "epochs")} checkpoints, not test-selected
checkpoints.

## Classification and finite attacks

The clean attack-subset accuracy and FGSM/PGD robust accuracies below use the same
{config.integer("dataset", "attack_per_class") * config.integer("dataset", "classes")}
hash-selected test images, balanced by target species. The full official test contains
{config.integer("dataset", "expected_test")} images and its accuracy is reported separately.

{_attack_table(config, evaluation)}

{_attack_label(config)} is finite untargeted `L∞` search with
`epsilon={config.number("attack", "epsilon"):.8g}`. Attack success is measured among
clean-correct samples. The search is not certified, physical, or unrestricted robustness.

## Hierarchical feature diagnostics

{_feature_table(features)}

Channels were selected from clean calibration responses, not test-image appearance.
First-layer kernels show learned weights. Activation-maximization images are synthetic
inputs optimized to excite a fixed channel; they are not recovered training photographs.
Response changes compare the same selected channels on the same calibration images against
the matched ImageNet initialization. They do not identify a named semantic concept.

{_synthetic_optimization_note(features)}

Activation maps locate responses on the transformed input. Highlighted receptive-field
boxes describe theoretical input support for an activation, not the precise pixels that
caused it. Later receptive fields can exceed the entire input image.

## Class localization and controls

{_localization_table(features)}

Grad-CAM differentiates the fixed true-species logit, including on classification errors.
Occlusion measures the true-species-minus-other-species logit margin on the same input:
these are related but different scalar objectives. Equal-area image regions are replaced,
comparing high-CAM tiles with deterministic random tiles. A positive drop means the
intervention reduced the true-species margin. Randomization checks whether CAM changes
when the trained model is
randomized. These are limited descriptive diagnostics, not causal proof of what a filter
learned, not training-source attribution, and not proof that the object is the only cue.
Constant trained or randomized CAMs have undefined correlation; the mean excludes them
and the defined-count column makes that omission explicit. Undefined is not zero.

{_anchor_table(features)}

## Standard arm: full official test and registered shifts

{_condition_table("standard", evaluation)}

## Adversarial arm: full official test and registered shifts

{_condition_table("adversarial", evaluation)}

Scalar temperatures and {target_coverage:g}%-target confidence thresholds use only clean
calibration data; they are frozen for every test, corruption, and attack condition.
Confidence is not an out-of-distribution detector.

## Figures and distribution

Synthetic feature montages, kernel grids, and aggregate charts may be exported.
Actual pet images, feature-map overlays, Grad-CAM, occlusion panels, and receptive-field
crops remain in ignored local analysis outputs. The generated notebook can display them
locally, but they are not silently included in the public repository.

## Limitations

{limitations}

There is no bootstrap representation-retention claim in this revised study. No named
concept, exact causal training source, physical robustness, or safer recognition claim
is made. Configuration SHA-256: `{config.sha256}`. Checkpoint/source/evidence hashes are
recorded in `local-release-manifest.json`.
"""


def _long_form(
    config: ExperimentConfig, evaluation: dict[str, Any], features: dict[str, Any]
) -> str:
    standard = evaluation["arms"]["standard"]
    adversarial = evaluation["arms"]["adversarial"]
    return f"""# Seeing what a cat/dog CNN responds to

{_species_diagnostic_section(evaluation)}

Two ImageNet-initialized ResNet-18 models were fine-tuned for a binary cat-versus-dog task from the
official Oxford-IIIT Pet dataset. Matched initialization, image order, crops, and optimizer
updates isolate clean versus PGD-{config.integer("attack", "train_steps")} training.

The feature story is now visual and layer-by-layer: kernel weights in the first layer,
synthetic activation-maximization stimuli, actual-image activation maps, and receptive-field
crops for selected channels. Channels are selected using calibration responses so the
test images are explanations, not a mechanism for cherry-picking the selection.

{_synthetic_optimization_note(features)}

The distinction is important: a synthesized edge or texture is an input that excites a
channel. It does not prove that the channel learned the named concept we attach to it.
An activation map shows where a response occurs; its receptive-field box is theoretical
support, not an exact causal explanation.

Class-specific Grad-CAM asks a different question: where does the fixed true-species logit
respond? Equal-area top-CAM versus random occlusion measures the true-vs-other margin,
a related but different objective, and randomized-model controls probe weight dependence.
They provide descriptive checks, not a causal account of the training process.

Clean full-test accuracy on {config.integer("dataset", "expected_test")} official test images
is {_percent(standard["full_test"]["clean"]["accuracy"])} for
standard training and {_percent(adversarial["full_test"]["clean"]["accuracy"])} for PGD
training. Finite {_attack_label(config)} robust accuracy on the paired
{config.integer("dataset", "attack_per_class") * config.integer("dataset", "classes")}-image
species-balanced subset is
{_percent(standard["attack_subset"]["pgd"]["robust_accuracy"])} and
{_percent(adversarial["attack_subset"]["pgd"]["robust_accuracy"])}, respectively.

{_localization_table(features)}

The scope remains one dataset, one architecture, and one split/seed family.
ImageNet-pretrained features, synthetic optimization artifacts, backgrounds, finite attack
search, and coarse localization are credible limitations. No physical or safe-use
conclusion follows. Original pet photos remain local; public visuals are synthetic or
aggregate.
"""


def _sunday_draft(
    config: ExperimentConfig, evaluation: dict[str, Any], features: dict[str, Any]
) -> str:
    return f"""# Sunday draft — what a CNN feature picture actually tells us

{_species_diagnostic_section(evaluation)}

I changed this pet-recognition study from breed embedding plots to visual inspection of
a cat/dog CNN: early/middle/late channel stimuli, image activation maps, receptive-field
crops, class-specific Grad-CAM, and occlusion controls.

An optimized feature image answers: what synthetic input excites this channel?
An activation map answers: where did this channel respond on this image?
Grad-CAM answers: where does the fixed true-species logit receive supporting signal?
Occlusion measures changes to its true-vs-other logit margin, not the identical scalar.

None alone identifies the exact training pixels that created the feature, proves a named
semantic concept, or establishes robustness. The same study retains matched clean and
PGD training, finite {_attack_label(config)} evaluation, and frozen clean-calibrated
confidence policies.

{_localization_table(features)}

Scope: one dataset, one architecture, one split/seed family, and bounded digital attacks.
No causal training-source, physical-robustness, or safety claim.

Status: draft only; not approved or published.
"""


def load_current_feature_evidence(config: ExperimentConfig) -> dict[str, Any]:
    """Require complete eight-section, two-model current evidence, never legacy substitutes."""
    from src.data import DataError
    from src.feature_section_types import FEATURE_SECTIONS
    from src.feature_sections import FeatureEvidenceError, validate_feature_evidence
    from src.training import ResumeError
    from src.training_monitoring import MonitoringError

    features = _load(config.project_path("results") / "feature_visualizations.json")
    if features.get("config_sha256") != config.sha256:
        raise ReportError("feature diagnostics belong to another configuration")
    if (
        features.get("status") != "complete"
        or features.get("completion", {}).get("missing")
        or any(
            set(features.get("section_receipts", {}).get(arm, {})) != set(FEATURE_SECTIONS)
            for arm in ("standard", "adversarial")
        )
    ):
        raise ReportError("complete current feature sections are required; fresh results pending")
    try:
        validate_feature_evidence(config, features)
    except (FeatureEvidenceError, DataError, MonitoringError, ResumeError, OSError) as error:
        raise ReportError(f"complete current feature sections are required: {error}") from error
    for arm in ("standard", "adversarial"):
        measured = features.get("arms", {}).get(arm, {})
        if measured.get("state_unchanged") is not True:
            raise ReportError(f"{arm} model state changed during feature analysis")
        path = checkpoint_path(config, cast(Any, arm))
        if not path.is_file() or measured.get("checkpoint_sha256") != sha256_file(path):
            raise ReportError(f"{arm} feature checkpoint identity is invalid")
        if {
            record.get("section")
            for record in features.get("figures", [])
            if record.get("arm") == arm
        } != set(FEATURE_SECTIONS):
            raise ReportError(f"complete registered feature images are required for {arm}")
    for figure in features.get("figures", []):
        path = (config.root / figure["path"]).resolve()
        if (
            not path.is_relative_to(config.root)
            or not path.is_file()
            or sha256_file(path) != figure.get("sha256")
        ):
            raise ReportError("feature image alignment/hash mismatch")
    return features


def _feature_summary(config: ExperimentConfig, features: dict[str, Any]) -> str:
    from src.feature_section_types import FEATURE_SECTIONS

    rows = ["| Model | Verified feature families | Local figures |", "|---|---:|---:|"]
    for arm in ("standard", "adversarial"):
        count = sum(figure.get("arm") == arm for figure in features["figures"])
        rows.append(f"| {arm} | {len(FEATURE_SECTIONS)}/{len(FEATURE_SECTIONS)} | {count} |")
    limits = "\n".join(f"- {item}" for item in features.get("limitations", []))
    table = "\n".join(rows)
    return f"""# Cat/dog CNN — current feature walkthrough

Complete current-run feature evidence for the standard and PGD-trained ResNet-18 models.
Configured training length: {config.integer("training", "epochs")} epochs. PGD is a training
objective here; this report does not measure attack accuracy or certified robustness.

{table}

## What the pictures show

- Input-to-score stages: stem, max-pooling, layer1–layer4, pooled 512 features and cat/dog scores.
- Pattern preference: early kernels, synthetic preferred stimuli and strongest real patches.
- Spatial response: channel activation maps and input gradients.
- Prediction influence: true-species Grad-CAM, equal-area occlusion and randomized-weight controls.

Channels are selected from registered reference/calibration images, not attractive test examples.
Both models' fixed hash-selected cat/dog anchors remain visible. Their scores and predictions
describe those images only: four anchors are not a test-accuracy estimate.
Feature maps are channel responses, not reconstructed photographs or verified anatomical detectors.
These methods cannot recover which training pixels originally taught a filter.

{_synthetic_optimization_note(features)}

## Local illustrated evidence

[Read-only feature notebook](cat_dog_feature_results.ipynb) embeds the registered images.
Photographs, crops and overlays stay ignored local artifacts. Only reviewed kernel,
synthetic-stimulus and aggregate figures are eligible for a separate public export gate.
Saved training history may supply compact loss/validation charts; no missing curve is inferred.

## Limitations

{limits}

Configuration SHA-256: {config.sha256}.
Feature-method SHA-256: {features.get("method_sha256", "not recorded")}.
Status: local measured walkthrough; no publishing, profile edits or applications performed.
"""


def _release_files(config: ExperimentConfig) -> list[Path]:
    """Inventory current feature receipts, not unrelated attack/corruption caches."""
    files = [
        config.root / name
        for name in ("README.md", "pyproject.toml", "requirements.txt", "setup_venv.sh")
    ]
    for directory in ("src", "scripts", "notebooks", "configs"):
        root = config.root / directory
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    features_path = config.project_path("results") / "feature_visualizations.json"
    files.extend((features_path, config.project_path("results") / "summary.json"))
    if features_path.is_file():
        features = _load(features_path)
        files.extend(config.root / figure["path"] for figure in features["figures"])
        visited: set[Path] = set()

        def inventory_receipt(receipt: dict[str, Any]) -> None:
            path = (config.root / receipt["path"]).resolve()
            if path in visited:
                return
            visited.add(path)
            if not path.is_relative_to(config.root) or not path.is_file():
                raise ReportError("registered feature receipt is missing or outside project")
            if sha256_file(path) != receipt["sha256"]:
                raise ReportError("registered feature receipt hash mismatch")
            files.append(path)
            section = _load(path)
            arrays = (config.root / section["arrays_path"]).resolve()
            if (
                not arrays.is_relative_to(config.root)
                or not arrays.is_file()
                or sha256_file(arrays) != section["arrays_sha256"]
            ):
                raise ReportError("registered feature array hash mismatch")
            files.append(arrays)
            for dependency in section.get("dependencies", []):
                inventory_receipt(dependency)

        for registry in features.get("section_receipts", {}).values():
            for receipt in registry.values():
                inventory_receipt(receipt)
    for arm in ("standard", "adversarial"):
        files.extend(
            (
                checkpoint_path(config, cast(Any, arm)),
                config.project_path("artifacts") / "training" / f"{arm}.json",
            )
        )
    reports = config.project_path("reports")
    files.extend(
        reports / name for name in ("technical-report.md", "cat_dog_feature_results.ipynb")
    )
    files.append(config.project_path("artifacts") / "features" / "results-notebook.json")
    from src.runtime_visuals import current_visual_manifests

    for manifest in current_visual_manifests(config, feature_only=True):
        if manifest["group"] == "report" or manifest["group"].startswith("training-"):
            files.append(
                config.project_path("results") / "visualizations" / f"{manifest['group']}.json"
            )
            files.extend(config.root / figure["path"] for figure in manifest["figures"])
    return sorted(
        {
            path.resolve()
            for path in files
            if path.is_file() and path.resolve().is_relative_to(config.root)
        }
    )


def report(config: ExperimentConfig, *, progress: bool = True) -> dict[str, Any]:
    status("Report: checking current CNN feature sections...", enabled=progress)
    features = load_current_feature_evidence(config)
    technical = config.project_path("reports") / "technical-report.md"
    atomic_write_text(technical, _feature_summary(config, features))
    summary = {
        "schema_version": 3,
        "created_at": utc_now(),
        "config_sha256": config.sha256,
        "study": "binary species CNN feature learning and localization",
        "status": "complete current feature walkthrough",
        "feature_evidence_sha256": sha256_file(
            config.project_path("results") / "feature_visualizations.json"
        ),
        "feature_method_sha256": features["method_sha256"],
        "feature_diagnostics": {
            "reference_sample_count": len(features["selection"]["calibration_sample_ids"]),
            "test_anchor_count": len(features["selection"]["test_anchor_ids"]),
            "model_states_unchanged": True,
            "population_accuracy_measured": False,
        },
        "reports": [str(technical.relative_to(config.root))],
        "approval": "not requested",
        "publication": "not performed",
    }
    atomic_write_json(config.project_path("results") / "summary.json", summary)
    from src.runtime_visuals import ensure_report_visuals

    ensure_report_visuals(config)
    from src.feature_results_notebook import build_feature_results_notebook

    notebook = build_feature_results_notebook(config)
    summary["reports"].append(str(notebook.relative_to(config.root)))
    atomic_write_json(config.project_path("results") / "summary.json", summary)
    files = _release_files(config)
    manifest = {
        "schema_version": 2,
        "created_at": utc_now(),
        "status": "local feature evidence bundle; no public release",
        "config_sha256": config.sha256,
        "feature_method_sha256": features["method_sha256"],
        "source_sha256": source_hash(config.root),
        "provenance": features.get("provenance", {}),
        "environment": environment_snapshot(),
        "git": git_state(config.root),
        "fixed_checkpoints": {
            arm: {
                "path": str(checkpoint_path(config, cast(Any, arm)).relative_to(config.root)),
                "sha256": sha256_file(checkpoint_path(config, cast(Any, arm))),
            }
            for arm in ("standard", "adversarial")
        },
        "files": {str(path.relative_to(config.root)): sha256_file(path) for path in files},
        "contains_original_photographs": any(
            not figure["shareable"] for figure in features["figures"]
        ),
        "public_actions": [],
    }
    manifest_path = config.project_path("artifacts") / "release" / "local-release-manifest.json"
    atomic_write_json(manifest_path, manifest)
    status(
        "Report complete: current feature summary and illustrated notebook saved locally.",
        enabled=progress,
    )
    return {**summary, "local_release_manifest": str(manifest_path)}
