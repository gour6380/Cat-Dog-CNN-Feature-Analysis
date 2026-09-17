"""Export reviewed cat/dog metrics and photograph-free feature figures.

This presentation-only command does not load models or perform training, inference,
attacks, or downloads. Missing/currently superseded evidence produces a pending page.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classifier_diagnostics import (  # noqa: E402
    species_diagnostic_warning,
    species_prediction_diagnostic,
)
from src.config import ExperimentConfig, load_config  # noqa: E402
from src.io_utils import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402
from src.reporting import (  # noqa: E402
    _attack_table,
    _condition_table,
    _feature_table,
    _localization_table,
)

SAFE_FIGURE_KINDS = {
    "kernels",
    "activation_maximization",
    "species_response",
    "initial_response_change",
    "localization_summary",
}
ANCHOR_FIELDS = (
    "sample_id",
    "label",
    "prediction",
    "clean_margin",
    "pgd_prediction",
    "pgd_margin",
    "top_gradcam_occlusion_mean_drop",
    "random_occlusion_mean_drop",
    "randomization_cam_correlation",
)
LIMITATIONS = [
    "One ImageNet-initialized ResNet-18 pair and one split/seed family limit scope.",
    "Synthetic channel stimuli are optimized inputs, not recovered training images.",
    "Channel responses do not prove named concepts or identify their causal training source.",
    "Receptive-field boxes describe theoretical support, not exact contributing pixels.",
    "Grad-CAM, occlusion, and randomization are descriptive localization diagnostics.",
    "Finite digital attacks and synthetic corruptions do not establish physical or safe use.",
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path.name}")
    return cast(dict[str, Any], value)


def _validate_configuration(config_sha256: str, documents: list[dict[str, Any]]) -> None:
    observed = {document.get("config_sha256") for document in documents}
    if observed != {config_sha256}:
        raise RuntimeError("saved evidence has mixed configuration identities")


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
        "attack_subset_count": (
            config.integer("dataset", "attack_per_class") * config.integer("dataset", "classes")
        ),
        "attack": {
            "targeted": False,
            "norm": config.value("attack", "norm", str),
            "epsilon": config.number("attack", "epsilon"),
            "training_steps": config.integer("attack", "train_steps"),
            "evaluation_steps": config.integer("attack", "evaluation_steps"),
            "evaluation_restarts": config.integer("attack", "evaluation_restarts"),
        },
        "train_calibration_stratification": "breed",
        "fixed_subset_stratification": "target_class",
    }


def _write_pending(config: ExperimentConfig) -> tuple[Path, Path]:
    markdown = config.root / "docs/results.md"
    aggregate = config.root / "docs/results.json"
    atomic_write_json(
        aggregate,
        {
            "schema_version": 2,
            "project": config.value("experiment", "title", str),
            "status": "pending binary feature-study execution",
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
        """# Results: cat/dog CNN feature learning and localization

Status: pending execution of the revised binary study.

The task is cat=0 versus dog=1. The earlier 37-breed results do not establish the
performance or learned channels of a newly trained binary classifier. They are not
reported as results of this study.

The new evidence will include early/middle/late channel visualizations, synthetic
activation-maximization images, actual-image feature maps, receptive-field crops,
class-specific Grad-CAM, and equal-area occlusion/randomization controls. Actual pet
photographs and overlays stay local; only synthetic/kernel/aggregate charts are exported.

These are descriptive diagnostics, not proof of named concepts, causal training-source
attribution, physical or safe use.

[Configuration](../configs/experiment.yaml), [protocol](PROTOCOL.md),
[guided notebook](../notebooks/cat_dog_cnn_features.ipynb),
and [machine-readable status](results.json) are included.
""",
    )
    return markdown, aggregate


def _shareable_figures(config: ExperimentConfig, features: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for figure in features["figures"]:
        if figure.get("shareable") is not True:
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
        ):
            raise RuntimeError("shareable figure must be a local generated PNG")
        selected.append({**figure, "source": source})
    return selected


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _synthetic_response_summary(layer: dict[str, Any]) -> dict[str, Any]:
    """Report the recorded single-start outcome, not whether a channel is alive."""

    recorded = layer.get("synthetic_responses")
    if not isinstance(recorded, list) or not recorded:
        return {"available": False}
    selected = layer["selected_channels"]
    calibration = layer["calibration_mean_by_species"]
    responses: list[dict[str, Any]] = []
    for record in recorded:
        if not isinstance(record, dict) or not all(
            key in record
            for key in (
                "channel",
                "initial_unregularized_mean_response",
                "final_unregularized_mean_response",
            )
        ):
            continue
        initial = record["initial_unregularized_mean_response"]
        final = record["final_unregularized_mean_response"]
        if initial == 0 and final == 0:
            status = "unsuccessful_zero_response_single_start"
        elif final > initial:
            status = "measured_response_increased"
        else:
            status = "no_measured_response_increase_single_start"
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
        if record["channel"] in selected:
            index = selected.index(record["channel"])
            response["calibration_mean_response_cat"] = calibration[0][index]
            response["calibration_mean_response_dog"] = calibration[1][index]
        responses.append(response)
    if not responses:
        return {"available": False}
    return {
        "available": True,
        "responses": responses,
        "zero_response_single_start_count": sum(
            item["single_start_status"] == "unsuccessful_zero_response_single_start"
            for item in responses
        ),
        "interpretation": "Single recorded optimization start, not a dead-channel test. "
        "Initial/final raw responses are distinct from the regularized jittered loss.",
    }


def _public_feature_summary(arm: dict[str, Any]) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for name, layer in arm["layers"].items():
        responses = layer["calibration_mean_by_species"]
        layers[name] = {
            "selected_channels": ", ".join(str(item) for item in layer["selected_channels"]),
            "selected_channel_count": len(layer["selected_channels"]),
            "spatial_grid": "×".join(str(item) for item in layer["spatial_shape"]),
            "receptive_field_pixels": layer["receptive_field_pixels"],
            "calibration_mean_response_cat": _mean(responses[0]),
            "calibration_mean_response_dog": _mean(responses[1]),
            "mean_relative_response_change_from_initial": _mean(
                layer["relative_change_from_initial"]
            ),
            "activation_maximization": _synthetic_response_summary(layer),
        }
    anchors = [{key: anchor[key] for key in ANCHOR_FIELDS} for anchor in arm["anchors"]]
    for anchor in anchors:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", anchor["sample_id"]):
            raise RuntimeError("anchor identifiers must not contain paths")
    return {
        "layers": layers,
        "anchors": anchors,
        "state_unchanged": arm["state_unchanged"],
    }


def _arm_summary(
    arm: dict[str, Any], features: dict[str, Any], training: dict[str, Any]
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition, metrics in arm["full_test"].items():
        conditions[condition] = {
            "accuracy": metrics["accuracy"],
            "macro_accuracy": metrics["macro_accuracy"],
            "per_class_accuracy": metrics.get("per_class_accuracy", {}),
            "risk": {
                key: metrics["risk"][key]
                for key in ("nll", "brier", "ece_15", "coverage", "selective_risk", "aurc")
            },
        }
    attacks = {
        name: {
            key: metrics[key]
            for key in (
                "clean_accuracy_on_subset",
                "robust_accuracy",
                "attack_success_clean_correct",
                "clean_correct_count",
                "attack_success_count",
                "per_class",
            )
            if key in metrics
        }
        for name, metrics in arm["attack_subset"].items()
    }
    return {
        "training": {
            "epochs": training["completed_epochs"],
            "optimizer_updates": training["completed_updates"],
            "elapsed_seconds": training["elapsed_this_invocation_seconds"],
        },
        "full_test": conditions,
        "clean_prediction_diagnostic": species_prediction_diagnostic(arm["full_test"]["clean"]),
        "attack_subset": attacks,
        "calibration_policy": {
            key: arm["policy"][key]
            for key in (
                "temperature",
                "threshold",
                "target_coverage",
                "fitted_count",
                "fitted_coverage",
                "threshold_tie_count",
            )
        },
        "calibration_nll_before": arm["calibration_nll_before"],
        "calibration_nll_after": arm["calibration_nll_after"],
        "feature_diagnostics": _public_feature_summary(features),
    }


def _prediction_warning(arm: str, diagnostic: dict[str, Any]) -> str:
    """Keep a failed decision rule prominent beside nominal headline accuracies."""

    warning = species_diagnostic_warning(diagnostic)
    if not warning:
        return ""
    recalls = diagnostic["per_species_recall"]
    species = diagnostic["single_predicted_species_on_clean_test"]
    baseline = diagnostic["constant_prediction_baseline_accuracy"]
    return (
        f"> **Classifier failure — {arm}.** {warning} Cat recall is "
        f"**{recalls['cat']:.2%}**, dog recall is **{recalls['dog']:.2%}**. "
        f"The nominal clean accuracy equals the **{baseline:.2%} constant-{species} baseline** "
        "on this species-imbalanced test partition. Balanced attack-set survival does not "
        "establish useful binary recognition."
    )


def _activation_maximization_note(arm: str, feature_summary: dict[str, Any]) -> str:
    zero_records: list[dict[str, Any]] = []
    channel_groups: list[str] = []
    for layer_name, layer in feature_summary["layers"].items():
        records = layer["activation_maximization"].get("responses", [])
        failed = [
            item
            for item in records
            if item["single_start_status"] == "unsuccessful_zero_response_single_start"
        ]
        if not failed:
            continue
        zero_records.extend(failed)
        channels = ", ".join(str(item["channel"]) for item in failed)
        channel_groups.append(f"{layer_name} channel(s) {channels}")
    if not zero_records:
        return ""
    positive_calibration = all(
        item.get("calibration_mean_response_cat", 0) > 0
        or item.get("calibration_mean_response_dog", 0) > 0
        for item in zero_records
    )
    real_response_note = (
        " The same channels have positive measured responses on real clean calibration inputs."
        if positive_calibration
        else ""
    )
    return (
        f"**Unsuccessful single-start activation maximization — {arm}:** "
        + "; ".join(channel_groups)
        + ". Their recorded unregularized response was exactly 0 → 0; gray tiles are "
        "unsuccessful optimized stimuli from this one start, **not evidence of dead channels "
        "or that the model learned nothing**."
        + real_response_note
        + " Recorded channel choices and protocol are preserved: no retry, reselection, "
        "or retraining was used to replace these tiles. Per-channel responses/statuses are "
        "included in the aggregate JSON."
    )


def build(root: Path = ROOT) -> tuple[Path, Path]:
    config = load_config(root / "configs/experiment.yaml")
    results = config.project_path("results")
    evaluation_path = results / "evaluation.json"
    features_path = results / "feature_visualizations.json"
    summary_path = results / "summary.json"
    artifacts = config.project_path("artifacts")
    data_path = artifacts / "data/manifest.json"
    preflight_path = artifacts / "preflight.json"
    release_path = artifacts / "release/local-release-manifest.json"
    training_paths = {
        arm: artifacts / "training" / f"{arm}.json" for arm in ("standard", "adversarial")
    }
    reports = {
        config.project_path("reports") / "technical-report.md": root
        / "reports/technical_report.md",
        config.project_path("reports") / "robust-vision-long-form-report.md": (
            root / "reports/long_form_report.md"
        ),
    }
    required = [
        evaluation_path,
        features_path,
        summary_path,
        data_path,
        preflight_path,
        release_path,
        *training_paths.values(),
        *reports,
    ]
    if not all(path.is_file() for path in required):
        return _write_pending(config)
    evaluation = _read_json(evaluation_path)
    features = _read_json(features_path)
    summary = _read_json(summary_path)
    data = _read_json(data_path)
    preflight = _read_json(preflight_path)
    release = _read_json(release_path)
    training = {arm: _read_json(path) for arm, path in training_paths.items()}
    documents = [
        evaluation,
        features,
        summary,
        preflight,
        release,
        *(item["provenance"] for item in training.values()),
    ]
    if any(document.get("config_sha256") != config.sha256 for document in documents):
        return _write_pending(config)
    _validate_configuration(config.sha256, documents)
    epochs = config.integer("training", "epochs")
    if any(
        item.get("status") != "complete" or item.get("completed_epochs") != epochs
        for item in training.values()
    ):
        raise RuntimeError("training evidence is incomplete or has a different epoch count")
    if len({item["completed_updates"] for item in training.values()}) != 1:
        raise RuntimeError("training arms have unequal optimizer-update counts")
    figures = _shareable_figures(config, features)
    release_files = release.get("files")
    if not isinstance(release_files, dict):
        raise RuntimeError("release file manifest is invalid")
    for path in [
        evaluation_path,
        features_path,
        summary_path,
        *training_paths.values(),
        *(figure["source"] for figure in figures),
        *reports,
    ]:
        if release_files.get(str(path.relative_to(root))) != sha256_file(path):
            raise RuntimeError(f"saved evidence hash mismatch: {path.name}")
    for arm in ("standard", "adversarial"):
        expected_hash = release["fixed_checkpoints"][arm]["sha256"]
        if (
            features["arms"][arm].get("state_unchanged") is not True
            or features["arms"][arm]["checkpoint_sha256"] != expected_hash
            or evaluation["arms"][arm]["checkpoint_sha256"] != expected_hash
        ):
            raise RuntimeError(
                "feature/evaluation checkpoint identity or unchanged-state check failed"
            )
    docs_assets = root / "docs/assets"
    docs_assets.mkdir(parents=True, exist_ok=True)
    asset_hashes: dict[str, str] = {}
    public_figures: list[dict[str, Any]] = []
    for figure in figures:
        stem = re.sub(r"[^A-Za-z0-9_-]", "-", figure["source"].stem)
        name = f"{figure['arm']}-{figure['kind']}-{stem}.png"
        destination = docs_assets / name
        shutil.copyfile(figure["source"], destination)
        relative = str(destination.relative_to(root))
        asset_hashes[relative] = sha256_file(destination)
        public_figures.append(
            {
                "path": relative,
                "caption": figure["caption"],
                "kind": figure["kind"],
                "arm": figure["arm"],
            }
        )
    report_hashes: dict[str, str] = {}
    for source, destination in reports.items():
        atomic_write_text(destination, source.read_text(encoding="utf-8"))
        report_hashes[str(destination.relative_to(root))] = sha256_file(destination)
    public = {
        "schema_version": 2,
        "project": config.value("experiment", "title", str),
        "status": f"completed {epochs}-epoch binary feature-study reference experiment",
        "observed_at": features["created_at"],
        "protocol": {
            **_protocol(config),
            "device": evaluation["device"],
            "dataset_source": data["source"],
            "dataset_license": data["license"],
            "split": {
                "training": data["training_count"],
                "calibration": data["calibration_count"],
                "official_test": data["integrity"]["test_count"],
            },
        },
        "arms": {
            arm: _arm_summary(evaluation["arms"][arm], features["arms"][arm], training[arm])
            for arm in ("standard", "adversarial")
        },
        "secondary_policy_shift": evaluation["secondary_policy_shift"],
        "figures": public_figures,
        "provenance": {
            "configuration_sha256": config.sha256,
            "dataset_sha256": data["dataset_content_sha256"],
            "split_sha256": data["split_sha256"],
            "initialization_sha256": release["provenance"]["initialization_sha256"],
            "preflight_source_sha256": preflight["current_source_sha256"],
            "feature_method_source_sha256": features.get("method_sha256"),
            "pre_export_source_sha256": release["source_sha256"],
            "pre_export_git_commit": release["git"]["commit"],
            "pre_export_worktree_dirty": (
                bool(release["git"]["status_porcelain"])
                if release["git"].get("status_porcelain") is not None
                else None
            ),
            "evaluation_sha256": sha256_file(evaluation_path),
            "feature_visualizations_sha256": sha256_file(features_path),
            "checkpoints": {
                arm: {"sha256": item["sha256"]}
                for arm, item in release["fixed_checkpoints"].items()
            },
            "public_assets": asset_hashes,
            "public_reports": report_hashes,
        },
        "limitations": list(dict.fromkeys([*LIMITATIONS, *features["limitations"]])),
    }
    serialized = json.dumps(public)
    if any(marker in serialized for marker in ("/Users/", "/private/", "/tmp/", "image_path")):
        raise RuntimeError("public aggregate contains a private path")
    aggregate = root / "docs/results.json"
    markdown = root / "docs/results.md"
    atomic_write_json(aggregate, public)
    figure_markdown = "\n\n".join(
        f"![{figure['arm']} {figure['kind'].replace('_', ' ')}]"
        f"({Path(figure['path']).relative_to('docs')})\n\n"
        f"**Figure caption.** {figure['caption']}"
        + (
            "\n\n"
            + _activation_maximization_note(
                figure["arm"], public["arms"][figure["arm"]]["feature_diagnostics"]
            )
            if figure["kind"] == "activation_maximization" and figure["arm"] in public["arms"]
            else ""
        )
        for figure in public_figures
    )
    limitations = "\n".join(f"- {item}" for item in public["limitations"])
    prediction_warnings = "\n\n".join(
        warning
        for arm, arm_result in public["arms"].items()
        if (warning := _prediction_warning(arm, arm_result["clean_prediction_diagnostic"]))
    )
    text = f"""# Results: cat/dog CNN feature learning and localization

Matched standard and adversarial ResNet-18 arms completed {epochs} epochs for cat=0
versus dog=1. Both begin from identical ImageNet tensors and a new binary classifier.
This revised study is descriptive feature/localization analysis, not the old breed
geometry/retention experiment.

{prediction_warnings}

## Classification and finite attacks

The attack table uses the same
{config.integer("dataset", "attack_per_class") * config.integer("dataset", "classes")}-image
species-balanced test subset for clean/FGSM/PGD accuracy. Full clean/corruption accuracy
below uses {config.integer("dataset", "expected_test")} official test images. These
denominators must not be interchanged.

{_attack_table(config, evaluation)}

## Selected early/middle/late channels

{_feature_table(features)}

{figure_markdown}

## Class-localization controls

{_localization_table(features)}

Grad-CAM targets the fixed true-species logit, including classification errors.
Positive occlusion drops mean the true-species-minus-other-species logit margin decreased:
a related but different scalar objective. Equal-area random occlusion and randomized-model
CAM checks are limited controls, not causal proof. Undefined constant-map correlations
remain null in JSON and are excluded from the mean with a displayed defined count.
Channel stimuli are synthetic optimized inputs, not recovered training photographs.
Receptive-field boxes describe theoretical support, not exact contributing pixels.

## Standard arm: full-test performance and clean-fitted confidence

{_condition_table("standard", evaluation)}

## Adversarial arm: full-test performance and clean-fitted confidence

{_condition_table("adversarial", evaluation)}

Temperatures and confidence thresholds are fitted only on clean calibration data and
frozen for all shifts. Confidence is not an out-of-distribution detector.

## Limitations

{limitations}

## Evidence and reproduction

[Aggregate JSON and hashes](results.json), [configuration](../configs/experiment.yaml),
[protocol](PROTOCOL.md),
[guided notebook](../notebooks/cat_dog_cnn_features.ipynb),
[technical report](../reports/technical_report.md), and
[long-form interpretation](../reports/long_form_report.md) are included.
Photographs, activation overlays, receptive-field crops, checkpoints, and per-sample
arrays stay local. Reading this page performs no training or inference.

Tracked provenance records the pre-export source/worktree base, not a self-referential
final commit hash. After presentation files are reviewed and locally committed, the ignored
`artifacts/release/local-release-manifest.json` records the actual clean candidate commit
and complete active-file hashes. No remote or publication is implied.
"""
    atomic_write_text(markdown, text)
    return markdown, aggregate


if __name__ == "__main__":
    markdown_path, aggregate_path = build()
    print(markdown_path)
    print(aggregate_path)
