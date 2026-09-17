from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scripts.build_public_results import (
    _public_feature_summary,
    _shareable_figures,
    build,
)
from src.config import ExperimentConfig, load_config
from src.io_utils import atomic_write_json, atomic_write_text, sha256_file

ROOT = Path(__file__).parents[1]
PRIVATE_FIXTURE_ROOT = "/" + "Users/" + "private/"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _config(tmp_path: Path) -> ExperimentConfig:
    destination = tmp_path / "configs/experiment.yaml"
    atomic_write_text(destination, (ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    return load_config(destination)


def _bundle(tmp_path: Path) -> ExperimentConfig:
    config = _config(tmp_path)
    risk = {
        "nll": 0.3,
        "brier": 0.2,
        "ece_15": 0.05,
        "coverage": 0.9,
        "selective_risk": 0.1,
        "aurc": 0.03,
    }
    metrics = {"accuracy": 0.9, "macro_accuracy": 0.9, "risk": risk}
    attacks = {
        "clean_accuracy_on_subset": 0.9,
        "robust_accuracy": 0.5,
        "attack_success_clean_correct": 0.4,
    }
    policy = {
        "temperature": 1.5,
        "threshold": 0.6,
        "target_coverage": 0.9,
        "fitted_count": 20,
        "fitted_coverage": 0.9,
        "threshold_tie_count": 1,
    }
    evaluation = {
        "created_at": "fixed-evidence-time",
        "config_sha256": config.sha256,
        "device": "mps",
        "arms": {
            arm: {
                "checkpoint": PRIVATE_FIXTURE_ROOT + "checkpoint.pt",
                "checkpoint_sha256": arm + "-checkpoint",
                "full_test": {"clean": metrics, "brightness_0.6": metrics},
                "attack_subset": {"fgsm": attacks, "pgd": attacks},
                "policy": policy,
                "calibration_nll_before": 0.4,
                "calibration_nll_after": 0.3,
            }
            for arm in ("standard", "adversarial")
        },
        "secondary_policy_shift": {},
    }
    anchor = {
        "sample_id": "pet_001",
        "label": 0,
        "prediction": 0,
        "clean_margin": 1.2,
        "pgd_prediction": 1,
        "pgd_margin": -0.3,
        "top_gradcam_occlusion_mean_drop": 0.4,
        "random_occlusion_mean_drop": 0.1,
        "randomization_cam_correlation": 0.2,
        "image_path": PRIVATE_FIXTURE_ROOT + "pet.jpg",
        "pixels": [[1, 2, 3]],
    }
    layer = {
        "selected_channels": [2, 4],
        "calibration_mean_by_species": [[0.4, 0.6], [0.2, 0.8]],
        "relative_change_from_initial": [0.1, 0.3],
        "spatial_shape": [14, 14],
        "receptive_field_pixels": 211,
    }
    figure_dir = config.project_path("figures") / "features"
    figure_dir.mkdir(parents=True)
    synthetic = figure_dir / "synthetic.png"
    photograph = figure_dir / "actual-pet-overlay.png"
    Image.new("RGB", (8, 8), (80, 90, 100)).save(synthetic)
    Image.new("RGB", (8, 8), (140, 150, 160)).save(photograph)
    features = {
        "config_sha256": config.sha256,
        "created_at": "fixed-evidence-time",
        "device": "mps",
        "selection": {"calibration_sample_ids": ["cal_1"], "test_anchor_ids": ["pet_001"]},
        "arms": {
            arm: {
                "checkpoint_sha256": arm + "-checkpoint",
                "layers": {"network.layer2": layer},
                "anchors": [anchor],
                "state_unchanged": True,
            }
            for arm in ("standard", "adversarial")
        },
        "figures": [
            {
                "path": str(synthetic.relative_to(tmp_path)),
                "caption": "Synthetic channel inputs",
                "shareable": True,
                "arm": "standard",
                "kind": "activation_maximization",
            },
            {
                "path": str(photograph.relative_to(tmp_path)),
                "caption": "Local pet overlay",
                "shareable": False,
                "arm": "standard",
                "kind": "gradcam",
            },
        ],
        "limitations": ["Synthetic inputs do not prove semantic concepts."],
    }
    results = config.project_path("results")
    artifacts = config.project_path("artifacts")
    docs: dict[Path, dict[str, Any]] = {
        results / "evaluation.json": evaluation,
        results / "feature_visualizations.json": features,
        results / "summary.json": {"config_sha256": config.sha256},
        artifacts / "data/manifest.json": {
            "source": "Official Oxford-IIIT Pet",
            "license": "CC BY-SA 4.0",
            "training_count": 32,
            "calibration_count": 8,
            "integrity": {"test_count": 40},
            "dataset_content_sha256": "dataset-hash",
            "split_sha256": "split-hash",
        },
        artifacts / "preflight.json": {
            "config_sha256": config.sha256,
            "current_source_sha256": "execution-source",
        },
    }
    for arm in ("standard", "adversarial"):
        docs[artifacts / "training" / f"{arm}.json"] = {
            "status": "complete",
            "completed_epochs": config.integer("training", "epochs"),
            "completed_updates": 10,
            "elapsed_this_invocation_seconds": 120,
            "provenance": {"config_sha256": config.sha256},
        }
    for path, value in docs.items():
        atomic_write_json(path, value)
    report_paths = [
        config.project_path("reports") / "technical-report.md",
        config.project_path("reports") / "robust-vision-long-form-report.md",
    ]
    for path in report_paths:
        atomic_write_text(path, "# Descriptive feature report\n")
    release = {
        "config_sha256": config.sha256,
        "source_sha256": "release-source",
        "git": {"commit": "local-commit"},
        "provenance": {"initialization_sha256": "initial"},
        "fixed_checkpoints": {
            arm: {"sha256": arm + "-checkpoint"} for arm in ("standard", "adversarial")
        },
        "files": {
            str(path.relative_to(tmp_path)): sha256_file(path)
            for path in [*docs, *report_paths, synthetic, photograph]
        },
    }
    atomic_write_json(artifacts / "release/local-release-manifest.json", release)
    return config


def test_missing_evidence_creates_an_honest_pending_page(tmp_path: Path) -> None:
    config = _config(tmp_path)
    markdown, aggregate = build(tmp_path)
    public = _read(aggregate)
    assert public["status"] == "pending binary feature-study execution"
    assert public["provenance"]["configuration_sha256"] == config.sha256
    assert public["protocol"]["classes"] == 2
    assert public["arms"] == {}
    assert "earlier 37-breed results do not establish" in markdown.read_text(encoding="utf-8")
    assert "primary_hypothesis" not in public


def test_public_export_uses_only_safe_figures_and_whitelisted_aggregates(tmp_path: Path) -> None:
    config = _bundle(tmp_path)
    markdown, aggregate = build(tmp_path)
    public = _read(aggregate)
    text = markdown.read_text(encoding="utf-8")
    assert public["provenance"]["configuration_sha256"] == config.sha256
    assert public["protocol"]["attack_subset_count"] == 200
    assert "primary_hypothesis" not in public
    assert "geometry" not in public
    assert len(public["figures"]) == 1
    assert len(list((tmp_path / "docs/assets").glob("*.png"))) == 1
    assert not any("actual-pet" in str(path) for path in (tmp_path / "docs/assets").glob("*"))
    serialized = json.dumps(public)
    assert "/Users/" not in serialized
    assert "image_path" not in serialized
    assert '"pixels":' not in serialized
    assert "0.4000" in text
    assert "NLL | Brier" in text
    assert "physical or safe use" in text
    assert "fixed true-species logit" in text
    assert "200-image" in text and "3669 official test images" in text
    assert public["provenance"]["pre_export_git_commit"] == "local-commit"
    assert public["provenance"]["pre_export_worktree_dirty"] is None
    for arm in ("standard", "adversarial"):
        actual = public["arms"][arm]
        assert actual["full_test"]["clean"]["accuracy"] == 0.9
        assert actual["clean_prediction_diagnostic"] == {"available": False}
        assert actual["attack_subset"]["pgd"]["robust_accuracy"] == 0.5
        assert actual["calibration_policy"]["temperature"] == 1.5
        assert (
            actual["feature_diagnostics"]["layers"]["network.layer2"][
                "calibration_mean_response_cat"
            ]
            == 0.5
        )
    for relative, expected in public["provenance"]["public_assets"].items():
        assert sha256_file(tmp_path / relative) == expected
    for figure in public["figures"]:
        assert f"**Figure caption.** {figure['caption']}" in text


def test_public_export_prominently_discloses_all_dog_classifier_failure(tmp_path: Path) -> None:
    config = _bundle(tmp_path)
    path = config.project_path("results") / "evaluation.json"
    evaluation = _read(path)
    clean = evaluation["arms"]["adversarial"]["full_test"]["clean"]
    clean["confusion_matrix"] = [[0, 1183], [0, 2486]]
    clean["accuracy"] = 2486 / 3669
    clean["macro_accuracy"] = 0.5
    clean["per_class_accuracy"] = {"0": 0.0, "1": 1.0}
    atomic_write_json(path, evaluation)
    release_path = config.project_path("artifacts") / "release/local-release-manifest.json"
    release = _read(release_path)
    release["files"][str(path.relative_to(tmp_path))] = sha256_file(path)
    atomic_write_json(release_path, release)

    markdown, aggregate = build(tmp_path)
    public = _read(aggregate)
    diagnostic = public["arms"]["adversarial"]["clean_prediction_diagnostic"]
    assert diagnostic["available"] is True
    assert diagnostic["observed_single_class_prediction"] is True
    assert diagnostic["single_predicted_species_on_clean_test"] == "dog"
    assert diagnostic["prediction_counts"] == {"cat": 0, "dog": 3669}
    assert diagnostic["true_counts"] == {"cat": 1183, "dog": 2486}
    assert diagnostic["per_species_recall"] == {"cat": 0.0, "dog": 1.0}
    assert diagnostic["macro_accuracy"] == 0.5
    assert diagnostic["constant_prediction_baseline_accuracy"] == pytest.approx(2486 / 3669)
    text = markdown.read_text(encoding="utf-8")
    assert text.index("Classifier failure — adversarial") < text.index("## Classification")
    assert "predicts only dog on all 3669 clean test images" in text
    assert "67.76% constant-dog baseline" in text
    assert "must not be presented as useful robust cat/dog recognition" in text
    assert "not proof that all hidden features are constant" in text
    assert public["arms"]["standard"]["clean_prediction_diagnostic"] == {"available": False}


def test_public_export_does_not_warn_when_both_species_are_predicted(tmp_path: Path) -> None:
    config = _bundle(tmp_path)
    path = config.project_path("results") / "evaluation.json"
    evaluation = _read(path)
    for arm in ("standard", "adversarial"):
        evaluation["arms"][arm]["full_test"]["clean"]["confusion_matrix"] = [
            [18, 2],
            [2, 18],
        ]
    atomic_write_json(path, evaluation)
    release_path = config.project_path("artifacts") / "release/local-release-manifest.json"
    release = _read(release_path)
    release["files"][str(path.relative_to(tmp_path))] = sha256_file(path)
    atomic_write_json(release_path, release)
    markdown, aggregate = build(tmp_path)
    public = _read(aggregate)
    for arm in ("standard", "adversarial"):
        diagnostic = public["arms"][arm]["clean_prediction_diagnostic"]
        assert diagnostic["available"] is True
        assert diagnostic["observed_single_class_prediction"] is False
        assert diagnostic["prediction_counts"] == {"cat": 20, "dog": 20}
        assert diagnostic["per_species_recall"] == {"cat": 0.9, "dog": 0.9}
    assert "Classifier failure" not in markdown.read_text(encoding="utf-8")


def test_public_export_discloses_unsuccessful_gray_stimuli_and_visible_captions(
    tmp_path: Path,
) -> None:
    config = _bundle(tmp_path)
    path = config.project_path("results") / "feature_visualizations.json"
    features = _read(path)
    layer2 = features["arms"]["adversarial"]["layers"]["network.layer2"]
    layer2["selected_channels"] = [73, 74]
    layer2["synthetic_responses"] = [
        {
            "channel": channel,
            "initial_unregularized_mean_response": 0.0,
            "final_unregularized_mean_response": final,
            "unregularized_response_gain": final,
        }
        for channel, final in ((73, 0.0), (74, 2.0))
    ]
    layer4 = {
        "selected_channels": [491, 418, 61],
        "calibration_mean_by_species": [[0.9, 0.8, 0.6], [0.6, 0.4, 0.9]],
        "relative_change_from_initial": [0.1, 0.3, 0.2],
        "spatial_shape": [7, 7],
        "receptive_field_pixels": 435,
        "synthetic_responses": [
            {
                "channel": channel,
                "initial_unregularized_mean_response": 0.0,
                "final_unregularized_mean_response": 0.0,
                "unregularized_response_gain": 0.0,
            }
            for channel in (491, 418, 61)
        ],
    }
    features["arms"]["adversarial"]["layers"]["network.layer4"] = layer4
    features["figures"][0]["arm"] = "adversarial"
    caption = "Aggregate localization. WARNING: failed all-dog classifier."
    features["figures"].append(
        {
            **features["figures"][0],
            "kind": "localization_summary",
            "caption": caption,
        }
    )
    atomic_write_json(path, features)
    release_path = config.project_path("artifacts") / "release/local-release-manifest.json"
    release = _read(release_path)
    release["files"][str(path.relative_to(tmp_path))] = sha256_file(path)
    atomic_write_json(release_path, release)

    markdown, aggregate = build(tmp_path)
    public = _read(aggregate)
    layers = public["arms"]["adversarial"]["feature_diagnostics"]["layers"]
    assert (
        layers["network.layer2"]["activation_maximization"]["zero_response_single_start_count"] == 1
    )
    assert (
        layers["network.layer4"]["activation_maximization"]["zero_response_single_start_count"] == 3
    )
    response = layers["network.layer2"]["activation_maximization"]["responses"][0]
    assert response["single_start_status"] == "unsuccessful_zero_response_single_start"
    assert response["initial_unregularized_mean_response"] == 0.0
    assert response["final_unregularized_mean_response"] == 0.0
    assert response["calibration_mean_response_cat"] == 0.4
    assert response["calibration_mean_response_dog"] == 0.2
    text = markdown.read_text(encoding="utf-8")
    assert f"**Figure caption.** {caption}" in text
    assert "network.layer2 channel(s) 73" in text
    assert "network.layer4 channel(s) 491, 418, 61" in text
    assert "exactly 0 → 0" in text
    assert "not evidence of dead channels or that the model learned nothing" in text
    assert "positive measured responses on real clean calibration inputs" in text
    assert "no retry, reselection, or retraining" in text
    assert public["arms"]["standard"]["feature_diagnostics"]["layers"]["network.layer2"][
        "activation_maximization"
    ] == {"available": False}


def test_public_export_rejects_shareable_pet_overlays(tmp_path: Path) -> None:
    config = _bundle(tmp_path)
    features = _read(config.project_path("results") / "feature_visualizations.json")
    features["figures"][1]["shareable"] = True
    with pytest.raises(RuntimeError, match="unapproved shareable figure kind"):
        _shareable_figures(config, features)


def test_public_export_retains_null_cam_correlations(tmp_path: Path) -> None:
    config = _bundle(tmp_path)
    path = config.project_path("results") / "feature_visualizations.json"
    features = _read(path)
    for arm in ("standard", "adversarial"):
        features["arms"][arm]["anchors"][0]["randomization_cam_correlation"] = None
    atomic_write_json(path, features)
    release_path = config.project_path("artifacts") / "release/local-release-manifest.json"
    release = _read(release_path)
    release["files"][str(path.relative_to(tmp_path))] = sha256_file(path)
    atomic_write_json(release_path, release)
    markdown, aggregate = build(tmp_path)
    public = _read(aggregate)
    for arm in ("standard", "adversarial"):
        assert (
            public["arms"][arm]["feature_diagnostics"]["anchors"][0][
                "randomization_cam_correlation"
            ]
            is None
        )
    assert "n/a | 0/1" in markdown.read_text(encoding="utf-8")


def test_public_export_rejects_unsafe_anchor_paths(tmp_path: Path) -> None:
    config = _bundle(tmp_path)
    features = _read(config.project_path("results") / "feature_visualizations.json")
    arm = features["arms"]["standard"]
    arm["anchors"][0]["sample_id"] = PRIVATE_FIXTURE_ROOT + "pet"
    with pytest.raises(RuntimeError, match="must not contain paths"):
        _public_feature_summary(arm)


def test_public_export_validates_current_evidence_hashes(tmp_path: Path) -> None:
    config = _bundle(tmp_path)
    path = config.project_path("results") / "evaluation.json"
    evaluation = _read(path)
    evaluation["arms"]["standard"]["full_test"]["clean"]["accuracy"] = 0.1
    atomic_write_json(path, evaluation)
    with pytest.raises(RuntimeError, match="evidence hash mismatch"):
        build(tmp_path)


def test_tracked_status_page_matches_active_config_and_local_links() -> None:
    public = _read(ROOT / "docs/results.json")
    markdown = ROOT / "docs/results.md"
    config = load_config(ROOT / "configs/experiment.yaml")
    assert public["provenance"]["configuration_sha256"] == config.sha256
    assert public["protocol"]["classes"] == 2
    assert public["protocol"]["label_mode"] == "species"
    assert "primary_hypothesis" not in public
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown.read_text(encoding="utf-8")):
        assert "://" not in target
        path = (markdown.parent / target.split("#", maxsplit=1)[0]).resolve()
        assert path.is_relative_to(ROOT)
        assert path.exists(), target
