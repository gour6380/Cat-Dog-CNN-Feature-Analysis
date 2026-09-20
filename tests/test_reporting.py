from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.config import ExperimentConfig, _canonical_hash, load_config
from src.corruptions import registered_corruptions
from src.feature_section_types import FEATURE_SECTIONS
from src.io_utils import atomic_write_json, sha256_file
from src.reporting import (
    ReportError,
    _attack_label,
    _attack_table,
    _feature_table,
    _localization_table,
    _long_form,
    _release_files,
    _species_diagnostic_section,
    _synthetic_optimization_note,
    _technical_report,
    report,
)
from src.training import checkpoint_path

ROOT = Path(__file__).parents[1]


def _evaluation_features(config: ExperimentConfig) -> tuple[dict[str, Any], dict[str, Any]]:
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
    evaluation = {
        "config_sha256": config.sha256,
        "created_at": "fixed-evidence-time",
        "arms": {
            arm: {
                "checkpoint_sha256": "placeholder",
                "full_test": {
                    condition: deepcopy(metrics)
                    for condition in (
                        "clean",
                        *[
                            c.identifier
                            for c in registered_corruptions(config.section("corruptions"))
                        ],
                    )
                },
                "attack_subset": {"fgsm": attacks, "pgd": attacks},
            }
            for arm in ("standard", "adversarial")
        },
        "secondary_policy_shift": {},
    }
    layer = {
        "selected_channels": [2, 4],
        "calibration_mean_by_species": [[0.4, 0.6], [0.2, 0.8]],
        "relative_change_from_initial": [0.1, 0.3],
        "spatial_shape": [14, 14],
        "receptive_field_pixels": 211,
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
    }
    features = {
        "config_sha256": config.sha256,
        "created_at": "fixed-evidence-time",
        "device": "cpu",
        "selection": {
            "calibration_sample_ids": ["cal_001", "cal_002"],
            "test_anchor_ids": ["pet_001"],
        },
        "arms": {
            arm: {
                "checkpoint_sha256": "placeholder",
                "layers": {"network.layer2": deepcopy(layer)},
                "anchors": [deepcopy(anchor)],
                "state_unchanged": True,
            }
            for arm in ("standard", "adversarial")
        },
        "figures": [],
        "limitations": ["Synthetic stimuli do not prove semantic concepts."],
    }
    return evaluation, features


def test_report_labels_follow_configured_attack() -> None:
    config = load_config(ROOT / "configs/experiment.yaml")
    config.raw["attack"]["evaluation_steps"] = 10
    config.raw["attack"]["evaluation_restarts"] = 2
    evaluation, features = _evaluation_features(config)
    assert _attack_label(config) == "PGD-10×2"
    assert "PGD-10×2 robust" in _attack_table(config, evaluation)
    assert "network.layer2" in _feature_table(features)
    assert "14×14" in _feature_table(features)
    assert "0.2000" in _feature_table(features)
    assert "0.4000" in _localization_table(features)
    assert "0.1000" in _localization_table(features)


def test_report_is_deterministic_and_descriptively_scoped() -> None:
    config = load_config(ROOT / "configs/experiment.yaml")
    evaluation, features = _evaluation_features(config)
    text = _technical_report(config, evaluation, features)
    assert text == _technical_report(config, evaluation, features)
    assert "not causal proof" in text
    assert "not training-source attribution" in text
    assert "not recovered training photographs" in text
    assert "theoretical input support" in text
    assert "binary head" in text
    assert "NLL | Brier" in text
    assert "bootstrapped median" not in text
    assert "PCA plane" not in text
    assert "fixed true-species logit" in text
    assert "related but different scalar objectives" in text
    assert "200" in text and "3669" in text


def test_localization_report_preserves_undefined_correlations() -> None:
    config = load_config(ROOT / "configs/experiment.yaml")
    _evaluation, features = _evaluation_features(config)
    for arm in ("standard", "adversarial"):
        first = features["arms"][arm]["anchors"][0]
        undefined = deepcopy(first)
        undefined["sample_id"] = "constant_001"
        undefined["randomization_cam_correlation"] = None
        features["arms"][arm]["anchors"].append(undefined)
    text = _localization_table(features)
    assert "Defined CAM correlations" in text
    assert "0.2000 | 1/2" in text
    for arm in ("standard", "adversarial"):
        features["arms"][arm]["anchors"][0]["randomization_cam_correlation"] = None
    assert "n/a | 0/2" in _localization_table(features)


def test_clean_species_failure_is_prominent_and_not_feature_collapse() -> None:
    config = load_config(ROOT / "configs/experiment.yaml")
    evaluation, features = _evaluation_features(config)
    evaluation["arms"]["standard"]["full_test"]["clean"] = {
        **evaluation["arms"]["standard"]["full_test"]["clean"],
        "confusion_matrix": [[1173, 10], [13, 2473]],
    }
    evaluation["arms"]["adversarial"]["full_test"]["clean"] = {
        **evaluation["arms"]["adversarial"]["full_test"]["clean"],
        "confusion_matrix": [[0, 1183], [0, 2486]],
    }
    section = _species_diagnostic_section(evaluation)
    assert "adversarial arm — WARNING" in section
    assert "only dog on all 3669" in section
    assert "| adversarial | 1183 | 2486 | 0 | 3669 | 0.00% | 100.00% | 50.00% |" in section
    assert "FAILED comparison" in section
    assert "not useful robust cat/dog recognition" in section
    assert "Hidden features may remain variable" in section
    assert "cause of the classifier failure remains unproven" in section
    text = _technical_report(config, evaluation, features)
    assert text.index("WARNING") < text.index("## Question and scope")


def test_missing_clean_confusion_matrix_does_not_infer_failure_from_accuracy() -> None:
    config = load_config(ROOT / "configs/experiment.yaml")
    evaluation, features = _evaluation_features(config)
    assert _species_diagnostic_section(evaluation) == ""
    assert "WARNING" not in _technical_report(config, evaluation, features)


def test_zero_gain_synthetic_trials_are_disclosed_without_dead_channel_claims() -> None:
    config = load_config(ROOT / "configs/experiment.yaml")
    evaluation, features = _evaluation_features(config)
    layer = features["arms"]["adversarial"]["layers"]["network.layer2"]
    layer["synthetic_responses"] = [
        {
            "channel": 2,
            "initial_unregularized_mean_response": 0.1,
            "final_unregularized_mean_response": 1.2,
            "unregularized_response_gain": 1.1,
        },
        {
            "channel": 4,
            "initial_unregularized_mean_response": 0.0,
            "final_unregularized_mean_response": 0.0,
            "unregularized_response_gain": 0.0,
        },
    ]
    note = _synthetic_optimization_note(features)
    assert "1 of 2 recorded single-start" in note
    assert "| adversarial | `network.layer2` | 4 | 0.0000 | 0.0000 | 0.0000 |" in note
    assert "not evidence of dead channels" in note
    assert "respond positively to real calibration images" in note
    assert "no retry or replacement" in note
    assert note in _technical_report(config, evaluation, features)
    assert note in _long_form(config, evaluation, features)


def test_missing_synthetic_response_metadata_gracefully_skips_note() -> None:
    config = load_config(ROOT / "configs/experiment.yaml")
    _evaluation, features = _evaluation_features(config)
    assert _synthetic_optimization_note(features) == ""


def _report_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExperimentConfig:
    base = load_config(ROOT / "configs/experiment.yaml")
    raw = deepcopy(base.raw)
    # Legacy report-shape fixture: no training/array provenance is fabricated.
    # New runtime manifests and visuals are covered in test_runtime_visuals.
    raw["training_monitoring"]["enabled"] = False
    config = ExperimentConfig(base.path, tmp_path, raw, _canonical_hash(raw))
    evaluation, features = _evaluation_features(config)
    # Root engine integrity is exercised with real synthetic receipts in
    # test_feature_sections; here only rendering/orchestration is bounded.
    monkeypatch.setattr(
        "src.feature_sections.validate_feature_evidence", lambda *_args, **_kw: None
    )
    features.update(
        {
            "status": "complete",
            "method_sha256": "synthetic-method",
            "completion": {"missing": []},
            "section_receipts": {},
            "provenance": {"initialization_sha256": "initialization"},
        }
    )
    for arm in ("standard", "adversarial"):
        checkpoint = checkpoint_path(config, arm)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"fixture-checkpoint-{arm}".encode())
        checkpoint_hash = sha256_file(checkpoint)
        evaluation["arms"][arm]["checkpoint_sha256"] = checkpoint_hash
        features["arms"][arm]["checkpoint_sha256"] = checkpoint_hash
        features["section_receipts"][arm] = {}
        for section in FEATURE_SECTIONS:
            image = config.project_path("figures") / f"{arm}-{section}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), "white").save(image)
            features["figures"].append(
                {
                    "arm": arm,
                    "section": section,
                    "kind": section,
                    "caption": "Synthetic fixture, not scientific evidence.",
                    "shareable": section in {"synthetic", "kernels"},
                    "path": str(image.relative_to(tmp_path)),
                    "sha256": sha256_file(image),
                }
            )
            arrays = tmp_path / f"artifacts/features/{arm}-{section}.npz"
            arrays.parent.mkdir(parents=True, exist_ok=True)
            arrays.write_bytes(b"bounded synthetic array fixture")
            receipt = arrays.with_suffix(".json")
            atomic_write_json(
                receipt,
                {
                    "arrays_path": str(arrays.relative_to(tmp_path)),
                    "arrays_sha256": sha256_file(arrays),
                    "dependencies": [],
                },
            )
            features["section_receipts"][arm][section] = {
                "path": str(receipt.relative_to(tmp_path)),
                "sha256": sha256_file(receipt),
            }
    atomic_write_json(config.project_path("results") / "feature_visualizations.json", features)
    atomic_write_json(
        config.project_path("artifacts") / "data/manifest.json",
        {"dataset_content_sha256": "dataset", "split_sha256": "split"},
    )
    atomic_write_json(
        config.project_path("artifacts") / "model/initialization.json",
        {"state_sha256": "initialization"},
    )
    return config


def test_report_consumes_feature_evidence_and_preserves_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _report_bundle(tmp_path, monkeypatch)
    summary = report(config, progress=False)
    assert summary["study"] == "binary species CNN feature learning and localization"
    assert summary["feature_diagnostics"]["model_states_unchanged"] is True
    assert "primary_supported" not in summary
    assert "clean_species_prediction_diagnostics" not in summary
    assert "secondary_policy_shift" not in summary
    assert summary["feature_diagnostics"]["population_accuracy_measured"] is False
    assert not (config.project_path("results") / "evaluation.json").exists()
    assert not (config.project_path("reports") / "sunday-draft.md").exists()
    import json

    release = json.loads(Path(summary["local_release_manifest"]).read_text(encoding="utf-8"))
    assert release["config_sha256"] == config.sha256
    assert release["provenance"]["initialization_sha256"] == "initialization"
    for arm in ("standard", "adversarial"):
        assert release["fixed_checkpoints"][arm]["sha256"] == sha256_file(
            checkpoint_path(config, arm)
        )
    path = "results/generated/feature_visualizations.json"
    assert release["files"][path] == sha256_file(config.root / path)


def test_report_rejects_mutated_feature_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _report_bundle(tmp_path, monkeypatch)
    path = config.project_path("results") / "feature_visualizations.json"
    import json

    features = json.loads(path.read_text(encoding="utf-8"))
    features["arms"]["standard"]["state_unchanged"] = False
    atomic_write_json(path, features)
    with pytest.raises(ReportError, match="model state changed"):
        report(config, progress=False)


def test_report_ignores_unrelated_partial_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _report_bundle(tmp_path, monkeypatch)
    atomic_write_json(config.project_path("results") / "evaluation.json", {"status": "partial"})
    assert report(config, progress=False)["status"] == "complete current feature walkthrough"


def test_report_rejects_incomplete_feature_sections_without_writing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    config = _report_bundle(tmp_path, monkeypatch)
    path = config.project_path("results") / "feature_visualizations.json"
    features = json.loads(path.read_text())
    features["status"] = "partial"
    features["completion"] = {"missing": ["adversarial/diagnostics"]}
    atomic_write_json(path, features)
    with pytest.raises(ReportError, match="complete current feature sections"):
        report(config, progress=False)
    assert not config.project_path("reports").exists()


def test_binary_release_inventory_excludes_retained_superseded_breed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _report_bundle(tmp_path, monkeypatch)
    old_checkpoint = tmp_path / "checkpoints/old-breed-config/standard/epoch-015.pt"
    old_checkpoint.parent.mkdir(parents=True)
    old_checkpoint.write_bytes(b"retained historical evidence")
    old_result = config.project_path("results") / "representations.json"
    atomic_write_json(old_result, {"config_sha256": "old-breed-config"})
    inventory = _release_files(config)
    assert old_checkpoint not in inventory and old_result not in inventory
    assert old_checkpoint.is_file() and old_result.is_file()
    assert checkpoint_path(config, "standard") in inventory
