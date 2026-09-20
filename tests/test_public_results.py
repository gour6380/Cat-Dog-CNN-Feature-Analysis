from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scripts.build_public_results import _public_feature_summary, _shareable_figures, build
from src.config import ExperimentConfig, load_config
from src.feature_section_types import FEATURE_SECTIONS
from src.feature_sections import FeatureEvidenceError
from src.io_utils import atomic_write_json, atomic_write_text, sha256_file
from src.training import checkpoint_path

ROOT = Path(__file__).parents[1]
PRIVATE_FIXTURE_ROOT = "/" + "Users/private/"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _config(root: Path) -> ExperimentConfig:
    destination = root / "configs/experiment.yaml"
    atomic_write_text(destination, (ROOT / "configs/experiment.yaml").read_text())
    return load_config(destination)


def _bundle(root: Path, monkeypatch: pytest.MonkeyPatch) -> ExperimentConfig:
    config = _config(root)
    # Actual cache/source validation is tested in test_feature_sections.
    # This fixture isolates export policy without scientific computation.
    monkeypatch.setattr("src.feature_sections.validate_feature_evidence", lambda *_a, **_k: None)
    features = {
        "config_sha256": config.sha256,
        "status": "complete",
        "method_sha256": "fixture-method",
        "created_at": "fixture-time",
        "completion": {"missing": []},
        "section_receipts": {},
        "arms": {},
        "figures": [],
        "limitations": ["Descriptive only."],
    }
    layer = {
        "selected_channels": [2],
        "calibration_mean_by_species": [[0.4], [0.2]],
        "relative_change_from_initial": [0.1],
        "spatial_shape": [14, 14],
        "receptive_field_pixels": 99,
        "synthetic_responses": [
            {
                "channel": 2,
                "initial_unregularized_mean_response": 0.0,
                "final_unregularized_mean_response": 0.0,
                "unregularized_response_gain": 0.0,
            }
        ],
    }
    anchor = {
        "sample_id": "fixed-cat",
        "label": 0,
        "prediction": 1,
        "clean_margin": -0.2,
        "randomization_cam_correlation": None,
        "pgd_prediction": 0,
        "image_path": PRIVATE_FIXTURE_ROOT + "pet.jpg",
        "pixels": [[1]],
    }
    for arm in ("standard", "adversarial"):
        checkpoint = checkpoint_path(config, arm)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(arm.encode())
        features["arms"][arm] = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "state_unchanged": True,
            "layers": {"network.layer2": layer},
            "anchors": [anchor],
        }
        features["section_receipts"][arm] = {section: {} for section in FEATURE_SECTIONS}
        for section in FEATURE_SECTIONS:
            image = config.project_path("figures") / f"{arm}-{section}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), "gray").save(image)
            kind = {
                "synthetic": "activation_maximization",
                "diagnostics": "localization_summary",
            }.get(section, section)
            features["figures"].append(
                {
                    "path": str(image.relative_to(root)),
                    "sha256": sha256_file(image),
                    "caption": f"Fixture {arm} {section}",
                    "arm": arm,
                    "section": section,
                    "kind": kind,
                    "shareable": section in {"kernels", "synthetic", "diagnostics"},
                }
            )
    atomic_write_json(config.project_path("results") / "feature_visualizations.json", features)
    return config


def test_missing_evidence_creates_fresh_pending_snapshot(tmp_path: Path) -> None:
    config = _config(tmp_path)
    markdown, aggregate = build(tmp_path)
    public = _read(aggregate)
    assert public["status"] == "fresh results pending"
    assert public["provenance"]["configuration_sha256"] == config.sha256
    assert public["protocol"]["label_mode"] == "species"
    assert public["protocol"]["epochs"] == config.integer("training", "epochs")
    assert public["arms"] == {} and public["figures"] == []
    assert public["provenance"]["public_assets"] == {}
    assert "No fresh result has been substituted" in markdown.read_text()


def test_export_is_feature_only_safe_and_whitelisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bundle(tmp_path, monkeypatch)
    markdown, aggregate = build(tmp_path)
    public = _read(aggregate)
    assert not (config.project_path("results") / "evaluation.json").exists()
    assert public["status"] == "complete current feature walkthrough"
    assert len(public["figures"]) == 4
    assert len(list((tmp_path / "docs/assets").glob("*.png"))) == 4
    assert {"standard", "adversarial"} == {item["arm"] for item in public["figures"]}
    serialized = json.dumps(public)
    for forbidden in (
        "image_path",
        "/Users/",
        '"pixels":',
        "pgd_prediction",
        "attack_subset",
        "calibration_policy",
        "full_test",
        "secondary_policy_shift",
    ):
        assert forbidden not in serialized
    assert public["protocol"]["official_test_accuracy_measured"] is False
    assert public["protocol"]["post_training_attack_accuracy_measured"] is False
    for arm in ("standard", "adversarial"):
        result = public["arms"][arm]["feature_diagnostics"]
        assert result["anchors"][0]["randomization_cam_correlation"] is None
        assert result["layers"]["network.layer2"]["calibration_mean_response_cat"] == 0.4
    for relative, expected in public["provenance"]["public_assets"].items():
        assert sha256_file(tmp_path / relative) == expected
    text = markdown.read_text()
    assert "not a population experiment" in text
    assert "not evidence of a dead channel" in text
    assert "Fixture adversarial synthetic" not in text
    assert "## Classification" not in text


def test_unrelated_evaluation_and_confidence_documents_are_not_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bundle(tmp_path, monkeypatch)
    atomic_write_text(config.project_path("results") / "evaluation.json", "not valid JSON")
    atomic_write_text(config.project_path("results") / "policy.json", "not valid JSON")
    assert _read(build(tmp_path)[1])["status"] == "complete current feature walkthrough"


def test_public_export_rejects_shareable_photo_overlays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bundle(tmp_path, monkeypatch)
    features = _read(config.project_path("results") / "feature_visualizations.json")
    features["figures"][0]["kind"] = "gradcam"
    features["figures"][0]["shareable"] = True
    with pytest.raises(RuntimeError, match="unapproved shareable figure kind"):
        _shareable_figures(config, features)


def test_public_export_rejects_unsafe_anchor_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bundle(tmp_path, monkeypatch)
    arm = _read(config.project_path("results") / "feature_visualizations.json")["arms"]["standard"]
    arm["anchors"][0]["sample_id"] = PRIVATE_FIXTURE_ROOT + "pet"
    with pytest.raises(RuntimeError, match="must not contain paths"):
        _public_feature_summary(arm)


@pytest.mark.parametrize("failure", ["partial", "image", "checkpoint", "method"])
def test_incomplete_or_stale_feature_evidence_exports_pending_not_historical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    config = _bundle(tmp_path, monkeypatch)
    path = config.project_path("results") / "feature_visualizations.json"
    features = _read(path)
    if failure == "partial":
        features["status"] = "partial"
    elif failure == "image":
        features["figures"][0]["sha256"] = "stale"
    elif failure == "checkpoint":
        features["arms"]["adversarial"]["checkpoint_sha256"] = "stale"
    else:

        def invalid(*_a: Any, **_k: Any) -> None:
            raise FeatureEvidenceError("stale method identity")

        monkeypatch.setattr("src.feature_sections.validate_feature_evidence", invalid)
    atomic_write_json(path, features)
    public = _read(build(tmp_path)[1])
    assert public["status"] == "fresh results pending" and public["figures"] == []
    assert public["provenance"]["public_assets"] == {}


def test_tracked_status_page_matches_completed_current_run() -> None:
    current = load_config(ROOT / "configs/experiment.yaml")
    public = _read(ROOT / "docs/results.json")
    assert public["status"] == "complete current feature walkthrough"
    assert public["provenance"]["configuration_sha256"] == current.sha256
    assert public["protocol"]["label_mode"] == "species"
    assert public["protocol"]["epochs"] == 15
    assert public["figures"] and public["provenance"]["public_assets"]
    assert set(public["arms"]) == {"standard", "adversarial"}
    monitoring = public["training_monitoring"]
    assert monitoring["arms"]["standard"]["validation"]["correct_count"] == 294
    assert monitoring["arms"]["adversarial"]["validation"]["prediction_counts"] == {
        "0": 0,
        "1": 295,
    }
    assert "historical_evidence" not in public
    markdown = ROOT / "docs/results.md"
    text = markdown.read_text()
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        assert "://" not in target
        path = (markdown.parent / target.split("#", maxsplit=1)[0]).resolve()
        assert path.is_relative_to(ROOT) and path.exists(), target
