from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import nbformat
import pytest
from PIL import Image

from src.config import ExperimentConfig, load_config
from src.feature_results_notebook import build_feature_results_notebook
from src.io_utils import atomic_write_json, sha256_file
from src.training import checkpoint_path

ROOT = Path(__file__).parents[1]


def _bundle(root: Path) -> ExperimentConfig:
    base = load_config(ROOT / "configs/experiment.yaml")
    config = ExperimentConfig(base.path, root, deepcopy(base.raw), base.sha256)
    image = root / "figures/generated/features/local-panel.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), "white").save(image)
    features = {
        "config_sha256": config.sha256,
        "arms": {},
        "figures": [
            {
                "path": str(image.relative_to(root)),
                "sha256": sha256_file(image),
                "arm": "standard",
                "kind": "real_patches",
                "caption": "Fixture local panel; not model evidence.",
                "shareable": False,
            }
        ],
        "limitations": ["Interpretation is descriptive."],
    }
    evaluation = {"config_sha256": config.sha256, "arms": {}}
    for arm in ("standard", "adversarial"):
        path = checkpoint_path(config, arm)
        path.parent.mkdir(parents=True)
        path.write_bytes(arm.encode())
        identity = sha256_file(path)
        features["arms"][arm] = {"checkpoint_sha256": identity, "anchors": []}
        evaluation["arms"][arm] = {
            "checkpoint_sha256": identity,
            "full_test": {"clean": {"accuracy": 0.9, "risk": {"selective_risk": 0.1}}},
            "attack_subset": {
                "fgsm": {"robust_accuracy": 0.6},
                "pgd": {"robust_accuracy": 0.5},
            },
        }
    atomic_write_json(config.project_path("results") / "feature_visualizations.json", features)
    atomic_write_json(config.project_path("results") / "evaluation.json", evaluation)
    return config


def test_companion_embeds_verified_saved_images_without_model_execution(tmp_path: Path) -> None:
    config = _bundle(tmp_path)
    with patch("src.model.build_model", side_effect=AssertionError("unexpected model run")):
        path = build_feature_results_notebook(config)
        first_hash = sha256_file(path)
        assert build_feature_results_notebook(config) == path
        assert sha256_file(path) == first_hash
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.cell_type == "markdown" for cell in notebook.cells)
    assert sum(bool(cell.get("attachments")) for cell in notebook.cells) == 1
    assert notebook.metadata.evidence.built_from_saved_evidence is True
    assert notebook.metadata.evidence.contains_pet_photographs is True
    manifest = json.loads((tmp_path / "artifacts/features/results-notebook.json").read_text())
    assert manifest["public_export"] is False
    assert manifest["code_cells"] == 0 and manifest["figure_count"] == 1


def test_companion_prominently_reports_single_class_prediction_failure(tmp_path: Path) -> None:
    config = _bundle(tmp_path)
    evaluation_path = config.project_path("results") / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["arms"]["adversarial"]["full_test"]["clean"]["confusion_matrix"] = [[0, 2], [0, 3]]
    atomic_write_json(evaluation_path, evaluation)
    notebook = nbformat.read(build_feature_results_notebook(config), as_version=4)
    assert "Failed comparison arm" in notebook.cells[1].source
    assert "only dog" in notebook.cells[1].source
    assert "not established" in notebook.cells[1].source
    assert (
        notebook.metadata.evidence.clean_classifier_diagnostics.adversarial.prediction_counts.dog
        == 5
    )


def test_companion_explains_unsuccessful_synthetic_probes(tmp_path: Path) -> None:
    config = _bundle(tmp_path)
    feature_path = config.project_path("results") / "feature_visualizations.json"
    features = json.loads(feature_path.read_text())
    features["arms"]["adversarial"]["layers"] = {
        "network.layer4": {
            "synthetic_responses": [{"channel": 491, "unregularized_response_gain": 0.0}]
        }
    }
    atomic_write_json(feature_path, features)
    notebook = nbformat.read(build_feature_results_notebook(config), as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "not dead channels" in source and "channel 491" in source
    assert "positive responses on real calibration inputs" in source
    assert len(notebook.metadata.evidence.synthetic_nonpositive_gain_units) == 1


@pytest.mark.parametrize("mismatch", ["configuration", "checkpoint", "image"])
def test_companion_rejects_misaligned_saved_evidence(tmp_path: Path, mismatch: str) -> None:
    config = _bundle(tmp_path)
    feature_path = config.project_path("results") / "feature_visualizations.json"
    features = json.loads(feature_path.read_text())
    if mismatch == "configuration":
        features["config_sha256"] = "old-breed-study"
    elif mismatch == "checkpoint":
        features["arms"]["standard"]["checkpoint_sha256"] = "different-model"
    else:
        features["figures"][0]["sha256"] = "different-image"
    atomic_write_json(feature_path, features)
    with pytest.raises(ValueError):
        build_feature_results_notebook(config)
    assert not (config.project_path("reports") / "cat_dog_feature_results.ipynb").exists()
