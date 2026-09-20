from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import nbformat
import pytest
from nbconvert import HTMLExporter
from PIL import Image

from src.config import ExperimentConfig, load_config
from src.feature_results_notebook import build_feature_results_notebook
from src.feature_section_types import FEATURE_SECTIONS
from src.io_utils import atomic_write_json, sha256_file
from src.reporting import ReportError
from src.training import checkpoint_path

ROOT = Path(__file__).parents[1]


def _bundle(root: Path, monkeypatch: pytest.MonkeyPatch) -> ExperimentConfig:
    base = load_config(ROOT / "configs/experiment.yaml")
    config = ExperimentConfig(base.path, root, deepcopy(base.raw), base.sha256)
    # Scientific cache integrity is covered with actual synthetic receipts in
    # test_feature_sections. This fixture isolates presentation without a dataset.
    monkeypatch.setattr("src.feature_sections.validate_feature_evidence", lambda *_a, **_k: None)
    features = {
        "config_sha256": config.sha256,
        "status": "complete",
        "created_at": "fixture-time",
        "method_sha256": "fixture-method",
        "completion": {"missing": []},
        "section_receipts": {},
        "arms": {},
        "figures": [],
        "limitations": ["Interpretation is descriptive."],
        "selection": {
            "calibration_sample_ids": ["ref-cat", "ref-dog"],
            "test_anchor_ids": ["anchor-cat", "anchor-dog"],
        },
    }
    for arm in ("standard", "adversarial"):
        path = checkpoint_path(config, arm)
        path.parent.mkdir(parents=True)
        path.write_bytes(arm.encode())
        features["arms"][arm] = {
            "checkpoint_sha256": sha256_file(path),
            "state_unchanged": True,
            "anchors": [],
            "layers": {},
        }
        features["section_receipts"][arm] = {section: {} for section in FEATURE_SECTIONS}
        for section in FEATURE_SECTIONS:
            image = config.project_path("figures") / f"{arm}-{section}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), "white" if arm == "standard" else "gray").save(image)
            features["figures"].append(
                {
                    "path": str(image.relative_to(root)),
                    "sha256": sha256_file(image),
                    "arm": arm,
                    "section": section,
                    "kind": section,
                    "caption": f"Fixture {arm} {section}; not scientific evidence.",
                    "shareable": section in {"kernels", "synthetic"},
                }
            )
    atomic_write_json(config.project_path("results") / "feature_visualizations.json", features)
    return config


def test_companion_embeds_both_models_without_evaluation_or_model_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bundle(tmp_path, monkeypatch)
    with patch("src.model.build_model", side_effect=AssertionError("unexpected model run")):
        path = build_feature_results_notebook(config)
        first_hash = sha256_file(path)
        assert build_feature_results_notebook(config) == path
        assert sha256_file(path) == first_hash
    assert not (config.project_path("results") / "evaluation.json").exists()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert all(cell.cell_type == "markdown" for cell in notebook.cells)
    assert sum(len(cell.get("attachments", {})) for cell in notebook.cells) == 16
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "Feature patterns" in source and "Prediction influence" in source
    assert "four anchors are not a test-accuracy" in source
    assert "PGD-trained" in source and "Standard" in source
    assert "evaluation_sha256" not in notebook.metadata.evidence
    assert notebook.metadata.evidence.population_accuracy_measured is False
    manifest = json.loads((tmp_path / "artifacts/features/results-notebook.json").read_text())
    assert manifest["public_export"] is False and manifest["code_cells"] == 0
    assert manifest["figure_count"] == 16


def test_markdown_attachments_render_as_embedded_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bundle(tmp_path, monkeypatch)
    notebook = nbformat.read(build_feature_results_notebook(config), as_version=4)
    html, _ = HTMLExporter().from_notebook_node(notebook)
    assert html.count('src="data:image/png;base64,') == 16
    assert 'src="attachment:' not in html


def test_companion_uses_first_registered_image_and_links_remaining_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bundle(tmp_path, monkeypatch)
    path = config.project_path("results") / "feature_visualizations.json"
    features = json.loads(path.read_text())
    extra = deepcopy(features["figures"][0])
    image = config.project_path("figures") / "later-stage.png"
    Image.new("RGB", (4, 4), "red").save(image)
    extra.update(
        {
            "path": str(image.relative_to(tmp_path)),
            "sha256": sha256_file(image),
            "caption": "later registered view",
        }
    )
    features["figures"].append(extra)
    atomic_write_json(path, features)
    notebook = nbformat.read(build_feature_results_notebook(config), as_version=4)
    assert sum(len(cell.get("attachments", {})) for cell in notebook.cells) == 16
    assert "later-stage.png" in notebook.cells[1].source
    assert "later registered view" not in notebook.cells[1].source


def test_companion_explains_unsuccessful_synthetic_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _bundle(tmp_path, monkeypatch)
    path = config.project_path("results") / "feature_visualizations.json"
    features = json.loads(path.read_text())
    features["arms"]["adversarial"]["layers"] = {
        "network.layer4": {
            "selected_channels": [491],
            "calibration_mean_by_species": [[0.2], [0.3]],
            "synthetic_responses": [
                {
                    "channel": 491,
                    "initial_unregularized_mean_response": 0.0,
                    "final_unregularized_mean_response": 0.0,
                    "unregularized_response_gain": 0.0,
                }
            ],
        }
    }
    atomic_write_json(path, features)
    source = "\n".join(
        cell.source
        for cell in nbformat.read(build_feature_results_notebook(config), as_version=4).cells
    )
    assert "not evidence of dead channels" in source
    assert "| 491 |" in source and "no retry or replacement" in source


@pytest.mark.parametrize(
    "mismatch", ["configuration", "checkpoint", "image", "partial", "missing_family"]
)
def test_companion_rejects_misaligned_or_incomplete_current_feature_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    config = _bundle(tmp_path, monkeypatch)
    path = config.project_path("results") / "feature_visualizations.json"
    features = json.loads(path.read_text())
    if mismatch == "configuration":
        features["config_sha256"] = "old-run"
    elif mismatch == "checkpoint":
        features["arms"]["standard"]["checkpoint_sha256"] = "old-model"
    elif mismatch == "image":
        features["figures"][0]["sha256"] = "old-image"
    elif mismatch == "partial":
        features["status"] = "partial"
    else:
        del features["section_receipts"]["adversarial"]["occlusion"]
    atomic_write_json(path, features)
    with pytest.raises(ReportError):
        build_feature_results_notebook(config)
    assert not (config.project_path("reports") / "cat_dog_feature_results.ipynb").exists()
