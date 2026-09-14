from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat

from src.config import load_config
from src.notebook_support import ROOT

NOTEBOOK = ROOT / "notebooks" / "oxford_pets_results_explained.ipynb"
CONFIG = ROOT / "configs" / "experiment.yaml"
PUBLIC_RESULTS = ROOT / "docs/results.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_results_notebook_is_visual_and_readable_without_execution() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call]
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in notebook.cells)
    attachments = {name for cell in notebook.cells for name in getattr(cell, "attachments", {})}
    assert len(attachments) == 13
    assert attachments == {
        "split-and-species.png",
        "breed-balance.png",
        "image-geometry.png",
        "cosine-drift.png",
        "knn-retention-by-breed.png",
        "pca-standard.png",
        "pca-adversarial.png",
        "tsne-standard.png",
        "tsne-adversarial.png",
        "umap-standard.png",
        "umap-adversarial.png",
        "pair-boundary-standard.png",
        "pair-boundary-adversarial.png",
    }
    assert all(cell.cell_type == "markdown" for cell in notebook.cells)
    assert "read without running any cell" in source
    assert "Exploratory data analysis" in source
    assert "7,349" in source
    assert "Original 512-D" in source
    config = load_config(CONFIG)
    public = _read_json(PUBLIC_RESULTS)
    standard = public["arms"]["standard"]
    adversarial = public["arms"]["adversarial"]
    assert f"{config.integer('training', 'epochs')}-epoch reference run" in source
    assert f"{standard['attack_subset']['pgd_robust_accuracy']:.2%}" in source
    assert f"{adversarial['attack_subset']['pgd_robust_accuracy']:.2%}" in source
    assert f"{standard['representation']['median_cosine_drift']:.4f}" in source
    assert f"{adversarial['representation']['median_cosine_drift']:.4f}" in source
    assert "one-epoch pilot" not in source
    assert "both models reached zero robust accuracy" not in source
    assert "restore 15 epochs" not in source
    assert "No retraining is needed" in source
    assert "/Users/" not in source


def test_results_notebook_records_the_current_evidence_identity() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call]
    metadata = notebook.metadata["result_snapshot"]
    config = load_config(CONFIG)
    public = _read_json(PUBLIC_RESULTS)
    provenance = public["provenance"]
    assert metadata["epochs"] == config.integer("training", "epochs")
    assert metadata["config_sha256"] == config.sha256
    assert metadata["evaluation_sha256"] == provenance["evaluation_sha256"]
    assert metadata["representations_sha256"] == provenance["representations_sha256"]
    assert metadata["standard_checkpoint_sha256"] == provenance["checkpoints"]["standard"]["sha256"]
    assert (
        metadata["adversarial_checkpoint_sha256"]
        == provenance["checkpoints"]["adversarial"]["sha256"]
    )
    assert Path(NOTEBOOK).is_file()
