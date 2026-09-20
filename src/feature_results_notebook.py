"""Compact read-only current feature walkthrough; reads saved evidence, never runs a model."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import nbformat

from src.config import ExperimentConfig
from src.feature_section_types import FEATURE_SECTIONS
from src.io_utils import atomic_write_json, atomic_write_text, sha256_file
from src.reporting import _synthetic_optimization_note, load_current_feature_evidence

SECTION_NOTES = {
    "stages": (
        "Image to score",
        "The actual model input passes through stem, pooling and "
        "layer1–layer4 to a 512-dimensional pooled vector and cat/dog scores. Maps are "
        "channel responses, not reconstructed photographs.",
    ),
    "kernels": (
        "Early filter weights",
        "The first-layer RGB kernels are learned weights; "
        "they are not photographs, anatomical detectors or entire high-level concepts.",
    ),
    "synthetic": (
        "Preferred synthetic patterns",
        "A fixed optimization start seeks inputs "
        "that excite selected channels. Gray or zero-gain tiles remain unsuccessful "
        "probes, not evidence of dead channels; there is no attractive-example retry.",
    ),
    "real_patches": (
        "Strongest real reference patches",
        "These are the strongest registered "
        "reference responses for the selected channels. A theoretical receptive-field "
        "crop is support, not an exact causal training source.",
    ),
    "activations": (
        "Channel location",
        "Activation maps and input gradients show where a "
        "selected channel responds on fixed anchors. Independently normalized overlays "
        "do not establish equal raw response strength.",
    ),
    "gradcam": (
        "Prediction influence: Grad-CAM",
        "The target is the fixed true-species logit. "
        "An overlay is a local attribution diagnostic, not proof of a named body-part detector.",
    ),
    "occlusion": (
        "Prediction influence: masking",
        "Positive drops mean masking reduced the "
        "true-species-minus-other logit margin. Equal-area top-CAM and random tiles "
        "need not agree; effects of individual masks are not additive.",
    ),
    "diagnostics": (
        "Controls and measured responses",
        "Randomized-weight controls and raw "
        "response charts expose method dependence. Four fixed anchors are descriptive "
        "examples, not a population accuracy or robustness evaluation.",
    ),
}


def build_feature_results_notebook(config: ExperimentConfig) -> Path:
    features = load_current_feature_evidence(config)
    destination = config.project_path("reports") / "cat_dog_feature_results.ipynb"

    def markdown(source: str) -> Any:
        return nbformat.v4.new_markdown_cell(source)  # type: ignore[no-untyped-call]

    cells = [
        markdown(
            "# Cat/dog CNN — illustrated current feature walkthrough\n\n"
            "Read-only saved evidence for **both standard and PGD-trained models**; no model, "
            "dataset or training execution occurs here. All eight current feature families are "
            "verified. This compact companion shows the first registered panel per family/model "
            "and links the remaining full-resolution images. "
            "No historical gallery is substituted.\n\n"
            "PGD is a training objective, not a measured robustness result. Fixed-anchor scores "
            "and predictions describe those examples only; **four anchors are not a test-accuracy "
            "estimate**. Training curves, when present, are held-out validation "
            "monitoring only.\n\n"
            "Photographs and derived photo panels remain Git-ignored local evidence."
        )
    ]
    included: list[dict[str, Any]] = []
    for section in FEATURE_SECTIONS:
        title, note = SECTION_NOTES[section]
        heading = "Feature patterns" if section in FEATURE_SECTIONS[:5] else "Prediction influence"
        cell = markdown(f"## {heading} · {title}\n\n{note}\n\n")
        cell["attachments"] = {}
        gallery: list[str] = []
        links: list[str] = []
        for arm in ("standard", "adversarial"):
            records = [
                record
                for record in features["figures"]
                if record.get("section") == section and record.get("arm") == arm
            ]
            if not records:
                raise ValueError(
                    f"verified feature family has no registered image: {arm}/{section}"
                )
            chosen = records[0]
            path = config.root / chosen["path"]
            name = f"{arm}-{section}-{chosen['sha256'][:12]}.png"
            cell["attachments"][name] = {"image/png": base64.b64encode(path.read_bytes()).decode()}
            label = "Standard" if arm == "standard" else "PGD-trained"
            gallery.append(f"![{label} · {title}](attachment:{name})")
            included.append(chosen)
            for record in records:
                relative = os.path.relpath(config.root / record["path"], destination.parent)
                links.append(f"- [{label} · {record['kind'].replace('_', ' ')}]({relative})")
        cell.source += (
            "| Standard | PGD-trained |\n|---|---|\n"
            + "| "
            + " | ".join(gallery)
            + " |\n\nFull saved panels:\n\n"
            + "\n".join(links)
        )
        cells.append(cell)
    synthetic_note = _synthetic_optimization_note(features)
    if synthetic_note:
        cells.append(markdown(synthetic_note))
    cells.append(
        markdown(
            "## Interpretation limits\n\n"
            + "\n".join(f"- {item}" for item in features.get("limitations", []))
            + "\n\nAn eye-like or fur-like response is a tentative visual description, not a "
            "verified semantic detector. No method identifies the exact training image pixels "
            "that taught a filter. Comparison is local and descriptive; there is no full-test "
            "accuracy, attack-accuracy, risk/calibration or safer-use claim "
            "in this feature-only run.\n\n"
            "[Current feature summary](technical-report.md)"
        )
    )
    for index, cell in enumerate(cells):
        cell["id"] = f"feature-results-{index:03d}"
    notebook = nbformat.v4.new_notebook(cells=cells)  # type: ignore[no-untyped-call]
    feature_path = config.project_path("results") / "feature_visualizations.json"
    notebook["metadata"]["evidence"] = {
        "config_sha256": config.sha256,
        "feature_sha256": sha256_file(feature_path),
        "feature_method_sha256": features["method_sha256"],
        "built_from_saved_evidence": True,
        "contains_pet_photographs": any(not record["shareable"] for record in included),
        "required_feature_sections": list(FEATURE_SECTIONS),
        "population_accuracy_measured": False,
        "representative_selection": "first registered figure per family and model; "
        "no attractiveness ranking",
    }
    nbformat.validate(notebook)
    atomic_write_text(destination, nbformat.writes(notebook, version=4))  # type: ignore[no-untyped-call]
    atomic_write_json(
        config.project_path("artifacts") / "features" / "results-notebook.json",
        {
            "config_sha256": config.sha256,
            "feature_evidence_sha256": sha256_file(feature_path),
            "path": str(destination.relative_to(config.root)),
            "sha256": sha256_file(destination),
            "code_cells": 0,
            "figure_count": len(included),
            "registered_figure_count": len(features["figures"]),
            "contains_original_photographs": any(not record["shareable"] for record in included),
            "public_export": False,
        },
    )
    return destination
