from __future__ import annotations

import io
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from scripts.refresh_notebook_views import _capture
from src import runtime_visuals
from src.config import ExperimentConfig
from src.io_utils import atomic_save_npz, atomic_write_json, sha256_file
from src.notebook_views import (
    compact_records,
    feature_note,
    real_patch_preview,
    section_measurements,
    synthetic_preview,
    validation_note,
)


def _history(matrix: list[list[int]]) -> list[dict[str, Any]]:
    return [{"epoch": 2, "monitoring": {"validation_clean": {"confusion_matrix": matrix}}}]


def test_validation_exposes_majority_guess_without_attack_claim() -> None:
    text = validation_note(_history([[0, 95], [0, 200]]), "adversarial")
    for phrase in ("200/295 (67.80%)", "0/95 (0.00%)", "0 cat / 295 dog", "all 295"):
        assert phrase in text
    assert "majority-class guess" in text and "not an attack test" in text
    assert "robust accuracy" not in text


def test_validation_preserves_species_counts_and_does_not_invent_missing_details() -> None:
    text = validation_note(_history([[90, 5], [0, 200]]), "standard")
    assert "290/295 (98.31%)" in text and "90/95 (94.74%)" in text
    assert "Watch out" not in text
    assert "unavailable" in validation_note([], "standard")
    assert "unavailable" in validation_note([{"epoch": 1}], "standard")


def test_compact_selection_pairs_first_registered_cat_dog_not_best_predictions() -> None:
    ids = ["cat-2", "cat-1", "dog-2", "dog-1"]
    evidence = {
        "selection": {"test_anchor_ids": ids},
        "arms": {
            "standard": {
                "anchors": [
                    {"sample_id": name, "label": int(name.startswith("dog")), "prediction": 1}
                    for name in ids
                ]
            }
        },
        "figures": [
            {"section": "stages", "arm": arm, "sample_id": name}
            for arm in ("adversarial", "standard")
            for name in reversed(ids)
        ],
    }
    before = deepcopy(evidence)
    records = compact_records(evidence, "stages")
    assert [(row["sample_id"], row["arm"]) for row in records] == [
        ("cat-2", "standard"),
        ("cat-2", "adversarial"),
        ("dog-2", "standard"),
        ("dog-2", "adversarial"),
    ]
    assert evidence == before


def test_optional_diagnostics_not_removed_from_saved_evidence() -> None:
    evidence = {
        "figures": [
            {"section": "diagnostics", "kind": kind, "arm": "standard"}
            for kind in (
                "species_response",
                "randomization_control",
                "initial_response_change",
                "localization_summary",
            )
        ]
    }
    result = compact_records(evidence, "diagnostics")
    assert [row["kind"] for row in result] == ["randomization_control", "localization_summary"]
    assert len(evidence["figures"]) == 4


def _synthetic() -> dict[str, np.ndarray[Any, Any]]:
    return {
        "synthetic_layers": np.asarray(["early", "middle", "late"]),
        "synthetic_channels": np.asarray([4, 8, 16]),
        "synthetic_pixels": np.full((3, 3, 8, 8), 0.5, dtype=np.float32),
        "initial_unregularized_mean_responses": np.asarray([0.25, 0.1, 0.0]),
        "final_unregularized_mean_responses": np.asarray([1.0, 0.2, 0.0]),
        "unregularized_response_gains": np.asarray([0.75, 0.1, 0.0]),
    }


def test_synthetic_relayout_uses_all_recorded_tiles_and_retains_failed_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arrays = _synthetic()
    titles: list[str] = []
    tiles: list[np.ndarray[Any, Any]] = []
    from matplotlib.figure import Figure

    original = Figure.savefig

    def save(figure: Figure, *args: Any, **kwargs: Any) -> None:
        titles.extend(axis.get_title() for axis in figure.axes)
        tiles.extend(np.asarray(axis.images[0].get_array()) for axis in figure.axes)
        original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    picture = synthetic_preview(arrays, "adversarial")
    assert Image.open(io.BytesIO(picture)).format == "PNG"
    assert len(tiles) == 3 and "No response increase" in titles[-1]
    for tile, pixels in zip(tiles, arrays["synthetic_pixels"], strict=True):
        np.testing.assert_array_equal(tile, pixels.transpose(1, 2, 0))
    assert "1/3" in feature_note("synthetic", {}, arrays)
    assert "does not mean the channel is dead" in feature_note("synthetic", {}, arrays)
    assert not plt.get_fignums()


@pytest.mark.parametrize("changed", ["pixels", "gains", "alignment"])
def test_synthetic_relayout_rejects_invalid_arrays(changed: str) -> None:
    arrays = _synthetic()
    if changed == "pixels":
        arrays["synthetic_pixels"][0, 0, 0, 0] = np.nan
    elif changed == "gains":
        arrays["unregularized_response_gains"][0] = 8.0
    else:
        arrays["synthetic_channels"] = np.asarray([1])
    with pytest.raises(ValueError):
        synthetic_preview(arrays, "standard")


def _patches() -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    layers = ["network.relu", "network.layer2", "network.layer4"]
    receipt = {"payload": {"layers": {layer: {"selected_channels": [5, 2, 9]} for layer in layers}}}
    return receipt, {
        "patch_layers": np.asarray([layer for layer in layers for _ in range(3)]),
        "patch_channels": np.asarray([5, 2, 9] * 3),
        "patch_ranks": np.ones(9, dtype=int),
        "patch_image_indices": np.arange(9),
        "patch_sample_ids": np.asarray([f"image-{i}" for i in range(9)]),
        "reference_image_ids": np.asarray([f"image-{i}" for i in range(9)]),
        "reference_pixels": np.arange(9 * 3 * 8 * 8, dtype=np.float32).reshape(9, 3, 8, 8)
        / (9 * 3 * 8 * 8),
        "patch_rf_boxes": np.tile([1, 2, 5, 6], (9, 1)),
    }


def test_real_patch_preview_retains_registered_order_pixels_and_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.figure import Figure

    receipt, arrays = _patches()
    images: list[np.ndarray[Any, Any]] = []
    titles: list[str] = []
    original = Figure.savefig

    def save(figure: Figure, *args: Any, **kwargs: Any) -> None:
        images.extend(np.asarray(axis.images[0].get_array()) for axis in figure.axes)
        titles.extend(axis.get_title() for axis in figure.axes)
        original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    picture = real_patch_preview(receipt, arrays, "standard")
    assert Image.open(io.BytesIO(picture)).format == "PNG"
    assert len(images) == 12
    assert "channel 5" in titles[0] and "channel 2" in titles[2]
    assert not any("channel 9" in title for title in titles)
    for row, reference_index in enumerate([0, 1, 3, 4, 6, 7]):
        pixels = arrays["reference_pixels"][reference_index].transpose(1, 2, 0)
        np.testing.assert_array_equal(images[row * 2], pixels)
        np.testing.assert_array_equal(images[row * 2 + 1], pixels[2:6, 1:5])


@pytest.mark.parametrize("changed", ["image", "box", "rank"])
def test_real_patch_preview_rejects_misaligned_or_invalid_images(changed: str) -> None:
    receipt, arrays = _patches()
    if changed == "image":
        arrays["patch_sample_ids"][0] = "other"
    elif changed == "box":
        arrays["patch_rf_boxes"][0] = [-1, 0, 5, 6]
    else:
        arrays["patch_ranks"][0] = 2
    with pytest.raises(ValueError):
        real_patch_preview(receipt, arrays, "adversarial")


def test_empty_gradcam_and_undefined_controls_are_explained_not_removed() -> None:
    receipt = {
        "payload": {"anchors": [{"sample_id": "cat", "randomization_cam_correlation": None}]}
    }
    arrays = {"anchor_0_gradcam": np.zeros((7, 7))}
    assert "No positive Grad-CAM map" in feature_note("gradcam", receipt, arrays)
    note = feature_note("diagnostics", receipt, arrays)
    assert "undefined for 1/1" in note and "not zero or a passed check" in note


def test_saved_measurement_loading_verifies_hashes_and_identity(tmp_path: Path) -> None:
    config = ExperimentConfig(tmp_path / "configs/experiment.yaml", tmp_path, {}, "config")
    arrays = tmp_path / "artifacts/synthetic.npz"
    atomic_save_npz(arrays, _synthetic())
    receipt_path = tmp_path / "artifacts/synthetic.json"
    receipt = {
        "arm": "standard",
        "section": "synthetic",
        "identity": "run",
        "config_sha256": "config",
        "method_sha256": "method",
        "arrays_path": "artifacts/synthetic.npz",
        "arrays_sha256": sha256_file(arrays),
    }
    atomic_write_json(receipt_path, receipt)
    evidence = {
        "identity": "run",
        "method_sha256": "method",
        "section_receipts": {
            "standard": {
                "synthetic": {
                    "path": "artifacts/synthetic.json",
                    "sha256": sha256_file(receipt_path),
                }
            }
        },
    }
    saved, values = section_measurements(config, evidence, "standard", "synthetic")
    assert saved == receipt
    np.testing.assert_array_equal(values["synthetic_pixels"], _synthetic()["synthetic_pixels"])
    evidence["identity"] = "other-run"
    with pytest.raises(ValueError, match="identity"):
        section_measurements(config, evidence, "standard", "synthetic")
    evidence["identity"] = "run"
    atomic_save_npz(arrays, {"wrong": np.ones(3)})
    with pytest.raises(ValueError, match="hash"):
        section_measurements(config, evidence, "standard", "synthetic")


def test_compact_curve_shows_balanced_and_majority_baseline_from_saved_values() -> None:
    history = _history([[0, 95], [0, 200]])
    history[0]["mean_training_loss"] = 0.7
    history[0]["monitoring"]["validation_clean"].update(
        {"accuracy": 200 / 295, "macro_accuracy": 0.5}
    )
    figure = runtime_visuals.plot_learning_curves(history, "adversarial", compact=True)
    try:
        assert len(figure.axes) == 2
        assert [line.get_ydata()[0] for line in figure.axes[1].lines] == pytest.approx(
            [100 * 200 / 295, 50, 100 * 200 / 295]
        )
        assert figure.axes[1].get_xticks().tolist() == [2]
    finally:
        plt.close(figure)


def test_read_only_output_capture_keeps_image_and_evidence_metadata() -> None:
    pixels = io.BytesIO()
    Image.new("RGB", (8, 8)).save(pixels, format="PNG")

    def view() -> None:
        Picture, Markdown, display = runtime_visuals._ipython_display()
        display(Markdown("Recorded, not rerun"))
        display(Picture(data=pixels.getvalue()), metadata={"source_arrays_sha256": "recorded"})

    outputs = _capture(view)
    assert outputs[0]["data"]["text/markdown"] == "Recorded, not rerun"
    assert "image/png" in outputs[1]["data"]
    assert outputs[1]["metadata"]["source_arrays_sha256"] == "recorded"
