from __future__ import annotations

import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest
from PIL import Image

from src import runtime_visuals as visuals
from src.calibration import softmax, tie_aware_risk_coverage
from src.config import ExperimentConfig, load_config
from src.corruptions import registered_corruptions
from src.evaluation import EvaluationError
from src.feature_results_notebook import build_feature_results_notebook
from src.io_utils import atomic_save_npz, atomic_write_bytes, atomic_write_json, sha256_file
from src.metrics import accuracy_metrics
from src.training import checkpoint_path
from src.training_monitoring import prepare_training_monitoring, save_training_examples

ROOT = Path(__file__).parents[1]


def _config(tmp_path: Path) -> ExperimentConfig:
    raw = deepcopy(load_config(ROOT / "configs" / "experiment.yaml").raw)
    raw["training"].update({"epochs": 1, "num_workers": 0})
    raw["dataset"].update({"expected_test": 4, "attack_per_class": 2})
    return ExperimentConfig(tmp_path / "configs" / "experiment.yaml", tmp_path, raw, "a" * 64)


def _png(path: Path, value: int = 120) -> None:
    image = Image.new("RGB", (16, 16), (value, 60, 140))
    output = io.BytesIO()
    image.save(output, format="PNG")
    atomic_write_bytes(path, output.getvalue())


def _history() -> list[dict[str, Any]]:
    return [
        {
            "epoch": epoch,
            "mean_training_loss": loss,
            "last_learning_rates": [0.0001 / epoch, 0.001 / epoch],
            "monitoring": {
                "train_clean": {
                    "loss": 0.8 / epoch,
                    "accuracy": 0.7 + 0.1 * epoch,
                    "macro_accuracy": 0.65 + 0.1 * epoch,
                },
                "validation_clean": {
                    "loss": 1.0 / epoch,
                    "accuracy": 0.5 + 0.1 * epoch,
                    "macro_accuracy": 0.4 + 0.1 * epoch,
                    "per_class_accuracy": {"0": 0.2 + 0.1 * epoch, "1": 0.8},
                    "confusion_matrix": [[1, 1], [0, 2]],
                },
            },
        }
        for epoch, loss in ((1, 1.2), (2, 0.6))
    ]


@pytest.fixture
def evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from src.data import SampleRecord

    config = _config(tmp_path)
    photo = tmp_path / "synthetic-photo.png"
    _png(photo)

    def record(prefix: str, species: int, index: int) -> SampleRecord:
        return SampleRecord(
            f"{prefix}-{'cat' if species == 0 else 'dog'}-{index:02d}",
            str(photo),
            species,
            species,
            f"synthetic-breed-{species}",
            "test" if prefix == "test" else "trainval",
            species,
        )

    records = [record("test", label, index) for label in (0, 1) for index in (0, 1)]
    splits = {
        "test": records,
        "attack": records,
        "calibration": [record("cal", label, index) for label in (0, 1) for index in (0, 1)],
        "training": [record("fit", label, index) for label in (0, 1) for index in range(20)],
    }
    monkeypatch.setattr("src.data.load_registered_splits", lambda _: splits)
    monkeypatch.setattr("src.training_monitoring.load_registered_splits", lambda _: splits)
    provenance = {"config_sha256": config.sha256, "scientific_source": "synthetic"}
    monkeypatch.setattr(visuals, "experiment_provenance", lambda _: provenance)

    def artifact_provenance(_: Any, arm: str, checkpoint_hash: str) -> dict[str, str]:
        return {**provenance, "arm": arm, "checkpoint_sha256": checkpoint_hash}

    monkeypatch.setattr(visuals, "_artifact_provenance", artifact_provenance)
    conditions = [
        "clean",
        *[c.identifier for c in registered_corruptions(config.raw["corruptions"])],
    ]
    evaluation: dict[str, Any] = {
        "config_sha256": config.sha256,
        "status": "complete",
        "arms": {},
        "secondary_policy_shift": {},
    }
    saved_arrays: dict[tuple[str, str, str], dict[str, np.ndarray[Any, Any]]] = {}

    def arrays_for(
        arm: str, condition: str, selected: list[Any]
    ) -> dict[str, np.ndarray[Any, Any]]:
        labels = np.asarray([r.label for r in selected], dtype=np.int64)
        logits = (
            np.array([[3, 0], [0, 3], [0, 3], [0, 3]], dtype=np.float64)
            if arm == "standard"
            else np.tile([0.0, 3.0], (4, 1))
        )
        if condition != "clean":
            logits = logits / 3
        pixels = np.stack(
            [np.full((3, 224, 224), 0.2 + index * 0.1, dtype=np.float32) for index in range(4)]
        )
        return {
            "sample_ids": np.asarray([r.sample_id for r in selected]),
            "labels": labels,
            "species": labels.copy(),
            "logits": logits,
            "features": np.zeros((4, 512), dtype=np.float32),
            "gallery_sample_ids": np.asarray([r.sample_id for r in selected]),
            "gallery_labels": labels.copy(),
            "gallery_species": labels.copy(),
            "gallery_pixels": pixels,
        }

    def write_arrays(arm: str, partition: str, condition: str, arrays: dict[str, Any]) -> None:
        group = "calibration" if partition == "calibration" else f"{partition}/{arm}"
        name = arm if partition == "calibration" else condition
        path = config.project_path("results") / "per_sample" / group / f"{name}.npz"
        atomic_save_npz(path, arrays)
        atomic_write_json(
            path.with_suffix(".json"),
            {
                "provenance": artifact_provenance(
                    config, arm, sha256_file(checkpoint_path(config, arm))
                ),
                "npz_sha256": sha256_file(path),
            },
        )
        saved_arrays[arm, partition, condition] = arrays

    for arm in visuals.ARMS:
        atomic_write_bytes(checkpoint_path(config, arm), f"synthetic-checkpoint-{arm}".encode())
        clean = arrays_for(arm, "clean", records)
        metrics = accuracy_metrics(clean["logits"], clean["labels"], 2)
        risk = {"coverage": 0.9, "selective_risk": 0.1, "nll": 0.3, "brier": 0.2, "ece_15": 0.05}
        full = {}
        for index, condition in enumerate(conditions):
            full[condition] = {
                **metrics,
                "risk": {
                    **risk,
                    "coverage": 0.9 - index * 0.02,
                    "selective_risk": 0.1 + index * 0.01,
                },
            }
            write_arrays(arm, "full-test", condition, arrays_for(arm, condition, records))
        write_arrays(arm, "calibration", "clean", arrays_for(arm, "clean", splits["calibration"]))
        write_arrays(arm, "attack", "clean", clean)
        attack_metrics = {
            "clean_accuracy_on_subset": metrics["accuracy"],
            "robust_accuracy": 0.25,
            "attack_success_clean_correct": 0.75,
            "per_class": {"0": {"robust_accuracy": 0.0}, "1": {"robust_accuracy": 0.5}},
            "cumulative_restart_robust_accuracy": [0.5, 0.25, 0.25, 0.25, 0.25],
        }
        for condition in ("fgsm", "pgd"):
            attacked = arrays_for(arm, condition, records)
            clean_pixels = attacked.pop("gallery_pixels")
            attacked["gallery_clean_pixels"] = clean_pixels
            attacked["gallery_attack_pixels"] = clean_pixels + np.float32(0.01)
            attacked["cumulative_predictions"] = np.tile(np.array([1, 1, 0, 1]), (5, 1))
            write_arrays(arm, "attack", condition, attacked)
        evaluation["arms"][arm] = {
            "checkpoint_sha256": sha256_file(checkpoint_path(config, arm)),
            "policy": {"temperature": 2.0, "confidence_threshold": 0.9},
            "full_test": full,
            "attack_subset": {"fgsm": deepcopy(attack_metrics), "pgd": deepcopy(attack_metrics)},
        }
    atomic_write_json(config.project_path("results") / "evaluation.json", evaluation)
    return {
        "config": config,
        "evaluation": evaluation,
        "arrays": saved_arrays,
        "provenance": provenance,
        "splits": splits,
        "write_arrays": write_arrays,
    }


@pytest.fixture
def captured_figures(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    figures: list[Any] = []
    original = visuals._save

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        figures.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(visuals, "_save", capture)
    return figures


def test_learning_curves_match_history_and_leave_missing_measurements_as_gaps() -> None:
    history = _history()
    history[1]["monitoring"] = {}
    figure = visuals.plot_learning_curves(history, "adversarial")
    try:
        axes = figure.axes
        assert np.allclose(axes[0].lines[0].get_ydata(), [1.2, 0.6])
        assert "PGD" in axes[0].lines[0].get_label()
        assert "augmented CE" in axes[0].lines[0].get_label()
        assert axes[1].lines[0].get_ydata()[0] == 0.8
        assert np.isnan(axes[1].lines[0].get_ydata()[1])
        assert axes[2].lines[0].get_ydata()[0] == pytest.approx(80)
        assert axes[4].lines[0].get_ydata()[0] == pytest.approx(30)
        assert np.allclose(axes[5].lines[0].get_ydata(), [0.0001, 0.00005])
    finally:
        plt.close(figure)


def test_one_epoch_learning_curve_does_not_fabricate_additional_epochs() -> None:
    figure = visuals.plot_learning_curves(_history()[:1], "standard")
    try:
        assert len(figure.axes[0].lines[0].get_xdata()) == 1
        assert figure.axes[0].lines[0].get_xdata()[0] == 1
        assert "Clean augmented" in figure.axes[0].lines[0].get_label()
    finally:
        plt.close(figure)


def test_final_training_registry_has_measured_curves_and_validation_health(
    evidence: dict[str, Any], captured_figures: list[Any]
) -> None:
    manifest = visuals.training_visuals(evidence["config"], _history()[:1], "standard", final=True)
    assert manifest["completed_epoch"] == 1
    assert manifest["final"] is True
    assert [row["kind"] for row in manifest["figures"]] == [
        "standard-learning-curves",
        "standard-validation",
    ]
    assert np.allclose(captured_figures[0].axes[0].lines[0].get_ydata(), [1.2])
    health = captured_figures[1]
    assert np.array_equal(health.axes[0].images[0].get_array(), [[1, 1], [0, 2]])
    assert [bar.get_height() for bar in health.axes[1].patches] == [1, 3]
    for record in manifest["figures"]:
        assert sha256_file(evidence["config"].root / record["path"]) == record["sha256"]


def test_missing_training_history_does_not_invent_a_curve(evidence: dict[str, Any]) -> None:
    with pytest.raises(visuals.VisualEvidenceError, match="history is unavailable"):
        visuals.training_visuals(evidence["config"], [], "standard")


def test_clean_runtime_charts_are_metric_matched_and_photo_panels_are_local(
    evidence: dict[str, Any], captured_figures: list[Any]
) -> None:
    manifest = visuals.evaluation_visuals(evidence["config"], "clean")
    assert manifest["population"] == "full official test"
    assert len(manifest["figures"]) == 4
    assert [r["shareable"] for r in manifest["figures"]] == [True, True, False, False]
    first = captured_figures[0]
    assert [bar.get_height() for bar in first.axes[0].patches[:4]] == [75, 75, 50, 100]
    assert np.array_equal(first.axes[1].images[0].get_array(), [[1, 1], [0, 2]])
    assert "Full official test" in first.axes[0].get_title()


@pytest.mark.parametrize("family", ["gaussian_noise", "gaussian_blur", "brightness", "contrast"])
def test_each_corruption_chart_uses_only_that_family_and_exact_recorded_values(
    evidence: dict[str, Any], captured_figures: list[Any], family: str
) -> None:
    config = evidence["config"]
    manifest = visuals.evaluation_visuals(config, family)
    assert manifest["population"] == "full official test"
    dashboard = captured_figures[0]
    selected = [
        "clean",
        *[
            c.identifier
            for c in registered_corruptions(config.raw["corruptions"])
            if c.name == family
        ],
    ]
    expected = [
        100 * evidence["evaluation"]["arms"]["standard"]["full_test"][c]["risk"]["coverage"]
        for c in selected
    ]
    assert np.allclose(dashboard.axes[4].lines[0].get_ydata(), expected)
    assert family.replace("_", " ").title() in dashboard._suptitle.get_text()
    assert len(dashboard.axes[0].lines[0].get_xdata()) == 3
    assert all("transform parameter" in axis.get_xlabel().lower() for axis in dashboard.axes)


def test_attack_gallery_displays_exact_captured_pixels_predictions_and_amplification(
    evidence: dict[str, Any],
) -> None:
    config = evidence["config"]
    figure = visuals._input_gallery(config, "standard", ["clean", "pgd"], 2.0, attack=True)
    arrays = evidence["arrays"]["standard", "attack", "pgd"]
    try:
        assert np.array_equal(
            figure.axes[0].images[0].get_array(),
            arrays["gallery_clean_pixels"][0].transpose(1, 2, 0),
        )
        assert np.array_equal(
            figure.axes[1].images[0].get_array(),
            arrays["gallery_attack_pixels"][0].transpose(1, 2, 0),
        )
        expected = np.clip(
            np.abs(arrays["gallery_attack_pixels"][0] - arrays["gallery_clean_pixels"][0]) * 32,
            0,
            1,
        ).transpose(1, 2, 0)
        assert np.array_equal(figure.axes[2].images[0].get_array(), expected)
        assert "display only" in figure.axes[2].get_title()
        assert "confidence" in figure.axes[1].get_title()
        assert "test-cat-00" in figure.axes[0].get_ylabel()
    finally:
        plt.close(figure)


def test_pgd_charts_label_subset_and_use_cumulative_restart_values(
    evidence: dict[str, Any], captured_figures: list[Any]
) -> None:
    manifest = visuals.evaluation_visuals(evidence["config"], "pgd")
    assert manifest["population"] == "balanced attack subset"
    assert np.allclose(captured_figures[1].axes[0].lines[0].get_ydata(), [50, 25, 25, 25, 25])
    assert "Balanced" in captured_figures[0].axes[0].get_title()
    assert [bar.get_height() for bar in captured_figures[0].axes[0].patches[:2]] == [75, 25]


def test_confidence_summary_uses_arrays_only_and_accepts_ties_together(
    evidence: dict[str, Any], captured_figures: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbid(*_: Any, **__: Any) -> Any:
        raise AssertionError("visual summary must not construct a model or rerun inference")

    monkeypatch.setattr("src.model.build_model", forbid)
    monkeypatch.setattr("src.evaluation.load_checkpoint_model", forbid)
    monkeypatch.setattr("src.evaluation._collect_clean_or_shift", forbid)
    monkeypatch.setattr("src.evaluation._collect_attack", forbid)
    manifest = visuals.confidence_visuals(evidence["config"])
    assert manifest["no_additional_inference"] is True
    arrays = evidence["arrays"]["standard", "full-test", "clean"]
    probabilities = softmax(arrays["logits"] / 2)
    coverage, risk, _ = tie_aware_risk_coverage(
        probabilities.max(axis=1), probabilities.argmax(axis=1) == arrays["labels"]
    )
    curve = captured_figures[1].axes[0].lines[0]
    assert np.array_equal(curve.get_xdata(), 100 * coverage)
    assert np.array_equal(curve.get_ydata(), 100 * risk)
    assert curve.get_xdata().tolist() == [100.0]


@pytest.mark.parametrize("problem", ["configuration", "checkpoint", "npz", "ids"])
def test_current_array_verification_rejects_stale_or_misaligned_evidence(
    evidence: dict[str, Any], problem: str
) -> None:
    config = evidence["config"]
    path = config.project_path("results") / "per_sample/full-test/standard/clean.npz"
    sidecar = path.with_suffix(".json")
    if problem == "configuration":
        metadata = json.loads(sidecar.read_text())
        metadata["provenance"]["config_sha256"] = "historical"
        atomic_write_json(sidecar, metadata)
    elif problem == "checkpoint":
        atomic_write_bytes(checkpoint_path(config, "standard"), b"different checkpoint")
    elif problem == "npz":
        atomic_write_bytes(path, b"changed arrays")
    else:
        arrays = deepcopy(evidence["arrays"]["standard", "full-test", "clean"])
        arrays["sample_ids"][0] = "foreign"
        evidence["write_arrays"]("standard", "full-test", "clean", arrays)
    with pytest.raises((visuals.VisualEvidenceError, EvaluationError)):
        visuals._array(config, "standard", "full-test")


def test_missing_gallery_never_substitutes_historical_docs_image(evidence: dict[str, Any]) -> None:
    config = evidence["config"]
    _png(config.root / "docs/assets/historical.png")
    arrays = deepcopy(evidence["arrays"]["standard", "full-test", "clean"])
    for key in list(arrays):
        if key.startswith("gallery_"):
            del arrays[key]
    evidence["write_arrays"]("standard", "full-test", "clean", arrays)
    with pytest.raises(visuals.VisualEvidenceError, match="no historical substitute"):
        visuals._input_gallery(config, "standard", ["clean"], 2.0)


def test_confidence_summary_requires_all_registered_conditions(evidence: dict[str, Any]) -> None:
    config = evidence["config"]
    evaluation = deepcopy(evidence["evaluation"])
    del evaluation["arms"]["standard"]["full_test"]["contrast-0.5"]
    atomic_write_json(config.project_path("results") / "evaluation.json", evaluation)
    with pytest.raises(visuals.VisualEvidenceError, match="completed conditions"):
        visuals.confidence_visuals(config)


def test_registry_checks_configuration_method_checkpoint_and_plot_hashes(
    evidence: dict[str, Any],
) -> None:
    config = evidence["config"]
    path = config.project_path("figures") / "synthetic.png"
    _png(path)
    record = {
        "path": str(path.relative_to(config.root)),
        "sha256": sha256_file(path),
        "kind": "synthetic",
        "caption": "synthetic test figure",
        "shareable": True,
    }
    manifest = visuals._manifest(
        config,
        "synthetic",
        [record],
        {arm: sha256_file(checkpoint_path(config, arm)) for arm in visuals.ARMS},
    )
    assert visuals.current_visual_manifests(config) == [manifest]
    receipt_path = config.project_path("results") / "visualizations/synthetic.json"
    for field in ("config_sha256", "plot_source_sha256"):
        stale = {**manifest, field: "stale"}
        atomic_write_json(receipt_path, stale)
        assert visuals.current_visual_manifests(config) == []
    stale = deepcopy(manifest)
    stale["checkpoint_sha256"]["standard"] = "stale"
    atomic_write_json(receipt_path, stale)
    assert visuals.current_visual_manifests(config) == []
    atomic_write_json(receipt_path, manifest)
    _png(path, 10)
    with pytest.raises(visuals.VisualEvidenceError, match="hash mismatch"):
        visuals.current_visual_manifests(config)


def test_image_paths_cannot_escape_the_current_project(
    evidence: dict[str, Any], tmp_path: Path
) -> None:
    outside = tmp_path.parent / "outside-synthetic.png"
    _png(outside)
    with pytest.raises(visuals.VisualEvidenceError, match="missing current-run"):
        visuals._verified_path(
            evidence["config"], {"path": str(outside), "sha256": sha256_file(outside)}
        )


def test_training_examples_match_actual_captured_inputs_and_final_epoch(
    evidence: dict[str, Any],
) -> None:
    config = evidence["config"]
    protocol = prepare_training_monitoring(config, evidence["splits"])
    rows = [
        {
            "sample_id": r.sample_id,
            "label": r.label,
            "clean_pixels": np.full((3, 224, 224), 0.3, dtype=np.float32),
            "objective_pixels": np.full((3, 224, 224), 0.31, dtype=np.float32),
            "logits": np.array([0.1, 0.2], dtype=np.float32),
            "batch_index": 1,
            "update_index": 1,
            "original_image_path": r.image_path,
        }
        for r in protocol.gallery_records
    ]
    save_training_examples(config, "adversarial", 1, rows, evidence["provenance"], protocol)
    figure = visuals._training_examples(config, "adversarial")
    assert figure is not None
    try:
        assert np.allclose(figure.axes[0].images[0].get_array(), 0.3)
        assert np.allclose(figure.axes[1].images[0].get_array(), 0.31)
        assert np.allclose(figure.axes[2].images[0].get_array(), 0.32)
        assert "Actual PGD-5" in figure.axes[1].get_title()
        assert "epoch 1" in figure.axes[0].get_ylabel()
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "problem",
    [
        "id",
        "pixel_nan",
        "pixel_shape",
        "labels",
        "attack_bounds",
        "empty",
        "logits_nan",
        "logits_shape",
    ],
)
def test_training_gallery_rejects_bad_captured_evidence_even_with_fresh_file_hash(
    evidence: dict[str, Any], problem: str
) -> None:
    config = evidence["config"]
    protocol = prepare_training_monitoring(config, evidence["splits"])
    rows = [
        {
            "sample_id": record.sample_id,
            "label": record.label,
            "clean_pixels": np.full((3, 224, 224), 0.3, dtype=np.float32),
            "objective_pixels": np.full((3, 224, 224), 0.31, dtype=np.float32),
            "logits": np.array([0.1, 0.2], dtype=np.float32),
            "batch_index": 1,
            "update_index": 1,
            "original_image_path": record.image_path,
        }
        for record in protocol.gallery_records
    ]
    save_training_examples(config, "adversarial", 1, rows, evidence["provenance"], protocol)
    path = config.project_path("artifacts") / "training/adversarial-examples.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if problem == "id":
        arrays["sample_ids"][0] = "foreign"
    elif problem == "pixel_nan":
        arrays["objective_pixels"][0, 0, 0, 0] = np.nan
    elif problem == "pixel_shape":
        arrays["clean_pixels"] = arrays["clean_pixels"][:, :, :16, :16]
    elif problem == "labels":
        arrays["labels"][0] = 1 - arrays["labels"][0]
    elif problem == "empty":
        arrays = {key: value[:0] for key, value in arrays.items()}
    elif problem == "logits_nan":
        arrays["logits"][0, 0] = np.nan
    elif problem == "logits_shape":
        arrays["logits"] = arrays["logits"][:, :1]
    else:
        arrays["objective_pixels"][0, 0, 0, 0] = 0.7
    atomic_save_npz(path, arrays)
    sidecar = path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text())
    metadata.update({"sha256": sha256_file(path), "npz_sha256": sha256_file(path)})
    atomic_write_json(sidecar, metadata)
    with pytest.raises(visuals.VisualEvidenceError):
        visuals._training_examples(config, "adversarial")


def test_future_unsaved_gallery_epoch_is_not_shown_for_earlier_completed_checkpoint(
    evidence: dict[str, Any],
) -> None:
    config = evidence["config"]
    config.raw["training"]["epochs"] = 2
    protocol = prepare_training_monitoring(config, evidence["splits"])

    def rows(value: float) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": record.sample_id,
                "label": record.label,
                "clean_pixels": np.full((3, 224, 224), value, dtype=np.float32),
                "objective_pixels": np.full((3, 224, 224), value + 0.01, dtype=np.float32),
                "logits": np.array([0.1, 0.2], dtype=np.float32),
                "batch_index": 1,
                "update_index": 1,
                "original_image_path": record.image_path,
            }
            for record in protocol.gallery_records
        ]

    save_training_examples(config, "adversarial", 1, rows(0.3), evidence["provenance"], protocol)
    # Simulate capture succeeding but checkpoint writing being interrupted.
    save_training_examples(config, "adversarial", 2, rows(0.5), evidence["provenance"], protocol)
    figure = visuals._training_examples(config, "adversarial", completed_epoch=1)
    assert figure is not None
    try:
        assert np.allclose(figure.axes[0].images[0].get_array(), 0.3)
        assert np.allclose(figure.axes[1].images[0].get_array(), 0.31)
        assert "epoch 1" in figure.axes[0].get_ylabel()
        assert "epoch 2" not in figure.axes[0].get_ylabel()
    finally:
        plt.close(figure)


def _feature_bundle(evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    config = evidence["config"]
    monkeypatch.setattr("src.feature_sections.validate_feature_evidence", lambda *_a, **_k: None)
    features: dict[str, Any] = {
        "config_sha256": config.sha256,
        "status": "complete",
        "method_sha256": "synthetic-method",
        "created_at": "fixture-time",
        "completion": {"missing": []},
        "section_receipts": {},
        "selection": {"calibration_sample_ids": [], "test_anchor_ids": []},
        "arms": {},
        "figures": [],
        "limitations": ["not proof of learned anatomy"],
    }
    for arm in visuals.ARMS:
        features["arms"][arm] = {
            "checkpoint_sha256": sha256_file(checkpoint_path(config, arm)),
            "state_unchanged": True,
            "anchors": [],
            "layers": {},
        }
        features["section_receipts"][arm] = {section: {} for section in visuals.FEATURE_SECTIONS}
        for section in visuals.FEATURE_SECTIONS:
            image = config.project_path("figures") / f"{arm}-{section}.png"
            _png(image, 70 if arm == "standard" else 180)
            features["figures"].append(
                {
                    "arm": arm,
                    "section": section,
                    "kind": "stage_walkthrough" if section == "stages" else section,
                    "path": str(image.relative_to(config.root)),
                    "sha256": sha256_file(image),
                    "caption": f"current {arm} {section} view",
                    "shareable": section in {"kernels", "synthetic"},
                }
            )
    atomic_write_json(config.project_path("results") / "feature_visualizations.json", features)
    return features


def test_feature_report_and_companion_use_both_models_without_evaluation_dependencies(
    evidence: dict[str, Any],
    captured_figures: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = evidence["config"]
    _feature_bundle(evidence, monkeypatch)
    monkeypatch.setattr(
        visuals,
        "_artifact_provenance",
        lambda *_a: pytest.fail("feature report must not query evaluation provenance"),
    )
    # Deliberately invalid, unrelated legacy evidence is not read.
    atomic_write_bytes(config.project_path("results") / "evaluation.json", b"invalid old JSON")
    manifest = visuals.report_visuals(config)
    assert {f["kind"] for f in manifest["figures"]} == {
        "artifact-completion",
        "feature-gallery-preview",
    }
    assert "evaluation_provenance" not in manifest
    assert np.array_equal(captured_figures[0].axes[0].images[0].get_array(), np.ones((2, 8)))
    assert [axis.get_title() for axis in captured_figures[1].axes] == [
        "Standard · stages",
        "Standard · synthetic",
        "PGD-trained · stages",
        "PGD-trained · synthetic",
    ]
    atomic_write_bytes(
        config.project_path("results") / "visualizations/evaluation-stale.json", b"invalid JSON"
    )
    assert visuals.current_visual_manifests(config, feature_only=True) == [manifest]
    _png(config.root / "docs/assets/historical.png")
    notebook = nbformat.read(build_feature_results_notebook(config), as_version=4)
    assert all(cell.cell_type == "markdown" for cell in notebook.cells)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "current-run-dashboard" not in source and "historical.png" not in source
    assert "fixture-time" not in source
    assert "PGD-trained" in source and "Standard" in source
    assert sum(len(cell.get("attachments", {})) for cell in notebook.cells) == 16


def test_report_rejects_partial_features_instead_of_generating_finished_dashboard(
    evidence: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = evidence["config"]
    features = _feature_bundle(evidence, monkeypatch)
    features["status"] = "partial"
    features["completion"]["missing"] = ["adversarial/diagnostics"]
    atomic_write_json(config.project_path("results") / "feature_visualizations.json", features)
    with pytest.raises(visuals.VisualEvidenceError, match="incomplete"):
        visuals.report_visuals(config)
    assert not (config.project_path("results") / "visualizations/report.json").exists()


def test_feature_report_registry_rejects_changed_feature_manifest(
    evidence: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = evidence["config"]
    features = _feature_bundle(evidence, monkeypatch)
    visuals.report_visuals(config)
    features["method_sha256"] = "changed-method"
    atomic_write_json(config.project_path("results") / "feature_visualizations.json", features)
    assert visuals.current_visual_manifests(config, feature_only=True) == []


def test_compact_learning_curves_show_only_objective_and_validation_accuracy() -> None:
    history = _history()
    history[1]["monitoring"] = {}
    figure = visuals.plot_learning_curves(history, "adversarial", compact=True)
    try:
        assert len(figure.axes) == 2
        assert np.allclose(figure.axes[0].lines[0].get_ydata(), [1.2, 0.6])
        assert figure.axes[1].lines[0].get_ydata()[0] == pytest.approx(60)
        assert np.isnan(figure.axes[1].lines[0].get_ydata()[1])
        assert figure.axes[1].get_title() == "Validation accuracy"
    finally:
        plt.close(figure)


def test_compact_final_training_does_not_construct_health_or_image_galleries(
    evidence: dict[str, Any],
    captured_figures: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        visuals, "_training_examples", lambda *_a, **_k: pytest.fail("gallery call")
    )
    monkeypatch.setattr(visuals, "_confusion", lambda *_a: pytest.fail("health call"))
    manifest = visuals.training_visuals(
        evidence["config"], _history()[:1], "standard", final=True, compact=True
    )
    assert len(manifest["figures"]) == 1 and manifest["compact"] is True
    assert len(captured_figures[0].axes) == 2


def test_compact_observer_and_final_display_route_compact_parameter(
    evidence: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = evidence["config"]
    calls: list[dict[str, Any]] = []
    original = visuals.training_visuals

    def observe(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(visuals, "training_visuals", observe)
    monkeypatch.setattr(
        visuals, "_ipython_display", lambda: (lambda **_k: object(), str, lambda *_a, **_k: None)
    )
    update = visuals.training_observer(config, compact=True)
    update(_history()[:1], "standard")
    atomic_write_json(
        config.project_path("artifacts") / "training" / "standard.json",
        {"provenance": evidence["provenance"], "history": _history()[:1]},
    )
    visuals.display_training(config, "standard", compact=True)
    assert [row.get("compact") for row in calls] == [True, True]
    assert calls[-1]["final"] is True


def test_optional_stale_training_history_is_explicit_without_blocking_feature_report(
    evidence: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = evidence["config"]
    _feature_bundle(evidence, monkeypatch)
    atomic_write_json(
        config.project_path("artifacts") / "training" / "standard.json",
        {
            "provenance": {"config_sha256": "stale"},
            "history": _history()[:1],
        },
    )
    monkeypatch.setattr(visuals, "training_visuals", lambda *_a, **_k: pytest.fail("stale curve"))
    visuals.ensure_report_visuals(config)
    result = json.loads((config.project_path("results") / "visualizations/report.json").read_text())
    assert result["training_history_status"]["standard"].startswith("unavailable")
    assert result["training_history_status"]["adversarial"].startswith("unavailable")
    assert "no curve inferred" in capsys.readouterr().out
    assert "evaluation_provenance" not in result
