from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.config import ExperimentConfig, load_config
from src.data import SampleRecord
from src.feature_section_types import FEATURE_SECTIONS, FeatureSection, SectionData
from src.feature_sections import (
    METHOD_FILES,
    FeatureEvidenceError,
    FeatureWorkspace,
    SectionContext,
    _preserved_randomness,
    compute_feature_sections,
    validate_feature_evidence,
)
from src.io_utils import atomic_save_npz, atomic_write_json, atomic_write_text, sha256_file
from src.training import checkpoint_path

ROOT = Path(__file__).parents[1]


@pytest.fixture
def registered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    raw = deepcopy(load_config(ROOT / "configs/experiment.yaml").raw)
    raw["feature_visualization"].update({"calibration_per_class": 2, "anchors_per_class": 1})
    raw["training"]["num_workers"] = 0
    raw["training_monitoring"]["enabled"] = False
    config = ExperimentConfig(tmp_path / "configs/experiment.yaml", tmp_path, raw, "a" * 64)
    image = tmp_path / "synthetic.png"
    Image.new("RGB", (8, 8), (70, 100, 150)).save(image)

    def records(prefix: str) -> list[SampleRecord]:
        return [
            SampleRecord(
                f"{prefix}_{label}_{index}",
                str(image),
                label,
                label,
                f"breed_{label}",
                "test" if prefix == "test" else "trainval",
                label,
            )
            for label in (0, 1)
            for index in range(3)
        ]

    splits = {"calibration": records("cal"), "test": records("test")}
    monkeypatch.setattr("src.feature_sections.load_registered_splits", lambda _: splits)
    monkeypatch.setattr(
        "src.feature_sections.experiment_provenance",
        lambda _: {
            "config_sha256": config.sha256,
            "dataset_sha256": "synthetic",
            "initialization_sha256": "initial",
        },
    )
    for name in METHOD_FILES:
        atomic_write_text(tmp_path / name, f"synthetic source receipt for {name}\n")
    for arm in ("standard", "adversarial"):
        path = checkpoint_path(config, arm)
        path.parent.mkdir(parents=True)
        path.write_bytes(arm.encode())
    calls: list[tuple[str, str]] = []
    models: list[nn.Module] = []
    initial: list[bool] = []

    def load(*_: Any) -> nn.Module:
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        model[0].weight.grad = torch.ones_like(model[0].weight)
        models.append(model)
        return model

    def compute(section: FeatureSection, context: Any) -> SectionData:
        calls.append((context.arm, section))
        if section == "diagnostics":
            context.dependency("gradcam")
            context.dependency("occlusion")
        figure, axis = plt.subplots(figsize=(2, 2))
        axis.plot([0, 1], [0, len(calls)])
        context.register(
            figure,
            "synthetic-evidence",
            "Synthetic fixture, not a measured pet result.",
            "species_response",
            shareable=True,
        )
        return SectionData(
            payload={
                "anchors": [
                    {"sample_id": record.sample_id, "label": record.label}
                    for record in context.anchors
                ]
            },
            arrays={
                "anchor_sample_ids": np.asarray([record.sample_id for record in context.anchors]),
                "measurement": np.asarray([len(calls)], dtype=np.float32),
            },
        )

    monkeypatch.setattr("src.feature_sections.load_checkpoint_model", load)
    monkeypatch.setattr("src.feature_pattern_sections.compute_patterns", compute)
    monkeypatch.setattr("src.feature_anchor_sections.compute_anchors", compute)
    prepare_initial = FeatureWorkspace.prepare_initial
    monkeypatch.setattr(
        FeatureWorkspace, "prepare_initial", lambda _, *, baseline: initial.append(baseline)
    )
    return {
        "config": config,
        "calls": calls,
        "models": models,
        "initial": initial,
        "prepare_initial": prepare_initial,
    }


@pytest.mark.parametrize("section", FEATURE_SECTIONS)
def test_selected_family_isolated_and_repeated_cell_is_cache_only(
    registered: dict[str, Any], section: FeatureSection
) -> None:
    config = registered["config"]
    cpu_state, python_state, numpy_state = (
        torch.get_rng_state().clone(),
        random.getstate(),
        np.random.get_state(),
    )
    result = compute_feature_sections(config, torch.device("cpu"), progress=False, section=section)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    assert np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    expected = {section, "gradcam", "occlusion"} if section == "diagnostics" else {section}
    assert set(registered["calls"]) == {
        (arm, group) for arm in ("standard", "adversarial") for group in expected
    }
    assert result["status"] == "partial"
    assert not (config.project_path("results") / "feature_visualizations.json").exists()
    assert {figure["section"] for figure in result["figures"]} == expected
    assert all(model.training for model in registered["models"])
    assert all(
        torch.equal(model[0].weight.grad, torch.ones_like(model[0].weight))
        for model in registered["models"]
    )
    counts = (len(registered["calls"]), len(registered["models"]), len(registered["initial"]))
    repeated = compute_feature_sections(
        config, torch.device("cpu"), progress=False, section=section
    )
    assert (
        len(registered["calls"]),
        len(registered["models"]),
        len(registered["initial"]),
    ) == counts
    assert repeated["elapsed_seconds"] == result["elapsed_seconds"]
    validate_feature_evidence(config, repeated, require_complete=False)


def test_complete_manifest_only_after_all_sections_and_hydrates_without_inference(
    registered: dict[str, Any],
) -> None:
    config = registered["config"]
    for section in FEATURE_SECTIONS[:-1]:
        result = compute_feature_sections(
            config, torch.device("cpu"), progress=False, section=section
        )
        assert result["status"] == "partial"
        assert not (config.project_path("results") / "feature_visualizations.json").exists()
    result = compute_feature_sections(
        config, torch.device("cpu"), progress=False, section="diagnostics"
    )
    assert result["status"] == "complete" and result["completion"]["missing"] == []
    assert all(result["arms"][arm]["state_unchanged"] for arm in result["arms"])
    validate_feature_evidence(config, result)
    # Atomic JSON sorts dictionary keys; natural figure order must survive a
    # disk round-trip rather than relying on in-memory registry insertion order.
    saved = json.loads((config.project_path("results") / "feature_visualizations.json").read_text())
    validate_feature_evidence(config, saved)
    calls = len(registered["calls"])
    assert (
        compute_feature_sections(config, torch.device("cpu"), progress=False)["status"]
        == "complete"
    )
    assert len(registered["calls"]) == calls == 16


def test_diagnostics_reuses_existing_prerequisites(registered: dict[str, Any]) -> None:
    config = registered["config"]
    compute_feature_sections(config, torch.device("cpu"), progress=False, section="gradcam")
    registered["calls"].clear()
    compute_feature_sections(config, torch.device("cpu"), progress=False, section="diagnostics")
    assert set(registered["calls"]) == {
        (arm, section)
        for arm in ("standard", "adversarial")
        for section in ("occlusion", "diagnostics")
    }


@pytest.mark.parametrize("change", ["method", "checkpoint", "photograph"])
def test_identity_changes_reject_old_evidence_and_only_compute_requested_family(
    registered: dict[str, Any], change: str
) -> None:
    config = registered["config"]
    result = compute_feature_sections(config, torch.device("cpu"), progress=False, section="stages")
    if change == "method":
        atomic_write_text(config.root / "src/feature_methods.py", "changed synthetic method\n")
    elif change == "checkpoint":
        checkpoint_path(config, "standard").write_bytes(b"changed synthetic checkpoint")
    else:
        Image.new("RGB", (8, 8), "blue").save(config.root / "synthetic.png")
    with pytest.raises(FeatureEvidenceError, match="stale"):
        validate_feature_evidence(config, result, require_complete=False)
    registered["calls"].clear()
    changed = compute_feature_sections(
        config, torch.device("cpu"), progress=False, section="kernels"
    )
    assert changed["identity"] != result["identity"]
    assert set(registered["calls"]) == {("standard", "kernels"), ("adversarial", "kernels")}


@pytest.mark.parametrize("change", ["npz", "ids", "figure", "state"])
def test_invalid_condition_cache_recomputes_only_affected_arm(
    registered: dict[str, Any], change: str
) -> None:
    config = registered["config"]
    result = compute_feature_sections(
        config, torch.device("cpu"), progress=False, section="gradcam"
    )
    path = config.root / result["section_receipts"]["standard"]["gradcam"]["path"]
    receipt = json.loads(path.read_text())
    if change == "figure":
        (config.root / receipt["figures"][0]["path"]).write_bytes(b"changed figure")
    elif change == "state":
        receipt["payload"]["state_unchanged"] = False
        atomic_write_json(path, receipt)
    else:
        arrays_path = config.root / receipt["arrays_path"]
        if change == "npz":
            arrays_path.write_bytes(b"not an array archive")
        else:
            with np.load(arrays_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["__anchor_sample_ids"] = np.asarray(["incorrect"])
            atomic_save_npz(arrays_path, arrays)
            receipt["arrays_sha256"] = sha256_file(arrays_path)
            atomic_write_json(path, receipt)
    with pytest.raises(FeatureEvidenceError):
        validate_feature_evidence(config, result, require_complete=False)
    registered["calls"].clear()
    compute_feature_sections(config, torch.device("cpu"), progress=False, section="gradcam")
    assert registered["calls"] == [("standard", "gradcam")]


def test_partial_manifest_not_accepted_as_complete(registered: dict[str, Any]) -> None:
    result = compute_feature_sections(
        registered["config"], torch.device("cpu"), progress=False, section="occlusion"
    )
    with pytest.raises(FeatureEvidenceError, match="incomplete"):
        validate_feature_evidence(registered["config"], result)


def test_invalid_selector_stops_before_model_construction(registered: dict[str, Any]) -> None:
    with pytest.raises(FeatureEvidenceError, match="unknown"):
        compute_feature_sections(registered["config"], torch.device("cpu"), section="not-a-section")  # type: ignore[arg-type]
    assert registered["models"] == []


def test_interrupted_diagnostics_preserves_completed_prerequisite_progress(
    registered: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import feature_anchor_sections

    original = feature_anchor_sections.compute_anchors

    def interrupted(section: FeatureSection, context: Any) -> SectionData:
        if section == "diagnostics":
            raise RuntimeError("synthetic interruption")
        return original(section, context)

    monkeypatch.setattr(feature_anchor_sections, "compute_anchors", interrupted)
    config = registered["config"]
    with pytest.raises(RuntimeError, match="interruption"):
        compute_feature_sections(config, torch.device("cpu"), progress=False, section="diagnostics")
    saved = json.loads(
        (config.project_path("results") / "feature_visualizations.partial.json").read_text()
    )
    assert set(saved["section_receipts"]["standard"]) == {"gradcam", "occlusion"}
    assert "standard/diagnostics" in saved["completion"]["missing"]
    assert not (config.project_path("results") / "feature_visualizations.json").exists()
    validate_feature_evidence(config, saved, require_complete=False)


def test_read_only_display_filters_saved_section_without_loading_models(
    registered: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import runtime_visuals

    config = registered["config"]
    compute_feature_sections(config, torch.device("cpu"), progress=False, section="kernels")
    compute_feature_sections(config, torch.device("cpu"), progress=False, section="gradcam")
    displayed: list[Any] = []
    monkeypatch.setattr(
        runtime_visuals,
        "_ipython_display",
        lambda: (
            lambda **kwargs: kwargs,
            lambda value: value,
            displayed.append,
        ),
    )
    calls = len(registered["models"])
    runtime_visuals.display_features(config, section="kernels")
    images = [item for item in displayed if isinstance(item, dict)]
    assert len(images) == 2 and all("/kernels/" in item["filename"] for item in images)
    assert len(registered["models"]) == calls


def test_read_only_validation_does_not_create_missing_training_identity(
    registered: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = registered["config"]
    config.raw["training_monitoring"]["enabled"] = True
    result = compute_feature_sections(config, torch.device("cpu"), progress=False, section="stages")
    monkeypatch.setattr(
        "src.feature_sections.experiment_provenance",
        lambda _: pytest.fail("read-only validation attempted registration"),
    )
    with pytest.raises(FeatureEvidenceError, match="read-only"):
        validate_feature_evidence(config, result, require_complete=False)
    assert not (config.project_path("artifacts") / "training").exists()


def test_initial_kernels_are_reusable_without_trained_arm_state_marker(
    registered: dict[str, Any],
) -> None:
    workspace = FeatureWorkspace(registered["config"], torch.device("cpu"), False)
    kernels = np.ones((2, 3, 7, 7), dtype=np.float32)
    workspace._save("initial", "kernels", SectionData(arrays={"kernels": kernels}))
    context = SectionContext(workspace, "standard", "kernels", nn.Linear(2, 2))
    assert torch.equal(context.initial_kernels(), torch.from_numpy(kernels))
    assert context.dependencies[0]["path"].endswith("initial/kernels.json")


def test_initial_kernel_preparation_builds_only_once_and_never_probes(
    registered: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    builds: list[nn.Module] = []

    def build(_: ExperimentConfig) -> nn.Module:
        model = nn.Module()
        model.network = nn.Module()
        model.network.conv1 = nn.Conv2d(3, 2, 7)
        builds.append(model)
        return model

    monkeypatch.setattr("src.feature_sections.build_model", build)
    monkeypatch.setattr(
        "src.feature_visualization._probe", lambda *_: pytest.fail("unnecessary initial probe")
    )
    workspace = FeatureWorkspace(registered["config"], torch.device("cpu"), False)
    registered["prepare_initial"](workspace, baseline=False)
    registered["prepare_initial"](workspace, baseline=False)
    assert len(builds) == 1
    context = SectionContext(workspace, "standard", "kernels", nn.Linear(2, 2))
    assert context.initial_kernels().shape == (2, 3, 7, 7)


def test_corrupted_prerequisite_arrays_invalidate_parent_even_with_intact_receipt(
    registered: dict[str, Any],
) -> None:
    config = registered["config"]
    workspace = FeatureWorkspace(config, torch.device("cpu"), False)
    workspace._save("standard", "probe", SectionData(arrays={"synthetic_probe": np.ones(3)}))
    prerequisite = workspace._path("standard", "probe")
    workspace._save(
        "standard",
        "synthetic",
        SectionData(payload={"state_unchanged": True}, arrays={"response": np.ones(2)}),
        dependencies=[
            {
                "path": str(prerequisite.relative_to(config.root)),
                "sha256": sha256_file(prerequisite),
            }
        ],
    )
    assert workspace._load("standard", "synthetic") is not None
    prerequisite.with_suffix(".npz").write_bytes(b"damaged prerequisite measurements")
    assert workspace._load("standard", "synthetic") is None
    assert "standard/synthetic" in workspace.manifest()["completion"]["missing"]


def test_circular_cache_dependencies_are_rejected(registered: dict[str, Any]) -> None:
    workspace = FeatureWorkspace(registered["config"], torch.device("cpu"), False)
    workspace._save("standard", "probe", SectionData(arrays={"synthetic": np.ones(1)}))
    path = workspace._path("standard", "probe")
    receipt = json.loads(path.read_text())
    # A self-reference cannot have a matching content hash; either way, it
    # must be treated as invalid evidence rather than followed indefinitely.
    receipt["dependencies"] = [
        {"path": str(path.relative_to(workspace.config.root)), "sha256": sha256_file(path)}
    ]
    atomic_write_json(path, receipt)
    assert workspace._load("standard", "probe") is None


def test_mps_random_state_restored_after_failure_without_hardware_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mps_state = torch.tensor([7, 11], dtype=torch.uint8)
    restored: list[torch.Tensor] = []
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: mps_state.clone())
    monkeypatch.setattr(torch.mps, "set_rng_state", restored.append)
    cpu_state = torch.get_rng_state().clone()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with (
        pytest.raises(RuntimeError, match="synthetic interruption"),
        _preserved_randomness(torch.device("mps")),
    ):
        random.random()
        np.random.random()
        torch.rand(3)
        raise RuntimeError("synthetic interruption")
    assert len(restored) == 1 and torch.equal(restored[0], mps_state)
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])


def test_assembled_features_generate_report_and_read_only_notebook_without_evaluation(
    registered: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import nbformat

    from src.reporting import report

    config = registered["config"]
    compute_feature_sections(config, torch.device("cpu"), progress=False)
    calls = len(registered["models"])

    def forbidden(*_: Any, **__: Any) -> None:
        pytest.fail("feature reporting must not run inference, attacks or confidence fitting")

    monkeypatch.setattr("src.feature_sections.load_checkpoint_model", forbidden)
    monkeypatch.setattr("src.evaluation.evaluate", forbidden)
    monkeypatch.setattr("src.calibration.fit_temperature", forbidden)
    monkeypatch.setattr("src.attacks.pgd_attack", forbidden)
    summary = report(config, progress=False)
    assert summary["feature_diagnostics"]["population_accuracy_measured"] is False
    assert len(registered["models"]) == calls
    assert not (config.project_path("results") / "evaluation.json").exists()
    notebook = nbformat.read(
        config.project_path("reports") / "cat_dog_feature_results.ipynb", as_version=4
    )
    assert all(cell.cell_type == "markdown" for cell in notebook.cells)
    assert notebook.metadata.evidence.required_feature_sections == list(FEATURE_SECTIONS)
    assert sum(len(cell.get("attachments", {})) for cell in notebook.cells) == 16
    release = json.loads(Path(summary["local_release_manifest"]).read_text())
    feature_manifest = json.loads(
        (config.project_path("results") / "feature_visualizations.json").read_text()
    )
    for registry in feature_manifest["section_receipts"].values():
        for registered_receipt in registry.values():
            receipt = json.loads((config.root / registered_receipt["path"]).read_text())
            assert release["files"][receipt["arrays_path"]] == receipt["arrays_sha256"]
