from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import nbformat
import pytest

from scripts import build_notebook, check_repository
from src.config import ExperimentConfig
from src.notebook_support import ROOT


def test_repository_candidate_passes_read_only_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Review the generated shareable source in an isolated candidate. The owner
    # may have private executed pictures in the working notebook during review;
    # the public gate must still reject those, not erase them as a test side effect.
    paths = check_repository._candidate_paths()
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == Path("notebooks/cat_dog_cnn_features.ipynb"):
            nbformat.write(build_notebook.make_notebook(run_full=True), destination)
        else:
            shutil.copyfile(ROOT / relative, destination)
    monkeypatch.setattr(check_repository, "ROOT", tmp_path)
    monkeypatch.setattr(check_repository, "_candidate_paths", lambda: paths)
    assert check_repository.review() == 0
    assert "Repository candidate passed" in capsys.readouterr().out


def test_deleted_tracked_files_are_not_candidate_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "kept.md").write_text("kept", encoding="utf-8")
    monkeypatch.setattr(check_repository, "ROOT", tmp_path)
    monkeypatch.setattr(
        check_repository.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"kept.md\0deleted-old-notebook.ipynb\0"
        ),
    )
    assert check_repository._candidate_paths() == [Path("kept.md")]


def _public_manifest(tmp_path: Path, *, kind: str = "activation_maximization") -> Path:
    asset = tmp_path / "docs/assets/synthetic-channel.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"synthetic picture placeholder")
    (tmp_path / "docs/results.json").write_text(
        json.dumps(
            {
                "protocol": {"label_mode": "species"},
                "figures": [
                    {
                        "path": "docs/assets/synthetic-channel.png",
                        "kind": kind,
                        "arm": "standard",
                    }
                ],
                "provenance": {
                    "configuration_sha256": "current-config",
                    "public_assets": {
                        "docs/assets/synthetic-channel.png": hashlib.sha256(
                            asset.read_bytes()
                        ).hexdigest()
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return asset


def test_public_assets_require_safe_kind_and_matching_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset = _public_manifest(tmp_path)
    monkeypatch.setattr(check_repository, "ROOT", tmp_path)
    monkeypatch.setattr(check_repository, "_current_config_sha256", lambda: "current-config")
    paths = [asset.relative_to(tmp_path)]
    errors: list[str] = []
    check_repository._check_public_assets(paths, errors)
    assert errors == []
    asset.write_bytes(b"replaced figure")
    check_repository._check_public_assets(paths, errors)
    assert any("hash differs" in error for error in errors)


def test_photo_figure_cannot_be_whitelisted_as_public_asset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset = _public_manifest(tmp_path, kind="top_real_patches")
    monkeypatch.setattr(check_repository, "ROOT", tmp_path)
    monkeypatch.setattr(check_repository, "_current_config_sha256", lambda: "current-config")
    errors: list[str] = []
    check_repository._check_public_assets([asset.relative_to(tmp_path)], errors)
    assert any("not shareable" in error for error in errors)


def test_stale_public_asset_is_flagged_even_when_other_figures_are_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset = _public_manifest(tmp_path)
    stale = tmp_path / "docs/assets/old-breed-drift.png"
    stale.write_bytes(b"old evidence")
    monkeypatch.setattr(check_repository, "ROOT", tmp_path)
    monkeypatch.setattr(check_repository, "_current_config_sha256", lambda: "current-config")
    errors: list[str] = []
    check_repository._check_public_assets(
        [asset.relative_to(tmp_path), stale.relative_to(tmp_path)], errors
    )
    assert any("old-breed-drift.png: stale or unmanifested" in error for error in errors)


def _historical_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, kind: str = "activation_maximization"
) -> tuple[Path, dict[str, Any]]:
    asset = _public_manifest(tmp_path, kind=kind)
    path = tmp_path / "docs/results.json"
    manifest = json.loads(path.read_text())
    original_sha = next(iter(check_repository.HISTORICAL_PUBLIC_STUDIES))
    registration = check_repository.HISTORICAL_PUBLIC_STUDIES[original_sha]
    manifest["provenance"]["configuration_sha256"] = original_sha
    manifest["historical_evidence"] = {
        **registration,
        "status": "archived_completed_experiment",
        "configuration_sha256": original_sha,
        "current_run_results_available": False,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    archived_path = tmp_path / registration["configuration_path"]
    archived_path.parent.mkdir(exist_ok=True)
    archived_path.write_text("synthetic archived protocol fixture", encoding="utf-8")
    archived = ExperimentConfig(
        archived_path,
        tmp_path,
        {"experiment": {"id": registration["experiment_id"]}},
        original_sha,
    )
    monkeypatch.setattr(check_repository, "ROOT", tmp_path)
    monkeypatch.setattr(check_repository, "_current_config_sha256", lambda: "fresh-v3-config")
    monkeypatch.setattr(
        check_repository, "_current_experiment_id", lambda: registration["superseded_by"]
    )
    monkeypatch.setattr(check_repository, "load_config", lambda *args, **kwargs: archived)
    return asset, manifest


def test_registered_historical_assets_are_explicitly_distinct_from_fresh_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset, manifest = _historical_manifest(monkeypatch, tmp_path)
    errors: list[str] = []
    check_repository._check_public_assets([asset.relative_to(tmp_path)], errors)
    assert errors == []
    assert not manifest["historical_evidence"]["current_run_results_available"]


def test_unmarked_superseded_public_evidence_is_still_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset, manifest = _historical_manifest(monkeypatch, tmp_path)
    del manifest["historical_evidence"]
    (tmp_path / "docs/results.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors: list[str] = []
    check_repository._check_public_assets([asset.relative_to(tmp_path)], errors)
    assert any("superseded configuration" in error for error in errors)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_revision", "unknown-source"),
        ("configuration_path", "configs/unregistered.yaml"),
        ("experiment_id", "unregistered-experiment"),
        ("superseded_by", "another-experiment"),
        ("configuration_sha256", "different-original-config"),
        ("status", "fresh_current_results"),
        ("current_run_results_available", True),
    ],
)
def test_historical_label_cannot_bypass_registered_protocol_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: object
) -> None:
    asset, manifest = _historical_manifest(monkeypatch, tmp_path)
    manifest["historical_evidence"][field] = value
    (tmp_path / "docs/results.json").write_text(json.dumps(manifest), encoding="utf-8")
    errors: list[str] = []
    check_repository._check_public_assets([asset.relative_to(tmp_path)], errors)
    assert errors


def test_changed_archived_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset, manifest = _historical_manifest(monkeypatch, tmp_path)
    registration = manifest["historical_evidence"]
    modified = ExperimentConfig(
        tmp_path / registration["configuration_path"],
        tmp_path,
        {"experiment": {"id": registration["experiment_id"]}},
        "modified-archive",
    )
    monkeypatch.setattr(check_repository, "load_config", lambda *args, **kwargs: modified)
    errors: list[str] = []
    check_repository._check_public_assets([asset.relative_to(tmp_path)], errors)
    assert any("archived configuration identity differs" in error for error in errors)


def test_history_cannot_claim_a_different_current_experiment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset, _manifest = _historical_manifest(monkeypatch, tmp_path)
    monkeypatch.setattr(check_repository, "_current_experiment_id", lambda: "unregistered-v4")
    errors: list[str] = []
    check_repository._check_public_assets([asset.relative_to(tmp_path)], errors)
    assert any("different current experiment" in error for error in errors)


@pytest.mark.parametrize("kind", ["stage_walkthrough", "class_attribution", "real_patches"])
def test_historical_evidence_does_not_allow_local_photographic_figures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    asset, _manifest = _historical_manifest(monkeypatch, tmp_path, kind=kind)
    errors: list[str] = []
    check_repository._check_public_assets([asset.relative_to(tmp_path)], errors)
    assert any("not shareable" in error for error in errors)


def test_historical_assets_still_require_bytes_and_no_unmanifested_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset, _manifest = _historical_manifest(monkeypatch, tmp_path)
    asset.write_bytes(b"replaced historical figure")
    unmanifested = tmp_path / "docs/assets/unknown.png"
    unmanifested.write_bytes(b"unreviewed evidence")
    errors: list[str] = []
    check_repository._check_public_assets(
        [asset.relative_to(tmp_path), unmanifested.relative_to(tmp_path)], errors
    )
    assert any("hash differs" in error for error in errors)
    assert any("stale or unmanifested" in error for error in errors)


@pytest.mark.parametrize("run_full", [False, True])
def test_source_notebook_is_the_only_distributable_notebook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_full: bool
) -> None:
    allowed = check_repository.ALLOWED_NOTEBOOKS
    assert allowed == {Path("notebooks/cat_dog_cnn_features.ipynb")}
    notebook = tmp_path / "notebooks/cat_dog_cnn_features.ipynb"
    notebook.parent.mkdir()
    content: dict[str, Any] = {
        "cells": [
            {
                "cell_type": "code",
                "source": [f"RUN_FULL_EXPERIMENT = {run_full}"],
                "execution_count": None,
                "outputs": [],
            }
        ]
    }
    notebook.write_text(json.dumps(content), encoding="utf-8")
    monkeypatch.setattr(check_repository, "ROOT", tmp_path)
    errors: list[str] = []
    check_repository._check_notebooks(errors)
    assert errors == []
    content["cells"].append(
        {
            "cell_type": "markdown",
            "source": ["Local pet photo"],
            "attachments": {"pet.png": {"image/png": "photo"}},
        }
    )
    notebook.write_text(json.dumps(content), encoding="utf-8")
    check_repository._check_notebooks(errors)
    assert any("image attachments" in error for error in errors)


@pytest.mark.parametrize(
    "source",
    [
        "unrelated_setting = True",
        "# RUN_FULL_EXPERIMENT = False\nunrelated_setting = True",
        "RUN_FULL_EXPERIMENT = 1",
        "RUN_FULL_EXPERIMENT = bool('1')",
        "RUN_FULL_EXPERIMENT = None",
    ],
)
def test_source_guide_requires_an_explicit_literal_boolean_execution_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str
) -> None:
    notebook = tmp_path / "notebooks/cat_dog_cnn_features.ipynb"
    notebook.parent.mkdir()
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": [source],
                        "execution_count": None,
                        "outputs": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(check_repository, "ROOT", tmp_path)
    errors: list[str] = []
    check_repository._check_notebooks(errors)
    assert any("requires literal RUN_FULL_EXPERIMENT" in error for error in errors)


@pytest.mark.parametrize(
    "field,value", [("outputs", [{"output_type": "stream"}]), ("execution_count", 1)]
)
def test_full_true_does_not_permit_executed_notebook_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: object
) -> None:
    notebook = tmp_path / "notebooks/cat_dog_cnn_features.ipynb"
    notebook.parent.mkdir()
    cell = {
        "cell_type": "code",
        "source": ["RUN_FULL_EXPERIMENT = True"],
        "execution_count": None,
        "outputs": [],
        field: value,
    }
    notebook.write_text(json.dumps({"cells": [cell]}), encoding="utf-8")
    monkeypatch.setattr(check_repository, "ROOT", tmp_path)
    errors: list[str] = []
    check_repository._check_notebooks(errors)
    assert any("contains execution output" in error for error in errors)
