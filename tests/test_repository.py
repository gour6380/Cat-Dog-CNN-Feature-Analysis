from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import check_repository
from src.notebook_support import ROOT


def test_repository_candidate_passes_read_only_review() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_repository.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Repository candidate passed" in result.stdout


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


def test_source_notebook_is_the_only_distributable_notebook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    allowed = check_repository.ALLOWED_NOTEBOOKS
    assert allowed == {Path("notebooks/cat_dog_cnn_features.ipynb")}
    notebook = tmp_path / "notebooks/cat_dog_cnn_features.ipynb"
    notebook.parent.mkdir()
    content: dict[str, Any] = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["RUN_FULL_EXPERIMENT = False"],
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
