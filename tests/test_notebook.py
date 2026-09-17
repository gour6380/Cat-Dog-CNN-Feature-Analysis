from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import nbformat
import pytest

from scripts import build_notebook
from src.config import load_config
from src.io_utils import source_hash
from src.notebook_runner import CallingKernelSpecs
from src.notebook_support import (
    ROOT,
    NotebookStage,
    notebook_context,
    require_project_environment,
    run_stage,
)
from src.progress import NotebookProgress, tqdm

CONFIG = ROOT / "configs" / "experiment.yaml"
NOTEBOOK = ROOT / "notebooks" / "cat_dog_cnn_features.ipynb"


def _public_gallery_manifest(tmp_path: Path) -> tuple[dict[str, Any], Path]:
    records = [
        ("experiment-workflow.png", "species_response", "experiment"),
        ("standard-synthetic-atlas.png", "activation_maximization", "standard"),
        ("adversarial-synthetic-atlas.png", "activation_maximization", "adversarial"),
        ("experiment-accuracy-context.png", "species_response", "experiment"),
        ("standard-kernels.png", "kernels", "standard"),
        ("adversarial-kernels.png", "kernels", "adversarial"),
        ("standard-species-response.png", "species_response", "standard"),
        ("adversarial-species-response.png", "species_response", "adversarial"),
    ]
    figures: list[dict[str, str]] = []
    assets: dict[str, str] = {}
    asset_root = tmp_path / "docs/assets"
    asset_root.mkdir(parents=True)
    for name, kind, arm in records:
        path = asset_root / name
        path.write_bytes(name.encode())
        relative = path.relative_to(tmp_path).as_posix()
        figures.append({"path": relative, "kind": kind, "arm": arm, "caption": f"Saved {name}"})
        assets[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "figures": figures,
        "provenance": {"configuration_sha256": "current", "public_assets": assets},
    }
    manifest_path = tmp_path / "docs/results.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, manifest_path


def test_public_gallery_links_compact_matching_synthetic_assets(tmp_path: Path) -> None:
    _public_gallery_manifest(tmp_path)
    cells = build_notebook._public_preview_cells(tmp_path, config_sha256="current")
    assert len(cells) == 7  # Intro plus workflow, two atlases, accuracy context, two kernels.
    assert (
        "not a dead channel" in cells[0].source and "failed trials are retained" in cells[0].source
    )
    assert all(cell.cell_type == "markdown" and not cell.get("attachments") for cell in cells)
    assert all("../docs/assets/" in cell.source for cell in cells[1:])
    assert "standard-synthetic-atlas.png" in cells[2].source
    assert "adversarial-synthetic-atlas.png" in cells[3].source
    assert not any("species-response.png" in cell.source for cell in cells)
    assert [cell.id for cell in cells] == [
        cell.id for cell in build_notebook._public_preview_cells(tmp_path, config_sha256="current")
    ]


@pytest.mark.parametrize(
    "invalid_kind", ["stale-config", "unsafe-kind", "altered-hash", "private-path", "pending"]
)
def test_public_gallery_omits_unverified_or_pending_evidence(
    tmp_path: Path, invalid_kind: str
) -> None:
    manifest, path = _public_gallery_manifest(tmp_path)
    manifest["figures"] = [manifest["figures"][1]]
    if invalid_kind == "stale-config":
        manifest["provenance"]["configuration_sha256"] = "old-breed-config"
    elif invalid_kind == "unsafe-kind":
        manifest["figures"][0]["kind"] = "top_real_patches"
    elif invalid_kind == "altered-hash":
        (tmp_path / manifest["figures"][0]["path"]).write_bytes(b"altered asset bytes")
    elif invalid_kind == "private-path":
        private_path = tmp_path / "figures/private-photo.png"
        private_path.parent.mkdir()
        private_path.write_bytes(b"local photograph")
        manifest["figures"][0]["path"] = "figures/private-photo.png"
        manifest["provenance"]["public_assets"]["figures/private-photo.png"] = hashlib.sha256(
            private_path.read_bytes()
        ).hexdigest()
    else:
        manifest["figures"] = []
        manifest["provenance"]["public_assets"] = {}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert build_notebook._public_preview_cells(tmp_path, config_sha256="current") == []


def test_notebook_gallery_adds_only_markdown_and_preserves_safe_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _public_gallery_manifest(tmp_path)
    gallery = build_notebook._public_preview_cells(tmp_path, config_sha256="current")
    monkeypatch.setattr(build_notebook, "_public_preview_cells", lambda: gallery)
    notebook = build_notebook.make_notebook()
    base = build_notebook.make_notebook(include_public_figures=False)
    assert notebook.cells[: len(base.cells)] == base.cells
    assert notebook.cells[len(base.cells) :] == gallery
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert "RUN_FULL_EXPERIMENT = False" in code[0].source
    assert all(not cell.outputs and cell.execution_count is None for cell in code)
    nbformat.validate(notebook)


def test_source_notebook_is_valid_ordered_and_output_free() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call]
    nbformat.validate(notebook)
    headings = [
        line.strip()
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("## ")
    ]
    assert headings == [
        "## 1. Question and claim boundary",
        "## 2. Editable protocol and matched comparison",
        "## 3. Environment and native PyTorch MPS",
        "## 4. Dataset and EDA",
        "## 5. Model depth and spatial features",
        "## 6. Native MPS preflight",
        "## 7. Standard fine-tuning",
        "## 8. PGD-5 adversarial fine-tuning",
        "## 9. Attacks, corruptions, and risk-aware evaluation",
        "## 10. Learned filters, real image parts, and attribution",
        "## 11. Reports, interpretation, and local artifact inventory",
        "## 12. Reproduction and references",
    ]
    assert notebook.metadata.kernelspec.name == "oxford-pets-adversarial-representations"
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert "RUN_FULL_EXPERIMENT = False" in code_cells[0].source
    assert all(cell.execution_count is None and cell.outputs == [] for cell in code_cells)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "run_stage(" in source
    assert "generate_eda(config)" in source
    assert 'for figure_key, relative_path in eda["figures"].items()' in source
    assert "breed_balance" not in source
    assert "feature_visualizations.json" in source
    assert 'figure["path"]' in source
    assert "Grad-CAM and occlusion" in source
    assert "true-species logit" in source
    assert "true-species-vs-other logit margin" in source
    assert "two cats and two dogs" in source
    assert "cat=0, dog=1" in source
    assert "figures/generated/projections/" not in source
    assert "primary_hypothesis" not in source
    assert "37-class head" not in source
    assert "def build_model" not in source
    assert "def pgd" not in source
    assert "optimizer.step" not in source


def test_feature_cell_reads_matching_manifest_and_dynamic_image_paths(tmp_path: Path) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call]
    cell = next(cell for cell in notebook.cells if cell.id == "run-features")
    results = tmp_path / "results" / "generated"
    results.mkdir(parents=True)
    image = tmp_path / "figures" / "custom-channel.png"
    image.parent.mkdir()
    image.write_bytes(b"image contents are not read by the mocked display")
    (results / "feature_visualizations.json").write_text(
        json.dumps(
            {
                "config_sha256": "matching-config",
                "created_at": "saved-evidence-time",
                "arms": {},
                "figures": [
                    {
                        "path": "figures/custom-channel.png",
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "caption": "A dynamically selected feature",
                        "shareable": False,
                        "arm": "standard",
                        "kind": "activation map",
                    }
                ],
                "limitations": ["local sensitivity is not causal learning"],
            }
        ),
        encoding="utf-8",
    )
    displays: list[object] = []
    calls: list[tuple[str, bool]] = []

    def stage(_: object, name: str, *, full: bool, device: str) -> dict[str, object]:
        assert device == "mps"
        calls.append((name, full))
        return {"results_available": False}

    namespace: dict[str, Any] = {
        "ROOT": tmp_path,
        "config": SimpleNamespace(sha256="matching-config"),
        "RUN_FULL_EXPERIMENT": False,
        "device": "mps",
        "run_stage": stage,
        "json": json,
        "show": displays.append,
        "display": displays.append,
        "Markdown": lambda value: value,
        "Image": lambda *, filename, width: {"filename": filename, "width": width},
    }
    exec(compile(cell.source, "feature-notebook-cell", "exec"), namespace)
    assert calls == [("represent", False)]
    assert {"filename": str(image.resolve()), "width": 1200} in displays
    assert any("local-only" in str(item) for item in displays)
    assert not any("pca" in str(item).lower() for item in displays)


@pytest.mark.parametrize("stale_kind", ["altered-bytes", "missing-file", "missing-hash"])
def test_feature_cell_does_not_display_stale_image_bytes(tmp_path: Path, stale_kind: str) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call]
    cell = next(cell for cell in notebook.cells if cell.id == "run-features")
    results = tmp_path / "results/generated"
    results.mkdir(parents=True)
    image = tmp_path / "figures/channel.png"
    image.parent.mkdir()
    image.write_bytes(b"verified original image")
    figure: dict[str, Any] = {
        "path": "figures/channel.png",
        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "caption": "Saved feature",
        "shareable": False,
        "arm": "standard",
        "kind": "activation map",
    }
    if stale_kind == "altered-bytes":
        image.write_bytes(b"unverified replacement image")
    elif stale_kind == "missing-file":
        image.unlink()
    else:
        del figure["sha256"]
    (results / "feature_visualizations.json").write_text(
        json.dumps(
            {
                "config_sha256": "matching-config",
                "created_at": "saved-time",
                "figures": [figure],
                "limitations": [],
            }
        ),
        encoding="utf-8",
    )
    messages: list[object] = []
    displays: list[object] = []
    namespace: dict[str, Any] = {
        "ROOT": tmp_path,
        "config": SimpleNamespace(sha256="matching-config"),
        "RUN_FULL_EXPERIMENT": False,
        "device": "mps",
        "run_stage": lambda *args, **kwargs: {"results_available": False},
        "json": json,
        "show": messages.append,
        "display": displays.append,
        "Markdown": lambda value: value,
        "Image": lambda *, filename, width: {"filename": filename, "width": width},
    }
    exec(compile(cell.source, "feature-notebook-cell", "exec"), namespace)
    stale_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("status") == "stale feature evidence"
    ]
    assert len(stale_messages) == 1
    assert stale_messages[0]["figure"] == "figures/channel.png"
    assert not any(isinstance(value, dict) and "filename" in value for value in displays)


def test_feature_cell_rejects_stale_evidence_without_running_model(tmp_path: Path) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call]
    cell = next(cell for cell in notebook.cells if cell.id == "run-features")
    results = tmp_path / "results" / "generated"
    results.mkdir(parents=True)
    (results / "feature_visualizations.json").write_text(
        json.dumps({"config_sha256": "superseded-breed-config", "figures": []}),
        encoding="utf-8",
    )
    namespace: dict[str, Any] = {
        "ROOT": tmp_path,
        "config": SimpleNamespace(sha256="new-species-config"),
        "RUN_FULL_EXPERIMENT": False,
        "device": "mps",
        "run_stage": lambda *args, **kwargs: {"results_available": False},
        "json": json,
        "show": lambda value: None,
    }
    with pytest.raises(RuntimeError, match="different configuration"):
        exec(compile(cell.source, "feature-notebook-cell", "exec"), namespace)


def test_feature_cell_without_saved_evidence_is_read_only_and_explicit(tmp_path: Path) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)  # type: ignore[no-untyped-call]
    cell = next(cell for cell in notebook.cells if cell.id == "run-features")
    messages: list[object] = []
    calls: list[tuple[str, bool]] = []
    unavailable = {
        "stage": "represent",
        "status": "skipped in safe mode",
        "results_available": False,
    }

    def stage(_: object, name: str, *, full: bool, device: str) -> dict[str, object]:
        assert device == "mps"
        calls.append((name, full))
        return unavailable

    namespace: dict[str, Any] = {
        "ROOT": tmp_path,
        "config": SimpleNamespace(sha256="new-species-config"),
        "RUN_FULL_EXPERIMENT": False,
        "device": "mps",
        "run_stage": stage,
        "json": json,
        "show": messages.append,
    }
    exec(compile(cell.source, "feature-notebook-cell", "exec"), namespace)
    assert messages == [unavailable]
    assert calls == [("represent", False)]
    assert list(tmp_path.iterdir()) == []


def test_readme_project_links_are_repository_local_and_resolve() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    assert targets
    for target in targets:
        assert "://" not in target
        assert not target.startswith(("/", ".."))
        local = target.split("#", maxsplit=1)[0]
        assert (ROOT / local).exists(), target
    assert "Instructions/checklist.md" not in readme
    assert "/Users/" not in readme


def test_safe_mode_never_invokes_a_scientific_stage() -> None:
    config = load_config(CONFIG)
    stages: list[tuple[NotebookStage, str | None]] = [
        ("preflight", None),
        ("train", "standard"),
        ("train", "adversarial"),
        ("evaluate", None),
        ("represent", None),
        ("report", None),
    ]
    for stage, arm in stages:
        result = run_stage(
            config,
            stage,
            full=False,
            device="mps",
            arm=cast(Any, arm),
            progress=False,
        )
        assert isinstance(result, dict)
        assert result == {
            "stage": stage,
            "status": "skipped in safe mode",
            "results_available": False,
        }


def test_notebook_override_and_interpreter_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("src.notebook_support.sys.prefix", str(ROOT / ".venv"))
    monkeypatch.setenv("OXFORD_PETS_NOTEBOOK_RUN_FULL", "0")
    assert notebook_context(full=True)[1] is False
    monkeypatch.setenv("OXFORD_PETS_NOTEBOOK_RUN_FULL", "1")
    assert notebook_context(full=False)[1] is True
    monkeypatch.delenv("OXFORD_PETS_NOTEBOOK_RUN_FULL")
    assert notebook_context(full=True)[1] is True
    assert notebook_context()[1] is False

    monkeypatch.setattr("src.notebook_support.sys.prefix", str(tmp_path / "sibling/.venv"))
    with pytest.raises(RuntimeError, match="Wrong notebook interpreter"):
        require_project_environment()


def test_headless_runner_uses_calling_interpreter() -> None:
    specification = CallingKernelSpecs().get_kernel_spec("ignored")
    assert specification.argv[0] == sys.executable
    assert specification.argv[1:3] == ["-m", "ipykernel_launcher"]


def test_headless_runner_is_a_direct_script_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "src/notebook_runner.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--full" in result.stdout


def test_nested_notebook_progress_has_no_terminal_controls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rendered: list[str] = []

    class Handle:
        def display(self, value: dict[str, str], **_: Any) -> None:
            rendered.append(value["text/html"])

        def update(self, value: dict[str, str], **_: Any) -> None:
            rendered.append(value["text/html"])

    monkeypatch.setattr("src.progress.in_notebook", lambda: True)
    monkeypatch.setattr("src.progress.DisplayHandle", Handle)
    with tqdm(total=2, desc="standard epochs", mininterval=0) as epochs:
        for epoch in range(2):
            for _ in tqdm(range(3), desc=f"epoch {epoch + 1}", leave=False, mininterval=0):
                pass
            epochs.update()
            epochs.set_postfix(loss="0.5", selected=epoch + 1)
    assert isinstance(epochs, NotebookProgress)
    assert epochs.n == 2
    assert any('aria-label="standard epochs"' in html and 'value="2"' in html for html in rendered)
    assert any("selected=2" in html for html in rendered)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
    assert all("\x1b" not in html and "\r" not in html for html in rendered)


def test_notebook_stack_and_kernel_registration_are_pinned() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "jupyterlab==4.5.0" in requirements
    assert "ipykernel==7.0.1" in requirements
    assert "nbclient==0.10.2" in requirements
    assert "nbformat==5.10.4" in requirements
    setup = (ROOT / "setup_venv.sh").read_text(encoding="utf-8")
    assert "oxford-pets-adversarial-representations" in setup
    assert "ipykernel install" in setup


def test_source_hash_includes_notebook_sources(tmp_path: Path) -> None:
    notebook = tmp_path / "guide.ipynb"
    notebook.write_text('{"cells": []}', encoding="utf-8")
    before = source_hash(tmp_path)
    notebook.write_text('{"cells": [{"cell_type": "markdown"}]}', encoding="utf-8")
    assert source_hash(tmp_path) != before
