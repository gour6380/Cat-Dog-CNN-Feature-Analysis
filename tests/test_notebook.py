from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import nbformat
import pytest

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
NOTEBOOK = ROOT / "notebooks" / "oxford_pets_adversarial_representations.ipynb"


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
        "## 1. Question, hypotheses, and claim boundary",
        "## 2. Locked protocol",
        "## 3. Environment and native MPS",
        "## 4. Data integrity and protocol-neutral EDA",
        "## 5. Model and feature interface",
        "## 6. Native MPS preflight",
        "## 7. Standard fine-tuning",
        "## 8. PGD-5 adversarial fine-tuning",
        "## 9. Evaluation, calibration, and input shifts",
        "## 10. Original-feature geometry and explanatory projections",
        "## 11. Reports, limitations, and artifact inventory",
        "## 12. Reproduction",
    ]
    assert notebook.metadata.kernelspec.name == "oxford-pets-adversarial-representations"
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert "RUN_FULL_EXPERIMENT = False" in code_cells[0].source
    assert all(cell.execution_count is None and cell.outputs == [] for cell in code_cells)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "run_stage(" in source
    assert "generate_eda(config)" in source
    assert '"split_and_species", "breed_balance", "image_geometry"' in source
    assert "def build_model" not in source
    assert "def pgd" not in source
    assert "optimizer.step" not in source


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
