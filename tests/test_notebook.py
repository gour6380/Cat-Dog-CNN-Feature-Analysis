from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import nbformat
import pytest
from tqdm.std import tqdm as TerminalProgress

from scripts import build_notebook
from src.config import load_config
from src.feature_section_types import FEATURE_SECTIONS
from src.io_utils import source_hash
from src.notebook_runner import CallingKernelSpecs
from src.notebook_support import ROOT, notebook_context, require_project_environment, run_stage
from src.progress import NotebookProgress, tqdm

CONFIG = ROOT / "configs/experiment.yaml"
NOTEBOOK = ROOT / "notebooks/cat_dog_cnn_features.ipynb"


def _cell(identifier: str) -> Any:
    # Presentation tests exercise generated cells, not the owner's saved execution.
    notebook = build_notebook.make_notebook(run_full=True)
    return next(cell for cell in notebook.cells if cell.id == identifier)


def test_generated_notebook_is_compact_valid_ordered_and_output_free() -> None:
    notebook = build_notebook.make_notebook(run_full=True)
    nbformat.validate(notebook)
    headings = [
        line
        for cell in notebook.cells
        if cell.cell_type == "markdown"
        for line in cell.source.splitlines()
        if line.startswith("## ")
    ]
    assert headings == [
        "## 1. Setup and data",
        "## 2. Training",
        "## 3. Learned features",
        "## 4. Prediction influence",
        "## 5. Current-run results",
    ]
    assert notebook.metadata.kernelspec.name == "oxford-pets-adversarial-representations"
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert "RUN_FULL_EXPERIMENT = True" in code[0].source
    assert code[0].metadata.tags == ["parameters"]
    assert all(cell.execution_count is None and cell.outputs == [] for cell in code)
    assert all(not cell.get("attachments") for cell in notebook.cells)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "show(" not in source and "config.raw" not in source
    assert "display_evaluation" not in source
    assert "../docs/assets/" not in source and "docs/results.json" not in source
    assert not any(cell.id.startswith("public-") for cell in notebook.cells)
    assert "cat=0, dog=1" in source and "two cats and two dogs" in source
    assert "true-species logit" in source and "true-species-vs-other logit margin" in source
    assert "exact PGD" not in source and "PGD-20" not in source and "FGSM" not in source
    assert "optimizer.step" not in source and "def pgd" not in source


def test_owner_notebook_remains_valid_with_or_without_saved_execution() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata.kernelspec.name == "oxford-pets-adversarial-representations"
    assert next(cell for cell in notebook.cells if cell.id == "setup").metadata.tags == [
        "parameters"
    ]
    identifiers = [cell.id for cell in notebook.cells]
    assert len(identifiers) == len(set(identifiers))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.execution_count is None or isinstance(cell.execution_count, int)
            assert isinstance(cell.outputs, list)


def test_generated_explanations_are_plain_and_avoid_permanent_pending_claim() -> None:
    notebook = build_notebook.make_notebook()
    text = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    assert (
        "Fresh results are pending" not in text
        and "Keep this tracked guide output-free" not in text
    )
    for phrase in (
        "small pattern tester",
        "512 summary numbers",
        "not trained from scratch",
        "does not mean the filter is useless",
        "which tiny pixel changes",
        "Red/positive",
        "Blue/negative",
        "correlation undefined",
        "first fixed cat and first fixed dog",
        "All four anchors",
    ):
        assert phrase in text


def test_factory_is_safe_by_default_and_never_reads_historical_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_notebook, "ROOT", Path("/nonexistent-historical-checkout"), raising=True
    )
    notebook = build_notebook.make_notebook()
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert "RUN_FULL_EXPERIMENT = False" in code[0].source
    assert all(cell.outputs == [] and cell.execution_count is None for cell in code)
    assert not hasattr(build_notebook, "_public_preview_cells")


@pytest.mark.parametrize("run_full", [False, True])
def test_builder_preserves_owner_parameters_kernel_and_clears_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, run_full: bool
) -> None:
    path = tmp_path / "notebooks/cat_dog_cnn_features.ipynb"
    path.parent.mkdir()
    existing = build_notebook.make_notebook(run_full=run_full)
    existing.metadata.kernelspec.display_name = "Owner's selected kernel"
    existing.metadata.language_info.version = "3.13.15-owner"
    existing.cells[1].execution_count = 4
    existing.cells[1].outputs = [nbformat.v4.new_output("stream", name="stdout", text="old run")]
    nbformat.write(existing, path)
    monkeypatch.setattr(build_notebook, "ROOT", tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(
        build_notebook.subprocess,
        "run",
        lambda arguments, **_: commands.append(arguments),
    )
    assert build_notebook.build() == path
    rebuilt = nbformat.read(path, as_version=4)
    assert f"RUN_FULL_EXPERIMENT = {run_full}" in rebuilt.cells[1].source
    assert rebuilt.metadata.kernelspec == existing.metadata.kernelspec
    assert rebuilt.metadata.language_info == existing.metadata.language_info
    assert all(cell.outputs == [] for cell in rebuilt.cells if cell.cell_type == "code")
    assert len(commands) == 3 and all("ruff" in command for command in commands)


def test_builder_explicitly_preserves_owner_outputs_and_metadata_by_cell_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "notebooks/cat_dog_cnn_features.ipynb"
    path.parent.mkdir()
    existing = build_notebook.make_notebook(run_full=True)
    existing.metadata["owner_note"] = {"run": "keep this execution"}
    setup = next(cell for cell in existing.cells if cell.id == "setup")
    setup.metadata["execution"] = {"iopub.execute_input": "owner timestamp"}
    setup.execution_count = 11
    setup.outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={"text/plain": "saved response"},
            metadata={"owner_display": "unchanged"},
        )
    ]
    feature = next(cell for cell in existing.cells if cell.id == "run-feature-stages")
    feature.execution_count = 17
    feature.outputs = [nbformat.v4.new_output("stream", name="stdout", text="measured run")]
    existing.cells.reverse()  # Matching IDs, not positions, must preserve the right outputs.
    nbformat.write(existing, path)
    monkeypatch.setattr(build_notebook, "ROOT", tmp_path)
    monkeypatch.setattr(build_notebook.subprocess, "run", lambda *_, **__: None)

    assert build_notebook.build(preserve_outputs=True) == path
    rebuilt = nbformat.read(path, as_version=4)
    preserved_setup = next(cell for cell in rebuilt.cells if cell.id == "setup")
    preserved_feature = next(cell for cell in rebuilt.cells if cell.id == "run-feature-stages")
    assert "RUN_FULL_EXPERIMENT = True" in preserved_setup.source
    assert rebuilt.metadata == existing.metadata
    assert preserved_setup.metadata == setup.metadata
    assert preserved_setup.execution_count == 11 and preserved_setup.outputs == setup.outputs
    assert preserved_feature.execution_count == 17 and preserved_feature.outputs == feature.outputs
    assert "compact=True" in preserved_feature.source
    assert next(cell for cell in rebuilt.cells if cell.id == "run-feature-kernels").outputs == []


def test_only_setup_training_feature_and_report_stages_are_present() -> None:
    notebook = build_notebook.make_notebook()
    calls: list[tuple[str, str | None]] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for node in ast.walk(ast.parse(cell.source)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_stage"
            ):
                stage = cast(ast.Constant, node.args[1]).value
                sections = [keyword.value for keyword in node.keywords if keyword.arg == "section"]
                if stage == "represent":
                    assert len(sections) == 1 and isinstance(sections[0], ast.Constant)
                calls.append((stage, cast(ast.Constant, sections[0]).value if sections else None))
    assert calls == [
        ("setup", None),
        ("train", None),
        ("train", None),
        *(("represent", section) for section in FEATURE_SECTIONS),
        ("report", None),
    ]
    ids = [cell.id for cell in notebook.cells]
    assert ids.index("prepare-data") < ids.index("inspect-data") < ids.index("train-standard")
    assert ids.index("feature-walkthrough") < ids.index("run-feature-activations")
    assert ids.index("run-feature-activations") < ids.index("prediction-influence")
    assert ids.index("prediction-influence") < ids.index("run-feature-gradcam")
    assert not any(identifier.startswith("run-evaluation") for identifier in ids)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("section", FEATURE_SECTIONS)
def test_feature_cells_compute_and_display_only_their_own_family(full: bool, section: str) -> None:
    config = object()
    events: list[tuple[Any, ...]] = []
    result = {"section": section}

    def stage(value: Any, name: str, *, full: bool, device: str, section: str) -> Any:
        assert value is config and name == "represent" and device == "mps"
        events.append(("compute", section, full))
        return result

    def figures(value: Any, *, section: str, compact: bool) -> None:
        assert value is config and compact is True
        events.append(("display", section))

    namespace = {
        "config": config,
        "RUN_FULL_EXPERIMENT": full,
        "device": "mps",
        "run_stage": stage,
        "display_features": figures,
    }
    cell = _cell(f"run-feature-{section}")
    exec(compile(cell.source, "feature-cell", "exec"), namespace)
    assert namespace[f"feature_{section}_result"] is result
    assert events == [("compute", section, full), ("display", section)]
    assert "Image(" not in cell.source and "docs/assets" not in cell.source


def test_safe_display_does_not_hide_stale_evidence() -> None:
    def reject(*_: Any, **__: Any) -> None:
        raise ValueError("stale method/model/image identity")

    namespace = {
        "config": object(),
        "RUN_FULL_EXPERIMENT": False,
        "device": "mps",
        "run_stage": lambda *_, **__: {"status": "skipped"},
        "display_features": reject,
    }
    with pytest.raises(ValueError, match="stale method/model/image identity"):
        exec(compile(_cell("run-feature-stages").source, "feature-cell", "exec"), namespace)


@pytest.mark.parametrize("section", FEATURE_SECTIONS)
def test_notebook_facade_dispatches_only_selected_feature_family(
    monkeypatch: pytest.MonkeyPatch, section: str
) -> None:
    monkeypatch.setattr("src.notebook_support.require_project_environment", lambda: None)
    monkeypatch.setattr("src.notebook_support.require_device", lambda requested: requested)
    calls: list[str] = []

    def selected(config: Any, device: Any, *, progress: bool, section: str) -> object:
        calls.append(section)
        return section

    monkeypatch.setattr("src.feature_visualization.visualize_features", selected)
    assert (
        run_stage(
            load_config(CONFIG),
            "represent",
            full=True,
            device="mps",
            progress=False,
            section=cast(Any, section),
        )
        == section
    )
    assert calls == [section]


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("arm", ["standard", "adversarial"])
def test_training_cells_use_compact_live_charts_without_gallery(arm: str, full: bool) -> None:
    config = object()
    observer = object()
    events: list[tuple[Any, ...]] = []

    def factory(value: Any, *, compact: bool) -> object:
        assert value is config and compact is True
        events.append(("observer",))
        return observer

    def stage(
        value: Any, name: str, *, full: bool, device: str, arm: str, epoch_observer: Any
    ) -> dict[str, Any]:
        assert value is config and name == "train" and device == "mps"
        assert epoch_observer is (observer if full else None)
        events.append(("train", arm, full))
        return {"status": "complete"}

    def training(value: Any, selected: str, *, compact: bool, summary_only: bool) -> None:
        assert value is config and compact is True and summary_only is full
        events.append(("display", selected))

    namespace = {
        "config": config,
        "RUN_FULL_EXPERIMENT": full,
        "device": "mps",
        "run_stage": stage,
        "training_observer": factory,
        "display_training": training,
    }
    exec(compile(_cell(f"train-{arm}").source, "training-cell", "exec"), namespace)
    assert events == [*([("observer",)] if full else []), ("train", arm, full), ("display", arm)]


@pytest.mark.parametrize("full", [False, True])
def test_setup_cell_runs_only_setup(full: bool, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []

    def stage(config: Any, name: str, **_: Any) -> dict[str, str]:
        calls.append(name)
        return {"status": "setup"}

    namespace = {
        "config": object(),
        "RUN_FULL_EXPERIMENT": full,
        "device": "mps",
        "run_stage": stage,
    }
    exec(compile(_cell("prepare-data").source, "setup-cell", "exec"), namespace)
    assert calls == ["setup"]
    output = capsys.readouterr().out
    assert ("Dataset manifests prepared." if full else "Setup skipped in safe mode.") in output


def test_data_cell_displays_only_counts_and_one_balance_chart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "data").mkdir(parents=True)
    (artifact_root / "data/manifest.json").write_text("{}", encoding="utf-8")

    class Config:
        def project_path(self, name: str) -> Path:
            assert name == "artifacts"
            return artifact_root

    calls: list[Any] = []
    monkeypatch.setattr(
        "src.eda.generate_eda",
        lambda _: {"figures": {"split_and_species": "balance.png", "ignored": "other.png"}},
    )
    summary = {"training": 2649, "validation": 295, "calibration": 736, "official_test": 3669}
    namespace = {
        "config": Config(),
        "ROOT": tmp_path,
        "inspect_data": lambda _: {**summary, "attack_subset": 200, "hash": "do-not-dump"},
        "Markdown": lambda text: ("markdown", text),
        "Image": lambda **kwargs: ("image", kwargs),
        "display": calls.append,
    }
    exec(compile(_cell("inspect-data").source, "data-cell", "exec"), namespace)
    assert len(calls) == 2 and calls[0][0] == "markdown" and calls[1][0] == "image"
    assert all(str(value) in calls[0][1] for value in summary.values())
    assert "do-not-dump" not in calls[0][1] and "attack_subset" not in calls[0][1]
    assert calls[1][1]["filename"] == str(tmp_path / "balance.png")


def test_missing_data_is_explicit_without_computation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Config:
        def project_path(self, _: str) -> Path:
            return tmp_path

    def forbid(*_: Any, **__: Any) -> None:
        pytest.fail("missing data must not generate an EDA substitute")

    monkeypatch.setattr("src.eda.generate_eda", forbid)
    messages: list[str] = []
    namespace = {
        "config": Config(),
        "ROOT": tmp_path,
        "inspect_data": forbid,
        "Markdown": lambda text: text,
        "display": messages.append,
    }
    exec(compile(_cell("inspect-data").source, "data-cell", "exec"), namespace)
    assert messages == ["Data unavailable. Run the setup cell in full mode first."]


@pytest.mark.parametrize("full", [False, True])
def test_report_cell_has_no_evaluation_prerequisite(full: bool) -> None:
    calls: list[tuple[Any, ...]] = []

    def stage(value: Any, name: str, *, full: bool, device: str) -> dict[str, str]:
        assert name == "report" and device == "mps"
        calls.append((name, full))
        return {"status": "report"}

    namespace = {
        "config": object(),
        "RUN_FULL_EXPERIMENT": full,
        "device": "mps",
        "run_stage": stage,
        "display_report": lambda _, *, compact: calls.append(("display-report", compact)),
    }
    exec(compile(_cell("run-report").source, "report-cell", "exec"), namespace)
    assert calls == [("report", full), ("display-report", True)]


def test_readme_project_links_are_repository_local_and_resolve() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    assert targets
    for target in targets:
        assert "://" not in target and not target.startswith(("/", ".."))
        assert (ROOT / target.split("#", maxsplit=1)[0]).exists(), target
    assert "Instructions/checklist.md" not in readme and "/Users/" not in readme
    assert "99.37%" not in readme and "67.76%" not in readme
    assert "99.66%" in readme and "67.80%" in readme
    assert "docs/assets/validation-monitoring-comparison.png" in readme


def test_safe_mode_skips_active_stages_without_loading_models() -> None:
    config = load_config(CONFIG)
    for stage, arm, section in [
        ("setup", None, None),
        ("train", "standard", None),
        ("train", "adversarial", None),
        *(("represent", None, section) for section in FEATURE_SECTIONS),
        ("report", None, None),
    ]:
        result = run_stage(
            config,
            cast(Any, stage),
            full=False,
            device="mps",
            arm=cast(Any, arm),
            progress=False,
            section=cast(Any, section),
        )
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
    assert notebook_context(full=True)[1] is True and notebook_context()[1] is False
    monkeypatch.setattr("src.notebook_support.sys.prefix", str(tmp_path / "sibling/.venv"))
    with pytest.raises(RuntimeError, match="Wrong notebook interpreter"):
        require_project_environment()


def test_headless_runner_uses_calling_interpreter_and_direct_entrypoint() -> None:
    specification = CallingKernelSpecs().get_kernel_spec("ignored")
    assert specification.argv[0] == sys.executable
    assert specification.argv[1:3] == ["-m", "ipykernel_launcher"]
    result = subprocess.run(
        [sys.executable, "src/notebook_runner.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "--full" in result.stdout


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
    assert isinstance(epochs, NotebookProgress) and epochs.n == 2
    assert any('aria-label="standard epochs"' in html and 'value="2"' in html for html in rendered)
    output = capsys.readouterr()
    assert output.out == "" and output.err == ""
    assert all("\x1b" not in html and "\r" not in html for html in rendered)


def test_notebook_progress_does_not_publish_from_tqdm_monitor_thread() -> None:
    assert NotebookProgress.monitor_interval == 0
    assert NotebookProgress._instances is not TerminalProgress._instances


def test_notebook_stack_and_kernel_registration_are_pinned() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    for requirement in (
        "jupyterlab==4.5.0",
        "ipykernel==7.0.1",
        "nbclient==0.10.2",
        "nbformat==5.10.4",
    ):
        assert requirement in requirements
    setup = (ROOT / "setup_venv.sh").read_text(encoding="utf-8")
    assert "oxford-pets-adversarial-representations" in setup and "ipykernel install" in setup


def test_source_hash_includes_notebook_sources(tmp_path: Path) -> None:
    notebook = tmp_path / "guide.ipynb"
    notebook.write_text('{"cells": []}', encoding="utf-8")
    before = source_hash(tmp_path)
    notebook.write_text('{"cells": [{"cell_type": "markdown"}]}', encoding="utf-8")
    assert source_hash(tmp_path) != before
