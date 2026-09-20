from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.cli import _parser, _run
from src.config import ExperimentConfig, load_config
from src.feature_section_types import FEATURE_SECTIONS
from src.progress import status, tqdm
from src.runtime import SafetyStop, require_device

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "experiment.yaml"


def test_all_commands_accept_no_progress() -> None:
    arguments = {
        "setup": ["setup", "--config", str(CONFIG), "--no-progress"],
        "preflight": [
            "preflight",
            "--config",
            str(CONFIG),
            "--device",
            "mps",
            "--no-progress",
        ],
        "train": [
            "train",
            "--config",
            str(CONFIG),
            "--arm",
            "standard",
            "--no-progress",
        ],
        "evaluate": [
            "evaluate",
            "--config",
            str(CONFIG),
            "--device",
            "mps",
            "--no-progress",
        ],
        "represent": ["represent", "--config", str(CONFIG), "--no-progress"],
        "report": ["report", "--config", str(CONFIG), "--no-progress"],
        "reproduce": [
            "reproduce",
            "--config",
            str(CONFIG),
            "--device",
            "mps",
            "--no-progress",
        ],
    }
    for command, argv in arguments.items():
        parsed = _parser().parse_args(argv)
        assert parsed.command == command
        assert parsed.no_progress is True


def test_disabled_progress_and_status_are_silent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert list(tqdm(range(3), disable=True)) == [0, 1, 2]
    status("hidden", enabled=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_requirements_txt_pins_pytorch_mps_stack() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "torch==2.13.0" in requirements
    assert "torchvision==0.28.0" in requirements
    assert "tqdm==4.70.0" in requirements
    assert not (ROOT / "uv.lock").exists()
    assert not (ROOT / "requirements.in").exists()


def test_mps_device_rejects_enabled_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(SafetyStop, match="must be 0"):
        require_device("mps")


@pytest.mark.parametrize("section", FEATURE_SECTIONS)
def test_represent_section_selector_matches_feature_contract(section: str) -> None:
    parsed = _parser().parse_args(
        ["represent", "--config", str(CONFIG), "--device", "mps", "--section", section]
    )
    assert parsed.section == section


def test_represent_section_selector_rejects_other_stage_family() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["represent", "--config", str(CONFIG), "--section", "brightness"])


@pytest.mark.parametrize("section", [None, *FEATURE_SECTIONS])
def test_cli_dispatches_one_feature_family_or_legacy_full_workflow(
    monkeypatch: pytest.MonkeyPatch, section: str | None
) -> None:
    argv = ["represent", "--config", str(CONFIG), "--device", "mps", "--no-progress"]
    if section is not None:
        argv.extend(["--section", section])
    args = _parser().parse_args(argv)
    monkeypatch.setattr("src.cli.require_device", lambda requested: requested)
    calls: list[dict[str, Any]] = []

    def features(config: Any, device: str, *, progress: bool, **options: Any) -> Any:
        assert not progress and device == "mps"
        calls.append(options)
        return options

    monkeypatch.setattr("src.feature_visualization.visualize_features", features)
    result = _run("represent", load_config(CONFIG), args)
    assert calls == [{} if section is None else {"section": section}]
    assert result == calls[0]


def test_reproduce_runs_only_training_and_clean_feature_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = load_config(CONFIG)
    config = ExperimentConfig(base.path, tmp_path, deepcopy(base.raw), base.sha256)
    args = _parser().parse_args(
        ["reproduce", "--config", str(CONFIG), "--device", "mps", "--no-progress"]
    )
    calls: list[str] = []
    monkeypatch.setattr("src.cli.require_device", lambda requested: requested)

    def setup(_: ExperimentConfig, *, progress: bool) -> dict[str, str]:
        assert not progress
        calls.append("setup")
        return {"status": "complete"}

    def train(_: ExperimentConfig, arm: str, device: str, *, progress: bool) -> dict[str, str]:
        assert not progress and device == "mps"
        calls.append(f"train_{arm}")
        return {"status": "complete"}

    def features(_: ExperimentConfig, device: str, *, progress: bool) -> dict[str, str]:
        assert not progress and device == "mps"
        calls.append("represent")
        return {"status": "complete"}

    def summary(_: ExperimentConfig, *, progress: bool) -> dict[str, str]:
        assert not progress
        calls.append("report")
        return {"status": "complete"}

    def forbidden(*_: Any, **__: Any) -> None:
        pytest.fail("reproduce must not run attack/evaluation/calibration stages")

    monkeypatch.setattr("src.setup_stage.setup_experiment", setup)
    monkeypatch.setattr("src.training.train_arm", train)
    monkeypatch.setattr("src.feature_visualization.visualize_features", features)
    monkeypatch.setattr("src.reporting.report", summary)
    monkeypatch.setattr("src.evaluation.evaluate", forbidden)
    monkeypatch.setattr("src.preflight.run_preflight", forbidden)
    monkeypatch.setattr("src.calibration.fit_temperature", forbidden)
    result = _run("reproduce", config, args)
    assert calls == ["setup", "train_standard", "train_adversarial", "represent", "report"]
    assert isinstance(result, dict) and list(result["stages"]) == calls
    assert not (config.project_path("results") / "evaluation.json").exists()
    assert (config.project_path("artifacts") / "reproduction.json").is_file()
