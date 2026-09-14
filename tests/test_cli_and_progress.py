from __future__ import annotations

from pathlib import Path

import pytest

from src.cli import _parser
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
