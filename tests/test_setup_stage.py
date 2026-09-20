from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.config import load_config
from src.setup_stage import setup_experiment

ROOT = Path(__file__).parents[1]


def test_setup_records_git_remote_without_blocking_local_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "configs/experiment.yaml"
    config_path.parent.mkdir()
    shutil.copyfile(ROOT / "configs/experiment.yaml", config_path)
    shutil.copyfile(ROOT / "requirements.txt", tmp_path / "requirements.txt")
    config = load_config(config_path)
    config.raw["training_monitoring"]["enabled"] = False

    monkeypatch.setattr("src.setup_stage._remote_names", lambda _root: ["origin"])
    monkeypatch.setattr("src.setup_stage.download_official_dataset", lambda _path: None)
    monkeypatch.setattr(
        "src.setup_stage.create_data_manifests", lambda _config: {"status": "fixture"}
    )
    monkeypatch.setattr(
        "src.setup_stage.download_pretrained_state", lambda _config: {"status": "fixture"}
    )
    monkeypatch.setattr(
        "src.setup_stage.initialization_manifest", lambda _config: {"status": "fixture"}
    )
    monkeypatch.setattr("src.setup_stage.source_hash", lambda _root: "fixture-source")
    monkeypatch.setattr("src.setup_stage.environment_snapshot", lambda: {"python": "fixture"})
    monkeypatch.setattr("src.setup_stage.git_state", lambda _root: {"commit": "fixture"})

    result = setup_experiment(config, progress=False)

    assert result["status"] == "complete"
    assert result["remote_names"] == ["origin"]
    assert result["public_actions"] == "none"
    assert (config.project_path("artifacts") / "setup.json").is_file()
