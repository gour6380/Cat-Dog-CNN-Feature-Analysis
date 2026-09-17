"""Typed command-line interface for the configured experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal, cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Apply the no-fallback policy before importing modules that load PyTorch.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

from src.config import ExperimentConfig, load_config
from src.io_utils import atomic_write_json, utc_now
from src.progress import status, tqdm
from src.runtime import record_failure, require_device

Command = Literal[
    "setup", "preflight", "train", "evaluate", "represent", "report", "reproduce", "pilot"
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cat/dog CNN feature learning and localization")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "represent", "report"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--no-progress", action="store_true")
        if name == "represent":
            command.add_argument("--device", choices=("mps", "cpu"))
    for name in ("preflight", "evaluate", "reproduce", "pilot"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--device", required=True, choices=("mps", "cpu"))
        command.add_argument("--no-progress", action="store_true")
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--arm", required=True, choices=("standard", "adversarial"))
    train.add_argument("--no-progress", action="store_true")
    return parser


def _run(command: Command, config: ExperimentConfig, args: argparse.Namespace) -> object:
    from src.pilot_protocol import pilot_enabled

    progress = not bool(args.no_progress)
    is_pilot = pilot_enabled(config)
    if is_pilot:
        from src.pilot import assert_pilot_isolated

        assert_pilot_isolated(config)
    if command == "pilot" or (command == "reproduce" and is_pilot):
        from src.pilot import run_pilot

        return run_pilot(config, require_device(cast(str, args.device)), progress=progress)
    if command == "setup":
        from src.setup_stage import setup_experiment

        return setup_experiment(config, progress=progress)
    if command == "preflight":
        from src.preflight import run_preflight

        return run_preflight(config, require_device(cast(str, args.device)), progress=progress)
    if command == "train":
        from src.training import train_arm

        locked_device = config.value("training", "device", str)
        arm = cast(Literal["standard", "adversarial"], args.arm)
        return train_arm(config, arm, require_device(locked_device), progress=progress)
    if command == "evaluate":
        from src.evaluation import evaluate

        return evaluate(
            config,
            require_device(cast(str, args.device)),
            progress=progress,
            arms=("adversarial",) if is_pilot else ("standard", "adversarial"),
        )
    if command == "represent":
        if is_pilot:
            raise ValueError("pilot has no matched standard arm; use its separate report charts")
        from src.feature_visualization import visualize_features

        return visualize_features(
            config,
            require_device(args.device or config.value("training", "device", str)),
            progress=progress,
        )
    if command == "report":
        if is_pilot:
            from src.pilot_reporting import report_pilot

            return report_pilot(config, progress=progress)
        from src.reporting import report

        return report(config, progress=progress)
    if command == "reproduce":
        from src.evaluation import evaluate
        from src.feature_visualization import visualize_features
        from src.preflight import run_preflight
        from src.reporting import report
        from src.setup_stage import setup_experiment
        from src.training import train_arm

        device = require_device(cast(str, args.device))
        stages: dict[str, object] = {}
        status("Reproduce: starting the seven cat/dog feature-study stages...", enabled=progress)
        with tqdm(
            total=7, desc="reproduce stages", unit="stage", disable=not progress
        ) as stage_bar:
            stages["setup"] = setup_experiment(config, progress=progress)
            stage_bar.update(1)
            stages["preflight"] = run_preflight(config, device, progress=progress)
            stage_bar.update(1)
            stages["train_standard"] = train_arm(config, "standard", device, progress=progress)
            stage_bar.update(1)
            stages["train_adversarial"] = train_arm(
                config, "adversarial", device, progress=progress
            )
            stage_bar.update(1)
            stages["evaluate"] = evaluate(config, device, progress=progress)
            stage_bar.update(1)
            stages["represent"] = visualize_features(config, device, progress=progress)
            stage_bar.update(1)
            stages["report"] = report(config, progress=progress)
            stage_bar.update(1)
        result = {"status": "complete", "created_at": utc_now(), "stages": stages}
        atomic_write_json(config.project_path("artifacts") / "reproduction.json", result)
        status("Reproduce complete: all seven stages passed.", enabled=progress)
        return result
    raise AssertionError(f"unhandled command: {command}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = cast(Command, args.command)
    config: ExperimentConfig | None = None
    try:
        config = load_config(args.config)
        result = _run(command, config, args)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except BaseException as error:
        if config is not None:
            record_failure(
                config.project_path("artifacts") / "failures" / f"cli-{command}.json",
                f"cli:{command}",
                error,
            )
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
