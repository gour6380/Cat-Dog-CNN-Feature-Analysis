"""Typed command-line interface for the locked experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, cast

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig, load_config
from src.io_utils import atomic_write_json, utc_now
from src.runtime import record_failure, require_device

Command = Literal["setup", "preflight", "train", "evaluate", "represent", "report", "reproduce"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Oxford Pets adversarial representation experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("setup", "represent", "report"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
    for name in ("preflight", "evaluate", "reproduce"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--device", required=True, choices=("mps", "cpu"))
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--arm", required=True, choices=("standard", "adversarial"))
    return parser


def _run(command: Command, config: ExperimentConfig, args: argparse.Namespace) -> object:
    if command == "setup":
        from src.setup_stage import setup_experiment

        return setup_experiment(config)
    if command == "preflight":
        from src.preflight import run_preflight

        return run_preflight(config, require_device(cast(str, args.device)))
    if command == "train":
        from src.training import train_arm

        locked_device = config.value("training", "device", str)
        arm = cast(Literal["standard", "adversarial"], args.arm)
        return train_arm(config, arm, require_device(locked_device))
    if command == "evaluate":
        from src.evaluation import evaluate

        return evaluate(config, require_device(cast(str, args.device)))
    if command == "represent":
        from src.representations import represent

        return represent(config)
    if command == "report":
        from src.reporting import report

        return report(config)
    if command == "reproduce":
        from src.evaluation import evaluate
        from src.preflight import run_preflight
        from src.reporting import report
        from src.representations import represent
        from src.setup_stage import setup_experiment
        from src.training import train_arm

        device = require_device(cast(str, args.device))
        stages: dict[str, object] = {}
        stages["setup"] = setup_experiment(config)
        stages["preflight"] = run_preflight(config, device)
        stages["train_standard"] = train_arm(config, "standard", device)
        stages["train_adversarial"] = train_arm(config, "adversarial", device)
        stages["evaluate"] = evaluate(config, device)
        stages["represent"] = represent(config)
        stages["report"] = report(config)
        result = {"status": "complete", "created_at": utc_now(), "stages": stages}
        atomic_write_json(config.project_path("artifacts") / "reproduction.json", result)
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
