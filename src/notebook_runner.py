"""Execute the guided notebook with this project's exact Python interpreter."""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Match the CLI's no-fallback policy before importing notebook support (and PyTorch).
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import nbformat
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
from jupyter_client.manager import AsyncKernelManager
from nbclient import NotebookClient

from src.config import load_config
from src.io_utils import (
    atomic_write_json,
    atomic_write_text,
    environment_snapshot,
    sha256_file,
    source_hash,
    utc_now,
)
from src.notebook_support import ROOT, require_project_environment
from src.runtime import record_failure


class CallingKernelSpecs(KernelSpecManager):
    def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
        return KernelSpec(
            argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            display_name="Oxford Pets calling interpreter",
            language="python",
        )


class CallingKernelManager(AsyncKernelManager):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(kernel_spec_manager=CallingKernelSpecs(), **kwargs)


def _serialize(notebook: Any) -> str:
    return str(nbformat.writes(notebook))  # type: ignore[no-untyped-call]


def run_notebook(config_path: Path, *, full: bool, requested: str | None = None) -> Path:
    require_project_environment()
    config = load_config(config_path)
    requested_device = requested or config.value("training", "device", str)
    source_path = ROOT / "notebooks" / "oxford_pets_adversarial_representations.ipynb"
    notebook = nbformat.read(source_path, as_version=4)  # type: ignore[no-untyped-call]
    nbformat.validate(notebook)
    if any(
        cell.get("outputs") or cell.get("execution_count") is not None for cell in notebook.cells
    ):
        raise RuntimeError("source notebook must remain output-free")
    run_id = f"{utc_now().replace(':', '').replace('+', '-')}-{uuid.uuid4().hex[:8]}"
    directory = config.project_path("artifacts") / "notebooks" / run_id
    output = directory / "oxford_pets_adversarial_representations.executed.ipynb"
    manifest_path = directory / "manifest.json"
    source_before = source_hash(ROOT)
    environment = {
        **os.environ,
        "OXFORD_PETS_NOTEBOOK_CONFIG": str(config.path),
        "OXFORD_PETS_NOTEBOOK_RUN_FULL": "1" if full else "0",
        "OXFORD_PETS_NOTEBOOK_DEVICE": requested_device,
        "OXFORD_PETS_NOTEBOOK_ROOT": str(ROOT),
    }

    def save_cell(cell: Any, cell_index: int, **_: Any) -> None:
        atomic_write_text(output, _serialize(notebook))

    try:
        NotebookClient(
            notebook,
            timeout=None,
            kernel_name="oxford-pets-calling-interpreter",
            kernel_manager_class=CallingKernelManager,
            allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}},
            on_cell_executed=save_cell,
            on_cell_error=save_cell,
        ).execute(env=environment)
        atomic_write_text(output, _serialize(notebook))
        if source_hash(ROOT) != source_before:
            raise RuntimeError("notebook execution changed a tracked source input")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_at": utc_now(),
            "mode": "full" if full else "safe",
            "requested_device": requested_device,
            "python": sys.executable,
            "config_sha256": config.sha256,
            "source_sha256": source_before,
            "source_notebook": str(source_path.relative_to(ROOT)),
            "source_notebook_sha256": sha256_file(source_path),
            "executed_notebook": str(output.relative_to(ROOT)),
            "executed_notebook_sha256": sha256_file(output),
            "environment": environment_snapshot(),
        }
        atomic_write_json(manifest_path, manifest)
        return output
    except BaseException as error:
        record_failure(
            manifest_path,
            "notebook",
            error,
            {
                "mode": "full" if full else "safe",
                "requested_device": requested_device,
                "config_sha256": config.sha256,
                "partial_output": str(output) if output.is_file() else None,
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute the Week 3 guide; --full explicitly enables scientific stages"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--device", choices=("mps", "cpu"))
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    try:
        requested = None if args.device is None else str(args.device)
        print(run_notebook(args.config, full=bool(args.full), requested=requested))
        return 0
    except BaseException as error:
        print(f"Notebook stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
