"""Refresh saved notebook pictures from verified evidence, without running any models."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import nbformat
from IPython.core.formatters import DisplayFormatter
from IPython.display import Image, Markdown

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_notebook  # noqa: E402
from src import runtime_visuals  # noqa: E402
from src.config import ExperimentConfig, load_config  # noqa: E402
from src.feature_section_types import FEATURE_SECTIONS  # noqa: E402
from src.io_utils import atomic_write_bytes, utc_now  # noqa: E402


def _capture(view: Callable[[], None]) -> list[Any]:
    formatter = DisplayFormatter()
    outputs: list[Any] = []

    def display(value: Any, *, metadata: dict[str, Any] | None = None) -> None:
        data, own_metadata = formatter.format(value)  # type: ignore[no-untyped-call]
        outputs.append(
            nbformat.v4.new_output(  # type: ignore[no-untyped-call]
                "display_data", data=data, metadata={**own_metadata, **(metadata or {})}
            )
        )

    with patch.object(runtime_visuals, "_ipython_display", lambda: (Image, Markdown, display)):
        view()
    return outputs


def refresh(config: ExperimentConfig) -> Path:
    """Preserve the owner's parameters, run counts and scientific files; change views only."""
    if config.root != ROOT:
        raise ValueError("Use this checkout's configuration for its saved notebook")
    from src.reporting import load_current_feature_evidence

    features = load_current_feature_evidence(config)
    path = config.root / "notebooks/cat_dog_cnn_features.ipynb"
    previous = nbformat.read(path, as_version=4)  # type: ignore[no-untyped-call]
    refreshed: dict[str, list[Any]] = {}
    for arm in runtime_visuals.ARMS:
        refreshed[f"train-{arm}"] = _capture(
            partial(runtime_visuals.display_training, config, arm, compact=True)
        )
    for section in FEATURE_SECTIONS:
        refreshed[f"run-feature-{section}"] = _capture(
            partial(runtime_visuals.display_features, config, section=section, compact=True)
        )
    refreshed["run-report"] = _capture(lambda: runtime_visuals.display_report(config, compact=True))
    # All views must validate before rebuilding the owner notebook. No stage execution.
    build_notebook.build(preserve_outputs=True)
    notebook = nbformat.read(path, as_version=4)  # type: ignore[no-untyped-call]
    for cell in notebook.cells:
        if cell.id in refreshed:
            cell.outputs = refreshed[cell.id]
            cell.metadata["saved_evidence_display"] = True
        elif cell.cell_type == "code" and cell.id in {"prepare-data", "inspect-data"}:
            cell.outputs = [
                output
                for output in cell.outputs
                if output.output_type == "error"
                or (
                    output.output_type == "display_data"
                    and any(
                        output.get("data", {}).get(mime) for mime in ("image/png", "text/markdown")
                    )
                )
                or (
                    output.output_type == "stream"
                    and "Dataset manifests prepared." in output.get("text", "")
                )
            ]
    notebook.metadata["saved_view_refresh"] = {
        "updated_at": utc_now(),
        "config_sha256": config.sha256,
        "feature_identity": features["identity"],
        "scope": "Presentation refreshed from saved measurements; no training or inference.",
    }
    assert notebook.metadata.kernelspec == previous.metadata.kernelspec
    nbformat.validate(notebook)
    contents = nbformat.writes(notebook).encode("utf-8")  # type: ignore[no-untyped-call]
    atomic_write_bytes(path, contents)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    arguments = parser.parse_args()
    output = refresh(load_config(arguments.config))
    notebook = json.loads(output.read_text())
    print(f"Saved-only notebook views refreshed: {output} ({len(notebook['cells'])} cells)")
