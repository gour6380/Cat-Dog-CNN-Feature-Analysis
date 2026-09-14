"""Generate the canonical output-free Week 3 notebook."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT
RUFF_CONFIG = PROJECT_ROOT / "pyproject.toml"


def _markdown(source: str, identifier: str) -> Any:
    cell = nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())
    cell["id"] = identifier
    return cell


def _code(source: str, identifier: str, *, parameters: bool = False) -> Any:
    normalized = textwrap.dedent(source).strip()
    ast.parse(normalized)
    cell = nbformat.v4.new_code_cell(normalized)
    cell["id"] = identifier
    if parameters:
        cell["metadata"]["tags"] = ["parameters"]
    return cell


def build() -> Path:
    cells = [
        _markdown(
            """
            # Adversarial Representation Drift in Fine-Grained Pet Recognition

            ## 1. Question, hypotheses, and claim boundary

            **Question.** Does matched PGD adversarial fine-tuning preserve penultimate representation structure under bounded attacks better than standard fine-tuning, and how do clean-fitted confidence policies behave after input shift?

            The primary decision requires both lower median clean-to-PGD cosine feature drift and higher five-nearest-neighbour breed retention for the adversarial model, with class-stratified bootstrap bounds supporting both differences. PCA, t-SNE, and UMAP are explanatory views only. The experiment cannot establish physical robustness, safe pet recognition, unrestricted adversarial robustness, or production readiness.

            The next cell defaults to safe inspection. Training, attack evaluation, representation fitting, and report generation require an explicit full-mode choice.
            """,
            "question",
        ),
        _code(
            """
            import os
            import sys
            from pathlib import Path

            configured_root = os.environ.get("OXFORD_PETS_NOTEBOOK_ROOT")
            candidates = [Path(configured_root).resolve()] if configured_root else []
            candidates.extend([Path.cwd().resolve(), *Path.cwd().resolve().parents])
            ROOT = next(
                (
                    candidate
                    for candidate in candidates
                    if (candidate / "configs/experiment.yaml").is_file()
                    and (candidate / "src/notebook_support.py").is_file()
                ),
                None,
            )
            if ROOT is None:
                raise FileNotFoundError("Open the notebook from its project checkout")
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            from src.notebook_support import (
                inspect_data,
                inspect_environment,
                inspect_model,
                inventory,
                notebook_context,
                run_stage,
                show,
            )

            RUN_FULL_EXPERIMENT = False
            # Set True only to run preflight, both training arms, evaluation, analysis, and reports.
            config, RUN_FULL_EXPERIMENT, device = notebook_context(full=RUN_FULL_EXPERIMENT)
            show(
                {
                    "full": RUN_FULL_EXPERIMENT,
                    "device": device,
                    "config": str(config.path.relative_to(ROOT)),
                    "config_sha256": config.sha256,
                }
            )
            """,
            "setup",
            parameters=True,
        ),
        _markdown(
            """
            ## 2. Locked protocol

            Both ResNet-18 arms start from byte-identical `IMAGENET1K_V1` tensors and an identical new 37-class head. The official test partition remains untouched. The deterministic training/calibration split, attack subset, projection subset, seeds, 224-pixel transforms, 15 epochs, optimizer, learning-rate schedule, and attack budgets are locked in the repository configuration.
            """,
            "protocol",
        ),
        _code("show(config.raw)", "show-config"),
        _markdown(
            """
            ## 3. Environment and native MPS

            The notebook must run with this checkout's `.venv` interpreter. PyTorch MPS is the locked training backend and silent operation fallback is disabled. This inspection reports availability but does not allocate a training model on MPS.
            """,
            "environment",
        ),
        _code("show(inspect_environment(config, device))", "inspect-environment"),
        _markdown(
            """
            ## 4. Data integrity and protocol-neutral EDA

            Read the registered official-data manifest and deterministic split. The class-count chart is descriptive only: it cannot change the split, hypotheses, checkpoint rule, attack samples, or metrics. No pet photograph is embedded in this shareable notebook.
            """,
            "data",
        ),
        _code(
            """
            import matplotlib.pyplot as plt
            from IPython.display import display

            data_summary = inspect_data(config)
            show({key: value for key, value in data_summary.items() if not key.endswith("_per_class")})
            class_ids = list(range(data_summary["classes"]))
            figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
            axis.plot(class_ids, data_summary["training_per_class"], marker="o", label="training")
            axis.plot(class_ids, data_summary["calibration_per_class"], marker="s", label="calibration")
            axis.set(title="Registered examples per breed", xlabel="breed class ID", ylabel="images")
            axis.grid(alpha=0.2)
            axis.legend()
            display(figure)
            plt.close(figure)
            """,
            "inspect-data",
        ),
        _markdown(
            """
            ## 5. Model and feature interface

            Construct the registered model on CPU and verify its 37 logits and original 512-dimensional penultimate feature interface. This is an architecture inspection, not a trained-model result.
            """,
            "model",
        ),
        _code("show(inspect_model(config))", "inspect-model"),
        _markdown(
            """
            ## 6. Native MPS preflight

            Full mode first checks memory, dataset isolation, deterministic augmentation/order, identical initialization, CPU/MPS logit parity, PGD bounds, BatchNorm preservation, finite gradients, cleanup, and numerical invariants. Failure stops the notebook before training and remains recorded.
            """,
            "preflight",
        ),
        _code(
            "show(run_stage(config, 'preflight', full=RUN_FULL_EXPERIMENT, device=device))",
            "run-preflight",
        ),
        _markdown(
            """
            ## 7. Standard fine-tuning

            Fifteen float32 epochs use clean augmented cross-entropy, micro-batch 16, two-step accumulation, AdamW, one-epoch warm-up, and cosine decay. Epoch checkpoints and durable tqdm summaries preserve progress without consulting test results.
            """,
            "standard",
        ),
        _code(
            "show(run_stage(config, 'train', full=RUN_FULL_EXPERIMENT, device=device, arm='standard'))",
            "train-standard",
        ),
        _markdown(
            """
            ## 8. PGD-5 adversarial fine-tuning

            The matched arm changes only the training inputs: untargeted random-start PGD-5 at `L∞ 4/255`, step `1/255`, projected and clipped in raw `[0,1]` pixel space. Initialization, ordered augmentations, optimizer updates, and schedule remain matched.
            """,
            "adversarial",
        ),
        _code(
            "show(run_stage(config, 'train', full=RUN_FULL_EXPERIMENT, device=device, arm='adversarial'))",
            "train-adversarial",
        ),
        _markdown(
            """
            ## 9. Evaluation, calibration, and input shifts

            Evaluate fixed epoch-15 checkpoints on the official test partition, registered corruptions, and the paired 740-image FGSM/PGD-20×5 subset. Temperatures and 90%-coverage thresholds are fitted on clean calibration only. Finite attacks do not certify robustness.
            """,
            "evaluation",
        ),
        _code(
            "show(run_stage(config, 'evaluate', full=RUN_FULL_EXPERIMENT, device=device))",
            "run-evaluation",
        ),
        _markdown(
            """
            ## 10. Original-feature geometry and explanatory projections

            Quantitative evidence uses aligned original 512-D features: cosine and relative-L2 drift, five-NN retention, nearest centroids, scatter, margins, linear CKA, and class-stratified uncertainty. PCA, t-SNE, UMAP, and difficult-pair boundary slices are explanatory and are never compared by raw coordinates across independently fitted model views.
            """,
            "representations",
        ),
        _code(
            "show(run_stage(config, 'represent', full=RUN_FULL_EXPERIMENT, device=device))",
            "run-representations",
        ),
        _markdown(
            """
            ## 11. Reports, limitations, and artifact inventory

            Reports are generated only from complete, matching machine-readable evidence. Safe mode shows existing setup metadata while unavailable scientific results remain explicitly skipped. Executed notebook copies stay under ignored local artifacts; this source notebook remains output-free.
            """,
            "report",
        ),
        _code(
            """
            report_result = run_stage(
                config,
                "report",
                full=RUN_FULL_EXPERIMENT,
                device=device,
            )
            show(report_result)
            show(inventory(config))
            """,
            "run-report",
        ),
        _markdown(
            """
            ## 12. Reproduction

            Interactive full run: change `RUN_FULL_EXPERIMENT = False` to `True`, rerun the setup cell, then run every following cell in order. Headless safe run: `.venv/bin/python src/notebook_runner.py --device mps`. Headless full run: add `--full`. The canonical CLI remains `.venv/bin/python src/cli.py reproduce --config configs/experiment.yaml --device mps`.
            """,
            "reproduction",
        ),
    ]
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (Oxford Pets Adversarial Representations)",
                "language": "python",
                "name": "oxford-pets-adversarial-representations",
            },
            "language_info": {"name": "python", "version": "3.13.15"},
        },
    )
    nbformat.validate(notebook)
    output = ROOT / "notebooks" / "oxford_pets_adversarial_representations.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    for arguments in (
        ["check", "--fix", "--select", "I,E401"],
        ["format"],
        ["check"],
    ):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                *arguments,
                "--config",
                str(RUFF_CONFIG),
                str(output),
            ],
            cwd=ROOT,
            check=True,
        )
    return output


if __name__ == "__main__":
    print(build())
