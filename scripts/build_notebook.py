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


def make_notebook() -> Any:
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

            Both ResNet-18 arms start from byte-identical `IMAGENET1K_V1` tensors and an identical new 37-class head. The official test partition remains untouched. The deterministic training/calibration split, attack subset, projection subset, seeds, 224-pixel transforms, configured epoch count, optimizer, learning-rate schedule, and attack budgets live in the repository configuration.
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

            Read the registered official-data manifest and deterministic split, then visualize partition size, breed balance, species composition, and stored image geometry. EDA is descriptive only: it cannot change the split, hypotheses, checkpoint rule, attack samples, or metrics. No pet photograph is embedded in this shareable notebook.
            """,
            "data",
        ),
        _code(
            """
            from IPython.display import Image, Markdown, display

            from src.eda import generate_eda

            data_summary = inspect_data(config)
            eda = generate_eda(config)
            records = eda["record_summary"]
            geometry = eda["image_geometry"]["overall"]
            display(
                Markdown(
                    "\\n".join(
                        [
                            "### Registered-data EDA",
                            "",
                            "| Check | Value |",
                            "|---|---:|",
                            f"| Images | {records['total_unique_images']:,} |",
                            f"| Breeds | {records['classes']} |",
                            f"| Cat / dog breeds | {records['breeds_by_species']['cat']} / "
                            f"{records['breeds_by_species']['dog']} |",
                            f"| Training / calibration / official test | "
                            f"{data_summary['training']:,} / {data_summary['calibration']:,} / "
                            f"{data_summary['official_test']:,} |",
                            f"| Median stored width × height | "
                            f"{geometry['width_pixels']['median']:.0f} × "
                            f"{geometry['height_pixels']['median']:.0f} px |",
                            "",
                            "The figures are aggregate diagnostics. They contain no source photograph.",
                        ]
                    )
                )
            )
            for figure_key in ("split_and_species", "breed_balance", "image_geometry"):
                figure_path = ROOT / eda["figures"][figure_key]
                if not figure_path.is_file():
                    raise FileNotFoundError(figure_path)
                display(Image(filename=str(figure_path), width=1200))
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

            Full mode records memory telemetry and checks dataset isolation, deterministic augmentation/order, identical initialization, CPU/MPS logit parity, PGD bounds, BatchNorm preservation, finite gradients, cleanup, and numerical invariants. Memory readings do not block execution; a failed correctness check stops the notebook before training and remains recorded.
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

            The configured float32 epochs use clean augmented cross-entropy, micro-batch 16, two-step accumulation, AdamW, the configured warm-up, and cosine decay when training extends beyond warm-up. Epoch checkpoints and durable tqdm summaries preserve progress without consulting test results.
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

            Evaluate the configured final-epoch checkpoints on the official test partition, registered corruptions, and the paired 740-image FGSM/PGD-20×5 subset. Temperatures and 90%-coverage thresholds are fitted on clean calibration only. Finite attacks do not certify robustness.
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
            """
            from IPython.display import Image, Markdown, display

            representation_result = run_stage(
                config,
                "represent",
                full=RUN_FULL_EXPERIMENT,
                device=device,
            )
            if (
                isinstance(representation_result, dict)
                and representation_result.get("results_available") is False
            ):
                show(representation_result)
            else:
                if not isinstance(representation_result, dict):
                    raise TypeError("representation stage returned an invalid result")
                primary = representation_result["primary_hypothesis"]
                geometry = representation_result["geometry"]
                standard = geometry["standard"]
                adversarial = geometry["adversarial"]
                verdict = "supported" if primary["primary_supported"] else "not supported"
                lines = [
                    f"### Registered representation result: **{verdict}**",
                    "",
                    f"Configured checkpoint: **epoch {config.integer('training', 'epochs')}**. "
                    "Lower drift is better; higher neighbour retention and CKA are better.",
                    "",
                    "| Original 512-D metric | Standard | PGD-trained |",
                    "|---|---:|---:|",
                    f"| Median cosine drift | {standard['cosine_drift']['median']:.4f} | "
                    f"{adversarial['cosine_drift']['median']:.4f} |",
                    f"| Median relative-L2 drift | {standard['relative_l2_drift']['median']:.4f} | "
                    f"{adversarial['relative_l2_drift']['median']:.4f} |",
                    f"| Clean 5-NN accuracy | {standard['knn']['clean_accuracy']:.2%} | "
                    f"{adversarial['knn']['clean_accuracy']:.2%} |",
                    f"| PGD 5-NN accuracy | {standard['knn']['pgd_accuracy']:.2%} | "
                    f"{adversarial['knn']['pgd_accuracy']:.2%} |",
                    f"| PGD 5-NN breed retention | {standard['knn']['pgd_retention']:.2%} | "
                    f"{adversarial['knn']['pgd_retention']:.2%} |",
                    f"| Clean→PGD linear CKA | {standard['linear_cka_clean_to_pgd']:.4f} | "
                    f"{adversarial['linear_cka_clean_to_pgd']:.4f} |",
                    "",
                    "The decision above is the registered numerical test. The figures below are "
                    "explanatory views and cannot establish robustness by themselves.",
                ]
                display(Markdown("\\n".join(lines)))

                figure_groups = [
                    (
                        "Quantitative geometry",
                        "Original-space drift and per-breed local label retention.",
                        [
                            "figures/generated/geometry/cosine-drift.png",
                            "figures/generated/geometry/knn-retention-by-breed.png",
                        ],
                    ),
                    (
                        "PCA views",
                        "PCA is fitted independently on each model's clean calibration features.",
                        [
                            "figures/generated/projections/pca-standard.png",
                            "figures/generated/projections/pca-adversarial.png",
                        ],
                    ),
                    (
                        "t-SNE views",
                        "Clean and PGD features are embedded jointly within each model only.",
                        [
                            "figures/generated/projections/tsne-standard.png",
                            "figures/generated/projections/tsne-adversarial.png",
                        ],
                    ),
                    (
                        "UMAP views",
                        "Each UMAP is fitted on that model's clean calibration features.",
                        [
                            "figures/generated/projections/umap-standard.png",
                            "figures/generated/projections/umap-adversarial.png",
                        ],
                    ),
                    (
                        "Difficult-pair boundary slices",
                        "These are two-class slices of the 37-class head, not input-space boundaries.",
                        [
                            "figures/generated/boundaries/pair-boundary-standard.png",
                            "figures/generated/boundaries/pair-boundary-adversarial.png",
                        ],
                    ),
                ]
                for title, caption, relative_paths in figure_groups:
                    display(Markdown(f"### {title}\\n\\n{caption}"))
                    for relative_path in relative_paths:
                        figure_path = ROOT / relative_path
                        if not figure_path.is_file():
                            raise FileNotFoundError(figure_path)
                        display(Image(filename=str(figure_path), width=1100))
            """,
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
    return notebook


def build() -> Path:
    notebook = make_notebook()
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
