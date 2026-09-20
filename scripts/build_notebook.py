"""Generate a compact Cat/Dog guide; clear outputs unless explicitly preserving an owner run."""

from __future__ import annotations

import argparse
import ast
import copy
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
    cell = nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())  # type: ignore[no-untyped-call]
    cell["id"] = identifier
    return cell


def _code(source: str, identifier: str, *, parameters: bool = False) -> Any:
    normalized = textwrap.dedent(source).strip()
    ast.parse(normalized)
    cell = nbformat.v4.new_code_cell(normalized)  # type: ignore[no-untyped-call]
    cell["id"] = identifier
    if parameters:
        cell["metadata"]["tags"] = ["parameters"]
    return cell


def _feature_cells(*, influence: bool) -> list[Any]:
    """Each cell computes and displays only its named current-run figure family."""
    sections = (
        (
            "stages",
            "Image → layers → prediction",
            "Follow the actual 224px input through the network. A channel is a small pattern "
            "tester; its map shows where it responds, not a restored photograph. Later maps "
            "are coarser. Pooling turns the final maps into 512 summary numbers, which the "
            "classifier combines into cat/dog scores.",
        ),
        (
            "kernels",
            "First-layer filters",
            "A first-layer filter is a small colored stencil that responds to patterns such "
            "as edges or color changes. Compare the starting ImageNet stencil, the fine-tuned "
            "one, and their difference. These are weights, not a heatmap for a particular pet "
            "or proof of an eye/ear/fur detector.",
        ),
        (
            "synthetic",
            "Synthetic channel preferences",
            "Let an optimizer draw a picture that increases one channel's response. The "
            "patterns show preferences, not remembered pets or training photographs. A gray "
            "tile means this trial did not find a stronger pattern from its starting noise; "
            "it does not mean the filter is useless. Unsuccessful trials stay visible.",
        ),
        (
            "real_patches",
            "Strong real-image patches",
            "Find real reference pictures that strongly activate the selected pattern testers. "
            "The box is the region that could feed that response (its receptive field), not "
            "proof that every enclosed pixel matters or that this picture taught the filter. "
            "A late-layer box may cover the whole input. The compact view shows the first "
            "two reference-ranked channels per level and their strongest recorded image; "
            "the fuller galleries remain linked.",
        ),
        (
            "activations",
            "Clean activation maps and sensitivity",
            "An activation map asks 'where does this channel respond?' An input gradient asks "
            "'which tiny pixel changes could alter that response?' These are different views, "
            "not reconstructed images. Heatmaps are display-scaled; brighter colors do not "
            "make one model better.",
        ),
        (
            "gradcam",
            "Grad-CAM class influence",
            "Highlight regions connected to the score for the animal's true species, even "
            "when the prediction is wrong. A cat picture always targets the cat score. "
            "Grad-CAM is coarse and a blank map means no positive map for this target under "
            "this method—not that the network has no features. It is not causal proof.",
        ),
        (
            "occlusion",
            "Masking and region controls",
            "Cover one tile at a time and watch the correct-species score minus the other "
            "score (the margin). Red/positive means covering the tile lowers that margin: "
            "it was helping the correct species. Blue/negative means covering it raises the "
            "margin. Masks are artificial inputs, and model score scales can differ.",
        ),
        (
            "diagnostics",
            "Randomized-weight checks and region controls",
            "Replace the learned weights with random ones and compare Grad-CAM on the same "
            "pictures. This checks dependence on learned weights; a changed map does not "
            "prove the explanation is correct. Compare masking Grad-CAM-ranked tiles with "
            "random equal-area tiles. Flat maps can make correlation undefined. Extra "
            "response/change charts are linked as optional details.",
        ),
    )
    selected = sections[5:] if influence else sections[:5]
    cells: list[Any] = []
    for index, (section, title, explanation) in enumerate(selected, start=1):
        cells.extend(
            [
                _markdown(
                    f"### {'4' if influence else '3'}{chr(ord('a') + index - 1)}. {title}"
                    f"\n\n{explanation}",
                    f"feature-{section}",
                ),
                _code(
                    f"feature_{section}_result = run_stage(config, 'represent', "
                    f"full=RUN_FULL_EXPERIMENT, device=device, section='{section}')\n"
                    f"display_features(config, section='{section}', compact=True)",
                    f"run-feature-{section}",
                ),
            ]
        )
    return cells


def make_notebook(*, run_full: bool = False) -> Any:
    """Build safe-by-default source cells without reading saved results or images."""
    cells = [
        _markdown(
            """
            # Cat/Dog CNN Features: What Responds, and Where?

            Train two ResNet-18 Cat/Dog classifiers, then see what their channels respond to
            and which regions influence a prediction. The labels are **cat=0, dog=1**;
            the 37 breed annotations are used for splitting, not a breed prediction head.

            Display saved pictures only when their registered evidence matches the current
            run; otherwise show what is missing. There is no historical substitute. These
            pictures describe current model behavior, not how a filter originally learned
            a pattern or a robustness guarantee.

            ## 1. Setup and data

            Select this project's `.venv/bin/python` (CPython 3.13.15, requirements.txt,
            PyTorch/MPS, tqdm). `RUN_FULL_EXPERIMENT=True` intentionally runs setup,
            both training arms, eight feature families and the report. Set it to `False` for
            read-only inspection; missing evidence is reported rather than substituted.
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

            import torch
            from IPython.display import Image, Markdown, display

            from src.notebook_support import inspect_data, notebook_context, run_stage
            from src.runtime_visuals import (
                display_features,
                display_report,
                display_training,
                training_observer,
            )

            RUN_FULL_EXPERIMENT = False
            config, RUN_FULL_EXPERIMENT, device = notebook_context(full=RUN_FULL_EXPERIMENT)
            print(f"Python {sys.version.split()[0]} · PyTorch {torch.__version__} · device: {device}")
            print(f"Epochs: {config.integer('training', 'epochs')} · full execution: {RUN_FULL_EXPERIMENT}")
            """,
            "setup",
            parameters=True,
        ),
        _markdown(
            """
            ### Prepare the registered split

            Oxford-IIIT Pet: about 7,349 photographs, 37 breeds (12 cat / 25 dog breeds).
            Preserve the official test split. Hash-split trainval 80/20 within breed, then
            reserve 10% of the original training pool for clean validation:
            approximately **2,649 fitting / 295 validation / 736 reference / 3,669 test**.
            Both models use the same IDs. Reference images select channels; validation
            monitors training. Neither selects the final checkpoint.
            """,
            "prepare-data-heading",
        ),
        _code(
            "setup_result = run_stage(config, 'setup', full=RUN_FULL_EXPERIMENT, device=device)\n"
            "print('Dataset manifests prepared.' if RUN_FULL_EXPERIMENT else 'Setup skipped in safe mode.')",
            "prepare-data",
        ),
        _code(
            """
            from src.eda import generate_eda

            if (config.project_path("artifacts") / "data/manifest.json").is_file():
                data_summary = inspect_data(config)
                rows = [
                    ("Fitting", "training"),
                    ("Validation", "validation"),
                    ("Reference", "calibration"),
                    ("Official test", "official_test"),
                ]
                table = "| Partition | Images |\\n|---|---:|\\n"
                table += "\\n".join(f"| {label} | {data_summary.get(key, 'unavailable')} |" for label, key in rows)
                display(Markdown(table))
                eda = generate_eda(config)
                display(Image(filename=str(ROOT / eda["figures"]["split_and_species"]), width=1000))
            else:
                display(Markdown("Data unavailable. Run the setup cell in full mode first."))
            """,
            "inspect-data",
        ),
        _markdown(
            """
            ## 2. Training

            Start from identical ImageNet ResNet-18 weights and a new two-class head:
            the networks already know many image patterns; they are not trained from scratch.
            Match fitting IDs, augmentation, sample order, optimizer updates and schedule.
            Use the configured final epoch, not a validation-selected checkpoint.

            Each completed epoch refreshes **training-objective loss** (the mistake penalty;
            lower is better for that model's objective) and **clean validation accuracy**
            (correct answers on pictures not used to fit the weights). Read the final
            cat/dog recalls too: overall accuracy can hide always choosing the common class.
            The dashed line gives cats and dogs equal weight. The dotted line shows what
            simply guessing the more common animal would achieve—without learning the task.
            Monitoring does not update the model. One point means one measured epoch.

            ### Standard model

            Cross-entropy on actual clean augmented inputs. Training uses float32 native MPS,
            micro-batch 16, two-step accumulation, AdamW and tqdm.
            """,
            "standard",
        ),
        _code(
            "standard_result = run_stage(config, 'train', full=RUN_FULL_EXPERIMENT, device=device, "
            "arm='standard', epoch_observer=training_observer(config, compact=True) if RUN_FULL_EXPERIMENT else None)\n"
            "display_training(config, 'standard', compact=True, summary_only=RUN_FULL_EXPERIMENT)",
            "train-standard",
        ),
        _markdown(
            """
            ### PGD-trained model

            Train on inputs changed slightly to make classification harder. PGD takes five
            small steps, keeping each pixel change within `4/255` (step `1/255`) and valid
            `[0,1]` pixels. The loss is measured on these difficult inputs, so it is not the
            standard model's clean loss. Validation uses clean pictures for both models.
            PGD training alone does not establish robustness.
            """,
            "adversarial",
        ),
        _code(
            "adversarial_result = run_stage(config, 'train', full=RUN_FULL_EXPERIMENT, device=device, "
            "arm='adversarial', epoch_observer=training_observer(config, compact=True) if RUN_FULL_EXPERIMENT else None)\n"
            "display_training(config, 'adversarial', compact=True, summary_only=RUN_FULL_EXPERIMENT)",
            "train-adversarial",
        ),
        _markdown(
            """
            ## 3. Learned features

            Use the same fixed clean test anchors: **two cats and two dogs**. Select channels
            on reference images before looking at the test pictures. Each cell below computes
            only its named family and necessary prerequisites, then reads verified current-run
            figures. Photo panels remain ignored local artifacts.

            Compact layer/activation previews show the first fixed cat and first fixed dog
            for both models—not the prettiest examples. All four anchors and full figures
            remain linked; use `display_features(config, section="stages", compact=False)`
            (or another section) to display everything.

            Read in order: layer walkthrough → learned kernels → synthetic preferences →
            strong real patches → clean responses. Do not assign anatomical detector labels
            from visual resemblance alone.
            """,
            "feature-walkthrough",
        ),
        *_feature_cells(influence=False),
        _markdown(
            """
            ## 4. Prediction influence

            Grad-CAM highlights regions linked to the animal's correct-class score
            (**true-species logit**). Occlusion covers regions and measures how the correct
            score's lead over the other score changes (**true-species-vs-other logit margin**).
            Keep that target fixed even when the model is wrong. These methods ask different
            questions; neither reveals which training picture taught a filter.
            """,
            "prediction-influence",
        ),
        *_feature_cells(influence=True),
        _markdown(
            """
            ## 5. Current-run results

            Build a feature-only report and read-only picture companion from the saved training
            history and eight registered figure families. No separate attack, corruption or
            confidence evaluation is needed. Incomplete evidence stays explicitly incomplete.
            The companion under `reports/generated/` can be read later without training again.

            You can review saved outputs without clearing them or retraining. Local photographs
            and derived panels are ignored; reviewed synthetic/aggregate exports are separate
            from executed notebook outputs.
            """,
            "report",
        ),
        _code(
            "report_result = run_stage(config, 'report', full=RUN_FULL_EXPERIMENT, device=device)\n"
            "display_report(config, compact=True)",
            "run-report",
        ),
        _markdown(
            """
            Reproduce with `.venv/bin/python src/cli.py reproduce --config configs/experiment.yaml --device mps`.
            Edit valid parameters in `configs/experiment.yaml` before a fresh run; changed
            identities cannot silently reuse old checkpoints. The registered run used 15 epochs;
            its pure-PGD arm collapsed to the dog-majority rule, so its feature pictures are
            failure diagnostics rather than evidence of model superiority or robustness.

            Feature-visualization inspiration:
            [Lee et al., 2009](https://ai.stanford.edu/~ang/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf),
            [Grad-CAM](https://arxiv.org/abs/1610.02391),
            [sanity checks](https://arxiv.org/abs/1810.03292).
            Our discriminative ResNet does not reproduce Lee et al.'s generative method.
            """,
            "reproduction",
        ),
    ]
    notebook = nbformat.v4.new_notebook(  # type: ignore[no-untyped-call]
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (Oxford Pets Cat/Dog CNN Features)",
                "language": "python",
                "name": "oxford-pets-adversarial-representations",
            },
            "language_info": {"name": "python", "version": "3.13.15"},
        },
    )
    parameters = next(cell for cell in notebook.cells if cell.id == "setup")
    parameters.source = parameters.source.replace(
        "RUN_FULL_EXPERIMENT = False", f"RUN_FULL_EXPERIMENT = {run_full}"
    )
    nbformat.validate(notebook)
    return notebook


def _existing_run_full(notebook: Any) -> bool:
    for cell in notebook.cells:
        if cell.cell_type != "code" or cell.id != "setup":
            continue
        for statement in ast.parse(cell.source).body:
            if (
                isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "RUN_FULL_EXPERIMENT"
                    for target in statement.targets
                )
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, bool)
            ):
                return statement.value.value
    return False


def build(*, run_full: bool | None = None, preserve_outputs: bool = False) -> Path:
    """Refresh stable cell IDs; saved owner outputs are retained only by explicit opt-in."""
    output = ROOT / "notebooks" / "cat_dog_cnn_features.ipynb"
    existing = nbformat.read(output, as_version=4) if output.is_file() else None  # type: ignore[no-untyped-call]
    if run_full is None:
        run_full = _existing_run_full(existing) if existing is not None else False
    notebook = make_notebook(run_full=run_full)
    if existing is not None:
        if preserve_outputs:
            notebook.metadata.update(copy.deepcopy(existing.metadata))
        for key in ("kernelspec", "language_info"):
            if key in existing.metadata:
                notebook.metadata[key] = copy.deepcopy(existing.metadata[key])
        if preserve_outputs:
            previous_cells = {cell.id: cell for cell in existing.cells}
            for cell in notebook.cells:
                previous = previous_cells.get(cell.id)
                if previous is None or previous.cell_type != cell.cell_type:
                    continue
                cell.metadata.update(copy.deepcopy(previous.metadata))
                if cell.cell_type == "code":
                    cell.execution_count = previous.execution_count
                    cell.outputs = copy.deepcopy(previous.outputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)  # type: ignore[no-untyped-call]
    for arguments in (
        ["check", "--fix", "--select", "I,E401"],
        ["format"],
        ["check"],
    ):
        subprocess.run(
            [sys.executable, "-m", "ruff", *arguments, "--config", str(RUFF_CONFIG), str(output)],
            cwd=ROOT,
            check=True,
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preserve-outputs",
        action="store_true",
        help="Preserve a saved owner run by cell ID during an explicit presentation refresh",
    )
    arguments = parser.parse_args()
    print(build(preserve_outputs=arguments.preserve_outputs))
