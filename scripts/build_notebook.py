"""Generate the output-free Cat/Dog CNN feature walkthrough."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT
RUFF_CONFIG = PROJECT_ROOT / "pyproject.toml"
SAFE_PUBLIC_FIGURE_KINDS = {
    "kernels",
    "activation_maximization",
    "species_response",
    "initial_response_change",
    "localization_summary",
}


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


def _public_preview_cells(
    root: Path | None = None, *, config_sha256: str | None = None
) -> list[Any]:
    """Link a small, current photograph-free gallery without executing the model."""

    root = ROOT if root is None else root
    manifest_path = root / "docs/results.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        figures = manifest["figures"]
        assets = manifest["provenance"]["public_assets"]
        recorded_config = manifest["provenance"]["configuration_sha256"]
    except (ValueError, TypeError, KeyError, OSError):
        return []
    if not isinstance(figures, list) or not figures or not isinstance(assets, dict):
        return []
    if config_sha256 is None:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from src.config import load_config

        config_sha256 = load_config(root / "configs/experiment.yaml", enforce_python=False).sha256
    if recorded_config != config_sha256:
        return []

    selected: dict[tuple[int, str], dict[str, Any]] = {}
    for figure in figures:
        if (
            not isinstance(figure, dict)
            or figure.get("kind") not in SAFE_PUBLIC_FIGURE_KINDS
            or figure.get("shareable") is False
            or not isinstance(figure.get("path"), str)
        ):
            continue
        relative = Path(figure["path"])
        path = root / relative
        if (
            relative.is_absolute()
            or not relative.is_relative_to(Path("docs/assets"))
            or not path.resolve().is_relative_to((root / "docs/assets").resolve())
            or path.is_symlink()
            or not path.is_file()
            or path.suffix.lower() != ".png"
        ):
            continue
        expected_hash = assets.get(str(relative))
        try:
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            continue

        name = relative.stem.lower().replace("_", "-")
        kind, arm = figure.get("kind"), figure.get("arm")
        if kind == "species_response" and any(tag in name for tag in ("workflow", "architecture")):
            key = (0, "workflow")
        elif kind == "activation_maximization" and arm in {"standard", "adversarial"}:
            key = (1, "0-standard" if arm == "standard" else "1-adversarial")
        elif kind == "species_response" and "accuracy-context" in name:
            key = (2, "accuracy")
        elif kind == "kernels" and arm in {"standard", "adversarial"}:
            key = (3, "0-standard" if arm == "standard" else "1-adversarial")
        else:
            continue
        # One image per intended slot: no crowded dump of every registered panel.
        selected.setdefault(key, figure)
    if not selected:
        return []
    cells = [
        _markdown(
            """
            ### Current public-safe feature gallery — no execution needed

            These are saved, configuration-matched synthetic/kernel/aggregate figures, with
            verified asset hashes. They are linked from this checkout rather than embedded as
            code outputs. Synthetic tiles are optimized channel stimuli, not reconstructed pet
            photographs or proof of named eye/ear detectors. Real input patches and attribution
            overlays remain in the ignored local result companion. Separately normalized tiles
            cannot establish absolute response strength or robustness.

            A gray zero-response tile can be an unsuccessful single-start stimulus
            optimization, not a dead channel or proof that nothing was learned. Check the
            recorded response gains and real calibration responses in `docs/results.json`
            and the visible failure notes in `docs/results.md`; failed trials are retained.
            """,
            "public-feature-gallery",
        )
    ]
    for key in sorted(selected):
        figure = selected[key]
        relative = Path(figure["path"])
        link = (Path("..") / relative).as_posix()
        identifier = "public-figure-" + hashlib.sha256(str(relative).encode()).hexdigest()[:12]
        arm = str(figure.get("arm", "experiment"))
        kind = str(figure["kind"]).replace("_", " ")
        caption = str(figure.get("caption", "Saved public-safe figure"))
        cell = _markdown(f"### {arm}: {kind}\n\n![{arm} {kind}]({link})\n\n{caption}", identifier)
        cell.metadata["tags"] = ["public-safe-preview"]
        cells.append(cell)
    return cells


def make_notebook(*, include_public_figures: bool = True) -> Any:
    cells = [
        _markdown(
            """
            # Cat/Dog CNN Features: What Responds, and Where?

            ## 1. Question and claim boundary

            This is a learned-feature walkthrough: see low-, mid-, and high-level channel patterns,
            connect them to parts of real cat/dog images, and examine which regions affect the
            class score. It is not a two-dimensional embedding study.

            The model predicts **cat=0, dog=1**, not the 37 breeds. Matched standard and PGD-trained
            models let us examine the same images under a fixed digital attack. A feature picture
            cannot prove semantic understanding, safety, or physical robustness. An attractive
            synthetic pattern is not a real training example or a decoded memory.

            The earlier breed experiment is superseded. Its numerical results are not reused.
            The new two-class head needs matching new checkpoints and evidence.
            """,
            "question",
        ),
        _code(
            """
            import json
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
            # True intentionally runs preflight, both arms, attacks, feature figures, and reports.
            config, RUN_FULL_EXPERIMENT, device = notebook_context(full=RUN_FULL_EXPERIMENT)
            show(
                {
                    "full": RUN_FULL_EXPERIMENT,
                    "device": device,
                    "config": str(config.path.relative_to(ROOT)),
                    "config_sha256": config.sha256,
                    "labels": {"cat": 0, "dog": 1},
                }
            )
            """,
            "setup",
            parameters=True,
        ),
        _markdown(
            """
            ## 2. Editable protocol and matched comparison

            The YAML is the source of truth. Both ResNet-18 arms start from identical pinned
            ImageNet weights and an identical new two-class head; they use the same ordered
            samples, deterministic augmentations, optimizer updates, and schedule. Only the
            training inputs differ. The configured final epoch, not the best test epoch, is compared.

            Changing a valid value such as epochs is allowed and creates a new run identity.
            Old evidence is not silently relabeled. No automatic protocol downgrade is used.
            """,
            "protocol",
        ),
        _code("show(config.raw)", "show-config"),
        _markdown(
            """
            ## 3. Environment and native PyTorch MPS

            Select this checkout's `.venv/bin/python`: CPython 3.13.15, pinned `requirements.txt`,
            PyTorch, native MPS, and tqdm. The setup uses venv/pip, not uv. Silent MPS fallback
            is disabled. This cell inspects the runtime without starting training.
            """,
            "environment",
        ),
        _code("show(inspect_environment(config, device))", "inspect-environment"),
        _markdown(
            """
            ## 4. Dataset and EDA

            Oxford-IIIT Pet has roughly 7,349 images, 37 breeds, 12 cat breeds and 25 dog breeds.
            Learning labels are species. The official test stays intact; official `trainval` is
            split 80/20 **within each breed** into training/calibration. Stored breed metadata
            verifies coverage; the class-balance chart counts the actual Cat/Dog targets.

            Inspect actual registered counts, species imbalance, breed coverage, image sizes, and
            aspect ratios. Do not infer clean accuracy from class imbalance alone. Report macro
            and per-species metrics alongside overall accuracy. EDA does not alter the split.
            """,
            "data",
        ),
        _code(
            """
            from IPython.display import Image, Markdown, display

            from src.eda import generate_eda

            data_summary = inspect_data(config)
            eda = generate_eda(config)
            show(data_summary)
            show(eda["record_summary"])
            for figure_key, relative_path in eda["figures"].items():
                figure_path = ROOT / relative_path
                if not figure_path.is_file():
                    raise FileNotFoundError(figure_path)
                display(Markdown(f"### EDA: {figure_key.replace('_', ' ')}"))
                display(Image(filename=str(figure_path), width=1200))
            """,
            "inspect-data",
        ),
        _markdown(
            """
            ## 5. Model depth and spatial features

            `RGB → conv1/low network.relu → residual blocks/mid network.layer2 → high network.layer4
            → global average pool → two-class head`.

            Hooks preserve spatial channel maps at low, mid, and high stages. Early responses
            often involve colors/edges; later stages combine larger patterns. This is a tendency,
            not proof that an individual channel is an eye, ear, or fur detector. The 512-dimensional
            pooled feature is still part of the classifier, but pooled embeddings are not the
            centerpiece of this walkthrough.
            """,
            "model",
        ),
        _code("show(inspect_model(config))", "inspect-model"),
        _markdown(
            """
            ## 6. Native MPS preflight

            Check data isolation, deterministic augmentation/order, identical initialization,
            CPU/MPS logit parity, finite gradients, attack bounds, unchanged BatchNorm state during
            attack generation, and cleanup. Memory is recorded as telemetry, not a minimum-memory
            gate. Actual OOM, invalid attacks, and numerical failures preserve a failure record.
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

            Train on clean augmented images with species cross-entropy. The reference is 15 epochs,
            float32 MPS, micro-batch 16 and two-step accumulation, AdamW, one-epoch warm-up followed
            by cosine decay. Configuration values remain editable. Atomic epoch checkpoints and
            nested tqdm progress retain the configured final checkpoint without test selection.
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

            Change only training inputs: untargeted PGD-5, `L∞ 4/255`, step `1/255`, uniform random
            start, projection and clipping in raw `[0,1]` pixels before normalization. Match the
            clean arm's initialization, samples, augmentations, updates, and schedule. Neither
            prettier filters nor this finite training attack establishes unrestricted robustness.
            """,
            "adversarial",
        ),
        _code(
            "show(run_stage(config, 'train', full=RUN_FULL_EXPERIMENT, device=device, arm='adversarial'))",
            "train-adversarial",
        ),
        _markdown(
            """
            ## 9. Attacks, corruptions, and risk-aware evaluation

            Evaluate the full official test clean and under registered noise/blur/brightness/contrast.
            Use 100 fixed test images per species (200 total) for paired FGSM and PGD-20×5. Report
            clean overall/macro/per-species accuracy, robust accuracy, and attack success among
            clean-correct images. Finite attacks are lower-bound search, not certification.

            Fit temperature and 90%-coverage confidence threshold on clean calibration only.
            Compare NLL/Brier/ECE, coverage, selective risk, and tie-aware AURC after shift without
            refitting. Low ECE does not mean low error or a safe classifier.
            """,
            "evaluation",
        ),
        _code(
            "show(run_stage(config, 'evaluate', full=RUN_FULL_EXPERIMENT, device=device))",
            "run-evaluation",
        ),
        _markdown(
            """
            ## 10. Learned filters, real image parts, and attribution

            Channels are chosen on **clean calibration**, before inspecting fixed test anchors:
            two cats and two dogs. The figure manifest records their IDs and settings.

            **Read the pictures in this order:**

            1. **conv1 kernels:** actual learned RGB weights. They are not maps of the pet input.
            2. **Low/mid/high activation maximization:** synthetic inputs optimized to excite a
               selected channel. They show a possible high-response stimulus, not a training memory.
            3. **Top real calibration patches:** high-response locations plus receptive-field boxes
               connect channels to real image regions. A late theoretical field can exceed the
               whole image; the clipped box is possible support, not equal pixel importance.
            4. **Anchor activation maps and input-gradient sensitivity:** activation locates response;
               gradient locates local sensitivity. They answer different questions.
            5. **Grad-CAM and occlusion:** Grad-CAM targets the **true-species logit**; occlusion
               measures the drop in the **true-species-vs-other logit margin** after masking a
               region. These are related but distinct scalar objectives: agreement does not
               exactly validate the same score. Targets stay fixed even on misclassified inputs.
               Masking can create out-of-distribution inputs; neither proves causal learning.
            6. **Randomized-weight control and aggregate response charts:** explanation should depend
               on learned parameters. A changed control is necessary evidence, not proof of validity.
               Compare saved response numbers, not brightness from separately normalized panels.

            These figures explain present model behavior. They cannot identify which original
            training image or causal learning event created a filter. No named eye/ear/fur detector
            is verified merely because a picture looks familiar.

            The next cell computes this stage in full mode, or reads only already saved,
            configuration-matched figures in safe mode. Missing binary evidence is never filled
            with an old breed plot. Photo-containing outputs remain local and ignored.
            """,
            "feature-walkthrough",
        ),
        _code(
            """
            import hashlib

            feature_result = run_stage(
                config,
                "represent",
                full=RUN_FULL_EXPERIMENT,
                device=device,
            )
            feature_manifest = ROOT / "results/generated/feature_visualizations.json"
            if (
                isinstance(feature_result, dict)
                and feature_result.get("results_available") is False
            ):
                if feature_manifest.is_file():
                    feature_result = json.loads(feature_manifest.read_text(encoding="utf-8"))
                else:
                    show(feature_result)

            if isinstance(feature_result, dict) and "figures" in feature_result:
                if feature_result.get("config_sha256") != config.sha256:
                    raise RuntimeError("Saved feature figures belong to a different configuration")
                figures = feature_result["figures"]
                if not isinstance(figures, list):
                    raise TypeError("Feature manifest figures must be a list")
                display(
                    Markdown(
                        "### Saved learned-feature walkthrough\\n\\n"
                        f"Evidence completed: {feature_result.get('created_at', 'not recorded')}. "
                        "Pet-photo panels are local-only; source guide stays output-free."
                    )
                )
                for figure in figures:
                    if not isinstance(figure, dict) or not isinstance(figure.get("path"), str):
                        raise TypeError("Invalid feature figure record")
                    figure_path = (ROOT / figure["path"]).resolve()
                    if not figure_path.is_relative_to(ROOT) or not figure_path.is_file():
                        show(
                            {
                                "status": "stale feature evidence",
                                "figure": figure["path"],
                                "reason": "figure is missing or outside the project",
                            }
                        )
                        continue
                    expected_sha256 = figure.get("sha256")
                    try:
                        current_sha256 = hashlib.sha256(figure_path.read_bytes()).hexdigest()
                    except OSError as error:
                        show(
                            {
                                "status": "stale feature evidence",
                                "figure": figure["path"],
                                "reason": f"figure cannot be read: {error}",
                            }
                        )
                        continue
                    if not isinstance(expected_sha256, str) or current_sha256 != expected_sha256:
                        show(
                            {
                                "status": "stale feature evidence",
                                "figure": figure["path"],
                                "reason": "recorded figure hash is missing or mismatched",
                            }
                        )
                        continue
                    sharing = "aggregate/synthetic export candidate" if figure.get("shareable") else "local-only"
                    display(
                        Markdown(
                            f"### {figure.get('arm', '')}: {figure.get('kind', 'feature view')}\\n\\n"
                            f"{figure.get('caption', '')}\\n\\nSharing: **{sharing}**."
                        )
                    )
                    display(Image(filename=str(figure_path), width=1200))
                show(feature_result.get("limitations", []))
            """,
            "run-features",
        ),
        _markdown(
            """
            ## 11. Reports, interpretation, and local artifact inventory

            Reports must use complete matching binary evidence. Interpret actual channel responses
            and class-score changes separately from possible semantic stories. One model pair,
            one split/seed family, ImageNet pretraining, selected anchors, explanation-method limits,
            and finite attacks constrain conclusions.

            A completed picture-rich local result companion is saved under `reports/generated/`
            and can be read without rerunning training. It contains real photographs and remains
            ignored. Only reviewed aggregate charts/synthetic features may be exported publicly.
            Do not save this executed guide over the tracked output-free source.
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
            ## 12. Reproduction and references

            Full run: deliberately set `RUN_FULL_EXPERIMENT = True`, rerun setup, then run cells
            in order. CLI: `.venv/bin/python src/cli.py reproduce --config configs/experiment.yaml
            --device mps`. Existing checkpoint/evidence reuse requires matching provenance.

            [Lee et al. 2009](https://ai.stanford.edu/~ang/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf)
            inspires the low-to-high illustration; this discriminative ResNet does **not** reproduce
            their generative convolutional deep belief network. Original method references:
            [Zeiler/Fergus](https://arxiv.org/abs/1311.2901),
            [Grad-CAM](https://arxiv.org/abs/1610.02391),
            [Adebayo sanity checks](https://arxiv.org/abs/1810.03292), and
            [Distill Feature Visualization](https://distill.pub/2017/feature-visualization/).
            """,
            "reproduction",
        ),
    ]
    if include_public_figures:
        cells.extend(_public_preview_cells())
    notebook = nbformat.v4.new_notebook(
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
    nbformat.validate(notebook)
    return notebook


def build(*, include_public_figures: bool = True) -> Path:
    notebook = make_notebook(include_public_figures=include_public_figures)
    output = ROOT / "notebooks" / "cat_dog_cnn_features.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
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
        "--no-public-preview", action="store_true", help="omit the saved public-safe figure gallery"
    )
    arguments = parser.parse_args()
    print(build(include_public_figures=not arguments.no_public_preview))
