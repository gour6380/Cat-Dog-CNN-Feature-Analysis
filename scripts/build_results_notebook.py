"""Build a visual, read-only explanation from saved experiment evidence."""

# Markdown prose remains readable in the generated notebook.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import base64
import json
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.eda import generate_eda  # noqa: E402
from src.io_utils import atomic_write_json, sha256_file, utc_now  # noqa: E402

OUTPUT = ROOT / "notebooks" / "oxford_pets_results_explained.ipynb"

FIGURE_GROUPS = [
    (
        "5. Original 512-D geometry",
        "The box plot summarizes paired 512-D cosine drift. The breed chart shows how "
        "often the five nearest clean-calibration features still carry the query breed.",
        [
            "figures/generated/geometry/cosine-drift.png",
            "figures/generated/geometry/knn-retention-by-breed.png",
        ],
    ),
    (
        "6. PCA: dominant linear variance",
        "Each PCA is fitted on that model's clean calibration features. Compare clean and "
        "PGD within a panel; raw coordinates are not comparable between models.",
        [
            "figures/generated/projections/pca-standard.png",
            "figures/generated/projections/pca-adversarial.png",
        ],
    ),
    (
        "7. t-SNE: local relationships in a joint sample",
        "Clean and PGD features are embedded together within each model. Apparent islands and "
        "distances are explanatory, not quantitative robustness evidence.",
        [
            "figures/generated/projections/tsne-standard.png",
            "figures/generated/projections/tsne-adversarial.png",
        ],
    ),
    (
        "8. UMAP: transformed calibration neighbourhoods",
        "Each UMAP learns a clean-calibration graph and transforms the paired test features. "
        "Its geometry answers a different visualization question from PCA or t-SNE.",
        [
            "figures/generated/projections/umap-standard.png",
            "figures/generated/projections/umap-adversarial.png",
        ],
    ),
    (
        "9. Difficult-pair classifier slices",
        "The line is exact only for the selected two logits inside that model's own PCA plane. "
        "It is neither an input-space boundary nor the complete 37-class decision surface.",
        [
            "figures/generated/boundaries/pair-boundary-standard.png",
            "figures/generated/boundaries/pair-boundary-adversarial.png",
        ],
    ),
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _points(value: float) -> str:
    return f"{value * 100:+.2f} percentage points"


def _training_seconds(training: dict[str, Any]) -> float:
    """Return the complete recorded epoch time, including resumed invocations."""

    history = training.get("history")
    if not isinstance(history, list) or not history:
        raise RuntimeError("training history is missing")
    values = [record.get("seconds") for record in history if isinstance(record, dict)]
    if len(values) != len(history) or any(not isinstance(value, int | float) for value in values):
        raise RuntimeError("training history contains an invalid duration")
    return sum(float(value) for value in values)


def _highest_risk_shift(arm: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    full_test = arm.get("full_test")
    if not isinstance(full_test, dict):
        raise RuntimeError("full-test evidence is missing")
    shifted = [
        (str(name), result)
        for name, result in full_test.items()
        if name != "clean" and isinstance(result, dict)
    ]
    if not shifted:
        raise RuntimeError("registered shift evidence is missing")
    return max(shifted, key=lambda item: float(item[1]["risk"]["selective_risk"]))


def _table(headers: Sequence[object], rows: Sequence[Sequence[object]]) -> str:
    def line(values: Sequence[object]) -> str:
        return "| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers)), *map(line, rows)])


def _markdown(source: str, identifier: str) -> Any:
    cell = nbformat.v4.new_markdown_cell(  # type: ignore[no-untyped-call]
        textwrap.dedent(source).strip()
    )
    cell["id"] = identifier
    return cell


def _figure_cell(
    title: str,
    caption: str,
    relative_paths: list[str],
    identifier: str,
) -> Any:
    if len(relative_paths) != 2:
        raise ValueError("figure comparison cells require exactly two images")
    first, second = (ROOT / value for value in relative_paths)
    for path in (first, second):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = (
        f"## {title}\n\n{caption}\n\n"
        "| Standard fine-tuning | PGD-5 fine-tuning |\n"
        "|---|---|\n"
        f"| ![Standard: {title}](attachment:{first.name}) "
        f"| ![PGD-trained: {title}](attachment:{second.name}) |\n"
    )
    cell = nbformat.v4.new_markdown_cell(source)  # type: ignore[no-untyped-call]
    cell["id"] = identifier
    cell["attachments"] = {
        first.name: {"image/png": base64.b64encode(first.read_bytes()).decode("ascii")},
        second.name: {"image/png": base64.b64encode(second.read_bytes()).decode("ascii")},
    }
    return cell


def _eda_cell(eda: dict[str, Any]) -> Any:
    figures = eda["figures"]
    if not isinstance(figures, dict):
        raise TypeError("EDA figure map is invalid")
    relative_paths = [
        str(figures["split_and_species"]),
        str(figures["breed_balance"]),
        str(figures["image_geometry"]),
    ]
    paths = [ROOT / relative for relative in relative_paths]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    record_summary = eda["record_summary"]
    geometry = eda["image_geometry"]["overall"]
    split_counts = record_summary["counts_by_split"]
    breed_species = record_summary["breeds_by_species"]
    orientations = geometry["orientation"]
    aspect = geometry["aspect_ratio_width_over_height"]
    width = geometry["width_pixels"]
    height = geometry["height_pixels"]
    total = geometry["count"]
    source = (
        "## 2. Exploratory data analysis\n\n"
        "EDA checks what the model receives before interpreting accuracy or representations. "
        "It uses the registered split metadata and stored image headers; no original pet "
        "photograph is embedded.\n\n"
        + _table(
            ["Dataset property", "Observed value"],
            [
                ["Registered images", f"{total:,}"],
                ["Prediction classes", f"{record_summary['classes']} breeds"],
                [
                    "Breed families",
                    f"{breed_species['cat']} cat breeds · {breed_species['dog']} dog breeds",
                ],
                [
                    "Training / calibration / test",
                    f"{split_counts['training']:,} / {split_counts['calibration']:,} / "
                    f"{split_counts['test']:,}",
                ],
                [
                    "Median stored width × height",
                    f"{width['median']:.0f} × {height['median']:.0f} px",
                ],
                [
                    "Aspect ratio, median [IQR]",
                    f"{aspect['median']:.2f} [{aspect['q25']:.2f}, {aspect['q75']:.2f}]",
                ],
                [
                    "Portrait / square / landscape",
                    f"{orientations['portrait'] / total:.1%} / "
                    f"{orientations['square'] / total:.1%} / "
                    f"{orientations['landscape'] / total:.1%}",
                ],
            ],
        )
        + "\n\n"
        "| Partition and species balance | Per-breed balance |\n"
        "|---|---|\n"
        f"| ![Split and species EDA](attachment:{paths[0].name}) "
        f"| ![Breed balance EDA](attachment:{paths[1].name}) |\n\n"
        "### Stored image dimensions and aspect ratios\n\n"
        f"![Image geometry EDA](attachment:{paths[2].name})\n\n"
        "**What this changes:** the official test set is slightly smaller than trainval, "
        "but every breed remains represented. Cats and dogs are not balanced as breed families, "
        "so species coloring is contextual only; it is not a second prediction task. The varied "
        "source geometry also explains why the registered resize/crop transform is necessary."
    )
    cell = nbformat.v4.new_markdown_cell(source)  # type: ignore[no-untyped-call]
    cell["id"] = "eda"
    cell["attachments"] = {
        path.name: {"image/png": base64.b64encode(path.read_bytes()).decode("ascii")}
        for path in paths
    }
    return cell


def _validate_evidence() -> dict[str, Any]:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    paths = {
        "summary": ROOT / "results/generated/summary.json",
        "evaluation": ROOT / "results/generated/evaluation.json",
        "representations": ROOT / "results/generated/representations.json",
        "standard_training": ROOT / "artifacts/training/standard.json",
        "adversarial_training": ROOT / "artifacts/training/adversarial.json",
        "data": ROOT / "artifacts/data/manifest.json",
        "figures": ROOT / "artifacts/representations/figures.json",
        "release": ROOT / "artifacts/release/local-release-manifest.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    evidence = {name: _read_json(path) for name, path in paths.items()}
    hashes = {
        config.sha256,
        evidence["summary"].get("config_sha256"),
        evidence["evaluation"].get("config_sha256"),
        evidence["representations"].get("config_sha256"),
        evidence["standard_training"].get("provenance", {}).get("config_sha256"),
        evidence["adversarial_training"].get("provenance", {}).get("config_sha256"),
        evidence["release"].get("config_sha256"),
    }
    if hashes != {config.sha256}:
        raise RuntimeError("saved evidence does not match the current configuration")
    epochs = config.integer("training", "epochs")
    for arm in ("standard", "adversarial"):
        training = evidence[f"{arm}_training"]
        if training.get("status") != "complete" or training.get("completed_epochs") != epochs:
            raise RuntimeError(f"{arm} training is incomplete for epoch {epochs}")
    if evidence["summary"].get("primary") != evidence["representations"].get("primary_hypothesis"):
        raise RuntimeError("summary and representation decisions disagree")
    recorded_hashes = evidence["figures"].get("figure_sha256")
    if not isinstance(recorded_hashes, dict):
        raise RuntimeError("figure manifest is invalid")
    release_files = evidence["release"].get("files")
    if not isinstance(release_files, dict):
        raise RuntimeError("release file manifest is invalid")
    required_files = [path for _, _, group in FIGURE_GROUPS for path in group]
    required_files.extend(
        [
            "results/generated/summary.json",
            "results/generated/evaluation.json",
            "results/generated/representations.json",
            "artifacts/training/standard.json",
            "artifacts/training/adversarial.json",
        ]
    )
    for relative in required_files:
        path = ROOT / relative
        actual = sha256_file(path)
        expected = (
            recorded_hashes.get(relative)
            if relative.startswith("figures/")
            else release_files.get(relative)
        )
        if actual != expected:
            raise RuntimeError(f"saved evidence hash mismatch: {relative}")
    return {"config": config, **evidence}


def build(*, force: bool = False) -> Path:
    if OUTPUT.exists() and not force:
        raise FileExistsError(f"{OUTPUT} exists; pass --force to rebuild it")
    evidence = _validate_evidence()
    config = evidence["config"]
    eda = generate_eda(config, progress=False, force=force)
    evaluation = evidence["evaluation"]
    representations = evidence["representations"]
    primary = representations["primary_hypothesis"]
    geometry = representations["geometry"]
    standard = evaluation["arms"]["standard"]
    adversarial = evaluation["arms"]["adversarial"]
    standard_geometry = geometry["standard"]
    adversarial_geometry = geometry["adversarial"]
    epochs = config.integer("training", "epochs")
    updates = evidence["standard_training"]["completed_updates"]
    if evidence["adversarial_training"]["completed_updates"] != updates:
        raise RuntimeError("the two training arms have different optimizer-update counts")
    standard_seconds = _training_seconds(evidence["standard_training"])
    adversarial_seconds = _training_seconds(evidence["adversarial_training"])
    attack_label = (
        f"PGD-{config.integer('attack', 'evaluation_steps')}×"
        f"{config.integer('attack', 'evaluation_restarts')}"
    )
    run_name = "one-epoch pilot" if epochs == 1 else f"{epochs}-epoch reference run"
    verdict = "passed" if primary["primary_supported"] else "did not pass"
    standard_pgd = standard["attack_subset"]["pgd"]
    adversarial_pgd = adversarial["attack_subset"]["pgd"]
    standard_clean = standard["full_test"]["clean"]
    adversarial_clean = adversarial["full_test"]["clean"]
    clean_difference = adversarial_clean["accuracy"] - standard_clean["accuracy"]
    robust_difference = adversarial_pgd["robust_accuracy"] - standard_pgd["robust_accuracy"]
    standard_worst_name, standard_worst = _highest_risk_shift(standard)
    adversarial_worst_name, adversarial_worst = _highest_risk_shift(adversarial)
    shift_names = sorted(set(standard["full_test"]) - {"clean"})
    standard_higher_on_all_shifts = all(
        standard["full_test"][name]["accuracy"] > adversarial["full_test"][name]["accuracy"]
        for name in shift_names
    )
    corruption_comparison = (
        "The standard model also had higher absolute accuracy on every registered corruption."
        if standard_higher_on_all_shifts
        else "Neither model had higher absolute accuracy on every registered corruption."
    )
    primary_decision = (
        "Both registered interval signs met the conjunctive decision rule."
        if primary["primary_supported"]
        else "The registered conjunctive decision rule was not satisfied."
    )
    headline_rows = [
        [
            "Clean accuracy · full official test",
            _pct(standard["full_test"]["clean"]["accuracy"]),
            _pct(adversarial["full_test"]["clean"]["accuracy"]),
        ],
        [
            "Clean accuracy · 740-image attack subset",
            _pct(standard["attack_subset"]["pgd"]["clean_accuracy_on_subset"]),
            _pct(adversarial["attack_subset"]["pgd"]["clean_accuracy_on_subset"]),
        ],
        [
            "FGSM robust accuracy · attack subset",
            _pct(standard["attack_subset"]["fgsm"]["robust_accuracy"]),
            _pct(adversarial["attack_subset"]["fgsm"]["robust_accuracy"]),
        ],
        [
            f"{attack_label} robust accuracy · attack subset",
            _pct(standard["attack_subset"]["pgd"]["robust_accuracy"]),
            _pct(adversarial["attack_subset"]["pgd"]["robust_accuracy"]),
        ],
    ]
    geometry_rows = [
        [
            "Median cosine drift ↓",
            f"{standard_geometry['cosine_drift']['median']:.4f}",
            f"{adversarial_geometry['cosine_drift']['median']:.4f}",
        ],
        [
            "Median relative-L2 drift ↓",
            f"{standard_geometry['relative_l2_drift']['median']:.4f}",
            f"{adversarial_geometry['relative_l2_drift']['median']:.4f}",
        ],
        [
            "Clean 5-NN accuracy ↑",
            _pct(standard_geometry["knn"]["clean_accuracy"]),
            _pct(adversarial_geometry["knn"]["clean_accuracy"]),
        ],
        [
            f"{attack_label} 5-NN accuracy ↑",
            _pct(standard_geometry["knn"]["pgd_accuracy"]),
            _pct(adversarial_geometry["knn"]["pgd_accuracy"]),
        ],
        [
            f"{attack_label} 5-NN breed retention ↑",
            _pct(standard_geometry["knn"]["pgd_retention"]),
            _pct(adversarial_geometry["knn"]["pgd_retention"]),
        ],
        [
            "Clean→PGD linear CKA ↑",
            f"{standard_geometry['linear_cka_clean_to_pgd']:.4f}",
            f"{adversarial_geometry['linear_cka_clean_to_pgd']:.4f}",
        ],
    ]
    risk_rows = [
        [
            "Clean calibrated coverage",
            _pct(standard["full_test"]["clean"]["risk"]["coverage"]),
            _pct(adversarial["full_test"]["clean"]["risk"]["coverage"]),
        ],
        [
            "Clean selective risk ↓",
            _pct(standard["full_test"]["clean"]["risk"]["selective_risk"]),
            _pct(adversarial["full_test"]["clean"]["risk"]["selective_risk"]),
        ],
        [
            "Clean 15-bin ECE ↓",
            f"{standard['full_test']['clean']['risk']['ece_15']:.4f}",
            f"{adversarial['full_test']['clean']['risk']['ece_15']:.4f}",
        ],
        [
            "Clean AURC ↓",
            f"{standard['full_test']['clean']['risk']['aurc']:.4f}",
            f"{adversarial['full_test']['clean']['risk']['aurc']:.4f}",
        ],
    ]
    first_class = str(representations["difficult_pair_selection"]["first_class"])
    second_class = str(representations["difficult_pair_selection"]["second_class"])
    label_to_name = evidence["data"]["integrity"]["label_to_name"]
    pair = f"{label_to_name[first_class]} (class {first_class}) and {label_to_name[second_class]} (class {second_class})"
    chance = 1.0 / config.integer("dataset", "classes")
    standard_pca_variance = sum(
        representations["projection_settings"]["standard"]["pca_explained_variance_ratio"]
    )
    adversarial_pca_variance = sum(
        representations["projection_settings"]["adversarial"]["pca_explained_variance_ratio"]
    )
    figure_groups = [
        (
            FIGURE_GROUPS[0][0],
            f"The standard feature drift is centered near "
            f"{standard_geometry['cosine_drift']['median']:.3f}, while the PGD-trained drift "
            f"is centered near {adversarial_geometry['cosine_drift']['median']:.3f}. Standard "
            f"five-neighbour breed retention is almost zero across breeds; PGD-trained retention "
            f"is higher but varies substantially by class. These are the decision-useful original-space "
            f"measurements, not projection scores.",
            FIGURE_GROUPS[0][2],
        ),
        (
            FIGURE_GROUPS[1][0],
            f"The first two clean-calibration PCs explain only {_pct(standard_pca_variance)} of "
            f"standard variance and {_pct(adversarial_pca_variance)} of PGD-trained variance. "
            f"The standard attacked cloud moves to a much wider, differently located region; the "
            f"PGD-trained clean and attacked layouts occupy more similar ranges. Read only within "
            f"each model—the two PCA bases are independent.",
            FIGURE_GROUPS[1][2],
        ),
        (
            FIGURE_GROUPS[2][0],
            "For the standard model, clean breed-local groups become tight mixed-colour attacked "
            "islands. The PGD-trained clean and attacked samples occupy broadly similar manifold "
            "shapes, although breed mixing remains. t-SNE was fitted jointly within each model and "
            "emphasizes local neighbourhoods; island spacing and global axes are not evidence.",
            FIGURE_GROUPS[2][2],
        ),
        (
            FIGURE_GROUPS[3][0],
            "The standard model's attacked samples transform into compact mixed-breed packets that "
            "differ sharply from its clean graph. The PGD-trained clean and attacked views retain "
            "more of the same broad layout, with visible class overlap. UMAP uses a clean-calibration "
            "reference graph and explains neighbourhood change; it does not validate robustness.",
            FIGURE_GROUPS[3][2],
        ),
        (
            FIGURE_GROUPS[4][0],
            f"The registered difficult pair is {pair}. In both panels, the exact two-logit equality "
            f"line lies away from most displayed points. That is evidence about this two-PC slice: "
            f"the classifier can depend strongly on the 510 hidden feature directions. It is not a "
            f"complete 37-class or image-space decision boundary.",
            FIGURE_GROUPS[4][2],
        ),
    ]

    cells = [
        _markdown(
            f"""
            # Oxford Pets adversarial representations: results explained
            ## A visual reading of the {run_name}

            **Completed local run · {epochs} epoch(s) · {updates} optimizer updates per arm ·
            configuration `{config.sha256[:16]}…`**

            This companion is built from saved machine-readable experiment evidence, registered
            dataset metadata, and verified aggregate figures. It can be read without running any cell.
            It performs no training, inference, attack, download, or public action, and it contains
            no original pet photographs.

            **Bottom line:** the registered representation criterion {verdict}. Under {attack_label},
            robust accuracy was {_pct(standard_pgd["robust_accuracy"])} for standard fine-tuning and
            {_pct(adversarial_pgd["robust_accuracy"])} for PGD-5 fine-tuning. The representation result
            coincided with partial—not complete—robust recognition and a substantial clean-accuracy cost.
            """,
            "title",
        ),
        _markdown(
            "## 1. What was actually run\n\n"
            + _table(
                ["Control", "Recorded value"],
                [
                    ["Dataset", "Oxford-IIIT Pet · 37 breeds"],
                    ["Training / calibration / official test", "2,944 / 736 / 3,669 images"],
                    ["Training budget", f"{epochs} epoch(s) · {updates} updates per arm"],
                    [
                        "Standard summed epoch time",
                        f"{standard_seconds / 60:.1f} minutes ({standard_seconds:.1f} seconds)",
                    ],
                    [
                        "PGD-5 summed epoch time",
                        f"{adversarial_seconds / 60:.1f} minutes ({adversarial_seconds:.1f} seconds)",
                    ],
                    ["Attack subset", f"740 images · 20 per breed · FGSM and {attack_label}"],
                    [
                        "Feature analyzed",
                        "512-D vector after ResNet average pooling, before its head",
                    ],
                ],
            )
            + (
                "\n\nBecause `warmup_epochs` also equals 1, this entire pilot stayed inside "
                "learning-rate warm-up; it did not enter cosine decay."
                if epochs == 1
                else ""
            ),
            "run-context",
        ),
        _eda_cell(eda),
        _markdown(
            "## 3. Prediction results before interpreting embeddings\n\n"
            + _table(["Metric", "Standard", "PGD-trained"], headline_rows)
            + f"\n\n**Observation:** PGD-5 training changed full-test clean accuracy by "
            f"**{_points(clean_difference)}** and {attack_label} robust accuracy by "
            f"**{_points(robust_difference)}** relative to standard fine-tuning. Attack success among "
            f"clean-correct samples was {_pct(standard_pgd['attack_success_clean_correct'])} versus "
            f"{_pct(adversarial_pgd['attack_success_clean_correct'])}. {corruption_comparison} "
            "**Interpretation:** the adversarial arm traded clean and corruption accuracy for partial "
            "resistance to the registered bounded attack; it did not establish general robustness.",
            "prediction-results",
        ),
        _markdown(
            "## 4. The registered 512-D representation test\n\n"
            + _table(["Original-feature metric", "Standard", "PGD-trained"], geometry_rows)
            + f"\n\nThe adversarial-minus-standard median-drift difference was "
            f"**{primary['drift_difference_adversarial_minus_standard']:.4f}** with 95% interval "
            f"`[{primary['drift_ci95'][0]:.4f}, {primary['drift_ci95'][1]:.4f}]`. The retention "
            f"difference was **{primary['retention_difference_adversarial_minus_standard'] * 100:.2f} "
            f"percentage points** with interval "
            f"`[{primary['retention_ci95'][0] * 100:.2f}, "
            f"{primary['retention_ci95'][1] * 100:.2f}]`. {primary_decision}\n\n"
            f"**Practical check:** PGD-trained neighbour retention was "
            f"{_pct(adversarial_geometry['knn']['pgd_retention'])}; a balanced 37-class chance "
            f"reference is approximately {_pct(chance)}. Its {attack_label} five-NN accuracy was "
            f"{_pct(adversarial_geometry['knn']['pgd_accuracy'])}, alongside model robust accuracy "
            f"of {_pct(adversarial_pgd['robust_accuracy'])}. The geometry retained useful class "
            "information, but neither metric approached the model's clean performance.",
            "registered-result",
        ),
    ]
    cells.extend(
        _figure_cell(title, caption, paths, f"figure-{index}")
        for index, (title, caption, paths) in enumerate(figure_groups, 1)
    )
    cells.extend(
        [
            _markdown(
                "## 10. Calibration and selective prediction\n\n"
                + _table(["Clean-test policy metric", "Standard", "PGD-trained"], risk_rows)
                + f"\n\nOn clean test data, the standard policy accepted "
                f"{_pct(standard_clean['risk']['coverage'])} with "
                f"{_pct(standard_clean['risk']['selective_risk'])} selective risk; the PGD-trained "
                f"policy accepted {_pct(adversarial_clean['risk']['coverage'])} with "
                f"{_pct(adversarial_clean['risk']['selective_risk'])} risk. The highest selective-risk "
                f"registered shifts were `{standard_worst_name}` for standard fine-tuning "
                f"({_pct(standard_worst['risk']['coverage'])} coverage, "
                f"{_pct(standard_worst['risk']['selective_risk'])} risk) and "
                f"`{adversarial_worst_name}` for PGD-5 fine-tuning "
                f"({_pct(adversarial_worst['risk']['coverage'])} coverage, "
                f"{_pct(adversarial_worst['risk']['selective_risk'])} risk). Both registered "
                "policy-shift checks fired. Calibration on clean inputs did not preserve the target "
                "operating point after every shift.",
                "risk",
            ),
            _markdown(
                f"""
                ## 11. How to read the difficult pair

                The registered tie-break selected **{pair}**. The two boundary figures show their
                linear-head equality line only after restricting the 512-D representation to two PCA
                coordinates and fixing all hidden coordinates at the calibration mean. A steep or distant
                line can be far from the displayed point cloud because the other 510 feature directions
                are fixed at their calibration mean. It does not show the full image-space boundary.
                """,
                "boundary-reading",
            ),
            _markdown(
                f"""
                ## 12. Defensible conclusion and limits

                **Observation:** during this {run_name}, PGD-5 training reduced median cosine drift
                from {standard_geometry["cosine_drift"]["median"]:.4f} to
                {adversarial_geometry["cosine_drift"]["median"]:.4f}, increased five-neighbour breed
                retention from {_pct(standard_geometry["knn"]["pgd_retention"])} to
                {_pct(adversarial_geometry["knn"]["pgd_retention"])}, and increased {attack_label}
                robust accuracy from {_pct(standard_pgd["robust_accuracy"])} to
                {_pct(adversarial_pgd["robust_accuracy"])}. Clean accuracy decreased from
                {_pct(standard_clean["accuracy"])} to {_pct(adversarial_clean["accuracy"])}.

                **Interpretation:** the adversarial arm retained substantially more local and global
                representation structure under this attack and gained partial prediction robustness,
                while specializing away from clean and registered-corruption performance.
                Representation stability and prediction robustness were related but not identical.

                **Main limitation:** this is one ImageNet-initialized architecture, one split and seed
                family, and one finite digital threat model. The bootstrap intervals quantify paired
                test-sample uncertainty, not retraining variability. PCA, t-SNE, and UMAP remain
                explanatory projections rather than evidence of intrinsic representation quality.

                **Stop decision:** the registered Week 3 experiment is complete. No retraining is needed
                for this release candidate. A new seed, architecture, or stronger evaluator would be a
                separately registered extension rather than a repair to these results.
                """,
                "conclusion",
            ),
            _markdown(
                f"""
                ## 13. Evidence map

                - Numerical summary: `results/generated/summary.json`
                - Aggregate EDA: `results/generated/eda.json`
                - Full evaluation: `results/generated/evaluation.json`
                - Representation metrics: `results/generated/representations.json`
                - Aggregate figures: `figures/generated/`
                - Technical report: `reports/generated/technical-report.md`
                - Shareable result summary: `docs/results.md`
                - Shareable aggregate JSON: `docs/results.json`
                - Local release manifest: `artifacts/release/local-release-manifest.json`
                - Configuration SHA-256: `{config.sha256}`

                The notebook embeds the exact PNG bytes verified against the saved EDA, figure, and
                release manifests. It contains no pet photographs and authorizes no publication.
                """,
                "evidence",
            ),
        ]
    )
    notebook = nbformat.v4.new_notebook(  # type: ignore[no-untyped-call]
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (Oxford Pets Adversarial Representations)",
                "language": "python",
                "name": "oxford-pets-adversarial-representations",
            },
            "language_info": {"name": "python", "version": "3.13.15"},
            "result_snapshot": {
                "config_sha256": config.sha256,
                "epochs": epochs,
                "evaluation_sha256": sha256_file(ROOT / "results/generated/evaluation.json"),
                "representations_sha256": sha256_file(
                    ROOT / "results/generated/representations.json"
                ),
                "standard_checkpoint_sha256": evidence["release"]["fixed_checkpoints"]["standard"][
                    "sha256"
                ],
                "adversarial_checkpoint_sha256": evidence["release"]["fixed_checkpoints"][
                    "adversarial"
                ]["sha256"],
                "generated_at": utc_now(),
                "source": "verified saved evidence plus aggregate EDA; no model computation",
            },
        },
    )
    nbformat.validate(notebook)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT)  # type: ignore[no-untyped-call]
    atomic_write_json(
        ROOT / "artifacts/notebooks/results-explained.json",
        {
            "created_at": utc_now(),
            "config_sha256": config.sha256,
            "notebook": str(OUTPUT.relative_to(ROOT)),
            "notebook_sha256": sha256_file(OUTPUT),
            "embedded_figures": [
                *[str(path) for path in eda["figures"].values()],
                *[path for _, _, paths in FIGURE_GROUPS for path in paths],
            ],
            "contains_original_photographs": False,
        },
    )
    return OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace an existing result notebook")
    arguments = parser.parse_args()
    print(build(force=arguments.force))


if __name__ == "__main__":
    main()
