"""Export a compact GitHub-facing result page from verified saved evidence.

This presentation-only command never loads a model, runs an attack, or trains.
"""

# The long lines below are Markdown table rows kept readable as complete records.
# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.io_utils import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402

RESULTS_JSON = ROOT / "docs" / "results.json"
RESULTS_MARKDOWN = ROOT / "docs" / "results.md"
ASSETS = {
    "cosine-drift.png": ROOT / "figures/generated/geometry/cosine-drift.png",
    "knn-retention-by-breed.png": (ROOT / "figures/generated/geometry/knn-retention-by-breed.png"),
}
REPORTS = {
    ROOT / "reports/generated/technical-report.md": ROOT / "reports/technical_report.md",
    ROOT / "reports/generated/robust-vision-long-form-report.md": (
        ROOT / "reports/long_form_report.md"
    ),
}
LIMITATIONS = [
    "One ImageNet-initialized ResNet-18 pair and one split/seed family were evaluated.",
    "Bootstrap intervals quantify paired test-sample uncertainty, not retraining variability.",
    "PGD-20 with five restarts is finite digital attack search, not certified robustness.",
    "PCA, t-SNE, and UMAP distort different relationships and are explanatory only.",
    "Synthetic corruptions and bounded pixel attacks do not establish physical or safe use.",
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _arm_summary(
    arm: dict[str, Any],
    geometry: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    clean = arm["full_test"]["clean"]
    fgsm = arm["attack_subset"]["fgsm"]
    pgd = arm["attack_subset"]["pgd"]
    return {
        "training": {
            "epochs": training["completed_epochs"],
            "optimizer_updates": training["completed_updates"],
            "elapsed_seconds": training["elapsed_this_invocation_seconds"],
        },
        "clean_test": {
            "accuracy": clean["accuracy"],
            "macro_accuracy": clean["macro_accuracy"],
            "nll": clean["risk"]["nll"],
            "brier": clean["risk"]["brier"],
            "ece_15": clean["risk"]["ece_15"],
            "coverage": clean["risk"]["coverage"],
            "selective_risk": clean["risk"]["selective_risk"],
            "aurc": clean["risk"]["aurc"],
        },
        "attack_subset": {
            "clean_accuracy": pgd["clean_accuracy_on_subset"],
            "fgsm_robust_accuracy": fgsm["robust_accuracy"],
            "pgd_robust_accuracy": pgd["robust_accuracy"],
            "pgd_attack_success_among_clean_correct": pgd["attack_success_clean_correct"],
        },
        "representation": {
            "median_cosine_drift": geometry["cosine_drift"]["median"],
            "median_relative_l2_drift": geometry["relative_l2_drift"]["median"],
            "clean_knn_accuracy": geometry["knn"]["clean_accuracy"],
            "pgd_knn_accuracy": geometry["knn"]["pgd_accuracy"],
            "pgd_knn_breed_retention": geometry["knn"]["pgd_retention"],
            "clean_to_pgd_linear_cka": geometry["linear_cka_clean_to_pgd"],
            "clean_nearest_centroid_accuracy": geometry["nearest_centroid"]["clean_accuracy"],
            "pgd_nearest_centroid_accuracy": geometry["nearest_centroid"]["pgd_accuracy"],
        },
    }


def _validate_configuration(config_sha256: str, documents: list[dict[str, Any]]) -> None:
    observed = {document.get("config_sha256") for document in documents}
    if observed != {config_sha256}:
        raise RuntimeError(f"saved evidence has mixed configuration identities: {observed}")


def build() -> tuple[Path, Path]:
    config = load_config(ROOT / "configs" / "experiment.yaml")
    evaluation_path = ROOT / "results/generated/evaluation.json"
    representations_path = ROOT / "results/generated/representations.json"
    summary_path = ROOT / "results/generated/summary.json"
    data_path = ROOT / "artifacts/data/manifest.json"
    preflight_path = ROOT / "artifacts/preflight.json"
    release_path = ROOT / "artifacts/release/local-release-manifest.json"
    standard_training_path = ROOT / "artifacts/training/standard.json"
    adversarial_training_path = ROOT / "artifacts/training/adversarial.json"
    required = [
        evaluation_path,
        representations_path,
        summary_path,
        data_path,
        preflight_path,
        release_path,
        standard_training_path,
        adversarial_training_path,
        *ASSETS.values(),
        *REPORTS,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    evaluation = _read_json(evaluation_path)
    representations = _read_json(representations_path)
    summary = _read_json(summary_path)
    data = _read_json(data_path)
    preflight = _read_json(preflight_path)
    release = _read_json(release_path)
    standard_training = _read_json(standard_training_path)
    adversarial_training = _read_json(adversarial_training_path)
    _validate_configuration(
        config.sha256,
        [
            evaluation,
            representations,
            summary,
            preflight,
            release,
            standard_training["provenance"],
            adversarial_training["provenance"],
        ],
    )
    release_files = release.get("files")
    if not isinstance(release_files, dict):
        raise RuntimeError("release file manifest is invalid")
    for path in [
        evaluation_path,
        representations_path,
        summary_path,
        standard_training_path,
        adversarial_training_path,
        *ASSETS.values(),
        *REPORTS,
    ]:
        relative = str(path.relative_to(ROOT))
        if release_files.get(relative) != sha256_file(path):
            raise RuntimeError(f"saved evidence hash mismatch: {relative}")
    if standard_training.get("status") != "complete":
        raise RuntimeError("standard training is incomplete")
    if adversarial_training.get("status") != "complete":
        raise RuntimeError("adversarial training is incomplete")
    epochs = config.integer("training", "epochs")
    if {
        standard_training.get("completed_epochs"),
        adversarial_training.get("completed_epochs"),
    } != {epochs}:
        raise RuntimeError("training evidence does not match the configured epoch count")

    docs_assets = ROOT / "docs" / "assets"
    docs_assets.mkdir(parents=True, exist_ok=True)
    asset_hashes: dict[str, str] = {}
    for name, source in ASSETS.items():
        destination = docs_assets / name
        shutil.copyfile(source, destination)
        asset_hashes[str(destination.relative_to(ROOT))] = sha256_file(destination)
    report_hashes: dict[str, str] = {}
    for source, destination in REPORTS.items():
        atomic_write_text(destination, source.read_text(encoding="utf-8"))
        report_hashes[str(destination.relative_to(ROOT))] = sha256_file(destination)

    geometry = representations["geometry"]
    standard = evaluation["arms"]["standard"]
    adversarial = evaluation["arms"]["adversarial"]
    primary = representations["primary_hypothesis"]
    pair = representations["difficult_pair_selection"]
    label_to_name = data["integrity"]["label_to_name"]
    first_class = str(pair["first_class"])
    second_class = str(pair["second_class"])
    corruptions = {
        condition: {
            "standard_accuracy": standard["full_test"][condition]["accuracy"],
            "adversarial_accuracy": adversarial["full_test"][condition]["accuracy"],
            "standard_coverage": standard["full_test"][condition]["risk"]["coverage"],
            "adversarial_coverage": adversarial["full_test"][condition]["risk"]["coverage"],
            "standard_selective_risk": standard["full_test"][condition]["risk"]["selective_risk"],
            "adversarial_selective_risk": adversarial["full_test"][condition]["risk"][
                "selective_risk"
            ],
        }
        for condition in sorted(set(standard["full_test"]) - {"clean"})
    }
    public = {
        "schema_version": 1,
        "project": config.value("experiment", "title", str),
        "status": f"completed {epochs}-epoch local reference experiment",
        "observed_at": evaluation["created_at"],
        "protocol": {
            "dataset": config.value("dataset", "name", str),
            "dataset_source": data["source"],
            "dataset_license": data["license"],
            "classes": config.integer("dataset", "classes"),
            "architecture": config.value("model", "architecture", str),
            "pretrained_weights": config.value("model", "weights", str),
            "device": evaluation["device"],
            "dtype": config.value("training", "dtype", str),
            "epochs": epochs,
            "attack_subset_count": data["attack_subset_count"],
            "attack": {
                "targeted": False,
                "norm": config.value("attack", "norm", str),
                "epsilon": config.number("attack", "epsilon"),
                "training_steps": config.integer("attack", "train_steps"),
                "evaluation_steps": config.integer("attack", "evaluation_steps"),
                "evaluation_restarts": config.integer("attack", "evaluation_restarts"),
            },
            "split": {
                "training": data["training_count"],
                "calibration": data["calibration_count"],
                "official_test": data["integrity"]["test_count"],
            },
        },
        "primary_hypothesis": primary,
        "arms": {
            "standard": _arm_summary(standard, geometry["standard"], standard_training),
            "adversarial": _arm_summary(
                adversarial,
                geometry["adversarial"],
                adversarial_training,
            ),
        },
        "registered_corruptions": corruptions,
        "secondary_policy_shift": evaluation["secondary_policy_shift"],
        "difficult_pair": {
            **pair,
            "first_name": label_to_name[first_class],
            "second_name": label_to_name[second_class],
        },
        "provenance": {
            "configuration_sha256": config.sha256,
            "dataset_sha256": data["dataset_content_sha256"],
            "split_sha256": data["split_sha256"],
            "initialization_sha256": release["provenance"]["initialization_sha256"],
            "execution_source_sha256": preflight["current_source_sha256"],
            "evaluation_sha256": sha256_file(evaluation_path),
            "representations_sha256": sha256_file(representations_path),
            "checkpoints": {
                arm: {"sha256": details["sha256"]}
                for arm, details in release["fixed_checkpoints"].items()
            },
            "public_assets": asset_hashes,
            "public_reports": report_hashes,
        },
        "limitations": LIMITATIONS,
    }
    atomic_write_json(RESULTS_JSON, public)

    standard_public = public["arms"]["standard"]
    adversarial_public = public["arms"]["adversarial"]
    rows = [
        (
            condition,
            _pct(values["standard_accuracy"]),
            _pct(values["adversarial_accuracy"]),
        )
        for condition, values in corruptions.items()
    ]
    corruption_rows = "\n".join(
        f"| `{condition}` | {standard_value} | {adversarial_value} |"
        for condition, standard_value, adversarial_value in rows
    )
    limitations = "\n".join(f"- {limitation}" for limitation in LIMITATIONS)
    report = f"""# Results: adversarial representation drift

One matched pair of ImageNet-initialized ResNet-18 models on Oxford-IIIT Pet, trained
for {epochs} epochs in float32 with PyTorch MPS. The standard arm used clean examples;
the comparison arm used five-step `L∞` PGD examples at `epsilon=4/255`.

![Distribution of paired cosine feature drift](assets/cosine-drift.png)

## Registered result

The primary conjunction **{"passed" if primary["primary_supported"] else "did not pass"}**.
PGD-5 training produced both lower median clean-to-PGD cosine drift and higher
five-nearest-neighbour breed retention in the original 512-dimensional features.

| Metric | Standard | PGD-trained |
|---|---:|---:|
| Clean accuracy, full official test | {_pct(standard_public["clean_test"]["accuracy"])} | {_pct(adversarial_public["clean_test"]["accuracy"])} |
| FGSM robust accuracy, 740-image subset | {_pct(standard_public["attack_subset"]["fgsm_robust_accuracy"])} | {_pct(adversarial_public["attack_subset"]["fgsm_robust_accuracy"])} |
| PGD-20×5 robust accuracy, 740-image subset | {_pct(standard_public["attack_subset"]["pgd_robust_accuracy"])} | {_pct(adversarial_public["attack_subset"]["pgd_robust_accuracy"])} |
| Attack success among clean-correct samples | {_pct(standard_public["attack_subset"]["pgd_attack_success_among_clean_correct"])} | {_pct(adversarial_public["attack_subset"]["pgd_attack_success_among_clean_correct"])} |
| Median cosine feature drift | {standard_public["representation"]["median_cosine_drift"]:.4f} | {adversarial_public["representation"]["median_cosine_drift"]:.4f} |
| Median relative-L2 feature drift | {standard_public["representation"]["median_relative_l2_drift"]:.4f} | {adversarial_public["representation"]["median_relative_l2_drift"]:.4f} |
| PGD five-NN breed retention | {_pct(standard_public["representation"]["pgd_knn_breed_retention"])} | {_pct(adversarial_public["representation"]["pgd_knn_breed_retention"])} |
| PGD five-NN accuracy | {_pct(standard_public["representation"]["pgd_knn_accuracy"])} | {_pct(adversarial_public["representation"]["pgd_knn_accuracy"])} |
| Clean-to-PGD linear CKA | {standard_public["representation"]["clean_to_pgd_linear_cka"]:.4f} | {adversarial_public["representation"]["clean_to_pgd_linear_cka"]:.4f} |

The adversarial-minus-standard drift difference was
`{primary["drift_difference_adversarial_minus_standard"]:.6f}`, with a class-stratified
95% bootstrap interval of `[{primary["drift_ci95"][0]:.6f}, {primary["drift_ci95"][1]:.6f}]`.
The neighbour-retention difference was
`{primary["retention_difference_adversarial_minus_standard"] * 100:.2f}` percentage points,
interval `[{primary["retention_ci95"][0] * 100:.2f}, {primary["retention_ci95"][1] * 100:.2f}]`.
These intervals measure paired sample uncertainty, not variation from retraining.

![Five-nearest-neighbour breed retention by breed](assets/knn-retention-by-breed.png)

## Corruptions and clean-fitted confidence policies

| Full-test condition | Standard accuracy | PGD-trained accuracy |
|---|---:|---:|
{corruption_rows}

The standard model had higher absolute accuracy on every registered corruption. On clean
test data, its 90%-target confidence policy realized
{_pct(standard_public["clean_test"]["coverage"])} coverage at
{_pct(standard_public["clean_test"]["selective_risk"])} selective risk. The PGD-trained
policy realized {_pct(adversarial_public["clean_test"]["coverage"])} coverage at
{_pct(adversarial_public["clean_test"]["selective_risk"])} risk. Registered shifts moved
at least one clean-fitted operating point outside its allowed coverage/risk tolerance for
both arms. Confidence is not treated as an out-of-distribution detector.

## Interpretation

Under this protocol, adversarial fine-tuning preserved much more local and global feature
structure and improved finite-attack robust accuracy, while reducing clean and corruption
accuracy. Representation retention and classifier correctness were related but not
equivalent. PCA, t-SNE, UMAP, and the difficult-pair boundary slices in the
[visual results notebook](../notebooks/oxford_pets_results_explained.ipynb) explain the
geometry; the registered decision comes from the original 512-dimensional measurements.

## Limitations

{limitations}

## Evidence and reproduction

[results.json](results.json) contains the reviewed aggregate values and hashes. Raw
photographs, checkpoints, logits, features, and per-sample arrays remain local. The
[configuration](../configs/experiment.yaml), [protocol](PROTOCOL.md),
[implementation](../src/), [output-free guided notebook](../notebooks/oxford_pets_adversarial_representations.ipynb),
[technical report](../reports/technical_report.md), and
[long-form interpretation](../reports/long_form_report.md) are included. A fresh run uses:

```bash
.venv/bin/python src/cli.py reproduce --config configs/experiment.yaml --device mps
```

This command performs the full experiment. Reading this page or the visual results
notebook performs no training, inference, attack, or download.
"""
    atomic_write_text(RESULTS_MARKDOWN, report)
    return RESULTS_MARKDOWN, RESULTS_JSON


if __name__ == "__main__":
    markdown, aggregate = build()
    print(markdown)
    print(aggregate)
