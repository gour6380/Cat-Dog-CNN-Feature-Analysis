"""Evidence-backed reports and local release manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from src.config import ExperimentConfig
from src.io_utils import (
    atomic_write_json,
    atomic_write_text,
    environment_snapshot,
    git_state,
    sha256_file,
    source_hash,
    utc_now,
)
from src.progress import status
from src.training import checkpoint_path, experiment_provenance


class ReportError(RuntimeError):
    """Required measured evidence is missing."""


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReportError(f"required measured result is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReportError(f"invalid result document: {path}")
    return cast(dict[str, Any], value)


def _percent(value: object) -> str:
    return "n/a" if value is None else f"{100.0 * float(cast(float, value)):.2f}%"


def _number(value: object, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(cast(float, value)):.{digits}f}"


def _condition_table(arm: str, evaluation: dict[str, Any]) -> str:
    rows = [
        "| Condition | Accuracy | Macro accuracy | Coverage | Selective risk | ECE | AURC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    conditions = evaluation["arms"][arm]["full_test"]
    for condition, metrics in conditions.items():
        risk = metrics["risk"]
        rows.append(
            "| "
            + " | ".join(
                (
                    condition,
                    _percent(metrics["accuracy"]),
                    _percent(metrics["macro_accuracy"]),
                    _percent(risk["coverage"]),
                    _percent(risk["selective_risk"]),
                    _number(risk["ece_15"]),
                    _number(risk["aurc"]),
                )
            )
            + " |"
        )
    return "\n".join(rows)


def _attack_table(evaluation: dict[str, Any]) -> str:
    rows = [
        "| Arm | Clean subset | FGSM robust | PGD-20×5 robust | PGD success among clean-correct |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("standard", "adversarial"):
        attacks = evaluation["arms"][arm]["attack_subset"]
        rows.append(
            f"| {arm} | {_percent(attacks['pgd20x5']['clean_accuracy_on_subset'])} | "
            f"{_percent(attacks['fgsm']['robust_accuracy'])} | "
            f"{_percent(attacks['pgd20x5']['robust_accuracy'])} | "
            f"{_percent(attacks['pgd20x5']['attack_success_clean_correct'])} |"
        )
    return "\n".join(rows)


def _geometry_table(representations: dict[str, Any]) -> str:
    rows = [
        "| Arm | Median cosine drift | Median relative-L2 | "
        "PGD 5-NN accuracy | PGD 5-NN retention | CKA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("standard", "adversarial"):
        geometry = representations["geometry"][arm]
        rows.append(
            f"| {arm} | {_number(geometry['cosine_drift']['median'])} | "
            f"{_number(geometry['relative_l2_drift']['median'])} | "
            f"{_percent(geometry['five_nn']['pgd_accuracy'])} | "
            f"{_percent(geometry['five_nn']['pgd_retention'])} | "
            f"{_number(geometry['linear_cka_clean_to_pgd'])} |"
        )
    return "\n".join(rows)


def _claim(primary: dict[str, Any]) -> str:
    if primary["primary_supported"]:
        return (
            "Both registered criteria passed: under this fixed finite digital attack, the "
            "adversarially trained model retained the measured penultimate structure better."
        )
    return (
        "The conjunctive primary hypothesis was not supported. Any favorable individual metric "
        "is reported as mixed evidence rather than promoted to a robustness conclusion."
    )


def _technical_report(
    config: ExperimentConfig, evaluation: dict[str, Any], representations: dict[str, Any]
) -> str:
    primary = representations["primary_hypothesis"]
    return f"""# Technical report: adversarial representation drift in Oxford-IIIT Pet

Generated from fixed machine-readable evidence on {utc_now()}.

## Registered decision

{_claim(primary)} The adversarial-minus-standard median cosine-drift difference was
`{_number(primary["drift_difference_adversarial_minus_standard"], 6)}` with 95% interval
`[{_number(primary["drift_ci95"][0], 6)}, {_number(primary["drift_ci95"][1], 6)}]`.
The five-neighbour retention difference was
`{_number(primary["retention_difference_adversarial_minus_standard"], 6)}` with 95% interval
`[{_number(primary["retention_ci95"][0], 6)}, {_number(primary["retention_ci95"][1], 6)}]`.

## Classification and finite attacks

{_attack_table(evaluation)}

PGD-20×5 is a finite untargeted `L∞` evaluation at `epsilon=4/255`; it is not a proof
against stronger, adaptive, physical, or unrestricted attacks. Attack success uses only
samples classified correctly before attack. Both models are their fixed epoch-15 states.

## Original-feature geometry

{_geometry_table(representations)}

All values above use the original 512-dimensional penultimate features. PCA, t-SNE,
and UMAP figures are explanatory and their independently fitted model coordinates are
not compared numerically.

## Standard model: full official test and registered shifts

{_condition_table("standard", evaluation)}

## Adversarial model: full official test and registered shifts

{_condition_table("adversarial", evaluation)}

Temperatures and 90%-coverage thresholds were fitted only on the deterministic clean
calibration partition. They were not refitted to the official test set or any attack or
corruption. Confidence is not presented as an out-of-distribution detector.

## Difficult-pair boundary

The shared pair was classes `{representations["difficult_pair_selection"]["first_class"]}`
and `{representations["difficult_pair_selection"]["second_class"]}`, selected by maximum
symmetric clean-calibration confusion, then minimum centroid distance and class ID. Each
line is the exact equality of those two linear-head logits restricted to that model's own
PCA plane with undisplayed coordinates fixed at the calibration mean. It is not an input-
space boundary or a complete 37-class decision map.

## Limitations and competing explanations

- ImageNet initialization may account for substantial geometry before fine-tuning.
- Oxford-IIIT Pet has limited examples per breed; a single split and seed limit scope.
- PCA, t-SNE, and UMAP distort different relationships and cannot validate robustness.
- PGD-20×5 is finite; failure to find an adversarial example does not prove none exists.
- Synthetic pixel corruptions do not establish physical robustness or safe recognition.
- Optimization dynamics—not adversarial invariance alone—may explain observed geometry.

Configuration SHA-256: `{config.sha256}`. Full provenance and artifact hashes are in
`local-release-manifest.json`.
"""


def _long_form(evaluation: dict[str, Any], representations: dict[str, Any]) -> str:
    primary = representations["primary_hypothesis"]
    standard_clean = evaluation["arms"]["standard"]["full_test"]["clean"]["accuracy"]
    adversarial_clean = evaluation["arms"]["adversarial"]["full_test"]["clean"]["accuracy"]
    standard_robust = evaluation["arms"]["standard"]["attack_subset"]["pgd20x5"]["robust_accuracy"]
    adversarial_robust = evaluation["arms"]["adversarial"]["attack_subset"]["pgd20x5"][
        "robust_accuracy"
    ]
    return f"""# What adversarial training changed inside a 37-breed pet classifier

Two ResNet-18 models began from the same ImageNet tensors, the same newly initialized
37-class head, the same ordered samples and deterministic crops, and the same number of
optimizer updates. The only experimental arm difference was whether training examples
were clean or generated by five-step bounded PGD.

The registered question was not whether one embedding picture looked cleaner. It was
whether paired features moved less under PGD and kept more same-breed neighbours in the
original 512-dimensional space. {_claim(primary)}

Clean official-test accuracy was {_percent(standard_clean)} for standard fine-tuning and
{_percent(adversarial_clean)} for adversarial fine-tuning. On the fixed 740-image finite
PGD-20×5 subset, robust accuracy was {_percent(standard_robust)} and
{_percent(adversarial_robust)}, respectively. These clean results are trade-offs, not a
pre-registered directional win.

The visualization lesson matters as much as the result. PCA preserves dominant linear
variance; t-SNE reconstructs local relationships in a joint sample; UMAP transforms from
a learned neighbourhood graph. Their stories can differ without any contradiction—and
without proving robustness. That is why the decision used aligned original features,
class-stratified uncertainty, attacks with checked bounds, and a conjunctive rule.

Confidence policies exposed a second boundary. Each temperature and threshold came only
from clean calibration data. After the inputs shifted, realized coverage and selective
risk were measured without refitting. This is evidence about one frozen confidence rule,
not an OOD detector and not a safety system.

The narrow conclusion applies to Oxford-IIIT Pet, this split, these seeds, ResNet-18,
224-pixel inputs, and a finite digital `L∞` threat model. ImageNet pretraining, small
per-breed sample sizes, projection distortion, and incomplete attack coverage remain
credible competing explanations and limitations.
"""


def _sunday_draft(evaluation: dict[str, Any], representations: dict[str, Any]) -> str:
    primary = representations["primary_hypothesis"]
    return f"""# Sunday draft — the embedding plot was not the result

I trained matched standard and PGD-5 ResNet-18 models on all 37 Oxford-IIIT Pet breeds.
The tempting output was six colourful PCA, t-SNE, and UMAP panels. The actual test was
in the original 512-dimensional feature space.

Registered outcome: {_claim(primary)}

I required both a lower bootstrapped median cosine drift and higher five-neighbour breed
retention under finite PGD-20×5. I also froze clean-calibrated temperature/coverage rules
before testing noise, blur, brightness, and contrast shifts.

The durable lesson: projections answer different visualization questions. They do not,
by themselves, show that a model is robust. That claim needs paired quantitative evidence,
attack validation, uncertainty, counterexamples, and a threat model.

Scope: one dataset, one architecture, one split/seed family, and bounded digital attacks.
No physical-robustness or safety claim.

Status: draft only; not approved or published.
"""


def _release_files(config: ExperimentConfig) -> list[Path]:
    roots = [
        config.root / "src",
        config.root / "configs",
        config.root / "docs",
        config.project_path("artifacts"),
        config.project_path("results"),
        config.project_path("figures"),
        config.project_path("reports"),
        config.project_path("checkpoints"),
    ]
    files: list[Path] = [
        config.root / "README.md",
        config.root / "pyproject.toml",
        config.root / "requirements.txt",
        config.root / "setup_venv.sh",
    ]
    for root in roots:
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    manifest_path = config.project_path("artifacts") / "release" / "local-release-manifest.json"
    return sorted(
        {path.resolve() for path in files if path.is_file() and path.resolve() != manifest_path}
    )


def report(config: ExperimentConfig, *, progress: bool = True) -> dict[str, Any]:
    status("Report: loading verified evaluation and representation evidence...", enabled=progress)
    evaluation = _load(config.project_path("results") / "evaluation.json")
    representations = _load(config.project_path("results") / "representations.json")
    if evaluation.get("config_sha256") != config.sha256:
        raise ReportError("evaluation belongs to another configuration")
    if representations.get("config_sha256") != config.sha256:
        raise ReportError("representation results belong to another configuration")
    destination = config.project_path("reports")
    technical = destination / "technical-report.md"
    long_form = destination / "robust-vision-long-form-report.md"
    sunday = destination / "sunday-draft.md"
    atomic_write_text(technical, _technical_report(config, evaluation, representations))
    atomic_write_text(long_form, _long_form(evaluation, representations))
    atomic_write_text(sunday, _sunday_draft(evaluation, representations))
    primary = representations["primary_hypothesis"]
    summary = {
        "schema_version": 1,
        "created_at": utc_now(),
        "config_sha256": config.sha256,
        "primary_supported": primary["primary_supported"],
        "primary": primary,
        "secondary_policy_shift": evaluation["secondary_policy_shift"],
        "reports": [str(path.relative_to(config.root)) for path in (technical, long_form, sunday)],
        "approval": "not requested",
        "publication": "not performed",
    }
    atomic_write_json(config.project_path("results") / "summary.json", summary)
    files = _release_files(config)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "status": "local evidence bundle; no public release",
        "config_sha256": config.sha256,
        "source_sha256": source_hash(config.root),
        "provenance": experiment_provenance(config),
        "environment": environment_snapshot(),
        "git": git_state(config.root),
        "fixed_checkpoints": {
            arm: {
                "path": str(checkpoint_path(config, cast(Any, arm)).relative_to(config.root)),
                "sha256": sha256_file(checkpoint_path(config, cast(Any, arm))),
            }
            for arm in ("standard", "adversarial")
        },
        "files": {str(path.relative_to(config.root)): sha256_file(path) for path in files},
        "contains_original_photographs": False,
        "public_actions": [],
    }
    manifest_path = config.project_path("artifacts") / "release" / "local-release-manifest.json"
    atomic_write_json(manifest_path, manifest)
    status(
        "Report complete: local reports and the release manifest were written.", enabled=progress
    )
    return {**summary, "local_release_manifest": str(manifest_path)}
