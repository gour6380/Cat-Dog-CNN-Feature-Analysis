"""Training-only registration and monitoring for the exploratory PGD pilot.

The completed matched comparison is deliberately unaffected by these opt-in
helpers. Validation comes only from its original fitting partition, never from
calibration or test, and monitoring does not select a checkpoint.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from src.attacks import AttackSpec, pgd_attack
from src.config import ExperimentConfig
from src.data import (
    SampleRecord,
    load_registered_splits,
    stratified_hash_selection,
    stratified_train_calibration_split,
)
from src.io_utils import (
    atomic_save_npz,
    atomic_write_json,
    canonical_json_bytes,
    environment_snapshot,
    git_state,
    sha256_bytes,
    sha256_file,
    tree_hash,
    utc_now,
)
from src.metrics import accuracy_metrics
from src.model import NormalizedResNet18
from src.runtime import ensure_finite


class PilotError(RuntimeError):
    """An exploratory pilot's training-only protocol is inconsistent."""


PILOT_SCIENTIFIC_SOURCE_FILES = (
    "src/training.py",
    "src/pilot_protocol.py",
    "src/attacks.py",
    "src/data.py",
    "src/model.py",
    "src/config.py",
    "src/runtime.py",
    "src/io_utils.py",
    "src/evaluation.py",
    "src/calibration.py",
    "src/metrics.py",
    "src/corruptions.py",
    "src/progress.py",
    "requirements.txt",
)


def register_pilot_training_source(config: ExperimentConfig) -> str:
    """Freeze scientific code identity without including reports or entry points."""

    files = [config.root / name for name in PILOT_SCIENTIFIC_SOURCE_FILES]
    if any(not path.is_file() for path in files):
        raise PilotError("pilot scientific source or dependency lock is missing")
    fingerprint = tree_hash(config.root, files)
    file_hashes = {name: sha256_file(config.root / name) for name in PILOT_SCIENTIFIC_SOURCE_FILES}
    path = config.project_path("artifacts") / "training" / "source.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or (
            existing.get("pilot_training_source_sha256") != fingerprint
            or existing.get("files") != file_hashes
            or existing.get("config_sha256") != config.sha256
        ):
            raise PilotError("pilot scientific source changed; refusing to overwrite or resume")
    else:
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "config_sha256": config.sha256,
                "pilot_training_source_sha256": fingerprint,
                "files": file_hashes,
                "environment": environment_snapshot(),
                "git": git_state(config.root),
            },
        )
    return fingerprint


@dataclass(frozen=True)
class PilotProtocol:
    fit_records: list[SampleRecord]
    validation_records: list[SampleRecord]
    attack_records: list[SampleRecord]
    weights: tuple[float, float]
    sha256: str
    manifest: dict[str, Any]


def pilot_enabled(config: ExperimentConfig) -> bool:
    section = config.raw.get("pilot")
    return isinstance(section, dict) and section.get("enabled") is True


def _ids(records: list[SampleRecord]) -> list[str]:
    return [record.sample_id for record in records]


def _records_hash(records: list[SampleRecord]) -> str:
    return sha256_bytes(canonical_json_bytes([record.to_json() for record in records]))


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"created_at", "pilot_protocol_sha256"}
            }
        )
    )


def prepare_pilot_protocol(
    config: ExperimentConfig,
    splits: dict[str, list[SampleRecord]] | None = None,
) -> PilotProtocol:
    """Register or verify the fixed, breed-stratified training validation split."""

    if not pilot_enabled(config):
        raise PilotError("pilot protocol requires pilot.enabled: true")
    if config.label_mode != "species":
        raise PilotError("the species-balanced pilot requires cat/dog targets")
    fraction = config.number("pilot", "validation_fraction")
    clean_weight = config.number("pilot", "clean_loss_weight")
    warmup_epochs = config.integer("pilot", "clean_warmup_epochs")
    attack_per_species = config.integer("pilot", "validation_attack_per_species")
    seed = config.value("pilot", "validation_seed", str)
    if not 0 < fraction < 1 or not 0 < clean_weight < 1:
        raise PilotError("pilot split fraction and mixed clean weight must be between zero and one")
    if not 0 <= warmup_epochs < config.integer("training", "epochs"):
        raise PilotError("pilot clean warmup must leave at least one mixed-training epoch")
    if attack_per_species <= 0 or not seed:
        raise PilotError("pilot validation selection needs a positive sample count and seed")
    registered = load_registered_splits(config) if splits is None else splits
    original = sorted(registered["training"], key=lambda record: record.sample_id)
    if len(set(_ids(original))) != len(original):
        raise PilotError("original fitting partition contains duplicate sample IDs")
    if any(
        record.official_split != "trainval" or record.label != record.species for record in original
    ):
        raise PilotError("pilot fitting samples must have original trainval species targets")
    original_ids = set(_ids(original))
    excluded = {*_ids(registered["calibration"]), *_ids(registered["test"])}
    if original_ids & excluded:
        raise PilotError("original fitting partition overlaps calibration or test")
    fit, validation = stratified_train_calibration_split(original, 1 - fraction, seed)
    fit_ids, validation_ids = set(_ids(fit)), set(_ids(validation))
    if fit_ids & validation_ids or fit_ids | validation_ids != original_ids:
        raise PilotError("pilot fitting and validation IDs must partition original fitting IDs")
    if {record.breed_name for record in fit} != {record.breed_name for record in validation}:
        raise PilotError("pilot validation reserve must retain every original breed in fitting")
    counts = Counter(record.label for record in fit)
    if set(counts) != {0, 1}:
        raise PilotError("both species must remain in the effective fitting partition")
    weights = (len(fit) / (2 * counts[0]), len(fit) / (2 * counts[1]))
    attack = stratified_hash_selection(validation, attack_per_species, seed + ":pgd-monitor")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "config_sha256": config.sha256,
        "status": "registered_before_fitting",
        "scientific_status": "exploratory_after_original_test_inspection",
        "validation_source": "original_registered_training_only",
        "stratification": "breed",
        "validation_fraction": fraction,
        "validation_seed": seed,
        "clean_warmup_epochs": warmup_epochs,
        "mixed_clean_loss_weight": clean_weight,
        "species_weight_formula": "N_fit / (2 * n_fit_species)",
        "species_weights": {str(label): weight for label, weight in enumerate(weights)},
        "checkpoint_selection": "none; fixed configured final epoch",
        "fixed_final_epoch": config.integer("training", "epochs"),
        "original_training_sha256": _records_hash(original),
        "fit_sha256": _records_hash(fit),
        "validation_sha256": _records_hash(validation),
        "validation_attack_sha256": _records_hash(attack),
        "fit_ids": _ids(fit),
        "validation_ids": _ids(validation),
        "validation_attack_ids": _ids(attack),
        "counts": {
            "original_training": len(original),
            "fit": len(fit),
            "validation": len(validation),
            "validation_attack": len(attack),
        },
        "per_species_counts": {
            name: {str(label): sum(record.label == label for record in records) for label in (0, 1)}
            for name, records in (
                ("original_training", original),
                ("fit", fit),
                ("validation", validation),
                ("validation_attack", attack),
            )
        },
        "untouched_calibration_ids_sha256": sha256_bytes(
            canonical_json_bytes(sorted(_ids(registered["calibration"])))
        ),
        "untouched_test_ids_sha256": sha256_bytes(
            canonical_json_bytes(sorted(_ids(registered["test"])))
        ),
        "fit_records": [record.to_json() for record in fit],
        "validation_records": [record.to_json() for record in validation],
        "validation_attack": {
            "epsilon": config.number("attack", "epsilon"),
            "step_size": config.number("attack", "step_size"),
            "steps": config.integer("attack", "evaluation_steps"),
            "restarts": config.integer("attack", "evaluation_restarts"),
            "random_start": bool(config.section("attack").get("random_start")),
            "seed": config.integer("attack", "evaluation_seed"),
        },
    }
    protocol_hash = _manifest_hash(manifest)
    manifest["pilot_protocol_sha256"] = protocol_hash
    path = config.project_path("artifacts") / "training" / "validation-split.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or _manifest_hash(existing) != protocol_hash:
            raise PilotError("registered pilot protocol differs; refusing to overwrite or resume")
        if existing.get("pilot_protocol_sha256") != protocol_hash:
            raise PilotError("registered pilot protocol hash is invalid")
        manifest = cast(dict[str, Any], existing)
    else:
        atomic_write_json(path, manifest)
    return PilotProtocol(fit, validation, attack, weights, protocol_hash, manifest)


def pilot_phase(config: ExperimentConfig, epoch: int) -> str:
    return (
        "clean_warmup"
        if epoch <= config.integer("pilot", "clean_warmup_epochs")
        else "mixed_clean_pgd"
    )


def pilot_training_loss(
    model: nn.Module,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    epoch: int,
    attack_spec: AttackSpec,
    protocol: PilotProtocol,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute weighted warmup or mixed loss, keeping training PGD unchanged."""

    weights = torch.tensor(protocol.weights, device=pixels.device, dtype=torch.float32)
    adversarial: torch.Tensor | None = None
    if pilot_phase(config, epoch) == "mixed_clean_pgd":
        attack_ids = [f"{sample_id}:epoch:{epoch}" for sample_id in sample_ids]
        # Do not retain a clean forward graph while PGD constructs input gradients.
        adversarial = pgd_attack(model, pixels, labels, attack_ids, attack_spec).adversarial
    clean_logits = model(pixels)
    ensure_finite("pilot clean training logits", clean_logits)
    clean_loss = nnf.cross_entropy(clean_logits, labels, weight=weights)
    if adversarial is None:
        return clean_loss, clean_loss, None
    adversarial_logits = model(adversarial)
    ensure_finite("pilot adversarial training logits", adversarial_logits)
    adversarial_loss = nnf.cross_entropy(adversarial_logits, labels, weight=weights)
    clean_weight = config.number("pilot", "clean_loss_weight")
    return (
        clean_weight * clean_loss + (1 - clean_weight) * adversarial_loss,
        clean_loss,
        adversarial_loss,
    )


def _validation_metrics(arrays: dict[str, np.ndarray[Any, Any]]) -> dict[str, Any]:
    predictions = arrays["logits"].argmax(axis=1)
    labels = arrays["labels"].astype(np.int64)
    return {
        **accuracy_metrics(arrays["logits"].astype(np.float64), labels, 2),
        "count": len(labels),
        "correct_count": int((predictions == labels).sum()),
        "prediction_counts": {str(label): int((predictions == label).sum()) for label in (0, 1)},
    }


def collect_pilot_validation(
    model: NormalizedResNet18,
    protocol: PilotProtocol,
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    """Record clean and strong-attack validation without test-based selection."""

    # Lazy imports avoid the existing evaluation -> training provenance cycle.
    from src.evaluation import _attack_metrics, _collect_attack, _collect_clean_or_shift

    was_training = model.training
    try:
        model.eval()
        clean = _collect_clean_or_shift(
            model,
            protocol.validation_records,
            config,
            device,
            progress=progress,
            description="pilot validation clean",
        )
        by_id = {str(sample_id): index for index, sample_id in enumerate(clean["sample_ids"])}
        attacked = _collect_attack(
            model,
            protocol.attack_records,
            config,
            device,
            "pgd",
            progress=progress,
            description="pilot validation PGD",
        )
        # Evaluation loaders sort globally by filename, whereas the hash selector
        # returns label-then-filename order. Pair by actual collected sample IDs.
        attacked_ids = [str(sample_id) for sample_id in attacked["sample_ids"]]
        if len(set(attacked_ids)) != len(protocol.attack_records) or set(attacked_ids) != set(
            _ids(protocol.attack_records)
        ):
            raise PilotError("validation PGD collection changed fixed sample IDs")
        indices = np.asarray([by_id[sample_id] for sample_id in attacked_ids])
        paired_clean = {key: values[indices] for key, values in clean.items()}
        paths: dict[str, dict[str, str]] = {}
        for name, arrays in (
            ("clean", clean),
            ("clean_attack_subset", paired_clean),
            ("pgd", attacked),
        ):
            path = (
                config.project_path("results") / "validation" / f"epoch-{epoch:03d}" / f"{name}.npz"
            )
            atomic_save_npz(path, arrays)
            paths[name] = {"path": str(path.relative_to(config.root)), "sha256": sha256_file(path)}
        result = {
            "epoch": epoch,
            "protocol_sha256": protocol.sha256,
            "checkpoint_selection": "none",
            "clean": _validation_metrics(clean),
            "clean_attack_subset": _validation_metrics(paired_clean),
            "pgd": _validation_metrics(attacked),
            "attack": {
                **protocol.manifest["validation_attack"],
                **_attack_metrics(paired_clean, attacked, 2),
            },
            "per_sample": paths,
        }
        atomic_write_json(
            config.project_path("artifacts")
            / "training"
            / "validation"
            / f"epoch-{epoch:03d}.json",
            result,
        )
        return result
    finally:
        model.train(was_training)
