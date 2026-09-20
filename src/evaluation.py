"""Aligned clean, attack, corruption, calibration, and feature evaluation."""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch

from src.attacks import AttackSpec, fgsm_attack, pgd_attack
from src.calibration import CalibrationPolicy, apply_policy, fit_policy, negative_log_likelihood
from src.config import ExperimentConfig
from src.corruptions import Corruption, apply_corruption, registered_corruptions
from src.data import SampleRecord, build_loader, load_registered_splits
from src.io_utils import atomic_save_npz, atomic_write_json, sha256_file, tree_hash, utc_now
from src.metrics import accuracy_metrics
from src.model import NormalizedResNet18, load_checkpoint_model
from src.progress import status, tqdm
from src.runtime import ensure_finite, memory_snapshot, record_failure, synchronize
from src.training import Arm, checkpoint_path, experiment_provenance

BatchTransform = Callable[[torch.Tensor, list[str]], torch.Tensor]
EvaluationSection = Literal[
    "clean", "fgsm", "pgd", "gaussian_noise", "gaussian_blur", "brightness", "contrast"
]
EVALUATION_SECTIONS: tuple[EvaluationSection, ...] = (
    "clean",
    "fgsm",
    "pgd",
    "gaussian_noise",
    "gaussian_blur",
    "brightness",
    "contrast",
)
EVALUATION_METHOD_FILES = (
    "src/evaluation.py",
    "src/calibration.py",
    "src/corruptions.py",
    "src/metrics.py",
    "src/model.py",
    "src/attacks.py",
    "src/data.py",
    "src/config.py",
    "src/runtime.py",
    "src/io_utils.py",
    "src/training.py",
    "requirements.txt",
)


class EvaluationError(RuntimeError):
    """Evaluation evidence is absent, stale, or misaligned."""


def _arrays_path(config: ExperimentConfig, group: str, name: str) -> Path:
    return config.project_path("results") / "per_sample" / group / f"{name}.npz"


def _metadata_path(array_path: Path) -> Path:
    return array_path.with_suffix(".json")


def _cache_valid(path: Path, expected: dict[str, str]) -> bool:
    metadata = _metadata_path(path)
    if not path.is_file() or not metadata.is_file():
        return False
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EvaluationError(f"invalid cache metadata: {metadata}") from error
    if not isinstance(value, dict) or value.get("provenance") != expected:
        return False
    if value.get("npz_sha256") != sha256_file(path):
        raise EvaluationError(f"per-sample cache hash mismatch: {path}")
    return True


def _save_arrays(
    path: Path,
    arrays: dict[str, np.ndarray[Any, Any]],
    provenance: dict[str, str],
    extra: dict[str, Any],
) -> None:
    atomic_save_npz(path, arrays)
    atomic_write_json(
        _metadata_path(path),
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "provenance": provenance,
            "npz_sha256": sha256_file(path),
            **extra,
        },
    )


def load_arrays(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    if not path.is_file():
        raise EvaluationError(f"missing per-sample array: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {name: loaded[name] for name in loaded.files}
    except (OSError, ValueError) as error:
        raise EvaluationError(f"unreadable per-sample array: {path}") from error


def _validate_arrays(
    arrays: dict[str, np.ndarray[Any, Any]],
    records: list[SampleRecord],
    classes: int,
) -> None:
    """Validate against filename-ordered registered records, not another cache."""

    ordered = sorted(records, key=lambda record: record.sample_id)
    expected = {
        "sample_ids": np.asarray([record.sample_id for record in ordered], dtype=np.str_),
        "labels": np.asarray([record.label for record in ordered], dtype=np.int64),
        "species": np.asarray([record.species for record in ordered], dtype=np.int64),
    }
    for name, values in expected.items():
        if name not in arrays or not np.array_equal(arrays[name], values):
            raise EvaluationError(f"registered {name} are not aligned")
    count = len(ordered)
    logits = arrays.get("logits")
    features = arrays.get("features")
    if logits is None or logits.shape != (count, classes) or not np.isfinite(logits).all():
        raise EvaluationError("evaluation logits have invalid shape or non-finite values")
    if (
        features is None
        or features.ndim != 2
        or features.shape[0] != count
        or not np.isfinite(features).all()
    ):
        raise EvaluationError("evaluation features have invalid shape or non-finite values")
    cumulative = arrays.get("cumulative_predictions")
    if cumulative is not None and (
        cumulative.ndim != 2
        or cumulative.shape[1] != count
        or not np.isin(cumulative, np.arange(classes)).all()
    ):
        raise EvaluationError("cumulative attack predictions are invalid or misaligned")
    gallery_ids = arrays.get("gallery_sample_ids")
    if gallery_ids is not None:
        required = (
            ("gallery_labels", "gallery_species", "gallery_pixels")
            if "gallery_pixels" in arrays
            else (
                "gallery_labels",
                "gallery_species",
                "gallery_clean_pixels",
                "gallery_attack_pixels",
            )
        )
        if any(name not in arrays for name in required):
            raise EvaluationError("attack gallery is incomplete")
        if len(gallery_ids) != len(set(gallery_ids.tolist())):
            raise EvaluationError("attack gallery IDs are duplicated")
        index = {record.sample_id: record for record in ordered}
        if any(sample_id not in index for sample_id in gallery_ids.tolist()):
            raise EvaluationError("attack gallery IDs are not registered")
        for field in ("labels", "species"):
            attribute = "label" if field == "labels" else "species"
            values = np.asarray(
                [getattr(index[str(value)], attribute) for value in gallery_ids],
                dtype=np.int64,
            )
            if not np.array_equal(arrays[f"gallery_{field}"], values):
                raise EvaluationError(f"attack gallery {field} are not aligned")
        clean = arrays.get("gallery_pixels", arrays.get("gallery_clean_pixels"))
        attacked = arrays.get("gallery_attack_pixels", clean)
        assert clean is not None and attacked is not None
        if (
            clean.ndim != 4
            or clean.shape[0] != len(gallery_ids)
            or clean.shape[1] != 3
            or clean.shape != attacked.shape
            or not np.isfinite(clean).all()
            or not np.isfinite(attacked).all()
            or clean.min() < 0.0
            or clean.max() > 1.0
            or attacked.min() < 0.0
            or attacked.max() > 1.0
        ):
            raise EvaluationError("attack gallery pixels are invalid")


def _collect_clean_or_shift(
    model: NormalizedResNet18,
    records: list[SampleRecord],
    config: ExperimentConfig,
    device: torch.device,
    transform: BatchTransform | None = None,
    *,
    progress: bool = False,
    description: str = "evaluation",
) -> dict[str, np.ndarray[Any, Any]]:
    logits_parts: list[np.ndarray[Any, Any]] = []
    feature_parts: list[np.ndarray[Any, Any]] = []
    label_parts: list[np.ndarray[Any, Any]] = []
    species_parts: list[np.ndarray[Any, Any]] = []
    ids: list[str] = []
    gallery_records = {
        record.sample_id
        for species_id in (0, 1)
        for record in sorted(
            (item for item in records if item.species == species_id),
            key=lambda item: item.sample_id,
        )[:2]
    }
    gallery_ids: list[str] = []
    gallery_labels: list[int] = []
    gallery_species: list[int] = []
    gallery_pixels: list[np.ndarray[Any, Any]] = []
    loader = build_loader(records, config, training=False)
    model.eval()
    for pixels, labels, sample_ids_raw, species in tqdm(
        loader,
        total=len(loader),
        desc=description,
        unit="batch",
        leave=False,
        disable=not progress,
    ):
        sample_ids = list(sample_ids_raw)
        pixels = pixels.to(device=device, dtype=torch.float32)
        if transform is not None:
            pixels = transform(pixels, sample_ids)
        with torch.inference_mode():
            logits, features = model.forward_with_features(pixels)
        ensure_finite("evaluation logits", logits)
        ensure_finite("evaluation features", features)
        logits_parts.append(logits.detach().cpu().numpy().astype(np.float32))
        feature_parts.append(features.detach().cpu().numpy().astype(np.float32))
        label_parts.append(labels.numpy().astype(np.int64))
        species_parts.append(species.numpy().astype(np.int64))
        ids.extend(sample_ids)
        for index, sample_id in enumerate(sample_ids):
            if sample_id in gallery_records:
                gallery_ids.append(sample_id)
                gallery_labels.append(int(labels[index]))
                gallery_species.append(int(species[index]))
                gallery_pixels.append(pixels[index].detach().cpu().numpy().astype(np.float32))
    arrays = {
        "sample_ids": np.asarray(ids, dtype=np.str_),
        "labels": np.concatenate(label_parts),
        "species": np.concatenate(species_parts),
        "logits": np.concatenate(logits_parts),
        "features": np.concatenate(feature_parts),
    }
    if gallery_ids:
        arrays.update(
            gallery_sample_ids=np.asarray(gallery_ids, dtype=np.str_),
            gallery_labels=np.asarray(gallery_labels, dtype=np.int64),
            gallery_species=np.asarray(gallery_species, dtype=np.int64),
            gallery_pixels=np.stack(gallery_pixels),
        )
    return arrays


def _collect_attack(
    model: NormalizedResNet18,
    records: list[SampleRecord],
    config: ExperimentConfig,
    device: torch.device,
    kind: str,
    *,
    progress: bool = False,
    description: str = "attack evaluation",
) -> dict[str, np.ndarray[Any, Any]]:
    logits_parts: list[np.ndarray[Any, Any]] = []
    feature_parts: list[np.ndarray[Any, Any]] = []
    label_parts: list[np.ndarray[Any, Any]] = []
    species_parts: list[np.ndarray[Any, Any]] = []
    cumulative_parts: list[list[np.ndarray[Any, Any]]] | None = None
    gallery_records = {
        record.sample_id
        for species_id in (0, 1)
        for record in sorted(
            (item for item in records if item.species == species_id),
            key=lambda item: item.sample_id,
        )[:2]
    }
    gallery_ids: list[str] = []
    gallery_labels: list[int] = []
    gallery_species: list[int] = []
    gallery_clean: list[np.ndarray[Any, Any]] = []
    gallery_attack: list[np.ndarray[Any, Any]] = []
    ids: list[str] = []
    loader = build_loader(records, config, training=False)
    model.eval()
    for pixels, labels, sample_ids_raw, species in tqdm(
        loader,
        total=len(loader),
        desc=description,
        unit="batch",
        leave=False,
        disable=not progress,
    ):
        sample_ids = list(sample_ids_raw)
        pixels = pixels.to(device=device, dtype=torch.float32)
        labels_device = labels.to(device=device, dtype=torch.long)
        if kind == "fgsm":
            attack = fgsm_attack(
                model,
                pixels,
                labels_device,
                sample_ids,
                config.number("attack", "epsilon"),
                config.integer("attack", "evaluation_seed"),
            )
        elif kind == "pgd":
            attack = pgd_attack(
                model,
                pixels,
                labels_device,
                sample_ids,
                AttackSpec(
                    epsilon=config.number("attack", "epsilon"),
                    step_size=config.number("attack", "step_size"),
                    steps=config.integer("attack", "evaluation_steps"),
                    random_start=bool(config.section("attack").get("random_start")),
                    restarts=config.integer("attack", "evaluation_restarts"),
                    seed=config.integer("attack", "evaluation_seed"),
                ),
            )
        else:
            raise EvaluationError(f"unknown attack condition: {kind}")
        with torch.inference_mode():
            logits, features = model.forward_with_features(attack.adversarial)
        ensure_finite("attacked logits", logits)
        ensure_finite("attacked features", features)
        logits_parts.append(logits.detach().cpu().numpy().astype(np.float32))
        feature_parts.append(features.detach().cpu().numpy().astype(np.float32))
        label_parts.append(labels.numpy().astype(np.int64))
        species_parts.append(species.numpy().astype(np.int64))
        ids.extend(sample_ids)
        for index, sample_id in enumerate(sample_ids):
            if sample_id in gallery_records:
                gallery_ids.append(sample_id)
                gallery_labels.append(int(labels[index]))
                gallery_species.append(int(species[index]))
                gallery_clean.append(pixels[index].detach().cpu().numpy().astype(np.float32))
                gallery_attack.append(
                    attack.adversarial[index].detach().cpu().numpy().astype(np.float32)
                )
        if cumulative_parts is None:
            cumulative_parts = [[] for _ in attack.cumulative_predictions]
        if len(cumulative_parts) != len(attack.cumulative_predictions):
            raise EvaluationError("attack restart count changed across batches")
        for restart, predictions in enumerate(attack.cumulative_predictions):
            cumulative_parts[restart].append(predictions.numpy().astype(np.int64))
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "sample_ids": np.asarray(ids, dtype=np.str_),
        "labels": np.concatenate(label_parts),
        "species": np.concatenate(species_parts),
        "logits": np.concatenate(logits_parts),
        "features": np.concatenate(feature_parts),
    }
    if cumulative_parts:
        arrays["cumulative_predictions"] = np.stack(
            [np.concatenate(parts) for parts in cumulative_parts]
        )
    if gallery_ids:
        arrays.update(
            gallery_sample_ids=np.asarray(gallery_ids, dtype=np.str_),
            gallery_labels=np.asarray(gallery_labels, dtype=np.int64),
            gallery_species=np.asarray(gallery_species, dtype=np.int64),
            gallery_clean_pixels=np.stack(gallery_clean),
            gallery_attack_pixels=np.stack(gallery_attack),
        )
    return arrays


def _evaluation_method_sha256(config: ExperimentConfig) -> str:
    """Fingerprint evaluation methods without changing training/checkpoint identity."""

    paths = [config.root / relative for relative in EVALUATION_METHOD_FILES]
    missing = [
        relative
        for relative, path in zip(EVALUATION_METHOD_FILES, paths, strict=True)
        if not path.is_file()
    ]
    if missing:
        raise EvaluationError("missing evaluation scientific source files: " + ", ".join(missing))
    return tree_hash(config.root, paths)


def _artifact_provenance(
    config: ExperimentConfig, arm: Arm, checkpoint_sha256: str
) -> dict[str, str]:
    return {
        **experiment_provenance(config),
        "arm": arm,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_method_sha256": _evaluation_method_sha256(config),
    }


def _get_or_compute(
    path: Path,
    provenance: dict[str, str],
    compute: Callable[[], dict[str, np.ndarray[Any, Any]]],
    extra: dict[str, Any],
    *,
    records: list[SampleRecord] | None = None,
    classes: int | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    if _cache_valid(path, provenance):
        arrays = load_arrays(path)
        if records is not None and classes is not None:
            _validate_arrays(arrays, records, classes)
        return arrays
    arrays = compute()
    if records is not None and classes is not None:
        _validate_arrays(arrays, records, classes)
    _save_arrays(path, arrays, provenance, extra)
    return arrays


def _assert_alignment(
    reference: dict[str, np.ndarray[Any, Any]], other: dict[str, np.ndarray[Any, Any]]
) -> None:
    if not np.array_equal(reference["sample_ids"], other["sample_ids"]):
        raise EvaluationError("sample IDs are not aligned")
    if not np.array_equal(reference["labels"], other["labels"]):
        raise EvaluationError("labels are not aligned")
    if not np.array_equal(reference["species"], other["species"]):
        raise EvaluationError("species are not aligned")


def _attack_metrics(
    clean: dict[str, np.ndarray[Any, Any]],
    attacked: dict[str, np.ndarray[Any, Any]],
    classes: int,
) -> dict[str, Any]:
    _assert_alignment(clean, attacked)
    labels = clean["labels"].astype(np.int64)
    clean_predictions = clean["logits"].argmax(axis=1)
    attacked_predictions = attacked["logits"].argmax(axis=1)
    clean_correct = clean_predictions == labels
    attacked_correct = attacked_predictions == labels
    successes = clean_correct & ~attacked_correct
    per_class: dict[str, Any] = {}
    for label in range(classes):
        selected = labels == label
        selected_clean = clean_correct[selected]
        denominator = int(selected_clean.sum())
        per_class[str(label)] = {
            "robust_accuracy": float(attacked_correct[selected].mean()),
            "attack_success_clean_correct": (
                float(successes[selected].sum() / denominator) if denominator else float("nan")
            ),
            "count": int(selected.sum()),
        }
    cumulative = attacked.get("cumulative_predictions")
    cumulative_robust: list[float] = []
    if cumulative is not None:
        cumulative_robust = [float((row == labels).mean()) for row in cumulative]
    return {
        "clean_accuracy_on_subset": float(clean_correct.mean()),
        "robust_accuracy": float(attacked_correct.mean()),
        "attack_success_clean_correct": (
            float(successes.sum() / clean_correct.sum()) if clean_correct.any() else float("nan")
        ),
        "clean_correct_count": int(clean_correct.sum()),
        "attack_success_count": int(successes.sum()),
        "cumulative_restart_robust_accuracy": cumulative_robust,
        "per_class": per_class,
    }


def _policy_from_calibration(
    config: ExperimentConfig, arrays: dict[str, np.ndarray[Any, Any]]
) -> CalibrationPolicy:
    bounds_raw = config.section("calibration").get("temperature_bounds")
    if not isinstance(bounds_raw, list) or len(bounds_raw) != 2:
        raise EvaluationError("calibration.temperature_bounds must have two values")
    return fit_policy(
        arrays["logits"].astype(np.float64),
        arrays["labels"].astype(np.int64),
        config.number("calibration", "target_coverage"),
        (float(bounds_raw[0]), float(bounds_raw[1])),
    )


def _policy_path(config: ExperimentConfig, arm: Arm) -> Path:
    return _arrays_path(config, "calibration", arm).with_suffix(".policy.json")


def _cached_policy(
    config: ExperimentConfig,
    arm: Arm,
    calibration: dict[str, np.ndarray[Any, Any]],
    provenance: dict[str, str],
) -> CalibrationPolicy:
    """Fit once per exact calibration-array/provenance identity."""

    path = _policy_path(config, arm)
    arrays_hash = sha256_file(_arrays_path(config, "calibration", arm))
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise EvaluationError(f"invalid calibration policy cache: {path}") from error
        if (
            isinstance(value, dict)
            and value.get("provenance") == provenance
            and value.get("calibration_npz_sha256") == arrays_hash
        ):
            try:
                policy = CalibrationPolicy(**value["policy"])
            except (KeyError, TypeError, ValueError) as error:
                raise EvaluationError(f"invalid cached calibration policy: {path}") from error
            if (
                not np.isfinite(policy.temperature)
                or policy.temperature <= 0.0
                or not np.isfinite(policy.threshold)
                or not 0.0 <= policy.threshold <= 1.0
                or policy.fitted_count != len(calibration["labels"])
                or policy.target_coverage != config.number("calibration", "target_coverage")
            ):
                raise EvaluationError(f"cached calibration policy is inconsistent: {path}")
            return policy
    policy = _policy_from_calibration(config, calibration)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "provenance": provenance,
            "calibration_npz_sha256": arrays_hash,
            "fit_partition": "clean calibration only",
            "policy": asdict(policy),
        },
    )
    return policy


def _register_array(arm_result: dict[str, Any], key: str, path: Path) -> None:
    arm_result["per_sample"][key] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "metadata_path": str(_metadata_path(path)),
        "metadata_sha256": sha256_file(_metadata_path(path)),
    }


def _condition_metrics(
    arrays: dict[str, np.ndarray[Any, Any]],
    policy: CalibrationPolicy,
    config: ExperimentConfig,
) -> dict[str, Any]:
    logits = arrays["logits"].astype(np.float64)
    labels = arrays["labels"].astype(np.int64)
    metrics = accuracy_metrics(logits, labels, config.integer("dataset", "classes"))
    metrics["risk"] = apply_policy(
        logits, labels, policy, config.integer("calibration", "ece_bins")
    )
    return metrics


def _update_summaries(
    config: ExperimentConfig,
    result: dict[str, Any],
    arms: tuple[Arm, ...],
    conditions: list[Corruption],
) -> bool:
    required_full = ["clean", *[condition.identifier for condition in conditions]]
    missing: dict[str, Any] = {}
    policy_shift: dict[str, Any] = {}
    target = config.number("calibration", "target_coverage")
    for arm in arms:
        arm_result = result["arms"][arm]
        full = arm_result["full_test"]
        attacks = arm_result["attack_subset"]
        missing[arm] = {
            "full_test": [name for name in required_full if name not in full],
            "attack_subset": [name for name in ("fgsm", "pgd") if name not in attacks],
        }
        if all(name in attacks for name in ("fgsm", "pgd")):
            fgsm = attacks["fgsm"]["robust_accuracy"]
            pgd = attacks["pgd"]["robust_accuracy"]
            arm_result["attack_strength_check"] = {
                "pgd_not_weaker_by_robust_accuracy": bool(pgd <= fgsm),
                "fgsm_robust_accuracy": fgsm,
                "pgd_robust_accuracy": pgd,
            }
        if "clean" in full:
            clean_risk = full["clean"]["risk"]
            flagged: list[dict[str, Any]] = []
            for condition, metrics in full.items():
                risk = metrics["risk"]
                coverage_delta = float(risk["coverage"] - target)
                risk_delta = float(risk["selective_risk"] - clean_risk["selective_risk"])
                if abs(coverage_delta) > 0.05 or abs(risk_delta) > 0.02:
                    flagged.append(
                        {
                            "condition": condition,
                            "coverage_delta_from_target": coverage_delta,
                            "selective_risk_delta_from_clean": risk_delta,
                        }
                    )
            policy_shift[arm] = {
                "hypothesis_met": bool(flagged),
                "flagged_conditions": flagged,
                "evaluated_full_test_conditions": list(full),
            }
    complete = not any(value["full_test"] or value["attack_subset"] for value in missing.values())
    result["secondary_policy_shift"] = policy_shift
    result["status"] = "complete" if complete else "partial"
    result["completion"] = {
        "required_arms": list(arms),
        "required_full_test": required_full,
        "required_attacks": ["fgsm", "pgd"],
        "missing_conditions": missing,
    }
    result["updated_at"] = utc_now()
    return complete


def assert_evaluation_complete(
    config: ExperimentConfig,
    result: dict[str, Any],
    *,
    arms: tuple[Arm, ...] | None = None,
) -> None:
    """Reject incomplete/stale evidence before reporting or feature analysis.

    Older complete schema-v1 records remain readable. New records additionally
    register the exact NPZ/sidecar receipts used to construct their metrics.
    """

    if result.get("config_sha256") != config.sha256:
        raise EvaluationError("evaluation belongs to a different configuration")
    if result.get("status", "complete") != "complete":
        raise EvaluationError("evaluation is incomplete; run the missing sections first")
    recorded = result.get("arms")
    if not isinstance(recorded, dict) or not recorded:
        raise EvaluationError("evaluation has no model-arm evidence")
    selected = arms or cast(tuple[Arm, ...], tuple(recorded))
    required_full = {
        "clean",
        *[item.identifier for item in registered_corruptions(config.section("corruptions"))],
    }
    for arm in selected:
        arm_result = recorded.get(arm)
        if not isinstance(arm_result, dict):
            raise EvaluationError(f"evaluation is missing the {arm} arm")
        if not required_full.issubset(arm_result.get("full_test", {})) or not {
            "fgsm",
            "pgd",
        }.issubset(arm_result.get("attack_subset", {})):
            raise EvaluationError(f"evaluation is missing required conditions for {arm}")
        checkpoint = checkpoint_path(config, arm)
        if not checkpoint.is_file() or arm_result.get("checkpoint_sha256") != sha256_file(
            checkpoint
        ):
            raise EvaluationError(f"evaluation checkpoint is stale or absent for {arm}")
        receipts = arm_result.get("per_sample", {})
        if "per_sample" in arm_result:
            expected_provenance = _artifact_provenance(config, arm, sha256_file(checkpoint))
            if arm_result.get("provenance") != expected_provenance:
                raise EvaluationError(f"evaluation method/provenance is stale or absent for {arm}")
            required_receipts = {
                "calibration",
                "attack_subset/clean",
                "attack_subset/fgsm",
                "attack_subset/pgd",
                *[f"full_test/{name}" for name in required_full],
            }
            if not isinstance(receipts, dict) or not required_receipts.issubset(receipts):
                raise EvaluationError(f"evaluation array receipts are incomplete for {arm}")
        for receipt in receipts.values():
            if not isinstance(receipt, dict):
                raise EvaluationError("evaluation array receipt is invalid")
            for path_field, hash_field in (
                ("path", "sha256"),
                ("metadata_path", "metadata_sha256"),
            ):
                path = Path(receipt.get(path_field, ""))
                if not path.is_file() or sha256_file(path) != receipt.get(hash_field):
                    raise EvaluationError(f"evaluation array receipt is stale or absent: {path}")
            if "per_sample" in arm_result:
                metadata = json.loads(Path(receipt["metadata_path"]).read_text(encoding="utf-8"))
                if metadata.get("provenance") != expected_provenance:
                    raise EvaluationError("evaluation array method/provenance is stale or absent")
        if "policy_cache" in arm_result:
            receipt = arm_result["policy_cache"]
            path = Path(receipt.get("path", ""))
            if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
                raise EvaluationError(f"evaluation policy receipt is stale or absent: {path}")


def evaluate(
    config: ExperimentConfig,
    device: torch.device,
    *,
    progress: bool = True,
    arms: tuple[Arm, ...] = ("standard", "adversarial"),
    section: EvaluationSection | None = None,
) -> dict[str, Any]:
    """Evaluate one condition family, or the entire registered experiment.

    All sections require clean calibration and full-test clean inference. Only
    attack sections additionally require paired clean attack-subset inference.
    Existing valid caches are hydrated without model construction or forwards.
    Every completed condition updates an atomic partial manifest; evaluation.json
    is published only once the full registered evidence for the selected arms is
    available. A repeated section cannot silently execute unrelated conditions.
    """

    failure = config.project_path("artifacts") / "failures" / "evaluate.json"
    started = time.perf_counter()
    try:
        if (
            not arms
            or len(set(arms)) != len(arms)
            or any(arm not in {"standard", "adversarial"} for arm in arms)
        ):
            raise EvaluationError("evaluation arms must be nonempty, unique and supported")
        if section is not None and section not in EVALUATION_SECTIONS:
            raise EvaluationError(f"unknown evaluation section: {section}")
        epochs = config.integer("training", "epochs")
        attack_steps = config.integer("attack", "evaluation_steps")
        attack_restarts = config.integer("attack", "evaluation_restarts")
        attack_per_class = config.integer("dataset", "attack_per_class")
        classes = config.integer("dataset", "classes")
        attack_label = f"PGD-{attack_steps}×{attack_restarts}"
        attack_partition = f"fixed {attack_per_class}-per-target-class test subset"
        status(
            f"Evaluation: epoch-{epochs} on {device.type}; section={section or 'all'}.",
            enabled=progress,
        )
        splits = load_registered_splits(config)
        conditions = registered_corruptions(config.section("corruptions"))
        selected_conditions = [
            item for item in conditions if section is None or item.name == section
        ]
        selected_attacks = [name for name in ("fgsm", "pgd") if section is None or name == section]
        result: dict[str, Any] = {
            "schema_version": 1,
            "created_at": utc_now(),
            "config_sha256": config.sha256,
            "device": str(device),
            "requested_section": section or "all",
            "arms": {},
            "finite_attack_scope": {
                "attacks": ["FGSM", attack_label],
                "norm": config.value("attack", "norm", str),
                "epsilon": config.number("attack", "epsilon"),
                "step_size": config.number("attack", "step_size"),
                "pgd_steps": attack_steps,
                "pgd_restarts": attack_restarts,
                "samples_per_class": attack_per_class,
            },
        }
        provenances: dict[Arm, dict[str, str]] = {}
        arrays_by_arm: dict[Arm, dict[str, dict[str, np.ndarray[Any, Any]]]] = {}
        policies: dict[Arm, CalibrationPolicy] = {}

        def publish_progress() -> None:
            complete = _update_summaries(config, result, arms, conditions)
            result["elapsed_seconds"] = time.perf_counter() - started
            atomic_write_json(config.project_path("results") / "evaluation.partial.json", result)
            if complete:
                atomic_write_json(config.project_path("results") / "evaluation.json", result)

        # Rebuild evidence from verified NPZs, never from unverified partial metrics.
        # Hydrating every selected arm first preserves already completed sections
        # even if a later collection fails in the first arm.
        for arm in arms:
            checkpoint = checkpoint_path(config, arm)
            if not checkpoint.is_file():
                raise EvaluationError(
                    f"configured epoch-{epochs} checkpoint is missing: {checkpoint}"
                )
            checkpoint_hash = sha256_file(checkpoint)
            provenance = _artifact_provenance(config, arm, checkpoint_hash)
            provenances[arm] = provenance
            arrays_by_arm[arm] = {}
            arm_result: dict[str, Any] = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "provenance": provenance,
                "policy_fit_partition": "clean calibration only",
                "full_test": {},
                "attack_subset": {},
                "per_sample": {},
            }
            result["arms"][arm] = arm_result

            def cached(
                path: Path,
                records: list[SampleRecord],
                expected: dict[str, str] = provenance,
            ) -> dict[str, np.ndarray[Any, Any]] | None:
                if not _cache_valid(path, expected):
                    return None
                arrays = load_arrays(path)
                _validate_arrays(arrays, records, classes)
                return arrays

            calibration_path = _arrays_path(config, "calibration", arm)
            calibration = cached(calibration_path, splits["calibration"])
            if calibration is None:
                continue
            arrays_by_arm[arm]["calibration"] = calibration
            policy = _cached_policy(config, arm, calibration, provenance)
            policies[arm] = policy
            arm_result["policy"] = asdict(policy)
            arm_result["policy_cache"] = {
                "path": str(_policy_path(config, arm)),
                "sha256": sha256_file(_policy_path(config, arm)),
            }
            arm_result["calibration_nll_before"] = negative_log_likelihood(
                calibration["logits"].astype(np.float64),
                calibration["labels"].astype(np.int64),
            )
            arm_result["calibration_nll_after"] = negative_log_likelihood(
                calibration["logits"].astype(np.float64),
                calibration["labels"].astype(np.int64),
                policy.temperature,
            )
            _register_array(arm_result, "calibration", calibration_path)
            clean_path = _arrays_path(config, f"full-test/{arm}", "clean")
            clean = cached(clean_path, splits["test"])
            if clean is not None:
                arrays_by_arm[arm]["clean"] = clean
                arm_result["full_test"]["clean"] = _condition_metrics(clean, policy, config)
                _register_array(arm_result, "full_test/clean", clean_path)
                for corruption in conditions:
                    path = _arrays_path(config, f"full-test/{arm}", corruption.identifier)
                    shifted = cached(path, splits["test"])
                    if shifted is not None:
                        _assert_alignment(clean, shifted)
                        arm_result["full_test"][corruption.identifier] = _condition_metrics(
                            shifted, policy, config
                        )
                        _register_array(arm_result, f"full_test/{corruption.identifier}", path)
            attack_clean_path = _arrays_path(config, f"attack/{arm}", "clean")
            attack_clean = cached(attack_clean_path, splits["attack"])
            if attack_clean is not None:
                arrays_by_arm[arm]["attack_clean"] = attack_clean
                _register_array(arm_result, "attack_subset/clean", attack_clean_path)
                for attack_name in ("fgsm", "pgd"):
                    path = _arrays_path(config, f"attack/{arm}", attack_name)
                    attacked = cached(path, splits["attack"])
                    if attacked is not None:
                        arm_result["attack_subset"][attack_name] = _attack_metrics(
                            attack_clean, attacked, classes
                        )
                        _register_array(arm_result, f"attack_subset/{attack_name}", path)
        publish_progress()

        for arm in tqdm(arms, desc="evaluation arms", unit="arm", disable=not progress):
            arm_result = result["arms"][arm]
            provenance = provenances[arm]
            stored = arrays_by_arm[arm]
            model: NormalizedResNet18 | None = None

            def get_model(selected_arm: Arm = arm) -> NormalizedResNet18:
                nonlocal model
                if model is None:
                    model = load_checkpoint_model(
                        config, checkpoint_path(config, selected_arm), device
                    )
                return model

            try:
                calibration_path = _arrays_path(config, "calibration", arm)
                if "calibration" not in stored:
                    calibration = _get_or_compute(
                        calibration_path,
                        provenance,
                        lambda selected_arm=arm: _collect_clean_or_shift(  # type: ignore[misc]
                            get_model(),
                            splits["calibration"],
                            config,
                            device,
                            progress=progress,
                            description=f"{selected_arm} calibration",
                        ),
                        {"partition": "calibration", "condition": "clean"},
                        records=splits["calibration"],
                        classes=classes,
                    )
                    stored["calibration"] = calibration
                    policy = _cached_policy(config, arm, calibration, provenance)
                    policies[arm] = policy
                    arm_result["policy"] = asdict(policy)
                    arm_result["policy_cache"] = {
                        "path": str(_policy_path(config, arm)),
                        "sha256": sha256_file(_policy_path(config, arm)),
                    }
                    arm_result["calibration_nll_before"] = negative_log_likelihood(
                        calibration["logits"].astype(np.float64),
                        calibration["labels"].astype(np.int64),
                    )
                    arm_result["calibration_nll_after"] = negative_log_likelihood(
                        calibration["logits"].astype(np.float64),
                        calibration["labels"].astype(np.int64),
                        policy.temperature,
                    )
                    _register_array(arm_result, "calibration", calibration_path)
                    publish_progress()
                policy = policies[arm]
                clean_path = _arrays_path(config, f"full-test/{arm}", "clean")
                if "clean" not in stored:
                    clean = _get_or_compute(
                        clean_path,
                        provenance,
                        lambda selected_arm=arm: _collect_clean_or_shift(  # type: ignore[misc]
                            get_model(),
                            splits["test"],
                            config,
                            device,
                            progress=progress,
                            description=f"{selected_arm} clean test",
                        ),
                        {"partition": "official test", "condition": "clean"},
                        records=splits["test"],
                        classes=classes,
                    )
                    stored["clean"] = clean
                    arm_result["full_test"]["clean"] = _condition_metrics(clean, policy, config)
                    _register_array(arm_result, "full_test/clean", clean_path)
                    publish_progress()
                clean = stored["clean"]
                for corruption in tqdm(
                    selected_conditions,
                    desc=f"{arm} corruptions",
                    unit="condition",
                    leave=False,
                    disable=not progress,
                ):
                    if corruption.identifier in arm_result["full_test"]:
                        continue
                    path = _arrays_path(config, f"full-test/{arm}", corruption.identifier)

                    def compute_shift(
                        selected: Corruption = corruption,
                        selected_arm: Arm = arm,
                    ) -> dict[str, np.ndarray[Any, Any]]:
                        return _collect_clean_or_shift(
                            get_model(),
                            splits["test"],
                            config,
                            device,
                            lambda pixels, ids: apply_corruption(
                                pixels, ids, selected, config.integer("attack", "evaluation_seed")
                            ),
                            progress=progress,
                            description=f"{selected_arm} {selected.identifier}",
                        )

                    shifted = _get_or_compute(
                        path,
                        provenance,
                        compute_shift,
                        {"partition": "official test", "condition": corruption.identifier},
                        records=splits["test"],
                        classes=classes,
                    )
                    _assert_alignment(clean, shifted)
                    arm_result["full_test"][corruption.identifier] = _condition_metrics(
                        shifted, policy, config
                    )
                    _register_array(arm_result, f"full_test/{corruption.identifier}", path)
                    publish_progress()
                if selected_attacks:
                    attack_clean_path = _arrays_path(config, f"attack/{arm}", "clean")
                    if "attack_clean" not in stored:
                        attack_clean = _get_or_compute(
                            attack_clean_path,
                            provenance,
                            lambda selected_arm=arm: _collect_clean_or_shift(  # type: ignore[misc]
                                get_model(),
                                splits["attack"],
                                config,
                                device,
                                progress=progress,
                                description=f"{selected_arm} clean attack subset",
                            ),
                            {"partition": attack_partition, "condition": "clean"},
                            records=splits["attack"],
                            classes=classes,
                        )
                        stored["attack_clean"] = attack_clean
                        _register_array(arm_result, "attack_subset/clean", attack_clean_path)
                        publish_progress()
                    attack_clean = stored["attack_clean"]
                    for attack_name in tqdm(
                        selected_attacks,
                        desc=f"{arm} attacks",
                        unit="attack",
                        leave=False,
                        disable=not progress,
                    ):
                        if attack_name in arm_result["attack_subset"]:
                            continue
                        path = _arrays_path(config, f"attack/{arm}", attack_name)

                        def compute_attack(
                            selected: str = attack_name,
                            selected_arm: Arm = arm,
                        ) -> dict[str, np.ndarray[Any, Any]]:
                            return _collect_attack(
                                get_model(),
                                splits["attack"],
                                config,
                                device,
                                selected,
                                progress=progress,
                                description=f"{selected_arm} {selected}",
                            )

                        attacked = _get_or_compute(
                            path,
                            provenance,
                            compute_attack,
                            {
                                "partition": attack_partition,
                                "condition": attack_name,
                                "finite_attack": True,
                            },
                            records=splits["attack"],
                            classes=classes,
                        )
                        arm_result["attack_subset"][attack_name] = _attack_metrics(
                            attack_clean, attacked, classes
                        )
                        _register_array(arm_result, f"attack_subset/{attack_name}", path)
                        publish_progress()
            finally:
                if model is not None:
                    model.to("cpu")
                    del model
                    gc.collect()
                    if device.type == "mps":
                        torch.mps.empty_cache()
        synchronize(device)
        result["elapsed_seconds"] = time.perf_counter() - started
        result["final_memory"] = memory_snapshot(device)
        publish_progress()
        status(
            "Evaluation complete: all registered conditions are available."
            if result["status"] == "complete"
            else f"Evaluation section {section} complete; remaining sections are pending.",
            enabled=progress,
        )
        return result
    except BaseException as error:
        record_failure(failure, "evaluate", error, {"device": str(device), "section": section})
        status(f"Evaluation stopped — {error}", enabled=progress)
        raise
