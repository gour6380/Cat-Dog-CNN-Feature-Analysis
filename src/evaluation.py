"""Aligned clean, attack, corruption, calibration, and feature evaluation."""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from src.attacks import AttackSpec, fgsm_attack, pgd_attack
from src.calibration import CalibrationPolicy, apply_policy, fit_policy, negative_log_likelihood
from src.config import ExperimentConfig
from src.corruptions import Corruption, apply_corruption, registered_corruptions
from src.data import SampleRecord, build_loader, load_registered_splits
from src.io_utils import atomic_save_npz, atomic_write_json, sha256_file, utc_now
from src.metrics import accuracy_metrics
from src.model import NormalizedResNet18, load_checkpoint_model
from src.runtime import ensure_finite, ensure_memory, memory_snapshot, record_failure, synchronize
from src.training import Arm, checkpoint_path, experiment_provenance

BatchTransform = Callable[[torch.Tensor, list[str]], torch.Tensor]


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
    value = json.loads(metadata.read_text(encoding="utf-8"))
    return isinstance(value, dict) and value.get("provenance") == expected


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
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _collect_clean_or_shift(
    model: NormalizedResNet18,
    records: list[SampleRecord],
    config: ExperimentConfig,
    device: torch.device,
    transform: BatchTransform | None = None,
) -> dict[str, np.ndarray[Any, Any]]:
    logits_parts: list[np.ndarray[Any, Any]] = []
    feature_parts: list[np.ndarray[Any, Any]] = []
    label_parts: list[np.ndarray[Any, Any]] = []
    species_parts: list[np.ndarray[Any, Any]] = []
    ids: list[str] = []
    loader = build_loader(records, config, training=False)
    model.eval()
    for pixels, labels, sample_ids_raw, species in loader:
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
    return {
        "sample_ids": np.asarray(ids, dtype=np.str_),
        "labels": np.concatenate(label_parts),
        "species": np.concatenate(species_parts),
        "logits": np.concatenate(logits_parts),
        "features": np.concatenate(feature_parts),
    }


def _collect_attack(
    model: NormalizedResNet18,
    records: list[SampleRecord],
    config: ExperimentConfig,
    device: torch.device,
    kind: str,
) -> dict[str, np.ndarray[Any, Any]]:
    logits_parts: list[np.ndarray[Any, Any]] = []
    feature_parts: list[np.ndarray[Any, Any]] = []
    label_parts: list[np.ndarray[Any, Any]] = []
    species_parts: list[np.ndarray[Any, Any]] = []
    cumulative_parts: list[list[np.ndarray[Any, Any]]] | None = None
    ids: list[str] = []
    loader = build_loader(records, config, training=False)
    for pixels, labels, sample_ids_raw, species in loader:
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
        elif kind == "pgd20x5":
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
    return arrays


def _artifact_provenance(
    config: ExperimentConfig, arm: Arm, checkpoint_sha256: str
) -> dict[str, str]:
    return {**experiment_provenance(config), "arm": arm, "checkpoint_sha256": checkpoint_sha256}


def _get_or_compute(
    path: Path,
    provenance: dict[str, str],
    compute: Callable[[], dict[str, np.ndarray[Any, Any]]],
    extra: dict[str, Any],
) -> dict[str, np.ndarray[Any, Any]]:
    if _cache_valid(path, provenance):
        return load_arrays(path)
    arrays = compute()
    _save_arrays(path, arrays, provenance, extra)
    return arrays


def _assert_alignment(
    reference: dict[str, np.ndarray[Any, Any]], other: dict[str, np.ndarray[Any, Any]]
) -> None:
    if not np.array_equal(reference["sample_ids"], other["sample_ids"]):
        raise EvaluationError("sample IDs are not aligned")
    if not np.array_equal(reference["labels"], other["labels"]):
        raise EvaluationError("labels are not aligned")


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


def evaluate(config: ExperimentConfig, device: torch.device) -> dict[str, Any]:
    failure = config.project_path("artifacts") / "failures" / "evaluate.json"
    started = time.perf_counter()
    try:
        ensure_memory(device, config.number("preflight", "minimum_available_memory_gib"))
        splits = load_registered_splits(config)
        conditions = registered_corruptions(config.section("corruptions"))
        all_results: dict[str, Any] = {
            "schema_version": 1,
            "created_at": utc_now(),
            "config_sha256": config.sha256,
            "device": str(device),
            "arms": {},
            "finite_attack_scope": "FGSM and PGD-20x5 at L-infinity epsilon 4/255",
        }
        for arm in ("standard", "adversarial"):
            selected_checkpoint = checkpoint_path(config, arm)
            if not selected_checkpoint.is_file():
                raise EvaluationError(
                    f"fixed epoch-15 checkpoint is missing: {selected_checkpoint}"
                )
            checkpoint_hash = sha256_file(selected_checkpoint)
            provenance = _artifact_provenance(config, arm, checkpoint_hash)
            model_for_arm = load_checkpoint_model(config, selected_checkpoint, device)
            calibration_path = _arrays_path(config, "calibration", arm)
            calibration = _get_or_compute(
                calibration_path,
                provenance,
                lambda selected_model=model_for_arm: _collect_clean_or_shift(  # type: ignore[misc]
                    selected_model, splits["calibration"], config, device
                ),
                {"partition": "calibration", "condition": "clean"},
            )
            policy = _policy_from_calibration(config, calibration)
            clean_calibration_nll = negative_log_likelihood(
                calibration["logits"].astype(np.float64),
                calibration["labels"].astype(np.int64),
            )
            arm_result: dict[str, Any] = {
                "checkpoint": str(selected_checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "policy_fit_partition": "clean calibration only",
                "policy": asdict(policy),
                "calibration_nll_before": clean_calibration_nll,
                "calibration_nll_after": negative_log_likelihood(
                    calibration["logits"].astype(np.float64),
                    calibration["labels"].astype(np.int64),
                    policy.temperature,
                ),
                "full_test": {},
                "attack_subset": {},
            }
            clean_path = _arrays_path(config, f"full-test/{arm}", "clean")
            clean_full = _get_or_compute(
                clean_path,
                provenance,
                lambda selected_model=model_for_arm: _collect_clean_or_shift(  # type: ignore[misc]
                    selected_model, splits["test"], config, device
                ),
                {"partition": "official test", "condition": "clean"},
            )
            classes = config.integer("dataset", "classes")
            labels_full = clean_full["labels"].astype(np.int64)
            clean_metrics = accuracy_metrics(
                clean_full["logits"].astype(np.float64), labels_full, classes
            )
            clean_metrics["risk"] = apply_policy(
                clean_full["logits"].astype(np.float64),
                labels_full,
                policy,
                config.integer("calibration", "ece_bins"),
            )
            arm_result["full_test"]["clean"] = clean_metrics
            for corruption in conditions:
                condition_path = _arrays_path(config, f"full-test/{arm}", corruption.identifier)

                def compute_shift(
                    selected: Corruption = corruption,
                    selected_model: NormalizedResNet18 = model_for_arm,
                ) -> dict[str, np.ndarray[Any, Any]]:
                    return _collect_clean_or_shift(
                        selected_model,
                        splits["test"],
                        config,
                        device,
                        lambda pixels, ids: apply_corruption(
                            pixels,
                            ids,
                            selected,
                            config.integer("attack", "evaluation_seed"),
                        ),
                    )

                shifted = _get_or_compute(
                    condition_path,
                    provenance,
                    compute_shift,
                    {"partition": "official test", "condition": corruption.identifier},
                )
                _assert_alignment(clean_full, shifted)
                shifted_logits = shifted["logits"].astype(np.float64)
                metrics = accuracy_metrics(shifted_logits, labels_full, classes)
                metrics["risk"] = apply_policy(
                    shifted_logits,
                    labels_full,
                    policy,
                    config.integer("calibration", "ece_bins"),
                )
                arm_result["full_test"][corruption.identifier] = metrics
            attack_clean_path = _arrays_path(config, f"attack/{arm}", "clean")
            attack_clean = _get_or_compute(
                attack_clean_path,
                provenance,
                lambda selected_model=model_for_arm: _collect_clean_or_shift(  # type: ignore[misc]
                    selected_model, splits["attack"], config, device
                ),
                {"partition": "fixed 20-per-breed test subset", "condition": "clean"},
            )
            for attack_name in ("fgsm", "pgd20x5"):
                attack_path = _arrays_path(config, f"attack/{arm}", attack_name)
                attacked = _get_or_compute(
                    attack_path,
                    provenance,
                    lambda selected=attack_name, selected_model=model_for_arm: _collect_attack(  # type: ignore[misc]
                        selected_model, splits["attack"], config, device, selected
                    ),
                    {
                        "partition": "fixed 20-per-breed test subset",
                        "condition": attack_name,
                        "finite_attack": True,
                    },
                )
                arm_result["attack_subset"][attack_name] = _attack_metrics(
                    attack_clean, attacked, classes
                )
            fgsm_robust = arm_result["attack_subset"]["fgsm"]["robust_accuracy"]
            pgd_robust = arm_result["attack_subset"]["pgd20x5"]["robust_accuracy"]
            arm_result["attack_strength_check"] = {
                "pgd20x5_not_weaker_by_robust_accuracy": bool(pgd_robust <= fgsm_robust),
                "fgsm_robust_accuracy": fgsm_robust,
                "pgd20x5_robust_accuracy": pgd_robust,
            }
            all_results["arms"][arm] = arm_result
            model_for_arm.to("cpu")
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
        clean_target = config.number("calibration", "target_coverage")
        policy_shift: dict[str, Any] = {}
        for arm, arm_result_any in all_results["arms"].items():
            arm_result = cast(dict[str, Any], arm_result_any)
            clean_risk = arm_result["full_test"]["clean"]["risk"]
            flagged: list[dict[str, Any]] = []
            for condition, metrics in arm_result["full_test"].items():
                risk = metrics["risk"]
                coverage_delta_target = float(risk["coverage"] - clean_target)
                risk_delta_clean = float(risk["selective_risk"] - clean_risk["selective_risk"])
                if abs(coverage_delta_target) > 0.05 or abs(risk_delta_clean) > 0.02:
                    flagged.append(
                        {
                            "condition": condition,
                            "coverage_delta_from_target": coverage_delta_target,
                            "selective_risk_delta_from_clean": risk_delta_clean,
                        }
                    )
            policy_shift[arm] = {"hypothesis_met": bool(flagged), "flagged_conditions": flagged}
        all_results["secondary_policy_shift"] = policy_shift
        synchronize(device)
        all_results["elapsed_seconds"] = time.perf_counter() - started
        all_results["final_memory"] = memory_snapshot(device)
        destination = config.project_path("results") / "evaluation.json"
        atomic_write_json(destination, all_results)
        return all_results
    except BaseException as error:
        record_failure(failure, "evaluate", error, {"device": str(device)})
        raise
