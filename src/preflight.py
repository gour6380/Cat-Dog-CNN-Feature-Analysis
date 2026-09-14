"""Hardware and scientific-validity preflight for the configured run."""

from __future__ import annotations

import gc
import json
import os
import time
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as nnf

from src.attacks import AttackSpec, batchnorm_state, fgsm_attack, pgd_attack
from src.config import ExperimentConfig
from src.data import PetRecordDataset, epoch_order, load_registered_splits
from src.io_utils import atomic_write_json, environment_snapshot, source_hash, utc_now
from src.model import build_model, state_dict_hash
from src.progress import status
from src.runtime import (
    SafetyStop,
    ensure_finite,
    ensure_memory,
    memory_snapshot,
    record_failure,
    synchronize,
)
from src.training import experiment_provenance, learning_rate_factor


class _LinearProbe(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs.flatten(1).mean(dim=1)
        return torch.stack((score, -score), dim=1)


def _assert_setup(config: ExperimentConfig) -> dict[str, Any]:
    path = config.project_path("artifacts") / "setup.json"
    if not path.is_file():
        raise SafetyStop("setup manifest is missing; run setup first")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise SafetyStop("setup manifest is invalid; run setup again")
    return cast(dict[str, Any], value)


def _software_invariants() -> dict[str, Any]:
    import numpy as np
    from sklearn.decomposition import PCA

    from src.calibration import fit_confidence_threshold, tie_aware_risk_coverage
    from src.metrics import class_centroids, knn_metrics, linear_cka

    confidence = np.asarray([0.9, 0.9, 0.7, 0.4], dtype=np.float64)
    correct = np.asarray([True, False, True, False], dtype=np.bool_)
    coverages, risks, aurc = tie_aware_risk_coverage(confidence, correct)
    if not np.allclose(coverages, [0.5, 0.75, 1.0]):
        raise SafetyStop("tie-aware risk/coverage grouping failed")
    threshold, coverage, ties = fit_confidence_threshold(confidence, 0.5)
    if threshold != 0.9 or coverage != 0.5 or ties != 2:
        raise SafetyStop("tie-aware threshold fitting failed")
    generator = np.random.default_rng(7)
    first = generator.normal(size=(20, 8))
    orthogonal, _ = np.linalg.qr(generator.normal(size=(8, 8)))
    if abs(linear_cka(first, first @ orthogonal * 3.0) - 1.0) > 1e-10:
        raise SafetyStop("linear CKA invariance failed")
    labels = np.repeat(np.arange(2), 10).astype(np.int64)
    centroids = class_centroids(first, labels, 2)
    if centroids.shape != (2, 8):
        raise SafetyStop("centroid geometry shape failed")
    knn = knn_metrics(first, labels, first, labels, neighbours=5)
    if len(knn.per_sample_retention) != len(first):
        raise SafetyStop("k-NN sample alignment failed")
    pca_a = PCA(n_components=2, svd_solver="full").fit_transform(first)
    pca_b = PCA(n_components=2, svd_solver="full").fit_transform(first)
    if not np.array_equal(pca_a, pca_b):
        raise SafetyStop("PCA transform is not fixed")
    schedule = [learning_rate_factor(index, 2, 6) for index in range(1, 7)]
    if schedule[:2] != [0.5, 1.0] or schedule[-1] != 0.0:
        raise SafetyStop("warm-up/cosine schedule failed")
    return {
        "tie_aware_aurc": aurc,
        "threshold": threshold,
        "threshold_ties": ties,
        "cka_orthogonal_scale_invariance": True,
        "centroid_and_knn": True,
        "pca_repeatability": True,
        "schedule": schedule,
    }


def _synthetic_attack_invariants() -> dict[str, Any]:
    model = _LinearProbe().eval()
    inputs = torch.full((4, 3, 4, 4), 0.4, dtype=torch.float32)
    labels = torch.zeros(4, dtype=torch.long)
    ids = [f"synthetic-{index}" for index in range(4)]
    zero = pgd_attack(
        model,
        inputs,
        labels,
        ids,
        AttackSpec(0.0, 1.0, 0, True, 1, 3),
    )
    if not torch.equal(inputs, zero.adversarial):
        raise SafetyStop("epsilon-zero attack is not exactly equivalent")
    fgsm = fgsm_attack(model, inputs, labels, ids, 4.0 / 255.0, 4)
    pgd = pgd_attack(
        model,
        inputs,
        labels,
        ids,
        AttackSpec(4.0 / 255.0, 1.0 / 255.0, 20, False, 1, 4),
    )
    if float((pgd.adversarial - inputs).abs().amax()) > 4.0 / 255.0 + 1e-7:
        raise SafetyStop("synthetic PGD exceeded its L-infinity bound")
    if float(pgd.best_loss.mean()) + 1e-7 < float(fgsm.best_loss.mean()):
        raise SafetyStop("PGD-20 is weaker than FGSM on the registered synthetic check")
    first = pgd_attack(
        model,
        inputs,
        labels,
        ids,
        AttackSpec(4.0 / 255.0, 1.0 / 255.0, 2, True, 1, 11),
    ).adversarial
    repeat = pgd_attack(
        model,
        inputs,
        labels,
        ids,
        AttackSpec(4.0 / 255.0, 1.0 / 255.0, 2, True, 1, 11),
    ).adversarial
    other = pgd_attack(
        model,
        inputs,
        labels,
        ids,
        AttackSpec(4.0 / 255.0, 1.0 / 255.0, 2, True, 1, 12),
    ).adversarial
    if not torch.equal(first, repeat) or torch.equal(first, other):
        raise SafetyStop("random-start seed isolation failed")
    return {
        "epsilon_zero_equivalence": True,
        "linf_and_clipping": True,
        "pgd20_loss_not_below_fgsm": True,
        "random_start_repeatability_and_isolation": True,
    }


def run_preflight(
    config: ExperimentConfig, device: torch.device, *, progress: bool = True
) -> dict[str, Any]:
    failure = config.project_path("artifacts") / "failures" / "preflight.json"
    started = time.perf_counter()
    try:
        status(
            "Preflight: validating the registered setup and configured backend...",
            enabled=progress,
        )
        setup = _assert_setup(config)
        configured_device = config.value("training", "device", str)
        if device.type != configured_device:
            raise SafetyStop(
                f"preflight device {device.type} does not match training.device {configured_device}"
            )
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
            raise SafetyStop("PYTORCH_ENABLE_MPS_FALLBACK must not enable silent fallback")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
        start_memory = ensure_memory(
            device, config.number("preflight", "minimum_available_memory_gib")
        )
        status(
            "Preflight: checking data splits, augmentations, and matched ordering...",
            enabled=progress,
        )
        splits = load_registered_splits(config)
        training = splits["training"]
        calibration = splits["calibration"]
        test = splits["test"]
        classes = config.integer("dataset", "classes")
        if {item.label for item in training} != set(range(classes)):
            raise SafetyStop("training class coverage failed")
        if {item.label for item in calibration} != set(range(classes)):
            raise SafetyStop("calibration class coverage failed")
        if {item.sample_id for item in training} & {item.sample_id for item in calibration}:
            raise SafetyStop("training/calibration overlap detected")
        if {item.sample_id for item in test} & {
            item.sample_id for item in [*training, *calibration]
        }:
            raise SafetyStop("official test partition overlap detected")
        dataset_a = PetRecordDataset(training[:1], config, training=True, epoch=1)
        dataset_b = PetRecordDataset(training[:1], config, training=True, epoch=1)
        if not torch.equal(dataset_a[0][0], dataset_b[0][0]):
            raise SafetyStop("deterministic augmentation repeatability failed")
        order_standard = [
            item.sample_id
            for item in epoch_order(training, config.integer("training", "data_seed"), 1)
        ]
        order_adversarial = [
            item.sample_id
            for item in epoch_order(training, config.integer("training", "data_seed"), 1)
        ]
        if order_standard != order_adversarial:
            raise SafetyStop("matched sample order failed")
        status(
            f"Preflight: checking initialization and CPU/{device.type} logit parity...",
            enabled=progress,
        )
        first_model = build_model(config)
        second_model = build_model(config)
        first_hash = state_dict_hash(first_model.state_dict())
        second_hash = state_dict_hash(second_model.state_dict())
        expected_init = experiment_provenance(config)["initialization_sha256"]
        if first_hash != second_hash or first_hash != expected_init:
            raise SafetyStop("matched initialization hash failed")
        probe_batch_size = min(config.integer("training", "micro_batch_size"), len(test))
        evaluation_dataset = PetRecordDataset(test[:probe_batch_size], config, training=False)
        pixels = torch.stack([evaluation_dataset[index][0] for index in range(probe_batch_size)])
        labels = torch.tensor([evaluation_dataset[index][1] for index in range(probe_batch_size)])
        ids = [evaluation_dataset[index][2] for index in range(probe_batch_size)]
        first_model.eval()
        with torch.inference_mode():
            cpu_logits = first_model(pixels[:2]).detach()
        mps_model = second_model.to(device=device, dtype=torch.float32).eval()
        with torch.inference_mode():
            mps_logits = mps_model(pixels[:2].to(device)).detach().cpu()
        ensure_finite("MPS parity logits", mps_logits)
        maximum_delta = float((cpu_logits - mps_logits).abs().max().item())
        atol = config.number("preflight", "parity_atol")
        rtol = config.number("preflight", "parity_rtol")
        if not torch.allclose(cpu_logits, mps_logits, atol=atol, rtol=rtol):
            raise SafetyStop(
                f"CPU/{device.type} logit parity failed: max delta {maximum_delta:.6g}, "
                f"atol={atol}, rtol={rtol}"
            )
        del first_model
        status(
            "Preflight: checking configured PGD, BatchNorm state, gradients, and memory...",
            enabled=progress,
        )
        pixels_mps = pixels.to(device=device, dtype=torch.float32)
        labels_mps = labels.to(device=device, dtype=torch.long)
        mps_model.train()
        bn_before = batchnorm_state(mps_model)
        attack = pgd_attack(
            mps_model,
            pixels_mps,
            labels_mps,
            ids,
            AttackSpec(
                epsilon=config.number("attack", "epsilon"),
                step_size=config.number("attack", "step_size"),
                steps=config.integer("attack", "train_steps"),
                random_start=bool(config.section("attack").get("random_start")),
                restarts=1,
                seed=config.integer("training", "attack_seed"),
            ),
        )
        bn_after = batchnorm_state(mps_model)
        for name in bn_before:
            if any(
                not torch.equal(left, right)
                for left, right in zip(bn_before[name], bn_after[name], strict=True)
            ):
                raise SafetyStop(f"BatchNorm state changed in configured PGD probe: {name}")
        mps_model.train()
        logits = mps_model(attack.adversarial)
        loss = nnf.cross_entropy(logits, labels_mps)
        ensure_finite("MPS preflight loss", loss)
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_count = 0
        for parameter in mps_model.parameters():
            if parameter.grad is not None:
                ensure_finite("MPS preflight gradient", parameter.grad)
                gradient_count += 1
        if gradient_count == 0:
            raise SafetyStop("MPS preflight produced no parameter gradients")
        actual_linf = float((attack.adversarial - pixels_mps).abs().amax().item())
        synchronize(device)
        peak_memory = memory_snapshot(device)
        del mps_model, attack, pixels_mps, labels_mps, logits, loss
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        synchronize(device)
        final_memory = ensure_memory(
            device, config.number("preflight", "minimum_available_memory_gib")
        )
        growth = final_memory.get("mps_driver_mib", 0.0) - start_memory.get("mps_driver_mib", 0.0)
        maximum_growth = config.number("preflight", "maximum_memory_growth_mib")
        if growth > maximum_growth:
            raise SafetyStop(
                "unstable MPS memory after cleanup: "
                f"grew {growth:.1f} MiB, limit {maximum_growth:.1f}"
            )
        result = {
            "schema_version": 1,
            "status": "passed",
            "created_at": utc_now(),
            "config_sha256": config.sha256,
            "setup_source_sha256": setup.get("source_sha256"),
            "current_source_sha256": source_hash(config.root),
            "provenance": experiment_provenance(config),
            "environment": environment_snapshot(),
            "device": str(device),
            "silent_fallback_disabled": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0",
            "data": {
                "training": len(training),
                "calibration": len(calibration),
                "test": len(test),
                "classes": classes,
                "split_disjoint": True,
                "official_test_preserved": True,
                "augmentation_repeatable": True,
                "sample_order_matched": True,
                "attack_subset": len(splits["attack"]),
                "projection_subset": len(splits["projection"]),
            },
            "model": {
                "initialization_sha256": first_hash,
                "matched_initialization": True,
                "feature_dimension": config.integer("model", "feature_dim"),
                "cpu_device_max_logit_delta": maximum_delta,
                "cpu_device_allclose": True,
                "finite_gradients": True,
                "gradient_parameter_count": gradient_count,
                "batchnorm_unchanged_during_attack": True,
            },
            "software_invariants": _software_invariants(),
            "synthetic_attack_invariants": _synthetic_attack_invariants(),
            "actual_attack": {
                "batch_size": probe_batch_size,
                "input_size": config.integer("input", "size"),
                "steps": config.integer("attack", "train_steps"),
                "linf_max": actual_linf,
                "finite": True,
            },
            "memory": {
                "start": start_memory,
                "peak_probe": peak_memory,
                "after_cleanup": final_memory,
                "driver_growth_mib": growth,
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(config.project_path("artifacts") / "preflight.json", result)
        status("Preflight passed: the configured experiment may train.", enabled=progress)
        return result
    except BaseException as error:
        record_failure(failure, "preflight", error, {"device": str(device)})
        raise
