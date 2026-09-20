"""Matched, resumable standard and adversarial training."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn import functional as nnf

from src.attacks import AttackSpec, pgd_attack
from src.config import ExperimentConfig
from src.data import build_loader, epoch_order, load_registered_splits
from src.io_utils import atomic_write_json, canonical_json_bytes, sha256_bytes, utc_now
from src.model import NormalizedResNet18, atomic_torch_save, build_model, state_dict_hash
from src.pilot_protocol import (
    collect_pilot_validation,
    pilot_enabled,
    pilot_phase,
    pilot_training_loss,
    prepare_pilot_protocol,
    register_pilot_training_source,
)
from src.progress import status, tqdm
from src.runtime import (
    SafetyStop,
    ensure_finite,
    memory_snapshot,
    record_failure,
    synchronize,
)
from src.training_monitoring import (
    collect_clean_monitoring,
    prepare_training_monitoring,
    register_training_monitoring_source,
    save_training_examples,
    training_monitoring_enabled,
    training_objective_metrics,
)

Arm = Literal["standard", "adversarial"]


class ResumeError(RuntimeError):
    """Checkpoint provenance does not match the registered run."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResumeError(f"required manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResumeError(f"invalid manifest: {path}")
    return cast(dict[str, Any], value)


def experiment_provenance(config: ExperimentConfig) -> dict[str, str]:
    artifacts = config.project_path("artifacts")
    data = _read_json(artifacts / "data" / "manifest.json")
    initialization = _read_json(artifacts / "model" / "initialization.json")
    dataset_hash = data.get("dataset_content_sha256")
    split_hash = data.get("split_sha256")
    initialization_hash = initialization.get("state_sha256")
    if not all(isinstance(value, str) for value in (dataset_hash, split_hash, initialization_hash)):
        raise ResumeError("provenance manifests do not contain required hashes")
    combined_fields = {
        "config": config.sha256,
        "dataset": dataset_hash,
        "split": split_hash,
        "initialization": initialization_hash,
    }
    optional_fields: dict[str, str] = {}
    if pilot_enabled(config):
        protocol = prepare_pilot_protocol(config)
        combined_fields["pilot_protocol"] = protocol.sha256
        optional_fields["pilot_protocol_sha256"] = protocol.sha256
        source_fingerprint = register_pilot_training_source(config)
        combined_fields["pilot_training_source"] = source_fingerprint
        optional_fields["pilot_training_source_sha256"] = source_fingerprint
    if training_monitoring_enabled(config):
        monitoring_protocol = prepare_training_monitoring(config)
        combined_fields["training_monitoring_protocol"] = monitoring_protocol.sha256
        optional_fields["training_monitoring_protocol_sha256"] = monitoring_protocol.sha256
        monitoring_source = register_training_monitoring_source(config)
        combined_fields["training_monitoring_source"] = monitoring_source
        optional_fields["training_monitoring_source_sha256"] = monitoring_source
    combined = sha256_bytes(canonical_json_bytes(combined_fields))
    return {
        "config_sha256": config.sha256,
        "dataset_sha256": cast(str, dataset_hash),
        "split_sha256": cast(str, split_hash),
        "initialization_sha256": cast(str, initialization_hash),
        "combined_sha256": combined,
        **optional_fields,
    }


def _optimizer(model: NormalizedResNet18, config: ExperimentConfig) -> torch.optim.AdamW:
    head_ids = {id(parameter) for parameter in model.classifier.parameters()}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    head = list(model.classifier.parameters())
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": config.number("training", "backbone_learning_rate")},
            {"params": head, "lr": config.number("training", "head_learning_rate")},
        ],
        weight_decay=config.number("training", "weight_decay"),
    )


def learning_rate_factor(update: int, warmup_updates: int, total_updates: int) -> float:
    if not 1 <= update <= total_updates:
        raise ValueError("update index is outside the schedule")
    if update <= warmup_updates:
        return update / warmup_updates
    if total_updates == warmup_updates:
        return 1.0
    progress = (update - warmup_updates) / (total_updates - warmup_updates)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def accumulation_window_size(batch_index: int, batches_per_epoch: int, accumulation: int) -> int:
    """Return the divisor for a complete or final partial accumulation window."""

    if not 1 <= batch_index <= batches_per_epoch or accumulation <= 0:
        raise ValueError("invalid gradient-accumulation position")
    window_start = ((batch_index - 1) // accumulation) * accumulation + 1
    window_end = min(window_start + accumulation - 1, batches_per_epoch)
    return window_end - window_start + 1


def _set_learning_rates(
    optimizer: torch.optim.AdamW,
    config: ExperimentConfig,
    update: int,
    warmup_updates: int,
    total_updates: int,
) -> list[float]:
    factor = learning_rate_factor(update, warmup_updates, total_updates)
    bases = (
        config.number("training", "backbone_learning_rate"),
        config.number("training", "head_learning_rate"),
    )
    values = []
    for group, base in zip(optimizer.param_groups, bases, strict=True):
        value = base * factor
        group["lr"] = value
        values.append(value)
    return values


def _order_hash(records: list[Any]) -> str:
    joined = "\n".join(record.sample_id for record in records).encode()
    return hashlib.sha256(joined).hexdigest()


def checkpoint_path(config: ExperimentConfig, arm: Arm, epoch: int | None = None) -> Path:
    selected_epoch = epoch or config.integer("training", "epochs")
    return (
        config.project_path("checkpoints")
        / config.sha256[:16]
        / arm
        / f"epoch-{selected_epoch:03d}.pt"
    )


def _validate_checkpoint(payload: dict[str, Any], arm: Arm, provenance: dict[str, str]) -> None:
    if payload.get("arm") != arm:
        raise ResumeError("checkpoint arm does not match")
    if payload.get("provenance") != provenance:
        raise ResumeError("checkpoint config/data/initialization provenance does not match")
    if not isinstance(payload.get("epoch"), int) or not isinstance(
        payload.get("update_count"), int
    ):
        raise ResumeError("checkpoint epoch/update metadata is invalid")


def _latest_checkpoint(config: ExperimentConfig, arm: Arm) -> Path | None:
    directory = config.project_path("checkpoints") / config.sha256[:16] / arm
    candidates = sorted(directory.glob("epoch-*.pt")) if directory.is_dir() else []
    return candidates[-1] if candidates else None


def _load_resume(
    config: ExperimentConfig,
    arm: Arm,
    model: NormalizedResNet18,
    optimizer: torch.optim.AdamW,
    provenance: dict[str, str],
) -> tuple[int, int, list[dict[str, Any]]]:
    latest = _latest_checkpoint(config, arm)
    if latest is None:
        return 0, 0, []
    raw = torch.load(latest, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ResumeError(f"invalid checkpoint payload: {latest}")
    payload = cast(dict[str, Any], raw)
    _validate_checkpoint(payload, arm, provenance)
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["model_state"]), strict=True)
    optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer_state"]))
    history = payload.get("history")
    if not isinstance(history, list):
        raise ResumeError("checkpoint history is invalid")
    return int(payload["epoch"]), int(payload["update_count"]), cast(list[dict[str, Any]], history)


def _assert_finite_gradients(model: nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        ensure_finite("training gradient", parameter.grad)
        squared += float(parameter.grad.detach().float().square().sum().item())
    norm = math.sqrt(squared)
    if not math.isfinite(norm):
        raise SafetyStop("non-finite aggregate gradient norm")
    return norm


def _repair_completed_journal(
    config: ExperimentConfig,
    arm: Arm,
    provenance: dict[str, str],
    initialization_hash: str,
    epoch: int,
    updates: int,
    updates_per_epoch: int,
    history: list[dict[str, Any]],
    extra: dict[str, Any],
) -> bool:
    """Reconcile the two-file commit gap from authoritative checkpoint evidence.

    An epoch checkpoint is written before its convenient JSON journal. A kernel
    interruption between those atomic writes must not leave completed-resume
    visualizations using older epochs. Invocation timing cannot be recovered
    from a missing journal; keep it absent rather than inventing measurements.
    """

    path = config.project_path("artifacts") / "training" / f"{arm}.json"
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = raw
        except (OSError, ValueError):
            pass  # The verified checkpoint, not this presentation journal, wins.
    authoritative = {
        "status": "complete",
        "arm": arm,
        "provenance": provenance,
        "initialization_sha256": initialization_hash,
        "completed_epochs": epoch,
        "completed_updates": updates,
        "updates_per_epoch": updates_per_epoch,
        "history": history,
        "checkpoint": str(checkpoint_path(config, arm, epoch)),
        **extra,
    }
    if all(existing.get(key) == value for key, value in authoritative.items()):
        return False
    same_run = existing.get("provenance") == provenance and existing.get("arm") == arm
    repaired = {**(existing if same_run else {}), **authoritative}
    if history and "memory" in history[-1]:
        repaired["current_memory"] = history[-1]["memory"]
    repaired["journal_recovery"] = {
        "repaired_at": utc_now(),
        "method": "authoritative_verified_completed_checkpoint",
        "previous_completed_epochs": existing.get("completed_epochs"),
        "invocation_telemetry_preserved": same_run,
        "missing_invocation_telemetry_not_reconstructed": not same_run,
        "preserved_telemetry_may_cover_an_earlier_epoch": (
            same_run and existing.get("completed_epochs") != epoch
        ),
    }
    atomic_write_json(path, repaired)
    return True


def train_arm(
    config: ExperimentConfig,
    arm: Arm,
    device: torch.device,
    *,
    progress: bool = True,
    epoch_observer: Callable[[list[dict[str, Any]], Arm], None] | None = None,
) -> dict[str, Any]:
    """Train or resume one configured arm without inspecting the test partition."""

    failure = config.project_path("artifacts") / "failures" / f"train-{arm}.json"
    try:
        status(
            f"{arm}: validating provenance before training...",
            enabled=progress,
        )
        start_memory = memory_snapshot(device)
        splits = load_registered_splits(config)
        training_records = splits["training"]
        protocol = None
        monitoring_protocol = None
        if pilot_enabled(config):
            if arm != "adversarial":
                raise ValueError("the exploratory pilot trains only the adversarial arm")
            protocol = prepare_pilot_protocol(config, splits)
            training_records = protocol.fit_records
        if training_monitoring_enabled(config):
            monitoring_protocol = prepare_training_monitoring(config, splits)
            training_records = monitoring_protocol.fit_records
        model = build_model(config).to(device=device, dtype=torch.float32)
        initialization_hash = state_dict_hash(model.state_dict())
        provenance = experiment_provenance(config)
        if initialization_hash != provenance["initialization_sha256"]:
            raise ResumeError(
                "constructed initialization hash differs from registered initialization"
            )
        optimizer = _optimizer(model, config)
        resumed_epoch, completed_updates, history = _load_resume(
            config, arm, model, optimizer, provenance
        )
        epochs = config.integer("training", "epochs")
        micro_batch = config.integer("training", "micro_batch_size")
        accumulation = config.integer("training", "gradient_accumulation_steps")
        batches_per_epoch = math.ceil(len(training_records) / micro_batch)
        updates_per_epoch = math.ceil(batches_per_epoch / accumulation)
        total_updates = epochs * updates_per_epoch
        warmup_updates = config.integer("training", "warmup_epochs") * updates_per_epoch
        expected_resume_updates = resumed_epoch * updates_per_epoch
        if completed_updates != expected_resume_updates:
            raise ResumeError(
                f"checkpoint update count {completed_updates} != expected {expected_resume_updates}"
            )
        if resumed_epoch > epochs:
            raise ResumeError("checkpoint epoch exceeds the configured fixed final epoch")
        repaired_journal = False
        if resumed_epoch == epochs:
            repaired_journal = _repair_completed_journal(
                config,
                arm,
                provenance,
                initialization_hash,
                resumed_epoch,
                completed_updates,
                updates_per_epoch,
                history,
                {
                    **({"pilot_protocol_sha256": protocol.sha256} if protocol is not None else {}),
                    **(
                        {
                            "training_monitoring_protocol_sha256": monitoring_protocol.sha256,
                            "effective_fit_count": len(monitoring_protocol.fit_records),
                            "validation_count": len(monitoring_protocol.validation_records),
                        }
                        if monitoring_protocol is not None
                        else {}
                    ),
                },
            )
        if history and epoch_observer is not None:
            # Reconstruct the live view from checkpoint-backed measurements.
            # The observer receives a copy and cannot modify checkpoint history.
            epoch_observer(deepcopy(history), arm)
        if resumed_epoch == epochs:
            status(
                f"{arm}: reusing the verified epoch-{resumed_epoch} checkpoint.",
                enabled=progress,
            )
            return {
                "status": "complete",
                "arm": arm,
                "checkpoint": str(checkpoint_path(config, arm)),
                "resumed_without_work": True,
                "journal_repaired_from_checkpoint": repaired_journal,
                "epochs": resumed_epoch,
                "updates": completed_updates,
            }
        attack_spec = AttackSpec(
            epsilon=config.number("attack", "epsilon"),
            step_size=config.number("attack", "step_size"),
            steps=config.integer("attack", "train_steps"),
            random_start=bool(config.section("attack").get("random_start")),
            restarts=1,
            seed=config.integer("training", "attack_seed"),
        )
        base_lrs = [
            config.number("training", "backbone_learning_rate"),
            config.number("training", "head_learning_rate"),
        ]
        started = time.perf_counter()
        remaining_epochs = epochs - resumed_epoch
        status(
            f"{arm}: starting {remaining_epochs} remaining epoch(s), "
            f"{batches_per_epoch} batches per epoch on {device.type}.",
            enabled=progress,
        )
        with tqdm(
            range(resumed_epoch + 1, epochs + 1),
            total=remaining_epochs,
            desc=f"{arm} epochs",
            unit="epoch",
            disable=not progress,
        ) as epoch_bar:
            for epoch in epoch_bar:
                epoch_started = time.perf_counter()
                status(f"{arm} — epoch {epoch}/{epochs}: training", enabled=progress)
                ordered = epoch_order(
                    training_records, config.integer("training", "data_seed"), epoch
                )
                loader = build_loader(training_records, config, training=True, epoch=epoch)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                running_loss = 0.0
                running_clean_loss = 0.0
                running_adversarial_loss = 0.0
                sample_count = 0
                objective_confusion = (
                    torch.zeros((config.integer("model", "classes"),) * 2, dtype=torch.int64)
                    if monitoring_protocol is not None
                    else None
                )
                example_rows: list[dict[str, Any]] = []
                gallery_by_id = (
                    {record.sample_id: record for record in monitoring_protocol.gallery_records}
                    if monitoring_protocol is not None and epoch in {1, epochs}
                    else {}
                )
                last_gradient_norm = 0.0
                last_lrs = base_lrs
                with tqdm(
                    loader,
                    total=len(loader),
                    desc=f"{arm} epoch {epoch}",
                    unit="batch",
                    leave=False,
                    disable=not progress,
                ) as batch_bar:
                    for batch_index, batch in enumerate(batch_bar, 1):
                        pixels, labels, sample_ids_raw, _species = batch
                        sample_ids = list(sample_ids_raw)
                        pixels = pixels.to(device=device, dtype=torch.float32)
                        labels = labels.to(device=device, dtype=torch.long)
                        clean_pixels = pixels
                        if protocol is not None:
                            loss, clean_loss, adversarial_loss = pilot_training_loss(
                                model,
                                pixels,
                                labels,
                                sample_ids,
                                epoch,
                                attack_spec,
                                protocol,
                                config,
                            )
                            running_clean_loss += float(clean_loss.detach().item()) * len(labels)
                            if adversarial_loss is not None:
                                running_adversarial_loss += float(
                                    adversarial_loss.detach().item()
                                ) * len(labels)
                        else:
                            if arm == "adversarial":
                                attack_ids = [
                                    f"{sample_id}:epoch:{epoch}" for sample_id in sample_ids
                                ]
                                pixels = pgd_attack(
                                    model, pixels, labels, attack_ids, attack_spec
                                ).adversarial
                            logits = model(pixels)
                            ensure_finite("training logits", logits)
                            loss = nnf.cross_entropy(logits, labels)
                            if monitoring_protocol is not None:
                                assert objective_confusion is not None
                                objective_predictions = logits.detach().argmax(dim=1).cpu()
                                objective_labels = labels.detach().cpu()
                                classes = config.integer("model", "classes")
                                objective_confusion += torch.bincount(
                                    objective_labels * classes + objective_predictions,
                                    minlength=classes * classes,
                                ).reshape(classes, classes)
                                for index, sample_id in enumerate(sample_ids):
                                    if sample_id not in gallery_by_id:
                                        continue
                                    example_rows.append(
                                        {
                                            "sample_id": sample_id,
                                            "label": int(objective_labels[index]),
                                            "clean_pixels": clean_pixels[index]
                                            .detach()
                                            .cpu()
                                            .numpy()
                                            .copy(),
                                            "objective_pixels": pixels[index]
                                            .detach()
                                            .cpu()
                                            .numpy()
                                            .copy(),
                                            "logits": logits[index].detach().cpu().numpy().copy(),
                                            "batch_index": batch_index,
                                            "update_index": completed_updates + 1,
                                            "original_image_path": gallery_by_id[
                                                sample_id
                                            ].image_path,
                                        }
                                    )
                        ensure_finite("training loss", loss)
                        window_size = accumulation_window_size(
                            batch_index, batches_per_epoch, accumulation
                        )
                        (loss / window_size).backward()  # type: ignore[no-untyped-call]
                        running_loss += float(loss.detach().item()) * len(labels)
                        sample_count += len(labels)
                        if batch_index % accumulation == 0 or batch_index == batches_per_epoch:
                            last_gradient_norm = _assert_finite_gradients(model)
                            next_update = completed_updates + 1
                            last_lrs = _set_learning_rates(
                                optimizer, config, next_update, warmup_updates, total_updates
                            )
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            completed_updates = next_update
                        batch_bar.set_postfix(
                            loss=f"{running_loss / sample_count:.4f}",
                            updates=completed_updates,
                        )
                if completed_updates != epoch * updates_per_epoch:
                    raise SafetyStop("optimizer-update count diverged from registered schedule")
                synchronize(device)
                mean_loss = running_loss / sample_count
                epoch_record: dict[str, Any] = {
                    "epoch": epoch,
                    "mean_training_loss": mean_loss,
                    "sample_count": sample_count,
                    "update_count": completed_updates,
                    "order_sha256": _order_hash(ordered),
                    "last_gradient_norm": last_gradient_norm,
                    "last_learning_rates": last_lrs,
                    "seconds": time.perf_counter() - epoch_started,
                    "memory": memory_snapshot(device),
                }
                if protocol is not None:
                    phase = pilot_phase(config, epoch)
                    epoch_record.update(
                        {
                            "pilot_phase": phase,
                            "clean_loss_weight": (
                                1.0
                                if phase == "clean_warmup"
                                else config.number("pilot", "clean_loss_weight")
                            ),
                            "mean_clean_training_loss": running_clean_loss / sample_count,
                            "mean_adversarial_training_loss": (
                                None
                                if phase == "clean_warmup"
                                else running_adversarial_loss / sample_count
                            ),
                        }
                    )
                    status(
                        f"pilot — epoch {epoch}/{epochs}: monitoring held-out training validation "
                        "(no checkpoint selection)",
                        enabled=progress,
                    )
                    epoch_record["validation"] = collect_pilot_validation(
                        model,
                        protocol,
                        config,
                        device,
                        epoch,
                        progress=progress,
                    )
                    synchronize(device)
                    epoch_record["seconds"] = time.perf_counter() - epoch_started
                    epoch_record["memory"] = memory_snapshot(device)
                if monitoring_protocol is not None:
                    assert objective_confusion is not None
                    status(
                        f"{arm} — epoch {epoch}/{epochs}: measuring clean fitting and validation "
                        "(evaluation mode; no checkpoint selection)",
                        enabled=progress,
                    )
                    epoch_record["training_objective"] = training_objective_metrics(
                        objective_confusion, running_loss, sample_count, arm
                    )
                    epoch_record["monitoring"] = {
                        "protocol_sha256": monitoring_protocol.sha256,
                        "checkpoint_selection": "none; fixed configured final epoch",
                        "train_clean": collect_clean_monitoring(
                            model,
                            monitoring_protocol.fit_records,
                            config,
                            device,
                            progress=progress,
                            description=f"{arm} clean fitting",
                        ),
                        "validation_clean": collect_clean_monitoring(
                            model,
                            monitoring_protocol.validation_records,
                            config,
                            device,
                            progress=progress,
                            description=f"{arm} clean validation",
                        ),
                    }
                    if gallery_by_id:
                        epoch_record["training_examples"] = save_training_examples(
                            config,
                            arm,
                            epoch,
                            example_rows,
                            provenance,
                            monitoring_protocol,
                        )
                    synchronize(device)
                    epoch_record["seconds"] = time.perf_counter() - epoch_started
                    epoch_record["memory"] = memory_snapshot(device)
                history.append(epoch_record)
                checkpoint = {
                    "schema_version": 1,
                    "saved_at": utc_now(),
                    "arm": arm,
                    "epoch": epoch,
                    "update_count": completed_updates,
                    "provenance": provenance,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                }
                atomic_torch_save(checkpoint_path(config, arm, epoch), checkpoint)
                atomic_write_json(
                    config.project_path("artifacts") / "training" / f"{arm}.json",
                    {
                        "status": "running" if epoch < epochs else "complete",
                        "arm": arm,
                        "provenance": provenance,
                        "initialization_sha256": initialization_hash,
                        "completed_epochs": epoch,
                        "completed_updates": completed_updates,
                        "updates_per_epoch": updates_per_epoch,
                        "history": history,
                        "checkpoint": str(checkpoint_path(config, arm, epoch)),
                        "started_from_epoch": resumed_epoch,
                        "elapsed_this_invocation_seconds": time.perf_counter() - started,
                        "start_memory": start_memory,
                        "current_memory": memory_snapshot(device),
                        **(
                            {"pilot_protocol_sha256": protocol.sha256}
                            if protocol is not None
                            else {}
                        ),
                        **(
                            {
                                "training_monitoring_protocol_sha256": monitoring_protocol.sha256,
                                "effective_fit_count": len(monitoring_protocol.fit_records),
                                "validation_count": len(monitoring_protocol.validation_records),
                            }
                            if monitoring_protocol is not None
                            else {}
                        ),
                    },
                )
                if epoch_observer is not None:
                    epoch_observer(deepcopy(history), arm)
                epoch_bar.set_postfix(loss=f"{mean_loss:.4f}", updates=completed_updates)
                status(
                    f"{arm} — epoch {epoch}/{epochs} complete: loss={mean_loss:.4f}, "
                    f"backbone_lr={last_lrs[0]:.3e}, head_lr={last_lrs[1]:.3e}, "
                    f"updates={completed_updates}, seconds={epoch_record['seconds']:.1f}.",
                    enabled=progress,
                )
                if protocol is not None:
                    clean_recalls = epoch_record["validation"]["clean"]["per_class_accuracy"]
                    pgd_recalls = epoch_record["validation"]["pgd"]["per_class_accuracy"]
                    status(
                        "pilot validation — clean cat/dog recall="
                        f"{clean_recalls['0']:.1%}/{clean_recalls['1']:.1%}; "
                        f"PGD cat/dog recall={pgd_recalls['0']:.1%}/{pgd_recalls['1']:.1%}.",
                        enabled=progress,
                    )
        standard_orders = [item["order_sha256"] for item in history]
        order_protocol_hash = sha256_bytes(canonical_json_bytes(standard_orders))
        result = {
            "status": "complete",
            "arm": arm,
            "checkpoint": str(checkpoint_path(config, arm)),
            "epochs": epochs,
            "updates": completed_updates,
            "order_protocol_sha256": order_protocol_hash,
            "resumed_from_epoch": resumed_epoch,
            "elapsed_seconds": time.perf_counter() - started,
        }
        status(
            f"{arm}: training complete at configured epoch {epochs}; checkpoint is registered.",
            enabled=progress,
        )
        return result
    except BaseException as error:
        record_failure(failure, f"train:{arm}", error, {"arm": arm, "device": str(device)})
        status(f"{arm}: training stopped — {error}", enabled=progress)
        raise
