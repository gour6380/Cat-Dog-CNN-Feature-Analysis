"""Matched, resumable standard and adversarial training."""

from __future__ import annotations

import hashlib
import json
import math
import time
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
from src.progress import status, tqdm
from src.runtime import (
    SafetyStop,
    ensure_finite,
    memory_snapshot,
    record_failure,
    synchronize,
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
    combined = sha256_bytes(
        canonical_json_bytes(
            {
                "config": config.sha256,
                "dataset": dataset_hash,
                "split": split_hash,
                "initialization": initialization_hash,
            }
        )
    )
    return {
        "config_sha256": config.sha256,
        "dataset_sha256": cast(str, dataset_hash),
        "split_sha256": cast(str, split_hash),
        "initialization_sha256": cast(str, initialization_hash),
        "combined_sha256": combined,
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


def train_arm(
    config: ExperimentConfig,
    arm: Arm,
    device: torch.device,
    *,
    progress: bool = True,
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
        if resumed_epoch >= epochs:
            status(
                f"{arm}: reusing the verified epoch-{resumed_epoch} checkpoint.",
                enabled=progress,
            )
            return {
                "status": "complete",
                "arm": arm,
                "checkpoint": str(checkpoint_path(config, arm)),
                "resumed_without_work": True,
                "epochs": resumed_epoch,
                "updates": completed_updates,
            }
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
                sample_count = 0
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
                        if arm == "adversarial":
                            attack_ids = [f"{sample_id}:epoch:{epoch}" for sample_id in sample_ids]
                            pixels = pgd_attack(
                                model, pixels, labels, attack_ids, attack_spec
                            ).adversarial
                        logits = model(pixels)
                        ensure_finite("training logits", logits)
                        loss = nnf.cross_entropy(logits, labels)
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
                epoch_record = {
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
                    },
                )
                epoch_bar.set_postfix(loss=f"{mean_loss:.4f}", updates=completed_updates)
                status(
                    f"{arm} — epoch {epoch}/{epochs} complete: loss={mean_loss:.4f}, "
                    f"backbone_lr={last_lrs[0]:.3e}, head_lr={last_lrs[1]:.3e}, "
                    f"updates={completed_updates}, seconds={epoch_record['seconds']:.1f}.",
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
