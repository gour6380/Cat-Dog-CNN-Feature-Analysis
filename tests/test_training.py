from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from src.config import ExperimentConfig
from src.model import atomic_torch_save
from src.training import ResumeError, _load_resume, _validate_checkpoint, learning_rate_factor


def test_one_epoch_warmup_then_cosine_to_zero() -> None:
    factors = [learning_rate_factor(index, 2, 6) for index in range(1, 7)]
    assert factors[0] == 0.5
    assert factors[1] == 1.0
    assert factors[-1] == 0.0
    assert factors[2] < factors[1]


def test_schedule_rejects_out_of_range_update() -> None:
    with pytest.raises(ValueError):
        learning_rate_factor(0, 2, 6)


def test_resume_requires_exact_provenance() -> None:
    expected = {"config_sha256": "a"}
    payload = {
        "arm": "standard",
        "provenance": {"config_sha256": "b"},
        "epoch": 1,
        "update_count": 2,
    }
    with pytest.raises(ResumeError, match="provenance"):
        _validate_checkpoint(payload, "standard", expected)


def test_resume_requires_matching_arm() -> None:
    expected = {"config_sha256": "a"}
    payload = {"arm": "adversarial", "provenance": expected, "epoch": 1, "update_count": 2}
    with pytest.raises(ResumeError, match="arm"):
        _validate_checkpoint(payload, "standard", expected)


def test_checkpoint_resume_round_trip(tmp_path: Path) -> None:
    config = ExperimentConfig(
        path=tmp_path / "configs" / "experiment.yaml",
        root=tmp_path,
        raw={"paths": {"checkpoints": "checkpoints"}},
        sha256="config",
    )
    provenance = {"combined_sha256": "registered"}
    original = nn.Linear(3, 2)
    original_optimizer = torch.optim.AdamW(original.parameters(), lr=0.01)
    loss = original(torch.ones(1, 3)).sum()
    loss.backward()
    original_optimizer.step()
    checkpoint = tmp_path / "checkpoints" / "standard" / "epoch-001.pt"
    atomic_torch_save(
        checkpoint,
        {
            "arm": "standard",
            "epoch": 1,
            "update_count": 1,
            "provenance": provenance,
            "model_state": original.state_dict(),
            "optimizer_state": original_optimizer.state_dict(),
            "history": [{"epoch": 1}],
        },
    )
    restored = nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    epoch, updates, history = _load_resume(
        config,
        "standard",
        cast(Any, restored),
        restored_optimizer,
        provenance,
    )
    assert (epoch, updates, history) == (1, 1, [{"epoch": 1}])
    assert all(
        torch.equal(original_value, restored.state_dict()[name])
        for name, original_value in original.state_dict().items()
    )
    assert restored_optimizer.state_dict()["state"]
