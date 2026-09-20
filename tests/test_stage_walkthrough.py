from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.feature_visualization import _save_figure_record
from src.io_utils import atomic_write_json, sha256_file
from src.stage_walkthrough import (
    CHANNEL_RULE,
    STAGE_LAYERS,
    create_stage_walkthrough,
    rank_stage_channels,
)


class SyntheticStages(nn.Module):
    """Small synthetic graph with the named stages; no dataset or pretrained model."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.ModuleDict(
            {
                "conv1": nn.Conv2d(3, 4, 1, stride=2),
                "bn1": nn.BatchNorm2d(4),
                "relu": nn.ReLU(inplace=True),
                "maxpool": nn.MaxPool2d(2),
                "layer1": self._block(4, 4, 1),
                "layer2": self._block(4, 6, 2),
                "layer3": self._block(6, 8, 2),
                "layer4": self._block(8, 512, 2),
                "avgpool": nn.AdaptiveAvgPool2d(1),
                "fc": nn.Linear(512, 2),
            }
        )
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    module.weight.zero_()
                    assert module.bias is not None
                    module.bias.copy_(torch.arange(module.out_channels))
            self.network["layer1"][0].bias[2:].fill_(5)

    @staticmethod
    def _block(inputs: int, outputs: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 1, stride=stride),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        result = pixels
        for layer in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            result = self.network[layer](result)
        return self.network["fc"](self.network["avgpool"](result).flatten(1))


def _model() -> SyntheticStages:
    with torch.random.fork_rng():
        torch.manual_seed(29)
        model = SyntheticStages()
    model.train()
    model.network["layer1"][1].eval()
    return model


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in model.state_dict().items()}


def _assert_unchanged(model: nn.Module, state: dict[str, torch.Tensor], modes: list[bool]) -> None:
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    assert modes == [module.training for module in model.modules()]
    assert all(not module._forward_hooks for module in model.modules())


def _photo(path: Path) -> None:
    pixels = np.zeros((48, 64, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(64)[None] * 4
    Image.fromarray(pixels).save(path)


def test_channels_are_calibration_ranked_deterministic_and_tie_by_id() -> None:
    model = _model()
    state = _state(model)
    modes = [module.training for module in model.modules()]
    pixels = torch.full((5, 3, 32, 32), 0.5)
    selected = rank_stage_channels(model, pixels, batch_size=2)
    repeated = rank_stage_channels(model, pixels, batch_size=3)
    assert selected == repeated
    assert selected["network.relu"] == [3, 2, 1]
    assert selected["network.layer1"] == [2, 3, 1]
    assert selected["network.layer4"] == [511, 510, 509]
    assert list(selected) == STAGE_LAYERS
    _assert_unchanged(model, state, modes)


def test_channel_accumulation_transfers_to_cpu_before_float64_cast() -> None:
    class MpsLikeResponse:
        def __init__(self) -> None:
            self.cpu_transfers = 0

        def flatten(self, _dimension: int) -> MpsLikeResponse:
            return self

        def amax(self, *, dim: int) -> MpsLikeResponse:
            assert dim == 2
            return self

        def detach(self) -> MpsLikeResponse:
            return self

        def cpu(self) -> torch.Tensor:
            self.cpu_transfers += 1
            return torch.tensor([[1.0, 5.0, 3.0]], dtype=torch.float32)

        def to(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("float64 must not be cast while the tensor is on MPS")

    response = MpsLikeResponse()
    with patch(
        "src.stage_walkthrough.capture_activations",
        return_value=dict.fromkeys(STAGE_LAYERS, response),
    ):
        selected = rank_stage_channels(_model(), torch.zeros(1, 3, 32, 32), 1)
    assert response.cpu_transfers == len(STAGE_LAYERS)
    assert all(channels == [1, 2, 0] for channels in selected.values())


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="native MPS unavailable")
def test_small_float32_mps_ranking_matches_cpu_and_preserves_model_state() -> None:
    cpu_model = _model()
    pixels = torch.linspace(0.1, 0.9, 3 * 3 * 32 * 32).reshape(3, 3, 32, 32)
    expected = rank_stage_channels(cpu_model, pixels, batch_size=2)
    model = copy.deepcopy(cpu_model).to("mps")
    state, modes = _state(model), [module.training for module in model.modules()]
    rng, mps_rng = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    try:
        selected = rank_stage_channels(model, pixels.to("mps"), batch_size=2)
        assert selected == expected
        _assert_unchanged(model, state, modes)
        assert torch.equal(torch.get_rng_state(), rng)
        assert torch.equal(torch.mps.get_rng_state(), mps_rng)
    finally:
        model.cpu()
        torch.mps.empty_cache()


def test_hooks_and_modes_are_restored_even_when_a_stage_fails() -> None:
    model = _model()
    state = _state(model)
    modes = [module.training for module in model.modules()]
    with (
        patch.object(model.network["layer2"], "forward", side_effect=RuntimeError("probe error")),
        pytest.raises(RuntimeError, match="probe error"),
    ):
        rank_stage_channels(model, torch.ones(1, 3, 32, 32), 1)
    _assert_unchanged(model, state, modes)


def test_stage_figure_records_real_shapes_pool_logits_and_png_manifest(tmp_path: Path) -> None:
    model = _model()
    state = _state(model)
    modes = [module.training for module in model.modules()]
    pixels = torch.full((1, 3, 32, 32), 0.5)
    selected = rank_stage_channels(model, pixels, 1)
    original = tmp_path / "synthetic-original.png"
    _photo(original)
    rng = torch.get_rng_state().clone()
    figure, metadata = create_stage_walkthrough(
        model,
        pixels,
        original,
        selected,
        sample_id="synthetic-anchor",
        arm="standard",
        identity="measured-identity",
        config_sha256="config",
        checkpoint_sha256="checkpoint",
        true_label=0,
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert metadata["original_shape_hwc"] == [48, 64, 3]
    assert metadata["original_image_sha256"] == sha256_file(original)
    assert metadata["input_shape_chw"] == [3, 32, 32]
    assert metadata["true_label"] == 0
    assert metadata["input_policy"] == "clean_only"
    assert metadata["arm_label"] == "Standard"
    assert [stage["activation_shape_chw"] for stage in metadata["stages"]] == [
        [4, 16, 16],
        [4, 8, 8],
        [4, 8, 8],
        [6, 4, 4],
        [8, 2, 2],
        [512, 1, 1],
    ]
    assert all(stage["selection_split"] == "calibration" for stage in metadata["stages"])
    assert all(stage["selection_rule"] == CHANNEL_RULE for stage in metadata["stages"])
    assert metadata["pooled_shape"] == [1, 512]
    # Constant channels must remain flat rather than display interpolation
    # rounding as apparent spatial localization.
    for row in range(1, 7):
        assert np.count_nonzero(figure.axes[row * 4 + 3].images[-1].get_array()) == 0
    assert len(metadata["pooled_features"]) == 512
    assert len(metadata["logits"]) == 2
    assert sum(metadata["raw_softmax_probabilities"]) == pytest.approx(1)
    assert "not reconstructions" in metadata["interpretation"]
    assert not metadata["shareable"]
    path = tmp_path / "figures" / "standard-stage-walkthrough-synthetic-anchor.png"
    record = _save_figure_record(
        figure,
        path,
        tmp_path,
        arm="standard",
        caption="Measured synthetic test graph, not a scientific result.",
        kind="stage_walkthrough",
        metadata=metadata,
    )
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert record["sha256"] == sha256_file(path)
    assert record["sample_id"] == "synthetic-anchor"
    assert record["identity"] == "measured-identity"
    assert record["config_sha256"] == "config"
    assert record["checkpoint_sha256"] == "checkpoint"
    assert record["kind"] == "stage_walkthrough" and not record["shareable"]
    manifest = tmp_path / "manifest.json"
    atomic_write_json(manifest, record)
    assert json.loads(manifest.read_text()) == record
    assert not plt.fignum_exists(figure.number)
    _assert_unchanged(model, state, modes)


def test_invalid_selection_is_rejected_without_leaving_hooks(tmp_path: Path) -> None:
    model = _model()
    pixels = torch.full((1, 3, 32, 32), 0.5)
    selected = rank_stage_channels(model, pixels, 1)
    selected["network.layer3"] = [999]
    original = tmp_path / "original.png"
    _photo(original)
    with pytest.raises(ValueError, match="calibration-selected"):
        create_stage_walkthrough(
            model,
            pixels,
            original,
            selected,
            sample_id="test",
            arm="standard",
            identity="identity",
            config_sha256="config",
            checkpoint_sha256="checkpoint",
        )
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("batch_size,count", [(0, 3), (1, 0)])
def test_invalid_channel_ranking_sizes(batch_size: int, count: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        rank_stage_channels(_model(), torch.ones(1, 3, 32, 32), batch_size, count)
