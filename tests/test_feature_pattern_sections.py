"""Tiny registered contexts; no dataset, full model, training or attacks."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import nn

from src import feature_visualization as fv
from src.config import ExperimentConfig
from src.data import SampleRecord
from src.feature_methods import receptive_field_box
from src.feature_pattern_sections import compute_patterns
from src.feature_section_types import FeatureContext, FeatureSection, SectionData
from src.io_utils import atomic_write_json


class TinyContext:
    device = torch.device("cpu")
    arm = "standard"
    identity = "registered-synthetic-test-identity"
    progress = False
    anchors: list[SampleRecord] = []

    def __init__(self, root: Path) -> None:
        self.config = ExperimentConfig(
            root / "configs/experiment.yaml",
            root,
            {
                "paths": {"artifacts": "artifacts"},
                "input": {"size": 32},
                "feature_visualization": {"seed": 103, "channels_per_layer": 6},
            },
            "synthetic-config",
        )
        with torch.random.fork_rng():
            torch.set_rng_state(torch.Generator().manual_seed(11).get_state())
            convolution = nn.Conv2d(3, 6, 7, padding=3)
            normalization = nn.BatchNorm2d(6)
        self.model = nn.Module()
        network = nn.Module()
        network.add_module("conv1", convolution)
        network.add_module("bn1", normalization)
        self.model.add_module("network", network)
        self.model.train()
        normalization.eval()  # Deliberately preserve mixed modes.
        for parameter in self.model.parameters():
            parameter.grad = torch.full_like(parameter, 0.25)
        self._initial = convolution.weight.detach().clone() - 0.125
        self.references = [
            SampleRecord(f"cal-{name}", f"/{name}.jpg", species, species, name, "trainval", species)
            for name, species in (("cat-a", 0), ("cat-b", 0), ("dog-a", 1), ("dog-b", 1))
        ]
        self._pixels = torch.stack(
            [torch.full((3, 32, 32), 0.1 * (index + 1)) for index in range(4)]
        )
        self._channels = {layer: list(range(6)) for layer in fv.LAYERS}
        peaks = torch.tensor(
            [[6, 4, 1, 9, 2, 9], [6, 2, 8, 5, 1, 8], [3, 9, 6, 3, 9, 7], [2, 8, 9, 1, 6, 6]],
            dtype=torch.float32,
        )
        self._probe = {
            layer: {
                "peaks": peaks.clone(),
                "means": peaks.clone() / 2,
                "positions": torch.tensor(
                    [
                        [[index % side, (index + channel) % side] for channel in range(6)]
                        for index in range(4)
                    ]
                ),
            }
            for layer, side in zip(fv.LAYERS, (16, 4, 1), strict=True)
        }
        self.events: list[str] = []
        self.figures: list[dict[str, Any]] = []

    def selected_channels(self) -> dict[str, list[int]]:
        self.events.append("selected")
        return self._channels

    def reference_pixels(self) -> torch.Tensor:
        self.events.append("reference-pixels")
        return self._pixels

    def reference_probe(self, *, initial: bool = False) -> dict[str, dict[str, torch.Tensor]]:
        if initial:
            pytest.fail("pattern sections must not run an initial-encoder baseline")
        self.events.append("current-probe")
        return self._probe

    def initial_kernels(self) -> torch.Tensor:
        self.events.append("initial-weights-only")
        return self._initial

    def stage_channels(self) -> dict[str, list[int]]:
        pytest.fail("pattern section must not compute major-stage rankings")

    def anchor_pixels(self, index: int) -> torch.Tensor:
        pytest.fail("pattern section must not load test anchors")

    def dependency(self, section: FeatureSection) -> SectionData:
        pytest.fail(f"unexpected pattern dependency: {section}")

    def register(
        self,
        figure: Any,
        name: str,
        caption: str,
        kind: str,
        *,
        shareable: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.figures.append(
            {
                "figure": figure,
                "name": name,
                "caption": caption,
                "kind": kind,
                "shareable": shareable,
                "metadata": metadata,
            }
        )


@pytest.fixture
def ctx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TinyContext]:
    context = TinyContext(tmp_path)

    def unrelated(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("unrelated feature computation was invoked")

    for name in ("pgd_attack", "_randomized", "grad_cam", "occlusion_map", "channel_saliency"):
        monkeypatch.setattr(fv, name, unrelated, raising=False)
    yield context
    for record in context.figures:
        plt.close(record["figure"])


def _snapshot(ctx: TinyContext) -> dict[str, Any]:
    return {
        "state": {key: value.clone() for key, value in ctx.model.state_dict().items()},
        "modes": [module.training for module in ctx.model.modules()],
        "flags": [parameter.requires_grad for parameter in ctx.model.parameters()],
        "gradients": [
            cast(torch.Tensor, parameter.grad).clone() for parameter in ctx.model.parameters()
        ],
        "torch_rng": torch.get_rng_state().clone(),
        "numpy_rng": np.random.get_state(),
    }


def _unchanged(ctx: TinyContext, before: dict[str, Any]) -> None:
    assert before["modes"] == [module.training for module in ctx.model.modules()]
    assert before["flags"] == [parameter.requires_grad for parameter in ctx.model.parameters()]
    assert all(
        torch.equal(value, before["state"][key]) for key, value in ctx.model.state_dict().items()
    )
    assert all(
        torch.equal(cast(torch.Tensor, parameter.grad), original)
        for parameter, original in zip(ctx.model.parameters(), before["gradients"], strict=True)
    )
    assert torch.equal(torch.get_rng_state(), before["torch_rng"])
    current = np.random.get_state()
    assert current[0] == before["numpy_rng"][0]
    assert np.array_equal(current[1], before["numpy_rng"][1])
    assert current[2:] == before["numpy_rng"][2:]


def test_kernel_section_reads_weights_only_and_matches_raw_display_arrays(
    ctx: TinyContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fv, "_pattern", lambda *args, **kwargs: pytest.fail("synthetic work"))
    before = _snapshot(ctx)
    result = compute_patterns("kernels", cast(FeatureContext, ctx))
    assert ctx.events == ["selected", "initial-weights-only"]
    assert len(ctx.figures) == 1 and ctx.figures[0]["shareable"] is True
    assert ctx.figures[0]["kind"] == "kernels"
    np.testing.assert_array_equal(result.arrays["initial_kernels"], ctx._initial.numpy())
    current = ctx.model.get_submodule("network.conv1").weight.detach().numpy()
    np.testing.assert_array_equal(result.arrays["current_kernels"], current)
    np.testing.assert_array_equal(
        result.arrays["kernel_differences"], current - ctx._initial.numpy()
    )
    images = [axis.images[0].get_array() for axis in ctx.figures[0]["figure"].axes]
    for row, field in enumerate(("initial_kernels", "current_kernels", "kernel_differences")):
        for channel in range(6):
            expected = fv.normalize_map(result.arrays[field][channel].transpose(1, 2, 0))
            np.testing.assert_array_equal(images[row * 6 + channel], expected)
    assert result.payload["layers"]["network.layer4"]["spatial_shape"] == [1, 1]
    _unchanged(ctx, before)


def _stub_patterns(ctx: TinyContext, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, int]]:
    calls: list[tuple[str, int, int]] = []

    def pattern(
        config: Any,
        model: Any,
        arm: str,
        layer: str,
        channel: int,
        device: torch.device,
        identity: str,
        seed: int,
    ) -> torch.Tensor:
        assert model is ctx.model and identity == ctx.identity and device.type == "cpu"
        calls.append((layer, channel, seed))
        failed = channel == 3
        initial, final = (0.0, 0.0) if failed else (channel / 10.0, 1.0 + channel / 10.0)
        path = (
            ctx.config.project_path("artifacts")
            / "features/synthetic"
            / f"{arm}-{layer.split('.')[-1]}-{channel}.json"
        )
        atomic_write_json(
            path,
            {
                "identity": identity,
                "layer": layer,
                "channel": channel,
                "seed": seed,
                "initial_unregularized_mean_response": initial,
                "final_unregularized_mean_response": final,
                "unregularized_response_gain": final - initial,
                "response_measurement": "synthetic bounded test probe",
                "sha256": "mocked-file-hash",
            },
        )
        return torch.full((3, 32, 32), 0.5 if failed else (channel + 1) / 10.0)

    monkeypatch.setattr(fv, "_pattern", pattern)
    return calls


def test_synthetic_section_preserves_18_seeds_measured_responses_and_failed_trials(
    ctx: TinyContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_patterns(ctx, monkeypatch)
    before = _snapshot(ctx)
    result = compute_patterns("synthetic", cast(FeatureContext, ctx))
    expected = [
        (layer, channel, 103 + level * 10000 + channel)
        for level, layer in enumerate(fv.LAYERS)
        for channel in range(6)
    ]
    assert calls == expected and ctx.events == ["selected"]
    assert len(ctx.figures) == 1 and ctx.figures[0]["shareable"] is True
    assert ctx.figures[0]["kind"] == "activation_maximization"
    np.testing.assert_array_equal(result.arrays["synthetic_seeds"], [item[2] for item in expected])
    for index, (layer, channel, seed) in enumerate(expected):
        response = result.payload["layers"][layer]["synthetic_responses"][channel]
        assert response["seed"] == seed
        gain = result.arrays["unregularized_response_gains"][index]
        assert gain == response["unregularized_response_gain"]
        axis = ctx.figures[0]["figure"].axes[index]
        np.testing.assert_array_equal(
            axis.images[0].get_array(), result.arrays["synthetic_pixels"][index].transpose(1, 2, 0)
        )
        if channel == 3:
            assert gain == 0.0 and np.all(result.arrays["synthetic_pixels"][index] == 0.5)
            assert (
                "zero-response trial" in axis.get_title()
                and "not a dead channel" in axis.get_title()
            )
        else:
            assert gain == pytest.approx(1.0)
    _unchanged(ctx, before)


def test_real_patches_keep_top_two_stable_rankings_exact_pixels_and_theoretical_boxes(
    ctx: TinyContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fv, "_pattern", lambda *args, **kwargs: pytest.fail("synthetic work"))
    before = _snapshot(ctx)
    result = compute_patterns("real_patches", cast(FeatureContext, ctx))
    assert ctx.events == ["selected", "current-probe", "reference-pixels"]
    assert len(ctx.figures) == 3 and all(record["shareable"] is False for record in ctx.figures)
    assert len(result.arrays["patch_channels"]) == 36
    for level, layer in enumerate(fv.LAYERS):
        patches = result.payload["layers"][layer]["real_patches"]
        for channel in range(6):
            expected = np.argsort(-ctx._probe[layer]["peaks"][:, channel].numpy(), kind="stable")[
                :2
            ]
            for rank, reference_index in enumerate(expected):
                flat_index = level * 12 + channel * 2 + rank
                patch = patches[channel * 2 + rank]
                assert patch["sample_id"] == ctx.references[reference_index].sample_id
                assert patch["peak_response"] == float(
                    ctx._probe[layer]["peaks"][reference_index, channel]
                )
                y, x = ctx._probe[layer]["positions"][reference_index, channel].tolist()
                box = receptive_field_box(layer, y, x, 32)
                assert patch["receptive_field_box"] == list(box)
                np.testing.assert_array_equal(result.arrays["patch_rf_boxes"][flat_index], box)
                image_index = result.arrays["patch_image_indices"][flat_index]
                np.testing.assert_array_equal(
                    result.arrays["reference_pixels"][image_index],
                    ctx._pixels[reference_index].numpy(),
                )
                axis = ctx.figures[level]["figure"].axes[channel * 4 + rank * 2]
                np.testing.assert_array_equal(
                    axis.images[0].get_array(),
                    ctx._pixels[reference_index].numpy().transpose(1, 2, 0),
                )
                rectangle = axis.patches[0]
                assert rectangle.get_xy() == (box[0], box[1])
                assert rectangle.get_width() == box[2] - box[0]
                assert rectangle.get_height() == box[3] - box[1]
    # Equal peaks retain the registered source order; never select for appearance.
    assert result.arrays["patch_sample_ids"][:2].tolist() == ["cal-cat-a", "cal-cat-b"]
    _unchanged(ctx, before)


def test_pattern_dispatch_rejects_other_families_without_computation(ctx: TinyContext) -> None:
    with pytest.raises(ValueError, match="not a pattern feature section"):
        compute_patterns("gradcam", cast(FeatureContext, ctx))
    assert not ctx.events and not ctx.figures


@pytest.mark.parametrize("damage", ["gain", "nonfinite_pixels", "unbounded_pixels"])
def test_invalid_synthetic_evidence_stops_without_retry_and_closes_only_new_panel(
    ctx: TinyContext, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    calls = _stub_patterns(ctx, monkeypatch)
    original = fv._pattern
    earlier = plt.figure()
    before_figures = set(plt.get_fignums())
    before = _snapshot(ctx)

    def damaged(*args: Any, **kwargs: Any) -> torch.Tensor:
        pixels = original(*args, **kwargs)
        if damage == "gain":
            path = ctx.config.project_path("artifacts") / "features/synthetic/standard-relu-0.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["unregularized_response_gain"] = 500.0
            atomic_write_json(path, record)
        else:
            pixels.fill_(float("nan") if damage == "nonfinite_pixels" else 1.1)
        return pixels

    monkeypatch.setattr(fv, "_pattern", damaged)
    try:
        with pytest.raises(ValueError, match="inconsistent|finite bounded"):
            compute_patterns("synthetic", cast(FeatureContext, ctx))
        assert calls == [("network.relu", 0, 103)]
        assert not ctx.figures and set(plt.get_fignums()) == before_figures
        _unchanged(ctx, before)
    finally:
        plt.close(earlier)
