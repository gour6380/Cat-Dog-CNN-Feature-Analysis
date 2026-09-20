"""Small synthetic section probes; never download data or initialize ResNet."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import nn

from src.config import ExperimentConfig
from src.data import SampleRecord
from src.feature_anchor_sections import compute_anchors
from src.feature_methods import capture_activations
from src.feature_section_types import FeatureSection, SectionData
from src.io_utils import atomic_write_bytes, sha256_file

LAYERS = ["network.relu", "network.layer2", "network.layer4"]


class TinyChannels(nn.Module):
    """Named layers with deliberately mixed modes and existing gradients."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.ModuleDict(
            {
                "conv1": nn.Conv2d(3, 4, 1),
                "bn1": nn.BatchNorm2d(4),
                "relu": nn.ReLU(inplace=True),
                "layer2": nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU()),
                "layer4": nn.Sequential(nn.Conv2d(4, 4, 1), nn.ReLU()),
                "fc": nn.Linear(4, 2),
            }
        )
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    module.weight.fill_(0.1)
                    assert module.bias is not None
                    module.bias.copy_(torch.tensor([0.0, 1.0, 2.0, 100.0]))
            self.network["fc"].weight.fill_(0.01)
            self.network["fc"].bias.copy_(torch.tensor([1.0, -1.0]))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        value = pixels
        for layer in ("conv1", "bn1", "relu", "layer2", "layer4"):
            value = self.network[layer](value)
        return self.network["fc"](value.mean(dim=(2, 3)))


class SyntheticContext:
    def __init__(self, root: Path) -> None:
        self.config = ExperimentConfig(
            root / "config.yaml",
            root,
            {
                "paths": {"checkpoints": "checkpoints"},
                "training": {"epochs": 2, "evaluation_batch_size": 2},
                "input": {"size": 8},
                "attack": {
                    "epsilon": 4 / 255,
                    "step_size": 1 / 255,
                    "evaluation_steps": 2,
                    "evaluation_restarts": 1,
                    "evaluation_seed": 29,
                },
                "feature_visualization": {
                    "seed": 31,
                    "occlusion_patch_size": 4,
                    "occlusion_stride": 4,
                    "top_occlusion_tiles": 2,
                    "random_occlusion_repeats": 5,
                },
            },
            "synthetic-config",
        )
        self.device = torch.device("cpu")
        self.arm = "standard"
        self.identity = "synthetic-method-identity"
        self.progress = False
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(9).get_state())
            self.model = TinyChannels()
        self.model.train()
        self.model.network["bn1"].eval()
        for parameter in self.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        self.anchors = [
            SampleRecord(
                f"anchor-{index}",
                str(root / f"anchor-{index}.png"),
                index % 2,
                index % 2,
                "synthetic",
                "test",
            )
            for index in range(4)
        ]
        self.references = [
            SampleRecord(
                f"calibration-{index}", "unused", index % 2, index % 2, "synthetic", "trainval"
            )
            for index in range(4)
        ]
        self.pixels = torch.linspace(0.1, 0.8, 3 * 8 * 8).reshape(1, 3, 8, 8)
        self.calls: list[str] = []
        self.registered: list[dict[str, Any]] = []
        self.dependencies: dict[FeatureSection, SectionData] = {}

    def anchor_pixels(self, index: int) -> torch.Tensor:
        self.calls.append(f"anchor:{index}")
        return self.pixels + index * 0.01

    def selected_channels(self) -> dict[str, list[int]]:
        self.calls.append("selected_channels")
        # Channel 3 is overwhelmingly stronger but deliberately not selected.
        return dict.fromkeys(LAYERS, [0, 1])

    def stage_channels(self) -> dict[str, list[int]]:
        self.calls.append("stage_channels")
        return {"network.relu": [0, 1]}

    def reference_probe(self, *, initial: bool = False) -> dict[str, dict[str, torch.Tensor]]:
        self.calls.append("initial_probe" if initial else "trained_probe")
        values = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
        return {layer: {"means": values * (1 if initial else 2)} for layer in LAYERS}

    def reference_pixels(self) -> torch.Tensor:
        raise AssertionError("anchor section does not need all reference pixels")

    def initial_kernels(self) -> torch.Tensor:
        raise AssertionError("anchor section does not need kernel weights")

    def dependency(self, section: FeatureSection) -> SectionData:
        self.calls.append(f"dependency:{section}")
        return self.dependencies[section]

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
        path = self.config.root / f"{name}.png"
        figure.savefig(path, dpi=20)
        self.registered.append(
            {
                "path": path,
                "sha256": sha256_file(path),
                "kind": kind,
                "shareable": shareable,
                "caption": caption,
                "metadata": metadata,
                "figure": figure,
            }
        )
        plt.close(figure)


@pytest.fixture
def context(tmp_path: Path) -> Iterator[SyntheticContext]:
    # No learned-feature family may generate PGD or FGSM interpretation inputs.
    with (
        patch("src.attacks.pgd_attack", side_effect=AssertionError("unexpected feature attack")),
        patch("src.attacks.fgsm_attack", side_effect=AssertionError("unexpected feature attack")),
    ):
        yield SyntheticContext(tmp_path)
    plt.close("all")


def _snapshot(
    model: nn.Module,
) -> tuple[dict[str, torch.Tensor], list[bool], list[torch.Tensor | None]]:
    return (
        {name: value.clone() for name, value in model.state_dict().items()},
        [module.training for module in model.modules()],
        [
            parameter.grad.clone() if parameter.grad is not None else None
            for parameter in model.parameters()
        ],
    )


def _unchanged(
    model: nn.Module,
    snapshot: tuple[dict[str, torch.Tensor], list[bool], list[torch.Tensor | None]],
) -> None:
    state, modes, gradients = snapshot
    assert all(torch.equal(value, state[name]) for name, value in model.state_dict().items())
    assert modes == [module.training for module in model.modules()]
    for previous, parameter in zip(gradients, model.parameters(), strict=True):
        if previous is None:
            assert parameter.grad is None
        else:
            assert torch.equal(previous, parameter.grad)
    assert all(not module._forward_hooks for module in model.modules())


def _assert_anchor_contract(data: SectionData, context: SyntheticContext) -> None:
    assert data.arrays["anchor_sample_ids"].tolist() == [
        record.sample_id for record in context.anchors
    ]
    assert len(data.payload["anchors"]) == 4
    for index, row in enumerate(data.payload["anchors"]):
        assert row["sample_id"] == context.anchors[index].sample_id
        assert row["label"] == context.anchors[index].label
        assert row["prediction"] in (0, 1)
        logits = data.arrays[f"anchor_{index}_clean_logits"]
        assert logits.shape == (2,)
        assert row["clean_margin"] == pytest.approx(logits[row["label"]] - logits[1 - row["label"]])
        assert data.arrays[f"anchor_{index}_clean_pixels"].shape == (3, 8, 8)


@pytest.mark.parametrize("section", ["gradcam", "occlusion"])
def test_clean_attribution_is_independent_and_records_aligned_raw_arrays(
    context: SyntheticContext, section: FeatureSection
) -> None:
    snapshot, rng = _snapshot(context.model), torch.get_rng_state().clone()
    data = compute_anchors(section, context)  # type: ignore[arg-type]
    _assert_anchor_contract(data, context)
    assert "selected_channels" not in context.calls
    assert "trained_probe" not in context.calls and "initial_probe" not in context.calls
    assert not any(call.startswith("dependency:") for call in context.calls)
    assert [record["kind"] for record in context.registered] == [section]
    assert not context.registered[0]["shareable"]
    assert context.registered[0]["path"].read_bytes().startswith(b"\x89PNG")
    if section == "gradcam":
        assert data.arrays["anchor_0_gradcam"].shape == (8, 8)
        assert data.arrays["anchor_0_gradcam_native"].shape == (8, 8)
    else:
        assert data.arrays["anchor_0_occlusion_drop"].shape == (2, 2)
        assert data.arrays["anchor_0_occlusion_positions"].shape == (4, 4)
        assert data.arrays["anchor_0_occlusion_positions"].dtype == np.int64
    assert torch.equal(rng, torch.get_rng_state())
    _unchanged(context.model, snapshot)


def test_attribution_targets_true_species_even_when_predictions_are_wrong(
    context: SyntheticContext,
) -> None:
    targets: list[int] = []

    def synthetic_cam(_model: nn.Module, pixels: torch.Tensor, target: int) -> torch.Tensor:
        targets.append(target)
        return pixels[0, 0]

    with patch("src.feature_anchor_sections.grad_cam", side_effect=synthetic_cam):
        data = compute_anchors("gradcam", context)  # type: ignore[arg-type]
    assert targets == [0, 1, 0, 1]
    assert all(row["prediction"] == 0 for row in data.payload["anchors"])


@pytest.mark.parametrize("arm,display", [("standard", "Standard"), ("adversarial", "PGD-trained")])
def test_activation_section_is_clean_only_and_preserves_maps_selection_and_modes(
    context: SyntheticContext,
    arm: str,
    display: str,
) -> None:
    context.arm = arm
    snapshot, rng = _snapshot(context.model), torch.get_rng_state().clone()
    # Clean interpretation no longer depends on evaluation attack settings.
    context.config.raw.pop("attack")
    with (
        patch(
            "src.feature_anchor_sections.grad_cam",
            side_effect=AssertionError("unrelated attribution"),
        ),
        patch(
            "src.feature_anchor_sections.occlusion_map",
            side_effect=AssertionError("unrelated masking"),
        ),
        # The tiny graph is not ResNet; its coordinates do not have ResNet strides.
        patch("src.feature_anchor_sections.receptive_field_box", return_value=(0, 0, 8, 8)),
    ):
        data = compute_anchors("activations", context)  # type: ignore[arg-type]
    _assert_anchor_contract(data, context)
    assert len(context.registered) == 4
    assert all(
        record["kind"] == "activation_walkthrough" and not record["shareable"]
        for record in context.registered
    )
    for index, row in enumerate(data.payload["anchors"]):
        for level, layer in enumerate(LAYERS):
            metadata = row["channel_peaks"][layer]
            assert metadata["channel"] == 1  # Not stronger unselected channel 3.
            clean = data.arrays[f"anchor_{index}_clean_map_{level}"]
            assert clean.shape == (8, 8)
            assert metadata["display_scale_max"] == pytest.approx(clean.max())
            assert data.arrays[f"anchor_{index}_clean_response_overlay_{level}"].shape == (8, 8)
            assert data.arrays[f"anchor_{index}_channel_saliency_{level}"].shape == (8, 8)
        assert "pgd_prediction" not in row and "pgd_margin" not in row
    assert not any("pgd" in name or "perturbation" in name for name in data.arrays)
    assert data.payload["input_policy"] == "clean_only"
    assert data.payload["arm_label"] == display
    titles = [axis.get_title() for axis in context.registered[0]["figure"].axes]
    assert any("Actual clean input" in title for title in titles)
    assert any("Channel response location" in title for title in titles)
    assert any("Input-gradient sensitivity" in title for title in titles)
    assert not any("PGD input" in title or "perturbation" in title for title in titles)
    assert context.registered[0]["figure"]._suptitle.get_text().startswith(display)
    assert torch.equal(rng, torch.get_rng_state())
    _unchanged(context.model, snapshot)


def _dependencies(context: SyntheticContext) -> None:
    context.dependencies["gradcam"] = compute_anchors("gradcam", context)  # type: ignore[arg-type]
    context.dependencies["occlusion"] = compute_anchors("occlusion", context)  # type: ignore[arg-type]
    context.registered.clear()
    context.calls.clear()


def test_constant_responses_cannot_create_false_overlay_localization(
    context: SyntheticContext,
) -> None:
    captures = dict.fromkeys(LAYERS, torch.full((1, 4, 8, 8), 3.0))
    with (
        patch("src.feature_anchor_sections.capture_activations", return_value=captures),
        patch("src.feature_anchor_sections.channel_saliency", return_value=torch.zeros(8, 8)),
        patch("src.feature_anchor_sections.receptive_field_box", return_value=(0, 0, 8, 8)),
    ):
        data = compute_anchors("activations", context)  # type: ignore[arg-type]
    for index in range(len(context.anchors)):
        for level in range(len(LAYERS)):
            assert np.all(data.arrays[f"anchor_{index}_clean_map_{level}"] == 3)
            assert (
                np.count_nonzero(data.arrays[f"anchor_{index}_clean_response_overlay_{level}"]) == 0
            )


def test_diagnostics_uses_dependencies_calibration_and_sequential_randomized_model(
    context: SyntheticContext,
) -> None:
    from src.feature_visualization import _randomized

    _dependencies(context)
    snapshot, rng = _snapshot(context.model), torch.get_rng_state().clone()
    moves: list[str] = []
    original_cpu, original_to = context.model.cpu, context.model.to

    def cpu() -> nn.Module:
        moves.append("trained_cpu")
        return original_cpu()

    def to(device: torch.device) -> nn.Module:
        moves.append("trained_restore")
        return original_to(device)

    def randomized(model: nn.Module, seed: int) -> nn.Module:
        assert moves == ["trained_cpu"]
        assert next(model.parameters()).device.type == "cpu"
        moves.append("randomized_created")
        # Do not deepcopy patched movement methods into the randomized fixture.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            fresh = TinyChannels()
        return _randomized(fresh, seed)

    with (
        patch.object(context.model, "cpu", side_effect=cpu),
        patch.object(context.model, "to", side_effect=to),
        patch("src.feature_visualization._randomized", side_effect=randomized),
        patch("torch.manual_seed", side_effect=AssertionError("global device RNG reseeding")),
    ):
        data = compute_anchors("diagnostics", context)  # type: ignore[arg-type]
    _assert_anchor_contract(data, context)
    assert moves[-1] == "trained_restore"
    assert context.calls.count("dependency:gradcam") == 1
    assert context.calls.count("dependency:occlusion") == 1
    assert context.calls.count("initial_probe") == context.calls.count("trained_probe") == 1
    assert {record["kind"]: record["shareable"] for record in context.registered} == {
        "species_response": True,
        "initial_response_change": True,
        "randomization_control": False,
        "localization_summary": True,
    }
    for row in data.payload["anchors"]:
        assert "top_gradcam_occlusion_mean_drop" in row
        assert "random_occlusion_mean_drop" in row
        assert "randomization_cam_correlation" in row
    for layer in LAYERS:
        assert data.payload["layers"][layer]["relative_change_from_initial"] == pytest.approx(
            [1, 1]
        )
    assert data.arrays["calibration_sample_ids"].tolist() == [
        record.sample_id for record in context.references
    ]
    assert torch.equal(rng, torch.get_rng_state())
    _unchanged(context.model, snapshot)


@pytest.mark.parametrize("mismatch", ["ids", "labels", "pixels", "logits"])
def test_diagnostics_rejects_misaligned_dependencies_without_randomizing(
    context: SyntheticContext, mismatch: str
) -> None:
    _dependencies(context)
    cached = context.dependencies["occlusion"]
    if mismatch == "ids":
        cached.arrays["anchor_sample_ids"] = np.asarray(["wrong"] * 4)
    elif mismatch == "labels":
        cached.payload["anchors"][0]["label"] = 1
    else:
        key = f"anchor_0_clean_{mismatch}"
        cached.arrays[key] = cached.arrays[key] + 1
    with (
        patch(
            "src.feature_visualization._randomized",
            side_effect=AssertionError("must not randomize"),
        ),
        pytest.raises(ValueError, match="aligned|mismatch"),
    ):
        compute_anchors("diagnostics", context)  # type: ignore[arg-type]


def test_randomization_failure_restores_trained_model_modes_state_and_rng(
    context: SyntheticContext,
) -> None:
    _dependencies(context)
    snapshot, rng = _snapshot(context.model), torch.get_rng_state().clone()
    with (
        patch(
            "src.feature_anchor_sections.grad_cam",
            side_effect=RuntimeError("synthetic probe failure"),
        ),
        pytest.raises(RuntimeError, match="synthetic probe failure"),
    ):
        compute_anchors("diagnostics", context)  # type: ignore[arg-type]
    assert next(context.model.parameters()).device.type == "cpu"
    assert torch.equal(rng, torch.get_rng_state())
    _unchanged(context.model, snapshot)
    assert not plt.get_fignums()


def test_stage_section_uses_calibration_channels_original_path_and_checkpoint_identity(
    context: SyntheticContext,
) -> None:
    from src.training import checkpoint_path

    path = checkpoint_path(context.config, "standard")
    atomic_write_bytes(path, b"synthetic checkpoint fixture, never a trained model")

    def stage(
        _model: nn.Module,
        pixels: torch.Tensor,
        original: Path,
        channels: dict[str, list[int]],
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        assert pixels.shape == (1, 3, 8, 8)
        assert original.name == f"{kwargs['sample_id']}.png"
        assert channels == {"network.relu": [0, 1]}
        assert kwargs["checkpoint_sha256"] == sha256_file(path)
        assert kwargs["identity"] == context.identity
        return plt.figure(), {"sample_id": kwargs["sample_id"], "stages": []}

    with (
        patch("src.feature_anchor_sections.create_stage_walkthrough", side_effect=stage) as helper,
    ):
        data = compute_anchors("stages", context)  # type: ignore[arg-type]
    assert helper.call_count == 4
    _assert_anchor_contract(data, context)
    assert len(data.payload["stage_walkthroughs"]) == 4
    assert all(
        record["kind"] == "stage_walkthrough" and not record["shareable"]
        for record in context.registered
    )
    assert context.calls.count("stage_channels") == 1
    assert all(
        row["calibration_sample_ids"] == [record.sample_id for record in context.references]
        for row in data.payload["stage_walkthroughs"]
    )


def test_capture_primitive_keeps_hooks_and_native_shape_independent(
    context: SyntheticContext,
) -> None:
    snapshot = _snapshot(context.model)
    captures = capture_activations(context.model, context.anchor_pixels(0), LAYERS)
    assert all(value.shape == (1, 4, 8, 8) for value in captures.values())
    _unchanged(context.model, snapshot)


def test_unknown_section_does_not_trigger_any_probe(context: SyntheticContext) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        compute_anchors("kernels", context)  # type: ignore[arg-type]
    assert not context.calls and not context.registered


def test_all_feature_families_have_no_attack_calls_or_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    for filename in (
        "feature_anchor_sections.py",
        "feature_pattern_sections.py",
        "feature_sections.py",
        "feature_methods.py",
        "feature_visualization.py",
        "stage_walkthrough.py",
    ):
        tree = ast.parse((root / "src" / filename).read_text())
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module == "src.attacks"
            for node in ast.walk(tree)
        ), filename
        calls = {
            node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute)
        }
        assert not calls.intersection({"pgd_attack", "fgsm_attack"}), filename
