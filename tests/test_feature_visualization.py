from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import nn

from src.config import ExperimentConfig
from src.feature_visualization import (
    _context_figures,
    _context_sources,
    _pattern,
    _randomized,
    _single_image_pixels,
    _workflow_figure,
    localization_comparison,
    map_correlation,
    normalize_map,
    visualize_features,
)
from src.io_utils import atomic_write_json


def test_single_image_probe_materializes_contiguous_nchw_without_changing_pixels() -> None:
    hwc = torch.rand(8, 8, 3)
    image = hwc.permute(2, 0, 1)
    assert not image.is_contiguous()
    pixels = _single_image_pixels(image, torch.device("cpu"))
    assert pixels.shape == (1, 3, 8, 8) and pixels.is_contiguous()
    assert torch.equal(pixels[0], image)
    with pytest.raises(ValueError, match="3×H×W"):
        _single_image_pixels(hwc, torch.device("cpu"))


def test_display_normalization_is_only_scaling_and_handles_constants() -> None:
    values = np.array([[-2.0, 0.0], [2.0, 6.0]])
    original = values.copy()
    assert np.array_equal(normalize_map(values), np.array([[0, 0.25], [0.5, 1]]))
    assert np.array_equal(values, original)
    assert np.array_equal(normalize_map(np.full((2, 2), 3.0)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="nonfinite"):
        normalize_map(np.array([[np.nan]]))
    with pytest.raises(ValueError, match="empty"):
        normalize_map(np.empty((0, 0)))


def _tiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = np.array([[1, 1, 4, 4], [1, 1, 4, 4], [3, 3, 2, 2], [3, 3, 2, 2]])
    boxes = np.array([[0, 0, 2, 2], [2, 0, 4, 2], [0, 2, 2, 4], [2, 2, 4, 4]])
    drops = np.array([[1.0, 40.0], [30.0, 2.0]])
    return cam, boxes, drops


def test_localization_ranks_aligned_tiles_and_uses_seeded_local_random_controls() -> None:
    cam, boxes, drops = _tiles()
    state = np.random.get_state()
    result = localization_comparison(cam, boxes, drops, top_count=2, repeats=10, seed=9)
    repeated = localization_comparison(cam, boxes, drops, top_count=2, repeats=10, seed=9)
    generator = np.random.default_rng(9)
    expected = np.mean(
        [drops.ravel()[generator.choice(4, 2, replace=False)].mean() for _repeat in range(10)]
    )
    assert result["top_gradcam_occlusion_mean_drop"] == 35
    assert result["random_occlusion_mean_drop"] == expected
    assert result == repeated
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_localization_rejects_area_alignment_and_spatial_errors() -> None:
    cam, boxes, drops = _tiles()
    with pytest.raises(ValueError, match="aligned equal-area"):
        localization_comparison(cam, boxes[:-1], drops, 1, 10, 2)
    unequal = boxes.copy()
    unequal[0, 2] = 1
    with pytest.raises(ValueError, match="equal-area"):
        localization_comparison(cam, unequal, drops, 1, 10, 2)
    outside = boxes.copy()
    outside[0, 0] = -1
    with pytest.raises(ValueError, match="within"):
        localization_comparison(cam, outside, drops, 1, 10, 2)
    with pytest.raises(ValueError, match="integer"):
        localization_comparison(cam, boxes.astype(float), drops, 1, 10, 2)
    with pytest.raises(ValueError, match="finite"):
        localization_comparison(cam, boxes, np.full((2, 2), np.nan), 1, 10, 2)
    with pytest.raises(ValueError, match="count"):
        localization_comparison(cam, boxes, drops, 5, 10, 2)


def test_correlation_has_no_value_for_constant_maps_and_checks_alignment() -> None:
    first = np.arange(9).reshape(3, 3)
    assert map_correlation(first, first * 2 + 1) == pytest.approx(1.0)
    assert map_correlation(first, -first) == pytest.approx(-1.0)
    assert map_correlation(first, np.ones((3, 3))) is None
    assert map_correlation(np.zeros((3, 3)), first) is None
    with pytest.raises(ValueError, match="alignment"):
        map_correlation(first, first[:2])
    with pytest.raises(ValueError, match="alignment"):
        map_correlation(first, first.reshape(1, 9))
    with pytest.raises(ValueError, match="finite"):
        map_correlation(first, np.full((3, 3), np.nan))


class TinyWrappedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 4, 3),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def test_randomized_control_is_repeatable_and_does_not_change_model_or_device_rngs() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(102)
        model = TinyWrappedModel()
    model.train()
    model.network[1].eval()
    with torch.no_grad():
        model.network[1].running_mean.fill_(2)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state().clone()
    # A call to manual_seed would also reseed MPS/CUDA; private CPU seeding must
    # avoid that API even when the test machine has no accelerator available.
    with patch("torch.manual_seed", side_effect=AssertionError("global device reseeding")):
        randomized = _randomized(model, seed=21)
        repeat = _randomized(model, seed=21)
        other = _randomized(model, seed=22)
    assert torch.equal(torch.get_rng_state(), rng)
    assert modes == [module.training for module in model.modules()]
    assert all(torch.equal(state[name], value) for name, value in model.state_dict().items())
    assert all(not module.training for module in randomized.modules())
    assert not torch.equal(state["network.0.weight"], randomized.network[0].weight)
    assert torch.equal(randomized.network[0].weight, repeat.network[0].weight)
    assert not torch.equal(randomized.network[0].weight, other.network[0].weight)
    assert torch.equal(randomized.network[1].running_mean, torch.zeros(4))


def _config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        root / "configs" / "experiment.yaml",
        root,
        {
            "paths": {"artifacts": "artifacts", "results": "results"},
            "dataset": {"expected_test": 3669, "attack_per_class": 100},
            "model": {"classes": 2},
            "attack": {"evaluation_steps": 20, "evaluation_restarts": 5, "train_steps": 5},
        },
        "current",
    )


def test_outer_wrapper_records_failures_even_before_baseline_or_eda(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with (
        patch(
            "src.feature_visualization._visualize_features",
            side_effect=RuntimeError("baseline failure"),
        ),
        pytest.raises(RuntimeError, match="baseline failure"),
    ):
        visualize_features(config, torch.device("cpu"), progress=False)
    failure = json.loads((tmp_path / "artifacts/failures/features.json").read_text())
    assert failure["status"] == "failed" and failure["stage"] == "features"
    assert failure["context"] == {"config_sha256": "current", "device": "cpu"}


def test_synthetic_manifest_records_achieved_responses_and_resumes_without_optimization(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.raw["input"] = {"size": 8}
    config.raw["feature_visualization"] = {
        "activation_maximization_steps": 0,
        "activation_maximization_learning_rate": 0.05,
        "total_variation_weight": 0.0005,
        "pixel_l2_weight": 0.001,
    }
    model = TinyWrappedModel().eval()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    arguments = (config, model, "standard", "network.2", 0, torch.device("cpu"), "identity", 10)
    pixels = _pattern(*arguments)
    manifest = json.loads((tmp_path / "artifacts/features/synthetic/standard-2-0.json").read_text())
    assert manifest["initial_unregularized_mean_response"] == pytest.approx(
        manifest["final_unregularized_mean_response"]
    )
    assert manifest["unregularized_response_gain"] == pytest.approx(0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(state[name], value) for name, value in model.state_dict().items())
    with patch(
        "src.feature_visualization.activation_maximize", side_effect=AssertionError("rerun")
    ):
        assert torch.equal(_pattern(*arguments), pixels)


def test_wrapper_synchronizes_and_preserves_original_cached_measurement(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with (
        patch("src.feature_visualization._visualize_features", return_value={"identity": "one"}),
        patch("src.feature_visualization.synchronize") as synchronize,
        patch("src.feature_visualization.time.perf_counter", side_effect=[10.0, 13.0]),
        patch("src.feature_visualization.environment_snapshot", return_value={"python": "3.13.15"}),
    ):
        result = visualize_features(config, torch.device("cpu"), progress=False)
    assert synchronize.call_count == 2
    assert result["elapsed_seconds"] == 3.0 and result["environment"]["python"] == "3.13.15"
    saved = json.loads((tmp_path / "results/feature_visualizations.json").read_text())
    assert saved == result
    with patch("src.feature_visualization._visualize_features", return_value=result):
        cached = visualize_features(config, torch.device("cpu"), progress=False)
    assert cached["elapsed_seconds"] == 3.0


def test_optional_context_uses_only_matching_complete_saved_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    for arm in ("standard", "adversarial"):
        atomic_write_json(
            tmp_path / "artifacts/training" / f"{arm}.json",
            {
                "status": "complete",
                "provenance": {"config_sha256": "current"},
                "history": [{"epoch": 1, "mean_training_loss": 0.4}],
            },
        )
    atomic_write_json(tmp_path / "results/evaluation.json", {"config_sha256": "old"})
    sources = _context_sources(config)
    assert len(sources) == 2 and "results/evaluation.json" not in sources
    figures = _context_figures(config, sources)
    assert len(figures) == 1 and figures[0][1] == "training-objectives"
    assert "clean augmented CE" in figures[0][0].axes[0].get_title()
    assert "PGD-5 input CE" in figures[0][0].axes[1].get_title()
    plt.close(figures[0][0])
    arms = {
        arm: {
            "full_test": {"clean": {"accuracy": 0.9}},
            "attack_subset": {
                name: {"clean_accuracy_on_subset": 0.8, "robust_accuracy": 0.4}
                for name in ("fgsm", "pgd")
            },
        }
        for arm in ("standard", "adversarial")
    }
    atomic_write_json(
        tmp_path / "results/evaluation.json", {"config_sha256": "current", "arms": arms}
    )
    figures = _context_figures(config, _context_sources(config))
    assert len(figures) == 2 and figures[0][1] == "accuracy-context"
    assert "n=3669" in figures[0][0].axes[0].get_title()
    assert "n=200" in figures[0][0].axes[1].get_title()
    for figure, _name, _caption in figures:
        plt.close(figure)


def test_workflow_has_actual_layer_dimensions_and_distinct_probe_questions() -> None:
    figure = _workflow_figure(224)
    texts = " ".join(text.get_text() for text in figure.axes[0].texts)
    assert "64×112×112" in texts and "128×28×28" in texts and "512×7×7" in texts
    assert "RF 435 px" in texts and "512-D vector" in texts and "2 logits" in texts
    assert "CHANNEL PREFERENCES" in texts and "SPATIAL RESPONSES" in texts
    assert "CLASS ATTRIBUTION" in texts and "pool + layer1" in texts and "layer3" in texts
    plt.close(figure)
