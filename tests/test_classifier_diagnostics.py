from __future__ import annotations

import pytest

from src.classifier_diagnostics import species_diagnostic_warning, species_prediction_diagnostic


def test_single_species_prediction_is_not_useful_robust_recognition() -> None:
    diagnostic = species_prediction_diagnostic({"confusion_matrix": [[0, 1183], [0, 2486]]})
    assert diagnostic["prediction_counts"] == {"cat": 0, "dog": 3669}
    assert diagnostic["per_species_recall"] == {"cat": 0.0, "dog": 1.0}
    assert diagnostic["macro_accuracy"] == 0.5
    assert diagnostic["constant_prediction_baseline_accuracy"] == 2486 / 3669
    warning = species_diagnostic_warning(diagnostic)
    assert "only dog" in warning and "not proof" in warning
    assert "useful robust cat/dog" in warning


def test_both_species_predictions_do_not_trigger_single_class_warning() -> None:
    diagnostic = species_prediction_diagnostic({"confusion_matrix": [[1173, 10], [13, 2473]]})
    assert diagnostic["prediction_counts"] == {"cat": 1186, "dog": 2483}
    assert diagnostic["single_predicted_species_on_clean_test"] is None
    assert species_diagnostic_warning(diagnostic) == ""
    assert species_prediction_diagnostic({}) == {"available": False}


@pytest.mark.parametrize(
    "matrix", [[[1]], [[True, 1], [2, 3]], [[1, -1], [2, 3]], [[0, 0], [1, 2]]]
)
def test_invalid_or_missing_species_counts_are_rejected(matrix: list[list[int]]) -> None:
    with pytest.raises(ValueError):
        species_prediction_diagnostic({"confusion_matrix": matrix})
