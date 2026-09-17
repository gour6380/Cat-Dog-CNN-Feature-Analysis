"""Separate species recognition from a single-class prediction failure."""

from __future__ import annotations

from typing import Any

NAMES = ("cat", "dog")


def species_prediction_diagnostic(metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize observed clean-test predictions, not behavior on every possible input."""
    matrix = metrics.get("confusion_matrix")
    if matrix is None:
        return {"available": False}
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
        or any(type(value) is not int or value < 0 for row in matrix for value in row)
    ):
        raise ValueError("species confusion matrix must contain 2×2 nonnegative integer counts")
    truth = [sum(row) for row in matrix]
    predicted = [sum(row[column] for row in matrix) for column in range(2)]
    total = sum(truth)
    if not total or not all(truth):
        raise ValueError("species diagnostic requires both species and a nonempty test partition")
    active = [column for column, count in enumerate(predicted) if count]
    single = NAMES[active[0]] if len(active) == 1 else None
    return {
        "available": True,
        "sample_count": total,
        "true_counts": dict(zip(NAMES, truth, strict=True)),
        "prediction_counts": dict(zip(NAMES, predicted, strict=True)),
        "single_predicted_species_on_clean_test": single,
        "observed_single_class_prediction": single is not None,
        "per_species_recall": {NAMES[i]: matrix[i][i] / truth[i] for i in range(2)},
        "macro_accuracy": sum(matrix[i][i] / truth[i] for i in range(2)) / 2,
        "constant_prediction_baseline_accuracy": truth[active[0]] / total if single else None,
        "scope": "Observed clean test only; does not imply constant penultimate features "
        "or constant predictions for every possible input.",
    }


def species_diagnostic_warning(diagnostic: dict[str, Any]) -> str:
    species = diagnostic.get("single_predicted_species_on_clean_test")
    if species is None:
        return ""
    return (
        f"WARNING: this arm predicts only {species} on all {diagnostic['sample_count']} clean test "
        "images (macro accuracy 50%). Nominal attack survival must not be presented as useful "
        "robust cat/dog recognition. This is a classifier prediction failure, not proof that "
        "all hidden features are constant."
    )
