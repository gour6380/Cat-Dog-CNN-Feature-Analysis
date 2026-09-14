from __future__ import annotations

import numpy as np

from src.metrics import (
    cosine_feature_drift,
    knn_metrics,
    linear_cka,
    nearest_centroid_metrics,
    relative_l2_drift,
    scatter_metrics,
    stratified_primary_bootstrap,
)


def test_zero_feature_drift_for_identical_pairs() -> None:
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(cosine_feature_drift(features, features), 0)
    assert np.allclose(relative_l2_drift(features, features), 0)


def test_linear_cka_is_orthogonal_and_scale_invariant() -> None:
    generator = np.random.default_rng(4)
    features = generator.normal(size=(30, 6))
    orthogonal, _ = np.linalg.qr(generator.normal(size=(6, 6)))
    assert np.isclose(linear_cka(features, 3 * features @ orthogonal), 1.0)


def test_knn_and_centroid_on_separated_classes() -> None:
    reference = np.asarray([[1.0, 0.0], [0.9, 0.1], [1.0, 0.1], [0.0, 1.0], [0.1, 0.9], [0.1, 1.0]])
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    query = np.asarray([[1.0, 0.05], [0.05, 1.0]])
    query_labels = np.asarray([0, 1], dtype=np.int64)
    knn = knn_metrics(reference, labels, query, query_labels, 3)
    centroid = nearest_centroid_metrics(reference, labels, query, query_labels, 2)
    assert knn.accuracy == 1
    assert knn.retention == 1
    assert centroid["accuracy"] == 1


def test_scatter_reports_positive_between_structure() -> None:
    features = np.asarray([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.1, 2.0]])
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    result = scatter_metrics(features, labels, 2)
    assert result["between_trace"] > result["within_trace"]


def test_primary_bootstrap_requires_both_bounds() -> None:
    labels = np.repeat(np.arange(4), 20).astype(np.int64)
    standard_drift = np.full(80, 0.4)
    adversarial_drift = np.full(80, 0.1)
    standard_retention = np.full(80, 0.2)
    adversarial_retention = np.full(80, 0.8)
    result = stratified_primary_bootstrap(
        labels,
        standard_drift,
        adversarial_drift,
        standard_retention,
        adversarial_retention,
        200,
        9,
    )
    assert result["primary_supported"] is True
    assert result["drift_ci95"][1] < 0
    assert result["retention_ci95"][0] > 0


def test_primary_bootstrap_rejects_partial_win() -> None:
    labels = np.repeat(np.arange(2), 20).astype(np.int64)
    result = stratified_primary_bootstrap(
        labels,
        np.full(40, 0.4),
        np.full(40, 0.1),
        np.full(40, 0.8),
        np.full(40, 0.8),
        100,
        3,
    )
    assert result["primary_supported"] is False
