from __future__ import annotations

import numpy as np

from src.calibration import (
    apply_policy,
    fit_confidence_threshold,
    fit_policy,
    negative_log_likelihood,
    tie_aware_risk_coverage,
)


def test_temperature_fit_does_not_worsen_fitting_nll() -> None:
    logits = np.asarray([[5.0, 0.0], [4.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    policy = fit_policy(logits, labels, 0.75, (0.05, 20.0))
    assert (
        negative_log_likelihood(logits, labels, policy.temperature)
        <= negative_log_likelihood(logits, labels) + 1e-10
    )


def test_threshold_is_tie_aware() -> None:
    confidence = np.asarray([0.9, 0.9, 0.7, 0.2])
    threshold, coverage, ties = fit_confidence_threshold(confidence, 0.25)
    assert threshold == 0.9
    assert coverage == 0.5
    assert ties == 2


def test_risk_coverage_groups_ties() -> None:
    confidence = np.asarray([0.9, 0.9, 0.7, 0.2])
    correct = np.asarray([True, False, True, False])
    coverage, risk, aurc = tie_aware_risk_coverage(confidence, correct)
    assert np.array_equal(coverage, [0.5, 0.75, 1.0])
    assert np.allclose(risk, [0.5, 1 / 3, 0.5])
    assert 0 <= aurc <= 1


def test_policy_metrics_have_registered_fields() -> None:
    logits = np.asarray([[4.0, 0.0], [0.0, 4.0], [2.0, 1.0], [1.0, 2.0]])
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    policy = fit_policy(logits, labels, 0.5, (0.05, 20.0))
    metrics = apply_policy(logits, labels, policy, 15)
    assert {"nll", "brier", "ece_15", "coverage", "selective_risk", "aurc"} <= metrics.keys()
