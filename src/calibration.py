"""Calibration and tie-aware selective-prediction policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CalibrationPolicy:
    temperature: float
    threshold: float
    target_coverage: float
    fitted_count: int
    fitted_coverage: float
    threshold_tie_count: int


def softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return cast(FloatArray, exponent / exponent.sum(axis=1, keepdims=True))


def negative_log_likelihood(
    logits: FloatArray, labels: IntArray, temperature: float = 1.0
) -> float:
    probabilities = softmax(logits / temperature)
    selected = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(selected, 1e-15, 1.0)).mean())


def fit_temperature(
    logits: FloatArray,
    labels: IntArray,
    bounds: tuple[float, float],
) -> float:
    """Fit one scalar by clean-calibration NLL only."""

    if len(logits) != len(labels) or not len(labels):
        raise ValueError("calibration logits/labels must be non-empty and aligned")
    lower, upper = bounds
    result = minimize_scalar(
        lambda log_temperature: negative_log_likelihood(
            logits, labels, float(np.exp(log_temperature))
        ),
        bounds=(float(np.log(lower)), float(np.log(upper))),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 256},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    return float(np.exp(result.x))


def fit_confidence_threshold(
    confidences: FloatArray, target_coverage: float
) -> tuple[float, float, int]:
    if not 0.0 < target_coverage <= 1.0 or len(confidences) == 0:
        raise ValueError("target coverage and confidences are invalid")
    descending = np.sort(confidences)[::-1]
    rank = max(0, int(np.ceil(target_coverage * len(descending))) - 1)
    threshold = float(descending[rank])
    accepted = confidences >= threshold
    tie_count = int(np.count_nonzero(confidences == threshold))
    return threshold, float(accepted.mean()), tie_count


def fit_policy(
    logits: FloatArray,
    labels: IntArray,
    target_coverage: float,
    temperature_bounds: tuple[float, float],
) -> CalibrationPolicy:
    temperature = fit_temperature(logits, labels, temperature_bounds)
    probabilities = softmax(logits / temperature)
    threshold, coverage, ties = fit_confidence_threshold(probabilities.max(axis=1), target_coverage)
    return CalibrationPolicy(
        temperature=temperature,
        threshold=threshold,
        target_coverage=target_coverage,
        fitted_count=len(labels),
        fitted_coverage=coverage,
        threshold_tie_count=ties,
    )


def tie_aware_risk_coverage(
    confidences: FloatArray, correct: NDArray[np.bool_]
) -> tuple[FloatArray, FloatArray, float]:
    """Group tied scores and integrate the resulting right-step risk curve."""

    if len(confidences) != len(correct) or len(confidences) == 0:
        raise ValueError("confidence/correct arrays must be non-empty and aligned")
    order = np.argsort(-confidences, kind="stable")
    scores = confidences[order]
    errors = (~correct[order]).astype(np.float64)
    coverages: list[float] = []
    risks: list[float] = []
    cumulative_errors = 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        cumulative_errors += float(errors[start:end].sum())
        coverage = end / len(scores)
        coverages.append(coverage)
        risks.append(cumulative_errors / end)
        start = end
    coverage_array = np.asarray(coverages, dtype=np.float64)
    risk_array = np.asarray(risks, dtype=np.float64)
    widths = np.diff(np.concatenate(([0.0], coverage_array)))
    aurc = float(np.sum(widths * risk_array))
    return coverage_array, risk_array, aurc


def apply_policy(
    logits: FloatArray,
    labels: IntArray,
    policy: CalibrationPolicy,
    ece_bins: int,
) -> dict[str, float | int]:
    probabilities = softmax(logits / policy.temperature)
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    accepted = confidence >= policy.threshold
    accepted_count = int(accepted.sum())
    selective_risk = float((~correct[accepted]).mean()) if accepted_count else float("nan")
    _coverage, _risk, aurc = tie_aware_risk_coverage(confidence, correct)
    one_hot = np.eye(logits.shape[1], dtype=np.float64)[labels]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    ece = expected_calibration_error(confidence, correct, ece_bins)
    return {
        "count": len(labels),
        "accuracy": float(correct.mean()),
        "nll": negative_log_likelihood(logits, labels, policy.temperature),
        "brier": brier,
        "ece_15": ece,
        "coverage": float(accepted.mean()),
        "accepted_count": accepted_count,
        "selective_risk": selective_risk,
        "aurc": aurc,
    }


def expected_calibration_error(
    confidence: FloatArray, correct: NDArray[np.bool_], bins: int
) -> float:
    if bins < 1:
        raise ValueError("ECE requires at least one bin")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    value = 0.0
    for index in range(bins):
        if index == 0:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        count = int(mask.sum())
        if count:
            value += (count / total) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(value)
