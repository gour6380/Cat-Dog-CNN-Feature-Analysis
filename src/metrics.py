"""Classification and original-feature representation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def accuracy_metrics(logits: FloatArray, labels: IntArray, classes: int) -> dict[str, Any]:
    predictions = logits.argmax(axis=1)
    per_class: dict[str, float] = {}
    for label in range(classes):
        mask = labels == label
        per_class[str(label)] = (
            float((predictions[mask] == label).mean()) if mask.any() else float("nan")
        )
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_accuracy": float(np.nanmean(list(per_class.values()))),
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=np.arange(classes)
        ).tolist(),
    }


def cosine_feature_drift(clean: FloatArray, shifted: FloatArray) -> FloatArray:
    _validate_pair(clean, shifted)
    numerator = np.sum(clean * shifted, axis=1)
    denominator = np.linalg.norm(clean, axis=1) * np.linalg.norm(shifted, axis=1)
    return cast(FloatArray, 1.0 - numerator / np.clip(denominator, 1e-15, None))


def relative_l2_drift(clean: FloatArray, shifted: FloatArray) -> FloatArray:
    _validate_pair(clean, shifted)
    return cast(
        FloatArray,
        np.linalg.norm(shifted - clean, axis=1)
        / np.clip(np.linalg.norm(clean, axis=1), 1e-15, None),
    )


def _validate_pair(first: FloatArray, second: FloatArray) -> None:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired feature matrices must have identical two-dimensional shape")


@dataclass(frozen=True)
class KnnResult:
    accuracy: float
    retention: float
    per_sample_retention: FloatArray
    predictions: IntArray


def knn_metrics(
    reference_features: FloatArray,
    reference_labels: IntArray,
    query_features: FloatArray,
    query_labels: IntArray,
    neighbours: int = 5,
) -> KnnResult:
    if len(reference_features) != len(reference_labels):
        raise ValueError("reference features and labels are not aligned")
    search = NearestNeighbors(n_neighbors=neighbours, metric="cosine")
    search.fit(reference_features)
    indices = search.kneighbors(query_features, return_distance=False)
    neighbour_labels = reference_labels[indices]
    retention = (neighbour_labels == query_labels[:, None]).mean(axis=1)
    classes = int(max(reference_labels.max(), query_labels.max())) + 1
    votes = np.zeros((len(query_labels), classes), dtype=np.int64)
    rows = np.arange(len(query_labels))
    for column in range(neighbours):
        np.add.at(votes, (rows, neighbour_labels[:, column]), 1)
    predictions = votes.argmax(axis=1).astype(np.int64)
    return KnnResult(
        accuracy=float((predictions == query_labels).mean()),
        retention=float(retention.mean()),
        per_sample_retention=retention.astype(np.float64),
        predictions=predictions,
    )


def class_centroids(features: FloatArray, labels: IntArray, classes: int) -> FloatArray:
    centroids = []
    for label in range(classes):
        selected = features[labels == label]
        if not len(selected):
            raise ValueError(f"class {label} has no reference features")
        centroids.append(selected.mean(axis=0))
    return np.stack(centroids)


def nearest_centroid_metrics(
    reference_features: FloatArray,
    reference_labels: IntArray,
    query_features: FloatArray,
    query_labels: IntArray,
    classes: int,
) -> dict[str, Any]:
    centroids = class_centroids(reference_features, reference_labels, classes)
    query_norm = query_features / np.clip(
        np.linalg.norm(query_features, axis=1, keepdims=True), 1e-15, None
    )
    centroid_norm = centroids / np.clip(
        np.linalg.norm(centroids, axis=1, keepdims=True), 1e-15, None
    )
    predictions = (query_norm @ centroid_norm.T).argmax(axis=1)
    return {
        "accuracy": float((predictions == query_labels).mean()),
        "predictions": predictions.astype(np.int64),
        "centroids": centroids,
    }


def scatter_metrics(features: FloatArray, labels: IntArray, classes: int) -> dict[str, float]:
    overall = features.mean(axis=0)
    within = 0.0
    between = 0.0
    for label in range(classes):
        selected = features[labels == label]
        if not len(selected):
            continue
        centroid = selected.mean(axis=0)
        within += float(np.square(selected - centroid).sum())
        between += len(selected) * float(np.square(centroid - overall).sum())
    return {
        "within_trace": within / len(features),
        "between_trace": between / len(features),
        "between_to_within": between / max(within, 1e-15),
    }


def logit_margins(logits: FloatArray, labels: IntArray) -> FloatArray:
    correct = logits[np.arange(len(labels)), labels]
    masked = logits.copy()
    masked[np.arange(len(labels)), labels] = -np.inf
    return cast(FloatArray, correct - masked.max(axis=1))


def linear_cka(first: FloatArray, second: FloatArray) -> float:
    _validate_pair(first, second)
    centered_first = first - first.mean(axis=0, keepdims=True)
    centered_second = second - second.mean(axis=0, keepdims=True)
    cross = centered_first.T @ centered_second
    numerator = float(np.square(cross).sum())
    first_norm = float(np.square(centered_first.T @ centered_first).sum())
    second_norm = float(np.square(centered_second.T @ centered_second).sum())
    return float(numerator / max(float(np.sqrt(first_norm * second_norm)), 1e-15))


def stratified_primary_bootstrap(
    labels: IntArray,
    standard_drift: FloatArray,
    adversarial_drift: FloatArray,
    standard_retention: FloatArray,
    adversarial_retention: FloatArray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    arrays = (standard_drift, adversarial_drift, standard_retention, adversarial_retention)
    if any(len(value) != len(labels) for value in arrays):
        raise ValueError("bootstrap inputs are not sample-aligned")
    generator = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    drift_differences = np.empty(replicates, dtype=np.float64)
    retention_differences = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = np.concatenate(
            [
                generator.choice(indices, size=len(indices), replace=True)
                for indices in class_indices
            ]
        )
        drift_differences[replicate] = np.median(adversarial_drift[sampled]) - np.median(
            standard_drift[sampled]
        )
        retention_differences[replicate] = (
            adversarial_retention[sampled].mean() - standard_retention[sampled].mean()
        )
    drift_point = float(np.median(adversarial_drift) - np.median(standard_drift))
    retention_point = float(adversarial_retention.mean() - standard_retention.mean())
    drift_interval = np.quantile(drift_differences, [0.025, 0.975])
    retention_interval = np.quantile(retention_differences, [0.025, 0.975])
    supported = bool(drift_interval[1] < 0.0 and retention_interval[0] > 0.0)
    return {
        "replicates": replicates,
        "seed": seed,
        "drift_difference_adversarial_minus_standard": drift_point,
        "drift_ci95": drift_interval.tolist(),
        "retention_difference_adversarial_minus_standard": retention_point,
        "retention_ci95": retention_interval.tolist(),
        "primary_supported": supported,
        "decision_rule": "drift upper < 0 and retention lower > 0",
    }
