"""Original-feature evidence and explicitly explanatory projections."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from src.config import ExperimentConfig
from src.data import load_registered_splits
from src.evaluation import EvaluationError, load_arrays
from src.io_utils import atomic_save_npz, atomic_write_json, sha256_file, utc_now
from src.metrics import (
    class_centroids,
    cosine_feature_drift,
    knn_metrics,
    linear_cka,
    logit_margins,
    nearest_centroid_metrics,
    relative_l2_drift,
    scatter_metrics,
    stratified_primary_bootstrap,
)
from src.model import load_checkpoint_model
from src.progress import status, tqdm
from src.training import Arm, checkpoint_path

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _path(config: ExperimentConfig, *parts: str) -> Path:
    return config.project_path("results") / "per_sample" / Path(*parts)


def _load_required(
    config: ExperimentConfig, arm: Arm
) -> dict[str, dict[str, np.ndarray[Any, Any]]]:
    return {
        "calibration": load_arrays(_path(config, "calibration", f"{arm}.npz")),
        "clean": load_arrays(_path(config, "attack", arm, "clean.npz")),
        "pgd": load_arrays(_path(config, "attack", arm, "pgd20x5.npz")),
    }


def _aligned(
    first: dict[str, np.ndarray[Any, Any]], second: dict[str, np.ndarray[Any, Any]]
) -> None:
    if not np.array_equal(first["sample_ids"], second["sample_ids"]):
        raise EvaluationError("representation inputs are not sample-aligned")
    if not np.array_equal(first["labels"], second["labels"]):
        raise EvaluationError("representation labels are not aligned")


def _projection_mask(ids: np.ndarray[Any, Any], selected_ids: set[str]) -> NDArray[np.bool_]:
    mask = np.asarray([str(value) in selected_ids for value in ids], dtype=np.bool_)
    return mask


def _arm_geometry(
    config: ExperimentConfig,
    values: dict[str, dict[str, np.ndarray[Any, Any]]],
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    calibration = values["calibration"]
    clean = values["clean"]
    pgd = values["pgd"]
    _aligned(clean, pgd)
    calibration_features = calibration["features"].astype(np.float64)
    calibration_labels = calibration["labels"].astype(np.int64)
    clean_features = clean["features"].astype(np.float64)
    shifted_features = pgd["features"].astype(np.float64)
    labels = clean["labels"].astype(np.int64)
    neighbours = config.integer("representations", "neighbours")
    classes = config.integer("dataset", "classes")
    clean_knn = knn_metrics(
        calibration_features, calibration_labels, clean_features, labels, neighbours
    )
    shifted_knn = knn_metrics(
        calibration_features, calibration_labels, shifted_features, labels, neighbours
    )
    clean_centroid = nearest_centroid_metrics(
        calibration_features, calibration_labels, clean_features, labels, classes
    )
    shifted_centroid = nearest_centroid_metrics(
        calibration_features, calibration_labels, shifted_features, labels, classes
    )
    cosine = cosine_feature_drift(clean_features, shifted_features)
    relative = relative_l2_drift(clean_features, shifted_features)
    clean_margins = logit_margins(clean["logits"].astype(np.float64), labels)
    shifted_margins = logit_margins(pgd["logits"].astype(np.float64), labels)
    per_class: dict[str, Any] = {}
    for label in range(classes):
        selected = labels == label
        per_class[str(label)] = {
            "count": int(selected.sum()),
            "median_cosine_drift": float(np.median(cosine[selected])),
            "median_relative_l2_drift": float(np.median(relative[selected])),
            "clean_knn_retention": float(clean_knn.per_sample_retention[selected].mean()),
            "pgd_knn_retention": float(shifted_knn.per_sample_retention[selected].mean()),
            "clean_margin_mean": float(clean_margins[selected].mean()),
            "pgd_margin_mean": float(shifted_margins[selected].mean()),
        }
    geometry = {
        "feature_dimension": int(clean_features.shape[1]),
        "sample_count": len(labels),
        "cosine_drift": {
            "median": float(np.median(cosine)),
            "mean": float(cosine.mean()),
        },
        "relative_l2_drift": {
            "median": float(np.median(relative)),
            "mean": float(relative.mean()),
        },
        "five_nn": {
            "clean_accuracy": clean_knn.accuracy,
            "pgd_accuracy": shifted_knn.accuracy,
            "clean_retention": clean_knn.retention,
            "pgd_retention": shifted_knn.retention,
        },
        "nearest_centroid": {
            "clean_accuracy": clean_centroid["accuracy"],
            "pgd_accuracy": shifted_centroid["accuracy"],
        },
        "scatter": {
            "clean": scatter_metrics(clean_features, labels, classes),
            "pgd": scatter_metrics(shifted_features, labels, classes),
        },
        "logit_margin": {
            "clean_mean": float(clean_margins.mean()),
            "pgd_mean": float(shifted_margins.mean()),
            "clean_median": float(np.median(clean_margins)),
            "pgd_median": float(np.median(shifted_margins)),
        },
        "linear_cka_clean_to_pgd": linear_cka(clean_features, shifted_features),
        "per_class": per_class,
    }
    aligned_arrays: dict[str, np.ndarray[Any, Any]] = {
        "sample_ids": clean["sample_ids"],
        "labels": labels,
        "cosine_drift": cosine,
        "relative_l2_drift": relative,
        "clean_knn_retention": clean_knn.per_sample_retention,
        "pgd_knn_retention": shifted_knn.per_sample_retention,
        "clean_knn_predictions": clean_knn.predictions,
        "pgd_knn_predictions": shifted_knn.predictions,
        "clean_centroid_predictions": cast(np.ndarray[Any, Any], clean_centroid["predictions"]),
        "pgd_centroid_predictions": cast(np.ndarray[Any, Any], shifted_centroid["predictions"]),
        "clean_margins": clean_margins,
        "pgd_margins": shifted_margins,
    }
    return geometry, aligned_arrays


def _select_difficult_pair(
    both: dict[Arm, dict[str, dict[str, np.ndarray[Any, Any]]]], classes: int
) -> tuple[int, int, dict[str, Any]]:
    combined_confusion = np.zeros((classes, classes), dtype=np.int64)
    distance_by_pair: dict[tuple[int, int], list[float]] = {}
    for values in both.values():
        calibration = values["calibration"]
        labels = calibration["labels"].astype(np.int64)
        predictions = calibration["logits"].argmax(axis=1)
        combined_confusion += confusion_matrix(labels, predictions, labels=np.arange(classes))
        centroids = class_centroids(calibration["features"].astype(np.float64), labels, classes)
        normalized = centroids / np.clip(
            np.linalg.norm(centroids, axis=1, keepdims=True), 1e-15, None
        )
        for first in range(classes):
            for second in range(first + 1, classes):
                distance = 1.0 - float(normalized[first] @ normalized[second])
                distance_by_pair.setdefault((first, second), []).append(distance)
    candidates: list[tuple[int, float, int, int]] = []
    for first in range(classes):
        for second in range(first + 1, classes):
            symmetric = int(combined_confusion[first, second] + combined_confusion[second, first])
            distance = float(np.mean(distance_by_pair[(first, second)]))
            candidates.append((-symmetric, distance, first, second))
    best = min(candidates)
    first, second = best[2], best[3]
    return (
        first,
        second,
        {
            "first_class": first,
            "second_class": second,
            "symmetric_confusion_count_across_models": -best[0],
            "mean_cosine_centroid_distance": best[1],
            "tie_breakers": "minimum centroid distance, then ascending class IDs",
        },
    )


def _scatter_projection(
    clean: FloatArray,
    shifted: FloatArray,
    labels: IntArray,
    species: IntArray,
    title: str,
    destination: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    colors = plt.get_cmap("turbo")(labels / max(int(labels.max()), 1))
    for axis, coordinates, state in zip(axes, (clean, shifted), ("clean", "PGD-20×5"), strict=True):
        for species_value, marker, label in ((0, "o", "cat"), (1, "^", "dog")):
            selected = species == species_value
            axis.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                c=colors[selected],
                marker=marker,
                s=18,
                alpha=0.72,
                linewidths=0,
                label=label,
            )
        axis.set_title(state)
        axis.set_xlabel("component 1")
        axis.set_ylabel("component 2")
        axis.grid(alpha=0.15)
    axes[1].legend(title="species marker", loc="best")
    figure.suptitle(
        title + "\nBreed is colour; coordinates are comparable only within this model/view."
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _projection_group(
    config: ExperimentConfig,
    arm: Arm,
    values: dict[str, dict[str, np.ndarray[Any, Any]]],
    projection_ids: set[str],
    *,
    progress: bool,
) -> tuple[dict[str, Any], PCA]:
    import umap

    calibration = values["calibration"]
    clean_all = values["clean"]
    pgd_all = values["pgd"]
    mask = _projection_mask(clean_all["sample_ids"], projection_ids)
    expected = config.integer("dataset", "projection_per_class") * config.integer(
        "dataset", "classes"
    )
    if int(mask.sum()) != expected:
        raise EvaluationError(
            f"projection selection contains {int(mask.sum())}, expected {expected}"
        )
    clean = clean_all["features"][mask].astype(np.float64)
    shifted = pgd_all["features"][mask].astype(np.float64)
    labels = clean_all["labels"][mask].astype(np.int64)
    species = clean_all["species"][mask].astype(np.int64)
    seed = config.integer("representations", "projection_seed")
    figure_root = config.project_path("figures") / "projections"
    coordinate_root = config.project_path("results") / "projection_coordinates"
    status(f"Represent: {arm} — fitting calibration PCA.", enabled=progress)
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(calibration["features"].astype(np.float64))
    pca_clean = pca.transform(clean)
    pca_shifted = pca.transform(shifted)
    _scatter_projection(
        pca_clean,
        pca_shifted,
        labels,
        species,
        f"PCA — {arm} model (fit on clean calibration features)",
        figure_root / f"pca-{arm}.png",
    )
    joint = np.concatenate([clean, shifted], axis=0)
    status(f"Represent: {arm} — fitting joint clean/PGD t-SNE.", enabled=progress)
    tsne = TSNE(
        n_components=2,
        init="pca",
        perplexity=config.number("representations", "tsne_perplexity"),
        learning_rate="auto",
        max_iter=config.integer("representations", "tsne_iterations"),
        random_state=seed,
        method="barnes_hut",
    )
    tsne_joint = tsne.fit_transform(joint)
    tsne_clean, tsne_shifted = np.split(tsne_joint, 2)
    _scatter_projection(
        tsne_clean,
        tsne_shifted,
        labels,
        species,
        f"t-SNE — {arm} model (joint clean + PGD fit)",
        figure_root / f"tsne-{arm}.png",
    )
    status(f"Represent: {arm} — fitting and transforming calibration UMAP.", enabled=progress)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=config.integer("representations", "umap_neighbours"),
        min_dist=config.number("representations", "umap_min_distance"),
        metric=config.value("representations", "umap_metric", str),
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )
    reducer.fit(calibration["features"].astype(np.float64))
    umap_clean = reducer.transform(clean)
    umap_shifted = reducer.transform(shifted)
    _scatter_projection(
        umap_clean,
        umap_shifted,
        labels,
        species,
        f"UMAP — {arm} model (fit on clean calibration features)",
        figure_root / f"umap-{arm}.png",
    )
    coordinates: dict[str, np.ndarray[Any, Any]] = {
        "sample_ids": clean_all["sample_ids"][mask],
        "labels": labels,
        "species": species,
        "pca_clean": pca_clean,
        "pca_pgd": pca_shifted,
        "tsne_clean": tsne_clean,
        "tsne_pgd": tsne_shifted,
        "umap_clean": np.asarray(umap_clean),
        "umap_pgd": np.asarray(umap_shifted),
    }
    atomic_save_npz(coordinate_root / f"{arm}.npz", coordinates)
    return {
        "sample_count": expected,
        "selection": "exactly 10 hash-selected official-test images per breed",
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_fit_partition": "clean calibration",
        "tsne_fit": "joint clean and PGD fixed subset within this model",
        "tsne_perplexity": config.number("representations", "tsne_perplexity"),
        "tsne_iterations": config.integer("representations", "tsne_iterations"),
        "umap_fit_partition": "clean calibration",
        "umap_neighbours": config.integer("representations", "umap_neighbours"),
        "umap_min_distance": config.number("representations", "umap_min_distance"),
        "umap_metric": config.value("representations", "umap_metric", str),
        "seed": seed,
        "interpretation": "explanatory view; raw coordinates are not compared between models",
    }, pca


def _boundary_figure(
    config: ExperimentConfig,
    arm: Arm,
    values: dict[str, dict[str, np.ndarray[Any, Any]]],
    pca: PCA,
    pair: tuple[int, int],
) -> dict[str, Any]:
    calibration = values["calibration"]
    labels = calibration["labels"].astype(np.int64)
    features = calibration["features"].astype(np.float64)
    selected = (labels == pair[0]) | (labels == pair[1])
    coordinates = pca.transform(features[selected])
    model = load_checkpoint_model(config, checkpoint_path(config, arm), torch.device("cpu"))
    weights = model.classifier.weight.detach().cpu().numpy().astype(np.float64)
    biases = model.classifier.bias.detach().cpu().numpy().astype(np.float64)
    difference = weights[pair[0]] - weights[pair[1]]
    normal = pca.components_ @ difference
    intercept = float(pca.mean_ @ difference + biases[pair[0]] - biases[pair[1]])
    x_min, x_max = float(coordinates[:, 0].min()), float(coordinates[:, 0].max())
    padding = max((x_max - x_min) * 0.08, 1e-6)
    x_values = np.linspace(x_min - padding, x_max + padding, 200)
    if abs(float(normal[1])) < 1e-12:
        y_values = None
        boundary_x = -intercept / float(normal[0]) if abs(float(normal[0])) >= 1e-12 else 0.0
    else:
        y_values = -(normal[0] * x_values + intercept) / normal[1]
        boundary_x = None
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for class_id, marker in ((pair[0], "o"), (pair[1], "s")):
        class_mask = labels[selected] == class_id
        axis.scatter(
            coordinates[class_mask, 0],
            coordinates[class_mask, 1],
            s=28,
            marker=marker,
            alpha=0.75,
            label=f"class {class_id}",
        )
    if y_values is not None:
        axis.plot(x_values, y_values, color="black", linewidth=2, label="exact head equality")
    else:
        assert boundary_x is not None
        axis.axvline(boundary_x, color="black", linewidth=2, label="exact head equality")
    axis.set_title(f"{arm}: exact {pair[0]} vs {pair[1]} classifier slice in PCA plane")
    axis.set_xlabel("PCA component 1")
    axis.set_ylabel("PCA component 2")
    axis.legend()
    axis.grid(alpha=0.15)
    destination = config.project_path("figures") / "boundaries" / f"pair-boundary-{arm}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return {
        "classes": list(pair),
        "normal_in_pca_plane": normal.tolist(),
        "intercept": intercept,
        "definition": (
            "exact equality of the two selected linear-head logits with all non-displayed "
            "PCA coordinates fixed at the calibration mean"
        ),
    }


def _geometry_figures(
    config: ExperimentConfig,
    aligned: dict[Arm, dict[str, np.ndarray[Any, Any]]],
) -> None:
    root = config.project_path("figures") / "geometry"
    root.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.boxplot(
        [aligned["standard"]["cosine_drift"], aligned["adversarial"]["cosine_drift"]],
        tick_labels=["standard", "adversarial"],
        showfliers=False,
    )
    axis.set_ylabel("1 − cosine(clean feature, PGD feature)")
    axis.set_title("Original 512-D clean-to-PGD representation drift")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(root / "cosine-drift.png", dpi=180)
    plt.close(figure)

    labels = aligned["standard"]["labels"].astype(np.int64)
    classes = config.integer("dataset", "classes")
    standard = aligned["standard"]["pgd_knn_retention"]
    adversarial = aligned["adversarial"]["pgd_knn_retention"]
    standard_means = [float(standard[labels == label].mean()) for label in range(classes)]
    adversarial_means = [float(adversarial[labels == label].mean()) for label in range(classes)]
    positions = np.arange(classes)
    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    axis.bar(positions - 0.2, standard_means, width=0.4, label="standard")
    axis.bar(positions + 0.2, adversarial_means, width=0.4, label="adversarial")
    axis.set_xlabel("breed class ID")
    axis.set_ylabel("fraction of five clean-calibration neighbours with true breed")
    axis.set_title("PGD five-neighbour breed retention in original 512-D space")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(root / "knn-retention-by-breed.png", dpi=180)
    plt.close(figure)


def _error_taxonomy(
    config: ExperimentConfig,
    both: dict[Arm, dict[str, dict[str, np.ndarray[Any, Any]]]],
    aligned: dict[Arm, dict[str, np.ndarray[Any, Any]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "created_at": utc_now(),
        "scope": "aggregate fixed attack subset; contains no pet photographs",
        "arms": {},
    }
    for arm in ("standard", "adversarial"):
        clean = both[arm]["clean"]
        pgd = both[arm]["pgd"]
        labels = clean["labels"].astype(np.int64)
        clean_prediction = clean["logits"].argmax(axis=1)
        pgd_prediction = pgd["logits"].argmax(axis=1)
        clean_correct = clean_prediction == labels
        pgd_correct = pgd_prediction == labels
        retention = aligned[arm]["pgd_knn_retention"]
        output["arms"][arm] = {
            "clean_correct_pgd_wrong": int((clean_correct & ~pgd_correct).sum()),
            "clean_wrong_pgd_wrong": int((~clean_correct & ~pgd_correct).sum()),
            "clean_wrong_pgd_correct": int((~clean_correct & pgd_correct).sum()),
            "pgd_correct_low_neighbour_retention": int((pgd_correct & (retention < 0.5)).sum()),
            "pgd_wrong_high_neighbour_retention": int((~pgd_correct & (retention >= 0.8)).sum()),
            "counterexample_definition": (
                "classification and local geometry can disagree; inspect aggregate IDs "
                "locally before any narrative"
            ),
        }
    atomic_write_json(config.project_path("results") / "error_taxonomy.json", output)
    return output


def represent(config: ExperimentConfig, *, progress: bool = True) -> dict[str, Any]:
    status(
        "Represent: loading aligned 512-D features for both fixed model arms...",
        enabled=progress,
    )
    arm_names: tuple[Arm, Arm] = ("standard", "adversarial")
    splits = load_registered_splits(config)
    projection_ids = {item.sample_id for item in splits["projection"]}
    both: dict[Arm, dict[str, dict[str, np.ndarray[Any, Any]]]] = {
        "standard": _load_required(config, "standard"),
        "adversarial": _load_required(config, "adversarial"),
    }
    _aligned(both["standard"]["clean"], both["adversarial"]["clean"])
    geometry: dict[Arm, dict[str, Any]] = {}
    aligned: dict[Arm, dict[str, np.ndarray[Any, Any]]] = {}
    projections: dict[Arm, dict[str, Any]] = {}
    pcas: dict[Arm, PCA] = {}
    for arm in tqdm(
        arm_names,
        desc="representation arms",
        unit="arm",
        disable=not progress,
    ):
        status(f"Represent: {arm} — quantitative geometry.", enabled=progress)
        geometry[arm], aligned[arm] = _arm_geometry(config, both[arm])
        atomic_save_npz(
            config.project_path("results") / f"representation-samples-{arm}.npz", aligned[arm]
        )
        projections[arm], pcas[arm] = _projection_group(
            config, arm, both[arm], projection_ids, progress=progress
        )
    status("Represent: class-stratified primary bootstrap.", enabled=progress)
    labels = aligned["standard"]["labels"].astype(np.int64)
    primary = stratified_primary_bootstrap(
        labels,
        aligned["standard"]["cosine_drift"].astype(np.float64),
        aligned["adversarial"]["cosine_drift"].astype(np.float64),
        aligned["standard"]["pgd_knn_retention"].astype(np.float64),
        aligned["adversarial"]["pgd_knn_retention"].astype(np.float64),
        config.integer("representations", "bootstrap_replicates"),
        config.integer("representations", "bootstrap_seed"),
    )
    pair_first, pair_second, pair_record = _select_difficult_pair(
        both, config.integer("dataset", "classes")
    )
    status("Represent: exact difficult-pair boundaries and geometry charts.", enabled=progress)
    boundaries = {
        arm: _boundary_figure(config, arm, both[arm], pcas[arm], (pair_first, pair_second))
        for arm in arm_names
    }
    _geometry_figures(config, aligned)
    taxonomy = _error_taxonomy(config, both, aligned)
    result = {
        "schema_version": 1,
        "created_at": utc_now(),
        "config_sha256": config.sha256,
        "quantitative_space": "original 512-dimensional penultimate features",
        "geometry": geometry,
        "primary_hypothesis": primary,
        "projection_settings": projections,
        "difficult_pair_selection": pair_record,
        "boundary_slices": boundaries,
        "error_taxonomy": taxonomy,
        "interpretation_limits": [
            "raw coordinates are not compared across independently fitted model projections",
            "PCA, t-SNE, and UMAP are explanatory views, not robustness evidence",
            "finite PGD-20x5 evidence does not establish physical or unrestricted robustness",
        ],
    }
    destination = config.project_path("results") / "representations.json"
    atomic_write_json(destination, result)
    figure_hashes = {
        str(path.relative_to(config.root)): sha256_file(path)
        for path in sorted(config.project_path("figures").rglob("*.png"))
    }
    atomic_write_json(
        config.project_path("artifacts") / "representations" / "figures.json",
        {"created_at": utc_now(), "figure_sha256": figure_hashes},
    )
    status(
        "Represent complete: quantitative results and projection figures are registered.",
        enabled=progress,
    )
    return result
