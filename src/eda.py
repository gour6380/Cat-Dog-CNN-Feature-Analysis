"""Deterministic, photograph-free exploratory data analysis for Oxford Pets."""

from __future__ import annotations

import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import ExperimentConfig
from src.data import SampleRecord, load_registered_splits
from src.io_utils import atomic_write_bytes, atomic_write_json, sha256_file, utc_now
from src.progress import status, tqdm

LogicalSplit = Literal["training", "validation", "calibration", "test"]
Orientation = Literal["portrait", "square", "landscape"]
PRIMARY_SPLITS: tuple[LogicalSplit, ...] = ("training", "calibration", "test")
ORIENTATIONS: tuple[Orientation, ...] = ("portrait", "square", "landscape")
SPECIES_NAMES = {0: "cat", 1: "dog"}
EDA_SCHEMA_VERSION = 1


def _primary_splits(splits: Mapping[str, Any]) -> tuple[LogicalSplit, ...]:
    return (
        ("training", "validation", "calibration", "test")
        if "validation" in splits
        else PRIMARY_SPLITS
    )


def _dimension_splits(dimensions: Sequence[ImageDimension]) -> tuple[LogicalSplit, ...]:
    names = {item.split for item in dimensions}
    return tuple(
        name for name in ("training", "validation", "calibration", "test") if name in names
    )


class EDAError(RuntimeError):
    """The registered data cannot support a trustworthy aggregate EDA."""


@dataclass(frozen=True)
class ImageDimension:
    """Stored source-image geometry without pixel content."""

    split: LogicalSplit
    width: int
    height: int
    aspect_ratio: float
    megapixels: float
    orientation: Orientation

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _records_for(
    splits: Mapping[str, Sequence[SampleRecord]], split: LogicalSplit
) -> Sequence[SampleRecord]:
    try:
        return splits[split]
    except KeyError as error:
        raise EDAError(f"registered split {split!r} is missing") from error


def summarize_records(splits: Mapping[str, Sequence[SampleRecord]], classes: int) -> dict[str, Any]:
    """Summarize class and species balance across the three disjoint partitions."""

    if classes <= 1:
        raise EDAError("EDA requires at least two classes")
    seen_ids: set[str] = set()
    label_names: dict[int, str] = {}
    label_species: dict[int, int] = {}
    counts_by_split: dict[str, int] = {}
    species_by_split: dict[str, dict[str, int]] = {}
    per_split_labels: dict[str, Counter[int]] = {}
    per_split_breeds: dict[str, Counter[str]] = {}
    breed_species: dict[str, int] = {}
    primary_splits = _primary_splits(splits)
    species_targets = classes == 2 and all(
        record.label == record.species
        for split in primary_splits
        for record in _records_for(splits, split)
    )

    for split in primary_splits:
        records = _records_for(splits, split)
        counts_by_split[split] = len(records)
        species_counts: Counter[int] = Counter()
        label_counts: Counter[int] = Counter()
        breed_counts_for_split: Counter[str] = Counter()
        for record in records:
            if record.sample_id in seen_ids:
                raise EDAError(f"sample occurs in more than one EDA split: {record.sample_id}")
            seen_ids.add(record.sample_id)
            if not 0 <= record.label < classes:
                raise EDAError(f"invalid class label in registered split: {record.label}")
            if record.species not in SPECIES_NAMES:
                raise EDAError(f"invalid species label in registered split: {record.species}")
            target_name = SPECIES_NAMES[record.species] if species_targets else record.breed_name
            previous_name = label_names.setdefault(record.label, target_name)
            previous_species = label_species.setdefault(record.label, record.species)
            if previous_name != target_name or previous_species != record.species:
                raise EDAError(f"class {record.label} has inconsistent breed metadata")
            prior_species = breed_species.setdefault(record.breed_name, record.species)
            if prior_species != record.species:
                raise EDAError("breed maps to inconsistent species")
            species_counts[record.species] += 1
            label_counts[record.label] += 1
            breed_counts_for_split[record.breed_name] += 1
        species_by_split[split] = {
            name: species_counts[value] for value, name in SPECIES_NAMES.items()
        }
        per_split_labels[split] = label_counts
        per_split_breeds[split] = breed_counts_for_split

    expected_labels = set(range(classes))
    if set(label_names) != expected_labels:
        raise EDAError("registered data does not cover every configured breed")

    target_counts: list[dict[str, Any]] = []
    for label in range(classes):
        row: dict[str, Any] = {
            "class_id": label,
            "breed": label_names[label],
            "species": SPECIES_NAMES[label_species[label]],
            **{split: per_split_labels[split][label] for split in primary_splits},
        }
        row["total"] = sum(per_split_labels[split][label] for split in primary_splits)
        target_counts.append(row)

    breed_counts: list[dict[str, Any]] = []
    for index, breed in enumerate(sorted(breed_species)):
        counts: dict[str, int] = {split: per_split_breeds[split][breed] for split in primary_splits}
        breed_counts.append(
            {
                "class_id": index,
                "breed": breed,
                "species": SPECIES_NAMES[breed_species[breed]],
                **counts,
                "total": sum(counts.values()),
            }
        )
    breeds_by_species = Counter(SPECIES_NAMES[value] for value in breed_species.values())
    return {
        "total_unique_images": len(seen_ids),
        "primary_splits": list(primary_splits),
        "classes": classes,
        "counts_by_split": counts_by_split,
        "species_by_split": species_by_split,
        "breeds_by_species": {name: breeds_by_species[name] for name in SPECIES_NAMES.values()},
        "breed_counts": breed_counts,
        "target_counts": target_counts,
        "label_mode": "species" if species_targets else "breed",
    }


def read_image_dimensions(
    splits: Mapping[str, Sequence[SampleRecord]], *, progress: bool = True
) -> list[ImageDimension]:
    """Read only image headers; original pixels are never copied into EDA artifacts."""

    queued = [
        (split, record)
        for split in _primary_splits(splits)
        for record in _records_for(splits, split)
    ]
    dimensions: list[ImageDimension] = []
    for split, record in tqdm(
        queued,
        desc="EDA image headers",
        unit="image",
        leave=False,
        disable=not progress,
    ):
        path = Path(record.image_path)
        try:
            with Image.open(path) as image:
                width, height = image.size
        except OSError as error:
            raise EDAError(f"cannot read image geometry: {path}") from error
        if width <= 0 or height <= 0:
            raise EDAError(f"invalid image geometry: {path}")
        orientation: Orientation
        if width < height:
            orientation = "portrait"
        elif width > height:
            orientation = "landscape"
        else:
            orientation = "square"
        dimensions.append(
            ImageDimension(
                split=split,
                width=width,
                height=height,
                aspect_ratio=width / height,
                megapixels=(width * height) / 1_000_000.0,
                orientation=orientation,
            )
        )
    return dimensions


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise EDAError("image-geometry distribution is empty or non-finite")
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "count": int(array.size),
        "minimum": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "maximum": float(quantiles[4]),
        "mean": float(array.mean()),
    }


def summarize_dimensions(dimensions: Sequence[ImageDimension]) -> dict[str, Any]:
    """Compute aggregate geometry and orientation summaries."""

    def one_group(values: Sequence[ImageDimension]) -> dict[str, Any]:
        orientations = Counter(item.orientation for item in values)
        return {
            "count": len(values),
            "width_pixels": _distribution([float(item.width) for item in values]),
            "height_pixels": _distribution([float(item.height) for item in values]),
            "aspect_ratio_width_over_height": _distribution([item.aspect_ratio for item in values]),
            "megapixels": _distribution([item.megapixels for item in values]),
            "orientation": {name: orientations[name] for name in ORIENTATIONS},
        }

    if not dimensions:
        raise EDAError("no image dimensions were collected")
    return {
        "overall": one_group(dimensions),
        "by_split": {
            split: one_group([item for item in dimensions if item.split == split])
            for split in _dimension_splits(dimensions)
        },
    }


def _save_figure(figure: Any, destination: Path) -> None:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    atomic_write_bytes(destination, buffer.getvalue())


def _plot_split_and_species(summary: dict[str, Any], destination: Path) -> None:
    colors = {
        "training": "#2563eb",
        "validation": "#7c3aed",
        "calibration": "#f59e0b",
        "test": "#0f766e",
    }
    splits = summary.get("primary_splits", list(PRIMARY_SPLITS))
    counts = cast(dict[str, int], summary["counts_by_split"])
    species = cast(dict[str, dict[str, int]], summary["species_by_split"])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)

    bars = axes[0].bar(
        splits,
        [counts[name] for name in splits],
        color=[colors[name] for name in splits],
        edgecolor="white",
    )
    axes[0].bar_label(bars, fmt="%d", padding=3, fontweight="bold")
    axes[0].set(
        title="Registered partition sizes",
        ylabel="images",
        ylim=(0, max(counts.values()) * 1.15),
    )
    axes[0].grid(axis="y", alpha=0.2)

    cats = np.asarray([species[name]["cat"] for name in splits], dtype=np.float64)
    dogs = np.asarray([species[name]["dog"] for name in splits], dtype=np.float64)
    totals = cats + dogs
    cat_share = cats / totals * 100.0
    dog_share = dogs / totals * 100.0
    axes[1].bar(splits, cat_share, label="cat", color="#7c3aed")
    axes[1].bar(splits, dog_share, bottom=cat_share, label="dog", color="#ea580c")
    for index, (cat_value, dog_value) in enumerate(zip(cat_share, dog_share, strict=True)):
        axes[1].text(
            index,
            cat_value / 2,
            f"{cat_value:.1f}%",
            ha="center",
            va="center",
            color="white",
        )
        axes[1].text(
            index,
            cat_value + dog_value / 2,
            f"{dog_value:.1f}%",
            ha="center",
            va="center",
            color="white",
        )
    axes[1].set(
        title="Species composition inside each split",
        ylabel="share of split (%)",
        ylim=(0, 100),
    )
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", alpha=0.15)
    figure.suptitle("Oxford-IIIT Pet EDA · registered data only", fontsize=15, fontweight="bold")
    _save_figure(figure, destination)


def _plot_breed_balance(summary: dict[str, Any], destination: Path) -> None:
    rows = cast(list[dict[str, Any]], summary["target_counts"])
    species_targets = summary["label_mode"] == "species"
    names = [str(row["breed"]).replace("_", " ") for row in rows]
    training = np.asarray([row["training"] for row in rows], dtype=np.int64)
    validation = np.asarray([row.get("validation", 0) for row in rows], dtype=np.int64)
    calibration = np.asarray([row["calibration"] for row in rows], dtype=np.int64)
    test = np.asarray([row["test"] for row in rows], dtype=np.int64)
    positions = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(14, 4 if species_targets else 12), constrained_layout=True)
    axis.barh(positions - 0.18, training, height=0.34, color="#2563eb", label="training")
    if "validation" in summary.get("primary_splits", []):
        axis.barh(
            positions - 0.18,
            validation,
            left=training,
            height=0.34,
            color="#7c3aed",
            label="validation",
        )
    axis.barh(
        positions - 0.18,
        calibration,
        left=training + validation,
        height=0.34,
        color="#f59e0b",
        label="calibration",
    )
    axis.barh(positions + 0.18, test, height=0.34, color="#0f766e", label="official test")
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set(
        title="Cat/dog target balance"
        if species_targets
        else "Breed balance across the deterministic partitions",
        xlabel="images per target class",
        ylabel="target class",
    )
    axis.grid(axis="x", alpha=0.2)
    axis.legend(ncols=3, loc="lower right")
    _save_figure(figure, destination)


def _plot_image_geometry(dimensions: Sequence[ImageDimension], destination: Path) -> None:
    widths = np.asarray([item.width for item in dimensions], dtype=np.float64)
    heights = np.asarray([item.height for item in dimensions], dtype=np.float64)
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)

    density = axes[0].hexbin(widths, heights, gridsize=32, mincnt=1, bins="log", cmap="viridis")
    limit = float(max(widths.max(), heights.max()))
    axes[0].plot(
        [0, limit],
        [0, limit],
        color="#dc2626",
        linestyle="--",
        linewidth=1,
        label="square",
    )
    axes[0].set(title="Stored width vs height", xlabel="width (px)", ylabel="height (px)")
    axes[0].legend()
    figure.colorbar(density, ax=axes[0], label="log image count")

    split_colors = {
        "training": "#2563eb",
        "validation": "#7c3aed",
        "calibration": "#f59e0b",
        "test": "#0f766e",
    }
    primary_splits = _dimension_splits(dimensions)
    low = max(0.0, float(np.quantile(widths / heights, 0.005)))
    high = float(np.quantile(widths / heights, 0.995))
    bins = np.linspace(low, high, 36)
    for split in primary_splits:
        values = [item.aspect_ratio for item in dimensions if item.split == split]
        axes[1].hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=split_colors[split],
            label=split,
        )
    axes[1].axvline(1.0, color="#111827", linestyle="--", linewidth=1)
    axes[1].set(title="Aspect-ratio distribution", xlabel="width / height", ylabel="density")
    axes[1].legend()
    axes[1].grid(alpha=0.15)

    orientation_colors = ("#7c3aed", "#64748b", "#ea580c")
    bottoms = np.zeros(len(primary_splits), dtype=np.float64)
    for orientation, color in zip(ORIENTATIONS, orientation_colors, strict=True):
        shares = []
        for split in primary_splits:
            selected = [item for item in dimensions if item.split == split]
            shares.append(
                sum(item.orientation == orientation for item in selected) / len(selected) * 100.0
            )
        axes[2].bar(primary_splits, shares, bottom=bottoms, color=color, label=orientation)
        bottoms += np.asarray(shares)
    axes[2].set(title="Orientation composition", ylabel="share of split (%)", ylim=(0, 100))
    axes[2].legend(loc="upper right")
    axes[2].grid(axis="y", alpha=0.15)
    figure.suptitle(
        "Source-image geometry · aggregate headers only · no photographs",
        fontsize=15,
        fontweight="bold",
    )
    _save_figure(figure, destination)


def _load_cached(config: ExperimentConfig, split_sha256: str) -> dict[str, Any] | None:
    result_path = config.project_path("results") / "eda.json"
    if not result_path.is_file():
        return None
    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return None
    result = cast(dict[str, Any], loaded)
    if (
        result.get("schema_version") != EDA_SCHEMA_VERSION
        or result.get("config_sha256") != config.sha256
        or result.get("split_sha256") != split_sha256
    ):
        return None
    figures = result.get("figure_sha256")
    if not isinstance(figures, dict):
        return None
    for relative, expected in figures.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            return None
        path = config.root / relative
        if not path.is_file() or sha256_file(path) != expected:
            return None
    return result


def generate_eda(
    config: ExperimentConfig, *, progress: bool = True, force: bool = False
) -> dict[str, Any]:
    """Create reusable EDA evidence and three aggregate figure groups."""

    data_manifest_path = config.project_path("artifacts") / "data" / "manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data_manifest, dict) or not isinstance(
        data_manifest.get("split_sha256"), str
    ):
        raise EDAError("registered data manifest is missing or invalid")
    split_sha256 = cast(str, data_manifest["split_sha256"])
    if not force:
        cached = _load_cached(config, split_sha256)
        if cached is not None:
            status("EDA: reusing matching aggregate figures.", enabled=progress)
            return cached

    status("EDA: summarizing splits, breeds, species, and image geometry...", enabled=progress)
    splits = load_registered_splits(config)
    from src.training_monitoring import prepare_training_monitoring, training_monitoring_enabled

    if training_monitoring_enabled(config):
        protocol = prepare_training_monitoring(config, splits)
        splits = {
            **splits,
            "training": protocol.fit_records,
            "validation": protocol.validation_records,
        }
    record_summary = summarize_records(splits, config.integer("dataset", "classes"))
    dimensions = read_image_dimensions(splits, progress=progress)
    dimension_summary = summarize_dimensions(dimensions)

    figure_root = config.project_path("figures") / "eda"
    destinations = {
        "split_and_species": figure_root / "split-and-species.png",
        "class_balance": figure_root / "target-balance.png",
        "image_geometry": figure_root / "image-geometry.png",
    }
    relative_figures = {
        name: str(path.relative_to(config.root)) for name, path in destinations.items()
    }
    _plot_split_and_species(record_summary, config.root / relative_figures["split_and_species"])
    _plot_breed_balance(record_summary, config.root / relative_figures["class_balance"])
    _plot_image_geometry(dimensions, config.root / relative_figures["image_geometry"])
    figure_sha256 = {
        relative: sha256_file(config.root / relative) for relative in relative_figures.values()
    }
    result = {
        "schema_version": EDA_SCHEMA_VERSION,
        "created_at": utc_now(),
        "config_sha256": config.sha256,
        "split_sha256": split_sha256,
        "record_summary": record_summary,
        "image_geometry": dimension_summary,
        "figures": relative_figures,
        "figure_sha256": figure_sha256,
        "contains_original_photographs": False,
        "method": "registered split metadata plus stored image headers; no image pixels exported",
    }
    result_path = config.project_path("results") / "eda.json"
    atomic_write_json(result_path, result)
    atomic_write_json(
        config.project_path("artifacts") / "eda" / "manifest.json",
        {
            "created_at": utc_now(),
            "config_sha256": config.sha256,
            "split_sha256": split_sha256,
            "result": str(result_path.relative_to(config.root)),
            "result_sha256": sha256_file(result_path),
            "figure_sha256": figure_sha256,
            "contains_original_photographs": False,
        },
    )
    status("EDA: aggregate figures and machine-readable summary saved.", enabled=progress)
    return result
