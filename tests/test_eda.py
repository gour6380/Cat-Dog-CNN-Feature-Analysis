from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data import SampleRecord
from src.eda import read_image_dimensions, summarize_dimensions, summarize_records


def _record(
    tmp_path: Path,
    sample_id: str,
    label: int,
    species: int,
    logical_split: str,
    size: tuple[int, int],
) -> SampleRecord:
    path = tmp_path / f"{sample_id}.jpg"
    Image.new("RGB", size, color=(32, 64, 96)).save(path)
    return SampleRecord(
        sample_id=sample_id,
        image_path=str(path),
        label=label,
        species=species,
        breed_name=("cat_breed" if label == 0 else "dog_breed"),
        official_split="test" if logical_split == "test" else "trainval",
    )


def test_eda_summarizes_registered_balance_and_image_geometry(tmp_path: Path) -> None:
    specifications = {
        "training": ((100, 200), (300, 100)),
        "calibration": ((100, 100), (200, 100)),
        "test": ((150, 300), (300, 150)),
    }
    splits = {
        split: [
            _record(tmp_path, f"{split}-cat", 0, 0, split, sizes[0]),
            _record(tmp_path, f"{split}-dog", 1, 1, split, sizes[1]),
        ]
        for split, sizes in specifications.items()
    }

    records = summarize_records(splits, classes=2)
    dimensions = read_image_dimensions(splits, progress=False)
    geometry = summarize_dimensions(dimensions)

    assert records["total_unique_images"] == 6
    assert records["counts_by_split"] == {"training": 2, "calibration": 2, "test": 2}
    assert records["breeds_by_species"] == {"cat": 1, "dog": 1}
    assert records["breed_counts"][0]["total"] == 3
    assert records["breed_counts"][1]["test"] == 1
    assert geometry["overall"]["orientation"] == {
        "portrait": 2,
        "square": 1,
        "landscape": 3,
    }
    assert geometry["overall"]["width_pixels"]["median"] == 175.0
    assert geometry["overall"]["height_pixels"]["median"] == 125.0
