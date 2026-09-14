from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from src.config import load_config
from src.io_utils import sha256_file
from src.notebook_support import ROOT

RESULTS_JSON = ROOT / "docs/results.json"
RESULTS_MARKDOWN = ROOT / "docs/results.md"
LOCAL_EVIDENCE = [
    ROOT / "results/generated/evaluation.json",
    ROOT / "results/generated/representations.json",
    ROOT / "artifacts/release/local-release-manifest.json",
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.skipif(
    not all(path.is_file() for path in LOCAL_EVIDENCE),
    reason="raw local experiment evidence is not distributed in the repository",
)
def test_public_aggregate_matches_scientific_evidence() -> None:
    public = _read_json(RESULTS_JSON)
    evaluation = _read_json(ROOT / "results/generated/evaluation.json")
    representations = _read_json(ROOT / "results/generated/representations.json")
    release = _read_json(ROOT / "artifacts/release/local-release-manifest.json")

    assert public["primary_hypothesis"] == representations["primary_hypothesis"]
    for arm in ("standard", "adversarial"):
        expected = evaluation["arms"][arm]
        actual = public["arms"][arm]
        assert actual["clean_test"]["accuracy"] == expected["full_test"]["clean"]["accuracy"]
        assert (
            actual["attack_subset"]["pgd_robust_accuracy"]
            == expected["attack_subset"]["pgd"]["robust_accuracy"]
        )
        assert (
            actual["representation"]["median_cosine_drift"]
            == representations["geometry"][arm]["cosine_drift"]["median"]
        )
        assert (
            public["provenance"]["checkpoints"][arm]["sha256"]
            == release["fixed_checkpoints"][arm]["sha256"]
        )


def test_public_aggregate_is_self_contained_and_matches_configuration() -> None:
    public = _read_json(RESULTS_JSON)
    config = load_config(ROOT / "configs/experiment.yaml")
    assert public["provenance"]["configuration_sha256"] == config.sha256
    assert public["protocol"]["epochs"] == config.integer("training", "epochs")
    assert public["protocol"]["attack_subset_count"] == 740
    assert public["primary_hypothesis"]["primary_supported"] is True
    assert set(public["arms"]) == {"standard", "adversarial"}
    assert len(public["registered_corruptions"]) == 8


def test_public_result_page_is_repository_local_and_scoped() -> None:
    public = _read_json(RESULTS_JSON)
    markdown = RESULTS_MARKDOWN.read_text(encoding="utf-8")
    serialized = json.dumps(public, sort_keys=True)
    assert "/Users/" not in markdown
    assert "/Users/" not in serialized
    assert "physical or safe use" in markdown
    assert "not variation from retraining" in markdown
    assert "PCA, t-SNE, UMAP" in markdown
    assert "No retraining" not in markdown

    targets = re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown)
    for target in targets:
        assert "://" not in target
        path = (RESULTS_MARKDOWN.parent / target.split("#", maxsplit=1)[0]).resolve()
        assert path.is_relative_to(ROOT)
        assert path.exists(), target

    for relative, expected in public["provenance"]["public_assets"].items():
        path = ROOT / relative
        assert path.is_file()
        assert sha256_file(path) == expected
