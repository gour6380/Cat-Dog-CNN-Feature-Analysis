"""Embed saved feature figures into an ignored local read-only notebook."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.feature_results_notebook import build_feature_results_notebook

if __name__ == "__main__":
    print(
        build_feature_results_notebook(
            load_config(Path(__file__).resolve().parents[1] / "configs/experiment.yaml")
        )
    )
