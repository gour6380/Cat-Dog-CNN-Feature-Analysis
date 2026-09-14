"""Oxford Pets adversarial representation experiment."""

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".local" / "matplotlib")
)

__version__ = "0.1.0"
