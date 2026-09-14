from __future__ import annotations

import subprocess
import sys

from src.notebook_support import ROOT


def test_repository_candidate_passes_read_only_review() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_repository.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Repository candidate passed" in result.stdout
