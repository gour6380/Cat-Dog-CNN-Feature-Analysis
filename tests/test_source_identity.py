from pathlib import Path

from src.io_utils import source_hash


def test_source_hash_excludes_generated_and_local_notebooks(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "mechanism.py").write_text("mechanism = 1\n")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "technical_report.md").write_text("Tracked report\n")
    before = source_hash(tmp_path)
    generated = reports / "generated"
    generated.mkdir()
    (generated / "results.ipynb").write_text("Generated photographs\n")
    assert source_hash(tmp_path) == before
    (reports / "technical_report.md").write_text("Changed tracked report\n")
    assert source_hash(tmp_path) != before
