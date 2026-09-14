"""Review the local Git candidate without changing the index or publishing."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 12 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".joblib",
    ".npy",
    ".npz",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".whl",
    ".zip",
}
ALLOWED_NOTEBOOKS = {
    Path("notebooks/oxford_pets_adversarial_representations.ipynb"),
    Path("notebooks/oxford_pets_results_explained.ipynb"),
}
SECRET_PATTERNS = {
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
PRIVATE_PATH = re.compile(r"/Users/[A-Za-z0-9._-]+/")


def _candidate_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    values = [Path(value.decode()) for value in result.stdout.split(b"\0") if value]
    return sorted(set(values), key=str)


def _markdown_targets(text: str) -> list[tuple[str, str]]:
    return re.findall(r"\[[^]]*\]\(([^)]+)\)|!\[[^]]*\]\(([^)]+)\)", text)


def _check_markdown(path: Path, text: str, errors: list[str]) -> None:
    for pair in _markdown_targets(text):
        target = next(value for value in pair if value)
        if target.startswith(("http://", "https://", "mailto:", "#", "attachment:")):
            continue
        cleaned = unquote(target.split("#", maxsplit=1)[0].strip("<>"))
        destination = (path.parent / cleaned).resolve()
        if not destination.is_relative_to(ROOT):
            errors.append(f"{path}: link escapes repository: {target}")
        elif not destination.exists():
            errors.append(f"{path}: unresolved local link: {target}")


def _check_notebooks(errors: list[str]) -> None:
    guide_path = ROOT / "notebooks/oxford_pets_adversarial_representations.ipynb"
    results_path = ROOT / "notebooks/oxford_pets_results_explained.ipynb"
    guide = json.loads(guide_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    guide_code = [cell for cell in guide["cells"] if cell["cell_type"] == "code"]
    guide_source = "\n".join("".join(cell["source"]) for cell in guide_code)
    if "RUN_FULL_EXPERIMENT = False" not in guide_source:
        errors.append(f"{guide_path.relative_to(ROOT)}: full experiment is enabled")
    if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in guide_code):
        errors.append(f"{guide_path.relative_to(ROOT)}: source guide contains execution output")
    if any(cell["cell_type"] != "markdown" for cell in results["cells"]):
        errors.append(
            f"{results_path.relative_to(ROOT)}: results notebook contains executable cells"
        )
    attachments = {name for cell in results["cells"] for name in cell.get("attachments", {})}
    if len(attachments) != 13:
        errors.append(
            f"{results_path.relative_to(ROOT)}: expected 13 aggregate figures, "
            f"found {len(attachments)}"
        )


def review(*, list_files: bool = False) -> int:
    paths = _candidate_paths()
    errors: list[str] = []
    total_bytes = 0
    for relative in paths:
        path = ROOT / relative
        if list_files:
            print(relative)
        if path.is_symlink():
            errors.append(f"{relative}: symbolic links are not part of the release candidate")
            continue
        size = path.stat().st_size
        total_bytes += size
        if size > MAX_FILE_BYTES:
            errors.append(f"{relative}: {size} bytes exceeds the 12 MiB candidate limit")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"{relative}: binary experiment artifact is not shareable")
        is_image = path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        if is_image and not relative.is_relative_to(Path("docs/assets")):
            errors.append(f"{relative}: images are allowed only under docs/assets")
        if path.suffix == ".ipynb" and relative not in ALLOWED_NOTEBOOKS:
            errors.append(f"{relative}: unreviewed notebook")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if PRIVATE_PATH.search(text):
            errors.append(f"{relative}: contains a private absolute path")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{relative}: possible {label}")
        if path.suffix.lower() == ".md":
            _check_markdown(path, text, errors)

    missing = [path for path in ALLOWED_NOTEBOOKS if path not in paths]
    errors.extend(f"{path}: required shareable notebook is missing" for path in missing)
    _check_notebooks(errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Repository candidate passed: {len(paths)} files, {total_bytes} bytes")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print candidate files")
    arguments = parser.parse_args()
    raise SystemExit(review(list_files=arguments.list))


if __name__ == "__main__":
    main()
