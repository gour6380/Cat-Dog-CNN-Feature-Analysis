"""Review the local Git candidate without changing the index or publishing."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

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
    Path("notebooks/cat_dog_cnn_features.ipynb"),
}
SAFE_FIGURE_KINDS = {
    "activation_maximization_comparison",
    "kernels",
    "activation_maximization",
    "species_response",
    "initial_response_change",
    "localization_summary",
    "validation_monitoring",
}
# Historical evidence is not a blanket stale-artifact exemption. Register the
# completed immutable study and verify its archived protocol before allowing it.
HISTORICAL_PUBLIC_STUDIES = {
    "3b96c26d328cffc308cb6de70e070748bba3c9fe7a7b229ccba6749f8aec2a91": {
        "experiment_id": "cat-dog-cnn-feature-visualization-v2",
        "configuration_path": "configs/archived_cat_dog_features_v2.yaml",
        "source_revision": "9aa7f2ac49cd938d377db7bb66aae6ee58246338",
        "superseded_by": "cat-dog-cnn-feature-visualization-v3",
    }
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
    # `--cached` includes deleted tracked paths until the owner stages the reset.
    # Review what exists now; a missing old file cannot enter a subsequent upload.
    existing = {path for path in values if (ROOT / path).exists() or (ROOT / path).is_symlink()}
    return sorted(existing, key=str)


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
    guide_path = ROOT / "notebooks/cat_dog_cnn_features.ipynb"
    if not guide_path.is_file():
        return  # The required-candidate check reports this without a traceback.
    try:
        guide = json.loads(guide_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        errors.append(f"{guide_path.relative_to(ROOT)}: unreadable notebook: {error}")
        return
    guide_code = [cell for cell in guide["cells"] if cell["cell_type"] == "code"]
    guide_source = "\n".join("".join(cell["source"]) for cell in guide_code)
    try:
        parsed = ast.parse(guide_source)
        switches: list[ast.expr | None] = []
        value: ast.expr | None
        for node in parsed.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            if any(
                isinstance(target, ast.Name) and target.id == "RUN_FULL_EXPERIMENT"
                for target in targets
            ):
                switches.append(value)
        if not switches or any(
            not isinstance(value, ast.Constant) or type(value.value) is not bool
            for value in switches
        ):
            errors.append(
                f"{guide_path.relative_to(ROOT)}: requires literal "
                "RUN_FULL_EXPERIMENT = True or False"
            )
    except SyntaxError as error:
        errors.append(f"{guide_path.relative_to(ROOT)}: invalid source code: {error.msg}")
    if any(cell.get("outputs") or cell.get("execution_count") is not None for cell in guide_code):
        errors.append(f"{guide_path.relative_to(ROOT)}: source guide contains execution output")
    if any(cell.get("attachments") for cell in guide["cells"]):
        errors.append(f"{guide_path.relative_to(ROOT)}: source guide contains image attachments")
    for cell in guide["cells"]:
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            if "data:image/" in source:
                errors.append(f"{guide_path.relative_to(ROOT)}: source guide embeds an image")
            _check_markdown(guide_path, source, errors)


def _current_config_sha256() -> str:
    return load_config(ROOT / "configs/experiment.yaml", enforce_python=False).sha256


def _current_experiment_id() -> str:
    return load_config(ROOT / "configs/experiment.yaml", enforce_python=False).value(
        "experiment", "id", str
    )


def _check_historical_identity(
    historical: object, configuration_sha256: object, errors: list[str]
) -> None:
    if not isinstance(historical, dict):
        errors.append("docs/results.json: historical evidence metadata must be an object")
        return
    registered = HISTORICAL_PUBLIC_STUDIES.get(str(configuration_sha256))
    if registered is None or any(historical.get(key) != value for key, value in registered.items()):
        errors.append("docs/results.json: historical study/source identity is not registered")
        return
    if (
        historical.get("status") != "archived_completed_experiment"
        or historical.get("configuration_sha256") != configuration_sha256
        or historical.get("current_run_results_available") is not False
    ):
        errors.append("docs/results.json: historical evidence must remain explicitly archived")
        return
    archived_path = ROOT / registered["configuration_path"]
    if archived_path.is_symlink() or not archived_path.resolve().is_relative_to(ROOT.resolve()):
        errors.append("docs/results.json: archived configuration escapes the repository")
        return
    try:
        archived = load_config(archived_path, enforce_python=False)
        if (
            archived.sha256 != configuration_sha256
            or archived.value("experiment", "id", str) != registered["experiment_id"]
        ):
            errors.append(
                "docs/results.json: archived configuration identity differs from evidence"
            )
        if historical["superseded_by"] != _current_experiment_id():
            errors.append(
                "docs/results.json: historical evidence names a different current experiment"
            )
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        errors.append(f"docs/results.json: invalid archived configuration: {error}")


def _check_public_assets(paths: list[Path], errors: list[str]) -> None:
    manifest_path = ROOT / "docs/results.json"
    if not manifest_path.is_file():
        errors.append("docs/results.json: public aggregate/status manifest is missing")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = manifest["provenance"]
        assets = provenance["public_assets"]
        config_sha256 = _current_config_sha256()
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        errors.append(f"docs/results.json: invalid public evidence manifest: {error}")
        return
    evidence_sha256 = provenance.get("configuration_sha256")
    if "historical_evidence" in manifest:
        _check_historical_identity(manifest["historical_evidence"], evidence_sha256, errors)
    elif evidence_sha256 != config_sha256:
        errors.append("docs/results.json: public evidence belongs to a superseded configuration")
    if manifest.get("protocol", {}).get("label_mode") != "species":
        errors.append(
            "docs/results.json: public evidence does not describe the binary species study"
        )
    if not isinstance(assets, dict):
        errors.append("docs/results.json: public_assets must be a path/hash mapping")
        return
    figures = manifest.get("figures", [])
    if not isinstance(figures, list):
        errors.append("docs/results.json: public figures must be a list")
        return
    approved: dict[Path, str] = {}
    for figure in figures:
        if not isinstance(figure, dict) or not isinstance(figure.get("path"), str):
            errors.append("docs/results.json: invalid public figure record")
            continue
        relative = Path(figure["path"])
        path = (ROOT / relative).resolve()
        if (
            relative.is_absolute()
            or not relative.is_relative_to(Path("docs/assets"))
            or not path.is_relative_to((ROOT / "docs/assets").resolve())
            or path.suffix.lower() != ".png"
        ):
            errors.append(f"{relative}: public figure must be a PNG inside docs/assets")
            continue
        if figure.get("kind") not in SAFE_FIGURE_KINDS or figure.get("shareable") is False:
            errors.append(f"{relative}: photo-containing/unapproved figure kind is not shareable")
            continue
        expected_hash = assets.get(str(relative))
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            errors.append(f"{relative}: public asset has no valid evidence hash")
            continue
        if relative in approved:
            errors.append(f"{relative}: duplicate public figure record")
        approved[relative] = expected_hash
    for asset in assets:
        if Path(asset) not in approved:
            errors.append(
                f"{asset}: asset hash is not backed by an approved synthetic/aggregate figure"
            )
    candidate_assets = {path for path in paths if path.is_relative_to(Path("docs/assets"))}
    for relative in candidate_assets - set(approved):
        errors.append(f"{relative}: stale or unmanifested public asset; cleanup/review is pending")
    for relative, expected_hash in approved.items():
        if relative not in candidate_assets or not (ROOT / relative).is_file():
            errors.append(f"{relative}: approved public asset is absent from the Git candidate")
        elif hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected_hash:
            errors.append(f"{relative}: public asset hash differs from saved evidence")


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
    _check_public_assets(paths, errors)
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
