#!/usr/bin/env bash
# Create the native Python 3.13 environment with pip; uv is not used.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_command="${PYTHON_BIN:-/opt/homebrew/bin/python3.13}"
offline=false

if [[ "${1:-}" == "--offline" && "$#" -eq 1 ]]; then
    offline=true
elif [[ "$#" -ne 0 ]]; then
    printf 'Usage: %s [--offline]\n' "$0" >&2
    exit 2
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    printf 'Week 3 requires native macOS on Apple Silicon.\n' >&2
    exit 1
fi

"$python_command" -c 'import platform; assert platform.python_version() == "3.13.15", "Use CPython 3.13.15"'

if [[ ! -x "$project_root/.venv/bin/python" ]]; then
    "$python_command" -m venv "$project_root/.venv"
fi

"$project_root/.venv/bin/python" -c 'import pathlib, platform, sys; expected = pathlib.Path(sys.argv[1]).resolve(); actual = pathlib.Path(sys.prefix).resolve(); assert actual == expected and platform.machine() == "arm64" and platform.python_version() == "3.13.15", "Existing .venv is not the registered native environment"' "$project_root/.venv"

pip_options=(--only-binary=:all:)
if "$offline"; then
    pip_options+=(--no-index --find-links "$project_root/.local/wheels")
fi

"$project_root/.venv/bin/python" -m pip install "${pip_options[@]}" -r "$project_root/requirements.txt"
"$project_root/.venv/bin/python" -m pip check
"$project_root/.venv/bin/python" -c 'import torch; assert torch.backends.mps.is_built(), "Installed PyTorch was not built with MPS"; print(f"PyTorch {torch.__version__}; MPS built={torch.backends.mps.is_built()}, available={torch.backends.mps.is_available()}")'
"$project_root/.venv/bin/python" -m ipykernel install --prefix "$project_root/.venv" \
    --name oxford-pets-adversarial-representations \
    --display-name "Python (Oxford Pets Adversarial Representations)"
printf '\nReady: %s/.venv/bin/python src/cli.py --help\n' "$project_root"
