# GitHub handoff

This checkout is a local release candidate. Preparing these files does not create a repository,
configure a remote, publish a release, or grant permission for a public upload.

## Candidate contents

The intended repository contains source, tests, the pinned environment, protocol/configuration,
the output-free experiment guide, the read-only visual result notebook, aggregate result JSON,
two aggregate figures, and reviewed reports. It excludes datasets, pretrained weights,
checkpoints, original photographs, logits, features, per-sample arrays, caches, and local run
manifests.

## Final local review

From the project root:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src
.venv/bin/python scripts/check_repository.py
.venv/bin/python scripts/check_repository.py --list
git status --short
```

The repository checker is read-only. It reviews tracked and unignored candidate files for private
absolute paths, common credential forms, oversized files, forbidden experiment binaries, notebook
state, and broken local Markdown links.

## Public action gate

Only after the owner separately approves a public repository and exact candidate should a GitHub
remote be created or a branch pushed. Re-run the checks immediately before that action and use the
actual uploaded commit for future evidence links. Do not describe the finite PGD result as
certified, physical, generally safe, or production-ready.
