# GitHub handoff

Preparing this checkout does not create a remote, push a branch, publish a release,
or authorize a public claim. The 15-epoch Cat/Dog experiment and public-safe exports
must pass the local gates below before any upload.

Suggested repository name: `cat-dog-cnn-feature-analysis`

Suggested description:

> Matched standard and pure-PGD ResNet-18 training on Oxford-IIIT Pet, with
> class-collapse diagnostics and reproducible CNN feature visualizations on Apple silicon.

Suggested topics: `pytorch`, `computer-vision`, `cnn-interpretability`,
`adversarial-training`, `grad-cam`, `apple-silicon`, `mps`, `reproducible-research`.

## Candidate contents

Include source, tests, the pinned environment, protocol/configuration, the output-free
`notebooks/cat_dog_cnn_features.ipynb` guide, evidence-backed aggregate JSON/reports,
and reviewed aggregate charts or synthetic feature images. The public result must say
that the pure-PGD arm collapsed to the majority dog class; it must not be described as
robust recognition.

Exclude datasets, pretrained weights, checkpoints, per-sample arrays, caches, local
run manifests/logs, original pet photographs, real top patches, activation/Grad-CAM
overlays, and rendered results notebooks. The old breed-results explanation notebook
is removed rather than reused for the binary task. A source notebook with photo
outputs must not enter the candidate by accident.

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

The checker is read-only and reviews candidate paths, secrets, sizes, forbidden
experiment binaries, notebook state, and local links. Also review images manually:
a filename or `.gitignore` alone cannot prove a figure contains no photographs.

Public result JSON records the pre-export source/worktree base, not its own future
Git commit hash. Once reviewed presentation files are committed locally, regenerate
the ignored local release manifest with the report command to record the actual clean
candidate commit and active-file inventory. The active workflow interprets clean images
from the standard and PGD-trained models; it does not run attack, corruption or
confidence-policy evaluation. A normal clone may have an `origin` remote; local setup
records its name but never pushes or publishes.

## Public action gate

Obtain a separate approval for the exact public candidate and destination before
creating a GitHub remote or pushing. Use the uploaded commit's evidence links.
Any photographs later approved for sharing require Oxford-IIIT Pet attribution and
license review; they are not covered by the project MIT license.

Claims must distinguish observed activations, local sensitivity, class attribution,
masking response, and scientific inference. Do not call a channel an eye/ear detector
without validation. This clean-input feature walkthrough does not measure robust
accuracy or establish physical robustness, safety or production readiness.
