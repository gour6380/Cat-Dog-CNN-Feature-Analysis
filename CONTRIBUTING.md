# Contributing

Keep recorded evidence, illustrative images and interpretation distinct. The current
15-epoch run is a negative pure-PGD result; do not replace it with historical metrics
or describe its feature panels as evidence of robustness.

## Development

Use the project's CPython 3.13.15 environment, PyTorch/MPS and pinned requirements:

```bash
./setup_venv.sh
.venv/bin/python -m pytest -q -p no:cacheprovider -m "not mps"
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q -p no:cacheprovider -m mps
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src
.venv/bin/python scripts/check_repository.py
```

The first pytest command matches hosted CPU CI. The marked MPS smoke test runs locally
on Apple silicon because hosted macOS MPS availability does not guarantee usable Metal
memory. Tests must use bounded synthetic inputs. Do not launch full training, download
datasets or restore checkpoints as a test/documentation side effect.

## Protocol changes

- Edit valid parameters in `configs/experiment.yaml`, not duplicated constants.
  The current 15-epoch setting identifies the recorded result. A changed configuration creates
  a new run identity; never relabel old results.
- Preserve species labels, official-test isolation, breed-stratified splits,
  identical fitting/validation IDs, initialization, order and update counts.
  Validation monitoring must not select or stop the fixed-final-epoch checkpoint.
- Preserve the standard clean-CE and pure PGD-5-CE objectives. Native MPS and tqdm
  remain the defaults; do not silently change settings or enable fallback.
- Clean monitoring must restore module modes, BatchNorm, weights, gradients,
  optimizer state and caller RNG streams. Extend durable checkpoint history.
  Charts use recorded epochs only, including honest gaps and one-epoch histories.
- Select channels on reference images, not attractive test pictures; retain
  deterministic anchors and failed probes.
- Test hooks, spatial alignment, receptive-field boxes, optimization bounds,
  gradients, model modes, randomized controls and artifact provenance.

The active workflow is feature-only after training. Legacy evaluation helpers may
remain available, but separate attack/corruption/confidence results are not required
by the notebook, reproduction command or feature report.

## Notebooks and figures

Regenerate the source guide with:

```bash
.venv/bin/python scripts/build_notebook.py
```

Keep the tracked guide at `RUN_FULL_EXPERIMENT=False` and preserve its registered
kernel metadata when rebuilding. Owner-executed copies and their outputs stay under
ignored local paths. Safe mode reads available evidence without computing missing
explanations. Keep tracked outputs, execution counts and attachments clear. Never
embed a historical gallery or local photo in the guide.

Training cells show only live objective-loss and clean validation-accuracy panels.
Keep five independent **Learned features** cells (`stages`, `kernels`, `synthetic`,
`real_patches`, clean-only `activations`) and three **Prediction influence** cells
(`gradcam`, `occlusion`, `diagnostics`). Match notebook compute/display selectors
to CLI selectors. Do not make one cell silently run unrelated families.

Cache receipts must match current method, data, configuration, model and figure
hashes. Save partial progress atomically. Displays validate saved evidence and do
not compute a replacement for missing/stale pictures. Reports are complete only
with the required current feature families for both models; no fixed figure count
or unrelated evaluation file is a completion gate.

Real photographs, patches, overlays, weights, checkpoints, arrays, logs and executed
result companions remain ignored. Only reviewed synthetic/aggregate exports are
public candidates; no publishing is implicit.

## Interpretation

Explain feature maps as channel responses, not reconstructed photographs or
verified anatomical detectors. Synthetic stimuli are probes, not training memories.
Grad-CAM, input gradients and occlusion use different objectives; visual agreement
is not causal validation. Record normalization and compare response numbers rather
than independently normalized brightness. Do not convert PGD training or attractive
images into robustness, safety, certification or deployment claims.
