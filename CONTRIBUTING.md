# Contributing

Keep scientific evidence, illustrative images, and speculation distinct.

## Development

Use native CPython 3.13.15 on Apple Silicon with `./setup_venv.sh`, then run:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src
.venv/bin/python scripts/check_repository.py
```

## Experiment changes

- Edit valid parameters in `configs/experiment.yaml`, not duplicated constants.
  A new configuration is a new experiment identity, not a license to relabel results.
- Preserve official-test isolation, breed-stratified training/calibration, species
  labels, matched-arm initialization/order/update counts, and final-checkpoint choice.
- Select channels on clean calibration and retain fixed test anchors. Never choose
  a channel or source image solely because its picture supports a preferred story.
- Test hooks, spatial alignment, receptive fields, optimization bounds, gradients,
  explanation model modes, randomized controls, calibration, and artifact provenance.
- Record visual normalization. Separately normalized panels are not absolute
  comparisons, and a top patch is not proof of a semantic concept.

## Notebooks and figures

Generate the output-free source guide from its builder:

```bash
.venv/bin/python scripts/build_notebook.py
```

Keep `RUN_FULL_EXPERIMENT = False` and do not embed local photo outputs in the tracked
guide. Completed local result notebooks can explain saved matching figures without
retraining, but remain ignored when they contain real photographs or overlays.
Do not reuse old 37-breed metrics as Cat/Dog results or hand-edit generated numbers.

Source images, real calibration patches, photo overlays, weights, checkpoints,
logits/responses, per-sample arrays, and local execution manifests remain ignored.
Only reviewed aggregate/synthetic exports are public candidates.

## Interpretation

Activation maps, input sensitivity, Grad-CAM, and occlusion are different views of
present model behavior, not an explanation of the original causal learning process.
No eye/ear/fur channel claim is verified by resemblance alone. Contributions must
not turn feature pictures or finite digital attacks into safety, physical robustness,
certification, or production-readiness claims.
