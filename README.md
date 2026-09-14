# Adversarial Representation Drift in Fine-Grained Pet Recognition

[Protocol](docs/PROTOCOL.md) · [Configuration](configs/experiment.yaml) ·
[Guided notebook](notebooks/oxford_pets_adversarial_representations.ipynb) ·
[Source](src/) · [Tests](tests/)

This local, self-contained experiment compares two matched ImageNet-initialized
ResNet-18 models across all 37 Oxford-IIIT Pet breeds. One model receives standard
fine-tuning and the other receives PGD-5 adversarial fine-tuning. The primary result
is based on original 512-dimensional penultimate features, not on a visually pleasing
projection.

The claim boundary is deliberately narrow: the study can support representation
retention under a fixed digital `L∞` attack. It cannot establish physical robustness,
safe pet recognition, out-of-distribution detection, or production readiness.

## Reference protocol

- Keep the official test partition intact. Split official `trainval` within every
  breed into deterministic 80% training and 20% calibration partitions.
- Initialize both 37-class models from identical `IMAGENET1K_V1` tensors and an
  identical new head. Train exactly 15 epochs at 224 px in float32 on MPS.
- Standard uses clean cross-entropy. Adversarial uses random-start PGD-5 with
  `epsilon=4/255` and `step=1/255`. Both arms share sample order, deterministic
  augmentation, update count, and learning-rate schedule.
- Evaluate FGSM and PGD-20 with five restarts on exactly 20 hash-selected test
  samples per breed. Evaluate all registered corruptions on the full test partition.
- Fit scalar temperature and a 90%-coverage threshold on clean calibration data only.
- Treat PCA, t-SNE, and UMAP as explanatory views. Quantitative conclusions use the
  original features and class-stratified bootstrap intervals.

The complete repository-local protocol is [`docs/PROTOCOL.md`](docs/PROTOCOL.md), and
the machine-readable copy is [`configs/experiment.yaml`](configs/experiment.yaml).
The project README does not rely on private program files outside this checkout.

The YAML file is the source of truth, not a second table of hard-coded values in Python.
You may edit epochs, batch sizes, attack settings, coverage, projection settings, and
other valid experiment parameters. Every edit produces a different configuration hash,
so checkpoints and computed results from another configuration are not reused silently.
Changes to dataset splitting or model initialization require `setup` to be run again;
ordinary training and evaluation parameter changes do not invalidate the registered data
split. Commands never change a requested value automatically.

## Environment

The supported runtime is native ARM64 CPython 3.13.15. From this directory:

```bash
./setup_venv.sh
```

The setup script uses the standard-library `venv` module and pip with the fully pinned
`requirements.txt`; uv is not used. To install from the ignored local wheel cache, use
`./setup_venv.sh --offline`. Dataset, weights, checkpoints, photographs, per-sample
arrays, and generated outputs are local and Git-ignored.

## Typed command interface

```text
python src/cli.py setup      --config configs/experiment.yaml
python src/cli.py preflight  --config configs/experiment.yaml --device mps
python src/cli.py train      --config configs/experiment.yaml --arm standard|adversarial
python src/cli.py evaluate   --config configs/experiment.yaml --device mps
python src/cli.py represent  --config configs/experiment.yaml
python src/cli.py report     --config configs/experiment.yaml
python src/cli.py reproduce  --config configs/experiment.yaml --device mps
```

Use `.venv/bin/python` in place of `python` unless the environment is activated.
Every command accepts `--no-progress`. Training shows nested tqdm epoch and batch bars
plus a durable epoch summary; setup, preflight, evaluation, representation analysis, and
reporting print explicit stage status. Disabling progress changes only rendering, never
the saved numerical evidence.

`reproduce` reuses only artifacts whose recorded configuration, data, model, and
initialization hashes match. It never silently reduces resolution, epochs, PGD steps,
or restarts. Memory use is recorded as telemetry but does not block execution. MPS OOM,
invalid attacks, and non-finite values still stop the run while preserving diagnostic state.

## Guided notebook

[`oxford_pets_adversarial_representations.ipynb`](notebooks/oxford_pets_adversarial_representations.ipynb)
is the output-free guided interface over the same typed modules used by the CLI. It
covers the question and claim boundary, configuration, environment, registered split,
architecture, MPS preflight, both training arms, evaluation, representation analysis,
reports, and the artifact inventory. It does not duplicate model, attack, or metric
implementations inside notebook cells.

Open it with this project's registered kernel:

```bash
.venv/bin/jupyter lab notebooks/oxford_pets_adversarial_representations.ipynb
```

The setup cell rejects an interpreter outside this checkout's `.venv`.
`RUN_FULL_EXPERIMENT = False` is the default, so preflight, training, attacks,
representation fitting, and reporting are skipped during safe inspection. Change it to
`True`, rerun the setup cell, and then run every cell in order only when intentionally
starting the configured experiment.

For headless execution with the exact calling interpreter:

```bash
# Safe inspection; long scientific stages remain disabled.
.venv/bin/python src/notebook_runner.py --config configs/experiment.yaml --device mps

# Explicitly enable preflight and the complete scientific pipeline.
.venv/bin/python src/notebook_runner.py --config configs/experiment.yaml --device mps --full
```

The headless safe mode overrides an edited notebook switch. Executed copies and their
manifests are saved under ignored local `artifacts/notebooks/`; the source notebook stays
output-free. Regenerate the canonical guide after editing its builder with:

```bash
.venv/bin/python scripts/build_notebook.py
```

## Evidence lifecycle

`setup` records the official split and hashes. `preflight` runs data, attack, parity,
gradient, BatchNorm, and determinism checks while recording memory telemetry. Each epoch
checkpoint is written atomically and is resumable only after provenance validation.
`evaluate` preserves aligned local logits/features; `represent` calculates geometry and
creates projection figures; `report` produces the technical report, long-form report,
Sunday draft, error taxonomy, and a hash-complete local release manifest.

## Repository map

```text
configs/          Locked machine-readable experiment settings
docs/             Protocol, controls, hypotheses, and limitations
notebooks/        Canonical output-free guided experiment notebook
scripts/          Deterministic notebook generator
src/              Typed data, model, attack, training, evaluation, and reporting code
tests/            Correctness, provenance, CLI, MPS, progress, and notebook gates
requirements.txt  Fully pinned Python 3.13 dependency environment
setup_venv.sh     Isolated venv/pip setup and project kernel registration
```

No remote is configured and no public action is part of these commands.
