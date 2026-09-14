# Adversarial Representation Drift in Fine-Grained Pet Recognition

This local, self-contained experiment compares two matched ImageNet-initialized
ResNet-18 models across all 37 Oxford-IIIT Pet breeds. One model receives standard
fine-tuning and the other receives PGD-5 adversarial fine-tuning. The primary result
is based on original 512-dimensional penultimate features, not on a visually pleasing
projection.

The claim boundary is deliberately narrow: the study can support representation
retention under a fixed digital `L∞` attack. It cannot establish physical robustness,
safe pet recognition, out-of-distribution detection, or production readiness.

## Locked protocol

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

The full registered protocol and state live in
[`Instructions/checklist.md`](../../Instructions/checklist.md). The machine-readable
copy is [`configs/experiment.yaml`](configs/experiment.yaml).

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
or restarts. Unsafe memory pressure, MPS OOM, invalid attacks, and non-finite values
stop the run while preserving diagnostic state.

## Evidence lifecycle

`setup` records the official split and hashes. `preflight` runs data, attack, parity,
gradient, BatchNorm, determinism, and memory checks. Each epoch checkpoint is written
atomically and is resumable only after provenance validation. `evaluate` preserves
aligned local logits/features; `represent` calculates geometry and creates projection
figures; `report` produces the technical report, long-form report, Sunday draft, error
taxonomy, and a hash-complete local release manifest.

No remote is configured and no public action is part of these commands.
