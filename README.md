# Adversarial Representation Drift in Fine-Grained Pet Recognition

[Results](docs/results.md) ·
[Visual explanation](notebooks/oxford_pets_results_explained.ipynb) ·
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

**Status:** the matched 15-epoch experiment completed on PyTorch MPS. The registered
representation hypothesis passed. PGD-5 fine-tuning preserved substantially more
penultimate structure and improved finite-attack robust accuracy, while reducing clean
and corruption accuracy. This is one model pair and one split/seed family.

## Completed results

| Metric | Standard | PGD-trained |
|---|---:|---:|
| Clean accuracy, full official test | 89.15% | 72.91% |
| FGSM robust accuracy, 740-image subset | 2.84% | 31.08% |
| PGD-20×5 robust accuracy, 740-image subset | 0.00% | 20.68% |
| PGD attack success among clean-correct images | 100.00% | 71.51% |
| Median clean-to-PGD cosine feature drift | 0.4531 | 0.0378 |
| PGD five-NN breed retention | 0.11% | 25.62% |
| PGD five-NN accuracy | 0.00% | 27.57% |
| Clean-to-PGD linear CKA | 0.1669 | 0.8627 |

The adversarial-minus-standard median-drift difference was `-0.415384`, with a
class-stratified 95% bootstrap interval of `[-0.423403, -0.408943]`. The neighbour-
retention difference was `+25.51` percentage points, interval `[23.54, 27.62]`. Both
registered signs passed. These intervals cover paired test-sample uncertainty, not
variation from retraining.

| Paired feature drift | Per-breed neighbour retention |
|---|---|
| ![Distribution of paired cosine feature drift](docs/assets/cosine-drift.png) | ![Five-nearest-neighbour breed retention by breed](docs/assets/knn-retention-by-breed.png) |

The standard model had higher absolute accuracy on every registered noise, blur,
brightness, and contrast condition. Clean-fitted 90%-coverage confidence policies also
moved outside their registered coverage/risk tolerances after at least one shift for both
arms. Lower clean ECE for the adversarial model did not compensate for its higher task
error or selective risk.

Read the [complete aggregate result page](docs/results.md), the
[machine-readable values](docs/results.json), or the
[self-contained visual explanation notebook](notebooks/oxford_pets_results_explained.ipynb).
The notebook has no code cells: it embeds thirteen aggregate EDA, geometry, PCA, t-SNE,
UMAP, and boundary figures and can be read directly on GitHub without running anything.

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
`./setup_venv.sh --offline`. Dataset files, weights, checkpoints, photographs, logits,
features, per-sample arrays, and run manifests remain local and Git-ignored. The reviewed
aggregate result page, result JSON, figures, reports, and visual notebook are shareable
snapshots rather than substitutes for an independent reproduction.

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

### Visual results companion

After a matching experiment completes, `scripts/build_results_notebook.py` creates
`notebooks/oxford_pets_results_explained.ipynb`: a read-only explanation with thirteen
embedded aggregate figures. Three cover EDA and ten cover geometry, PCA, t-SNE, UMAP,
and boundary views. The repository includes the reviewed 15-epoch snapshot; rebuilding it
requires matching local evidence and does not run a model or an attack.

```bash
.venv/bin/python scripts/build_results_notebook.py --force
```

The builder verifies the configuration identity plus saved result, training, release,
EDA, and figure hashes before embedding anything. Run it only after the notebook's report
stage completes. EDA reads registered split metadata and source-image headers only. It
includes no original pet photographs.

The GitHub-facing aggregate page, JSON, selected figures, and tracked reports are also
presentation-only exports from the same evidence:

```bash
.venv/bin/python scripts/build_public_results.py
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
docs/             Protocol, aggregate results, release handoff, and public-safe figures
notebooks/        Guided experiment notebook and visual results companion
scripts/          Deterministic notebook and results-presentation generators
src/              Typed data, model, attack, training, evaluation, and reporting code
tests/            Correctness, provenance, CLI, MPS, progress, and notebook gates
reports/          Reviewed technical and long-form reports
requirements.txt  Fully pinned Python 3.13 dependency environment
setup_venv.sh     Isolated venv/pip setup and project kernel registration
```

## Verification and GitHub handoff

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src
.venv/bin/python scripts/check_repository.py
```

The final command performs a read-only review of tracked and unignored candidate files:
private absolute paths, common credential forms, oversized files, forbidden experiment
binaries, notebook state, and local Markdown links. See the
[GitHub handoff guide](docs/github.md) and [contribution guide](CONTRIBUTING.md).

Original project code and documentation use the [MIT license](LICENSE). The Oxford-IIIT
Pet data and downloaded model weights are not distributed; see
[third-party notices](THIRD_PARTY_NOTICES.md). No remote is configured, and none of these
commands creates a repository, pushes code, publishes a release, or authorizes a public
claim.
