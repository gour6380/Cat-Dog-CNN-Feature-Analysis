# Cat/Dog CNN Features: What Responds, and Where?

[Guided notebook](notebooks/cat_dog_cnn_features.ipynb) ·
[Protocol](docs/PROTOCOL.md) · [Configuration](configs/experiment.yaml) ·
[Results](docs/results.md) · [Technical report](reports/technical_report.md) ·
[Source](src/) · [Tests](tests/) · [Contributing](CONTRIBUTING.md)

This project asks what low-, mid- and high-level ResNet-18 channels respond to—and
whether those pictures remain meaningful when the classifier itself has failed. Two
models share the same ImageNet initialization, data order and 15-epoch budget: standard
clean training versus pure PGD-5 training.

The main result is negative and useful: the PGD-trained model converged to the majority
dog rule. Its **67.80%** held-out validation accuracy sounds plausible, but macro accuracy
was **50%**, cat recall was **0%**, and all 295 validation images were predicted as dog.
The standard model reached **99.66%** overall and **99.75%** macro accuracy.

![Held-out validation comparison](docs/assets/validation-monitoring-comparison.png)

| Fixed epoch-15 checkpoint | Correct | Overall | Macro | Cat recall | Dog recall |
|---|---:|---:|---:|---:|---:|
| Standard | 294/295 | 99.66% | 99.75% | 100% | 99.5% |
| Pure PGD-5 | 200/295 | 67.80% | 50.00% | 0% | 100% |

These are clean metrics on a 295-image monitoring split that never selected a
checkpoint. This workflow does **not** measure official-test accuracy or adversarial
robust accuracy, so it cannot show that adversarial training generally fails. It shows
why overall accuracy and attractive internal visualizations cannot rescue a collapsed
comparison.

![Registered early, middle and late channel preferences](docs/assets/feature-preference-comparison.png)

The second figure contains activation-maximization probes—not pet photographs,
reconstructed training examples or verified anatomical detectors. The full public-safe
figures and their hashes are in [the results record](docs/results.md).

## Notebook workflow

The guided notebook has five compact groups:

1. **Setup and data:** environment summary, registered partition counts and a
   species-balance chart; data setup happens before inspection.
2. **Training:** separate standard and PGD-trained cells, each with live
   training-objective loss and clean validation-accuracy panels, with final cat/dog
   recall and prediction counts. Overall accuracy can hide always choosing the more
   common class. No training-image gallery or duplicate final live chart is displayed.
3. **Learned features:** five independent cells for image-to-layer walkthroughs,
   first-layer kernels, synthetic preferred inputs, strong real-image patches and
   clean activation/sensitivity maps.
4. **Prediction influence:** three independent cells for Grad-CAM, occlusion
   controls and response/randomized-weight diagnostics.
5. **Current-run results:** a feature-only visual report and read-only picture
   companion, without a separate evaluation prerequisite.

Every feature cell computes only its named family and required prerequisites.
Compact layer/activation previews show the first fixed cat and first fixed dog for
both models, not hand-picked attractive examples. All four anchors and full-size
figures remain linked. Extra response/change diagnostics are optional linked details;
randomized-weight and masking controls stay visible. Use
`display_features(config, section="stages", compact=False)` to display a full family.

Valid caches are reused; missing or stale evidence is never replaced with a
historical photograph, chart or result. Photographs and derived photo panels remain
ignored local artifacts.

## What the pictures show

| View | Meaning | Important limitation |
|---|---|---|
| Original input → stem/max-pool/layer1–4 → prediction | Spatial responses, dimensions, pooled features and class scores through the network | Maps are not reconstructed photographs |
| First-layer kernels | Actual learned RGB filter weights | Not an input-specific explanation |
| Synthetic channel preferences | Optimized stimuli that raise selected channel responses | Not a memory or real training example |
| Strong real patches and overlays | Where selected channels respond in clean reference images | Does not identify the causal learning event |
| Clean activation and gradient maps | Response location and local sensitivity | These answer different questions |
| Grad-CAM and occlusion | Class-score localization and score changes after masks | Different objectives; masks can create unfamiliar inputs |
| Randomized-weight checks | Whether an explanation depends on learned parameters | A changed control does not prove the explanation is correct |

In plain terms, a filter/channel is a small pattern tester. Its activation map shows
where it responds; an input gradient shows where tiny pixel changes could alter that
response. Later maps are coarse, and pooling condenses them into 512 numbers. The
models start with ImageNet pattern detectors, rather than learning from scratch.

A gray synthetic tile means the image optimizer failed to find a stronger pattern
in that trial—not that the channel is useless. Grad-CAM targets the true species even
when the prediction is wrong; a blank map means no positive map for that target under
this method. For occlusion, red/positive means masking that tile reduces the correct
species' score advantage; blue/negative means masking increases it. Flat maps can
make a randomized-map correlation undefined. None of these pictures proves how a
filter originally learned a pattern or establishes robustness.

Channels are selected using reference images, not attractive test pictures. Both
models use the same fixed anchors: two cats and two dogs. Do not infer an eye, ear
or fur detector from resemblance alone, or compare absolute response strength from
separately normalized panels. Theoretical receptive-field boxes describe possible
architectural support—not equal importance of all enclosed pixels.

## Data and matched training

Oxford-IIIT Pet has approximately 7,349 images across 37 breeds. The prediction task
is **cat=0, dog=1**; breed metadata is retained for stratification only.

- Preserve the official test partition. Split official trainval 80/20 within breed.
- From the original 2,944 training records, reserve a deterministic 10% per breed
  using `oxford-pets-validation-20260918`: approximately **2,649 fitting / 295
  validation**. The **736 reference / 3,669 test** records stay untouched.
- Both ResNet-18 arms start from identical pinned ImageNet weights and a new
  two-class head. Match IDs, sample order, augmentation, updates and schedule.
- Standard training uses clean cross-entropy. PGD training uses cross-entropy on
  PGD-5 inputs only: L∞ `4/255`, step `1/255`, random start and pixel clipping.
- Train at 224px, float32 native MPS, micro-batch 16 and two-step accumulation, with
  AdamW and tqdm. Use the configured final checkpoint, not the best validation one.
- Clean monitoring preserves weights, module modes, BatchNorm, gradients and
  training randomness. Its recorded history supplies the charts.

The [YAML configuration](configs/experiment.yaml) is editable. Changing a parameter
defines a new run identity; old checkpoints cannot silently become new results.
PGD training is retained as a comparison objective—not a claim of measured robustness.

## Run locally

Use native ARM64 CPython 3.13.15 and this project's existing environment:

```bash
./setup_venv.sh
.venv/bin/jupyter lab notebooks/cat_dog_cnn_features.ipynb
```

Setup uses `venv`, pip and exact [requirements.txt](requirements.txt), not uv.
Select this checkout's `.venv/bin/python` or its registered
`Python (Oxford Pets Adversarial Representations)` notebook kernel.

The distributed notebook uses `RUN_FULL_EXPERIMENT=False`, so opening it does not
silently retrain both models. Set it to `True` only when intentionally reproducing
training and feature generation. Builder regeneration preserves the existing flag and kernel.
It clears outputs by default; `scripts/build_notebook.py --preserve-outputs` is an
explicit presentation-refresh option that preserves owner outputs and metadata by
stable cell ID. Reviewing a saved notebook does not require clearing it or retraining.
To redraw the compact views from verified saved measurements without running models,
use `.venv/bin/python scripts/refresh_notebook_views.py --config configs/experiment.yaml`.
The existing run flag is preserved; reopening the refreshed notebook needs no execution.

```text
.venv/bin/python src/cli.py setup      --config configs/experiment.yaml
.venv/bin/python src/cli.py train      --config configs/experiment.yaml --arm standard
.venv/bin/python src/cli.py train      --config configs/experiment.yaml --arm adversarial
.venv/bin/python src/cli.py represent  --config configs/experiment.yaml --device mps
.venv/bin/python src/cli.py report     --config configs/experiment.yaml
.venv/bin/python src/cli.py reproduce  --config configs/experiment.yaml --device mps
```

Run a single feature family with `represent --section
stages|kernels|synthetic|real_patches|activations|gradcam|occlusion|diagnostics`.
Omitting the selector runs the full feature workflow. There is no separate attack
evaluation step in the active reproduction path.

## Evidence and reports

Epoch checkpoints, histories, partial feature receipts and figure hashes record
actual progress. The current 15-epoch validation outcome and all eight feature families
are exported in [machine-readable form](docs/results.json). Current-run
read-only notebooks and reports are generated locally under `reports/generated/`;
the main outputs are `technical-report.md` and `cat_dog_feature_results.ipynb`.
The read-only companion groups current feature pictures by family and model,
with links to additional local panels. These outputs explain saved evidence without
retraining. A completed report requires the
registered feature families for both current model checkpoints, not legacy
evaluation files.

Data, weights, checkpoints, logs, per-sample arrays and photo panels are ignored.
Only reviewed synthetic and aggregate exports under `docs/assets/` are shared. Generated
source guides are output-free by default; owner execution outputs stay local during
review. No remote repository creation, publishing or
profile updates are part of this workflow.

Native MPS fallback remains disabled. Actual OOM, non-finite values, invalid attacks
or provenance failures stop the run; settings are not silently downgraded.

## Verify changes

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
.venv/bin/python scripts/check_repository.py
```

Development tests use bounded synthetic inputs; they do not launch full training.
Methods and interpretation boundaries are in the [protocol](docs/PROTOCOL.md).
