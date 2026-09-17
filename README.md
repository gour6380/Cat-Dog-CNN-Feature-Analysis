# Cat/Dog CNN Features: What Responds, and Where?

[Protocol](docs/PROTOCOL.md) · [Configuration](configs/experiment.yaml) ·
[Guided notebook](notebooks/cat_dog_cnn_features.ipynb) · [Source](src/) ·
[Tests](tests/) · [GitHub handoff](docs/github.md)

This study visualizes learned CNN features, rather than two-dimensional embeddings.
It asks what patterns a ResNet-18 channel responds to, where those patterns occur in
real pet images, and which image regions affect a Cat/Dog prediction. A matched
standard/PGD-trained comparison adds a bounded digital-attack stress test.

**Task:** Oxford-IIIT Pet species classification: `cat=0`, `dog=1`. The original
37-breed annotations remain useful for stratifying the data split and describing EDA;
the network has a two-class head, not a breed prediction head.

**Status: fresh 15-epoch binary experiment and illustrated feature analysis complete.**
The earlier 37-breed PCA/t-SNE/UMAP study is superseded, not
relabeled. The obsolete `oxford_pets_results_explained.ipynb` is removed from the
active workflow. All current evidence must match the new two-class checkpoints.

The standard model correctly classifies **3,646/3,669 clean test images (99.37%)**.
The PGD-trained comparison **fails species recognition**: every clean-test prediction
is dog, so its 67.76% accuracy is just the dog proportion. Cat recall is 0%, dog recall
100%, and macro accuracy 50%. Its 50% survival on the balanced 200-image attack set
is not useful robust Cat/Dog recognition. Hidden features and scores still vary;
single-class decisions do not imply constant hidden features. The failure mechanism
is unproven and was not hidden by changing the protocol or selecting another checkpoint.
Use the standard model for the main learned-feature walkthrough, with the failed
PGD arm retained as a cautionary comparison.

[Measured results and charts](docs/results.md) ·
[Technical report](reports/technical_report.md) ·
[Long-form explanation](reports/long_form_report.md)

![Low/mid/high synthetic channel preferences](docs/assets/standard-activation_maximization-standard-synthetic-atlas.png)

These are optimized channel preferences, not recovered photographs or verified
object-part detectors. The local illustrated notebook pairs them with real input
regions and prediction-influence maps.

Four PGD-arm tiles stayed gray with zero response gain (`layer2` channel73 and
`layer4` channels491/418/61). These are unsuccessful fixed-start synthetic probes,
not dead channels: the same channels have positive real-image responses. Their
original choices/seeds are retained rather than retried for visual appeal.

## What the pictures mean

| View | Question it answers | What it cannot establish |
|---|---|---|
| First-layer kernels | What RGB/edge patterns are encoded by `conv1` weights? | That a channel detects a named object part |
| Activation-maximization images | What synthetic input increases one selected channel's response? | A real example, a decoded memory, or a training-image reconstruction |
| Top real calibration patches | Which clean images/locations strongly activate that channel? | Why the original training process learned it |
| Activation maps and input-gradient sensitivity | Where does a channel respond, and where can small input changes affect it? | A causal or complete explanation |
| Grad-CAM and occlusion | Where is the true-species logit localized, and how does masking change its margin over the other species? | Identical-score validation, guaranteed semantic understanding, or safety |
| Randomized-weight control | Does an explanation change when learned weights are destroyed? | That a surviving explanation is correct |
| Aggregate layer-response charts | How do responses change across depth, species, and input shift? | Superiority inferred from separately normalized pictures |

The low/mid/high views use `network.relu`, `network.layer2`, and `network.layer4`.
Early channels often respond to colors or edges; deeper channels can combine larger
patterns. Those are interpretive tendencies, not labels assigned to our channels.
We do not claim an “eye”, “ear”, or “fur” detector without independent validation.
Grad-CAM targets the true-species logit; occlusion targets the true-species-vs-other
logit margin. These related diagnostics use distinct scalar objectives and keep the
true target fixed even when a prediction is wrong. Agreement is not exact validation
of the same scalar attribution.
Late-layer theoretical receptive fields can exceed the whole 224-pixel input: a box
is architectural support, not proof that every enclosed pixel was used equally.

The Lee et al. 2009 visualization is inspiration for the low-to-high feature
walkthrough. This code uses a discriminative ResNet, not a convolutional deep belief
network, and does not reproduce that paper's generative learning method.

## Protocol at a glance

- Preserve the official test partition. Split official `trainval` 80/20 within each
  of the 37 breeds using a recorded filename-hash seed; labels for learning are species.
- Start both two-class ResNet-18 models from identical pinned ImageNet weights and
  an identical new head. Fully fine-tune for the configured 15 epochs at 224 px,
  float32, native PyTorch MPS, micro-batch 16 and two-step accumulation.
- Standard training uses clean cross-entropy. The adversarial arm uses random-start
  PGD-5 at `L∞ 4/255`, step `1/255`. Initialization, sample order, augmentation,
  optimizer-update count, and schedule are matched.
- Compare fixed final-epoch checkpoints. Evaluate clean accuracy, full-test
  corruptions, and FGSM/PGD-20 with five restarts on 100 fixed test images per species.
- Fit temperature and a 90%-coverage confidence policy using clean calibration only.
- Select channels using clean calibration, not attractive test pictures. Inspect
  fixed test anchors: two cats and two dogs. Compare both arms on the same images.
- Save identities, logits, responses, explanation settings, timing, and hashes.
  Separate quantitative observations from visual interpretation.

All experiment parameters live in [the YAML configuration](configs/experiment.yaml).
Valid parameters are editable: there are no duplicate hard-coded epoch or memory
minimum gates. A changed configuration defines a new run identity; old checkpoints
are not silently relabeled. Commands do not automatically downgrade requested settings.

## Environment and commands

Use native ARM64 CPython 3.13.15 on Apple Silicon:

```bash
./setup_venv.sh
```

This uses standard-library `venv`, pip, and the fully pinned `requirements.txt`, not
uv. Use `./setup_venv.sh --offline` with an existing ignored wheel cache. Select this
checkout's `.venv/bin/python` in the notebook's environment picker.

```text
python src/cli.py setup      --config configs/experiment.yaml
python src/cli.py preflight  --config configs/experiment.yaml --device mps
python src/cli.py train      --config configs/experiment.yaml --arm standard|adversarial
python src/cli.py evaluate   --config configs/experiment.yaml --device mps
python src/cli.py represent  --config configs/experiment.yaml --device mps
python src/cli.py report     --config configs/experiment.yaml
python src/cli.py reproduce  --config configs/experiment.yaml --device mps
```

Use `.venv/bin/python` unless the environment is activated. Every command accepts
`--no-progress`; PyTorch training uses tqdm without changing the saved evidence.
Silent MPS operation fallback is disabled. Memory is telemetry, not a minimum-memory
gate; actual OOM, invalid attacks, non-finite values, and provenance errors stop a run
and preserve diagnostics.

## Notebook walkthrough

Open the [output-free guided notebook](notebooks/cat_dog_cnn_features.ipynb):

```bash
.venv/bin/jupyter lab notebooks/cat_dog_cnn_features.ipynb
```

It covers species/breed EDA, model depth, matched training, attacks, channel selection,
synthetic features, real receptive-field patches, activation maps, sensitivity,
Grad-CAM, occlusion, sanity controls, quantitative charts, and report inventory.
It calls the same modules as the CLI; it does not reimplement training in cells.
After a matching public export, regenerating the guide also adds a compact static
gallery of the workflow, both synthetic feature atlases, accuracy context, and selected
kernels when available. Those repository-local Markdown images can be read on GitHub
without execution; their configuration, kind, path, and asset hashes are checked first.
They contain no real pet photographs. Use the builder's `--no-public-preview` option
to omit this gallery; pending or stale evidence adds no gallery.

`RUN_FULL_EXPERIMENT = False` is the default. Scientific stages are skipped in safe
mode; available matching figures may be read from their saved manifest. Set the
switch to `True` only to intentionally run the new study. A completed local notebook
or picture-rich result companion contains pet photographs and stays ignored; it is
not a public source notebook. A missing figure remains missing rather than borrowing
an old breed result.

For the completed local study, open `reports/generated/cat_dog_feature_results.ipynb`.
It embeds all 32 saved figures, including real top patches, cat/dog channel walkthroughs,
Grad-CAM and occlusion. It is entirely Markdown: no code cells, no execution required.
The GitHub guide instead includes the six-image photograph-free static gallery.

```bash
.venv/bin/jupyter lab reports/generated/cat_dog_feature_results.ipynb
```

```bash
# Regenerate the output-free source guide; no model execution.
.venv/bin/python scripts/build_notebook.py

# Explicit headless full run; leave out --full for safe inspection.
.venv/bin/python src/notebook_runner.py --config configs/experiment.yaml --device mps --full
```

`results/generated/feature_visualizations.json` records the current feature figure
paths, arm results, limitations, configuration hash, and completion time. The guide
loads those paths dynamically instead of assuming stale projection filenames.
Before displaying a saved image, it verifies the recorded SHA-256; missing or altered
images are explicitly marked stale and are not displayed.

## Evidence and sharing

Dataset files, ImageNet weights, checkpoints, photographs, per-sample arrays, execution
logs/manifests, photo-containing figures, and rendered results notebooks remain local
and ignored. Reviewed synthetic feature images and aggregate charts can be exported
for a repository, with the protocol and checkpoint identity attached. Do not publish
photo patches or overlays without a separate data-license/attribution review.

```text
configs/          Editable machine-readable protocol
docs/             Protocol, handoff, public-safe aggregate/synthetic exports
notebooks/        Output-free Cat/Dog feature walkthrough
scripts/          Notebook and result-presentation builders; repository checks
src/              Typed data, model, training, attacks, feature visualization, reports
tests/            Correctness, provenance, MPS, progress, explanation, notebook gates
reports/          Evidence-backed reports, or explicit pending state
requirements.txt  Exact Python dependency pins
setup_venv.sh     Isolated venv/pip setup and project kernel registration
```

The README is self-contained and does not depend on private program files outside
this checkout. The strongest eligible conclusion concerns this model pair's measured
responses and finite digital-attack behavior, not physical robustness or safer animals.

## Verification and release

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src
.venv/bin/python scripts/check_repository.py
```

See [contribution guidance](CONTRIBUTING.md), [third-party notices](THIRD_PARTY_NOTICES.md),
and the [local GitHub handoff](docs/github.md). Original code/documentation use
the [MIT license](LICENSE). No command here creates a remote, pushes, or publishes.
