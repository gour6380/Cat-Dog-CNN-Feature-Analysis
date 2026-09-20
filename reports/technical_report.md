# Cat/Dog CNN feature learning: technical report

## Question

What do low-, middle- and high-level ResNet-18 channels respond to, where do those
responses occur in clean pet images, and which regions influence a Cat/Dog score? The
study compares standard clean training with pure PGD-5 training under the same
initialization, fitting records, sample order, augmentations, update count and schedule.

## Protocol

- Oxford-IIIT Pet, with breed retained for stratification and `cat=0`, `dog=1` as targets.
- 2,649 fitting images, 295 held-out validation images, 736 reference images and the
  untouched 3,669-image official test partition.
- Two fully fine-tuned ImageNet-initialized ResNet-18 models, fixed epoch 15.
- Float32 PyTorch MPS on an M2 Pro; micro-batch 16 with two-step accumulation; AdamW.
- Standard objective: clean augmented cross-entropy.
- Adversarial objective: pure untargeted PGD-5 cross-entropy, L∞ `4/255`, step `1/255`,
  random start and clipping in `[0,1]`.
- Identical initialization and 1,245 optimizer updates per arm. Validation never selected
  or stopped a checkpoint.

The standard run took 14.67 minutes and the PGD run 30.95 minutes in their recorded
invocations. Machine runtime is not a focused-work estimate.

## Main result: the comparison failed

| Clean validation at epoch 15 | Standard | PGD-trained |
|---|---:|---:|
| Correct | 294/295 | 200/295 |
| Overall accuracy | 99.66% | 67.80% |
| Macro accuracy | 99.75% | 50.00% |
| Cat recall | 100% | 0% |
| Dog recall | 99.5% | 100% |
| Predictions: cat / dog | 96 / 199 | 0 / 295 |

The PGD model predicted every validation image as dog. Its final fitting-objective
accuracy, 67.72%, exactly matches the dog proportion in the fitting partition; its loss,
0.6307 nats, is also close to the class-prior entropy of 0.6289 nats. The evidence is
consistent with a majority-class rule rather than balanced species discrimination.

This is not an adversarial-robustness result. The active workflow did not evaluate the
fixed checkpoints under post-training attacks, corruptions or the official test set.
The scoped conclusion is that this exact pure-PGD-5 recipe failed on the imbalanced
binary task.

## Feature evidence

All eight registered feature families completed for both checkpoints: stage
walkthroughs, actual first-layer kernels, activation-maximization stimuli, strongest
reference patches, clean activation/input-gradient maps, Grad-CAM, occlusion and
randomized-weight diagnostics. Channel selection used 64 balanced reference images;
four hash-selected test anchors supplied descriptive walkthroughs.

Observed diagnostics:

- The standard model predicted all four anchors correctly. The PGD model predicted dog
  for all four, including both cat anchors.
- All six PGD-selected late-layer channels had higher mean dog than cat response on the
  balanced reference set. Standard late-layer selections separated in both directions.
  This is consistent with the classifier failure, not proof of its cause.
- Four of 18 PGD synthetic-optimization trials had zero measured gain; all four are
  retained. They do not establish dead channels because those channels responded to
  real reference images.
- Masking the standard model's top Grad-CAM tiles reduced the true-class margin more
  than random equal-area masks for all four anchors. PGD effects were small or
  inconsistent. Four anchors cannot support a population claim.

## Interpretation

A feature picture answers a narrower question than classification metrics. Kernels show
weights; optimized stimuli show patterns that raise one channel; activation maps show
where a channel responds; gradients show local sensitivity; Grad-CAM shows coarse
class-score localization; occlusion measures the effect of an artificial mask. None
shows which training image taught the feature, and agreement between methods is not
causal proof.

The important engineering lesson is earlier in the pipeline: overall accuracy alone
would have made the collapsed PGD model look moderately successful. Macro accuracy,
per-species recall and prediction counts exposed the failure immediately. Internal
visualizations then helped characterize a failed classifier; they did not repair it.

## Limitations

- One model pair, split and seed family.
- ImageNet initialization means many filters were inherited rather than learned from
  Oxford-IIIT Pet alone.
- The validation set has 95 cats and 200 dogs; it is not the official test partition.
- The four walkthrough anchors and 64 reference images are descriptive samples.
- Independently selected/normalized channels cannot be compared as semantic matches.
- No attack-test accuracy, certified robustness, physical robustness or safe-use claim.
- Dataset photographs and derived photo panels remain local under the dataset's license;
  repository figures are synthetic or aggregate only.

## Reproduce and verify

```bash
.venv/bin/python src/cli.py reproduce --config configs/experiment.yaml --device mps
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src
.venv/bin/python scripts/check_repository.py
```

Configuration SHA-256: `61fda0dad3180c5ffb9fd1b42cdd27815fed1e58ea433bbc5b0d431394b58e34`.
Standard checkpoint SHA-256: `6870b521f0f82f88609c5833da27d8a8a12c60d9955863a653ceac375eab8610`.
PGD checkpoint SHA-256: `4e7a736b951f3b338f150d67f4d50d8a1cc9e0d99e8512b137160d55530d9cf6`.

[Machine-readable public results](../docs/results.json) ·
[Protocol](../docs/PROTOCOL.md) · [Guided notebook](../notebooks/cat_dog_cnn_features.ipynb)
