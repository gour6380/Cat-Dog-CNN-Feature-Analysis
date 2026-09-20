# Cat/Dog learned-feature walkthrough protocol

Revised 19 September 2026. The current registered run completed both fixed
**15-epoch** checkpoints and all eight feature families. The pure-PGD comparison
collapsed to the dog-majority rule; it is preserved as a negative result rather than
relabeled as robust recognition.

## Question and scope

What patterns activate low-, mid- and high-level channels in matched standard and
PGD-trained ResNet-18 Cat/Dog classifiers? Where do those responses occur in clean
images, and which regions influence a species prediction?

The active workflow is training → learned features → prediction influence →
feature-only reports. It does not run breed classification, PCA/t-SNE/UMAP,
FGSM/PGD test evaluation, corruption sweeps, temperature scaling, confidence
thresholds or risk–coverage analysis. Legacy helper modules do not define the
active notebook or reproduction path.

## Dataset and partitions

Oxford-IIIT Pet contains approximately 7,349 images, 37 breeds (12 cat / 25 dog
breeds), official annotations/splits and a CC BY-SA 4.0 license. The supervised
targets are species: `cat=0`, `dog=1`. Preserve breed metadata for splitting only.

Preserve the official test partition. Deterministically divide official trainval
80/20 within breed using the recorded filename-hash seed. From the resulting
2,944 original training records, reserve 10% within each breed using seed
`oxford-pets-validation-20260918`:

| Partition | Expected count | Use |
|---|---:|---|
| Fitting | 2,649 | Parameter updates |
| Validation | 295 | Clean epoch monitoring only |
| Reference (registered calibration split) | 736 | Channel selection and real response patches |
| Official test | 3,669 | Fixed clean walkthrough anchors |

Register actual IDs/counts and disjointness. Both arms use identical fitting and
validation records. Validation does not select checkpoints or stop training. The
reference split is not used to fit a confidence policy in this workflow. Select
fixed test anchors deterministically: two cats and two dogs. Never replace a
channel or anchor because its picture is unflattering.

Setup prepares dataset, initialization and monitoring manifests before data
inspection. The notebook shows a compact partition table and species-balance
chart, not full manifest/configuration dumps.

## Matched training

Initialize both Torchvision ResNet-18 models from identical pinned
`IMAGENET1K_V1` weights and the same new two-class head. Fully fine-tune all layers.
Use 224×224 RGB, deterministic random resized crops/flips for fitting and
resize/center crop for clean monitoring/features. Normalize inside the model;
PGD operates on raw `[0,1]` inputs.

Keep float32 native MPS, micro-batch 16, two-step accumulation, AdamW, backbone LR
`1e-4`, head LR `1e-3`, weight decay `1e-4`, one-epoch warm-up and cosine decay.
The configuration controls epoch count. Match sample order, augmentation, optimizer
updates and schedule across arms; use the fixed configured final checkpoint.

- Standard objective: cross-entropy on clean augmented inputs.
- Adversarial objective: cross-entropy on PGD-5 inputs only, untargeted L∞ epsilon
  `4/255`, step `1/255`, uniform random start, projection and pixel clipping.

After each completed epoch, record clean fitting/validation metrics in durable
checkpoint history. Monitoring uses evaluation mode and restores every module
mode, BatchNorm state, gradients and caller RNG streams. It does not alter weights,
optimizer state or training order. The notebook refreshes only two panels:
training-objective loss and clean validation accuracy. Label adversarial-objective
loss distinctly; it is not clean loss. Missing epochs are gaps, not inferred points.
Resumed displays replay recorded history. There is no early stopping or
validation-based checkpoint selection.

## Observed fixed-checkpoint outcome

The standard model classified 294/295 held-out validation images correctly
(99.66% overall, 99.75% macro accuracy). The pure-PGD model classified 200/295
correctly (67.80% overall), but predicted every sample as dog: 0% cat recall,
100% dog recall and 50% macro accuracy. Its overall score equals the split's dog
majority baseline.

This monitoring partition was excluded from fitting and never selected a checkpoint,
but it is not the official test set. The active workflow does not run a post-training
attack benchmark, so the result does not establish robust accuracy or show that
adversarial training generally fails. It establishes failure of this exact pure-PGD-5,
class-imbalanced recipe and motivates class-aware monitoring before interpreting its
feature pictures.

## Eight independent feature families

Use typed `represent --section` selectors. Each family computes only its own
work and prerequisites, saves atomic partial progress and reuses identity-matched
caches. The notebook separates learned features from prediction influence:

| Group | Selector | Evidence |
|---|---|---|
| Learned features | `stages` | Original photo, actual input, stem, max-pool, layer1–4, pooled 512D features, scores/prediction |
| Learned features | `kernels` | Actual RGB conv1 weights |
| Learned features | `synthetic` | Synthetic preferred stimuli for selected low/mid/high channels |
| Learned features | `real_patches` | Strong reference-image responses, patches and receptive-field boxes |
| Learned features | `activations` | Clean fixed-anchor spatial maps, overlays and input-gradient sensitivity |
| Prediction influence | `gradcam` | Fixed true-species-logit localization |
| Prediction influence | `occlusion` | True-species-vs-other margin changes and equal-area region controls |
| Prediction influence | `diagnostics` | Response charts and randomized-weight sanity checks |

Select channels from clean reference images rather than test examples. Record
layer/channel IDs, response measurements, anchor filenames, seeds, normalization
and probe settings. Retain failed synthetic optimizations rather than retrying for
attractive pictures. The activations family is clean-only: it does not generate
feature-comparison attacks.

Show selected channel maps, dimensions and overlays. These are channel responses,
not reconstructed photographs or verified anatomical detectors. Grad-CAM and
occlusion use different scalar objectives; keep the true target fixed even when
the model predicts incorrectly. A heatmap is not a causal learning explanation.

## Provenance, reports and completion

Record configuration, data, initialization, scientific source and checkpoint
hashes, monitoring protocol, sample IDs, environment, timings, feature methods and
artifact hashes. Resume only matching runs. Valid parameter changes create a new
identity; no duplicate hard-coded epoch rule or silent settings downgrade is used.

Feature manifests remain partial until all required families for both models are
present and verified. Reports use matching training/feature evidence only; no
evaluation, confidence-policy or corruption artifact is a prerequisite. Missing
evidence is explicitly unavailable, never populated with historical images.
The visual report and read-only results notebook are generated from registered
current-run figures as `reports/generated/technical-report.md` and
`reports/generated/cat_dog_feature_results.ipynb`. The companion groups pictures
by family/model with links to additional local panels, and can be read without
model inference or retraining.

Photographs, real patches/overlays, dataset files, weights, checkpoints,
per-sample arrays and local execution manifests stay ignored. Reviewed
synthetic/aggregate exports may be prepared separately; execution does not publish
anything. Keep tracked notebook outputs, execution counts and attachments clear.
Builder regeneration preserves the owner's run flag and kernel metadata.

Native MPS fallback is disabled. Record memory as telemetry. Stop on actual OOM,
non-finite values, invalid training attacks or provenance failures; preserve the
failure rather than changing the experiment silently.

## Interpretation limits

- Kernel weights, synthetic preferences, activation maps, gradients and class
  influence diagnose different aspects of present model behavior.
- Resemblance does not verify an eye/ear/fur detector or identify the training
  photograph that caused a feature to develop.
- Separately normalized panels cannot establish absolute response superiority.
  A theoretical receptive field is possible support, not equal pixel importance.
- Occlusion can produce out-of-distribution inputs; randomized controls test
  dependence on learned weights, not explanation correctness.
- ImageNet pretraining, one seed/split family, selected anchors, short training
  and method-specific distortions constrain generalization.
- PGD training is an objective, not measured robustness. No safety, physical
  robustness, certification or production-readiness claim follows from these images.

## References

[Official Oxford-IIIT Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/),
[Torchvision ResNet-18 interface](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
[Lee et al., 2009](https://ai.stanford.edu/~ang/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf),
[Grad-CAM](https://arxiv.org/abs/1610.02391),
[Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292).
Lee et al. inspires the low-to-high illustration; this discriminative ResNet does
not reproduce the paper's convolutional deep belief network or generative training.
