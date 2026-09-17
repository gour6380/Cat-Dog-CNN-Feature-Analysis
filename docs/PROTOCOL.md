# Cat/Dog CNN feature-visualization protocol

Revised 17 September 2026. This replaces the earlier 37-breed projection study.
Its previous results are a superseded experiment, not binary-task evidence.

The YAML configuration is the source of truth. Valid values are editable and receive
a new configuration hash; Python does not enforce a duplicate table of constants.

## Question and scope

What patterns activate low-, mid-, and high-level channels in matched standard and
PGD-trained ResNet-18 Cat/Dog classifiers, where do these responses occur in real
images, and how do predictions/explanations change under fixed input shifts?

Feature visualization is the centerpiece. It is not an embedding-quality claim:
there is no PCA, t-SNE, UMAP, breed-neighbour-retention primary decision, or breed
boundary slice in the active workflow. Visual differences are descriptive and must
be supported by saved response values rather than interpreted as robustness alone.

## Dataset, splits, and labels

Oxford-IIIT Pet contains roughly 7,349 images from 37 breeds (12 cat breeds and 25 dog
breeds). Preserve the official test partition. Within each breed in official
`trainval`, use the recorded filename-hash seed to split 80% training and 20%
calibration. A breed-stratified split avoids accidentally excluding a breed even
though the supervised task has only two labels: `cat=0`, `dog=1`.

EDA records actual image counts, species/class balance, original image dimensions,
and split coverage from registered files. The species distribution is not balanced;
report overall, macro, and per-species accuracy rather than hiding that imbalance.
Report clean prediction counts as well as true counts. A model predicting only one
species is a failed recognition arm; attack survival by a majority-class decision
must not be presented as useful robustness. This diagnostic concerns decisions,
not whether its hidden features are constant. Preserve such failures without
post-test checkpoint selection or an undisclosed retuned rerun.
No test samples participate in fitting temperature, threshold, or channel selection.

Select the paired attack set deterministically: 100 official-test samples per species.
Select four fixed clean test anchors: two cats and two dogs. Record filenames and
hashes; do not replace an unflattering anchor after seeing its explanation.

## Matched training and evaluation

Both models start from byte-identical Torchvision `IMAGENET1K_V1` state and an identical
new two-class head. Fully fine-tune at 224×224 RGB using deterministic random resized
crops/flips for training and resize/center crop for evaluation. Attack raw `[0,1]`
pixels before ImageNet normalization inside the model.

Reference training: 15 float32 MPS epochs; micro-batch 16; two-step accumulation;
AdamW; backbone LR `1e-4`, head LR `1e-3`, weight decay `1e-4`; one-epoch warm-up and
cosine decay. Standard uses clean cross-entropy; adversarial uses random-start PGD-5,
untargeted `L∞ 4/255`, step `1/255`, projection and pixel clipping. Sample order,
augmentations, update counts, initialization, and schedule stay matched.

Compare fixed final-epoch checkpoints, never a test-selected checkpoint. Report clean
overall/macro/per-species accuracy, FGSM and PGD-20×5 robust accuracy, and attack
success among clean-correct samples. Full-test shifts are Gaussian noise
`0.02/0.05`, Gaussian blur `0.75/1.5`, brightness `0.8/0.6`, and contrast `0.75/0.5`.

Fit one scalar temperature and 90%-coverage confidence threshold per model on clean
calibration only. Report NLL, Brier score, 15-bin ECE, coverage, selective risk, and
tie-aware AURC. Flag shifted coverage outside 85–95% or selective-risk change above
two percentage points; do not refit on shifted/test samples.

## Layer/channel walkthrough

Use low `network.relu`, mid `network.layer2`, and high `network.layer4` activations.
All channel IDs and clean-calibration selection scores are recorded before fixed
test-anchor inspection. Preserve spatial responses rather than only pooled features.

1. Show actual `conv1` RGB kernels as weights, not as input-image explanations.
2. Optimize synthetic inputs for selected low/mid/high channels, with recorded seed,
   objective, regularization, steps, and response. Label them synthetic, not decoded
   training examples. Keep visual scaling consistent or disclose separate scaling.
3. Retrieve top-activating clean calibration locations and show their real input
   patches, activation coordinates, and theoretical receptive-field boxes.
4. For fixed anchors, show activation maps and input-gradient sensitivity for the
   selected channels. Preserve the source image alignment after the evaluation crop.
5. Compute Grad-CAM for the true-species logit and sliding occlusion for the
   true-species-vs-other logit margin on the same anchors. These are related but
   distinct scalar objectives; agreement does not exactly validate attribution to
   the same scalar. Keep the true target fixed on misclassified/attacked inputs.
   Save the baseline, mask size, stride, and margin change; do not change targets to
   improve the picture.
6. Include randomized-weight explanation controls and aggregate layer-response
   charts. Compare matched images and channel-selection rules across arms, not
   unaligned coordinates or apparently brighter heatmaps.

Real patches, source photographs, activation overlays, and photo-containing result
notebooks remain ignored local artifacts. Only reviewed aggregate/synthetic figures
are eligible for repository export.

## How to interpret the images

- A channel is not automatically a human concept such as an eye, ear, nose, or fur.
  Top patches and synthetic stimuli suggest possible response patterns, not verified
  semantic labels. No named detector claim is made without a validation protocol.
- An activation says where a channel responds to this input. A gradient is local
  sensitivity of an objective. Grad-CAM coarsely localizes the true-species logit.
  Occlusion measures the true-species-vs-other margin response to a particular masking
  intervention and can itself create out-of-distribution inputs. These methods answer
  different questions and do not use identical scalar targets.
- A theoretical receptive field indicates possible architectural support. Deep
  ResNet receptive fields may exceed the input and should be clipped for display;
  they are not measured effective receptive fields or equal-use guarantees.
- Heatmaps normalized independently cannot support an absolute response comparison.
  Use numeric response/score summaries for that comparison.
- These methods explain the trained model's present behavior, not which original
  training image or causal learning event produced a filter.
- ImageNet pretraining, one model pair/split/seed family, finite attacks, and selected
  anchors limit conclusions. Neither attractive features nor a PGD score certifies
  robustness, physical behavior, or safety.

## Provenance and failure behavior

The feature manifest is `results/generated/feature_visualizations.json`, including
`arms`, `figures`, `limitations`, `config_sha256`, and `created_at`. Save the selected
sample/channel IDs, data/checkpoint/config/source identities, responses, settings,
environment/MPS memory, timing, and figure hashes alongside it. Only matching
evidence is eligible for a report or export. The guide verifies each recorded figure
SHA-256 before display and explicitly skips missing or mismatched image evidence.

Memory snapshots are diagnostic telemetry, not minimum-memory checks. MPS OOM,
non-finite loss/gradients/logits, invalid attack bounds, parity failure, and provenance
mismatch preserve a failure record. Do not silently lower epochs, resolution, PGD
steps, restarts, or model scope to turn a failed experiment into a completed result.

## Method references

The low-to-high feature illustration is inspired by
[Lee et al., 2009, Convolutional Deep Belief Networks](https://ai.stanford.edu/~ang/papers/icml09-ConvolutionalDeepBeliefNetworks.pdf).
This ResNet study does not reproduce their generative model. Related original methods:
[Zeiler and Fergus, 2014](https://arxiv.org/abs/1311.2901),
[Grad-CAM, Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391),
[Adebayo et al., 2018, Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292),
and [Olah et al., 2017, Feature Visualization](https://distill.pub/2017/feature-visualization/).
