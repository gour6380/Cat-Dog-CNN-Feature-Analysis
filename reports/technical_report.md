# Technical report: cat and dog CNN feature learning and localization

Fixed evidence snapshot completed on 2026-09-17T17:45:29+05:30.

## Clean species predictions and failure status

> adversarial arm — WARNING: this arm predicts only dog on all 3669 clean test images (macro accuracy 50%). Nominal attack survival must not be presented as useful robust cat/dog recognition. This is a classifier prediction failure, not proof that all hidden features are constant.

| Arm | True cats | True dogs | Predicted cats | Predicted dogs | Cat recall | Dog recall | Macro accuracy | Clean-test status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| standard | 1183 | 2486 | 1186 | 2483 | 99.15% | 99.48% | 99.32% | Both species predicted; robustness not inferred |
| adversarial | 1183 | 2486 | 0 | 3669 | 0.00% | 100.00% | 50.00% | FAILED comparison: single-species prediction |

A single-species arm is retained as a failed comparison, not useful robust cat/dog recognition. Its channel and attribution pictures are failure diagnostics. Hidden features may remain variable even when all observed clean decisions select one species; the cause of the classifier failure remains unproven.


## Question and scope

What patterns activate early, middle, and late CNN channels, and where do the selected
channels and cat/dog classifier respond within actual images? This is a two-class species
study, not a 37-breed projection study. Breed metadata remains available and the official
trainval partition is split within each breed; the official test partition stays untouched.

Both ResNet-18 arms start from identical ImageNet tensors and an identical new binary head.
The standard arm trains on clean inputs; the adversarial arm trains on bounded PGD inputs.
These are fixed epoch-15 checkpoints, not test-selected
checkpoints.

## Classification and finite attacks

The clean attack-subset accuracy and FGSM/PGD robust accuracies below use the same
200
hash-selected test images, balanced by target species. The full official test contains
3669 images and its accuracy is reported separately.

| Arm | Clean subset | FGSM robust | PGD-20×5 robust | PGD success among clean-correct |
|---|---:|---:|---:|---:|
| standard | 99.50% | 17.50% | 0.00% | 100.00% |
| adversarial | 50.00% | 50.00% | 50.00% | 0.00% |

PGD-20×5 is finite untargeted `L∞` search with
`epsilon=0.015686275`. Attack success is measured among
clean-correct samples. The search is not certified, physical, or unrestricted robustness.

## Hierarchical feature diagnostics

| Arm | Layer | Spatial grid | Receptive field (px) | Selected channels | Mean response change vs initialization |
|---|---|---:|---:|---|---:|
| standard | `network.layer2` | 28×28 | 99 | 115, 91, 23, 62, 69, 71 | 0.0288 |
| standard | `network.layer4` | 7×7 | 435 | 417, 245, 17, 345, 186, 46 | 0.3294 |
| standard | `network.relu` | 112×112 | 7 | 24, 49, 39, 63, 10, 59 | 0.0249 |
| adversarial | `network.layer2` | 28×28 | 99 | 62, 71, 73, 91, 23, 124 | -0.0795 |
| adversarial | `network.layer4` | 7×7 | 435 | 491, 418, 133, 199, 47, 61 | -0.2562 |
| adversarial | `network.relu` | 112×112 | 7 | 49, 63, 20, 51, 10, 60 | 0.0209 |

Channels were selected from clean calibration responses, not test-image appearance.
First-layer kernels show learned weights. Activation-maximization images are synthetic
inputs optimized to excite a fixed channel; they are not recovered training photographs.
Response changes compare the same selected channels on the same calibration images against
the matched ImageNet initialization. They do not identify a named semantic concept.

### Gray synthetic tiles: zero-gain trials retained

4 of 36 recorded single-start synthetic optimization trials had zero unregularized response gain. Their gray/blank tiles are unsuccessful stimuli, not evidence of dead channels or that the model learned nothing.

| Arm | Layer | Channel | Seeded initial response | Final response | Gain | Real calibration mean: cat | Real calibration mean: dog |
|---|---|---:|---:|---:|---:|---:|---:|
| adversarial | `network.layer2` | 73 | 0.0000 | 0.0000 | 0.0000 | 0.1088 | 0.1115 |
| adversarial | `network.layer4` | 491 | 0.0000 | 0.0000 | 0.0000 | 0.8542 | 0.9773 |
| adversarial | `network.layer4` | 418 | 0.0000 | 0.0000 | 0.0000 | 0.4555 | 1.4753 |
| adversarial | `network.layer4` | 61 | 0.0000 | 0.0000 | 0.0000 | 0.4501 | 1.2567 |

These channels respond positively to real calibration images, as shown above. The fixed seeds, selected channels, and optimization budget were preserved; no retry or replacement was used to make the atlas look better. Initial/final responses are measured on unjittered inputs, not the regularized optimization loss.


Activation maps locate responses on the transformed input. Highlighted receptive-field
boxes describe theoretical input support for an activation, not the precise pixels that
caused it. Later receptive fields can exceed the entire input image.

## Class localization and controls

| Arm | Test anchors | Top-CAM occlusion mean margin drop | Equal-area random occlusion mean margin drop | Mean randomized-model CAM correlation | Defined CAM correlations |
|---|---:|---:|---:|---:|---:|
| standard | 4 | 1.8832 | 0.3178 | -0.1613 | 3/4 |
| adversarial | 4 | 0.0129 | 0.0135 | -0.1233 | 2/4 |

Grad-CAM differentiates the fixed true-species logit, including on classification errors.
Occlusion measures the true-species-minus-other-species logit margin on the same input:
these are related but different scalar objectives. Equal-area image regions are replaced,
comparing high-CAM tiles with deterministic random tiles. A positive drop means the
intervention reduced the true-species margin. Randomization checks whether CAM changes
when the trained model is
randomized. These are limited descriptive diagnostics, not causal proof of what a filter
learned, not training-source attribution, and not proof that the object is the only cue.
Constant trained or randomized CAMs have undefined correlation; the mean excludes them
and the defined-count column makes that omission explicit. Undefined is not zero.

| Arm | Sample | True species | Clean prediction | Clean margin | PGD prediction | PGD margin |
|---|---|---|---|---:|---|---:|
| standard | `Abyssinian_65` | cat | cat | 6.4615 | dog | -97.7517 |
| standard | `Ragdoll_255` | cat | cat | 8.6434 | dog | -122.7923 |
| standard | `english_cocker_spaniel_56` | dog | dog | 24.9317 | cat | -44.0627 |
| standard | `shiba_inu_67` | dog | dog | 12.9986 | cat | -44.5126 |
| adversarial | `Abyssinian_65` | cat | dog | -0.5485 | dog | -0.6524 |
| adversarial | `Ragdoll_255` | cat | dog | -0.6334 | dog | -0.7222 |
| adversarial | `english_cocker_spaniel_56` | dog | dog | 1.3051 | dog | 0.6838 |
| adversarial | `shiba_inu_67` | dog | dog | 1.3355 | dog | 0.9035 |

## Standard arm: full official test and registered shifts

| Condition | Accuracy | Macro accuracy | Cat accuracy | Dog accuracy | NLL | Brier | Coverage | Selective risk | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| brightness-0.6 | 99.21% | 99.04% | 98.56% | 99.52% | 0.0227 | 0.0125 | 87.65% | 0.00% | 0.0029 | 0.0002 |
| brightness-0.8 | 99.37% | 99.29% | 99.07% | 99.52% | 0.0202 | 0.0105 | 89.67% | 0.00% | 0.0035 | 0.0001 |
| clean | 99.37% | 99.32% | 99.15% | 99.48% | 0.0191 | 0.0099 | 90.05% | 0.00% | 0.0036 | 0.0001 |
| contrast-0.5 | 98.72% | 98.79% | 98.99% | 98.59% | 0.0370 | 0.0204 | 83.57% | 0.00% | 0.0051 | 0.0004 |
| contrast-0.75 | 99.24% | 99.22% | 99.15% | 99.28% | 0.0224 | 0.0115 | 89.02% | 0.00% | 0.0029 | 0.0002 |
| gaussian_blur-0.75 | 99.02% | 98.88% | 98.48% | 99.28% | 0.0288 | 0.0154 | 87.30% | 0.00% | 0.0049 | 0.0002 |
| gaussian_blur-1.5 | 97.36% | 96.41% | 93.74% | 99.07% | 0.0762 | 0.0416 | 76.83% | 0.04% | 0.0114 | 0.0015 |
| gaussian_noise-0.02 | 99.26% | 99.28% | 99.32% | 99.24% | 0.0200 | 0.0106 | 89.59% | 0.00% | 0.0036 | 0.0001 |
| gaussian_noise-0.05 | 98.75% | 98.68% | 98.48% | 98.87% | 0.0360 | 0.0192 | 83.05% | 0.00% | 0.0048 | 0.0004 |

## Adversarial arm: full official test and registered shifts

| Condition | Accuracy | Macro accuracy | Cat accuracy | Dog accuracy | NLL | Brier | Coverage | Selective risk | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| brightness-0.6 | 67.76% | 50.00% | 0.00% | 100.00% | 0.6119 | 0.4237 | 97.71% | 30.66% | 0.0654 | 0.0942 |
| brightness-0.8 | 67.76% | 50.00% | 0.00% | 100.00% | 0.5812 | 0.4009 | 93.38% | 27.44% | 0.1468 | 0.0769 |
| clean | 67.76% | 50.00% | 0.00% | 100.00% | 0.5505 | 0.3801 | 90.32% | 24.98% | 0.2052 | 0.0747 |
| contrast-0.5 | 67.76% | 50.00% | 0.00% | 100.00% | 0.6266 | 0.4352 | 99.73% | 32.06% | 0.0179 | 0.1486 |
| contrast-0.75 | 67.76% | 50.00% | 0.00% | 100.00% | 0.5978 | 0.4129 | 96.08% | 29.48% | 0.1085 | 0.0837 |
| gaussian_blur-0.75 | 67.76% | 50.00% | 0.00% | 100.00% | 0.5680 | 0.3928 | 95.83% | 29.29% | 0.1792 | 0.0857 |
| gaussian_blur-1.5 | 67.76% | 50.00% | 0.00% | 100.00% | 0.5933 | 0.4107 | 99.89% | 32.17% | 0.1257 | 0.1068 |
| gaussian_noise-0.02 | 67.76% | 50.00% | 0.00% | 100.00% | 0.5508 | 0.3804 | 90.73% | 25.32% | 0.2058 | 0.0754 |
| gaussian_noise-0.05 | 67.76% | 50.00% | 0.00% | 100.00% | 0.5538 | 0.3827 | 92.50% | 26.75% | 0.2043 | 0.0785 |

Scalar temperatures and 90%-target confidence thresholds use only clean
calibration data; they are frozen for every test, corruption, and attack condition.
Confidence is not an out-of-distribution detector.

## Figures and distribution

Synthetic feature montages, kernel grids, and aggregate charts may be exported.
Actual pet images, feature-map overlays, Grad-CAM, occlusion panels, and receptive-field
crops remain in ignored local analysis outputs. The generated notebook can display them
locally, but they are not silently included in the public repository.

## Limitations

- Responses do not identify which training photograph caused a filter to be learned.
- Synthetic patterns are regularized optimized stimuli, not reconstructed pet photographs.
- ImageNet pretraining already supplies feature detectors; changes from initialization are reported.
- No channel is claimed to be a verified eye, ear, fur, or other named semantic detector.
- Grad-CAM is coarse class attribution; channel sensitivity is a local input gradient.
- Theoretical high-layer receptive fields exceed the crop; an activation cell is not a tiny isolated part.
- Occlusion introduces an artificial shift; four fixed anchors support descriptive, not population, conclusions.
- Channels are ranked independently within each arm; equal channel IDs do not guarantee equal semantics.
- Only one initialization and finite digital attack family are evaluated; no physical or safety claim is made.

There is no bootstrap representation-retention claim in this revised study. No named
concept, exact causal training source, physical robustness, or safer recognition claim
is made. Configuration SHA-256: `3b96c26d328cffc308cb6de70e070748bba3c9fe7a7b229ccba6749f8aec2a91`. Checkpoint/source/evidence hashes are
recorded in `local-release-manifest.json`.
