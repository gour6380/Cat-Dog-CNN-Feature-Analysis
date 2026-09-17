# Seeing what a cat/dog CNN responds to

## Clean species predictions and failure status

> adversarial arm — WARNING: this arm predicts only dog on all 3669 clean test images (macro accuracy 50%). Nominal attack survival must not be presented as useful robust cat/dog recognition. This is a classifier prediction failure, not proof that all hidden features are constant.

| Arm | True cats | True dogs | Predicted cats | Predicted dogs | Cat recall | Dog recall | Macro accuracy | Clean-test status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| standard | 1183 | 2486 | 1186 | 2483 | 99.15% | 99.48% | 99.32% | Both species predicted; robustness not inferred |
| adversarial | 1183 | 2486 | 0 | 3669 | 0.00% | 100.00% | 50.00% | FAILED comparison: single-species prediction |

A single-species arm is retained as a failed comparison, not useful robust cat/dog recognition. Its channel and attribution pictures are failure diagnostics. Hidden features may remain variable even when all observed clean decisions select one species; the cause of the classifier failure remains unproven.


Two ImageNet-initialized ResNet-18 models were fine-tuned for a binary cat-versus-dog task from the
official Oxford-IIIT Pet dataset. Matched initialization, image order, crops, and optimizer
updates isolate clean versus PGD-5 training.

The feature story is now visual and layer-by-layer: kernel weights in the first layer,
synthetic activation-maximization stimuli, actual-image activation maps, and receptive-field
crops for selected channels. Channels are selected using calibration responses so the
test images are explanations, not a mechanism for cherry-picking the selection.

### Gray synthetic tiles: zero-gain trials retained

4 of 36 recorded single-start synthetic optimization trials had zero unregularized response gain. Their gray/blank tiles are unsuccessful stimuli, not evidence of dead channels or that the model learned nothing.

| Arm | Layer | Channel | Seeded initial response | Final response | Gain | Real calibration mean: cat | Real calibration mean: dog |
|---|---|---:|---:|---:|---:|---:|---:|
| adversarial | `network.layer2` | 73 | 0.0000 | 0.0000 | 0.0000 | 0.1088 | 0.1115 |
| adversarial | `network.layer4` | 491 | 0.0000 | 0.0000 | 0.0000 | 0.8542 | 0.9773 |
| adversarial | `network.layer4` | 418 | 0.0000 | 0.0000 | 0.0000 | 0.4555 | 1.4753 |
| adversarial | `network.layer4` | 61 | 0.0000 | 0.0000 | 0.0000 | 0.4501 | 1.2567 |

These channels respond positively to real calibration images, as shown above. The fixed seeds, selected channels, and optimization budget were preserved; no retry or replacement was used to make the atlas look better. Initial/final responses are measured on unjittered inputs, not the regularized optimization loss.


The distinction is important: a synthesized edge or texture is an input that excites a
channel. It does not prove that the channel learned the named concept we attach to it.
An activation map shows where a response occurs; its receptive-field box is theoretical
support, not an exact causal explanation.

Class-specific Grad-CAM asks a different question: where does the fixed true-species logit
respond? Equal-area top-CAM versus random occlusion measures the true-vs-other margin,
a related but different objective, and randomized-model controls probe weight dependence.
They provide descriptive checks, not a causal account of the training process.

Clean full-test accuracy on 3669 official test images
is 99.37% for
standard training and 67.76% for PGD
training. Finite PGD-20×5 robust accuracy on the paired
200-image
species-balanced subset is
0.00% and
50.00%, respectively.

| Arm | Test anchors | Top-CAM occlusion mean margin drop | Equal-area random occlusion mean margin drop | Mean randomized-model CAM correlation | Defined CAM correlations |
|---|---:|---:|---:|---:|---:|
| standard | 4 | 1.8832 | 0.3178 | -0.1613 | 3/4 |
| adversarial | 4 | 0.0129 | 0.0135 | -0.1233 | 2/4 |

The scope remains one dataset, one architecture, and one split/seed family.
ImageNet-pretrained features, synthetic optimization artifacts, backgrounds, finite attack
search, and coarse localization are credible limitations. No physical or safe-use
conclusion follows. Original pet photos remain local; public visuals are synthetic or
aggregate.
