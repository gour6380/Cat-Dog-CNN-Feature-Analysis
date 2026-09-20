# When 67.8% accuracy means the model learned one answer

Feature visualization is tempting because it turns a neural network into something we
can see. Early filters resemble edge or color detectors. Middle channels respond to
textures and repeated shapes. Later maps appear to highlight meaningful regions. But
those images become easy to overinterpret when the classifier underneath them is not
checked first.

This experiment trained two matched ResNet-18 Cat/Dog classifiers on Oxford-IIIT Pet.
Both started from the same ImageNet weights and binary head, saw the same fitting
records in the same order, and received 1,245 optimizer updates across 15 epochs. The
standard arm optimized clean cross-entropy. The second arm optimized only PGD-5 inputs
within an L∞ radius of `4/255`.

The goal was to compare what their low-, middle- and high-level channels preferred and
where those channels responded. The more important result arrived before the feature
gallery.

## A plausible number hid a complete failure

The standard checkpoint classified 294 of 295 held-out validation images correctly:
99.66% overall and 99.75% macro accuracy. The PGD checkpoint classified 200 correctly,
or 67.80%.

That second number could look respectable without context. The validation partition,
however, contains 95 cats and 200 dogs. The PGD model predicted **dog for all 295
images**. Cat recall was 0%, dog recall was 100%, and macro accuracy was exactly 50%.
Its overall accuracy simply reproduced the majority-class proportion.

The same pattern appears in fitting: all 855 cats were labeled dog and all 1,794 dogs
were labeled dog. The final attacked-training accuracy, 67.72%, exactly equals the dog
share. Its loss was also close to the entropy of that class prior. Under this recipe,
the model had converged to a shortcut rather than learning balanced discrimination.

This does not show that adversarial training generally fails. Pure PGD training, class
imbalance, BatchNorm exposure, optimization choices and the finite budget are all
plausible contributors. It does show that this particular comparison failed—and that
overall accuracy alone would have concealed the failure.

## What the internal pictures can and cannot add

The analysis generated eight registered feature families for both checkpoints:

1. input-to-stage walkthroughs through the stem, max-pool and four residual stages;
2. actual first-layer RGB kernels;
3. regularized activation-maximization stimuli;
4. strongest real reference patches with theoretical receptive-field boxes;
5. clean activation maps and channel input gradients;
6. true-species Grad-CAM;
7. signed occlusion effects; and
8. response summaries plus randomized-weight checks.

These tools do not all explain the same thing. A kernel is a learned weight pattern.
An optimized stimulus is an artificial input that raises one response. An activation
map shows where a channel fires. A gradient shows local sensitivity. Grad-CAM asks
where a class score has positive spatial support. Occlusion asks what happens after a
region is replaced. Similar-looking answers do not prove causality.

The diagnostics were consistent with the failure without explaining its cause. All six
selected late PGD channels responded more strongly to dogs on the balanced reference
sample, while standard selections separated in both directions. The PGD model labeled
both fixed cat anchors as dogs. Its attribution-to-occlusion effects were small and
inconsistent. Four activation-maximization attempts also produced zero gain; those gray
tiles were kept instead of being retried for a prettier atlas.

Each observation is descriptive. The channel samples were selected independently in
each model, the maps were normalized for display, and four anchors are not a population
study. The pictures cannot tell us which training photograph created a feature or prove
that a channel is an eye, ear or fur detector.

## The practical lesson

Interpretability should follow a minimum model-validity check, not replace it. Before
explaining a classifier, inspect at least:

- overall and macro accuracy;
- per-class recall;
- the prediction-count distribution;
- a simple class-prior baseline; and
- failures on fixed, non-cherry-picked examples.

Only after those checks does it make sense to ask what the internal representations are
doing. Here, visualization was still useful—but as a failure-analysis tool. It helped
show what a collapsed model responded to while preventing an attractive gallery from
being mistaken for evidence of useful robust features.

## Scope boundary

The validation partition was held out from fitting and did not select checkpoints, but
this active workflow did not compute official-test accuracy or post-training attack
accuracy. PGD was a training objective, not a measured robustness result. One
ImageNet-initialized model pair, one split/seed family, 64 reference images and four
walkthrough anchors cannot establish a general conclusion about adversarial training or
CNN semantics.

The defensible conclusion is narrower: **under this exact pure-PGD-5 and imbalanced
Cat/Dog protocol, the adversarial arm collapsed to the majority class; class-aware
monitoring exposed the failure, and feature visualization helped characterize it without
turning it into a robustness claim.**

[Technical report](technical_report.md) · [Results](../docs/results.md) ·
[Reproduction guide](../README.md)
