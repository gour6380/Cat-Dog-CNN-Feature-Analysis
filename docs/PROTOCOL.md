# Registered scientific protocol

Registered: 14 September 2026, before model training or test evaluation.

This document records the reference configuration. Experiment parameters remain editable
in `configs/experiment.yaml`; a changed file receives a new configuration hash and defines
a separate run rather than being rejected by duplicated constants in the source code.

## Primary decision

On the fixed 740-image (20 per breed) test subset, compare adversarial minus standard:

1. median clean-to-PGD-20×5 cosine feature drift; and
2. mean five-nearest-neighbour breed retention against clean calibration features.

The primary hypothesis is supported only when the 95% class-stratified bootstrap
interval has an upper bound below zero for drift and a lower bound above zero for
retention. Anything else is mixed, unsupported, or inconclusive—not a partial win.

## Secondary decisions

- Compare PGD-20×5 robust accuracy. Report clean overall/macro accuracy as an
  undirected trade-off.
- Flag operating-policy shift when any registered condition is outside 85–95%
  coverage or changes selective risk by more than 0.02 versus clean, using the clean
  calibration-fitted policy without refitting.

## Interpretation boundary

PCA and UMAP are independently fitted from each model's clean calibration features;
t-SNE is independently and jointly fitted to paired clean/PGD points within each
model. Raw coordinates are never compared between models. Projection separation,
cluster shape, or apparent boundary width does not establish robustness. The strongest
eligible conclusion is bounded digital-attack representation retention.

## Registered failure behavior

The 224 px, 15-epoch, PGD-5 training protocol is fixed. The runner stops and writes a
failure record on memory pressure, MPS OOM, invalid perturbation bounds, non-finite
loss/gradients/logits, parity failure, or provenance mismatch. It does not silently
lower batch size, resolution, epochs, steps, or model scope.
