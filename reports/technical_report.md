# Technical report: adversarial representation drift in Oxford-IIIT Pet

Generated from the fixed machine-readable evidence completed on
2026-09-14T20:11:03+05:30.

## Registered decision

Both registered criteria passed: under this fixed finite digital attack, the adversarially trained model retained the measured penultimate structure better. The adversarial-minus-standard median cosine-drift difference was
`-0.415384` with 95% interval
`[-0.423403, -0.408943]`.
The 5-neighbour retention difference was
`0.255135` with 95% interval
`[0.235399, 0.276216]`.

## Classification and finite attacks

| Arm | Clean subset | FGSM robust | PGD-20×5 robust | PGD success among clean-correct |
|---|---:|---:|---:|---:|
| standard | 87.97% | 2.84% | 0.00% | 100.00% |
| adversarial | 72.57% | 31.08% | 20.68% | 71.51% |

PGD-20×5 is a finite untargeted `L∞` evaluation at `epsilon=0.015686275`; it is not a proof
against stronger, adaptive, physical, or unrestricted attacks. Attack success uses only
samples classified correctly before attack. Both models are their configured epoch-15 states.

## Original-feature geometry

| Arm | Median cosine drift | Median relative-L2 | PGD 5-NN accuracy | PGD 5-NN retention | CKA |
|---|---:|---:|---:|---:|---:|
| standard | 0.4531 | 2.8832 | 0.00% | 0.11% | 0.1669 |
| adversarial | 0.0378 | 0.2833 | 27.57% | 25.62% | 0.8627 |

All values above use the original 512-dimensional penultimate features. PCA, t-SNE,
and UMAP figures are explanatory and their independently fitted model coordinates are
not compared numerically.

## Standard model: full official test and registered shifts

| Condition | Accuracy | Macro accuracy | Coverage | Selective risk | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|
| brightness-0.6 | 87.65% | 87.60% | 88.03% | 7.03% | 0.0289 | 0.0278 |
| brightness-0.8 | 88.96% | 88.92% | 88.69% | 6.21% | 0.0221 | 0.0230 |
| clean | 89.15% | 89.13% | 89.23% | 6.11% | 0.0224 | 0.0213 |
| contrast-0.5 | 84.08% | 84.04% | 82.86% | 8.03% | 0.0273 | 0.0379 |
| contrast-0.75 | 88.66% | 88.63% | 88.20% | 6.61% | 0.0243 | 0.0232 |
| gaussian_blur-0.75 | 85.39% | 85.23% | 85.17% | 8.26% | 0.0291 | 0.0371 |
| gaussian_blur-1.5 | 71.38% | 71.25% | 71.52% | 17.34% | 0.0838 | 0.1170 |
| gaussian_noise-0.02 | 88.47% | 88.48% | 88.03% | 6.50% | 0.0214 | 0.0241 |
| gaussian_noise-0.05 | 81.22% | 81.32% | 77.65% | 9.69% | 0.0315 | 0.0565 |

## Adversarial model: full official test and registered shifts

| Condition | Accuracy | Macro accuracy | Coverage | Selective risk | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|
| brightness-0.6 | 49.28% | 49.20% | 77.38% | 42.02% | 0.0913 | 0.2862 |
| brightness-0.8 | 67.57% | 67.48% | 87.35% | 26.52% | 0.0196 | 0.1262 |
| clean | 72.91% | 72.84% | 90.71% | 22.51% | 0.0188 | 0.0916 |
| contrast-0.5 | 25.76% | 25.68% | 65.39% | 68.11% | 0.2275 | 0.5932 |
| contrast-0.75 | 63.59% | 63.52% | 83.81% | 28.81% | 0.0180 | 0.1532 |
| gaussian_blur-0.75 | 69.01% | 68.93% | 87.76% | 25.12% | 0.0177 | 0.1156 |
| gaussian_blur-1.5 | 57.51% | 57.40% | 80.54% | 34.55% | 0.0416 | 0.2049 |
| gaussian_noise-0.02 | 72.88% | 72.81% | 90.81% | 22.57% | 0.0223 | 0.0915 |
| gaussian_noise-0.05 | 72.44% | 72.39% | 90.60% | 22.74% | 0.0186 | 0.0923 |

Temperatures and 90%-coverage thresholds were fitted only on the
deterministic clean calibration partition. They were not refitted to the official test
set or any attack or corruption. Confidence is not presented as an out-of-distribution
detector.

## Difficult-pair boundary

The shared pair was classes `5`
and `11`, selected by maximum
symmetric clean-calibration confusion, then minimum centroid distance and class ID. Each
line is the exact equality of those two linear-head logits restricted to that model's own
PCA plane with undisplayed coordinates fixed at the calibration mean. It is not an input-
space boundary or a complete 37-class decision map.

## Limitations and competing explanations

- ImageNet initialization may account for substantial geometry before fine-tuning.
- Oxford-IIIT Pet has limited examples per breed; a single split and seed limit scope.
- PCA, t-SNE, and UMAP distort different relationships and cannot validate robustness.
- PGD-20×5 is finite; failure to find an adversarial example does not prove none exists.
- Synthetic pixel corruptions do not establish physical robustness or safe recognition.
- Optimization dynamics—not adversarial invariance alone—may explain observed geometry.

Configuration SHA-256: `9e9727fe7a442e0e181349073577dca9c2e72828416c88dfe5f9d446f2ee9b14`. Full provenance and artifact hashes are in
`local-release-manifest.json`.
