# Exploratory PGD stabilization pilot

This is one owner-approved follow-up to the completed Cat/Dog study, not a rewrite
of its failed PGD arm. The original experiment and illustrated results remain
available with their original configuration and checkpoint identities. Its source
snapshot is local commit `9aa7f2ac49cd938d377db7bb66aae6ee58246338`.

## Completed result: not a successful robust model

Fixed epoch 15 completed on native MPS with 1,245 updates. Training plus all
per-epoch validation took 2,240.31 seconds; final evaluation took 311.90 seconds
(machine runtime, not a focused-hours claim).

| Measure | Failed original PGD arm | Exploratory pilot |
|---|---:|---:|
| Clean test accuracy | 67.76% | 85.06% (3,121/3,669) |
| Clean cat / dog recall | 0% / 100% | 53.68% / 100% |
| FGSM accuracy, paired subset | 50% | 49.5% (99/200) |
| PGD-20×5 accuracy, paired subset | 50% | 2.5% (5/200) |
| PGD cat / dog recall | 0% / 100% | 0% / 5% |

The original arm's nominal 50% attacked score was an all-dog decision rule, not
useful robustness. The pilot restores some clean cat recognition, but misses
548/1,183 clean cats and every attacked cat in the fixed 100-cat subset. Strong
PGD defeats 154 of its 159 clean-correct paired samples (96.86%). This is a
**negative strong-attack result**, not an accuracy-improvement success to present
as robust recognition. Keep the original standard model (99.37% clean) as the
main learned-feature walkthrough; do not replace it with this pilot.

Clean validation was volatile during mixed training. Final validation was
261/295 clean and 3/64 PGD-correct. All 15 epochs are retained; no earlier
checkpoint was substituted. Changing warm-up, mixing, weighting, fitting budget
and BatchNorm exposure together prevents attributing the result to one mechanism.

The clean-calibration policy achieves 90.24% test coverage with 11.45% selective
risk. Under blur sigma1.5, coverage becomes 95.42% and selective risk25.85%; the
confidence operating point does not stay fixed after shift. These are empirical
diagnostics, not a safety guarantee.

Evidence: config `a6a9b7a0…3bdedd`, dataset `a5373179…b4f3f`, initialization
`1ca4bb60…3fdd`, training protocol `a52f6c40…813df`, scientific source
`2fee7c7a…e7782`, checkpoint `a332c9c9…cc25f`, evaluation `2b3edde5…6fcd6`.
The local read-only report/notebook and full JSON comparison are listed below.
No original outputs were overwritten; no additional rescue run, architecture,
public repository, push or publication was performed.

## Registered recipe

Before fitting, register the exact [pilot configuration](../configs/pgd_pilot.yaml)
and deterministic fitting/validation sample IDs in its own output namespace.

- Start a fresh two-class ImageNet-initialized ResNet-18, not the failed checkpoint.
- Reserve 10% within each breed from the existing 2,944 training records. These
  samples become validation, never calibration or test. The exact counts are
  recorded by setup; the original 736 calibration and 3,669 test records stay fixed.
  The registered pilot has 2,649 fitting images (855 cats / 1,794 dogs) and 295
  validation images (95 cats / 200 dogs); its balanced attack monitor has 64 images.
- Compute species weights as `N / (2 × species_count)` from fitting records only.
  Registered weights are 1.549122807 for cats and 0.738294314 for dogs.
- Train 15 epochs: weighted clean cross-entropy for epochs 1–3, then
  `0.5 × weighted clean CE + 0.5 × weighted PGD CE` for epochs 4–15.
- Keep 224px inputs, MPS float32, micro-batch 16, accumulation 2, AdamW,
  backbone/head rates `1e-4/1e-3`, weight decay `1e-4`, one LR-warmup epoch,
  cosine decay, initialization seed and deterministic augmentation unchanged.
- Training attacks remain PGD-5, untargeted L∞ epsilon `4/255`, step `1/255`,
  projected uniform random start and clipping in raw `[0,1]` pixels. Attack
  generation uses the existing unweighted attack objective and freezes BatchNorm.
  The subsequent clean and adversarial training forwards update BatchNorm twice
  per mixed micro-batch; that exposure differs from the original pure-PGD arm.
- Monitor every epoch: full validation clean accuracy/recalls and clean/PGD
  accuracy/recalls on 32 fixed held-out validation images per species. Validation
  attacks are PGD-20×5. Monitoring does not select a checkpoint or alter training.
- Evaluate fixed epoch 15 on the unchanged official test partition, registered
  corruptions, and the same paired 200-image FGSM/PGD-20×5 test subset. Fit
  temperature/confidence only on clean calibration, after training.

The minimum diagnostic is whether both species receive correct decisions, rather
than reproducing a majority-only classifier. Report actual clean/attacked overall,
macro and cat/dog recall. No accuracy target or positive result is promised. A
single run cannot identify which recipe component caused any change.

## Important limits

The original test results informed this recipe, so all pilot comparisons are
**exploratory**. The held-out validation split also reduces the fitting set and
optimizer-update count relative to the original arms. This is not an exactly
matched training comparison or an independent confirmation. Report the old
all-dog model's nominal 50% PGD survival as a failed decision rule, not useful
robust recognition. A new score is finite attack survival, not certified, physical,
OOD or complete adversarial robustness. ImageNet pretraining remains a confound.

No additional architecture, dataset, hyperparameter grid, checkpoint selection,
AutoAttack, weaker attack, synthetic-probe retry, public upload or publication is
included. Focused work stays inside the Week 3 17-hour allocation; the 2.5-hour
application/profile and 0.5-hour review blocks remain protected. Machine runtime
is recorded separately and does not establish actual focused hours.

OOM, invalid attacks, nonfinite values, silent MPS fallback or provenance mismatch
stop and preserve the failure. There is no minimum available-memory gate and no
automatic protocol downgrade.

## Run and inspect

Use the existing Python 3.13.15 environment and `requirements.txt` (venv/pip, no uv):

This frozen comparison command is for the current completed local study: it needs
the original ignored checkpoints, evaluation arrays, release inventory and
`baseline-preservation.json` receipt. A fresh Git clone does not contain these
private artifacts and cannot reproduce this exact saved comparison by running the
command alone. To register a different fresh comparison, first complete the
original pipeline, preserve its own measured identities, and register a new pilot
configuration/receipt with those identities. Do not relabel new outcomes as this
pilot or bypass the preservation checks.

```bash
.venv/bin/python src/cli.py pilot --config configs/pgd_pilot.yaml --device mps
```

The command runs setup, protocol registration, MPS preflight, the adversarial pilot,
final evaluation and its separate report. Repeating it resumes compatible epoch
checkpoints and reuses compatible evaluation arrays; it does not launch a grid.
`reproduce` with the pilot configuration routes to the same pilot-only workflow.

All pilot output directories end in `pilots/balanced-mixed-pgd-v1`; checkpoints use
the pilot's distinct configuration-hash namespace. Original checkpoints, evaluation,
feature panels, guide notebook and reports are hash-verified before and after.
The old release manifest still describes its immutable original source commit,
not the newly modified working source.

After completion, open
`reports/generated/pilots/balanced-mixed-pgd-v1/pgd_pilot_results.ipynb` for embedded
charts and explanations without rerunning training, or read the adjacent
`pgd-pilot-report.md`. These are separate from the original feature walkthrough;
no new semantic filter labels or causal training-source attributions are inferred.

Regenerate only the separate pilot report:

```bash
.venv/bin/python src/cli.py report --config configs/pgd_pilot.yaml
```
