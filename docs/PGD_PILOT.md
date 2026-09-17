# Exploratory PGD stabilization pilot

This is one owner-approved follow-up to the completed Cat/Dog study, not a rewrite
of its failed PGD arm. The original experiment and illustrated results remain
available with their original configuration and checkpoint identities. Its source
snapshot is local commit `9aa7f2ac49cd938d377db7bb66aae6ee58246338`.

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
