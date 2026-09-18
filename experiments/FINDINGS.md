# Noise Nowcaster — model comparison findings

One question: does anything beat the baseline gradient-boosted model at predicting
band levels on a **glider it has never seen**? Everything below is scored
leave-one-glider-out (LOGO) on the archived 1,565-row fleet population, with each
target's reference median computed from training rows only and a 48-hour
reference-only calibration window for the held-out glider.

## Result

| model | LOGO RMSE | vs baseline | 95% interval | replicates across seeds |
|---|---|---|---|---|
| baseline (global TabPFN) | 6.374 dB | — | — | — |
| **forcing-gated mixture** | **6.196 dB** | **−0.176 dB** | **[−0.272, −0.069]** | **yes** |
| full-network fine-tuning | 6.229 dB | −0.144 dB | [−0.231, −0.048] | not tested |
| per-platform target scaling | 6.137 dB | −0.234 dB | [−0.474, +0.040] | not tested |

Intervals are a paired UTC-day block bootstrap (14 day-blocks), which keeps hours
from the same day together — adjacent hours are not independent samples.

The **forcing-gated mixture** is the recommended model. It is not the largest point
estimate, but it is the only arm whose interval excludes zero *and* that reproduces
under a second seed (−0.178 at seed 0, −0.162 at seed 1, P(better) = 1.000 in both).
The other two have larger or comparable point estimates with intervals that cross
zero on this many day-blocks.

## What the gate uses

Two experts on the original 13 predictors, routed by a training-only two-cluster
KMeans gate over **environmental forcing**: wind speed, significant wave height, a
deep-water wave-steepness proxy, and upslope current. Blend weight against the
global model is selected from {0, .25, .5, 1} by inner validation that mirrors the
outer protocol; α=0 (no expert contribution) is a selectable outcome.

Gate choice is the whole result. Alternative gates were tested on identical folds
and do essentially nothing:

| gate | seed 0 | seed 1 |
|---|---|---|
| forcing | −0.178 | −0.162 |
| geography | −0.023 | −0.009 |
| topography | −0.001 | −0.004 |
| added physics features (no gate) | −0.024 | −0.007 |

So the gain comes from routing on *sea state*, not from location, bathymetry, or
from adding physical predictors to a single model.

## Scope, honestly stated

- **The gain is LOGO-specific.** On the forward-time protocol the same gate is
  *worse* than the global model in both seeds (+0.31 / +0.22). It helps transfer to
  an unseen platform; it does not help forecasting forward in time.
- **SPWH and UNDO overlap.** SPWH (5–15 kHz) is a strict subset of UNDO (5–20 kHz)
  and the two measured series correlate at r = 0.9997. Every number above therefore
  **excludes SPWH** to avoid counting one result twice. Scoring all five bands would
  report −0.222 dB instead of −0.176 dB.
- More broadly, every band above 200 Hz correlates at r ≥ 0.9; only FIWH (10–30 Hz)
  is distinct. Five band scores are not five independent replications.
- **The interventions do not stack.** Combining the gate with per-platform target
  scaling was prespecified and tested: unscaled, the gate is worth −0.178 dB; once
  targets are scaled it is worth only −0.042 dB. Both are largely correcting the
  same per-platform structure, so ship one, not both.
- 6 platforms and 14 day-blocks is a small sample. These are exploratory follow-up
  results on a mission whose outer scores had already been inspected, not a
  confirmatory study on untouched data.

## Reproducing

```bash
bash experiments/followup/run_campaign.sh experts     # GPU, ~35 min
```

Predictions, per-fold metrics, inner-validation choices, split manifests, source
hashes and peak GPU memory are written per run. Run outputs and raw data are
gitignored.
