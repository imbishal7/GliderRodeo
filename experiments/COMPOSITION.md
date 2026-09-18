# Prespecification: do the three LOGO interventions compose?

Logged before running. Exploratory follow-up on an already-inspected mission.

## Motivation
Three independent interventions each recover almost exactly the same LOGO error,
on byte-identical rows (verified: 5,365 rows, max |diff| in targets 3.6e-15):

| intervention | LOGO pooled RMSE | vs global |
|---|---|---|
| global (tabpfn / catboost) | 6.392 / 6.388 | — |
| per-platform target scaling | 6.172 (catboost) | −0.22 |
| forcing-gated mixture | 6.168 (tabpfn) | −0.22 |
| full-network fine-tuning | 6.214 (tabpfn) | −0.18 |

Scaling vs forcing-gate head-to-head is a dead heat: +0.009 dB
[−0.218, +0.256], P(better) = 0.526. Three different mechanisms landing within
0.05 dB of each other is the signature of a shared ceiling, not a coincidence.

## Hypothesis
H0: they do not compose — all three capture the same platform-level transfer
    structure, so combining gains ~nothing beyond the best single one (~6.17).
H1: they are at least partly orthogonal — combining approaches ~6.0 or better.

## Prediction (recorded before running)
H0. The forcing gate (wind, Hs, wave steepness, upslope current) clusters rows
into environmental regimes that are strongly confounded with platform, because
each glider flew a distinct time/place window. Scaling removes the per-platform
output shift directly. Both are therefore proxies for the same latent variable,
and fine-tuning on 1,565 rows most plausibly learns the same thing a third way.

## Arms
`--stage experts --target-scaling platform`, protocol `logo`, seed 0, identical
folds. Compares scaled global / scaled forcing-gate / scaled topography-gate
within both families against the unscaled results already in hand.

## Decision rule
H1 only if the scaled forcing-gate arm beats BOTH the unscaled forcing-gate arm
and the scaled global arm, each with a paired UTC-day bootstrap interval
excluding zero. Otherwise H0 stands and ~6.17 is a real ceiling for this
approach class on this dataset.
