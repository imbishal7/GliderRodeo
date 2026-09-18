# Expert and pretrained-adaptation follow-up, 2026-09-18

Declared before running this campaign. Prior outer results have already been
inspected; this is exploratory follow-up on the existing mission, not untouched
confirmation data. No outer scores select gates, blend strengths, or training steps.

## Questions and fixed comparison

Use the archived 1,565-row fleet population, all eligible bands, and the existing
50 LOGO/forward/spatial band-folds. Preserve each target's training-only reference
and LOGO's first-48-hour reference-only calibration. Seed 0 is the full campaign.
Repeat seed 1 on the five forward band-folds as a prespecified stability check.

For both CatBoost and frozen pretrained TabPFN v2:

1. Global original-13-feature predictor.
2. Global original predictors plus six targeted physical predictors.
3. Two direct regional experts, geography/topography gate.
4. Two direct experts, topographic/water-column gate.
5. Two direct experts, environmental forcing gate.
6. An inner-selected gate/blend arm; its choice uses training data only.

Each expert uses the original 13 predictors. This separates model inputs from
gate inputs. Experts predict targets directly, not cross-fitted residuals: the
preceding audit found residual corrections harmful even without routing. Gates
are standardized training-only two-cluster KMeans. Both families fit experts on
the same hard cluster memberships; prediction blends them with soft weights.
Require at least 80 rows, 24 unique hours and two gliders per expert, otherwise
use the global predictor. Blend global and expert predictions with alpha selected
from {0, .25, .5, 1} by pooled inner validation MSE. Prefer smaller alpha on ties.
For a query farther from its nearest centroid than the training 99th-percentile
distance, use the global predictor. Record out-of-support fractions and activation.
Also retain unshrunk expert outputs as explicitly diagnostic comparisons.

Inner validation matches the outer question: three grouped glider folds with
48-hour reference calibration; two forward cutoffs (.55/.75) with a 24-hour gap;
or fixed buffered quadrants. Recompute reference medians inside each inner fold.
Require 100 training, 24 validation and 12 reference observations. If no inner
fold is viable, retain the original global predictor and record the reason.

CatBoost retains 400 iterations, depth 6, learning rate .06, L2=1 and four CPU
threads. TabPFN retains the public v2 checkpoint, four estimators and CUDA.
Compare these fixed configurations; do not call them capacity-matched.
Derived CSVs are loaded with round-trip float precision so the original 13
predictors and raw targets exactly retain their archived binary values. TabPFN
retains the original memory-saving inference setting; disable memory chunking
only during gradient computation, restoring normal inference at checkpoints.

## Physical hypotheses

Kaena observations support considering topographic slope, current direction
relative to topography, and depth above bottom; wind/wave conditions motivate
surface-noise regimes. They do not show that those mechanisms dominate these
specific recorded acoustic bands. Historical field sites are not assumed
identical to the glider tracks.

Use cached bathymetry to derive the gradient of positive-down water depth H.
Let g=grad(H), s=|g| and u=(current_u,current_v). Added predictors:
seafloor slope; relative depth; upslope current -u dot g / s; along-isobath
current (-u_x*g_y+u_y*g_x)/s; signed topographic forcing proxy -u dot g;
and deep-water wave-steepness proxy 2*pi*Hs/(gravity*Tp^2).
At flat cells, directional current proxies are zero. Mark out-of-grid or
invalid wave/depth inputs missing, and fit imputation only on training data.
The forcing proxy is not a measurement of vertical velocity or mixing. Model
currents at glider depth are not observed bottom tidal currents. No claim of
measured tide phase, critical slope or density stratification is made.

Gate sets:
- geography: east/north km, bottom depth, slope;
- topography: bottom depth, slope, relative depth;
- forcing: wind speed, wave height, wave steepness, upslope current.

Primary literature motivating, not validating, these hypotheses:
- https://journals.ametsoc.org/view/journals/phoc/36/6/jpo2888.1.xml
- https://journals.ametsoc.org/view/journals/phoc/38/2/2007jpo3728.1.xml
- https://arxiv.org/abs/2403.18728

## Supervised fine-tuning

Use installed TabPFN 2.0.9 and the same v2 checkpoint. Retain its actual
preprocessing and four-estimator inference, not a different simplified network
baseline. Compare frozen weights, decoder-only updates and full-network updates.
Reset weights for every fold/mode. Optimize episodic bar-distribution loss with
AdamW lr=1e-5, weight decay=.01, gradient clipping=1; query/context examples are
drawn only from permitted training rows. Use at most 192 context and 32 query rows
per episode; prefer episodes separated by glider/time/space to match the task.
Cycle the four fitted estimator preprocessing configurations during training.
Transform query targets using each context-fitted target transform.

Candidate checkpoints: 0, 8, 32 optimizer steps. Select steps separately for each
update mode by pooled inner validation MSE. Then restart the original checkpoint,
fit on all outer-training rows, and update for the selected number of steps.
Zero steps is a valid selection. Outer scoring labels are never used by training
or checkpoint selection. Record losses, gradients, parameter changes, selected
steps, runtime and peak GPU memory. Compare against frozen inference with the
identical preprocessing and context. This is a bounded adaptation experiment,
not a claim that all fine-tuning configurations have been exhausted.

## Execution and reporting

Dedicated new source directory and new run directories. After confirming both
GPUs idle, use GPU 1 for the expert campaign and GPU 0 for fine-tuning, with one
process per GPU and four CPU threads per process. The two campaigns have separate
outputs and share only read-only inputs/checkpoint files. Two-hour runner budgets
with resume and three-hour shell limits apply to each invocation.
Run meaningful split/feature/gate tests and a real-data execution smoke before
the full campaign. Freeze source/data identity in each run and save predictions,
inner choices, diagnostics and manifests. A bounded stop is not completion.

Report every band and protocol, pooled/macro/worst-fold error, paired UTC-day
bootstrap differences against the same family's original-feature global model,
expert use and physical feature effects. Highlight overlapping SPWH/UNDO and the
small number of independent gliders/days. A gain must survive the intended outer
task; negative or inconsistent results are valid outcomes. Report the observed
input policy explicitly; these physics inputs are not all planning-time inputs.
