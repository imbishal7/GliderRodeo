# Model experiments

These experiments are separate from the web demo. They compare:

- `hgb_base`: original 13 features, original HGB capacity, fixed 400 iterations.
- `hgb_geo`: the same HGB with geographic and physical features.
- `tabpfn_geo`: pretrained TabPFN v2 regression on the enhanced features, on CUDA.
- `moe_geo`: shared HGB plus two soft geographic residual experts, strength 0.25.
- `resnet_geo` (optional): a small residual MLP trained from scratch on CUDA.

The fixed HGB iteration budget disables the original random early-stopping
split. All model settings are declared before outer evaluation; no setting is
chosen from the scored outer folds. MoE corrections use inner whole-glider
cross-fitted residuals. Its two-cluster gate is fitted on training geography,
depth, and slope only, and logs average gate weights by platform.

## Installation on Aquaman

The working copy is `/data/suramya/GliderRodeo`. The isolated environment and
model caches do not modify other research environments. Python 3.10 and CUDA
12.1 wheels are used for compatibility with driver 535 and the 8 GB RTX 3070s.

```bash
uv venv --python python3.10 .venv
uv pip install --python .venv/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python -r experiments/requirements.txt
.venv/bin/python -m pytest tests/test_experiments.py tests/test_pipeline.py
```

TabPFN is pinned to the smaller, public v2 checkpoint supported by package
2.0.9. This tests one specific pretrained model, not the newest TabPFN release.
Each run records installed versions. The first use downloads the checkpoint;
the run launcher disables Hugging Face telemetry. Training data stays on Aquaman.

## Obtain the real inputs

The GitHub code repository does not contain `data/train.csv` or the acoustic
HDF5 archive. A separate checkout of NMFS-PAM-Glider/GliderRodeo contains the
science/engineering/GPS CSVs, but the checked copy has no acoustic HDF5s.

Either supply `data/train.csv`, `data/band_coverage.csv`, and preferably the
cached `data/env/*.csv.gz`, or supply the original archive and build the table:

```bash
export GLIDER_RODEO_DATA=/absolute/path/to/hackathon-data
.venv/bin/python fetch_env.py
.venv/bin/python noise_bands.py
.venv/bin/python build_table.py
```

`GLIDER_RODEO_DATA` must contain the `noise data/` and platform CSV folders
expected in `config.py`. No synthetic detection files are used as labels.

The public `gs://glider-rodeo-data` bucket can be accessed without cloud login:

```bash
.venv/bin/python -m experiments.fetch_public \
  --output /data/suramya/gliderrodeo-runtime/hackathon-data
```

As checked on 2026-09-18 UTC, it has 70 objects and only Risso and Belladonna
acoustic HDF5s. Belladonna is excluded by the existing registry. The downloader
therefore retrieves Risso and its metadata and explicitly reports the other
five active platforms as missing. It verifies object generations, sizes and
MD5 checksums, and maps flat public HDF5 paths into the original expected layout.
For this subset, run `noise_bands.py risso`, not the full default acoustic step.
One platform cannot support LOGO or learned multi-platform residual experts;
MoE falls back to the shared model and records that fact. Use temporal/spatial
results on this subset only as a pilot, never as the proposed fleet comparison.

## Prepare and run

```bash
.venv/bin/python -m experiments.data
bash experiments/run_aquaman.sh --output out/experiments/mission-observed
bash experiments/run_aquaman.sh --output out/experiments/mission-forecast \
  --protocols forward --feature-policy forecast_median
```

Preparation selects the configured mission dates, at least 15 deep acoustic
observations per hour, and full frequency coverage for each band. Targets come
from the raw band columns, never precomputed `y_*`. Unknown coverage requires
an explicit exploratory override and is prominently recorded in the audit.
Comparison populations therefore differ from the README's original scores.

Geographic features are local east/north coordinates, height above bottom,
relative depth, and (when cached source grids exist) seafloor slope, wind/current
components and salinity. Missing optional source grids are recorded. The
optional `--include-tide` adds a 12.4206012-hour harmonic clock proxy, not a tide
model. No temporal lags or same-hour acoustic features are used.

Protocols:

- `logo`: hold out an entire glider, reserve its first 48 hours to estimate its
  band reference, then score later observations. Other gliders use training-only
  references. This is calibrated platform transfer, not zero-calibration transfer
  or a future-time test; other platforms can contribute contemporaneous data.
- `forward`: a shared chronological cutoff at 70% of unique hours; remove 24
  hours before the cutoff from training, score the last 30%. Per-platform
  reference medians come only from training. Test platforms absent from training
  are excluded.
- `spatial`: hold out each quadrant around `config.CENTRE`, with a 2 km buffer
  between the training locations and the held quadrant. Test platforms must also
  occur in training. References come only from training locations. This is a
  known-platform geographic test, not a proof of distinct physical regimes.

Each band needs at least 12 reference observations per platform, 100 training
rows, and 24 test rows. Sparse folds are explicitly skipped and their full
membership and counts are recorded. No full-record median from a scored glider
is used. Partial frequency coverage is excluded separately per band.

`--feature-policy observed` uses available observed environmental/state inputs.
`forecast_median` replaces unavailable test features with medians computed from
training only, matching the demonstration's fallback concept at the supplied
locations. `forecast_only` removes those features from both training and test.
Neither simulates errors in archived weather forecasts; a forecast-issue-time
backtest would require those archives. Static location is assumed supplied.

The launcher uses GPU 1, at most four CPU threads, and an outer three-hour wall
limit. The runner stops between fits after two hours or 256 fits. Override with
CLI options as needed. It does not launch parallel GPU jobs or touch other jobs.
Use `--resume` with identical inputs, source, and options after a bounded stop;
completed fits are skipped. A lock prevents concurrent writes to the same run.
Use a new output directory after changing source or configuration.

## Artifacts

Every run stores input checksums, the prepared table, source snapshot, package
versions, hardware, configuration, calibration medians, row memberships,
per-fold metrics, individual held-out predictions, and events. `REPORT.md`,
`summary.csv`, `summary.json`, RMSE figures, and geographic residual maps are
generated automatically. Reports include pooled, macro-fold, and worst-fold RMSE.
Paired UTC-day bootstrap intervals against `hgb_base` are exploratory and
conditional on the fitted models. The optional neural baseline is enabled by
adding `resnet_geo` to `--models`.

## GPU software smoke test

```bash
.venv/bin/python -m experiments.smoke --output data/synthetic-smoke
bash experiments/run_aquaman.sh \
  --table data/synthetic-smoke/synthetic_experiments.csv \
  --output out/experiments/synthetic-smoke \
  --bands SPWH --models hgb_base hgb_geo tabpfn_geo moe_geo resnet_geo \
  --calibration-hours 12 --gap-hours 12 --min-train 40 --min-test 12 \
  --min-reference 6 --tabpfn-estimators 1 --resnet-epochs 5
```

This generates synthetic data purely to verify execution, split handling,
reporting, and CUDA inference/training. Its metrics are not mission results.
