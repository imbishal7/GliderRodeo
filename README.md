# Noise Nowcaster

Predicts how loud the ocean is in five whale-call frequency bands from the weather and
sea state, and converts that into how far a glider can hear.

Training data: **AQUAVIEW / PacIOOS** supplies every environmental input (~80%); the
**Glider Rodeo** hydrophones supply the labels and the glider's own state (~20%).

The Rodeo detection files are synthetic, so nothing here uses them. The labels are
measured noise levels, which are real.

## Quick start

```bash
cd noise-nowcaster
python3 run_all.py            # ~4 min first time, ~1 min after (downloads are cached)
python3 serve.py              # then open http://localhost:8000
python3 -m unittest discover -s tests
```

Needs `numpy pandas h5py scikit-learn joblib requests` (see `requirements.txt`).
The Rodeo archive is expected at `../hackathon-data`.

## Results

Each glider is predicted by a model trained on the other five, so it is always scored on
a glider it has never seen.

**How to read the score.** Two errors are compared: how far the model's prediction lands
from the measured level, and how far you would land by simply assuming conditions are
normal for that glider. "15% better than guessing" means the model's typical error is 15%
smaller. 0% means the model adds nothing.

| Band | Call | Model off by | Guessing normal off by | Better by | Verdict |
|---|---|---|---|---|---|
| SPWH | Sperm whale clicks | 5.55 dB | 6.55 dB | **15%** | clearly better |
| UNDO | Odontocete whistles | 5.57 dB | 6.48 dB | **14%** | clearly better |
| BBWH | Beaked whale upsweeps | 4.61 dB | 5.34 dB | **14%** | clearly better |
| HUWH | Humpback song | 5.79 dB | 6.20 dB | 7% | slightly better |
| FIWH | Fin whale 20 Hz | 6.99 dB | 6.89 dB | 0% | no better than guessing |

(The same numbers as a variance-based skill score, if you prefer that convention:
+0.28, +0.26, +0.25, +0.13, −0.03. They are in `out/metrics.json` as `skill_vs_median`.)

**Top drivers** (permutation importance, averaged over bands): `wind_ms` 0.376,
`depth_rate_ms` 0.102, `hs_m` 0.029, `temp_c` 0.026.

Two findings worth reporting:

1. **Wind dominates**, as the physics says it should, and the model recovers that
   without being told.
2. **The fin whale band has no skill.** It sits at 10–30 Hz, where shipping dominates,
   and there is no AIS data for this window (AQUAVIEW's is annual and stops at 2025).
   Weather explains four bands and not this one — that is the honest result, and the
   argument for getting vessel data next time.

Over the mission, measured noise in the sperm whale band swung from −7.9 dB to +5.8 dB
around its median, which is a **23× change in monitored area** between a quiet hour and
a loud one.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1 | `fetch_env.py` | `data/env/{wrf,ww3,roms,bathy}.csv.gz` from PacIOOS ERDDAP |
| 2 | `noise_bands.py` | `data/noise_minutes.csv.gz` — 117,487 glider-minutes, 5 bands + glider state |
| 3 | `build_table.py` | `data/train.csv` — 1,590 glider-hours with features and targets |
| 4–5 | `train.py` | `out/model.joblib`, `metrics.json`, `hindcast.csv`, `drivers.csv` |
| 6 | `detectability.py` | `out/detectability.json` — noise to range/area |
| 7 | `forecast.py` | `out/forecast.json` — live PacIOOS forecast run forward |
| UI | `serve.py` + `web/` | local web app |

### Inputs

From AQUAVIEW: `wrf_oa` (wind, rain), `ww3_hawaii_lon180` (wave height, period, swell),
`roms_hiig_assim` (temperature and currents at the glider's depth),
`hmrg_bathytopo_1km_mhi` (seafloor depth), plus hour-of-day.

From the Rodeo: `noise data/*.h5` band levels, and glider depth, dive rate and
surface fraction from the science, engineering and GPS files.

### Target

Band level **minus that glider's own median** for the band. The recorders disagree by
up to 57 dB in absolute terms (two of them use a different glider's calibration curve),
so the model learns how conditions move the noise, not how loud each hydrophone reads.
Everything downstream is therefore relative: "3 dB louder than usual", never "82 dB".

## Web UI

`python3 serve.py` then <http://localhost:8000>. Standard library only.

- **Try it** — move sliders for wind, waves, rain, glider depth and dive rate; see the
  predicted noise and listening range per species update live.
- **Hindcast** — predicted vs measured for each held-out glider, with its scores.
- **What drives noise** — permutation importance per band. Orange bars are glider
  state, so importance there is self-noise rather than weather.
- **Forecast** — the model run on the live PacIOOS forecast; "Refresh from PacIOOS"
  re-fetches. Bands without skill are drawn dashed.
- **About** — method and limitations.

## Limitations

- Relative decibels only, by design (see Target above).
- Range percentages assume spherical spreading: a guide, not a propagation model.
- 1,590 training hours is small. Gradient-boosted trees beat anything deeper here.
- Forecast features the model needs but a forecast cannot supply (currents,
  temperature, glider depth and dive rate) are held at the mission median; the JSON
  records which.
- belladonna is excluded: a GPS fix 2,120 km from the box, salinity from 1.81 to
  40.08 PSU, and acoustics uncorrelated with the rest of the fleet.
- Source levels in `config.BANDS` are literature values and are only used for labelling.
