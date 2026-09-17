"""Step 7 - run the model forward on the live PacIOOS forecast.

WRF Oahu and WaveWatch III publish several days ahead, so the same model that
hindcasts the mission can say what listening conditions will be like this week.

Features that a forecast cannot supply (currents at depth, water temperature,
seafloor, glider depth and dive activity) are held at the mission median; the
JSON records which those were.

Output: out/forecast.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from joblib import load

import config
from detectability import area_ratio, range_ratio
from io_utils import fetch_csv, griddap_url, log, to_ns

LAT, LON = config.CENTRE


def pull(dataset, variables, days, depth_axis=None):
    """Ask for now -> the end of the dataset; ERDDAP's `last` clamps to whatever
    the current forecast horizon happens to be."""
    now = pd.Timestamp.now(tz="UTC").floor("h")
    axes = [(now.strftime("%Y-%m-%dT%H:00:00Z"), 1, "last")]
    if depth_axis:
        axes.append(depth_axis)
    axes += [(LAT, 1, LAT), (LON, 1, LON)]
    return fetch_csv(griddap_url(dataset, variables, axes),
                     config.DATA / "env" / "raw" / "forecast" / f"{dataset}.csv", use_cache=False)


def main(days=5):
    bundle = load(config.OUT / "model.joblib")
    models, features = bundle["models"], bundle["features"]
    train = pd.read_csv(config.DATA / "train.csv")
    defaults = {f: float(train[f].median()) for f in features}

    log(f"pulling the PacIOOS forecast, next {days} days")
    wrf = pull("wrf_oa", ["Uwind", "Vwind", "rain"], days)
    ww3 = pull("ww3_hawaii_lon180", ["Thgt", "Tper", "shgt"], days, depth_axis=(0.0, 1, 0.0))
    if wrf.empty or ww3.empty:
        raise SystemExit("forecast unavailable from PacIOOS right now")

    wrf["time"] = to_ns(wrf["time"])
    ww3["time"] = to_ns(ww3["time"])
    df = wrf.merge(ww3, on="time", how="inner", suffixes=("", "_w"))
    df = df[df["time"] <= pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=days)].reset_index(drop=True)
    df["wind_ms"] = np.hypot(df["Uwind"], df["Vwind"])
    df["rain_mmhr"] = df["rain"] * 3600
    df = df.rename(columns={"Thgt": "hs_m", "Tper": "tp_s", "shgt": "swell_m"})
    h = df["time"].dt.hour + df["time"].dt.minute / 60
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)

    forecastable = {f for f, ok in config.FEATURES if ok}
    held = sorted(set(features) - set(df.columns))
    for f in held:
        df[f] = defaults[f]

    X = df[features]
    out = dict(
        generated_utc=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ"),
        held_at_median={f: round(defaults[f], 3) for f in held},
        horizon_hours=int(len(df)),
        hours=[t.strftime("%Y-%m-%dT%H:%MZ") for t in df["time"]],
        wind_ms=[round(float(v), 2) for v in df["wind_ms"]],
        hs_m=[round(float(v), 2) for v in df["hs_m"]],
        bands={})
    for code, model in models.items():
        p = model.predict(X)
        out["bands"][code] = dict(
            name=config.BANDS[code]["name"],
            anomaly_db=[round(float(v), 2) for v in p],
            range_pct=[round(float(v), 1) for v in 100 * range_ratio(p)],
            area_pct=[round(float(v), 1) for v in 100 * area_ratio(p)])

    json.dump(out, open(config.OUT / "forecast.json", "w"), indent=1)
    log(f"step 7 done: {len(df)} forecast hours -> out/forecast.json")
    best = int(np.argmin(out["bands"]["SPWH"]["anomaly_db"]))
    worst = int(np.argmax(out["bands"]["SPWH"]["anomaly_db"]))
    print(f"  quietest hour ahead: {out['hours'][best]}  wind {out['wind_ms'][best]} m/s  "
          f"range {out['bands']['SPWH']['range_pct'][best]}% of normal")
    print(f"  loudest  hour ahead: {out['hours'][worst]}  wind {out['wind_ms'][worst]} m/s  "
          f"range {out['bands']['SPWH']['range_pct'][worst]}% of normal")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=5)
    main(p.parse_args().days)
