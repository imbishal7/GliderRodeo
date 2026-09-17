"""Step 1 - download the AQUAVIEW / PacIOOS conditions for the Rodeo box.

  wrf.csv.gz    WRF Oahu       wind components + rain, hourly, ~1 km
  ww3.csv.gz    WaveWatch III  wave height / period / swell, hourly, 5 km
  roms.csv.gz   ROMS HIIG      temperature, salinity, currents, 3-hourly, 4 km, 0-1000 m
  bathy.csv.gz  HMRG           seafloor depth, 1 km

Requests are split by day and cached in data/env/raw, so reruns only fetch what's missing.
"""
import argparse

import pandas as pd

import config
from io_utils import fetch_csv, griddap_url, log

RAW = config.ENV / "raw"
B = config.BOX


def days():
    return pd.date_range(*config.MISSION_DAYS, freq="D", inclusive="left")


def daily(dataset, variables, tag, depth_axis=None, stride=1):
    frames = []
    for d in days():
        axes = [(d.strftime("%Y-%m-%dT00:00:00Z"), 1, d.strftime("%Y-%m-%dT23:59:00Z"))]
        if depth_axis:
            axes.append(depth_axis)
        axes += [(B["lat_min"], stride, B["lat_max"]), (B["lon_min"], stride, B["lon_max"])]
        df = fetch_csv(griddap_url(dataset, variables, axes), RAW / tag / f"{d:%Y%m%d}.csv")
        if len(df):
            frames.append(df)
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    log(f"  {dataset}: {len(out):,} rows, {out['time'].nunique()} times")
    return out


def main():
    log("WRF Oahu: wind + rain")
    daily("wrf_oa", ["Uwind", "Vwind", "rain"], "wrf", stride=4).to_csv(config.ENV / "wrf.csv.gz", index=False)

    log("WaveWatch III Hawaii: waves")
    daily("ww3_hawaii_lon180", ["Thgt", "Tper", "shgt"], "ww3",
          depth_axis=(0.0, 1, 0.0)).to_csv(config.ENV / "ww3.csv.gz", index=False)

    log("ROMS HIIG: temperature, salinity, currents")
    daily("roms_hiig_assim", ["temp", "salt", "u", "v"], "roms",
          depth_axis=(0.0, 1, 1000.0)).to_csv(config.ENV / "roms.csv.gz", index=False)

    log("HMRG bathymetry")
    url = griddap_url("hmrg_bathytopo_1km_mhi", ["z"],
                      [(B["lat_min"], 1, B["lat_max"]), (B["lon_min"], 1, B["lon_max"])])
    df = fetch_csv(url, RAW / "bathy" / "hmrg.csv")
    log(f"  hmrg_bathytopo_1km_mhi: {len(df):,} cells")
    df.to_csv(config.ENV / "bathy.csv.gz", index=False)
    log("step 1 done")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
