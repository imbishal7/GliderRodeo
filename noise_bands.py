"""Step 2 - read the Rodeo hydrophone files into per-minute band levels with glider state.

For each active glider:
  * integrate the hybrid-millidecade spectral density levels over each target band,
    clipped to the recorder's Nyquist frequency;
  * attach depth from the science file, falling back to the flight-engineering file
    (the Slocum science files only cover ~1,900 of ~20,000 minutes);
  * attach position from science, engineering, or interpolation between GPS fixes;
  * derive depth rate as a dive-activity proxy.

Output: data/noise_minutes.csv.gz, data/band_coverage.csv
"""
import argparse

import h5py
import numpy as np
import pandas as pd

import config
from io_utils import log, parse_time

TOL_SCIENCE = pd.Timedelta("45s")
TOL_ENG = pd.Timedelta("60s")
GPS_MAX_GAP = pd.Timedelta("6h")
LAT_OK, LON_OK = (20.5, 22.5), (-159.5, -157.0)
CHUNK = 2048


def band_level(levels_db, f_lo, f_hi, band_lo, band_hi):
    """Integrate spectral density levels (dB re 1 uPa^2/Hz) over a band -> dB re 1 uPa."""
    overlap = np.clip(np.minimum(f_hi, band_hi) - np.maximum(f_lo, band_lo), 0, None)
    keep = overlap > 0
    if not keep.any():
        return np.full(levels_db.shape[0], np.nan)
    power = (10 ** (levels_db[:, keep] / 10.0) * overlap[keep][None, :]).sum(axis=1)
    return 10.0 * np.log10(power)


def read_acoustics(glider, meta):
    with h5py.File(config.RODEO / meta["h5"], "r") as h:
        g = h["GliderRodeo"]
        times = pd.to_datetime([t.decode() for t in g["DateTime"][:]],
                               format="%Y%m%dT%H%M%S", utc=True).as_unit("ns")   # ns everywhere
        edges = g["hybridDecFreqHz"][:]
        lower, upper = edges[:, 0], edges[:, 2]
        nyquist = float(upper.max())
        levels = g["hybridMiliDecLevels"]
        out = {"time": times}
        coverage = []
        for code, b in config.BANDS.items():
            hi_eff = min(b["hi"], nyquist)
            if b["lo"] >= nyquist:
                out[code] = np.full(len(times), np.nan)
                coverage.append(dict(glider=glider, band=code, status="not_sampled", nyquist_hz=nyquist))
                continue
            cols = np.where((upper > b["lo"]) & (lower < hi_eff))[0]
            c0, c1 = cols.min(), cols.max() + 1
            vals = np.empty(len(times))
            for r0 in range(0, len(times), CHUNK):
                blk = levels[r0:r0 + CHUNK, c0:c1]
                vals[r0:r0 + CHUNK] = band_level(blk, lower[c0:c1], upper[c0:c1], b["lo"], hi_eff)
            out[code] = vals
            coverage.append(dict(glider=glider, band=code,
                                 status="full" if b["hi"] <= nyquist else "partial", nyquist_hz=nyquist))
    df = pd.DataFrame(out)
    df.insert(0, "glider", glider)
    return df, coverage


def _clean_positions(df):
    bad = ~(df.lat.between(*LAT_OK) & df.lon.between(*LON_OK))
    df.loc[bad, ["lat", "lon"]] = np.nan
    return df


def read_table(path, cols):
    df = pd.read_csv(config.RODEO / path, usecols=list(cols.values()))
    df = df.rename(columns={v: k for k, v in cols.items()})
    df["time"] = parse_time(df["time"])
    df = df.dropna(subset=["time"]).sort_values("time")
    if "lat" in df:
        df = _clean_positions(df)
    return df


def read_gps(meta):
    raw = pd.read_csv(config.RODEO / meta["gps"])
    fmt = meta["gps_format"]
    if fmt == "start_end":
        t0, t1 = parse_time(raw["startTime"]), parse_time(raw["endTime"])
        fixes = pd.DataFrame({"time": t0 + (t1 - t0) / 2, "lat": raw["latitude"], "lon": raw["longitude"]})
    elif fmt == "seaglider":
        fixes = pd.concat([
            pd.DataFrame({"time": parse_time(raw["startTime"]), "lat": raw["startLatitude"], "lon": raw["startLongitude"]}),
            pd.DataFrame({"time": parse_time(raw["endTime"]), "lat": raw["endLatitude"], "lon": raw["endLongitude"]})])
    else:
        tcol = "time_utc" if "time_utc" in raw else "time"
        fixes = pd.DataFrame({"time": parse_time(raw[tcol]), "lat": raw["latitude"], "lon": raw["longitude"]})
    return _clean_positions(fixes.dropna()).dropna().sort_values("time").drop_duplicates("time")


def interp_gps(times, fixes):
    t = times.astype("int64").to_numpy()
    ft = fixes["time"].astype("int64").to_numpy()
    lat = np.interp(t, ft, fixes["lat"].to_numpy(), left=np.nan, right=np.nan)
    lon = np.interp(t, ft, fixes["lon"].to_numpy(), left=np.nan, right=np.nan)
    i = np.clip(np.searchsorted(ft, t), 1, len(ft) - 1)
    far = np.maximum(t - ft[i - 1], ft[i] - t) > GPS_MAX_GAP.value
    lat[far] = np.nan
    lon[far] = np.nan
    return lat, lon


def attach_state(acoustic, meta):
    df = acoustic.sort_values("time")
    sci = read_table(meta["science"], meta["cols"])
    df = pd.merge_asof(df, sci, on="time", tolerance=TOL_SCIENCE, direction="nearest")
    df["pos_source"] = np.where(df.lat.notna(), "science", None)

    if "eng" in meta:
        e = meta["eng"]
        eng = read_table(e["file"], {k: v for k, v in e.items() if k != "file"})
        eng = eng.rename(columns={c: f"eng_{c}" for c in eng.columns if c != "time"})
        df = pd.merge_asof(df, eng, on="time", tolerance=TOL_ENG, direction="nearest")
        fill = df.depth.isna() & df.eng_depth.notna()
        df.loc[fill, "depth"] = df.loc[fill, "eng_depth"]
        if "eng_lat" in df:
            fill = df.lat.isna() & df.eng_lat.notna()
            df.loc[fill, ["lat", "lon"]] = df.loc[fill, ["eng_lat", "eng_lon"]].to_numpy()
            df.loc[fill, "pos_source"] = "engineering"
        df = df.drop(columns=[c for c in df.columns if c.startswith("eng_")])

    missing = df.lat.isna()
    if missing.any():
        fixes = read_gps(meta)
        if len(fixes) >= 2:
            lat, lon = interp_gps(df.loc[missing, "time"], fixes)
            df.loc[missing, "lat"], df.loc[missing, "lon"] = lat, lon
            df.loc[missing & df.lat.notna(), "pos_source"] = "gps_interp"

    dt = df["time"].diff().dt.total_seconds()
    df["depth_rate_ms"] = (df["depth"].diff() / dt).abs().where(dt.between(30, 180))
    return df


def main(only=None):
    frames, coverage = [], []
    for glider, meta in config.GLIDERS.items():
        if meta.get("exclude"):
            log(f"{glider}: EXCLUDED - {meta['exclude_reason']}")
            continue
        if only and glider not in only:
            continue
        ac, cov = read_acoustics(glider, meta)
        coverage += cov
        df = attach_state(ac, meta)
        log(f"{glider}: {len(df):,} minutes | depth {df.depth.notna().mean():.0%} | "
            f"position {df.pos_source.notna().mean():.0%}")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(config.DATA / "noise_minutes.csv.gz", index=False)
    pd.DataFrame(coverage).to_csv(config.DATA / "band_coverage.csv", index=False)
    log(f"step 2 done: {len(out):,} glider-minutes -> data/noise_minutes.csv.gz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gliders", nargs="*")
    main(p.parse_args().gliders)
