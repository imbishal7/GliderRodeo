"""Step 3 - build the training table: one row per glider-hour.

Columns:
  keys        glider, hour_utc, lat, lon
  features    wind_ms, rain_mmhr, hs_m, tp_s, swell_m, cur_ms, temp_c, seafloor_m,
              hour_sin, hour_cos, depth_m, depth_rate_ms, frac_surface
  targets     y_FIWH ... y_BBWH  = band level minus that glider's own median for the band

Working in anomalies removes the per-recorder calibration offset (the fleet disagrees by
up to 57 dB), so the model learns how conditions move the noise, not how loud each
hydrophone reads.

Outputs: data/train.csv, data/glider_medians.csv
"""
import numpy as np
import pandas as pd

import config
from io_utils import add_time_features, epoch_ns, log, nearest_index, to_ns


class Grid:
    """Regular (time, depth, lat, lon) grid from an ERDDAP long table."""

    def __init__(self, df, variables, has_depth=True):
        tns = epoch_ns(df["time"])
        self.t = np.unique(tns)
        self.z = np.sort(df["depth"].unique()) if has_depth else np.array([0.0])
        self.la, self.lo = np.sort(df["latitude"].unique()), np.sort(df["longitude"].unique())
        ti = np.searchsorted(self.t, tns)
        zi = np.searchsorted(self.z, df["depth"].to_numpy()) if has_depth else np.zeros(len(df), int)
        ai = np.searchsorted(self.la, df["latitude"].to_numpy())
        oi = np.searchsorted(self.lo, df["longitude"].to_numpy())
        self.v = {}
        for var in variables:
            arr = np.full((len(self.t), len(self.z), len(self.la), len(self.lo)), np.nan)
            arr[ti, zi, ai, oi] = df[var].to_numpy()
            self.v[var] = arr

    def _cell(self, t, lat, lon):
        return (nearest_index(self.t, np.asarray(t, dtype="int64")),
                nearest_index(self.la, lat), nearest_index(self.lo, lon))

    def surface(self, var, t, lat, lon, radius=4):
        ti, ai, oi = self._cell(t, lat, lon)
        vals = self.v[var][ti, 0, ai, oi].copy()
        arr = self.v[var]
        for k in np.where(~np.isfinite(vals))[0]:          # land / masked cell: nearest valid
            patch = arr[ti[k], 0, max(ai[k] - radius, 0):ai[k] + radius + 1,
                        max(oi[k] - radius, 0):oi[k] + radius + 1]
            if np.isfinite(patch).any():
                vals[k] = np.nanmean(patch)
        return vals

    def at_depth(self, var, t, lat, lon, depth):
        ti, ai, oi = self._cell(t, lat, lon)
        prof = self.v[var][ti, :, ai, oi]
        out = np.full(len(ti), np.nan)
        for k in range(len(ti)):
            ok = np.isfinite(prof[k])
            if ok.any():
                out[k] = np.interp(np.clip(depth[k], self.z[ok][0], self.z[ok][-1]), self.z[ok], prof[k][ok])
        return out


def hourly(minutes):
    m = minutes.copy()
    m["time"] = to_ns(m["time"])
    m["hour_utc"] = m["time"].dt.floor("h")
    m["deep"] = m["depth"] >= config.SURFACE_DEPTH_M
    deep = m[m["deep"]]
    codes = list(config.BANDS)

    base = m.groupby(["glider", "hour_utc"]).agg(
        lat=("lat", "mean"), lon=("lon", "mean"),
        depth_m=("depth", "mean"), depth_rate_ms=("depth_rate_ms", "mean"),
        n_minutes=("time", "size"), frac_surface=("deep", lambda s: 1 - s.mean()))
    levels = deep.groupby(["glider", "hour_utc"])[codes].median()
    n_deep = deep.groupby(["glider", "hour_utc"]).size().rename("n_deep")
    out = base.join(levels).join(n_deep).reset_index()
    out["n_deep"] = out["n_deep"].fillna(0).astype(int)
    return out


def main():
    codes = list(config.BANDS)
    log("aggregating minutes to glider-hours")
    hours = hourly(pd.read_csv(config.DATA / "noise_minutes.csv.gz"))
    hours = hours.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    log(f"  {len(hours):,} glider-hours with a position")

    t = epoch_ns(hours["hour_utc"])
    lat, lon = hours["lat"].to_numpy(), hours["lon"].to_numpy()
    depth = hours["depth_m"].fillna(400.0).clip(lower=config.SURFACE_DEPTH_M).to_numpy()

    log("joining WRF wind and rain")
    wrf = pd.read_csv(config.ENV / "wrf.csv.gz")
    wrf["wind"] = np.hypot(wrf["Uwind"], wrf["Vwind"])
    wrf["rain_mmhr"] = wrf["rain"] * 3600
    gw = Grid(wrf, ["wind", "rain_mmhr"], has_depth=False)
    hours["wind_ms"] = gw.surface("wind", t, lat, lon)
    hours["rain_mmhr"] = gw.surface("rain_mmhr", t, lat, lon)

    log("joining WaveWatch III")
    g3 = Grid(pd.read_csv(config.ENV / "ww3.csv.gz"), ["Thgt", "Tper", "shgt"], has_depth=False)
    hours["hs_m"] = g3.surface("Thgt", t, lat, lon)
    hours["tp_s"] = g3.surface("Tper", t, lat, lon)
    hours["swell_m"] = g3.surface("shgt", t, lat, lon)

    log("joining ROMS temperature and currents at glider depth")
    roms = pd.read_csv(config.ENV / "roms.csv.gz")
    gr = Grid(roms, ["temp", "u", "v"])
    hours["temp_c"] = gr.at_depth("temp", t, lat, lon, depth)
    hours["cur_ms"] = np.hypot(gr.at_depth("u", t, lat, lon, depth), gr.at_depth("v", t, lat, lon, depth))

    log("joining bathymetry")
    b = pd.read_csv(config.ENV / "bathy.csv.gz")
    bla, blo = np.sort(b.latitude.unique()), np.sort(b.longitude.unique())
    z = b.pivot(index="latitude", columns="longitude", values="z").loc[bla, blo].to_numpy()
    hours["seafloor_m"] = -z[nearest_index(bla, lat), nearest_index(blo, lon)]

    hours = add_time_features(hours)

    log("converting band levels to per-glider anomalies")
    medians = hours.groupby("glider")[codes].median()
    for c in codes:
        hours[f"y_{c}"] = hours[c] - hours["glider"].map(medians[c])
    medians.to_csv(config.DATA / "glider_medians.csv")

    hours.to_csv(config.DATA / "train.csv", index=False)
    log(f"step 3 done: {len(hours):,} rows -> data/train.csv")
    print(hours.groupby("glider")[["wind_ms", "hs_m", "temp_c", "cur_ms", "depth_m"]].median().round(2).to_string())
    print("\nrows with a usable target per band:")
    print(hours[[f"y_{c}" for c in codes]].notna().sum().to_string())


if __name__ == "__main__":
    main()
