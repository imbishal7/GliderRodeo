"""Feature construction and an auditable, consistent target population."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from build_table import Grid
from io_utils import epoch_ns, nearest_index

BASE_FEATURES = list(config.FEATURE_NAMES)
GEOGRAPHIC_FEATURES = ["east_km", "north_km", "height_above_bottom_m", "relative_depth"]
OPTIONAL_FEATURES = ["seafloor_slope", "wind_u_ms", "wind_v_ms", "current_u_ms",
                     "current_v_ms", "salinity_psu"]
TIDAL_FEATURES = ["m2_sin", "m2_cos"]
# Attribution split of the enhanced set: horizontal position vs added physical
# predictors. `hgb_geo` bundles both; these separate them on identical folds.
COORDINATE_FEATURES = ["east_km", "north_km"]
PHYSICAL_FEATURES = ["height_above_bottom_m", "relative_depth"] + OPTIONAL_FEATURES
FORECAST_HELD = ["cur_ms", "temp_c", "seafloor_m", "depth_m", "depth_rate_ms",
                 "frac_surface", "height_above_bottom_m", "relative_depth",
                 "current_u_ms", "current_v_ms", "salinity_psu", "seafloor_slope"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def geographic_features(frame):
    d = frame.copy()
    d["hour_utc"] = pd.to_datetime(d.hour_utc, utc=True)
    lat0, lon0 = config.CENTRE
    d["east_km"] = (d.lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
    d["north_km"] = (d.lat - lat0) * 111.32
    positive_bottom = d.seafloor_m.where(d.seafloor_m > 0)
    d["height_above_bottom_m"] = positive_bottom - d.depth_m
    d["relative_depth"] = d.depth_m / positive_bottom
    # A harmonic clock proxy, not a locally calibrated tide prediction.
    elapsed = (d.hour_utc - pd.Timestamp("2026-01-01", tz="UTC")).dt.total_seconds() / 3600
    phase = 2 * np.pi * elapsed / 12.4206012
    d["m2_sin"], d["m2_cos"] = np.sin(phase), np.cos(phase)
    return d


def enrich(frame, env_dir):
    d = geographic_features(frame)
    audit = {"environment_sources": {}, "omitted_optional_features": []}
    t, lat, lon = epoch_ns(d.hour_utc), d.lat.to_numpy(), d.lon.to_numpy()
    for filename, variables, names, at_depth in [
        ("wrf.csv.gz", ["Uwind", "Vwind"], ["wind_u_ms", "wind_v_ms"], False),
        ("roms.csv.gz", ["u", "v", "salt"],
         ["current_u_ms", "current_v_ms", "salinity_psu"], True),
    ]:
        source = Path(env_dir) / filename
        if not source.exists():
            audit["omitted_optional_features"].extend(names)
            continue
        source_df = pd.read_csv(source)
        grid = Grid(source_df, variables, has_depth=at_depth)
        valid = ((lat >= grid.la.min()) & (lat <= grid.la.max()) &
                 (lon >= grid.lo.min()) & (lon <= grid.lo.max()) &
                 (t >= grid.t.min()) & (t <= grid.t.max()))
        for var, name in zip(variables, names):
            values = (grid.at_depth(var, t, lat, lon, d.depth_m.to_numpy()) if at_depth
                      else grid.surface(var, t, lat, lon))
            d[name] = np.where(valid, values, np.nan)
        audit["environment_sources"][filename] = {
            "sha256": sha256(source), "out_of_domain_rows": int((~valid).sum())}
    bathy_file = Path(env_dir) / "bathy.csv.gz"
    if bathy_file.exists():
        bathy = pd.read_csv(bathy_file).pivot(index="latitude", columns="longitude", values="z")
        bathy = bathy.sort_index().sort_index(axis=1)
        la, lo = bathy.index.to_numpy(), bathy.columns.to_numpy()
        if min(len(la), len(lo)) >= 2:
            dy, dx = np.gradient(-bathy.to_numpy(), la * 111320,
                                 lo * 111320 * np.cos(np.deg2rad(config.CENTRE[0])))
            slope = np.hypot(dx, dy)
            valid = (lat >= la.min()) & (lat <= la.max()) & (lon >= lo.min()) & (lon <= lo.max())
            d["seafloor_slope"] = np.where(valid, slope[nearest_index(la, lat), nearest_index(lo, lon)], np.nan)
            audit["environment_sources"][bathy_file.name] = {"sha256": sha256(bathy_file)}
        else:
            audit["omitted_optional_features"].append("seafloor_slope")
    else:
        audit["omitted_optional_features"].append("seafloor_slope")
    return d.replace([np.inf, -np.inf], np.nan), audit


def load_population(table, coverage_file, *, min_deep=15, allow_unknown_coverage=False):
    d = pd.read_csv(table)
    required = {"glider", "hour_utc", "lat", "lon", *BASE_FEATURES, *config.BANDS}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Training table lacks {sorted(missing)}. Raw band levels are required; y_* alone is insufficient.")
    d["hour_utc"] = pd.to_datetime(d.hour_utc, utc=True, errors="raise")
    if d.duplicated(["glider", "hour_utc"]).any():
        raise ValueError("Duplicate glider-hour rows; resolve these before splitting.")
    start, end = map(lambda s: pd.Timestamp(s, tz="UTC"), config.MISSION_DAYS)
    audit = {"source_rows": len(d), "table_sha256": sha256(table), "min_deep": min_deep}
    keep = d.hour_utc.between(start, end, inclusive="left") & d.lat.notna() & d.lon.notna()
    if min_deep:
        if "n_deep" not in d:
            raise ValueError("n_deep missing: cannot enforce the deep-observation quality threshold.")
        keep &= d.n_deep >= min_deep
    d = d.loc[keep].sort_values(["hour_utc", "glider"]).reset_index(drop=True)
    audit["retained_rows"] = len(d)
    if not len(d):
        raise ValueError("No rows remain after mission/time/quality filtering.")
    if coverage_file and Path(coverage_file).exists():
        coverage = pd.read_csv(coverage_file)
        if coverage.duplicated(["glider", "band"]).any():
            raise ValueError("Duplicate glider-band coverage records.")
        audit["coverage_sha256"] = sha256(coverage_file)
        audit["coverage_policy"] = "full frequency coverage only"
        for band in config.BANDS:
            statuses = coverage.loc[coverage.band == band].set_index("glider").status
            covered = d.glider.map(statuses).eq("full")
            audit[f"{band}_excluded_partial_or_unknown"] = int((d[band].notna() & ~covered).sum())
            d.loc[~covered, band] = np.nan
    elif not allow_unknown_coverage:
        raise FileNotFoundError("band_coverage.csv is required. Explicit --allow-unknown-coverage is for exploratory runs only.")
    else:
        audit["coverage_policy"] = "UNKNOWN: explicit exploratory override"
    audit["band_counts"] = {b: int(d[b].notna().sum()) for b in config.BANDS}
    audit["glider_counts"] = d.glider.value_counts().to_dict()
    d["row_id"] = np.arange(len(d))
    return d, audit


def feature_set(frame, kind="enhanced", include_tide=False):
    """Predictor list for a comparison arm.

    base      original 13 predictors
    coords    original + local east/north coordinates only
    physical  original + added physical predictors, no coordinates
    enhanced  original + coordinates + physical (the bundled `hgb_geo` set)
    """
    names = list(BASE_FEATURES)
    if kind == "base":
        return names
    if kind == "coords":
        return names + [f for f in COORDINATE_FEATURES if f in frame]
    if kind == "physical":
        return names + [f for f in PHYSICAL_FEATURES if f in frame]
    if kind == "enhanced":
        names += GEOGRAPHIC_FEATURES + [f for f in OPTIONAL_FEATURES if f in frame]
        if include_tide:
            names += TIDAL_FEATURES
        return names
    raise ValueError(f"unknown feature set: {kind}")


def selected_features(frame, enhanced=True, include_tide=False):
    return feature_set(frame, "enhanced" if enhanced else "base", include_tide)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=config.DATA / "train.csv")
    parser.add_argument("--coverage", type=Path, default=config.DATA / "band_coverage.csv")
    parser.add_argument("--env-dir", type=Path, default=config.ENV)
    parser.add_argument("--output", type=Path, default=config.DATA / "experiments.csv")
    parser.add_argument("--min-deep", type=int, default=15)
    parser.add_argument("--allow-unknown-coverage", action="store_true")
    args = parser.parse_args()
    d, audit = load_population(args.table, args.coverage, min_deep=args.min_deep,
                               allow_unknown_coverage=args.allow_unknown_coverage)
    d, extra = enrich(d, args.env_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.output, index=False)
    audit.update(extra)
    audit["output_sha256"] = sha256(args.output)
    audit["features"] = selected_features(d)
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
