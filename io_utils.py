"""ERDDAP download with cache + retry, time parsing, small helpers."""
import io
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_time(series):
    """Epoch seconds or ISO strings -> tz-aware UTC, nanosecond unit."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.99:
        out = pd.to_datetime(num, unit="s", utc=True)
    else:
        out = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    return out.dt.as_unit("ns")


def griddap_url(dataset, variables, axes):
    """axes: list of (start, stride, stop) in dataset axis order.

    A bound may be the string "last", which ERDDAP clamps to the axis maximum -
    needed for forecasts, whose horizon changes through the day.
    """
    def bound(v):
        return "last" if v == "last" else f"({v})"
    constraint = "".join(f"[{bound(a)}:{s}:{bound(b)}]" for a, s, b in axes)
    query = ",".join(f"{v}{constraint}" for v in variables)
    return f"{config.ERDDAP}/{dataset}.csv?{urllib.parse.quote(query, safe='(),:=')}"


def fetch_csv(url, cache_path, retries=4, timeout=600, use_cache=True):
    cache_path = Path(cache_path)
    if use_cache and cache_path.exists() and cache_path.stat().st_size > 0:
        return pd.read_csv(cache_path)
    last = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and not r.text.startswith("Error"):
                df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_path, index=False)
                return df
            if any(s in r.text for s in ("no matching results", "axis maximum", "axis minimum")):
                return pd.DataFrame()          # outside the dataset's range: not an error
            last = f"HTTP {r.status_code}: {r.text[:160]}"
        except requests.RequestException as exc:
            last = str(exc)
        wait = min(45, 5 * attempt)
        log(f"  retry {attempt}/{retries} in {wait}s ({last})")
        time.sleep(wait)
    raise RuntimeError(f"ERDDAP failed: {url}\n{last}")


def to_ns(series):
    """Parse to tz-aware UTC at NANOSECOND resolution.

    pandas 3 parses strings to microseconds by default; mixing units silently breaks
    epoch-integer joins, so every timestamp in this project goes through here.
    """
    return pd.to_datetime(series, utc=True).dt.as_unit("ns")


def epoch_ns(series):
    return to_ns(series).astype("int64").to_numpy()


def nearest_index(sorted_values, x):
    v, x = np.asarray(sorted_values), np.asarray(x)
    i = np.clip(np.searchsorted(v, x), 1, len(v) - 1)
    return np.where(np.abs(x - v[i - 1]) <= np.abs(x - v[i]), i - 1, i)


def add_time_features(df, col="hour_utc"):
    h = pd.to_datetime(df[col], utc=True).dt.hour + pd.to_datetime(df[col], utc=True).dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * h / 24)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24)
    return df
