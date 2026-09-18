"""Explicit platform, future-time, and buffered geographic holdouts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    name: str
    train: np.ndarray
    test: np.ndarray
    calibration: np.ndarray


def make_folds(d, protocol, *, calibration_hours=48, gap_hours=24, buffer_km=2):
    indices = np.arange(len(d))
    empty = np.array([], dtype=int)
    if protocol == "logo":
        for glider in sorted(d.glider.unique()):
            held = d.glider.eq(glider)
            cutoff = d.loc[held, "hour_utc"].min() + pd.Timedelta(hours=calibration_hours)
            yield Fold(f"glider_{glider}", indices[~held],
                       indices[held & (d.hour_utc >= cutoff)],
                       indices[held & (d.hour_utc < cutoff)])
    elif protocol == "forward":
        times = np.sort(d.hour_utc.unique())
        cutoff = pd.Timestamp(times[min(int(0.7 * len(times)), len(times) - 1)])
        train = d.hour_utc < cutoff - pd.Timedelta(hours=gap_hours)
        # Evaluate only platforms with a reference established before the cutoff.
        test = (d.hour_utc >= cutoff) & d.glider.isin(d.loc[train, "glider"])
        yield Fold("future_30pct", indices[train], indices[test], empty)
    elif protocol == "spatial":
        # Fixed quadrants centred on the configured mission centre, in kilometres.
        for east in (False, True):
            for north in (False, True):
                sx, sy = (1 if east else -1), (1 if north else -1)
                x, y = sx * d.east_km, sy * d.north_km
                held = (x >= 0) & (y >= 0)
                # Euclidean distance to the held quadrant, including the corner.
                distance = np.hypot(np.maximum(-x, 0), np.maximum(-y, 0))
                train = (~held) & (distance >= buffer_km)
                test = held & d.glider.isin(d.loc[train, "glider"])
                label = ("N" if north else "S") + ("E" if east else "W")
                yield Fold(label, indices[train], indices[test], empty)
    else:
        raise ValueError(f"Unknown protocol {protocol}")


def fold_targets(d, band, fold, *, min_reference=12):
    """References use training or designated warmup labels, never scored labels."""
    training = d.iloc[fold.train]
    reference_rows = d.iloc[np.concatenate([fold.train, fold.calibration])]
    stats = reference_rows.groupby("glider")[band].agg(["median", "count"])
    medians = stats.loc[stats["count"] >= min_reference, "median"]
    train_ok = training[band].notna() & training.glider.isin(medians.index)
    test = d.iloc[fold.test]
    test_ok = test[band].notna() & test.glider.isin(medians.index)
    tr, te = fold.train[train_ok], fold.test[test_ok]
    y_train = (d.iloc[tr][band] - d.iloc[tr].glider.map(medians)).to_numpy()
    y_test = (d.iloc[te][band] - d.iloc[te].glider.map(medians)).to_numpy()
    if set(tr) & set(te) or set(te) & set(fold.calibration):
        raise AssertionError("Train, calibration, and scoring populations overlap.")
    return tr, te, y_train, y_test, medians.to_dict()


def fold_scales(d, band, fold, medians, *, min_reference=12, floor=0.5):
    """Per-platform anomaly SCALE from the same permitted rows as the medians.

    Platforms differ ~5x in anomaly spread (1.97-10.79 dB). Centring alone leaves
    that mismatch, so a model trained on wide-spread platforms overshoots on narrow
    ones. Scale is estimated ONLY from training + calibration rows, exactly like the
    medians, so this adds no leakage. Robust (MAD-based); floored to avoid blow-up.
    """
    reference_rows = d.iloc[np.concatenate([fold.train, fold.calibration])]
    out = {}
    for g, grp in reference_rows.groupby("glider"):
        v = grp[band].dropna()
        if len(v) < min_reference or g not in medians:
            continue
        mad = float(np.median(np.abs(v - medians[g]))) * 1.4826
        out[g] = max(mad, floor)
    return out
