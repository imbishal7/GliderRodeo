"""Per-platform, per-band and expert-activation summaries for a completed run.

`report.py` aggregates by protocol/band/model. The fleet deliverable also needs
metrics resolved to individual platforms, and evidence about whether the
mixture-of-experts actually did anything.

    python -m experiments.fleet_summary out/experiments/fleet-observed-seed0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def per_platform(pred):
    """RMSE/MAE per held-out platform. For LOGO the fold IS the platform; for
    forward/spatial a fold mixes platforms, so we resolve rows to their glider."""
    out = []
    for (protocol, band, model, glider), g in pred.groupby(["protocol", "band", "model", "glider"]):
        out.append(dict(protocol=protocol, band=band, model=model, glider=glider, n=len(g),
                        rmse_db=rmse(g.measured, g.predicted),
                        mae_db=float(np.mean(np.abs(g.measured - g.predicted))),
                        zero_rmse_db=rmse(g.measured, np.zeros(len(g)))))
    d = pd.DataFrame(out)
    if len(d):
        d["skill_vs_zero"] = 1 - (d.rmse_db / d.zero_rmse_db) ** 2
    return d


def expert_diagnostics(run_dir):
    """Did the MoE actually engage? Reports activation and gate occupancy."""
    rows = []
    for f in sorted((run_dir / "metrics").glob("*moe_geo.json")):
        m = json.loads(f.read_text())
        det = m.get("details", {}) or {}
        rows.append(dict(protocol=m["protocol"], band=m["band"], fold=m["fold"],
                         active_experts=det.get("active_experts", 0),
                         inner_folds=det.get("inner_folds", 0),
                         fallback=det.get("fallback", ""),
                         gate_weights=json.dumps(det.get("gate_glider_weights", {}))))
    return pd.DataFrame(rows)


def moe_vs_shared(pred):
    """Is MoE numerically distinguishable from the shared learner it corrects?"""
    rows = []
    for (protocol, band, fold), g in pred.groupby(["protocol", "band", "fold"]):
        a = g[g.model == "moe_geo"].sort_values("row_id")
        b = g[g.model == "hgb_geo"].sort_values("row_id")
        if len(a) and len(a) == len(b):
            rows.append(dict(protocol=protocol, band=band, fold=fold, n=len(a),
                             max_abs_diff=float(np.max(np.abs(a.predicted.to_numpy() - b.predicted.to_numpy())))))
    return pd.DataFrame(rows)


def cost(run_dir):
    rows = []
    for f in (run_dir / "metrics").glob("*.json"):
        m = json.loads(f.read_text())
        rows.append(dict(model=m["model"], seconds=m.get("elapsed_seconds", 0.0),
                         gpu_mib=m.get("gpu_peak_allocated_mib", 0.0) or 0.0))
    d = pd.DataFrame(rows)
    return (d.groupby("model").agg(fits=("seconds", "size"), total_s=("seconds", "sum"),
                                   mean_s=("seconds", "mean"), peak_gpu_mib=("gpu_mib", "max"))
            .reset_index()) if len(d) else d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    args = p.parse_args()
    run = args.run_dir
    pred = pd.read_csv(run / "predictions.csv")

    plat = per_platform(pred)
    plat.to_csv(run / "per_platform.csv", index=False)
    moe = expert_diagnostics(run)
    moe.to_csv(run / "expert_diagnostics.csv", index=False)
    diff = moe_vs_shared(pred)
    diff.to_csv(run / "moe_vs_hgb_geo.csv", index=False)
    c = cost(run)
    c.to_csv(run / "cost.csv", index=False)

    print(f"== {run.name} ==")
    print(f"\nrows scored: {len(pred):,} | platforms: {sorted(pred.glider.unique())}")
    print(f"bands: {sorted(pred.band.unique())} | protocols: {sorted(pred.protocol.unique())}")
    if len(moe):
        act = int((moe.active_experts > 0).sum())
        print(f"\nMoE: {act}/{len(moe)} folds with active experts")
        if len(diff):
            print(f"     max |moe - hgb_geo| over all folds = {diff.max_abs_diff.max():.3e}"
                  f"  -> {'DISTINGUISHABLE' if diff.max_abs_diff.max() > 1e-12 else 'IDENTICAL (experts never engaged)'}")
        if (moe.fallback != "").any():
            print("     fallback reasons:", sorted(set(moe.loc[moe.fallback != '', 'fallback'])))
    if len(c):
        print("\ncost:"); print(c.to_string(index=False))
    if len(plat):
        print("\nper-platform RMSE dB (pooled over folds within protocol/band):")
        piv = plat.pivot_table(index=["protocol", "band", "glider"], columns="model", values="rmse_db")
        print(piv.round(3).to_string())
    print(f"\nwrote: per_platform.csv, expert_diagnostics.csv, moe_vs_hgb_geo.csv, cost.csv")


if __name__ == "__main__":
    main()
