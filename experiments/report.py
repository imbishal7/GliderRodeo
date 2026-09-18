"""Aggregate paired predictions and write a compact report and static figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def score(measured, predicted):
    y, p = np.asarray(measured), np.asarray(predicted)
    mse = float(np.mean((y - p) ** 2))
    baseline = float(np.mean(y ** 2))
    return dict(n=len(y), rmse_db=float(np.sqrt(mse)), mae_db=float(np.mean(np.abs(y - p))),
                baseline_rmse_db=float(np.sqrt(baseline)),
                skill_vs_zero=1 - mse / baseline if baseline > 0 else None)


def paired_day_bootstrap(d, baseline, repetitions=500, seed=0):
    keys = ["fold", "row_id"]
    paired = d.merge(baseline[keys + ["predicted"]], on=keys, suffixes=("", "_base"), validate="one_to_one")
    if len(paired) != len(d) or len(paired) != len(baseline):
        return {"status": "incomplete paired population"}
    paired["day"] = pd.to_datetime(paired.hour_utc, utc=True).dt.floor("D")
    paired["sse"] = (paired.measured - paired.predicted) ** 2
    paired["sse_base"] = (paired.measured - paired.predicted_base) ** 2
    blocks = paired.groupby("day").agg(sse=("sse", "sum"), sse_base=("sse_base", "sum"), n=("sse", "size"))
    if len(blocks) < 5:
        return {"status": "fewer than five UTC-day blocks", "n_days": len(blocks)}
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(blocks), size=(repetitions, len(blocks)))
    totals = blocks.to_numpy()[draw].sum(axis=1)
    delta = np.sqrt(totals[:, 0] / totals[:, 2]) - np.sqrt(totals[:, 1] / totals[:, 2])
    return {"status": "exploratory paired UTC-day bootstrap",
            "delta_rmse_vs_hgb_base": float(np.sqrt(paired.sse.mean()) - np.sqrt(paired.sse_base.mean())),
            "ci95": np.quantile(delta, [0.025, 0.975]).tolist(), "n_days": len(blocks),
            "repetitions": repetitions}


def build_report(run_dir, plots=True):
    run_dir = Path(run_dir)
    paths = sorted((run_dir / "predictions").glob("*.csv"))
    if not paths:
        raise ValueError("No completed predictions to report")
    d = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    d["residual"] = d.measured - d.predicted
    d.to_csv(run_dir / "predictions.csv", index=False)
    summaries = []
    for (protocol, band, model), subset in d.groupby(["protocol", "band", "model"]):
        result = dict(protocol=protocol, band=band, model=model, **score(subset.measured, subset.predicted))
        folds = [score(x.measured, x.predicted) for _, x in subset.groupby("fold")]
        result.update(n_folds=len(folds), macro_fold_rmse_db=float(np.mean([x["rmse_db"] for x in folds])),
                      worst_fold_rmse_db=float(max(x["rmse_db"] for x in folds)))
        base = d[(d.protocol == protocol) & (d.band == band) & (d.model == "hgb_base")]
        result["paired_comparison"] = paired_day_bootstrap(subset, base) if len(base) else {"status": "baseline unavailable"}
        summaries.append(result)
    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    flat = pd.DataFrame([{k: v for k, v in row.items() if k != "paired_comparison"} for row in summaries])
    flat.to_csv(run_dir / "summary.csv", index=False)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    metrics = [json.loads(path.read_text()) for path in (run_dir / "metrics").glob("*.json")]
    fallbacks = [m for m in metrics if m.get("details", {}).get("fallback")]
    platform_counts = manifest.get("data_audit", {}).get("glider_counts", {})
    feature_policy = manifest.get("identity", {}).get("settings", {}).get("feature_policy", "unknown")
    lines = ["# GliderRodeo model comparison", "",
             f"Run: `{run_dir.name}`. Data: **{manifest.get('data_kind', 'observational')}**.", "",
             f"Prepared population: {sum(platform_counts.values())} rows across {len(platform_counts)} platform(s). Feature policy: `{feature_policy}`.", "",
             "All metrics use held-out predictions. RMSE is in dB relative to a reference established from training or calibration observations.", "",
             "| Protocol | Band | Model | Rows | Folds | RMSE dB | MAE dB | Skill vs zero |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        skill = f"{row['skill_vs_zero']:.3f}" if row["skill_vs_zero"] is not None else "undefined"
        lines.append(f"| {row['protocol']} | {row['band']} | {row['model']} | {row['n']} | {row['n_folds']} | {row['rmse_db']:.3f} | {row['mae_db']:.3f} | {skill} |")
    if fallbacks:
        reasons = sorted({m["details"]["fallback"] for m in fallbacks})
        lines += ["", f"**Expert-model limitation:** {len(fallbacks)} fits used only the shared model: {'; '.join(reasons)}. Those scores do not test the benefit of expert specialization."]
    lines += ["", "## Interpretation", "",
              "- Compare models within the same protocol, band, and feature policy. Missing folds are logged in events.jsonl.",
              "- LOGO includes an initial calibration period on the held-out platform; forward uses historical platform references; spatial tests known platforms in a held-out quadrant.",
              "- Bootstrap intervals resample whole UTC days across platforms. They are exploratory, conditional on these fitted models, and do not prove a physical mechanism.",
              "- Geographic gates may correlate with platform and mission time. Inspect expert diagnostics and residual maps before claiming distinct oceanographic regimes.",
              "- Fixed-M2 harmonic features, if enabled, are clock proxies, not tide-model predictions.",
              "- No same-hour acoustic measurements are model inputs. Pre-existing y_* columns are ignored.",
              "- Synthetic smoke-test numbers are software checks, not results on the glider mission.", ""]
    (run_dir / "REPORT.md").write_text("\n".join(lines))
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for protocol, rows in flat.groupby("protocol"):
            pivot = rows.pivot(index="band", columns="model", values="rmse_db")
            ax = pivot.plot.bar(figsize=(10, 5), ylabel="Held-out RMSE (dB)", rot=0,
                                title=f"{protocol}: {manifest.get('data_kind', 'observational')}")
            ax.figure.tight_layout()
            ax.figure.savefig(run_dir / f"rmse_{protocol}.png", dpi=160)
            plt.close(ax.figure)
        for (protocol, band, model), rows in d.groupby(["protocol", "band", "model"]):
            if model not in {"hgb_base", "hgb_geo", "moe_geo"}:
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            limit = max(float(np.quantile(np.abs(rows.residual), 0.95)), 1)
            points = ax.scatter(rows.lon, rows.lat, c=rows.residual, cmap="coolwarm", s=10,
                                vmin=-limit, vmax=limit, alpha=0.7)
            fig.colorbar(points, ax=ax, label="Measured − predicted (dB)")
            ax.set(xlabel="Longitude", ylabel="Latitude", title=f"{protocol} / {band} / {model}")
            ax.set_aspect(1 / np.cos(np.deg2rad(rows.lat.mean())))
            fig.tight_layout()
            fig.savefig(run_dir / f"residuals_{protocol}_{band}_{model}.png", dpi=130)
            plt.close(fig)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    build_report(args.run_dir, not args.no_plots)
