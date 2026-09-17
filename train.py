"""Steps 4-5 - train the noise model and rank what drives the noise.

Model: HistGradientBoostingRegressor, one per band, predicting the band-level anomaly
(dB relative to that glider's own median).

Honest test: leave-one-glider-out. Train on five gliders, predict the sixth. The
baseline is "predict 0", i.e. guess that glider's own median level; skill above that
means the model learned something about the ocean rather than about one hydrophone.

Outputs (out/):
  model.joblib      final models fitted on all gliders, plus metadata
  metrics.json      per band and per held-out glider scores
  drivers.csv       permutation importance, averaged over folds
  hindcast.csv      held-out predictions for every glider-hour
"""
import argparse
import json

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

import config
from io_utils import log

PARAMS = dict(max_iter=400, learning_rate=0.06, max_depth=6, min_samples_leaf=20,
              l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
              random_state=0)


def fit_band(df, code, features):
    """Leave-one-glider-out evaluation + a final model fitted on everything."""
    d = df.dropna(subset=[f"y_{code}"])
    if len(d) < 100 or d["glider"].nunique() < 3:
        return None
    X, y, g = d[features], d[f"y_{code}"].to_numpy(), d["glider"].to_numpy()

    folds, preds, imps = [], [], []
    for held in np.unique(g):
        tr, te = g != held, g == held
        if tr.sum() < 100 or te.sum() < 24:
            continue
        m = HistGradientBoostingRegressor(**PARAMS).fit(X[tr], y[tr])
        p = m.predict(X[te])
        mse_model = float(np.mean((y[te] - p) ** 2))
        mse_base = float(np.mean(y[te] ** 2))          # baseline: predict the glider's own median
        folds.append(dict(held_out=held, n=int(te.sum()),
                          r2=float(r2_score(y[te], p)),
                          mae_db=float(mean_absolute_error(y[te], p)),
                          rmse_db=float(np.sqrt(mse_model)),
                          baseline_rmse_db=float(np.sqrt(mse_base)),
                          skill_vs_median=float(1 - mse_model / mse_base) if mse_base > 0 else np.nan))
        preds.append(pd.DataFrame(dict(glider=held, hour_utc=d.loc[te, "hour_utc"].to_numpy(),
                                       band=code, measured=y[te], predicted=p)))
        pi = permutation_importance(m, X[te], y[te], n_repeats=5, random_state=0, scoring="r2")
        imps.append(pi.importances_mean)

    final = HistGradientBoostingRegressor(**PARAMS).fit(X, y)
    overall_mse = float(np.mean(np.concatenate([(f["rmse_db"] ** 2) * np.ones(f["n"]) for f in folds])))
    overall_base = float(np.mean(np.concatenate([(f["baseline_rmse_db"] ** 2) * np.ones(f["n"]) for f in folds])))
    summary = dict(
        band=code, n_rows=int(len(d)), n_gliders=int(d["glider"].nunique()),
        rmse_db=float(np.sqrt(overall_mse)), baseline_rmse_db=float(np.sqrt(overall_base)),
        skill_vs_median=float(1 - overall_mse / overall_base),
        mae_db=float(np.mean([f["mae_db"] for f in folds])),
        r2_mean=float(np.mean([f["r2"] for f in folds])),
        folds=folds)
    importance = pd.DataFrame(dict(band=code, feature=features,
                                   importance=np.mean(imps, axis=0) if imps else np.nan))
    return final, summary, pd.concat(preds, ignore_index=True), importance


def main(quick=False):
    df = pd.read_csv(config.DATA / "train.csv")
    features = config.FEATURE_NAMES
    log(f"{len(df):,} glider-hours, {len(features)} features, {df.glider.nunique()} gliders")

    models, metrics, hind, drivers = {}, [], [], []
    for code in config.BANDS:
        res = fit_band(df, code, features)
        if res is None:
            log(f"{code}: skipped (too few rows or gliders)")
            continue
        model, summary, preds, imp = res
        models[code] = model
        metrics.append(summary)
        hind.append(preds)
        drivers.append(imp)
        log(f"{code:5s} skill vs median {summary['skill_vs_median']:+.2f} | "
            f"RMSE {summary['rmse_db']:.2f} dB (baseline {summary['baseline_rmse_db']:.2f}) | "
            f"MAE {summary['mae_db']:.2f} dB | {summary['n_rows']:,} rows")

    dump(dict(models=models, features=features, bands=config.BANDS,
              medians=pd.read_csv(config.DATA / "glider_medians.csv", index_col=0).to_dict()),
         config.OUT / "model.joblib")
    json.dump(metrics, open(config.OUT / "metrics.json", "w"), indent=1)
    pd.concat(hind, ignore_index=True).to_csv(config.OUT / "hindcast.csv", index=False)
    imp = pd.concat(drivers, ignore_index=True)
    imp.to_csv(config.OUT / "drivers.csv", index=False)

    log("steps 4-5 done -> out/model.joblib, metrics.json, hindcast.csv, drivers.csv")
    top = (imp.groupby("feature").importance.mean().sort_values(ascending=False).head(8))
    print("\ntop drivers, averaged over bands:")
    print(top.round(4).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    main(p.parse_args().quick)
