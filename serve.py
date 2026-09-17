"""Web UI - live predictions plus the results of the pipeline.

    python3 serve.py            then open http://localhost:8000

Uses only the standard library. Endpoints:
    GET /                    the page
    GET /api/meta            bands, features, slider ranges, model scores
    GET /api/predict?...     live prediction for one set of conditions
    GET /api/hindcast        held-out predictions vs measurements, per band and glider
    GET /api/drivers         permutation importance per band
    GET /api/detectability   noise -> range/area curve and per-band swings
    GET /api/forecast        the saved PacIOOS forecast run (?refresh=1 to re-run)
"""
import argparse
import json
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from joblib import load

import config
from detectability import area_ratio, range_ratio

BUNDLE = load(config.OUT / "model.joblib")
MODELS, FEATURES = BUNDLE["models"], BUNDLE["features"]
TRAIN = pd.read_csv(config.DATA / "train.csv")
DEFAULTS = {f: float(TRAIN[f].median()) for f in FEATURES}
with open(config.OUT / "metrics.json") as _fh:
    METRICS = json.load(_fh)


def _verdict(err_pct):
    """Plain-language rating from the reduction in typical error vs guessing."""
    if err_pct >= 10:
        return "useful", "clearly better than guessing"
    if err_pct >= 2:
        return "marginal", "slightly better than guessing"
    return "useless", "no better than guessing"


def band_scores():
    """Per band: how much better the model is than assuming normal conditions."""
    out = {}
    for m in METRICS:
        err_pct = 100 * (1 - m["rmse_db"] / m["baseline_rmse_db"])
        rating, phrase = _verdict(err_pct)
        out[m["band"]] = dict(
            error_reduction_pct=round(err_pct, 1), rating=rating, phrase=phrase,
            rmse_db=round(m["rmse_db"], 2), baseline_rmse_db=round(m["baseline_rmse_db"], 2),
            skill_vs_median=round(m["skill_vs_median"], 3))
    return out


SCORES = band_scores()

SLIDERS = [
    ("wind_ms", "Wind speed", "m/s", 0, 22, 0.5),
    ("hs_m", "Wave height", "m", 0.5, 6, 0.1),
    ("tp_s", "Wave period", "s", 5, 20, 0.5),
    ("swell_m", "Swell height", "m", 0.2, 5, 0.1),
    ("rain_mmhr", "Rain", "mm/h", 0, 5, 0.1),
    ("depth_m", "Glider depth", "m", 10, 1000, 10),
    ("depth_rate_ms", "Dive rate", "m/s", 0, 0.4, 0.01),
    ("hour_utc", "Hour of day", "UTC", 0, 23, 1),
]


def _json(obj):
    return json.dumps(obj, allow_nan=False, default=float).encode()


def predict(params):
    row = dict(DEFAULTS)
    for f in FEATURES:
        if f in params:
            row[f] = float(params[f][0])
    if "hour_utc" in params:
        h = float(params["hour_utc"][0])
        row["hour_sin"], row["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
    out = {}
    for code, model in MODELS.items():
        a = float(model.predict(X)[0])
        out[code] = dict(name=config.BANDS[code]["name"], call=config.BANDS[code]["call"],
                         anomaly_db=round(a, 2),
                         range_pct=round(float(100 * range_ratio(a)), 1),
                         area_pct=round(float(100 * area_ratio(a)), 1),
                         score=SCORES.get(code))
    return dict(inputs={f: round(row[f], 3) for f in FEATURES}, bands=out)


def hindcast():
    h = pd.read_csv(config.OUT / "hindcast.csv")
    h["hour_utc"] = pd.to_datetime(h["hour_utc"], utc=True)
    out = {}
    for band, g in h.groupby("band"):
        series = {}
        for glider, gg in g.groupby("glider"):
            gg = gg.sort_values("hour_utc")
            series[glider] = dict(
                t=[t.strftime("%Y-%m-%dT%H:%MZ") for t in gg["hour_utc"]],
                measured=[round(float(v), 2) for v in gg["measured"]],
                predicted=[round(float(v), 2) for v in gg["predicted"]])
        out[band] = series
    return out


def drivers():
    d = pd.read_csv(config.OUT / "drivers.csv")
    return {band: [dict(feature=r.feature, importance=round(float(r.importance), 4))
                   for r in g.sort_values("importance", ascending=False).itertuples()]
            for band, g in d.groupby("band")}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(config.WEB), **kw)

    def log_message(self, fmt, *args):
        pass

    def _send(self, payload, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/meta":
                return self._send(_json(dict(
                    bands=config.BANDS, features=FEATURES, defaults=DEFAULTS,
                    sliders=[dict(key=k, label=l, unit=un, min=lo, max=hi, step=st)
                             for k, l, un, lo, hi, st in SLIDERS],
                    metrics=METRICS, scores=SCORES,
                    gliders={k: dict(platform=v["platform"], fs_khz=v["fs_khz"])
                             for k, v in config.ACTIVE.items()},
                    excluded={k: v.get("exclude_reason") for k, v in config.GLIDERS.items() if v.get("exclude")},
                    n_rows=int(len(TRAIN)))))
            if u.path == "/api/predict":
                return self._send(_json(predict(q)))
            if u.path == "/api/hindcast":
                return self._send(_json(hindcast()))
            if u.path == "/api/drivers":
                return self._send(_json(drivers()))
            if u.path == "/api/detectability":
                return self._send(open(config.OUT / "detectability.json", "rb").read())
            if u.path == "/api/forecast":
                if q.get("refresh"):
                    subprocess.run([sys.executable, str(config.ROOT / "forecast.py")], check=False)
                path = config.OUT / "forecast.json"
                if not path.exists():
                    return self._send(_json(dict(error="no forecast yet; run python3 forecast.py")))
                return self._send(path.read_bytes())
        except Exception as exc:                     # keep the UI alive, show the error
            return self._send(_json(dict(error=f"{type(exc).__name__}: {exc}")))
        return super().do_GET()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8000)
    port = p.parse_args().port
    print(f"Noise Nowcaster UI  ->  http://localhost:{port}   (ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
