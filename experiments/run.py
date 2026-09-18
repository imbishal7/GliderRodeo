"""Run bounded, resumable model comparisons on a prepared observational table."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import config
from experiments.data import FORECAST_HELD, feature_set, selected_features, sha256
from experiments.models import fit_model
from experiments.report import build_report, score
from experiments.splits import fold_scales, fold_targets, make_folds

MODELS = ["hgb_base", "hgb_geo", "tabpfn_geo", "moe_geo", "resnet_geo",
          "hgb_coords", "hgb_phys", "moe_full", "tabpfn_base",
          "catboost_base", "xgb_base", "lgbm_base"]

# Predictor set per model. Anything unlisted uses the bundled enhanced set.
MODEL_FEATURE_KIND = {"hgb_base": "base", "hgb_coords": "coords", "hgb_phys": "physical",
                      "tabpfn_base": "base",
                      "catboost_base": "base", "xgb_base": "base", "lgbm_base": "base"}


def command(args):
    try:
        return subprocess.run(args, text=True, capture_output=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)


def hardware():
    import platform
    record = {"hostname": platform.node(), "platform": platform.platform(), "python": sys.version,
              "cpu": command(["lscpu"]), "memory": command(["free", "-h"]),
              "gpu": command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version", "--format=csv"]),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}
    try:
        import torch
        record.update(torch=torch.__version__, cuda_build=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
    except ImportError:
        record["torch"] = "not installed"
    return record


def source_files():
    return sorted(list(config.ROOT.glob("*.py")) + list((config.ROOT / "experiments").glob("*.py")) +
                  list((config.ROOT / "experiments").glob("*.txt")))


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def apply_policy(X_train, X_test, policy):
    training, testing = X_train.copy(), X_test.copy()
    if policy == "forecast_median":
        for feature in set(FORECAST_HELD) & set(testing.columns):
            testing[feature] = training[feature].median()
    return training, testing


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", type=Path, default=config.DATA / "experiments.csv")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS[:4])
    p.add_argument("--protocols", nargs="+", choices=["logo", "forward", "spatial"], default=["logo", "forward", "spatial"])
    p.add_argument("--bands", nargs="+", choices=list(config.BANDS), default=list(config.BANDS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--calibration-hours", type=float, default=48)
    p.add_argument("--gap-hours", type=float, default=24)
    p.add_argument("--buffer-km", type=float, default=2)
    p.add_argument("--min-train", type=int, default=100)
    p.add_argument("--min-test", type=int, default=24)
    p.add_argument("--min-reference", type=int, default=12)
    p.add_argument("--tabpfn-estimators", type=int, default=4)
    p.add_argument("--moe-strength", type=float, default=0.25)
    p.add_argument("--resnet-epochs", type=int, default=100)
    p.add_argument("--include-tide", action="store_true")
    p.add_argument("--target-scaling", choices=["none", "platform"], default="none")
    p.add_argument("--feature-policy", choices=["observed", "forecast_median", "forecast_only"], default="observed")
    p.add_argument("--max-fits", type=int, default=256)
    p.add_argument("--max-seconds", type=int, default=7200)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p


def run(args):
    audit_path = args.table.with_suffix(".audit.json")
    if not args.table.exists() or not audit_path.exists():
        raise FileNotFoundError(f"Prepared data missing: {args.table}. Run python -m experiments.data first.")
    audit = json.loads(audit_path.read_text())
    data_hash = sha256(args.table)
    if audit.get("output_sha256") != data_hash:
        raise ValueError("Table checksum does not match its preparation audit")
    if args.min_reference < 1 or args.min_train < 2 or args.min_test < 1:
        raise ValueError("Invalid population thresholds")
    if args.threads < 1 or args.max_fits < 1 or args.max_seconds < 1:
        raise ValueError("Budgets and thread count must be positive")
    if args.calibration_hours <= 0 or args.gap_hours < 0 or args.buffer_km < 0:
        raise ValueError("Calibration must be positive; gap and buffer must be non-negative")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    lock = open(output / ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    source_hashes = {str(path.relative_to(config.ROOT)): sha256(path) for path in source_files()}
    identity = {"settings": settings, "data_sha256": data_hash, "source_sha256": source_hashes}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if not args.resume:
            raise FileExistsError("Run already exists; use --resume or choose a fresh output directory")
        if old["identity"] != identity:
            raise ValueError("Resume rejected: data, code, or configuration changed")
        manifest = old
    else:
        manifest = {"identity": identity, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "data_kind": audit.get("data_kind", "observational"), "data_audit": audit,
                    "hardware": hardware(), "git_commit": command(["git", "rev-parse", "HEAD"]),
                    "status": "running"}
        (output / "pip-freeze.txt").write_text(command([sys.executable, "-m", "pip", "freeze"]))
        # uv-created environments do not necessarily include pip.
        if not (output / "pip-freeze.txt").read_text().strip():
            import importlib.metadata
            packages = sorted(f"{p.metadata['Name']}=={p.version}" for p in importlib.metadata.distributions())
            (output / "pip-freeze.txt").write_text("\n".join(packages) + "\n")
        shutil.copy2(args.table, output / "input.csv")
        shutil.copy2(audit_path, output / "input.audit.json")
        with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
            for path in source_files():
                archive.add(path, arcname=str(path.relative_to(config.ROOT)))
    atomic_json(manifest_path, manifest)
    for subdir in ("predictions", "metrics", "splits"):
        (output / subdir).mkdir(exist_ok=True)
    events = open(output / "events.jsonl", "a", buffering=1)

    def event(kind, **info):
        record = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "event": kind, **info}
        events.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)

    d = pd.read_csv(args.table)
    d["hour_utc"] = pd.to_datetime(d.hour_utc, utc=True)
    if d.duplicated(["glider", "hour_utc"]).any() or not d.row_id.is_unique:
        raise ValueError("Prepared table identifiers are not unique")
    started, completed, failures = time.monotonic(), 0, 0
    bounded = False
    event("start", data_kind=manifest["data_kind"], rows=len(d), hardware=manifest["hardware"]["gpu"])
    try:
        with threadpool_limits(limits=args.threads):
            for protocol in args.protocols:
                for fold in make_folds(d, protocol, calibration_hours=args.calibration_hours,
                                       gap_hours=args.gap_hours, buffer_km=args.buffer_km):
                    for band in args.bands:
                        tr, te, y_train, y_test, references = fold_targets(d, band, fold, min_reference=args.min_reference)
                        split_key = f"{protocol}__{fold.name}__{band}"
                        scales = fold_scales(d, band, fold, references, min_reference=args.min_reference)
                        atomic_json(output / "splits" / f"{split_key}.json", {
                            "protocol": protocol, "fold": fold.name, "band": band,
                            "train_row_ids": d.iloc[tr].row_id.tolist(), "test_row_ids": d.iloc[te].row_id.tolist(),
                            "calibration_row_ids": d.iloc[fold.calibration].row_id.tolist(), "references_db": references})
                        if len(tr) < args.min_train or len(te) < args.min_test:
                            event("skip_fold", split=split_key, n_train=len(tr), n_test=len(te), reason="insufficient population")
                            continue
                        for model_name in args.models:
                            key = split_key + "__" + model_name
                            metric_path = output / "metrics" / f"{key}.json"
                            pred_path = output / "predictions" / f"{key}.csv"
                            if metric_path.exists() and pred_path.exists():
                                event("resume_skip", fit=key)
                                continue
                            if completed + failures >= args.max_fits or time.monotonic() - started >= args.max_seconds:
                                bounded = True
                                raise TimeoutError("Configured fit/time budget reached; resume to continue")
                            features = feature_set(d, MODEL_FEATURE_KIND.get(model_name, "enhanced"), include_tide=args.include_tide)
                            if args.feature_policy == "forecast_only":
                                features = [f for f in features if f not in FORECAST_HELD]
                            Xtr, Xte = apply_policy(d.iloc[tr][features], d.iloc[te][features], args.feature_policy)
                            event("fit_start", fit=key, n_train=len(tr), n_test=len(te), n_features=len(features))
                            fit_started = time.monotonic()
                            model = None
                            gpu = model_name in {"tabpfn_geo", "tabpfn_base", "resnet_geo"} and args.device.startswith("cuda")
                            try:
                                if gpu:
                                    import torch
                                    if not torch.cuda.is_available():
                                        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
                                    torch.set_num_threads(args.threads)
                                    torch.cuda.reset_peak_memory_stats()
                                y_fit = y_train
                                if args.target_scaling == "platform":
                                    s_tr = np.array([scales.get(g, 1.0) for g in d.iloc[tr].glider])
                                    y_fit = y_train / s_tr
                                model = fit_model(model_name, Xtr, y_fit, d.iloc[tr].glider.to_numpy(),
                                                  seed=args.seed, device=args.device, tabpfn_estimators=args.tabpfn_estimators,
                                                  moe_strength=args.moe_strength, resnet_epochs=args.resnet_epochs)
                                predicted = np.asarray(model.predict(Xte))
                                if args.target_scaling == "platform":
                                    # back to dB using the HELD-OUT platform's own permitted scale
                                    predicted = predicted * np.array([scales.get(g, 1.0) for g in d.iloc[te].glider])
                                if predicted.shape != y_test.shape or not np.isfinite(predicted).all():
                                    raise ValueError("Invalid model predictions")
                                result = {"protocol": protocol, "fold": fold.name, "band": band, "model": model_name,
                                          "features": features, "n_train": len(tr), **score(y_test, predicted),
                                          "elapsed_seconds": time.monotonic() - fit_started,
                                          "details": getattr(model, "details", {})}
                                if gpu:
                                    torch.cuda.synchronize()
                                    result["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024 ** 2
                                    result["gpu_device"] = torch.cuda.get_device_name()
                                pred = d.iloc[te][["row_id", "glider", "hour_utc", "lat", "lon"]].copy()
                                pred["protocol"], pred["fold"], pred["band"], pred["model"] = protocol, fold.name, band, model_name
                                pred["measured"], pred["predicted"] = y_test, predicted
                                pred["reference_db"] = d.iloc[te].glider.map(references).to_numpy()
                                temporary = pred_path.with_suffix(".tmp")
                                pred.to_csv(temporary, index=False)
                                temporary.replace(pred_path)
                                atomic_json(metric_path, result)
                                completed += 1
                                event("fit_complete", fit=key, rmse_db=result["rmse_db"], elapsed_seconds=result["elapsed_seconds"])
                            except Exception as error:
                                failures += 1
                                event("fit_failed", fit=key, error=str(error), traceback=traceback.format_exc())
                            finally:
                                del model
                                gc.collect()
                                if gpu and "torch" in sys.modules:
                                    sys.modules["torch"].cuda.empty_cache()
    except TimeoutError as error:
        if not bounded:
            raise
        event("budget_stop", error=str(error))
    except BaseException:
        manifest["status"] = "interrupted_or_failed"
        atomic_json(manifest_path, manifest)
        raise
    finally:
        events.close()
    manifest.update(status="budget_reached" if bounded else "completed_with_errors" if failures else "complete",
                    latest_execution={"completed_fits": completed, "failed_fits": failures,
                                      "elapsed_seconds": time.monotonic() - started},
                    updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    if not list((output / "predictions").glob("*.csv")):
        manifest["status"] = "no_predictions"
    atomic_json(manifest_path, manifest)
    if manifest["status"] != "no_predictions":
        build_report(output, plots=not args.no_plots)
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
