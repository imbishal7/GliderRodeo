"""Run the whole pipeline.

    python3 run_all.py               # everything (downloads are cached)
    python3 run_all.py --from train  # start at a later step
    python3 serve.py                 # then open the UI
"""
import argparse
import time

import build_table
import detectability
import fetch_env
import forecast
import noise_bands
import train
from io_utils import log

STEPS = [
    ("fetch", "download AQUAVIEW conditions", fetch_env.main),
    ("bands", "read hydrophones into band levels", noise_bands.main),
    ("table", "build the training table", build_table.main),
    ("train", "train + leave-one-glider-out test", train.main),
    ("detect", "noise to detectability", detectability.main),
    ("forecast", "run the live PacIOOS forecast", forecast.main),
]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="start", default="fetch", choices=[s[0] for s in STEPS])
    start = [s[0] for s in STEPS].index(p.parse_args().start)
    t0 = time.time()
    for key, name, fn in STEPS[start:]:
        log(f"===== {key}: {name} =====")
        try:
            fn()
        except SystemExit as exc:               # the forecast depends on a live service
            log(f"  skipped: {exc}")
    log(f"pipeline finished in {time.time() - t0:.0f}s - now run: python3 serve.py")
