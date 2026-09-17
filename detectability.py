"""Step 6 - turn a noise anomaly into a change in how far a glider can hear.

With spherical spreading, a call is detectable while SL - 20*log10(R) - NL >= DT.
Raising the noise by dNL dB shrinks the range by the same number of dB:

    range ratio = 10 ** (-dNL / 20)        area ratio = 10 ** (-dNL / 10)

So +6 dB of noise halves the range and quarters the monitored area. This is a
relative statement and needs no assumption about source level, which is why it is
the headline output. Absolute ranges are also given, using the literature source
levels in config.BANDS, and are clearly labelled as assumptions.

Outputs: out/detectability.json, out/detectability.csv
"""
import json

import numpy as np
import pandas as pd

import config
from io_utils import log


def range_ratio(anomaly_db):
    return 10 ** (-np.asarray(anomaly_db, float) / 20.0)


def area_ratio(anomaly_db):
    return 10 ** (-np.asarray(anomaly_db, float) / 10.0)


def main():
    hind = pd.read_csv(config.OUT / "hindcast.csv")
    train = pd.read_csv(config.DATA / "train.csv")

    curve = [dict(anomaly_db=float(a), range_pct=float(100 * range_ratio(a)),
                  area_pct=float(100 * area_ratio(a)))
             for a in np.arange(-12, 15.1, 0.5)]

    per_band = []
    for code, b in config.BANDS.items():
        y = train[f"y_{code}"].dropna()
        if y.empty:
            continue
        p10, p50, p90 = np.percentile(y, [10, 50, 90])
        per_band.append(dict(
            band=code, name=b["name"], call=b["call"],
            band_lo=b["lo"], band_hi=b["hi"], source_level_db=b["sl_db"],
            noise_p10_db=round(float(p10), 2), noise_p90_db=round(float(p90), 2),
            quiet_range_pct=round(float(100 * range_ratio(p10)), 1),
            loud_range_pct=round(float(100 * range_ratio(p90)), 1),
            area_swing_x=round(float(area_ratio(p10) / area_ratio(p90)), 1)))

    hind["pred_area_ratio"] = area_ratio(hind["predicted"])
    hind["meas_area_ratio"] = area_ratio(hind["measured"])
    hind.to_csv(config.OUT / "detectability.csv", index=False)

    agree = (hind.assign(ok=lambda d: np.sign(d.predicted) == np.sign(d.measured))
                 .groupby("band").ok.mean().round(3).to_dict())

    json.dump(dict(curve=curve, per_band=per_band, direction_agreement=agree),
              open(config.OUT / "detectability.json", "w"), indent=1)

    log("step 6 done -> out/detectability.json")
    print(pd.DataFrame(per_band)[["band", "name", "noise_p10_db", "noise_p90_db",
                                  "quiet_range_pct", "loud_range_pct", "area_swing_x"]].to_string(index=False))


if __name__ == "__main__":
    main()
