"""Sanity tests.  Run:  python3 -m unittest discover -s tests"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                   # noqa: E402
from detectability import area_ratio, range_ratio   # noqa: E402
from io_utils import epoch_ns, to_ns            # noqa: E402
from noise_bands import band_level              # noqa: E402


class TestDetectability(unittest.TestCase):
    def test_no_change(self):
        self.assertAlmostEqual(float(range_ratio(0)), 1.0)

    def test_six_db_halves_range_and_quarters_area(self):
        self.assertAlmostEqual(float(range_ratio(6)), 0.5, places=2)
        self.assertAlmostEqual(float(area_ratio(6)), 0.25, places=2)

    def test_quieter_means_further(self):
        self.assertGreater(float(range_ratio(-6)), 1.9)


class TestBandLevel(unittest.TestCase):
    def test_flat_spectrum(self):
        lower = np.arange(0, 2000, 10.0)
        upper = lower + 10
        levels = np.full((3, len(lower)), 60.0)      # 60 dB/Hz over 1000 Hz -> 90 dB
        np.testing.assert_allclose(band_level(levels, lower, upper, 500, 1500), 90.0, atol=1e-9)

    def test_partial_overlap_is_weighted(self):
        lower, upper = np.array([0.0, 100.0]), np.array([100.0, 200.0])
        levels = np.array([[60.0, 60.0]])
        self.assertAlmostEqual(float(band_level(levels, lower, upper, 50, 150)[0]), 80.0, places=9)

    def test_band_outside_range_is_nan(self):
        lower, upper = np.array([0.0]), np.array([10.0])
        self.assertTrue(np.isnan(band_level(np.array([[60.0]]), lower, upper, 100, 200)[0]))


class TestTimeHandling(unittest.TestCase):
    def test_nanosecond_resolution(self):
        """pandas 3 parses to microseconds; mixing units silently breaks the joins."""
        s = pd.Series(["2026-02-01T12:00:00Z"])
        self.assertEqual(to_ns(s).dt.unit, "ns")
        self.assertEqual(epoch_ns(s)[0], 1769947200 * 10**9)


class TestTrainedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from joblib import load
        if not (config.OUT / "model.joblib").exists():
            raise unittest.SkipTest("model not trained yet - run python3 run_all.py")
        cls.bundle = load(config.OUT / "model.joblib")
        cls.train = pd.read_csv(config.DATA / "train.csv")
        with open(config.OUT / "metrics.json") as fh:
            cls.metrics = json.load(fh)

    def _row(self, **over):
        med = {f: float(self.train[f].median()) for f in self.bundle["features"]}
        med.update(over)
        return pd.DataFrame([[med[f] for f in self.bundle["features"]]], columns=self.bundle["features"])

    def test_more_wind_predicts_more_noise(self):
        m = self.bundle["models"]["SPWH"]
        calm = float(m.predict(self._row(wind_ms=2))[0])
        windy = float(m.predict(self._row(wind_ms=16))[0])
        self.assertGreater(windy, calm)

    def test_predictions_are_finite_and_plausible(self):
        for code, m in self.bundle["models"].items():
            v = float(m.predict(self._row())[0])
            self.assertTrue(np.isfinite(v), code)
            self.assertLess(abs(v), 30, code)

    def test_wind_bands_have_skill(self):
        skill = {m["band"]: m["skill_vs_median"] for m in self.metrics}
        for band in ("SPWH", "UNDO", "BBWH"):
            self.assertGreater(skill[band], 0.1, f"{band} lost its skill")

    def test_targets_are_anomalies(self):
        """Each glider's target should be centred on zero by construction."""
        for code in config.BANDS:
            med = self.train.groupby("glider")[f"y_{code}"].median().dropna()
            if len(med):
                self.assertLess(float(med.abs().max()), 1e-6, code)


if __name__ == "__main__":
    unittest.main()
