"""Checks for leakage, population consistency, and regime-model invariants."""
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

import config
from experiments.data import (BASE_FEATURES, feature_set, geographic_features,
                              load_population, selected_features)
from experiments.models import ResidualExperts
from experiments.run import apply_policy, parser, run
from experiments.smoke import prepare, synthetic_frame
from experiments.splits import fold_targets, make_folds


@pytest.fixture
def frame():
    return geographic_features(synthetic_frame())


def test_scored_labels_cannot_change_reference_or_training_targets(frame):
    for protocol in ("logo", "forward", "spatial"):
        for fold in make_folds(frame, protocol, calibration_hours=12, gap_hours=12):
            before = fold_targets(frame, "SPWH", fold, min_reference=6)
            changed = frame.copy()
            changed.loc[changed.index[fold.test], "SPWH"] += 1000
            after = fold_targets(changed, "SPWH", fold, min_reference=6)
            assert before[4] == after[4]
            np.testing.assert_array_equal(before[2], after[2])
            if len(before[3]):
                np.testing.assert_allclose(after[3] - before[3], 1000)


def test_logo_gliders_and_calibration_are_disjoint(frame):
    for fold in make_folds(frame, "logo", calibration_hours=12):
        assert set(frame.iloc[fold.train].glider).isdisjoint(frame.iloc[fold.test].glider)
        assert set(fold.calibration).isdisjoint(fold.test)
        assert frame.iloc[fold.calibration].hour_utc.max() < frame.iloc[fold.test].hour_utc.min()


def test_future_split_has_shared_time_cutoff_and_gap(frame):
    fold = next(make_folds(frame, "forward", gap_hours=24))
    assert frame.iloc[fold.test].hour_utc.min() - frame.iloc[fold.train].hour_utc.max() > pd.Timedelta(hours=24)


def test_spatial_buffer_excludes_nearby_training_rows(frame):
    for fold in make_folds(frame, "spatial", buffer_km=2):
        training = frame.iloc[fold.train]
        sx, sy = (1 if fold.name[1] == "E" else -1), (1 if fold.name[0] == "N" else -1)
        distance = np.hypot(np.maximum(-sx * training.east_km, 0), np.maximum(-sy * training.north_km, 0))
        assert (distance >= 2).all()
        assert set(fold.train).isdisjoint(fold.test)


def test_partial_band_labels_are_not_compared_as_full_band(tmp_path):
    d = synthetic_frame(hours=48, gliders=3)
    table, coverage = tmp_path / "train.csv", tmp_path / "coverage.csv"
    d.to_csv(table, index=False)
    rows = [{"glider": g, "band": b, "status": "full"} for g in d.glider.unique() for b in config.BANDS]
    rows[0]["status"] = "partial"
    pd.DataFrame(rows).to_csv(coverage, index=False)
    filtered, audit = load_population(table, coverage)
    assert filtered.loc[filtered.glider == rows[0]["glider"], rows[0]["band"]].isna().all()
    assert audit[f"{rows[0]['band']}_excluded_partial_or_unknown"] == 48
    with pytest.raises(FileNotFoundError):
        load_population(table, tmp_path / "absent.csv")


def test_forecast_defaults_use_only_training_features(frame):
    features = selected_features(frame)
    tr, te = frame.iloc[:100][features], frame.iloc[100:][features]
    first = apply_policy(tr, te, "forecast_median")[1]
    shifted = te.copy()
    shifted["temp_c"] += 1000
    second = apply_policy(tr, shifted, "forecast_median")[1]
    np.testing.assert_array_equal(first.temp_c, second.temp_c)
    assert first.temp_c.iloc[0] == tr.temp_c.median()


def test_features_exclude_targets_and_recorder_identity(frame):
    features = set(selected_features(frame, include_tide=True))
    assert features.isdisjoint({"glider", *config.BANDS, *[f"y_{b}" for b in config.BANDS]})


def test_moe_gates_are_finite_and_normalized(frame):
    features = selected_features(frame)
    X, y = frame[features], frame.SPWH.to_numpy()
    with threadpool_limits(limits=2):
        model = ResidualExperts().fit(X, y, frame.glider.to_numpy())
        np.testing.assert_allclose(model.weights(X).sum(axis=1), 1)
        assert np.isfinite(model.predict(X)).all()
    assert model.details["inner_folds"] == 3
    assert model.details["active_experts"] == 2


def test_moe_falls_back_with_too_few_platforms(frame):
    subset = frame[frame.glider.isin(["synthetic_0", "synthetic_1"])]
    X = subset[selected_features(subset)]
    with threadpool_limits(limits=2):
        model = ResidualExperts().fit(X, subset.SPWH.to_numpy(), subset.glider.to_numpy())
        np.testing.assert_array_equal(model.predict(X), model.global_model.predict(X))
    assert model.details["active_experts"] == 0


def test_bounded_run_resumes_and_rejects_changed_settings(tmp_path):
    table = prepare(tmp_path / "synthetic")
    output = tmp_path / "run"
    arguments = ["--table", str(table), "--output", str(output), "--device", "cpu",
                 "--models", "hgb_base", "hgb_geo", "--protocols", "forward",
                 "--bands", "SPWH", "--max-fits", "1", "--threads", "2", "--no-plots"]
    assert run(parser().parse_args(arguments)) == 2  # first invocation reaches its bound
    assert len(list((output / "metrics").glob("*.json"))) == 1
    assert run(parser().parse_args(arguments + ["--resume"])) == 0
    assert len(list((output / "metrics").glob("*.json"))) == 2
    assert '"event": "resume_skip"' in (output / "events.jsonl").read_text()
    with pytest.raises(ValueError, match="Resume rejected"):
        run(parser().parse_args(arguments + ["--resume", "--seed", "1"]))


def test_attribution_feature_sets_partition_the_enhanced_set(frame):
    """coords and physical must split the enhanced additions with no overlap.

    `hgb_geo` bundles horizontal position with the added physical predictors;
    the attribution arms only isolate them if the split is clean.
    """
    base = set(feature_set(frame, "base"))
    coords = set(feature_set(frame, "coords"))
    physical = set(feature_set(frame, "physical"))
    enhanced = set(feature_set(frame, "enhanced"))

    assert base == set(BASE_FEATURES)
    assert base < coords and base < physical and base < enhanced
    assert coords | physical == enhanced           # together they rebuild the bundle
    assert (coords & physical) == base             # and share nothing beyond it
    assert {"east_km", "north_km"} <= coords
    assert not ({"east_km", "north_km"} & physical)


def test_attribution_arms_use_the_declared_predictors(frame):
    """The runner must map each attribution model to its own predictor set."""
    from experiments.run import MODEL_FEATURE_KIND
    assert MODEL_FEATURE_KIND["hgb_coords"] == "coords"
    assert MODEL_FEATURE_KIND["hgb_phys"] == "physical"
    assert MODEL_FEATURE_KIND["hgb_base"] == "base"
    # anything unlisted keeps the bundled enhanced set
    assert "hgb_geo" not in MODEL_FEATURE_KIND
    assert feature_set(frame, MODEL_FEATURE_KIND.get("hgb_geo", "enhanced")) == selected_features(frame)


def test_attribution_arms_fit_the_same_learner_as_the_baseline(frame):
    """hgb_coords/hgb_phys must be the same HGB learner, only the inputs differ."""
    from experiments.models import fit_model
    y = frame["SPWH"].to_numpy()
    groups = frame["glider"].to_numpy()
    models = {name: fit_model(name, frame[feature_set(frame, kind)], y, groups, seed=0)
              for name, kind in (("hgb_base", "base"), ("hgb_coords", "coords"),
                                 ("hgb_phys", "physical"))}
    for name, model in models.items():
        assert type(model).__name__ == "HistGradientBoostingRegressor", name
        assert model.max_iter == 400, name
