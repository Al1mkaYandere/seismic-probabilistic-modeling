"""The ETAS baseline must expose the predictions its own metrics are built on.

run_etas_static computed a full per-(cell, week) Poisson intensity for the test
split and then threw it away, returning only aggregate metrics. Every other
model in the pipeline stores its predictions per row, so ETAS - the actual
seismological baseline - could not take part in any distributional comparison:
there was nothing to score.

This test does not check that a file exists. It checks the property that makes
the file worth having: the numbers in it are the numbers the reported metrics
were computed from. A file written from some other intermediate state would
pass an existence check and fail this one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_poisson_deviance

from src import config, etas_baseline


@pytest.fixture()
def synthetic_catalog_and_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One active cell and one quiet cell over 40 weeks, fixed by construction.

    The catalog is regular rather than random: the point here is bookkeeping
    (are the saved rows the scored rows), not the quality of the fit.
    """
    origin = pd.Timestamp("2015-01-05")
    weeks = [origin + pd.Timedelta(weeks=w) for w in range(40)]

    events = []
    for day in range(0, 280, 9):
        events.append({"time": origin + pd.Timedelta(days=day), "mag": 3.4,
                       "latitude": 40.0, "longitude": 70.0, "cell_id": "ACTIVE"})
    for day in (30, 150):
        events.append({"time": origin + pd.Timedelta(days=day), "mag": 3.1,
                       "latitude": 43.0, "longitude": 73.0, "cell_id": "QUIET"})
    events_df = pd.DataFrame(events)

    rows = []
    for cell in ("ACTIVE", "QUIET"):
        for week in weeks:
            start, end = week, week + pd.Timedelta(days=7)
            n = int(((events_df["cell_id"] == cell)
                     & (events_df["time"] >= start)
                     & (events_df["time"] < end)).sum())
            rows.append({"cell_id": cell, "week": week, "Y": n,
                         "lat_grid": 40.0 if cell == "ACTIVE" else 43.0,
                         "lon_grid": 70.0 if cell == "ACTIVE" else 73.0})
    return events_df, pd.DataFrame(rows)


def test_saved_etas_predictions_reproduce_the_reported_metrics(
    synthetic_catalog_and_panel, tmp_path, monkeypatch
):
    events_df, panel_df = synthetic_catalog_and_panel
    out_path = tmp_path / "etas_test_predictions.csv"
    monkeypatch.setattr(config, "ETAS_TEST_PREDICTIONS_CSV", out_path)
    # Every path the step writes has to be redirected, not just the one this
    # test reads: the step also saves its training-period intensities, and a
    # test that leaves that one pointing at outputs/ quietly replaces a real
    # pipeline artefact with synthetic data. That happened once.
    monkeypatch.setattr(config, "ETAS_TRAIN_PREDICTIONS_CSV",
                        tmp_path / "etas_train_predictions.csv")

    metrics = etas_baseline.run_etas_static(events_df, panel_df)
    assert metrics["Status"] == "SUCCESS"

    saved = pd.read_csv(out_path)
    train_saved_path = tmp_path / "etas_train_predictions.csv"
    assert train_saved_path.exists(), (
        "the training intensities were written somewhere other than the redirected path"
    )

    # The training file has to be checked here, by calling the step, and not only
    # against the artefact in outputs/: a test that reads the pipeline's own file
    # cannot notice a defect introduced into the code until the pipeline is run
    # again, and by then the defect is already in the numbers.
    train_saved = pd.read_csv(train_saved_path)
    weeks = np.sort(np.unique(panel_df["week"]))
    split = int(np.floor(0.8 * len(weeks)))
    train_weeks = set(pd.to_datetime(weeks[:split]))
    test_weeks = set(pd.to_datetime(weeks[split:]))
    saved_weeks = set(pd.to_datetime(train_saved["week"]))

    assert saved_weeks == train_weeks, "the training file does not cover the training weeks"
    assert not saved_weeks & test_weeks, "the training file reaches into the forecast period"
    assert len(train_saved) == len(train_weeks) * panel_df["cell_id"].nunique()
    assert (train_saved["lambda_pred"] > 0).all(), (
        "an intensity of zero cannot carry a predictive distribution"
    )

    # Exact identity rather than a resemblance: refit the same model here and
    # recompute a few of those weeks. Anything that changed what the step fed to
    # the prediction - the completeness threshold, the parameters, the history -
    # breaks this, while "the numbers look plausible" would not.
    from src.etas_baseline import fit_etas_per_cell, predict_etas
    train_end = pd.Timestamp(weeks[split - 1]) + pd.Timedelta(days=7)
    params = fit_etas_per_cell(events_df, train_end)
    sample = train_saved.head(25)[["cell_id", "week"]].copy()
    sample["week"] = pd.to_datetime(sample["week"])
    recomputed = predict_etas(params, sample, events_df)
    # Joined on the key, never by position: predict_etas returns its rows
    # grouped by cell, which is not the order they were asked for.
    check = recomputed.assign(week=recomputed["week"].astype(str)).merge(
        train_saved.assign(week=train_saved["week"].astype(str)),
        on=["cell_id", "week"], how="inner", suffixes=("_recomputed", "_saved"),
    )
    assert len(check) == len(sample), "the sampled rows are not all in the saved file"
    np.testing.assert_allclose(
        check["lambda_pred_recomputed"].to_numpy(), check["lambda_pred_saved"].to_numpy(),
        rtol=0, atol=0,
        err_msg="the saved training intensities are not the ones this model produces",
    )

    # One row per (cell, week) of the test split, and the observed counts must
    # be the panel's own - not a re-derivation that could drift from it.
    n_test_weeks = len(np.unique(panel_df["week"])) - int(
        np.floor(0.8 * len(np.unique(panel_df["week"])))
    )
    assert len(saved) == n_test_weeks * panel_df["cell_id"].nunique()
    assert set(saved.columns) == {"cell_id", "week", "y_true", "lambda_pred"}

    merged = saved.merge(
        panel_df.assign(week=panel_df["week"].astype(str)),
        on=["cell_id", "week"], how="left",
    )
    assert not merged["Y"].isna().any(), "saved rows do not all exist in the panel"
    assert (merged["y_true"] == merged["Y"]).all()

    # The point of the test: scoring the saved file must give back exactly the
    # metric that was reported, so the file is the metric's own input.
    recomputed = mean_poisson_deviance(
        saved["y_true"].to_numpy(dtype=float),
        np.clip(saved["lambda_pred"].to_numpy(dtype=float), 1e-9, None),
    )
    assert recomputed == pytest.approx(metrics["Mean_Poisson_Deviance"], rel=1e-12)


def _events_with_cells(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw catalogue with the grid cell of each event, the way main.py builds it."""
    events = raw.copy()
    events["time"] = pd.to_datetime(events["time"], utc=True, format="mixed").dt.tz_localize(None)
    lat = np.floor((events["latitude"] - config.BBOX["minlatitude"]) / config.GRID_SIZE)
    lon = np.floor((events["longitude"] - config.BBOX["minlongitude"]) / config.GRID_SIZE)
    events["lat_grid"] = (lat * config.GRID_SIZE + config.BBOX["minlatitude"]).round(2)
    events["lon_grid"] = (lon * config.GRID_SIZE + config.BBOX["minlongitude"]).round(2)
    events["cell_id"] = (events["lat_grid"].map(lambda x: f"{float(x):.2f}") + "_"
                         + events["lon_grid"].map(lambda x: f"{float(x):.2f}"))
    return events


# ── Training-period intensities (step 2a) ────────────────────────────────────

def test_training_intensities_cover_the_training_weeks_and_nothing_later() -> None:
    """Anything fitted on top of ETAS needs its in-sample intensities.

    They have to be the TRAINING ones: fitting a dispersion parameter, or a level
    correction, on the test split is reading the answer. So the file must cover
    the training weeks exactly - one row per cell and week, and not a single week
    from the forecast period.
    """
    train_path = config.ETAS_TRAIN_PREDICTIONS_CSV
    test_path = config.ETAS_TEST_PREDICTIONS_CSV
    if not test_path.exists():
        pytest.skip("the pipeline has not been run yet")
    # Deliberately not a skip: once the step has run, a missing training file
    # means the step stopped saving them. A skip on both would turn "the code no
    # longer does this" into a silent pass.
    assert train_path.exists(), (
        f"{test_path.name} exists but {train_path.name} does not - the step ran "
        "and did not keep its training-period intensities"
    )

    train = pd.read_csv(train_path, parse_dates=["week"])
    test = pd.read_csv(test_path, parse_dates=["week"])
    panel = pd.read_csv(config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE,
                        parse_dates=["week"])

    weeks = np.sort(panel["week"].unique())
    split = int(np.floor(0.8 * len(weeks)))
    expected_train, expected_test = set(weeks[:split]), set(weeks[split:])

    assert set(train["week"]) == expected_train, "training weeks do not match the split"
    assert set(test["week"]) == expected_test
    assert not (set(train["week"]) & set(test["week"])), "the two files overlap in time"
    assert len(train) == len(expected_train) * panel["cell_id"].nunique()
    assert (train["lambda_pred"] > 0).all(), "an intensity of zero cannot carry a distribution"
    assert np.isfinite(train["lambda_pred"]).all()

    merged = panel.merge(train, on=["cell_id", "week"], how="inner")
    assert len(merged) == len(train)
    assert (merged["Y"].to_numpy() == merged["y_true"].to_numpy()).all(), (
        "the saved truth does not match the panel"
    )


def test_training_intensities_do_not_move_when_the_test_period_is_corrupted() -> None:
    """The property that makes them safe to fit on: they cannot see the future.

    A training week uses only the events before it, all of them inside the
    training period. Rewriting every event after the split must therefore leave
    every training intensity exactly where it was - not close, exactly.
    """
    raw_path = config.RAW_DATA_PATH / config.RAW_DATA_FILE
    panel_path = config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE
    if not raw_path.exists() or not panel_path.exists():
        pytest.skip("catalogue or panel missing")

    from src.etas_baseline import fit_etas_per_cell, predict_etas

    raw = _events_with_cells(pd.read_csv(raw_path))
    panel = pd.read_csv(panel_path, parse_dates=["week"])
    weeks = np.sort(panel["week"].unique())
    split = int(np.floor(0.8 * len(weeks)))
    train_end = pd.Timestamp(weeks[split - 1]) + pd.Timedelta(days=7)

    grid = panel[panel["week"].isin(weeks[:split])][["cell_id", "week"]]
    grid = grid[grid["cell_id"].isin(sorted(panel["cell_id"].unique())[:3])]

    params = fit_etas_per_cell(raw, train_end)
    clean = predict_etas(params, grid, raw).sort_values(["cell_id", "week"])

    corrupted = raw.copy()
    future = corrupted["time"] >= train_end
    assert future.sum() > 0, "nothing to corrupt - the test would prove nothing"
    corrupted.loc[future, "mag"] = corrupted.loc[future, "mag"] + 2.0
    dirty = predict_etas(params, grid, corrupted).sort_values(["cell_id", "week"])

    np.testing.assert_array_equal(
        clean["lambda_pred"].to_numpy(), dirty["lambda_pred"].to_numpy(),
        err_msg="a training-period intensity reacted to events from the forecast period",
    )
