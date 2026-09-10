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

    metrics = etas_baseline.run_etas_static(events_df, panel_df)
    assert metrics["Status"] == "SUCCESS"

    saved = pd.read_csv(out_path)

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
