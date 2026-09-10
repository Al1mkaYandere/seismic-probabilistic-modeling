"""Behavioural checks for ETAS: response to a strong event and reproducibility.

Both tests in this file used to be expected failures (xfail) documenting known
issues A1 and A2. A2 (unseeded optimizer restarts) was fixed in microstep 2a.
A1 (``predict_etas`` only seeing events up to ``train_end``, never events
inside the forecast period) is fixed in step 5: ``predict_etas`` now takes the
full event catalog and, for each (cell_id, week) row, uses that cell's events
strictly before the week's start — parameters are still fit only on the
training split, so this does not reopen the leakage A2 closed. Both xfail
markers have been removed; both tests now run as normal (passing) tests.
The synthetic catalog and the expected outcome were worked out and checked by
hand in advance, not fitted to whatever the code currently outputs — see
the comments inside each test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.etas_baseline import fit_etas_per_cell, predict_etas


def _synthetic_catalog_with_strong_event_inside_forecast_period() -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """A regular background (no clustering) up to train_end, plus one strong
    (M6.9) event AFTER train_end — i.e. inside the forecast period.

    Returns (events_df, train_end, week_before_start, week_after_start), where
    week_before_start is a week roughly a month before the strong event and
    week_after_start is the week right after it. Both weeks fall AFTER
    train_end, i.e. both are part of the forecast period.
    """
    origin_date = pd.Timestamp("2010-01-04")
    bg_days = np.arange(0, 720, 20)  # 36 background events, one every 20 days
    bg_times = [origin_date + pd.Timedelta(days=int(d)) for d in bg_days]
    bg_mags = [3.2] * len(bg_days)

    train_end = origin_date + pd.Timedelta(days=730)
    big_event_day = 800
    big_time = origin_date + pd.Timedelta(days=big_event_day)

    events_df = pd.DataFrame(
        {
            "time": bg_times + [big_time],
            "latitude": [40.0] * (len(bg_days) + 1),
            "longitude": [70.0] * (len(bg_days) + 1),
            "mag": bg_mags + [6.9],
            "cell_id": ["TEST"] * (len(bg_days) + 1),
        }
    )

    week_before_start = origin_date + pd.Timedelta(days=big_event_day - 28)  # a month (4 weeks) before
    week_after_start = origin_date + pd.Timedelta(days=big_event_day + 1)  # right after

    assert week_before_start > train_end and week_after_start > train_end, (
        "both forecast weeks must fall after train_end, otherwise this test "
        "does not exercise behaviour inside the forecast period"
    )
    return events_df, train_end, week_before_start, week_after_start


def test_etas_forecast_rises_after_strong_event_inside_forecast_period():
    events_df, train_end, week_before_start, week_after_start = (
        _synthetic_catalog_with_strong_event_inside_forecast_period()
    )
    params = fit_etas_per_cell(events_df, train_end)

    weeks_grid = pd.DataFrame({"cell_id": ["TEST", "TEST"], "week": [week_before_start, week_after_start]})
    pred = predict_etas(params, weeks_grid, events_df)

    pred_before = float(pred.loc[pred["week"] == week_before_start, "lambda_pred"].iloc[0])
    pred_after = float(pred.loc[pred["week"] == week_after_start, "lambda_pred"].iloc[0])

    assert pred_after > pred_before, (
        f"forecast after the strong event ({pred_after}) should be higher "
        f"than the forecast a month before it ({pred_before})"
    )


def test_fit_etas_per_cell_is_reproducible_across_calls():
    events_df, train_end, _, _ = _synthetic_catalog_with_strong_event_inside_forecast_period()

    params_1 = fit_etas_per_cell(events_df, train_end)["TEST"]
    params_2 = fit_etas_per_cell(events_df, train_end)["TEST"]

    for key in ("mu", "K", "c", "p", "a"):
        assert params_1[key] == pytest.approx(params_2[key], rel=1e-9), (
            f"parameter {key!r} differs between two runs on the same data: "
            f"{params_1[key]} vs {params_2[key]}"
        )
