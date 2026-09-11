"""Level bias must be measured without reading the answer, and removed exactly.

The defect that made this module necessary was not caught by four separate
checks of the same comparison, because all four re-sliced the same forecasts and
none asked whether their total was right. These tests are built around the one
property that pins the correction down exactly rather than approximately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.level_calibration import PREDICTION_COLUMNS, calibration_table, level_multiplier

RNG = np.random.default_rng(20260911)


def _rows(scale_by_year: dict[int, float], n_per_year: int = 60) -> pd.DataFrame:
    """Three test years. Every model predicts the truth times the year's factor."""
    frames = []
    for year, factor in scale_by_year.items():
        y = RNG.poisson(0.4, n_per_year).astype(float)
        frame = pd.DataFrame({
            "Year": year,
            "cell_id": [f"c{i % 5}" for i in range(n_per_year)],
            "week": pd.date_range(f"{year}-01-01", periods=n_per_year, freq="W-MON"),
            "y_true": y,
        })
        for column in PREDICTION_COLUMNS.values():
            frame[column] = np.clip(y * factor + 0.05, 1e-6, None)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_multiplier_is_fitted_on_earlier_years_and_cannot_see_the_scored_one() -> None:
    """A multiplier that peeks at the year it corrects flatters the worst model most."""
    calm = _rows({2018: 1.0, 2019: 1.0, 2020: 1.0})
    wild = calm.copy()
    later = wild["Year"] == 2020
    for column in PREDICTION_COLUMNS.values():
        wild.loc[later, column] = wild.loc[later, column] * 50.0      # absurd, last year only

    a = calibration_table(calm)
    b = calibration_table(wild)
    for year in (2018, 2019):
        ka = a.loc[a["Year"] == year, "level_multiplier"].to_numpy()
        kb = b.loc[b["Year"] == year, "level_multiplier"].to_numpy()
        np.testing.assert_allclose(np.nan_to_num(ka, nan=-1), np.nan_to_num(kb, nan=-1))

    # ...and the year that WAS changed must show it, or the test proves nothing.
    assert (b.loc[b["Year"] == 2020, "bias_ratio"] > 10 * a.loc[a["Year"] == 2020, "bias_ratio"].to_numpy()).all()

    # The harder case, and the one a mutation check exposed as untested: peeking
    # at the year being SCORED rather than at a later one. Spoiling 2019 alone
    # must leave 2019's own multiplier untouched, because it is fitted on 2018.
    # It must, however, move 2020's - that multiplier is allowed to see 2019.
    spoiled = calm.copy()
    scored = spoiled["Year"] == 2019
    for column in PREDICTION_COLUMNS.values():
        spoiled.loc[scored, column] = spoiled.loc[scored, column] * 50.0
    c = calibration_table(spoiled)

    k_before = a.loc[a["Year"] == 2019, "level_multiplier"].to_numpy()
    k_after = c.loc[c["Year"] == 2019, "level_multiplier"].to_numpy()
    np.testing.assert_allclose(k_before, k_after, rtol=1e-12,
                               err_msg="the multiplier for a year reacted to that year's own data")
    assert (c.loc[c["Year"] == 2020, "level_multiplier"].to_numpy()
            < 0.5 * a.loc[a["Year"] == 2020, "level_multiplier"].to_numpy()).all(), (
        "the 2020 multiplier ignored 2019 - it is not being fitted on the past at all"
    )


def test_scaling_a_forecast_leaves_the_calibrated_metrics_identical() -> None:
    """The exact identity the correction rests on.

    Multiply every forecast of one model by any constant and the fitted
    multiplier divides by the same constant, so the corrected forecast - and
    therefore every corrected metric - must come out bit for bit the same.
    A "the numbers moved" check would pass even if the correction reached only
    half of the calculation; this cannot.
    """
    rows = _rows({2018: 1.0, 2019: 1.3, 2020: 0.7})
    base = calibration_table(rows)

    for constant in (0.25, 3.0, 17.5):
        scaled = rows.copy()
        scaled["pred_etas"] = scaled["pred_etas"] * constant
        table = calibration_table(scaled)

        for year in (2019, 2020):
            b = base[(base["Year"] == year) & (base["Model"] == "ETAS_Per_Cell")].iloc[0]
            t = table[(table["Year"] == year) & (table["Model"] == "ETAS_Per_Cell")].iloc[0]
            for metric in ("MAE", "RMSE", "Mean_Poisson_Deviance"):
                assert t[f"{metric}_calibrated"] == pytest.approx(
                    b[f"{metric}_calibrated"], rel=1e-12
                ), f"{metric} moved under a pure rescaling by {constant}"
            assert t["level_multiplier"] == pytest.approx(b["level_multiplier"] / constant, rel=1e-12)
            # The uncorrected metric MUST move - otherwise the scaling did nothing.
            assert t["Mean_Poisson_Deviance_raw"] != pytest.approx(b["Mean_Poisson_Deviance_raw"])


def test_a_forecast_already_at_the_right_level_is_barely_touched() -> None:
    rows = _rows({2018: 1.0, 2019: 1.0, 2020: 1.0})
    table = calibration_table(rows)
    later = table[table["Year"] > 2018]
    assert np.allclose(later["level_multiplier"], 1.0, atol=0.25)
    assert np.allclose(later["Mean_Poisson_Deviance_calibrated"],
                       later["Mean_Poisson_Deviance_raw"], rtol=0.25)


def test_the_first_year_is_left_blank_rather_than_given_a_made_up_multiplier() -> None:
    """"Not measurable here" and "measured, and it was one" are different claims."""
    table = calibration_table(_rows({2018: 1.0, 2019: 2.0, 2020: 0.5}))
    first = table[table["Year"] == 2018]
    assert first["level_multiplier"].isna().all()
    assert first["Mean_Poisson_Deviance_calibrated"].isna().all()
    assert first["Mean_Poisson_Deviance_raw"].notna().all()
    assert table[table["Year"] > 2018]["level_multiplier"].notna().all()


def test_bias_ratio_points_the_right_way() -> None:
    """Over one means the forecast promises more events than happened."""
    over = calibration_table(_rows({2018: 1.0, 2019: 3.0}))
    assert (over[over["Year"] == 2019]["bias_ratio"] > 2.0).all()
    under = calibration_table(_rows({2018: 1.0, 2019: 0.25}))
    assert (under[under["Year"] == 2019]["bias_ratio"] < 0.6).all()


def test_multiplier_refuses_to_invent_a_number_for_an_empty_forecast() -> None:
    assert np.isnan(level_multiplier(np.array([1.0, 2.0]), np.array([0.0, 0.0])))
    assert level_multiplier(np.array([2.0, 2.0]), np.array([1.0, 1.0])) == pytest.approx(2.0)


def test_a_zero_forecast_is_scored_instead_of_ending_the_run() -> None:
    """A model is entitled to forecast zero; Poisson deviance is undefined there.

    Found by a reviewer as the gap it was: removing the floor inside the metrics
    left every test green, because neither the synthetic data here nor the real
    forecasts on disk contain an exact zero.
    """
    rows = _rows({2018: 1.0, 2019: 1.0})
    rows["pred_etas"] = 0.0

    table = calibration_table(rows)
    etas = table[table["Model"] == "ETAS_Per_Cell"]
    assert len(etas) == 2, "the model vanished from the table instead of being scored"
    assert np.isfinite(etas["Mean_Poisson_Deviance_raw"]).all(), (
        "a zero forecast produced a non-finite score"
    )
    assert etas["level_multiplier"].isna().all(), (
        "a forecast that predicts nothing at all cannot be rescaled into one that does"
    )
    # The other models must be unaffected by their neighbour's zeros.
    others = table[table["Model"] != "ETAS_Per_Cell"]
    assert np.isfinite(others["Mean_Poisson_Deviance_raw"]).all()


def test_one_missing_week_does_not_silence_every_later_year() -> None:
    """A gap must stay where it happened instead of spreading forward.

    Before this was fixed, a single missing forecast in one year dropped that
    model's whole year out of the table without a trace AND left every later
    year with no multiplier, because the fitting window demanded a spotless past.
    """
    rows = _rows({2018: 1.0, 2019: 1.0, 2020: 1.0, 2021: 1.0})
    hole = (rows["Year"] == 2019) & (rows.index % 7 == 0)
    rows.loc[hole, "pred_etas"] = np.nan

    table = calibration_table(rows)
    etas = table[table["Model"] == "ETAS_Per_Cell"].set_index("Year")

    assert set(etas.index) == {2018, 2019, 2020, 2021}, "a year disappeared from the table"
    assert etas.loc[2019, "rows_missing"] > 0, "the gap is not recorded anywhere"
    assert np.isnan(etas.loc[2019, "Mean_Poisson_Deviance_raw"]), (
        "an incomplete year was scored as if it were complete"
    )
    for year in (2020, 2021):
        assert np.isfinite(etas.loc[year, "level_multiplier"]), (
            f"{year} lost its multiplier because of a gap in 2019"
        )
        assert np.isfinite(etas.loc[year, "Mean_Poisson_Deviance_calibrated"])
    assert etas.loc[2020, "past_rows_used"] < etas.loc[2020, "rows_total"] * 2, (
        "the fitting window silently used rows it should have skipped"
    )


def test_a_year_with_no_events_reports_no_bias_rather_than_an_enormous_one() -> None:
    rows = _rows({2018: 1.0, 2019: 1.0})
    rows.loc[rows["Year"] == 2019, "y_true"] = 0.0
    table = calibration_table(rows)
    quiet = table[table["Year"] == 2019]
    assert quiet["bias_ratio"].isna().all(), "a year with nothing to predict was called biased"
