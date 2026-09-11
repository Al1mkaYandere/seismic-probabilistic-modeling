"""The year-by-year check keeps its forecasts, not just its verdicts.

The aggregated table says who won each year. It cannot say why, and the
completeness experiment produced a question only the forecasts can answer: at
the honest threshold ETAS cannot be fitted in 4 of 17 cells and falls back to a
constant rate there, so part of its loss may be missing data rather than a
property of the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_poisson_deviance

from src.validation import fold_prediction_rows

REPO_ROOT = Path(__file__).resolve().parent.parent
WF_RESULTS = REPO_ROOT / "outputs" / "walk_forward_results.csv"
WF_PREDICTIONS = REPO_ROOT / "outputs" / "walk_forward_predictions.csv"

MODEL_COLUMN = {
    "NB_Enhanced_MLE": "pred_nb_glm",
    "Hybrid_DL_Enhanced": "pred_hybrid_dl",
    "Neural_Poisson_Enhanced": "pred_neural_poisson",
    "ETAS_Per_Cell": "pred_etas",
}


def _fold() -> pd.DataFrame:
    return pd.DataFrame({
        "cell_id": ["38.00_65.00", "38.00_68.00", "41.00_65.00"],
        "week": pd.to_datetime(["2022-01-03", "2022-01-03", "2022-01-10"]),
        "Y": [0.0, 2.0, 1.0],
    })


def test_rows_keep_the_order_and_the_truth_of_the_fold() -> None:
    """A forecast that lands on the wrong row is worse than no forecast at all."""
    df_test = _fold()
    rows = fold_prediction_rows(
        df_test, df_test["Y"].to_numpy(), 2022,
        {"pred_hybrid_dl": np.array([0.11, 0.22, 0.33])},
        etas_fallback_cells=set(),
    )
    assert list(rows["cell_id"]) == list(df_test["cell_id"])
    assert list(rows["week"]) == list(df_test["week"])
    assert list(rows["y_true"]) == [0.0, 2.0, 1.0]
    assert list(rows["pred_hybrid_dl"]) == [0.11, 0.22, 0.33]
    assert list(rows["Year"]) == [2022, 2022, 2022]


def test_a_model_that_failed_this_year_leaves_blanks_not_zeros() -> None:
    """A failed fit must not read later as a forecast of zero events."""
    df_test = _fold()
    rows = fold_prediction_rows(
        df_test, df_test["Y"].to_numpy(), 2022,
        {"pred_etas": None, "pred_nb_glm": np.array([0.5, 0.5, 0.5])},
        etas_fallback_cells=set(),
    )
    assert rows["pred_etas"].isna().all()
    assert not rows["pred_nb_glm"].isna().any()


def test_fallback_cells_are_marked_so_the_comparison_can_exclude_them() -> None:
    """The flag exists to answer one question; it has to name the right rows."""
    df_test = _fold()
    rows = fold_prediction_rows(
        df_test, df_test["Y"].to_numpy(), 2022,
        {"pred_etas": np.array([0.1, 0.2, 0.3])},
        etas_fallback_cells={"41.00_65.00"},
    )
    marked = set(rows.loc[rows["etas_fallback_cell"], "cell_id"])
    assert marked == {"41.00_65.00"}
    assert not rows.loc[rows["cell_id"] == "38.00_65.00", "etas_fallback_cell"].any()


def test_saved_rows_reproduce_the_published_yearly_numbers() -> None:
    """The forecasts and the verdicts must be the same run, not two runs.

    Recomputes every year-model deviance from the per-row file and compares it
    with the aggregated table. Misaligned rows, a stale file or a model whose
    forecasts were collected from a different fold all show up here.
    """
    if not WF_PREDICTIONS.exists():
        pytest.skip("per-row walk-forward predictions not produced yet")

    rows = pd.read_csv(WF_PREDICTIONS)
    results = pd.read_csv(WF_RESULTS)

    checked = 0
    for _, r in results.iterrows():
        column = MODEL_COLUMN[r["Model"]]
        fold = rows[rows["Year"] == r["Year"]]
        assert not fold.empty, f"no rows kept for {r['Year']}"
        pred = fold[column].to_numpy(dtype=float)
        if np.isnan(pred).any() or not np.isfinite(r["Mean_Poisson_Deviance"]):
            continue
        recomputed = mean_poisson_deviance(
            fold["y_true"].to_numpy(dtype=float), np.clip(pred, 1e-9, None)
        )
        assert recomputed == pytest.approx(r["Mean_Poisson_Deviance"], rel=1e-9), (
            f"{r['Year']} {r['Model']}: rows give {recomputed}, table says "
            f"{r['Mean_Poisson_Deviance']}"
        )
        checked += 1

    assert checked >= 20, f"only {checked} year-model pairs checked - the file looks empty"
