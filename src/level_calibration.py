"""Level bias of each forecast, and the comparison that remains once it is removed.

A forecast can be wrong in two unrelated ways: it can put the average in the
wrong place, and it can distribute its mass badly. The first is repaired by one
multiplier and is rarely the interesting difference between two models; the
second is what a model comparison is supposed to be about. A table that ignores
the first therefore ranks calibration, not skill.

This is not hypothetical here. In the year-by-year protocol at completeness
threshold 4.5, ETAS predicts 1.8 times more events than occurred. Taken at face
value the neural network beat it by 8.4% of Poisson deviance, six years out of
six; once every model is allowed the same one-number level correction, fitted
only on earlier years, the gap is 0.8% and indistinguishable from zero. The
whole difference was the level.

The correction is applied to every model, not only to the biased one - correcting
only the baseline would be repairing the competition to fit our conclusion, while
correcting everyone makes the comparison conservative, because it helps whoever
is most biased.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

from src import config

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS: dict[str, str] = {
    "NB_Enhanced_MLE": "pred_nb_glm",
    "Hybrid_DL_Enhanced": "pred_hybrid_dl",
    "Neural_Poisson_Enhanced": "pred_neural_poisson",
    "ETAS_Per_Cell": "pred_etas",
}

MIN_PREDICTED_TOTAL: float = 1e-9


def level_multiplier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The single number that puts a forecast's total where the truth is.

    Deliberately the ratio of totals rather than a fitted regression slope: it
    is the correction a reader can apply by hand, and it leaves the shape of the
    forecast untouched, which is the part being compared.

    Scaling every forecast by a constant divides this multiplier by the same
    constant, so the corrected forecast is unchanged. That identity is what the
    correction rests on, and it holds while the totals stay in a range float64
    can represent: a review found it degenerating below about 1e-9 of the
    original scale, where the predicted total sinks under the guard below and
    the multiplier is refused rather than computed. Forecast totals anywhere
    near that are meaningless anyway.
    """
    predicted_total = float(np.sum(y_pred))
    if not np.isfinite(predicted_total) or predicted_total <= MIN_PREDICTED_TOTAL:
        return np.nan
    return float(np.sum(y_true)) / predicted_total


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Poisson deviance is undefined at a forecast of exactly zero, and a model
    # is entitled to forecast zero. The floor keeps that from ending the run.
    safe = np.clip(y_pred, 1e-9, None)
    return {
        "MAE": float(mean_absolute_error(y_true, safe)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, safe))),
        "Mean_Poisson_Deviance": float(mean_poisson_deviance(y_true, safe)),
    }


def calibration_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (test year, model): its level bias, and metrics with and without.

    The multiplier for a year is fitted on the years BEFORE it and on nothing
    else. Fitting it on the year being scored would be reading the answer: every
    model would improve, and the most biased one would improve most, which is
    exactly the comparison this table exists to avoid.

    The earliest test year therefore has no multiplier at all. It is reported
    with blanks rather than with a multiplier of one, because "not measurable
    here" and "measured, and it was one" are different statements.
    """
    out: list[dict] = []
    years = sorted(rows["Year"].unique())
    for year in years:
        past = rows[rows["Year"] < year]
        now = rows[rows["Year"] == year]
        for model, column in PREDICTION_COLUMNS.items():
            pred_now = now[column].to_numpy(dtype=np.float64)
            y_now = now["y_true"].to_numpy(dtype=np.float64)
            usable_now = np.isfinite(pred_now)

            record = {
                "Year": int(year),
                "Model": model,
                "rows_total": int(len(pred_now)),
                "rows_missing": int((~usable_now).sum()),
                "observed_total": float(np.sum(y_now)),
                "predicted_total": float(np.sum(pred_now[usable_now])),
            }
            # A year with no events at all would make this a division by nothing;
            # an enormous ratio there would say "wildly biased" about a year that
            # carries no information on bias either way.
            record["bias_ratio"] = (
                record["predicted_total"] / record["observed_total"]
                if record["observed_total"] > 0 else np.nan
            )

            # Metrics are only reported for a complete year. Computing them on the
            # surviving rows would quietly compare this model on a smaller sample
            # than the others in the same table. The row itself stays, with blanks
            # and a count, so a year cannot disappear from the comparison unnoticed.
            complete = usable_now.all()
            if complete:
                record.update({f"{k}_raw": v for k, v in _metrics(y_now, pred_now).items()})

            # The multiplier is fitted on whatever earlier rows this model actually
            # produced. Demanding a spotless past would let one missing week in one
            # year leave every later year uncorrected - the failure spreads forward
            # instead of staying where it happened.
            multiplier = np.nan
            past_used = 0
            if not past.empty:
                pred_past = past[column].to_numpy(dtype=np.float64)
                usable_past = np.isfinite(pred_past)
                past_used = int(usable_past.sum())
                if past_used > 0:
                    multiplier = level_multiplier(
                        past["y_true"].to_numpy(dtype=np.float64)[usable_past],
                        pred_past[usable_past],
                    )
            record["level_multiplier"] = multiplier
            record["past_rows_used"] = past_used
            if complete and np.isfinite(multiplier):
                record.update({f"{k}_calibrated": v
                               for k, v in _metrics(y_now, pred_now * multiplier).items()})
            out.append(record)
    return pd.DataFrame(out)


def run_level_calibration() -> pd.DataFrame:
    """Read the year-by-year forecasts, write the level-corrected comparison."""
    source = config.OUTPUT_DIR / "walk_forward_predictions.csv"
    if not source.exists():
        logger.warning("Level calibration skipped: %s not found", source)
        return pd.DataFrame()

    rows = pd.read_csv(source)
    table = calibration_table(rows)
    out_path = config.OUTPUT_DIR / "level_calibration.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    worst = table.loc[table["bias_ratio"].sub(1.0).abs().idxmax()] if not table.empty else None
    if worst is not None:
        logger.info("Level calibration written to %s; largest bias: %s in %d predicts "
                    "%.2fx the events that occurred",
                    out_path, worst["Model"], worst["Year"], worst["bias_ratio"])
    return table
