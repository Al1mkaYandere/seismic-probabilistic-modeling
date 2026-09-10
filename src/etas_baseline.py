"""Per-cell temporal ETAS (Ogata 1988) baseline for earthquake count prediction.

Model:
    lambda_c(t) = mu_c + K_c * sum_{t_i < t} exp(a*(m_i - M_c)) * (t - t_i + c_c)^{-p_c}

Parameters per cell: (mu, K, c, p, a).
Prediction: integrate lambda_c over each target weekly interval to get E[Y_c^(t)].
Cells with fewer than MIN_EVENTS training events use a constant-rate fallback.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

from src import config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

MIN_EVENTS: int = 5
M_C_DEFAULT: float = 3.0
WEEK_SECONDS: float = 7.0 * 24.0 * 3600.0


def _negative_log_likelihood(params: np.ndarray, t_days: np.ndarray, m: np.ndarray, m_c: float) -> float:
    """ETAS NLL on event times (in days since first event)."""
    mu, K, c, p, a = params
    if mu <= 0 or K < 0 or c <= 0 or p <= 0.9:
        return 1e12
    T = float(t_days[-1])
    n = len(t_days)

    # Conditional intensity at each event (excluding its own contribution)
    lam = np.full(n, mu, dtype=np.float64)
    for i in range(1, n):
        dt = t_days[i] - t_days[:i]
        lam[i] += K * np.sum(np.exp(a * (m[:i] - m_c)) / (dt + c) ** p)
    lam = np.maximum(lam, 1e-12)

    log_lam_sum = float(np.sum(np.log(lam)))

    # Integrated intensity (background + aftershock integral)
    integral_bg = mu * T
    integral_af = 0.0
    for i in range(n):
        # integral from t_i to T of K*exp(a*(m_i-Mc))*(s-t_i+c)^{-p} ds
        dt_remaining = T - t_days[i]
        if dt_remaining <= 0:
            continue
        integral_af += (K * np.exp(a * (m[i] - m_c)) / (p - 1.0)) * (
            c ** (1.0 - p) - (dt_remaining + c) ** (1.0 - p)
        )
    total_integral = integral_bg + integral_af

    nll = total_integral - log_lam_sum
    return float(nll) if np.isfinite(nll) else 1e12


def _fit_cell(
    t_days: np.ndarray,
    m: np.ndarray,
    m_c: float = M_C_DEFAULT,
    seed: int | np.random.SeedSequence = 42,
) -> dict:
    """Fit ETAS parameters for a single cell. Returns param dict.

    ``seed`` controls the optimizer's random restart jitter (see
    ``x0_jitter`` below). One RNG is created per call and reused across all
    three restarts, so the three restarts stay different from each other but
    identical from run to run given the same ``seed``.
    """
    x0 = np.array([0.01, 0.1, 0.01, 1.1, 1.0])
    bounds = [(1e-6, 10.0), (1e-6, 10.0), (1e-6, 5.0), (1.001, 3.0), (0.1, 3.0)]
    best_res = None
    best_val = np.inf
    rng = np.random.default_rng(seed)
    for _ in range(3):
        x0_jitter = x0 * np.exp(rng.normal(0, 0.3, size=x0.shape))
        x0_jitter = np.clip(x0_jitter, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    _negative_log_likelihood,
                    x0_jitter,
                    args=(t_days, m, m_c),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-10},
                )
            if res.fun < best_val:
                best_val = res.fun
                best_res = res
        except Exception:
            continue
    if best_res is None or not best_res.success:
        return {"mu": float(np.mean(np.diff(t_days)) ** -1 if len(t_days) > 1 else 0.01),
                "K": 0.0, "c": 0.01, "p": 1.1, "a": 1.0, "fallback": True}
    mu, K, c, p, a = best_res.x
    return {"mu": mu, "K": K, "c": c, "p": p, "a": a, "fallback": False}


def _predict_cell(
    params: dict,
    hist_t_days: np.ndarray,
    hist_m: np.ndarray,
    target_week_starts_days: np.ndarray,
    week_duration_days: float = 7.0,
    m_c: float = M_C_DEFAULT,
) -> np.ndarray:
    """Integrate lambda_c over each target week to get expected counts.

    ``hist_t_days``/``hist_m`` are the event history available AT PREDICTION
    TIME (relative days from the cell's ``origin`` and magnitudes) — the
    caller (``predict_etas``) is responsible for restricting this to events
    strictly before each target week's start. This is distinct from the
    events used to FIT ``params`` (which are restricted to ``train_end`` by
    ``fit_etas_per_cell``): the model's parameters are estimated only on the
    training split, but the aftershock history used at prediction time may
    extend further, up to (not including) the week being predicted.
    """
    mu = params["mu"]
    K = params.get("K", 0.0)
    c = params.get("c", 0.01)
    p = params.get("p", 1.1)
    a = params.get("a", 1.0)

    preds = np.full(len(target_week_starts_days), mu * week_duration_days, dtype=np.float64)
    if K <= 0 or len(hist_t_days) == 0:
        return np.maximum(preds, 1e-9)

    for j, t_start in enumerate(target_week_starts_days):
        t_end = t_start + week_duration_days
        af = 0.0
        for i in range(len(hist_t_days)):
            ti = float(hist_t_days[i])
            mi = float(hist_m[i])
            coef = K * np.exp(a * (mi - m_c))
            # integral from max(t_start, ti) to t_end of coef*(s-ti+c)^{-p} ds
            s0 = max(t_start, ti)
            if s0 >= t_end:
                continue
            dt0 = s0 - ti + c
            dt1 = t_end - ti + c
            if p == 1.0:
                af += coef * (np.log(dt1) - np.log(dt0))
            else:
                af += coef / (1.0 - p) * (dt1 ** (1.0 - p) - dt0 ** (1.0 - p))
        preds[j] += af
    return np.maximum(preds, 1e-9)


def fit_etas_per_cell(
    events_df: pd.DataFrame,
    train_end: pd.Timestamp,
    m_c: float = M_C_DEFAULT,
    seed: int = 42,
) -> dict[str, dict]:
    """Fit per-cell ETAS on events before train_end.

    Parameters
    ----------
    events_df : pd.DataFrame
        Raw events with columns ``time`` (datetime), ``latitude``, ``longitude``, ``mag``,
        ``cell_id``.
    train_end : pd.Timestamp
        Exclusive upper bound for training events.
    m_c : float
        Magnitude of completeness threshold.
    seed : int
        Base seed for the optimizer's restart jitter (see ``_fit_cell``). Each
        cell gets its own deterministic child stream spawned from this seed
        (via ``numpy.random.SeedSequence.spawn``), in the fixed order that
        ``groupby("cell_id")`` already sorts cells into. So cells differ from
        each other but the whole fit is identical across repeated calls with
        the same ``events_df``, ``train_end`` and ``seed``.

    Returns
    -------
    dict[str, dict]
        Mapping from cell_id to fitted parameter dict.
    """
    # Normalize time column to tz-naive for arithmetic
    events_work = events_df.copy()
    if hasattr(events_work["time"].dt, "tz") and events_work["time"].dt.tz is not None:
        events_work["time"] = events_work["time"].dt.tz_localize(None)
    if isinstance(train_end, pd.Timestamp) and train_end.tzinfo is not None:
        train_end = train_end.tz_localize(None)
    train = events_work[events_work["time"] < train_end].copy()
    params_by_cell: dict[str, dict] = {}

    origin = train["time"].min()
    if pd.isnull(origin):
        return params_by_cell
    # Normalize origin to tz-naive for consistency
    if hasattr(origin, "tzinfo") and origin.tzinfo is not None:
        origin = origin.tz_localize(None)

    seed_seq = np.random.SeedSequence(seed)
    for cell_id, grp in train.groupby("cell_id"):
        grp_sorted = grp.sort_values("time")
        t_days = (grp_sorted["time"] - origin).dt.total_seconds().to_numpy() / 86400.0
        m_arr = grp_sorted["mag"].to_numpy(dtype=np.float64)
        # One spawn per cell, in groupby's (sorted, deterministic) order, so
        # every cell gets its own reproducible stream regardless of whether
        # it ends up in the fallback branch below or actually gets fitted.
        cell_seed = seed_seq.spawn(1)[0]
        if len(t_days) < MIN_EVENTS:
            rate = len(t_days) / max((train_end - origin).total_seconds() / 86400.0, 1.0)
            params_by_cell[str(cell_id)] = {
                "mu": float(rate), "K": 0.0, "c": 0.01, "p": 1.1, "a": 1.0,
                "fallback": True, "origin": origin,
            }
        else:
            p = _fit_cell(t_days, m_arr, m_c, seed=cell_seed)
            p["origin"] = origin
            params_by_cell[str(cell_id)] = p

    return params_by_cell


def predict_etas(
    params_by_cell: dict[str, dict],
    weeks_grid: pd.DataFrame,
    events_df: pd.DataFrame,
    m_c: float = M_C_DEFAULT,
) -> pd.DataFrame:
    """Predict expected counts for each (cell_id, week) in weeks_grid.

    ETAS is a conditional-intensity model: its aftershock term depends on
    "what happened recently", so predicting a given week requires the event
    history up to (but not including) that week — not just the history
    available at fit time. ``fit_etas_per_cell`` still estimates PARAMETERS
    only from events before ``train_end`` (no leakage there); this function
    separately supplies the full event HISTORY, which may extend into the
    forecast period. For each (cell_id, week) row, only that cell's events
    with ``time`` strictly less than the week's start are used.

    Parameters
    ----------
    params_by_cell : dict
        Output of ``fit_etas_per_cell``.
    weeks_grid : pd.DataFrame
        Must have columns ``cell_id`` and ``week`` (Timestamp).
    events_df : pd.DataFrame
        Full raw event catalog (``time``, ``mag``, ``cell_id``) — not
        restricted to the training period. Events at or after a given week's
        start are excluded from that week's prediction, but events between
        ``train_end`` and the week's start (i.e. inside the forecast period)
        ARE included.

    Returns
    -------
    pd.DataFrame
        Columns: ``cell_id``, ``week``, ``lambda_pred``.
    """
    events_work = events_df[["time", "mag", "cell_id"]].copy()
    if hasattr(events_work["time"].dt, "tz") and events_work["time"].dt.tz is not None:
        events_work["time"] = events_work["time"].dt.tz_localize(None)

    weeks_grid = weeks_grid.copy()
    weeks_grid["week"] = pd.to_datetime(weeks_grid["week"])
    if hasattr(weeks_grid["week"].dt, "tz") and weeks_grid["week"].dt.tz is not None:
        weeks_grid["week"] = weeks_grid["week"].dt.tz_localize(None)

    # Sort each cell's events by time once, so that per-row history lookups
    # below can advance a pointer instead of re-filtering the whole catalog
    # for every (cell, week) row. Without this, a naive "filter events_df for
    # every panel row" implementation degrades to O(rows * total events).
    events_by_cell: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cid, grp in events_work.groupby("cell_id"):
        grp_sorted = grp.sort_values("time")
        events_by_cell[str(cid)] = (
            grp_sorted["time"].to_numpy(),
            grp_sorted["mag"].to_numpy(dtype=np.float64),
        )
    empty_times = np.array([], dtype="datetime64[ns]")
    empty_mags = np.array([], dtype=np.float64)

    rows = []
    for cid, wk_grp in weeks_grid.groupby("cell_id", sort=False):
        cid = str(cid)
        wk_sorted = wk_grp.sort_values("week")
        if cid not in params_by_cell:
            for wk in wk_sorted["week"]:
                rows.append({"cell_id": cid, "week": pd.Timestamp(wk), "lambda_pred": 0.1})
            continue

        p = params_by_cell[cid]
        origin = p["origin"]
        origin_dt64 = pd.Timestamp(origin).to_datetime64()
        ev_times, ev_mags = events_by_cell.get(cid, (empty_times, empty_mags))
        ev_t_days = (ev_times - origin_dt64) / np.timedelta64(1, "D")
        n_events = len(ev_t_days)

        # Weeks are processed in ascending order per cell, so the set of
        # "events strictly before this week's start" only ever grows: a
        # single forward-moving pointer replaces a fresh scan per row.
        ptr = 0
        for wk in wk_sorted["week"]:
            wk = pd.Timestamp(wk)
            t_start_days = (wk - origin).total_seconds() / 86400.0
            while ptr < n_events and ev_t_days[ptr] < t_start_days:
                ptr += 1
            hist_t = ev_t_days[:ptr]
            hist_m = ev_mags[:ptr]
            pred = _predict_cell(p, hist_t, hist_m, np.array([t_start_days]), m_c=m_c)[0]
            rows.append({"cell_id": cid, "week": wk, "lambda_pred": float(pred)})

    return pd.DataFrame(rows)


def run_etas_static(
    events_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    m_c: float = M_C_DEFAULT,
) -> dict[str, float]:
    """Run ETAS on 80/20 static split of panel weeks and return metrics dict.

    Parameters
    ----------
    events_df : pd.DataFrame
        Raw events (time, latitude, longitude, mag, cell_id).
    panel_df : pd.DataFrame
        Processed panel (cell_id, week, Y).

    Returns
    -------
    dict with keys MAE, RMSE, Mean_Poisson_Deviance, alpha_hat, Status.
    """
    unique_weeks = np.sort(panel_df["week"].dropna().unique())
    if len(unique_weeks) < 2:
        return {"MAE": np.nan, "RMSE": np.nan, "Mean_Poisson_Deviance": np.nan,
                "alpha_hat": np.nan, "Status": "FAILED"}
    split_w = int(np.floor(0.8 * len(unique_weeks)))
    train_weeks = unique_weeks[:split_w]
    test_weeks = unique_weeks[split_w:]
    train_end_raw = pd.Timestamp(train_weeks[-1])
    if train_end_raw.tzinfo is not None:
        train_end_raw = train_end_raw.tz_localize(None)
    train_end = train_end_raw + pd.Timedelta(days=7)

    logger.info("ETAS static: fitting on %d cells...", panel_df["cell_id"].nunique())
    params_by_cell = fit_etas_per_cell(events_df, train_end, m_c)

    test_panel = panel_df[panel_df["week"].isin(test_weeks)][["cell_id", "week", "Y"]].copy()
    pred_df = predict_etas(params_by_cell, test_panel[["cell_id", "week"]], events_df, m_c)
    test_panel = test_panel.merge(pred_df, on=["cell_id", "week"], how="left")
    test_panel["lambda_pred"] = test_panel["lambda_pred"].fillna(0.1).clip(lower=1e-9)

    # ETAS is a full Poisson predictive distribution, not just a point forecast:
    # lambda is the mean of P(Y = k) = exp(-lambda) lambda^k / k!. The rest of the
    # pipeline stores every other model's predictions per (cell, week), so that
    # the probabilistic evaluation can score them all on the same 2448 rows.
    # These were being computed and then dropped, which left the seismological
    # baseline out of every distributional comparison.
    out_path = config.ETAS_TEST_PREDICTIONS_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test_panel[["cell_id", "week", "Y", "lambda_pred"]].rename(
        columns={"Y": "y_true"}
    ).to_csv(out_path, index=False)
    logger.info("Saved ETAS test predictions to %s (%d rows)", out_path, len(test_panel))

    y_true = test_panel["Y"].astype(float).to_numpy()
    y_pred = test_panel["lambda_pred"].to_numpy()
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mpd = float(mean_poisson_deviance(y_true, y_pred))
    logger.info("ETAS static: MAE=%.4f RMSE=%.4f MPD=%.4f", mae, rmse, mpd)
    return {"MAE": mae, "RMSE": rmse, "Mean_Poisson_Deviance": mpd,
            "alpha_hat": np.nan, "Status": "SUCCESS"}
