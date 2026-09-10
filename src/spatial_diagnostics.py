"""Spatial residual diagnostics: Moran's I with queen contiguity and permutation test.

For each model, compute Pearson-style residuals averaged across weeks, then
test spatial autocorrelation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

logger = logging.getLogger(__name__)

OUTPUT_MORAN_CSV = config.MORAN_RESIDUALS_CSV
OUTPUT_MORAN_WEEKLY_CSV = OUTPUT_MORAN_CSV.parent / "moran_weekly.csv"
OUTPUT_MORAN_INFLUENCE_CSV = OUTPUT_MORAN_CSV.parent / "moran_influence.csv"
OUTPUT_MORAN_FIG = config.MORAN_RESIDUALS_FIG
PREDICTION_STORE = config.TEST_PREDICTIONS_CSV
CALIBRATION_PREDS = config.CALIBRATION_PREDICTIONS_CSV
MODEL_COMPARISON = config.MODEL_COMPARISON_CSV
PROCESSED_DATA = config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE


def _build_queen_weights(cells_df: pd.DataFrame, grid_size: float) -> np.ndarray:
    """Build row-standardised queen-contiguity weight matrix.

    Two cells are queen-neighbours if their lat_grid and lon_grid differ by
    at most grid_size in each direction.
    """
    lats = cells_df["lat_grid"].to_numpy(dtype=float)
    lons = cells_df["lon_grid"].to_numpy(dtype=float)
    n = len(lats)
    W = np.zeros((n, n), dtype=np.float64)
    tol = grid_size * 1.01
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(lats[i] - lats[j]) <= tol and abs(lons[i] - lons[j]) <= tol:
                W[i, j] = 1.0
    # Row-standardise
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return W / row_sums


def _morans_i(x: np.ndarray, W: np.ndarray) -> float:
    """Moran's I statistic."""
    n = len(x)
    xc = x - x.mean()
    numerator = float(n * xc @ W @ xc)
    denominator = float(np.sum(W) * (xc @ xc))
    if abs(denominator) < 1e-12:
        return np.nan
    return numerator / denominator


def _residuals(y: np.ndarray, mu: np.ndarray, alpha: float | None) -> np.ndarray:
    """Pearson-style residual (y - mu) / sqrt(Var); NB variance if alpha is given."""
    if alpha is not None and np.isfinite(alpha) and alpha > 0:
        denom = np.sqrt(np.maximum(mu + alpha * mu ** 2, 1e-12))
    else:
        denom = np.sqrt(np.maximum(mu, 1e-12))
    return (y - mu) / denom


def _permutation_test(x: np.ndarray, W: np.ndarray, B: int = 999) -> tuple[float, float, float]:
    """Returns (observed_I, z_score, p_value) under permutation null."""
    obs = _morans_i(x, W)
    if not np.isfinite(obs):
        return obs, np.nan, np.nan
    perm_vals = np.zeros(B, dtype=np.float64)
    rng = np.random.default_rng(42)
    for b in range(B):
        xp = rng.permutation(x)
        perm_vals[b] = _morans_i(xp, W)
    mean_p = float(np.mean(perm_vals))
    std_p = float(np.std(perm_vals, ddof=1))
    z = (obs - mean_p) / max(std_p, 1e-12)
    # Two-sided permutation p-value with the standard +1 correction: a
    # one-sided p = mean(perm >= obs) tests only for POSITIVE autocorrelation
    # and is structurally doomed to be large whenever the observed I is
    # negative (see A8). (1 + r) / (B + 1) avoids p = 0 with finite B.
    r = int(np.sum(np.abs(perm_vals - mean_p) >= np.abs(obs - mean_p)))
    p_val = (1 + r) / (B + 1)
    return obs, z, p_val


def _weekly_moran(models_to_check: list[dict], comparison_df: pd.DataFrame,
                   cell_order: list[str], W: np.ndarray, B: int = 999) -> pd.DataFrame:
    """Moran's I computed separately on each week's 17-cell residual vector.

    The main table above averages residuals across all 144 test weeks before
    testing for spatial autocorrelation; that averaging cancels out exactly
    the week-by-week signal the test is meant to detect. Here the same
    permutation test is re-run once per week, with no averaging beforehand.
    Aggregation across weeks (mean I, share of weeks with p < 0.05) is left
    to the caller.
    """
    rows: list[dict] = []
    for m in models_to_check:
        if m["source"] == "calib":
            model_name = m["col"]
            y = m["y_true"]
            mu = m["mu_pred"]
            alpha_arr = m.get("alpha_pred")
            alpha_scalar = float(np.mean(alpha_arr)) if alpha_arr is not None else None
            cell_ids_arr = m.get("cell_id")
            weeks_arr = m.get("week")
        else:
            df = m["df"]
            model_name = m["col"]
            y = df["y_true"].to_numpy(dtype=np.float64)
            mu = df[m["col"]].to_numpy(dtype=np.float64)
            alpha_row = comparison_df[comparison_df["model"] == model_name] if not comparison_df.empty else pd.DataFrame()
            alpha_scalar = float(alpha_row["alpha_hat"].iloc[0]) if not alpha_row.empty and "alpha_hat" in alpha_row.columns else None
            cell_ids_arr = df["cell_id"].astype(str).to_numpy() if "cell_id" in df.columns else None
            weeks_arr = df["week"].to_numpy() if "week" in df.columns else None

        if cell_ids_arr is None or weeks_arr is None:
            continue

        try:
            r = _residuals(y, mu, alpha_scalar)
            week_df = pd.DataFrame({
                "cell_id": np.asarray(cell_ids_arr, dtype=str),
                "week": weeks_arr,
                "resid": r,
            })
            for week, sub in week_df.groupby("week"):
                cell_vals = sub.groupby("cell_id")["resid"].mean().reindex(cell_order)
                if cell_vals.isna().any():
                    # Week does not cover all cells; skip rather than fabricate a 0 residual.
                    continue
                obs_I, z, p = _permutation_test(cell_vals.to_numpy(dtype=np.float64), W, B=B)
                rows.append({"model": model_name, "week": week, "moran_I": obs_I, "z_score": z, "p_perm": p})
        except Exception as exc:
            logger.warning("Weekly spatial diagnostics failed for %s: %s", model_name, exc)
    return pd.DataFrame(rows)


def _leave_one_out_moran(
    cell_values: np.ndarray,
    cells_df: pd.DataFrame,
    grid_size: float,
    b: int = 999,
) -> pd.DataFrame:
    """Refit Moran's I with each cell dropped in turn.

    With only ~17 cells a single cell can carry the whole statistic, so the
    full-sample value alone says little. This reports, for every cell, what
    Moran's I and its p-value become once that cell is removed, which makes an
    influential cell immediately visible.
    """
    order = cells_df["cell_id"].tolist()
    rows: list[dict] = []
    for i, dropped in enumerate(order):
        keep = [j for j in range(len(order)) if j != i]
        sub_cells = cells_df.iloc[keep].reset_index(drop=True)
        sub_w = _build_queen_weights(sub_cells, grid_size)
        obs_i, z, p = _permutation_test(cell_values[keep], sub_w, B=b)
        rows.append(
            {
                "dropped_cell": dropped,
                "moran_I_without": obs_i,
                "z_score_without": z,
                "p_perm_without": p,
            }
        )
    return pd.DataFrame(rows)


def _artefact_paths(out_dir: Path | None) -> tuple[Path, Path, Path, Path]:
    """Where the four Moran artefacts go: (residuals, weekly, influence, figure).

    ``out_dir=None`` means the pipeline's own configured locations, and those
    are NOT all in one directory: the three CSVs live in ``outputs/`` while the
    figure lives in ``outputs/figures/``. Deriving the figure's path from the
    CSV directory would silently move it up one level on every real run, and
    the frozen-baseline check would not notice — it only tracks CSVs.

    Kept as a separate function so both branches can be asserted directly
    instead of being re-implemented inside a test.
    """
    if out_dir is None:
        return (OUTPUT_MORAN_CSV, OUTPUT_MORAN_WEEKLY_CSV,
                OUTPUT_MORAN_INFLUENCE_CSV, OUTPUT_MORAN_FIG)
    base = Path(out_dir)
    return (base / OUTPUT_MORAN_CSV.name, base / OUTPUT_MORAN_WEEKLY_CSV.name,
            base / OUTPUT_MORAN_INFLUENCE_CSV.name, base / OUTPUT_MORAN_FIG.name)


def run_spatial_diagnostics(out_dir: Path | None = None) -> pd.DataFrame:
    """Compute Moran's I for residuals of available models and save artefacts.

    Parameters
    ----------
    out_dir : Path | None
        Directory to write all four artefacts into. ``None`` (the default,
        and what the pipeline uses) keeps every configured path exactly as it
        was — see ``_artefact_paths``. Tests pass a temporary directory
        instead: this function is called directly by the test suite, and
        without this argument every test run overwrites the pipeline's Moran
        outputs — harmless while the code is correct, but it silently poisons
        the frozen-baseline comparison when the test run is a deliberate
        mutation of that same code.
    """
    if not PROCESSED_DATA.exists():
        logger.warning("Spatial diagnostics skipped: processed data not found")
        return pd.DataFrame()

    moran_csv, moran_weekly_csv, moran_influence_csv, moran_fig = _artefact_paths(out_dir)

    panel = pd.read_csv(PROCESSED_DATA)
    panel["week"] = pd.to_datetime(panel["week"])

    cells_df = panel[["cell_id", "lat_grid", "lon_grid"]].drop_duplicates("cell_id").reset_index(drop=True)
    gs = float(getattr(config, "GRID_SIZE", 3.0))
    W = _build_queen_weights(cells_df, gs)
    cell_order = cells_df["cell_id"].tolist()
    cell_idx = {c: i for i, c in enumerate(cell_order)}
    n_cells = len(cell_order)

    rows: list[dict] = []
    influence_rows: list[pd.DataFrame] = []
    models_to_check: list[dict] = []

    if PREDICTION_STORE.exists():
        preds_df = pd.read_csv(PREDICTION_STORE)
        if "y_true" in preds_df.columns and "cell_id" in preds_df.columns:
            for col in preds_df.columns:
                if col in ("y_true", "cell_id", "week"):
                    continue
                models_to_check.append({"source": "preds_store", "col": col, "df": preds_df})

    if CALIBRATION_PREDS.exists():
        calib_df = pd.read_csv(CALIBRATION_PREDS)
        if "model" in calib_df.columns and "y_true" in calib_df.columns:
            for model_name, sub in calib_df.groupby("model"):
                models_to_check.append({
                    "source": "calib", "col": str(model_name),
                    "y_true": sub["y_true"].to_numpy(dtype=np.float64),
                    "mu_pred": sub["mu_pred"].to_numpy(dtype=np.float64),
                    "alpha_pred": sub["alpha_pred"].to_numpy(dtype=np.float64) if "alpha_pred" in sub.columns else None,
                    "cell_id": sub["cell_id"].to_numpy(dtype=str) if "cell_id" in sub.columns else None,
                    "week": sub["week"].to_numpy() if "week" in sub.columns else None,
                })

    comparison_df = pd.read_csv(MODEL_COMPARISON) if MODEL_COMPARISON.exists() else pd.DataFrame()

    def _get_cell_residuals(y: np.ndarray, mu: np.ndarray, alpha: float | None,
                            cell_ids_arr: np.ndarray | None) -> np.ndarray:
        r = _residuals(y, mu, alpha)
        if cell_ids_arr is not None:
            cell_mean = pd.Series(r).groupby(pd.Categorical(cell_ids_arr, categories=cell_order), observed=False).mean()
            return cell_mean.reindex(cell_order).fillna(0.0).to_numpy()
        # Average per cell using panel cell ordering if cell_ids not available
        # Fall back to global mean if no cell info
        return np.full(n_cells, float(np.mean(r)))

    for m in models_to_check:
        try:
            if m["source"] == "calib":
                model_name = m["col"]
                y = m["y_true"]
                mu = m["mu_pred"]
                alpha_arr = m.get("alpha_pred")
                alpha_scalar = float(np.mean(alpha_arr)) if alpha_arr is not None else None
                cell_ids_arr = m.get("cell_id")
                cell_r = _get_cell_residuals(y, mu, alpha_scalar, cell_ids_arr)
            else:
                df = m["df"]
                model_name = m["col"]
                y = df["y_true"].to_numpy(dtype=np.float64)
                mu = df[m["col"]].to_numpy(dtype=np.float64)
                alpha_row = comparison_df[comparison_df["model"] == model_name] if not comparison_df.empty else pd.DataFrame()
                alpha_s = float(alpha_row["alpha_hat"].iloc[0]) if not alpha_row.empty and "alpha_hat" in alpha_row.columns else None
                cell_ids_arr = df["cell_id"].astype(str).to_numpy() if "cell_id" in df.columns else None
                cell_r = _get_cell_residuals(y, mu, alpha_s, cell_ids_arr)

            obs_I, z, p = _permutation_test(cell_r, W, B=999)
            rows.append({"model": model_name, "moran_I": obs_I, "z_score": z, "p_perm": p})
            loo = _leave_one_out_moran(cell_r, cells_df, gs)
            loo.insert(0, "model", model_name)
            influence_rows.append(loo)
            logger.info("Moran's I for %s: I=%.4f z=%.3f p=%.4f", model_name, obs_I, z, p)
        except Exception as exc:
            logger.warning("Spatial diagnostics failed for %s: %s", m.get("col", "?"), exc)

    if influence_rows:
        influence = pd.concat(influence_rows, ignore_index=True)
        influence.to_csv(moran_influence_csv, index=False)
        for model_name, grp in influence.groupby("model"):
            worst = grp.loc[grp["p_perm_without"].idxmax()]
            logger.info(
                "Moran influence for %s: dropping %s moves p to %.3f (I=%.4f)",
                model_name, worst["dropped_cell"],
                worst["p_perm_without"], worst["moran_I_without"],
            )

    out_df = pd.DataFrame(rows)
    moran_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(moran_csv, index=False)
    logger.info("Saved Moran results to %s", moran_csv)

    weekly_df = _weekly_moran(models_to_check, comparison_df, cell_order, W, B=999)
    moran_weekly_csv.parent.mkdir(parents=True, exist_ok=True)
    weekly_df.to_csv(moran_weekly_csv, index=False)
    logger.info("Saved weekly Moran results to %s (%d rows)", moran_weekly_csv, len(weekly_df))
    if not weekly_df.empty:
        agg = weekly_df.groupby("model").agg(
            n_weeks=("moran_I", "count"),
            mean_moran_I=("moran_I", "mean"),
            share_p_lt_05=("p_perm", lambda s: float(np.mean(s < 0.05))),
        )
        for model_name, arow in agg.iterrows():
            logger.info(
                "Weekly Moran aggregate for %s: n_weeks=%d mean_I=%.4f share_p<0.05=%.3f",
                model_name, int(arow["n_weeks"]), arow["mean_moran_I"], arow["share_p_lt_05"],
            )

    if not out_df.empty and "moran_I" in out_df.columns:
        try:
            moran_fig.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["#c0392b" if p < 0.05 else "#1F4E79"
                      for p in out_df["p_perm"].fillna(1.0)]
            ax.barh(out_df["model"], out_df["moran_I"], color=colors, alpha=0.85)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
            ax.set_xlabel("Moran's I (red = significant p<0.05)")
            ax.set_title("Spatial autocorrelation of Pearson residuals")
            fig.tight_layout()
            fig.savefig(moran_fig, dpi=250)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Moran figure failed: %s", exc)

    return out_df
