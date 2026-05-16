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
    p_val = float(np.mean(perm_vals >= obs))
    return obs, z, p_val


def run_spatial_diagnostics() -> pd.DataFrame:
    """Compute Moran's I for residuals of available models and save artefacts."""
    if not PROCESSED_DATA.exists():
        logger.warning("Spatial diagnostics skipped: processed data not found")
        return pd.DataFrame()

    panel = pd.read_csv(PROCESSED_DATA)
    panel["week"] = pd.to_datetime(panel["week"])

    cells_df = panel[["cell_id", "lat_grid", "lon_grid"]].drop_duplicates("cell_id").reset_index(drop=True)
    gs = float(getattr(config, "GRID_SIZE", 3.0))
    W = _build_queen_weights(cells_df, gs)
    cell_order = cells_df["cell_id"].tolist()
    cell_idx = {c: i for i, c in enumerate(cell_order)}
    n_cells = len(cell_order)

    rows: list[dict] = []
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
                })

    comparison_df = pd.read_csv(MODEL_COMPARISON) if MODEL_COMPARISON.exists() else pd.DataFrame()

    def _get_cell_residuals(y: np.ndarray, mu: np.ndarray, alpha: float | None,
                            cell_ids_arr: np.ndarray | None) -> np.ndarray:
        if alpha is not None and np.isfinite(alpha) and alpha > 0:
            denom = np.sqrt(np.maximum(mu + alpha * mu ** 2, 1e-12))
        else:
            denom = np.sqrt(np.maximum(mu, 1e-12))
        r = (y - mu) / denom
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
            logger.info("Moran's I for %s: I=%.4f z=%.3f p=%.4f", model_name, obs_I, z, p)
        except Exception as exc:
            logger.warning("Spatial diagnostics failed for %s: %s", m.get("col", "?"), exc)

    out_df = pd.DataFrame(rows)
    OUTPUT_MORAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_MORAN_CSV, index=False)
    logger.info("Saved Moran results to %s", OUTPUT_MORAN_CSV)

    if not out_df.empty and "moran_I" in out_df.columns:
        try:
            OUTPUT_MORAN_FIG.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["#c0392b" if p < 0.05 else "#1F4E79"
                      for p in out_df["p_perm"].fillna(1.0)]
            ax.barh(out_df["model"], out_df["moran_I"], color=colors, alpha=0.85)
            ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
            ax.set_xlabel("Moran's I (red = significant p<0.05)")
            ax.set_title("Spatial autocorrelation of Pearson residuals")
            fig.tight_layout()
            fig.savefig(OUTPUT_MORAN_FIG, dpi=250)
            plt.close(fig)
        except Exception as exc:
            logger.warning("Moran figure failed: %s", exc)

    return out_df
