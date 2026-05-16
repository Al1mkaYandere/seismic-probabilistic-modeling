"""Tail-conditional evaluation of all models on static 80/20 test split.

Strata:
    Q1 / Q2 / Q3 / Q4 - quartiles of Y_test
    Y>=5              - extreme-event stratum

Metrics per stratum:
    MAE, RMSE, MPD, NLL (Poisson or NB closed-form), CRPS (discrete approximation)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import nbinom, poisson
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

from src import config

logger = logging.getLogger(__name__)

OUTPUT_TAIL_CSV = config.TAIL_EVALUATION_CSV
OUTPUT_TAIL_FIG = config.TAIL_METRICS_FIG

PREDICTION_STORE = config.TEST_PREDICTIONS_CSV
CALIBRATION_PREDS = config.CALIBRATION_PREDICTIONS_CSV
MODEL_COMPARISON = config.MODEL_COMPARISON_CSV


def _poisson_nll(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, 1e-9, None)
    return float(np.mean(mu - y * np.log(mu)))


def _nb_nll(y: np.ndarray, mu: np.ndarray, alpha: float | np.ndarray) -> float:
    from scipy.special import gammaln
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), 1e-9, None)
    mu = np.clip(mu, 1e-9, None)
    r = 1.0 / alpha
    term = (
        -gammaln(y + r)
        + gammaln(y + 1.0)
        + gammaln(r)
        - r * np.log(r / (r + mu))
        - y * np.log(mu / (r + mu))
    )
    return float(np.mean(term))


def _discrete_crps(y: np.ndarray, mu: np.ndarray, dist: str = "poisson",
                   alpha: float | np.ndarray | None = None) -> float:
    """Approximate CRPS for discrete distributions (Czado et al. 2009).

    CRPS = sum_{k=0}^{K_max} (F(k) - 1_{y<=k})^2
    """
    K_max = int(np.percentile(y, 99)) + 20
    crps_vals = np.zeros(len(y), dtype=np.float64)
    mu = np.clip(mu, 1e-9, None)
    for k in range(K_max + 1):
        if dist == "poisson":
            F_k = poisson.cdf(k, mu=mu)
        else:
            alpha_arr = np.clip(np.asarray(alpha, dtype=np.float64), 1e-9, None)
            r = 1.0 / alpha_arr
            p_nb = 1.0 / (1.0 + alpha_arr * mu)
            F_k = nbinom.cdf(k, n=r, p=p_nb)
        indicator = (y <= k).astype(np.float64)
        crps_vals += (F_k - indicator) ** 2
    return float(np.mean(crps_vals))


def _metrics_stratum(y: np.ndarray, mu: np.ndarray, dist: str = "poisson",
                     alpha: float | np.ndarray | None = None) -> dict[str, float]:
    if len(y) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "MPD": np.nan,
                "NLL": np.nan, "CRPS": np.nan}
    mu_c = np.clip(mu, 1e-9, None)
    mae = float(mean_absolute_error(y, mu_c))
    rmse = float(np.sqrt(mean_squared_error(y, mu_c)))
    try:
        mpd = float(mean_poisson_deviance(y, mu_c))
    except Exception:
        mpd = np.nan
    if dist == "poisson":
        nll = _poisson_nll(y, mu_c)
        crps = _discrete_crps(y, mu_c, dist="poisson")
    else:
        a = alpha if alpha is not None else 1.0
        nll = _nb_nll(y, mu_c, a)
        crps = _discrete_crps(y, mu_c, dist="nb", alpha=a)
    return {"n": int(len(y)), "MAE": mae, "RMSE": rmse, "MPD": mpd,
            "NLL": nll, "CRPS": crps}


def _strata_masks(y: np.ndarray) -> dict[str, np.ndarray]:
    q25, q50, q75 = float(np.percentile(y, 25)), float(np.percentile(y, 50)), float(np.percentile(y, 75))
    return {
        "Q1 (low)": y <= q25,
        "Q2": (y > q25) & (y <= q50),
        "Q3": (y > q50) & (y <= q75),
        "Q4 (high)": y > q75,
        "Y>=5": y >= 5,
        "All": np.ones(len(y), dtype=bool),
    }


def run_tail_metrics() -> pd.DataFrame:
    """Compute tail-conditional evaluation table and save artefacts."""
    if not PREDICTION_STORE.exists():
        logger.warning("Tail metrics skipped: %s not found", PREDICTION_STORE)
        return pd.DataFrame()

    preds_df = pd.read_csv(PREDICTION_STORE)
    calib_df = pd.read_csv(CALIBRATION_PREDS) if CALIBRATION_PREDS.exists() else pd.DataFrame()
    comparison_df = pd.read_csv(MODEL_COMPARISON) if MODEL_COMPARISON.exists() else pd.DataFrame()

    required_cols = {"y_true"}
    if not required_cols.issubset(preds_df.columns):
        logger.warning("Tail metrics: missing y_true in %s", PREDICTION_STORE)
        return pd.DataFrame()

    y_true = preds_df["y_true"].to_numpy(dtype=np.float64)
    masks = _strata_masks(y_true)

    all_rows: list[dict] = []
    model_configs: list[dict] = []

    # Build list of available model predictions
    NON_PRED_COLS = {"y_true", "cell_id", "week", "Y", "cell_id_idx"}
    for col in preds_df.columns:
        if col in NON_PRED_COLS:
            continue
        model_name = col.replace("pred_", "").replace("_", " ").title().replace(" ", "_")
        dist = "poisson"
        alpha = None
        if "nb" in col.lower() or "hybrid" in col.lower():
            dist = "nb"
            alpha_row = comparison_df[comparison_df["model"].str.lower().str.contains("nb_enhanced")] if not comparison_df.empty else pd.DataFrame()
            if not alpha_row.empty and "alpha_hat" in alpha_row.columns:
                alpha = float(alpha_row["alpha_hat"].iloc[0])
            if alpha is None or not np.isfinite(alpha):
                alpha = 1.0
        model_configs.append({"col": col, "model_name": col, "dist": dist, "alpha": alpha})

    # Add DL calibration models from calibration_predictions.csv
    if not calib_df.empty and "model" in calib_df.columns:
        for model_name, sub in calib_df.groupby("model"):
            if not ("y_true" in sub.columns and "mu_pred" in sub.columns):
                continue
            y_c = sub["y_true"].to_numpy(dtype=np.float64)
            mu_c = sub["mu_pred"].to_numpy(dtype=np.float64)
            alpha_c = sub["alpha_pred"].to_numpy(dtype=np.float64) if "alpha_pred" in sub.columns else None
            dist_c = "nb" if "Hybrid" in str(model_name) else "poisson"
            masks_c = _strata_masks(y_c)
            for stratum, mask in masks_c.items():
                if mask.sum() == 0:
                    continue
                m = _metrics_stratum(
                    y_c[mask], mu_c[mask], dist=dist_c,
                    alpha=alpha_c[mask] if alpha_c is not None else None,
                )
                all_rows.append({"model": model_name, "stratum": stratum, **m})

    for cfg in model_configs:
        mu = preds_df[cfg["col"]].to_numpy(dtype=np.float64)
        for stratum, mask in masks.items():
            if mask.sum() == 0:
                continue
            m = _metrics_stratum(y_true[mask], mu[mask], dist=cfg["dist"], alpha=cfg["alpha"])
            all_rows.append({"model": cfg["col"], "stratum": stratum, **m})

    out_df = pd.DataFrame(all_rows)
    OUTPUT_TAIL_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_TAIL_CSV, index=False)
    logger.info("Saved tail evaluation to %s", OUTPUT_TAIL_CSV)

    # Figure: MPD by model × stratum
    if not out_df.empty and "MPD" in out_df.columns:
        try:
            plot_df = out_df[out_df["stratum"].isin(["Q1 (low)", "Q2", "Q3", "Q4 (high)", "Y>=5"])].copy()
            plot_df = plot_df.dropna(subset=["MPD"])
            if not plot_df.empty:
                OUTPUT_TAIL_FIG.parent.mkdir(parents=True, exist_ok=True)
                strata_u = [s for s in ["Q1 (low)", "Q2", "Q3", "Q4 (high)", "Y>=5"]
                            if s in plot_df["stratum"].values]
                models_u = plot_df["model"].unique()
                x = np.arange(len(strata_u))
                width = min(0.8 / max(len(models_u), 1), 0.25)
                palette = sns.color_palette("Blues_d", max(len(models_u), 1))
                fig, ax = plt.subplots(figsize=(13, 6))
                for i, model in enumerate(models_u):
                    sub = plot_df[plot_df["model"] == model]
                    mpd_vals = [
                        float(sub[sub["stratum"] == s]["MPD"].iloc[0])
                        if len(sub[sub["stratum"] == s]) > 0 else np.nan
                        for s in strata_u
                    ]
                    offset = (i - len(models_u) / 2.0 + 0.5) * width
                    ax.bar(x + offset, mpd_vals, width,
                           label=model, color=palette[i], alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(strata_u, rotation=20)
                ax.set_ylabel("Mean Poisson Deviance")
                ax.set_title("Tail-conditional MPD by model and stratum")
                ax.legend(fontsize=8, ncol=2)
                fig.tight_layout()
                fig.savefig(OUTPUT_TAIL_FIG, dpi=250)
                plt.close(fig)
        except Exception as exc:
            logger.warning("Tail metrics figure failed: %s", exc)

    return out_df
