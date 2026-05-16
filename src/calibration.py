"""Probabilistic calibration diagnostics for count models."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from src import config

logger = logging.getLogger(__name__)

CALIBRATION_PREDS: Path = config.CALIBRATION_PREDICTIONS_CSV
OUTPUT_PIT_HIST: Path = config.PIT_HISTOGRAM_FIG
OUTPUT_CALIBRATION_SUMMARY: Path = config.CALIBRATION_SUMMARY_CSV


def _randomized_pit_discrete(y: np.ndarray, cdf_y: np.ndarray, cdf_prev: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(42)
    u = rng.uniform(size=len(y))
    pit = cdf_prev + u * (cdf_y - cdf_prev)
    return np.clip(pit, 0.0, 1.0)


def _compute_nb_pit(y: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    eps = 1e-8
    y_i = np.asarray(y, dtype=np.int64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), eps, None)
    alpha = np.clip(np.asarray(alpha, dtype=np.float64), eps, None)
    r = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * mu)
    cdf_y = nbinom.cdf(y_i, n=r, p=p)
    cdf_prev = nbinom.cdf(np.maximum(y_i - 1, 0), n=r, p=p)
    cdf_prev[y_i == 0] = 0.0
    return _randomized_pit_discrete(y_i, cdf_y, cdf_prev)


def _compute_poisson_pit(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    eps = 1e-8
    y_i = np.asarray(y, dtype=np.int64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), eps, None)
    cdf_y = poisson.cdf(y_i, mu=mu)
    cdf_prev = poisson.cdf(np.maximum(y_i - 1, 0), mu=mu)
    cdf_prev[y_i == 0] = 0.0
    return _randomized_pit_discrete(y_i, cdf_y, cdf_prev)


def _histogram_deviation_from_uniform(pit: np.ndarray, bins: int = 10) -> float:
    hist, _ = np.histogram(pit, bins=bins, range=(0.0, 1.0), density=False)
    if hist.sum() == 0:
        return float("nan")
    freq = hist / hist.sum()
    return float(np.mean(np.abs(freq - 1.0 / bins)))


def run_calibration() -> pd.DataFrame:
    """Compute PIT diagnostics and save histogram + summary CSV."""
    if not CALIBRATION_PREDS.exists():
        logger.warning("Calibration skipped: %s not found", CALIBRATION_PREDS)
        return pd.DataFrame(columns=["model", "n", "pit_mean", "pit_var", "pit_l1_uniform"])

    df = pd.read_csv(CALIBRATION_PREDS)
    required = {"model", "y_true", "mu_pred", "alpha_pred"}
    if not required.issubset(df.columns):
        logger.warning("Calibration skipped: missing required columns in %s", CALIBRATION_PREDS)
        return pd.DataFrame(columns=["model", "n", "pit_mean", "pit_var", "pit_l1_uniform"])

    rows: list[dict[str, float | str | int]] = []
    pits: dict[str, np.ndarray] = {}
    for model_name, sub in df.groupby("model", sort=True):
        y = sub["y_true"].to_numpy(dtype=np.int64)
        mu = sub["mu_pred"].to_numpy(dtype=np.float64)
        alpha = sub["alpha_pred"].to_numpy(dtype=np.float64)
        if model_name.startswith("Hybrid_DL"):
            pit = _compute_nb_pit(y=y, mu=mu, alpha=alpha)
        else:
            pit = _compute_poisson_pit(y=y, mu=mu)
        pits[model_name] = pit
        rows.append(
            {
                "model": model_name,
                "n": int(len(pit)),
                "pit_mean": float(np.mean(pit)),
                "pit_var": float(np.var(pit, ddof=1)) if len(pit) > 1 else np.nan,
                "pit_l1_uniform": _histogram_deviation_from_uniform(pit),
            }
        )

    summary = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    OUTPUT_CALIBRATION_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CALIBRATION_SUMMARY, index=False)

    OUTPUT_PIT_HIST.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, max(1, len(pits)), figsize=(5 * max(1, len(pits)), 4), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (model_name, pit) in zip(axes, sorted(pits.items())):
        ax.hist(pit, bins=10, range=(0.0, 1.0), color="#1F4E79", alpha=0.85, edgecolor="white")
        ax.axhline(len(pit) / 10.0, color="#c0392b", linestyle="--", linewidth=1.2)
        ax.set_title(model_name.replace("_", " "))
        ax.set_xlabel("PIT")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(OUTPUT_PIT_HIST, dpi=250)
    plt.close(fig)

    logger.info("Saved calibration summary to %s", OUTPUT_CALIBRATION_SUMMARY.resolve())
    logger.info("Saved PIT histogram to %s", OUTPUT_PIT_HIST.resolve())
    return summary

