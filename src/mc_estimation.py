"""Magnitude of completeness (M_c) estimation via Maximum Curvature method.

Reference: Wiemer & Wyss (2000), BSSA.
Method: M_c = bin with maximum frequency in frequency-magnitude distribution + 0.2 correction.
b-value: MLE Aki estimator.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

OUTPUT_MC_CSV = config.MC_ESTIMATE_CSV
OUTPUT_MC_FIG = config.MC_FREQUENCY_MAGNITUDE_FIG

BIN_WIDTH: float = 0.1


def _b_value_aki(magnitudes: np.ndarray, m_c: float) -> float:
    """MLE Aki (1965) b-value estimator."""
    mags = magnitudes[magnitudes >= m_c]
    if len(mags) < 10:
        return np.nan
    return float(1.0 / (np.log(10.0) * (np.mean(mags) - m_c + BIN_WIDTH / 2.0)))


def estimate_mc(magnitudes: np.ndarray) -> dict[str, float]:
    """Apply Maximum Curvature method and return M_c, b-value, n_used."""
    if len(magnitudes) == 0:
        return {"M_c_estimate": np.nan, "b_value": np.nan, "n_used": 0}
    m_min = float(np.floor(magnitudes.min() / BIN_WIDTH) * BIN_WIDTH)
    m_max = float(np.ceil(magnitudes.max() / BIN_WIDTH) * BIN_WIDTH)
    bins = np.arange(m_min, m_max + BIN_WIDTH, BIN_WIDTH)
    counts, edges = np.histogram(magnitudes, bins=bins)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    # Maximum curvature: bin with highest frequency
    max_idx = int(np.argmax(counts))
    m_c = float(bin_centers[max_idx]) + 0.2  # Wiemer & Wyss correction
    m_c = round(m_c / BIN_WIDTH) * BIN_WIDTH  # round to bin width
    b_val = _b_value_aki(magnitudes, m_c)
    n_used = int(np.sum(magnitudes >= m_c))
    return {"M_c_estimate": m_c, "b_value": b_val, "n_used": n_used,
            "bin_centers": bin_centers, "counts": counts}


def run_mc_estimation() -> pd.DataFrame:
    """Load raw events, estimate M_c, save CSV and figure."""
    raw_path = config.RAW_DATA_PATH / config.RAW_DATA_FILE
    if not raw_path.exists():
        logger.warning("M_c estimation skipped: raw data not found at %s", raw_path)
        return pd.DataFrame()

    df = pd.read_csv(raw_path)
    if "mag" not in df.columns:
        logger.warning("M_c estimation skipped: 'mag' column missing")
        return pd.DataFrame()

    mags = df["mag"].dropna().to_numpy(dtype=np.float64)
    result = estimate_mc(mags)
    m_c = result["M_c_estimate"]
    b_val = result["b_value"]
    n_used = result["n_used"]

    summary = pd.DataFrame([{
        "M_c_estimate": m_c, "b_value": b_val, "n_used": n_used,
        "n_total": len(mags), "method": "MaxCurvature+0.2 (Wiemer&Wyss 2000)",
    }])
    OUTPUT_MC_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_MC_CSV, index=False)
    logger.info("M_c estimate: %.2f (b=%.3f, n=%d)", m_c, b_val if np.isfinite(b_val) else 0, n_used)

    # Figure: frequency-magnitude distribution
    try:
        bin_centers = result["bin_centers"]
        counts = result["counts"]
        cumulative = np.array([int(np.sum(mags >= bc - BIN_WIDTH / 2)) for bc in bin_centers])
        OUTPUT_MC_FIG.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(bin_centers, counts, width=BIN_WIDTH * 0.9, color="#1F4E79",
               alpha=0.7, label="Incremental count")
        ax2 = ax.twinx()
        ax2.semilogy(bin_centers, np.maximum(cumulative, 1), "r-o",
                     markersize=4, linewidth=1.5, label="Cumulative (log)")
        ax.axvline(m_c, color="#c0392b", linestyle="--", linewidth=2.0,
                   label=f"$M_c = {m_c:.1f}$")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Frequency (incremental)")
        ax2.set_ylabel("Cumulative count (log)")
        ax.set_title(f"Frequency-Magnitude Distribution (b = {b_val:.2f})")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        fig.savefig(OUTPUT_MC_FIG, dpi=250)
        plt.close(fig)
    except Exception as exc:
        logger.warning("M_c figure failed: %s", exc)

    return summary
