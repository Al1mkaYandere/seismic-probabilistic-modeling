"""Poisson assumption diagnostics: dispersion indices, logging, and figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from src import config

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

FIGURES_DIR: Path = config.FIGURES_DIR

FIG_HIST_Y: Path = FIGURES_DIR / "hist_Y_logscale.png"
FIG_MEAN_VAR: Path = FIGURES_DIR / "mean_vs_variance.png"
FIG_DISP: Path = FIGURES_DIR / "dispersion_histogram.png"
FIG_HIST_Y_BEAUTIFUL: Path = FIGURES_DIR / "beautiful_hist_Y_logscale.png"
FIG_MEAN_VAR_BEAUTIFUL: Path = FIGURES_DIR / "beautiful_mean_vs_variance.png"
FIG_DISP_BEAUTIFUL: Path = FIGURES_DIR / "beautiful_dispersion_histogram.png"
CELL_STATS_CSV: Path = config.POISSON_CELL_STATS_CSV


def _ensure_output_directories() -> None:
    """Create ``outputs/`` and ``outputs/figures/`` if missing."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _validate_input(df: pd.DataFrame) -> None:
    """Assert non-empty panel, required column, and non-negative counts."""
    assert not df.empty, "Dataset must not be empty"
    assert "Y" in df.columns, "Column 'Y' is required"
    assert (df["Y"] >= 0).all(), "All Y must be >= 0"


def _compute_per_cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cell mean, sample variance, and dispersion index D = var/mean.

    Returns
    -------
    pd.DataFrame
        Columns: cell_id, mean_Y, var_Y, D_cell.
    """
    g = df.groupby("cell_id", sort=True)["Y"]
    stats = g.agg(mean_Y="mean", var_Y=lambda s: s.var(ddof=1)).reset_index()
    stats["mean_Y"] = stats["mean_Y"].astype(np.float64)
    stats["var_Y"] = stats["var_Y"].astype(np.float64)
    stats["D_cell"] = stats["var_Y"] / stats["mean_Y"]
    stats.loc[stats["mean_Y"] == 0, "D_cell"] = np.nan
    stats["D_cell"] = stats["D_cell"].replace([np.inf, -np.inf], np.nan)
    stats["D_cell"] = stats["D_cell"].astype(np.float64)
    return stats


def _compute_global_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Global mean, sample variance, and dispersion index.

    Returns
    -------
    tuple[float, float, float]
        global_mean, global_var, global_D (NaN if mean is 0).
    """
    y = df["Y"].astype(np.float64)
    global_mean = float(y.mean())
    global_var = float(y.var(ddof=1))
    if global_mean == 0:
        logger.warning("global_mean is 0; global_D set to NaN")
        global_D = np.nan
    else:
        global_D = global_var / global_mean
    return global_mean, global_var, global_D


def _log_summary_statistics(global_mean: float, global_var: float, global_D: float, cell_stats: pd.DataFrame) -> None:
    """Log global metrics, per-cell D summaries, and interpretation."""
    logger.info("Global Mean: %s", global_mean)
    logger.info("Global Variance: %s", global_var)
    logger.info("Global D (dispersion index): %s", global_D)

    d = cell_stats["D_cell"].to_numpy(dtype=float)
    mean_d = float(np.nanmean(d))
    median_d = float(np.nanmedian(d))
    logger.info("Mean of D_cell (NaN ignored): %s", mean_d)
    logger.info("Median of D_cell (NaN ignored): %s", median_d)

    finite = np.isfinite(d)
    n_finite = int(finite.sum())
    if n_finite > 0:
        pct_gt_1 = float((d[finite] > 1.0).sum() / n_finite * 100.0)
        pct_gt_1_5 = float((d[finite] > 1.5).sum() / n_finite * 100.0)
    else:
        pct_gt_1 = float("nan")
        pct_gt_1_5 = float("nan")
    logger.info("Percent of cells where D_cell > 1: %s", pct_gt_1)
    logger.info("Percent of cells where D_cell > 1.5: %s", pct_gt_1_5)

    if np.isfinite(global_D):
        if global_D <= 1.2:
            logger.info("Poisson assumption plausible")
        elif global_D <= 2.0:
            logger.warning("Moderate overdispersion detected")
        else:
            logger.warning(
                "Strong overdispersion. Poisson likely invalid, consider Negative Binomial models"
            )


def _plot_hist_y_logscale(df: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"]})
    
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    
    y_vals = df["Y"].astype(float).to_numpy()
    y_max = int(np.nanmax(y_vals)) if y_vals.size else 0
    bins = np.arange(-0.5, y_max + 1.5, 1.0)
    
    sns.histplot(
        x=y_vals,
        bins=bins,
        kde=False,
        stat="count",
        color="#4e79a7",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )
    
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8, top=ax.get_ylim()[1] * 2) 
    
    ax.set_title(r"Distribution of Cell-Week Earthquake Counts ($M \geq 3.0$)", pad=15)
    ax.set_xlabel(r"Weekly count $Y$")
    ax.set_ylabel("Frequency (log scale)")
    
    tick_step = 5 if y_max <= 30 else 10
    ax.set_xticks(np.arange(0, y_max + 1, tick_step))
    ax.set_xlim(-0.7, y_max + 0.7)
    
    ax.yaxis.grid(True, which="major", linestyle="--", alpha=0.3)
    ax.xaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(FIG_HIST_Y_BEAUTIFUL, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_vs_variance(cell_stats: pd.DataFrame) -> None:
    """Log-log scatter of mean vs variance with y = x reference line."""
    filt = cell_stats[(cell_stats["mean_Y"] > 0) & (cell_stats["var_Y"] > 0)].copy()
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    if not filt.empty:
        x = filt["mean_Y"].astype(float)
        y = filt["var_Y"].astype(float)
        sns.scatterplot(
            x=x,
            y=y,
            s=42,
            alpha=0.6,
            color=sns.color_palette("colorblind")[0],
            edgecolor=None,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        xmin, xmax = float(x.min()), float(x.max())
        ax.plot(
            [xmin, xmax],
            [xmin, xmax],
            linestyle=":",
            color="#d62728",
            linewidth=2.0,
            label=r"Poisson boundary: $y=x$",
        )
        d_overall = float(np.nanmean(filt["var_Y"] / filt["mean_Y"]))
        ax.text(
            0.03,
            0.94,
            rf"$D \approx {d_overall:.2f}$",
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#aaaaaa", "alpha": 0.9},
        )
        ax.legend(loc="upper left", frameon=True)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.set_xlabel(r"$\mathbb{E}[Y]$")
    ax.set_ylabel(r"$\mathrm{Var}(Y)$")
    ax.set_title(r"Mean-Variance Relationship Across Cells (log-log)")
    ax.grid(True, which="major", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_MEAN_VAR_BEAUTIFUL, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_dispersion_histogram(cell_stats: pd.DataFrame) -> None:
    """Histogram of per-cell dispersion index D."""
    d_vals = cell_stats["D_cell"].dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    if d_vals.size > 0:
        sns.histplot(
            d_vals,
            bins=30,
            kde=True,
            color=sns.color_palette("deep")[0],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            ax=ax,
        )
    ax.axvline(1.0, color="#d62728", linestyle="--", linewidth=2.0)
    ax.annotate(
        "Overdispersion",
        xy=(1.35, ax.get_ylim()[1] * 0.55 if ax.get_ylim()[1] > 0 else 1.0),
        xytext=(1.9, ax.get_ylim()[1] * 0.8 if ax.get_ylim()[1] > 0 else 1.5),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.2},
        color="#d62728",
        fontsize=10,
    )
    ax.set_title(r"Distribution of Dispersion Index $D=\mathrm{Var}(Y)/\mathbb{E}[Y]$")
    ax.set_xlabel(r"Dispersion index $D$")
    ax.set_ylabel("Frequency")
    ax.grid(True, which="major", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DISP_BEAUTIFUL, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_cell_stats_table(cell_stats: pd.DataFrame) -> None:
    """Write per-cell statistics to CSV."""
    out_path = Path(CELL_STATS_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cell_stats[["cell_id", "mean_Y", "var_Y", "D_cell"]].to_csv(out_path, index=False)
    logger.info("Saved per-cell statistics to %s", out_path.resolve())


def run_poisson_analysis(df: pd.DataFrame) -> None:
    """
    Validate counts, compute global and per-cell dispersion, log, plot, and save tables.

    Parameters
    ----------
    df : pd.DataFrame
        Processed spatiotemporal grid with column ``Y`` (non-negative counts).
    """
    _ensure_output_directories()
    _validate_input(df)

    cell_stats = _compute_per_cell_stats(df)
    global_mean, global_var, global_D = _compute_global_stats(df)

    _log_summary_statistics(global_mean, global_var, global_D, cell_stats)

    _plot_hist_y_logscale(df)
    _plot_mean_vs_variance(cell_stats)
    _plot_dispersion_histogram(cell_stats)

    _save_cell_stats_table(cell_stats)
