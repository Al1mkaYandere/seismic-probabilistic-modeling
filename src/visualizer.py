import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MaxNLocator
from pathlib import Path

# --- Global Scientific Styling ---
def set_academic_style():
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # Times-like math font
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.bbox": "tight"
    })

def plot_mean_vs_variance(cell_stats: pd.DataFrame, save_path: Path):
    """Log-log plot proving overdispersion."""
    set_academic_style()
    filt = cell_stats[(cell_stats["mean_Y"] > 0) & (cell_stats["var_Y"] > 0)].copy()
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    # Density via point transparency
    ax.scatter(filt["mean_Y"], filt["var_Y"], alpha=0.4, s=20, 
               color="#2c3e50", edgecolor="white", linewidth=0.3, label="Grid Cells")
    
    # Poisson line (y = x)
    lims = [filt["mean_Y"].min(), filt["mean_Y"].max()]
    ax.plot(lims, lims, color="#e74c3c", linestyle="--", lw=1.5, label=r"Poisson ($\sigma^2 = \mu$)")
    
    # Illustrative NB curve (y = x + alpha*x^2)
    x_range = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 100)
    ax.plot(x_range, x_range + 0.5 * x_range**2, color="#3498db", linestyle=":", lw=1.2, label=r"NB Trend")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    
    ax.set_xlabel(r"Empirical Mean $\mathbb{E}[Y]$")
    ax.set_ylabel(r"Empirical Variance $\mathrm{Var}(Y)$")
    ax.set_title("Mean-Variance Relationship", pad=10)
    ax.legend(frameon=False, loc="upper left")
    
    fig.savefig(save_path)
    plt.close()

def plot_forecast_comparison(test_df: pd.DataFrame, save_path: Path):
    """Step-plot for earthquake counts: Observed vs NB vs DL."""
    set_academic_style()
    # Use the most active cell for the figure
    cell_id = test_df.groupby("cell_id")["Y"].sum().idxmax()
    sub = test_df[test_df["cell_id"] == cell_id].sort_values("week").tail(52)  # Last year

    fig, ax = plt.subplots(figsize=(9, 4))
    
    # Observed as step plot
    ax.step(sub["week"], sub["Y"], where="post", color="black", lw=1.2, label="Observed", zorder=3)
    ax.fill_between(sub["week"], sub["Y"], step="post", color="black", alpha=0.05)
    
    # Model predictions
    if "pred_nb" in sub.columns:
        ax.plot(sub["week"], sub["pred_nb"], color="#e67e22", lw=1.5, alpha=0.8, label="NB Baseline")
    
    if "pred_dl" in sub.columns:
        ax.plot(sub["week"], sub["pred_dl"], color="#2980b9", lw=2, label=r"$\mathbf{Hybrid\ DL\ (Ours)}$")

    ax.set_ylabel(r"Weekly Event Count $Y$")
    ax.set_xlabel("Test Period (Weekly)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.set_title(f"Forecast Comparison: Cell {cell_id}")
    
    fig.savefig(save_path)
    plt.close()

def plot_walk_forward_stability(results_df: pd.DataFrame, save_path: Path):
    """Grouped bar chart for year-by-year error."""
    set_academic_style()
    plot_df = results_df.copy()
    
    # Normalize model display names
    model_map = {"NB_Enhanced": "NB Baseline", "Hybrid_DL_Enhanced": "Hybrid DL"}
    plot_df["Model"] = plot_df["Model"].map(model_map).fillna(plot_df["Model"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=plot_df, x="Year", y="Mean_Poisson_Deviance", hue="Model", 
                palette=["#bdc3c7", "#2c3e50"], ax=ax)
    
    ax.set_ylabel("Mean Poisson Deviance (Lower is better)")
    ax.set_xlabel("Test Year")
    ax.set_title("Walk-Forward Model Stability")
    
    sns.despine(left=True)
    ax.yaxis.grid(True)
    ax.legend(frameon=False, loc="upper right")
    
    fig.savefig(save_path)
    plt.close()

def plot_target_distribution(df: pd.DataFrame, save_path: Path):
    """Log-scale histogram of Y with integer bins."""
    set_academic_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Integer-aligned bins
    max_y = int(df["Y"].max())
    bins = np.arange(0, max_y + 2) - 0.5
    
    sns.histplot(df["Y"], bins=bins, color="#34495e", alpha=0.8, ax=ax)
    
    ax.set_yscale("log")
    ax.set_xticks(np.arange(0, max_y + 1, 5 if max_y > 10 else 1))
    
    ax.set_xlabel(r"Weekly Earthquake Count $Y$")
    ax.set_ylabel("Frequency (Log Scale)")
    ax.set_title("Target Variable Distribution")
    
    # Annotate zero-inflation
    zeros_pct = (df["Y"] == 0).mean() * 100
    ax.annotate(f"Zero-inflation: {zeros_pct:.1f}%", xy=(0, ax.get_ylim()[1]/2), 
                xytext=(3, ax.get_ylim()[1]*0.8),
                arrowprops=dict(arrowstyle="->", color="gray"))

    fig.savefig(save_path)
    plt.close()

def plot_dispersion_histogram(cell_stats: pd.DataFrame, save_path: Path):
    """Histogram of D with overdispersion threshold."""
    set_academic_style()
    d_vals = cell_stats["D_cell"].dropna()
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.histplot(d_vals, bins=35, kde=True, color="#95a5a6", ax=ax)
    
    ax.axvline(1.0, color="#c0392b", linestyle="--", lw=2, label="Poisson Threshold ($D=1$)")
    
    # Shade overdispersed region
    ax.axvspan(1.0, d_vals.max(), color="#c0392b", alpha=0.05)
    
    ax.set_xlabel(r"Dispersion Index $D = \mathrm{Var}(Y) / \mathbb{E}[Y]$")
    ax.set_ylabel("Cell Count")
    ax.set_title("Spatiotemporal Dispersion Diagnostics")
    ax.legend(frameon=False)
    
    fig.savefig(save_path)
    plt.close()
