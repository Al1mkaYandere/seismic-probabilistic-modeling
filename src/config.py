"""Central configuration: API parameters, paths, and filenames."""

from pathlib import Path

BASE_URL: str = "https://earthquake.usgs.gov/fdsnws/event/1/query"

START_DATE: str = "2010-01-01"
END_DATE: str = "2024-01-01"
MIN_MAGNITUDE: float = 3.0

BBOX: dict[str, float] = {
    "minlatitude": 38.0,
    "maxlatitude": 45.0,
    "minlongitude": 65.0,
    "maxlongitude": 85.0,
}

GRID_SIZE: float = 3.0

RAW_DATA_FILE: str = "usgs_central_asia_raw.csv"
PROCESSED_DATA_FILE: str = "spatiotemporal_grid.csv"

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH: Path = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"

MODEL_COMPARISON_CSV: Path = OUTPUT_DIR / "model_comparison.csv"
TEST_PREDICTIONS_CSV: Path = OUTPUT_DIR / "test_predictions.csv"
CALIBRATION_PREDICTIONS_CSV: Path = OUTPUT_DIR / "calibration_predictions.csv"
OVERDISPERSION_LR_TEST_CSV: Path = OUTPUT_DIR / "overdispersion_lr_test.csv"
WALK_FORWARD_RESULTS_CSV: Path = OUTPUT_DIR / "walk_forward_results.csv"
CALIBRATION_SUMMARY_CSV: Path = OUTPUT_DIR / "calibration_summary.csv"
TAIL_EVALUATION_CSV: Path = OUTPUT_DIR / "tail_evaluation.csv"
MORAN_RESIDUALS_CSV: Path = OUTPUT_DIR / "moran_residuals.csv"
MC_ESTIMATE_CSV: Path = OUTPUT_DIR / "mc_estimate.csv"
POISSON_CELL_STATS_CSV: Path = OUTPUT_DIR / "poisson_cell_stats.csv"
ALPHA_AUDIT_SUMMARY_CSV: Path = OUTPUT_DIR / "alpha_audit_summary.csv"
ALPHA_IDENTIFIABILITY_CSV: Path = OUTPUT_DIR / "alpha_identifiability.csv"

FORECAST_COMPARISON_FIG: Path = FIGURES_DIR / "forecast_comparison.png"
FORECAST_COMPARISON_BEAUTIFUL_FIG: Path = FIGURES_DIR / "beautiful_forecast_comparison.png"
WALK_FORWARD_STABILITY_FIG: Path = FIGURES_DIR / "beautiful_walk_forward_stability.png"
PIT_HISTOGRAM_FIG: Path = FIGURES_DIR / "pit_histogram.png"
TAIL_METRICS_FIG: Path = FIGURES_DIR / "tail_metrics_grouped.png"
MORAN_RESIDUALS_FIG: Path = FIGURES_DIR / "moran_residuals_bar.png"
MC_FREQUENCY_MAGNITUDE_FIG: Path = FIGURES_DIR / "mc_frequency_magnitude.png"
ALPHA_DISTRIBUTION_FIG: Path = FIGURES_DIR / "alpha_distribution.png"
ALPHA_IDENTIFIABILITY_FIG: Path = FIGURES_DIR / "alpha_identifiability.png"

REQUEST_TIMEOUT_SECONDS: int = 30
