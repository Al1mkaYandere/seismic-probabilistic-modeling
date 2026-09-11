"""Central configuration: API parameters, paths, and filenames."""

import os
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    """Read an override from the environment, falling back to the default.

    Three settings have to be overridable from outside the process rather than
    by assigning to this module after import: several modules bind their own
    constants from these values AT IMPORT TIME (spatial_diagnostics.PROCESSED_DATA,
    probabilistic_evaluation.OUTPUT_*, tail_metrics.PREDICTION_STORE), so a later
    assignment would move only part of the pipeline and leave the rest writing
    into the published outputs. The environment is read before any of that runs.

    Unset variables mean exactly the published behaviour.
    """
    value = os.environ.get(name)
    return default if value is None or value == "" else value

BASE_URL: str = "https://earthquake.usgs.gov/fdsnws/event/1/query"

START_DATE: str = "2010-01-01"
END_DATE: str = "2024-01-01"
MIN_MAGNITUDE: float = 3.0
"""Lower magnitude bound of the USGS query — what was downloaded."""

M_C: float = float(_env_str("SPM_M_C", "3.0"))
"""Completeness threshold used for MODELLING — above what magnitude the
catalogue is treated as complete.

This is a different quantity from MIN_MAGNITUDE, which only says what was
requested from the archive. They coincide today, and that is a choice rather
than a fact: mc_estimation.py estimates the real threshold at 4.5 (see
outputs/mc_estimate.csv), and the magnitude histogram rises up to 4.3 before it
falls, which under Gutenberg-Richter can only mean the smaller events were never
recorded. Modelling from 3.0 therefore counts 55% of the catalogue as data when
it is really absence of detection.

Raising this is the experiment that decides whether the tail claim is about
seismicity or about the catalogue. The default stays 3.0 so that the published
numbers are reproducible; set SPM_M_C=4.5 (together with SPM_PANEL_FILE and
SPM_OUTPUT_DIR, so the run cannot land on top of the published files) to run the
experiment. Nothing else in the code hard-codes a threshold."""

BBOX: dict[str, float] = {
    "minlatitude": 38.0,
    "maxlatitude": 45.0,
    "minlongitude": 65.0,
    "maxlongitude": 85.0,
}

GRID_SIZE: float = 3.0

RAW_DATA_FILE: str = "usgs_central_asia_raw.csv"
PROCESSED_DATA_FILE: str = _env_str("SPM_PANEL_FILE", "spatiotemporal_grid.csv")
"""Panel filename. A run at another completeness threshold writes a SEPARATE
file next to the published one - the published panel is never overwritten."""

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH: Path = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR: Path = Path(_env_str("SPM_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
"""Where every result file goes. Overridden so a parallel run at another
threshold cannot overwrite the published numbers."""
FIGURES_DIR: Path = OUTPUT_DIR / "figures"

MODEL_COMPARISON_CSV: Path = OUTPUT_DIR / "model_comparison.csv"
TEST_PREDICTIONS_CSV: Path = OUTPUT_DIR / "test_predictions.csv"
ETAS_TEST_PREDICTIONS_CSV: Path = OUTPUT_DIR / "etas_test_predictions.csv"
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
