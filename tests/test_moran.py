"""Checks for Moran's I test of spatial autocorrelation (spatial_diagnostics.py).

Two tests on synthetic data check that the test can actually detect a signal
(and does not detect one where there is none). The third test is an expected
failure (xfail); it documents known issues A7/A8: for the neural-network
models, run_spatial_diagnostics currently returns NaN instead of a number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.spatial_diagnostics import (
    _build_queen_weights,
    _permutation_test,
    run_spatial_diagnostics,
)

GRID_SIZE = 3.0


def _make_4x4_cells() -> pd.DataFrame:
    lat_vals = [38.0, 41.0, 44.0, 47.0]
    lon_vals = [65.0, 68.0, 71.0, 74.0]
    rows = [(la, lo) for la in lat_vals for lo in lon_vals]
    return pd.DataFrame({"lat_grid": [r[0] for r in rows], "lon_grid": [r[1] for r in rows]})


def test_moran_detects_strong_positive_spatial_correlation():
    """A built-in linear spatial gradient plus a small amount of noise.

    The seed and the noise parameters are fixed and chosen in advance with a
    known correct answer: a strong positive autocorrelation should give
    p < 0.05 (empirically, p ~ 0.000 is observed with B=999 permutations).
    """
    cells_df = _make_4x4_cells()
    W = _build_queen_weights(cells_df, GRID_SIZE)

    rng = np.random.default_rng(123)
    lat_idx = (cells_df["lat_grid"] - 38.0) / GRID_SIZE
    lon_idx = (cells_df["lon_grid"] - 65.0) / GRID_SIZE
    trend = lat_idx.to_numpy() + lon_idx.to_numpy()
    noise = rng.normal(0, 0.15, size=len(cells_df))
    x = trend + noise

    obs_I, _z, p = _permutation_test(x, W, B=999)
    assert obs_I > 0, "a positive gradient was built in — Moran's I must be positive"
    assert p < 0.05, f"the strong built-in autocorrelation was not detected: p={p}"


def test_moran_does_not_flag_white_noise():
    """White noise with no spatial structure — the test must not find a signal."""
    cells_df = _make_4x4_cells()
    W = _build_queen_weights(cells_df, GRID_SIZE)

    rng = np.random.default_rng(123)
    _ = rng.normal(0, 0.15, size=len(cells_df))  # same call sequence as the test above
    x_white = rng.normal(0, 1.0, size=len(cells_df))

    _obs_I, _z, p = _permutation_test(x_white, W, B=999)
    assert p > 0.05, f"the test found a signal in pure noise: p={p}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known issues A7 and A8: calibration_predictions.csv currently does "
        "not contain cell_id/week (dl_modeling.py, around lines 351-358), so "
        "run_spatial_diagnostics._get_cell_residuals falls into the branch "
        "that produces a constant residual vector (spatial_diagnostics.py, "
        "around line 133), and the denominator degenerates to 0 -> NaN. "
        "Verified empirically on the real outputs/: Hybrid_DL_Enhanced and "
        "Neural_Poisson_Enhanced both give NaN already."
    ),
)
def test_run_spatial_diagnostics_gives_finite_number_for_dl_models():
    out_df = run_spatial_diagnostics()
    for model_name in ("Hybrid_DL_Enhanced", "Neural_Poisson_Enhanced"):
        row = out_df.loc[out_df["model"] == model_name]
        assert not row.empty, f"{model_name} is missing from run_spatial_diagnostics output"
        moran_i = float(row["moran_I"].iloc[0])
        assert np.isfinite(moran_i), f"{model_name}: moran_I = {moran_i}, expected a finite number"
