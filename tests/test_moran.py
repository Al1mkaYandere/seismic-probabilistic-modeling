"""Checks for Moran's I test of spatial autocorrelation (spatial_diagnostics.py).

Tests on synthetic data check that Moran's I can actually detect a signal (and
does not detect one where there is none). ``test_run_spatial_diagnostics_...``
used to be an expected failure (xfail) documenting known issue A7: for the
neural-network models, run_spatial_diagnostics returned NaN instead of a
number, because the calibration output it reads was missing ``cell_id``. A7
was fixed in ``e3dbbb7``; the xfail marker is gone and the test now guards
against the NaN coming back.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.spatial_diagnostics import (
    _artefact_paths,
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

    B = 999
    obs_I, _z, p = _permutation_test(x, W, B=B)
    assert obs_I > 0, "a positive gradient was built in — Moran's I must be positive"
    assert p < 0.05, f"the strong built-in autocorrelation was not detected: p={p}"
    assert p >= 1.0 / (B + 1), (
        f"p={p} is below the smallest value {1.0 / (B + 1)} that B={B} permutations "
        "can support. A permutation p-value must use the (1 + r) / (B + 1) "
        "correction; the uncorrected r / B reports p = 0 here, which claims more "
        "resolution than the permutations actually provide."
    )


def test_moran_detects_strong_negative_spatial_correlation():
    """A built-in checkerboard of alternating ROWS — strong NEGATIVE correlation.

    This is the case a one-sided p-value cannot report. With
    p = mean(perm >= obs), a strongly negative observed I sits in the far LEFT
    tail of the permutation distribution, so almost every permutation exceeds
    it and p comes out close to 1 — the test then reads as "no spatial
    structure whatsoever" precisely when the structure is strongest. The
    two-sided p-value compares |obs - mean| against |perm - mean| and reports
    it correctly.

    The values are exact (+1 / -1), not noisy, and the expected sign is fixed
    by the geometry rather than by whatever the code returns: under queen
    contiguity an interior cell has 2 same-row neighbours (same sign) and up to
    6 neighbours in the rows above and below (opposite sign), so opposite-sign
    pairs dominate and Moran's I must be clearly negative.
    """
    cells_df = _make_4x4_cells()
    W = _build_queen_weights(cells_df, GRID_SIZE)

    row_idx = ((cells_df["lat_grid"] - 38.0) / GRID_SIZE).to_numpy()
    x = np.where(row_idx % 2 == 0, 1.0, -1.0)

    obs_I, _z, p = _permutation_test(x, W, B=999)

    assert obs_I < -0.3, (
        f"alternating rows must give a clearly negative Moran's I, got {obs_I}"
    )
    assert p < 0.05, (
        f"strong negative spatial autocorrelation was not detected: p={p}. "
        "A one-sided p = mean(perm >= obs) gives p close to 1 here by "
        "construction, which is exactly the defect this test guards against."
    )


def test_moran_does_not_flag_white_noise():
    """White noise with no spatial structure — the test must not find a signal."""
    cells_df = _make_4x4_cells()
    W = _build_queen_weights(cells_df, GRID_SIZE)

    rng = np.random.default_rng(123)
    _ = rng.normal(0, 0.15, size=len(cells_df))  # same call sequence as the test above
    x_white = rng.normal(0, 1.0, size=len(cells_df))

    _obs_I, _z, p = _permutation_test(x_white, W, B=999)
    assert p > 0.05, f"the test found a signal in pure noise: p={p}"


def test_artefact_paths_default_keeps_every_configured_location():
    """out_dir=None must reproduce the pipeline's paths, figure included.

    The three CSVs and the figure do NOT share a directory, so deriving one
    from the other silently relocates the figure on every real run — and the
    frozen-baseline check cannot see it, because it only compares CSVs.
    """
    residuals, weekly, influence, figure = _artefact_paths(None)

    assert residuals == config.MORAN_RESIDUALS_CSV
    assert weekly == config.MORAN_RESIDUALS_CSV.parent / "moran_weekly.csv"
    assert influence == config.MORAN_RESIDUALS_CSV.parent / "moran_influence.csv"
    assert figure == config.MORAN_RESIDUALS_FIG
    assert figure.parent != residuals.parent, (
        "this test only means anything while the figure lives in a different "
        "directory from the CSVs; if config changes, rewrite the test"
    )


def test_artefact_paths_redirect_sends_all_four_to_one_directory(tmp_path):
    paths = _artefact_paths(tmp_path)

    assert len(paths) == 4
    assert all(path.parent == tmp_path for path in paths), (
        f"not every artefact was redirected: {[str(p) for p in paths]}"
    )
    assert len({path.name for path in paths}) == 4, "artefacts would overwrite each other"


def test_run_spatial_diagnostics_gives_finite_number_for_dl_models(tmp_path):
    # out_dir keeps this call from overwriting the pipeline's own Moran outputs:
    # the test suite must not touch the artefacts the frozen baseline compares.
    out_df = run_spatial_diagnostics(out_dir=tmp_path)

    written = {path.name for path in tmp_path.iterdir()}
    for expected in ("moran_residuals.csv", "moran_weekly.csv", "moran_influence.csv"):
        assert expected in written, f"{expected} was not written into out_dir: {written}"

    for model_name in ("Hybrid_DL_Enhanced", "Neural_Poisson_Enhanced"):
        row = out_df.loc[out_df["model"] == model_name]
        assert not row.empty, f"{model_name} is missing from run_spatial_diagnostics output"
        moran_i = float(row["moran_I"].iloc[0])
        assert np.isfinite(moran_i), f"{model_name}: moran_I = {moran_i}, expected a finite number"
