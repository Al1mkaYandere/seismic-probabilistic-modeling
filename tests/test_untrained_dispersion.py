"""A network must not report a dispersion it never fitted.

``PoissonLoss`` takes an alpha argument and ignores it, so the alpha head of a
Poisson-trained network gets no gradient: its output is a random projection of the
trained layers. It used to be written into ``calibration_predictions.csv`` next to
the real one, and ``spatial_diagnostics`` averaged that column and scored the
Poisson network's residuals against a negative binomial variance built from it.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src import dl_modeling, predictive_family, spatial_diagnostics
from src.dl_modeling import FEATURES_ENHANCED

TEST_WEEKS = 160


def _short_panel() -> pd.DataFrame:
    full = dl_modeling._load_df()
    weeks = pd.Index(sorted(full["week"].unique()))[-TEST_WEEKS:]
    return full[full["week"].isin(weeks)].reset_index(drop=True)


def test_the_poisson_network_reports_no_dispersion_and_the_nb_one_does(
    tmp_path, monkeypatch
) -> None:
    # The NB variant saves an alpha audit to disk; send it to tmp_path rather than
    # into the published output directory.
    monkeypatch.setattr(dl_modeling, "OUTPUT_ALPHA_SUMMARY", tmp_path / "alpha.csv")
    monkeypatch.setattr(dl_modeling, "OUTPUT_ALPHA_HIST", tmp_path / "alpha.png")
    panel = _short_panel()

    _, poisson_calib = dl_modeling._train_one_variant(
        panel, FEATURES_ENHANCED, "Neural_Poisson_Enhanced", loss_kind="poisson"
    )
    _, nb_calib = dl_modeling._train_one_variant(
        panel, FEATURES_ENHANCED, "Hybrid_DL_Enhanced", loss_kind="nb"
    )

    assert poisson_calib is not None and nb_calib is not None
    assert poisson_calib["alpha_pred"].isna().all(), (
        "the Poisson network reported a dispersion; its alpha head never received a "
        "gradient, so whatever it reports is a random projection"
    )
    # Without this the test would pass on code that reports nothing for anyone.
    nb_alpha = nb_calib["alpha_pred"].to_numpy(float)
    assert np.isfinite(nb_alpha).all() and (nb_alpha > 0).all(), (
        "the negative binomial network stopped reporting the dispersion it did fit"
    )


def _rows(family: str, alpha) -> pd.DataFrame:
    return pd.DataFrame({
        predictive_family.COLUMN: family,
        "alpha_pred": alpha,
    })


def test_a_missing_dispersion_scores_residuals_as_poisson() -> None:
    y = np.array([0.0, 1.0, 3.0, 7.0])
    mu = np.array([0.2, 0.9, 2.0, 5.0])
    nan_column = _rows(predictive_family.NB, np.full(4, np.nan))
    declared_poisson = _rows(predictive_family.POISSON, np.full(4, 0.8))
    declared_nb = _rows(predictive_family.NB, np.full(4, 0.8))

    poisson = spatial_diagnostics._residuals(y, mu, None)
    all_nan = spatial_diagnostics._residuals(
        y, mu, predictive_family.mean_dispersion_of(nan_column)
    )
    # The declaration decides, not the numbers: these rows carry a perfectly finite
    # 0.8 and must still be scored as Poisson because that is what they declare.
    by_declaration = spatial_diagnostics._residuals(
        y, mu, predictive_family.mean_dispersion_of(declared_poisson)
    )
    real = spatial_diagnostics._residuals(
        y, mu, predictive_family.mean_dispersion_of(declared_nb)
    )

    np.testing.assert_array_equal(all_nan, poisson)
    assert np.isfinite(all_nan).all(), "NaN dispersion leaked into the residuals"
    np.testing.assert_array_equal(by_declaration, poisson)
    # A dispersion that is really declared must still change the scaling, or the
    # fallback has swallowed the negative binomial case along with the missing one.
    assert not np.allclose(real, poisson), "a declared dispersion stopped mattering"


def test_a_partly_missing_dispersion_averages_only_what_is_there() -> None:
    mixed = _rows(predictive_family.NB, np.array([1.0, np.nan, 3.0, np.nan]))
    assert predictive_family.mean_dispersion_of(mixed) == 2.0
