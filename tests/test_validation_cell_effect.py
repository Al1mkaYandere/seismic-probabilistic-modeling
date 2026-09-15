"""A feature-only NB GLM has no way to represent that two cells sit at different
average rates - every cell gets the same intercept. Giving it a dummy variable
per cell must let it match per-cell levels a feature-only model cannot. This is
the control the year-by-year comparison needs: if the neural net's edge over
the linear model were really just the net's cell embedding standing in for
different cell base rates, a linear model with its own per-cell intercept
should close most of that gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.validation import _cell_dummy_columns, _fit_nb_mle_glm, mean_poisson_deviance_safe

RNG = np.random.default_rng(20260915)


def _synthetic_panel(cell_rates: dict[str, float], n_weeks: int) -> pd.DataFrame:
    """A panel where cells differ ONLY in their average rate.

    One flat feature ``x`` is carried along so the fit goes through the same
    add_constant/GLM code path as the real one, and the only thing that changes
    between the two models compared below is the cell dummy columns.
    """
    frames = []
    for cell, rate in cell_rates.items():
        # Negative binomial with dispersion 3, mean `rate`: p = r / (r + mean).
        y = RNG.negative_binomial(3, 3.0 / (3.0 + rate), size=n_weeks).astype(float)
        frames.append(pd.DataFrame({
            "cell_id": cell, "y": y, "x": RNG.normal(size=n_weeks),
        }))
    return pd.concat(frames, ignore_index=True)


def test_cell_dummies_recover_the_known_rate_ratios() -> None:
    """Cell names sort in rate order, so the first (alphabetically) is the
    reference level and the fitted dummy coefficients are log rate ratios
    against it - the standard interpretation of a log-link NB GLM.
    """
    panel = _synthetic_panel({"cell_1_low": 1.0, "cell_2_mid": 3.0, "cell_3_high": 9.0}, n_weeks=500)
    x = panel[["x"]].to_numpy(dtype=float)
    y = panel["y"].to_numpy(dtype=float)

    train_dummies, _ = _cell_dummy_columns(panel["cell_id"], panel["cell_id"])
    assert train_dummies.shape == (len(panel), 2), "one dummy per non-reference cell"

    X_cell = sm.add_constant(np.hstack([x, train_dummies]), has_constant="add")
    result, _, _ = _fit_nb_mle_glm(y, X_cell)
    assert result is not None, "the fit failed; the test proves nothing"

    # Columns: [const, x, dummy(cell_2_mid), dummy(cell_3_high)].
    coef_mid, coef_high = result.params[2], result.params[3]
    assert coef_mid == pytest.approx(np.log(3.0), abs=0.3)
    assert coef_high == pytest.approx(np.log(9.0), abs=0.3)


def test_a_per_cell_intercept_gives_a_lower_deviance_than_features_alone() -> None:
    """The point of the control: on held-out weeks from the same cells, a model
    that cannot see cell identity pays for it in deviance.
    """
    rates = {"cell_a": 0.5, "cell_b": 2.0, "cell_c": 6.0, "cell_d": 15.0}
    train = _synthetic_panel(rates, n_weeks=300)
    test = _synthetic_panel(rates, n_weeks=150)

    x_train = train[["x"]].to_numpy(dtype=float)
    x_test = test[["x"]].to_numpy(dtype=float)
    y_train = train["y"].to_numpy(dtype=float)
    y_test = test["y"].to_numpy(dtype=float)

    X_train = sm.add_constant(x_train, has_constant="add")
    X_test = sm.add_constant(x_test, has_constant="add")
    no_cell, _, _ = _fit_nb_mle_glm(y_train, X_train)
    assert no_cell is not None
    dev_no_cell = mean_poisson_deviance_safe(
        y_test, np.clip(no_cell.predict(X_test), 1e-9, None)
    )

    train_dummies, test_dummies = _cell_dummy_columns(train["cell_id"], test["cell_id"])
    X_train_cell = sm.add_constant(np.hstack([x_train, train_dummies]), has_constant="add")
    X_test_cell = sm.add_constant(np.hstack([x_test, test_dummies]), has_constant="add")
    with_cell, _, _ = _fit_nb_mle_glm(y_train, X_train_cell)
    assert with_cell is not None
    dev_with_cell = mean_poisson_deviance_safe(
        y_test, np.clip(with_cell.predict(X_test_cell), 1e-9, None)
    )

    assert dev_with_cell < 0.5 * dev_no_cell, (
        f"the cell effect barely helped: {dev_with_cell} with it vs {dev_no_cell} without it"
    )


def test_a_cell_unseen_in_training_falls_back_to_the_reference_level() -> None:
    """No fold in the real walk-forward loop can hit this (every earlier year is
    in the training fold), but the function must not raise if one ever does.
    """
    train = _synthetic_panel({"cell_a": 1.0, "cell_b": 4.0}, n_weeks=100)
    unseen = pd.Series(["cell_never_trained_on"])

    train_dummies, unseen_dummies = _cell_dummy_columns(train["cell_id"], unseen)
    assert unseen_dummies.shape == (1, train_dummies.shape[1])
    assert np.all(unseen_dummies == 0.0), "an unseen cell must read as the reference level, not raise"
