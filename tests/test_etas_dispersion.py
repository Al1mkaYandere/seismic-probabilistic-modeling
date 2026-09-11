"""Fitting one extra parameter onto the seismological baseline.

The paper's claim is that Poisson forecasts - the baseline included - promise
far too few busy weeks, and that a dispersed distribution fixes it. The obvious
reply is to give the baseline the same dispersed distribution. These tests are
about making that reply honestly: the parameter is estimated, not guessed, it is
estimated away from the data it will be judged on, and it changes only the
spread of the forecast and never its mean.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gammaln

from src.probabilistic_evaluation import PredictiveDistribution, fit_dispersion


def _nb_log_likelihood(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    r = 1.0 / alpha
    return float(np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
                        + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))))


def test_a_known_dispersion_is_recovered_from_data_that_has_it() -> None:
    """Simulate a spread we chose, and get that number back."""
    rng = np.random.default_rng(20260911)
    n = 40_000
    mu = rng.uniform(0.05, 1.2, n)
    true_alpha = 2.0
    r = 1.0 / true_alpha
    y = rng.negative_binomial(r, r / (r + mu)).astype(float)

    fitted = fit_dispersion(y, mu)
    assert fitted == pytest.approx(true_alpha, rel=0.15), (
        f"asked for {true_alpha}, recovered {fitted}"
    )


def test_poisson_data_yields_a_dispersion_indistinguishable_from_none() -> None:
    """On Poisson data the estimate is near zero - near, not exactly.

    Written as exactly zero at first, and that was wrong: across seeds the
    maximum-likelihood estimate lands anywhere from 4e-7 to 7e-3, which is
    sampling noise and not a finding. What matters is not the digit but that a
    forecast built on it is the Poisson forecast for every threshold we report.
    """
    for seed in (1, 2, 3, 7):
        rng = np.random.default_rng(seed)
        mu = rng.uniform(0.05, 1.2, 40_000)
        y = rng.poisson(mu).astype(float)
        alpha = fit_dispersion(y, mu)
        assert 0.0 <= alpha < 0.05, f"seed {seed}: alpha={alpha} is not near zero"

        if alpha > 0:
            grid = np.array([0.05, 0.2, 0.5])
            plain = PredictiveDistribution("p", "poisson", grid)
            fitted = PredictiveDistribution("f", "nb", grid, np.full(len(grid), alpha))
            for k in (1, 2, 3):
                np.testing.assert_allclose(fitted.exceedance(k), plain.exceedance(k),
                                           rtol=0.05)


def test_data_with_less_spread_than_poisson_reports_exactly_zero() -> None:
    """Zero is the Poisson limit; 1/alpha is infinite there, so it must stop here.

    The case is counts that vary LESS than a Poisson would - mean 0.5, variance
    0.25. The first version of this test used all-zero counts instead and was
    wrong in the opposite direction: with a positive mean and nothing ever
    happening, the likelihood wants as much mass on zero as it can get, so the
    estimate runs to the top of the range rather than to the bottom.
    """
    mu = np.full(1000, 0.5)
    y = np.tile([0.0, 1.0], 500)
    assert np.var(y) < np.mean(y), "the premise is wrong: this data is not underdispersed"
    assert fit_dispersion(y, mu) == 0.0


def test_the_fitted_value_is_the_best_one_and_not_merely_a_plausible_one() -> None:
    """Checked against the likelihood itself, not against an expectation."""
    rng = np.random.default_rng(11)
    n = 8_000
    mu = rng.uniform(0.05, 1.0, n)
    r = 1.0 / 1.5
    y = rng.negative_binomial(r, r / (r + mu)).astype(float)

    fitted = fit_dispersion(y, mu)
    best = _nb_log_likelihood(y, mu, fitted)
    for candidate in (fitted * 0.5, fitted * 0.8, fitted * 1.25, fitted * 2.0):
        assert _nb_log_likelihood(y, mu, candidate) <= best + 1e-6, (
            f"alpha={candidate} explains the data better than the fitted {fitted}"
        )


def test_dispersion_moves_the_tail_and_leaves_the_mean_where_it_was() -> None:
    """The whole point of the comparison: same forecast, different spread.

    If the mean moved as well, a difference in the tail could be explained by
    the mean, and the test would say nothing about the distribution.
    """
    mu = np.array([0.02, 0.05, 0.1, 0.2, 0.5])     # the panel's range: 0.13 on average
    poisson = PredictiveDistribution("plain", "poisson", mu)
    dispersed = PredictiveDistribution("dispersed", "nb", mu, np.full(len(mu), 3.0))

    np.testing.assert_array_equal(poisson.mu, dispersed.mu)

    # Written first for means up to 2 and that was wrong: at mu = 2 dispersion
    # LOWERS P(Y >= 3), because it piles mass on zero and the bulk of the Poisson
    # already sits near 3. The tail is only lifted for thresholds well above the
    # mean - which is the regime this paper is about, and now the regime the test
    # is about.
    for k in (2, 3, 5):
        assert (dispersed.exceedance(k) > poisson.exceedance(k)).all(), (
            f"dispersion did not raise P(Y >= {k}) at means far below {k}"
        )
    assert (dispersed.exceedance(1) < poisson.exceedance(1)).all(), (
        "dispersion is supposed to pile mass on zero as well as on the tail"
    )


def test_fitting_ignores_nothing_and_reacts_to_the_data_it_is_given() -> None:
    """Two datasets that differ only in spread must not yield the same answer."""
    rng = np.random.default_rng(3)
    mu = rng.uniform(0.1, 1.0, 20_000)
    mild = rng.negative_binomial(1 / 0.5, (1 / 0.5) / ((1 / 0.5) + mu)).astype(float)
    wild = rng.negative_binomial(1 / 4.0, (1 / 4.0) / ((1 / 4.0) + mu)).astype(float)
    assert fit_dispersion(wild, mu) > 3 * fit_dispersion(mild, mu)


# ── Where the parameter comes from (step 2b) ─────────────────────────────────

def _write_predictions(tmp_path, monkeypatch, train_alpha: float, test_alpha: float):
    """Two synthetic files whose dispersions differ enough to tell apart.

    The point of the fixture: if the fit ever reads the test split, the answer
    jumps from one of these numbers to the other, and no tolerance can hide it.
    """
    import pandas as pd
    from src import config

    rng = np.random.default_rng(20260912)

    def draw(n, alpha, weeks_from):
        mu = rng.uniform(0.05, 0.8, n)
        r = 1.0 / alpha
        y = rng.negative_binomial(r, r / (r + mu)).astype(float)
        return pd.DataFrame({
            "cell_id": [f"c{i % 4}" for i in range(n)],
            "week": pd.date_range(weeks_from, periods=n, freq="D").astype(str),
            "y_true": y,
            "lambda_pred": mu,
        })

    train = draw(6000, train_alpha, "2010-01-04")
    test = draw(2000, test_alpha, "2021-01-04")
    train_path, test_path = tmp_path / "etas_train.csv", tmp_path / "etas_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    monkeypatch.setattr(config, "ETAS_TRAIN_PREDICTIONS_CSV", train_path)
    monkeypatch.setattr(config, "ETAS_TEST_PREDICTIONS_CSV", test_path)
    return train, test


def test_the_dispersion_comes_from_the_training_split_and_not_the_scored_one(
    tmp_path, monkeypatch
) -> None:
    """A baseline that has seen the answer cannot lose honestly.

    The two synthetic splits carry deliberately different spreads. Whichever one
    the fit reads is visible in the resulting number, so this cannot pass by
    accident or by tolerance.
    """
    train, test = _write_predictions(tmp_path, monkeypatch, train_alpha=0.4, test_alpha=6.0)

    from_train = fit_dispersion(train["y_true"].to_numpy(float),
                                train["lambda_pred"].to_numpy(float))
    from_test = fit_dispersion(test["y_true"].to_numpy(float),
                               test["lambda_pred"].to_numpy(float))
    assert from_test > 4 * from_train, "the fixture no longer separates the two splits"

    import pandas as pd
    from src import config
    from src.probabilistic_evaluation import collect_predictive_distributions

    # The rest of the machinery needs its own inputs; only the ETAS ones matter here.
    keys = test[["cell_id", "week"]].copy()
    keys["y_true"] = test["y_true"]
    preds = keys.copy()
    for column in ("pred_naive", "pred_poisson", "pred_nb"):
        preds[column] = test["lambda_pred"].to_numpy()
    preds_path = tmp_path / "test_predictions.csv"
    preds.to_csv(preds_path, index=False)
    monkeypatch.setattr(config, "TEST_PREDICTIONS_CSV", preds_path)

    comparison = pd.DataFrame([{"model": "NB_Enhanced", "alpha_hat": 1.0}])
    comparison_path = tmp_path / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    monkeypatch.setattr(config, "MODEL_COMPARISON_CSV", comparison_path)

    calib = pd.concat([
        keys.assign(model=name, mu_pred=test["lambda_pred"].to_numpy(), alpha_pred=1.0)
        for name in ("Neural_Poisson_Enhanced", "Hybrid_DL_Enhanced")
    ], ignore_index=True)
    calib_path = tmp_path / "calibration_predictions.csv"
    calib.to_csv(calib_path, index=False)
    monkeypatch.setattr(config, "CALIBRATION_PREDICTIONS_CSV", calib_path)

    _, dists = collect_predictive_distributions()
    by_name = {d.name: d for d in dists}
    assert "ETAS_NB" in by_name, "the dispersed baseline was not built at all"

    etas_nb, etas = by_name["ETAS_NB"], by_name["ETAS_Per_Cell"]
    assert etas_nb.family == "nb"
    np.testing.assert_array_equal(etas_nb.mu, etas.mu)
    used = float(np.unique(etas_nb.alpha)[0])
    # Not an exact identity: the module refits from the CSV, and a float does not
    # always survive the round trip through text unchanged. The tolerance is far
    # tighter than the twentyfold gap between the two splits, so it cannot mask
    # the thing being tested.
    assert used == pytest.approx(from_train, rel=1e-6), (
        f"the dispersion used is {used}: fitted on the split being scored ({from_test}), "
        f"not on the training period ({from_train})"
    )
    # Same mean, more mass far out: that is the only difference being claimed.
    assert (etas_nb.exceedance(3) > etas.exceedance(3)).all()


def test_every_scored_model_is_also_compared_against_something(tmp_path, monkeypatch) -> None:
    """A model that is scored but never compared is a result nobody will see.

    Found the hard way: deleting the two comparison pairs that produce this
    step's main number - the dispersed baseline against the plain one, and the
    network against the dispersed baseline - left all ninety-three tests green.
    The comparison table is where the paper's claims come from, so a model
    reaching the scores table and not the comparison table has to fail here
    rather than quietly vanish from the report.
    """
    from src.probabilistic_evaluation import COMPARISON_PAIRS

    _write_predictions(tmp_path, monkeypatch, train_alpha=1.5, test_alpha=1.5)

    import pandas as pd
    from src import config
    from src.probabilistic_evaluation import collect_predictive_distributions

    test = pd.read_csv(config.ETAS_TEST_PREDICTIONS_CSV)
    keys = test[["cell_id", "week"]].copy()
    keys["y_true"] = test["y_true"]
    preds = keys.copy()
    for column in ("pred_naive", "pred_poisson", "pred_nb"):
        preds[column] = test["lambda_pred"].to_numpy()
    preds_path = tmp_path / "test_predictions.csv"
    preds.to_csv(preds_path, index=False)
    monkeypatch.setattr(config, "TEST_PREDICTIONS_CSV", preds_path)

    comparison_path = tmp_path / "model_comparison.csv"
    pd.DataFrame([{"model": "NB_Enhanced", "alpha_hat": 1.0}]).to_csv(comparison_path, index=False)
    monkeypatch.setattr(config, "MODEL_COMPARISON_CSV", comparison_path)

    calib = pd.concat([
        keys.assign(model=name, mu_pred=test["lambda_pred"].to_numpy(), alpha_pred=1.0)
        for name in ("Neural_Poisson_Enhanced", "Hybrid_DL_Enhanced")
    ], ignore_index=True)
    calib_path = tmp_path / "calibration_predictions.csv"
    calib.to_csv(calib_path, index=False)
    monkeypatch.setattr(config, "CALIBRATION_PREDICTIONS_CSV", calib_path)

    _, dists = collect_predictive_distributions()
    compared = {name for pair in COMPARISON_PAIRS for name in pair}
    scored = {d.name for d in dists}

    # Naive_Persistence is deliberately outside the comparisons: it is there as a
    # floor to read the other numbers against, not as a competitor.
    orphans = scored - compared - {"Naive_Persistence", "Poisson_GLM"}
    assert not orphans, f"scored but never compared, so never reported: {sorted(orphans)}"

    # The two pairs this step exists to produce, named explicitly: they answer
    # "does one extra parameter rescue the baseline" and "does the network still
    # beat it once it has that parameter".
    assert ("ETAS_NB", "ETAS_Per_Cell") in COMPARISON_PAIRS
    assert ("Hybrid_DL_Enhanced", "ETAS_NB") in COMPARISON_PAIRS
