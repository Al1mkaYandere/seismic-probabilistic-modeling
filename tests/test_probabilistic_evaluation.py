"""Behavioural checks for the probabilistic evaluation of the full test split.

Each test pins a property that, if it broke, would change the paper's numbers
without changing anything visibly: an off-by-one in an exceedance probability, a
dropped log(y!), an interval computed as if the 17 cells of one week were
independent, or predictions silently misaligned to the wrong rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom, poisson

from src import config, probabilistic_evaluation as pe


def test_exceedance_matches_direct_summation_of_the_pmf():
    """P(Y >= k) must be the tail sum from k upwards, not from k+1.

    Computed here by explicit summation rather than by calling scipy's survival
    function, so an off-by-one in the implementation cannot be reproduced by the
    test making the same mistake.
    """
    mu = np.array([0.05, 0.4, 2.0, 7.5])
    dist = pe.PredictiveDistribution("test", "poisson", mu)

    for k in (1, 2, 3, 5):
        expected = np.array([
            sum(poisson.pmf(j, m) for j in range(k, k + 200)) for m in mu
        ])
        assert dist.exceedance(k) == pytest.approx(expected, abs=1e-12)


def test_nb_exceedance_matches_direct_summation_of_the_pmf():
    mu = np.array([0.05, 0.4, 2.0])
    alpha = np.array([3.0, 1.0, 0.25])
    dist = pe.PredictiveDistribution("test", "nb", mu, alpha)

    r = 1.0 / alpha
    p = r / (r + mu)
    for k in (1, 2, 5):
        expected = np.array([
            sum(nbinom.pmf(j, n=r[i], p=p[i]) for j in range(k, k + 400))
            for i in range(len(mu))
        ])
        assert dist.exceedance(k) == pytest.approx(expected, abs=1e-12)


def test_log_score_keeps_the_log_factorial_term():
    """The Poisson log score must be the true one, constant included.

    Dropping log(y!) is the defect fixed in 4f9aafc. It is invisible on rows
    where y is 0 or 1 and grows fast afterwards, so the check uses a row with
    y = 6, where the term is worth 6.579.
    """
    from scipy.special import gammaln

    mu = np.array([0.3, 1.5, 4.0])
    y = np.array([0.0, 2.0, 6.0])
    dist = pe.PredictiveDistribution("test", "poisson", mu)

    expected = mu - y * np.log(mu) + gammaln(y + 1.0)
    got = dist.log_score(y)

    assert got == pytest.approx(expected, abs=1e-12)
    without_constant = mu - y * np.log(mu)
    assert not np.allclose(got, without_constant), (
        "the log score equals the version without log(y!) - the constant is missing"
    )


def test_crps_terms_match_the_definition():
    mu = np.array([0.3, 1.5, 4.0])
    y = np.array([0.0, 2.0, 3.0])
    dist = pe.PredictiveDistribution("test", "poisson", mu)

    k_max = 40
    terms = dist.crps_terms(y, k_max)
    expected = np.array([
        [(poisson.cdf(k, mu[i]) - (1.0 if y[i] <= k else 0.0)) ** 2 for k in range(k_max + 1)]
        for i in range(len(y))
    ])
    assert terms == pytest.approx(expected, abs=1e-12)


def test_bootstrap_resamples_whole_weeks_not_rows():
    """Rows of one week must move together.

    Built so the two are impossible to confuse: every row inside a week carries
    the same value, and the weeks differ wildly. Resampling whole weeks then
    reproduces the bootstrap of the 4 week-values exactly, while resampling rows
    would give a far narrower interval.
    """
    weeks = np.repeat(["w1", "w2", "w3", "w4"], 17)
    values = np.repeat([0.0, 0.0, 0.0, 100.0], 17).astype(float)

    lo, hi = pe.block_bootstrap_ci(values, weeks, n_boot=4000, seed=7)

    # With one non-zero week out of four, a whole-week resample returns a mean of
    # 0, 25, 50, 75 or 100 - so the interval has to reach 0 at the bottom and
    # well past 25 at the top. Row resampling would concentrate near 25.
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi >= 50.0, f"upper end {hi} is too tight for whole-week resampling"


def test_diebold_mariano_finds_nothing_between_identical_models():
    rng = np.random.default_rng(3)
    weeks = np.repeat([f"w{i}" for i in range(60)], 5)
    loss = rng.gamma(2.0, 0.3, size=len(weeks))

    out = pe.diebold_mariano(loss, loss.copy(), weeks)

    assert out["n_weeks"] == 60
    assert out["mean_diff"] == pytest.approx(0.0, abs=1e-15)
    assert out["p_value"] > 0.99


def test_diebold_mariano_finds_a_constant_handicap():
    rng = np.random.default_rng(3)
    weeks = np.repeat([f"w{i}" for i in range(60)], 5)
    loss_b = rng.gamma(2.0, 0.3, size=len(weeks))
    loss_a = loss_b + 0.5  # A is worse on every single row

    out = pe.diebold_mariano(loss_a, loss_b, weeks)

    assert out["mean_diff"] == pytest.approx(0.5, abs=1e-12)
    assert out["p_value"] < 1e-6


def _write_tiny_pipeline_outputs(tmp_path, shuffle_seed: int | None):
    """Four small stand-ins for the pipeline's prediction files.

    Values differ per row on purpose: if the loader ever joined by position, a
    shuffled file would attach some other cell's forecast to a row and the
    numbers would change. Constant columns would hide exactly that.
    """
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cells = [f"c{i}" for i in range(4)]
    weeks = ["2021-01-04", "2021-01-11", "2021-01-18"]
    rows = [{"cell_id": c, "week": w} for c in cells for w in weeks]
    base = pd.DataFrame(rows)
    n = len(base)
    base["y_true"] = np.arange(n) % 3

    preds = base.copy()
    preds["pred_naive"] = 0.05 + 0.01 * np.arange(n)
    preds["pred_poisson"] = 0.10 + 0.02 * np.arange(n)
    preds["pred_nb"] = 0.15 + 0.03 * np.arange(n)

    etas = base[["cell_id", "week", "y_true"]].copy()
    etas["lambda_pred"] = 0.20 + 0.04 * np.arange(n)

    calib = []
    for name, off in (("Neural_Poisson_Enhanced", 0.3), ("Hybrid_DL_Enhanced", 0.4)):
        sub = base[["cell_id", "week", "y_true"]].copy()
        sub["model"] = name
        sub["mu_pred"] = off + 0.05 * np.arange(n)
        sub["alpha_pred"] = 1.0 + 0.1 * np.arange(n)
        calib.append(sub)
    calib_df = pd.concat(calib, ignore_index=True)

    comparison = pd.DataFrame([{"model": "NB_Enhanced", "alpha_hat": 2.5}])

    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        etas = etas.sample(frac=1.0, random_state=int(rng.integers(1e6))).reset_index(drop=True)
        calib_df = calib_df.sample(frac=1.0, random_state=int(rng.integers(1e6))).reset_index(drop=True)

    paths = {
        "TEST_PREDICTIONS_CSV": tmp_path / "test_predictions.csv",
        "ETAS_TEST_PREDICTIONS_CSV": tmp_path / "etas_test_predictions.csv",
        "CALIBRATION_PREDICTIONS_CSV": tmp_path / "calibration_predictions.csv",
        "MODEL_COMPARISON_CSV": tmp_path / "model_comparison.csv",
    }
    preds.to_csv(paths["TEST_PREDICTIONS_CSV"], index=False)
    etas.to_csv(paths["ETAS_TEST_PREDICTIONS_CSV"], index=False)
    calib_df.to_csv(paths["CALIBRATION_PREDICTIONS_CSV"], index=False)
    comparison.to_csv(paths["MODEL_COMPARISON_CSV"], index=False)
    return paths


def _collect_with(paths, monkeypatch):
    for attr, path in paths.items():
        monkeypatch.setattr(config, attr, path)
    return pe.collect_predictive_distributions()


def test_predictions_are_joined_by_key_not_by_row_order(tmp_path, monkeypatch):
    """Shuffling an input file must not change a single number.

    The prediction files are written by different pipeline steps and share no
    guaranteed row order. A positional join would still run, still produce a
    full table, and quietly score every model against the wrong cells.
    """
    ordered = _write_tiny_pipeline_outputs(tmp_path / "a", shuffle_seed=None)
    shuffled = _write_tiny_pipeline_outputs(tmp_path / "b", shuffle_seed=11)

    keys_o, dists_o = _collect_with(ordered, monkeypatch)
    keys_s, dists_s = _collect_with(shuffled, monkeypatch)

    pd.testing.assert_frame_equal(keys_o, keys_s)
    assert [d.name for d in dists_o] == [d.name for d in dists_s]
    for a, b in zip(dists_o, dists_s):
        assert a.family == b.family
        assert a.mu == pytest.approx(b.mu, abs=1e-12), f"{a.name}: mu moved when rows were shuffled"
        if a.alpha is None:
            assert b.alpha is None
        else:
            assert a.alpha == pytest.approx(b.alpha, abs=1e-12)


def test_hybrid_as_poisson_shares_the_mean_of_the_nb_model(tmp_path, monkeypatch):
    """The diagnostic model must differ from Hybrid_DL only in its family.

    It exists to separate the contribution of the distribution from that of the
    mean. If its mean drifted from Hybrid_DL's, the comparison would silently
    become a comparison of two different forecasts and prove nothing.
    """
    paths = _write_tiny_pipeline_outputs(tmp_path, shuffle_seed=None)
    _keys, dists = _collect_with(paths, monkeypatch)
    by_name = {d.name: d for d in dists}

    assert by_name["Hybrid_DL_as_Poisson"].family == "poisson"
    assert by_name["Hybrid_DL_Enhanced"].family == "nb"
    assert by_name["Hybrid_DL_as_Poisson"].mu == pytest.approx(
        by_name["Hybrid_DL_Enhanced"].mu, abs=1e-12
    )
    assert by_name["Hybrid_DL_as_Poisson"].alpha is None


def test_missing_rows_are_refused_rather_than_silently_dropped(tmp_path, monkeypatch):
    """A gap in one file must stop the run, not shrink the sample.

    Left to pandas, a missing ETAS row becomes NaN and then a model scored on
    fewer observations than the rest - a comparison of different samples
    presented as a comparison of models.
    """
    paths = _write_tiny_pipeline_outputs(tmp_path, shuffle_seed=None)
    etas = pd.read_csv(paths["ETAS_TEST_PREDICTIONS_CSV"])
    etas.iloc[1:].to_csv(paths["ETAS_TEST_PREDICTIONS_CSV"], index=False)

    with pytest.raises(ValueError, match="ETAS"):
        _collect_with(paths, monkeypatch)


def test_missing_calibration_rows_are_refused_too(tmp_path, monkeypatch):
    """The same guard must hold for the neural models, not only for ETAS.

    Found by mutation: deleting the calibration guard left every test passing,
    because the ETAS case was the only one covered.
    """
    paths = _write_tiny_pipeline_outputs(tmp_path, shuffle_seed=None)
    calib = pd.read_csv(paths["CALIBRATION_PREDICTIONS_CSV"])
    trimmed = calib[~((calib["model"] == "Hybrid_DL_Enhanced")
                      & (calib["cell_id"] == "c0"))]
    trimmed.to_csv(paths["CALIBRATION_PREDICTIONS_CSV"], index=False)

    with pytest.raises(ValueError, match="Hybrid_DL_Enhanced"):
        _collect_with(paths, monkeypatch)


def test_diebold_mariano_widens_the_variance_for_autocorrelated_weeks():
    """The Newey-West correction has to actually correct something.

    A large earthquake raises every model's loss for several weeks running, so
    the weekly differences are positively autocorrelated and their raw variance
    is too small - which inflates the statistic and manufactures significance.
    The series here is AR(1) with rho = 0.7, built so that ignoring the
    autocorrelation is visible rather than marginal.

    Found by mutation: with the correction switched off (lag 0) every other test
    in this file still passed, because they use differences that are either
    exactly zero or exactly constant.
    """
    rng = np.random.default_rng(5)
    n_weeks, rho = 120, 0.7
    series = np.empty(n_weeks)
    series[0] = rng.normal()
    for t in range(1, n_weeks):
        series[t] = rho * series[t - 1] + rng.normal(scale=0.5)
    series += 0.30  # a small genuine handicap on top of the dependence

    rows_per_week = 5
    weeks = np.repeat([f"w{t:03d}" for t in range(n_weeks)], rows_per_week)
    loss_a = np.repeat(series, rows_per_week)
    loss_b = np.zeros_like(loss_a)

    corrected = pe.diebold_mariano(loss_a, loss_b, weeks)
    uncorrected = pe.diebold_mariano(loss_a, loss_b, weeks, lag=0)

    assert corrected["lag"] > 0, "the default lag must not be zero"
    assert abs(corrected["dm_stat"]) < 0.8 * abs(uncorrected["dm_stat"]), (
        f"correction barely changed the statistic: {corrected['dm_stat']:.3f} "
        f"vs {uncorrected['dm_stat']:.3f} without it"
    )
    assert corrected["p_value"] > uncorrected["p_value"]
