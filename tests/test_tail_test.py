"""Behavioural checks for the standard tail test and comparison power.

Each test pins an identity the module has to satisfy regardless of what the
predictions happen to be: the Poisson-binomial convolution must be a genuine
probability distribution, must reduce to the textbook binomial in the
identical-probability case, must use exactly the row probabilities
``probabilistic_evaluation`` already reports as ``expected_count``, and the
p-value formula must match a value worked out by hand.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, norm

from src import config, tail_test as tt
from src import probabilistic_evaluation as pe


# ── poisson_binomial_pmf: identities that hold for ANY probabilities ───────

def test_poisson_binomial_pmf_sums_to_one():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.0, 1.0, size=200)

    pmf = tt.poisson_binomial_pmf(p)

    assert len(pmf) == 201
    assert pmf.sum() == pytest.approx(1.0, abs=1e-9)
    assert (pmf >= -1e-15).all(), "a probability went negative"


def test_poisson_binomial_pmf_of_zero_trials_is_a_point_mass_at_zero():
    pmf = tt.poisson_binomial_pmf(np.array([]))
    assert pmf == pytest.approx(np.array([1.0]))


def test_poisson_binomial_matches_scipy_binomial_for_equal_probabilities():
    """With every p_i equal, the Poisson-binomial IS the binomial.

    Checked against scipy's own implementation, computed a completely
    different way, so this cannot pass by the test repeating the same
    formula as the code under test.
    """
    n, p = 37, 0.23
    pmf = tt.poisson_binomial_pmf(np.full(n, p))
    expected = binom.pmf(np.arange(n + 1), n, p)
    assert pmf == pytest.approx(expected, abs=1e-12)


def test_poisson_binomial_matches_scipy_binomial_at_an_extreme_probability():
    n, p = 25, 0.97
    pmf = tt.poisson_binomial_pmf(np.full(n, p))
    expected = binom.pmf(np.arange(n + 1), n, p)
    assert pmf == pytest.approx(expected, abs=1e-12)


def test_discrete_quantile_uses_the_left_definition_on_a_tie():
    """The quantile is defined as the smallest n with P(N<=n) >= q. When q
    lands exactly on a cumulative-probability value, that is the index
    itself, not the next one - a 'first index strictly greater than q'
    reading would return one index too many.
    """
    pmf = np.array([0.25, 0.5, 0.25])  # cdf = [0.25, 0.75, 1.0]
    assert tt._discrete_quantile(pmf, 0.25) == 0
    assert tt._discrete_quantile(pmf, 0.75) == 1
    assert tt._discrete_quantile(pmf, 0.76) == 2


# ── _tail_test_row: the p-value formula, checked by hand ───────────────────

def test_p_value_matches_a_hand_computed_example():
    """Two coin flips, k picked so 'observed' is the maximum possible count.

    pmf = [0.25, 0.5, 0.25] by direct expansion of (1-p)(1-p), 2p(1-p), p^2 at
    p = 0.5. With observed = 2: P(N<=2) = 1, P(N>=2) = 0.25, so the two-sided
    p-value is 2*min(1, 0.25) = 0.5 - worked out independently of the code
    under test.
    """
    p = np.array([0.5, 0.5])
    row = tt._tail_test_row("toy", 3, p, observed=2)

    assert row["expected_count"] == pytest.approx(1.0, abs=1e-12)
    assert row["p_lower"] == pytest.approx(1.0, abs=1e-12)
    assert row["p_upper"] == pytest.approx(0.25, abs=1e-12)
    assert row["p_two_sided"] == pytest.approx(0.5, abs=1e-12)
    assert row["count_q025"] == pytest.approx(0.0, abs=1e-12)
    assert row["count_q975"] == pytest.approx(2.0, abs=1e-12)


def test_p_value_at_the_lower_boundary_of_the_observed_count():
    """observed = 0 must give P(N >= 0) = 1 exactly, not the formula's usual
    'one CDF step below' which would read a nonexistent cdf[-1]."""
    p = np.array([0.5, 0.5])
    row = tt._tail_test_row("toy", 3, p, observed=0)

    assert row["p_upper"] == pytest.approx(1.0, abs=1e-12)
    assert row["p_lower"] == pytest.approx(0.25, abs=1e-12)
    assert row["p_two_sided"] == pytest.approx(0.5, abs=1e-12)


def test_two_sided_p_value_never_exceeds_one():
    """When the observed count sits at the distribution's own mode, both
    one-sided tails can individually exceed 0.25, and 2 * min(...) must be
    capped rather than reported as a probability above 1."""
    p = np.full(10, 0.5)
    row = tt._tail_test_row("toy", 3, p, observed=5)  # the mode of Binomial(10, 0.5)
    # By symmetry P(N<=5) = P(N>=5) = 638/1024 ≈ 0.623, so 2*min(...) ≈ 1.246
    # before capping - this case only tests something if the cap actually fires.
    assert row["p_lower"] == pytest.approx(638 / 1024, abs=1e-9)
    assert row["p_two_sided"] == pytest.approx(1.0, abs=1e-12)


# ── Identity against probabilistic_evaluation's own numbers ────────────────

def _write_synthetic_pipeline_outputs() -> pd.DataFrame:
    """Tiny stand-ins for the four prediction files ``collect_predictive_
    distributions`` reads, written to the paths ``config`` currently points
    at (the isolated_outputs fixture must already be active).
    """
    cells = [f"c{i}" for i in range(4)]
    weeks = ["2021-01-04", "2021-01-11", "2021-01-18", "2021-01-25"]
    rows = [{"cell_id": c, "week": w} for c in cells for w in weeks]
    base = pd.DataFrame(rows)
    n = len(base)
    rng = np.random.default_rng(0)
    base["y_true"] = rng.poisson(1.5, size=n)

    preds = base.copy()
    preds["pred_naive"] = 0.3 + 0.05 * np.arange(n)
    preds["pred_poisson"] = 0.5 + 0.05 * np.arange(n)
    preds["pred_nb"] = 0.7 + 0.05 * np.arange(n)
    preds.to_csv(config.TEST_PREDICTIONS_CSV, index=False)

    etas = base[["cell_id", "week", "y_true"]].copy()
    etas["lambda_pred"] = 0.6 + 0.05 * np.arange(n)
    etas.to_csv(config.ETAS_TEST_PREDICTIONS_CSV, index=False)

    calib_parts = []
    for name, off, family in (("Neural_Poisson_Enhanced", 0.8, "poisson"),
                              ("Hybrid_DL_Enhanced", 1.0, "nb")):
        sub = base[["cell_id", "week", "y_true"]].copy()
        sub["model"] = name
        sub["family"] = family
        sub["mu_pred"] = off + 0.05 * np.arange(n)
        sub["alpha_pred"] = 1.5
        calib_parts.append(sub)
    pd.concat(calib_parts, ignore_index=True).to_csv(config.CALIBRATION_PREDICTIONS_CSV, index=False)

    pd.DataFrame([{"model": "NB_Enhanced", "alpha_hat": 2.0}]).to_csv(
        config.MODEL_COMPARISON_CSV, index=False)
    return base


def test_expected_count_matches_exceedance_calibration_to_1e_minus_9(isolated_outputs):
    """The row probabilities this module sums must be the SAME probabilities
    ``exceedance_calibration.csv`` already reports as ``expected_count`` -
    not a value that merely looks close.
    """
    _write_synthetic_pipeline_outputs()
    pe.run_probabilistic_evaluation()
    calib = pd.read_csv(pe.OUTPUT_EXCEEDANCE_CSV)

    _keys, dists = pe.collect_predictive_distributions()
    checked = 0
    for dist in dists:
        for k in tt.THRESHOLDS:
            row = calib.loc[(calib["model"] == dist.name) & (calib["threshold"] == k)]
            assert len(row) == 1
            from_csv = float(row["expected_count"].iloc[0])
            p_sum = float(dist.exceedance(k).sum())
            assert p_sum == pytest.approx(from_csv, abs=1e-9)
            checked += 1
    assert checked == len(dists) * len(tt.THRESHOLDS)


# ── Reading distributions from an arbitrary directory ───────────────────────

def test_collect_from_directory_reads_another_dir_and_restores_config(isolated_outputs, tmp_path):
    """Pointed at a directory that is NOT config.OUTPUT_DIR, the loader must
    read from there, and must leave config exactly as it found it afterwards
    - a later call with no arguments must still read the original directory.
    """
    _write_synthetic_pipeline_outputs()
    original_keys, original_dists = pe.collect_predictive_distributions()

    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    for name in ("test_predictions.csv", "etas_test_predictions.csv",
                 "calibration_predictions.csv", "model_comparison.csv"):
        (config.OUTPUT_DIR / name).replace(other_dir / name)

    before = {name: getattr(config, name) for name in
              ("TEST_PREDICTIONS_CSV", "MODEL_COMPARISON_CSV",
               "ETAS_TEST_PREDICTIONS_CSV", "CALIBRATION_PREDICTIONS_CSV")}

    keys, dists = tt.collect_predictive_distributions_from(other_dir)

    pd.testing.assert_frame_equal(keys, original_keys)
    assert [d.name for d in dists] == [d.name for d in original_dists]

    after = {name: getattr(config, name) for name in before}
    assert after == before, "collect_predictive_distributions_from left config pointed elsewhere"


def test_tail_test_table_for_directory_matches_manual_computation(isolated_outputs, tmp_path):
    _write_synthetic_pipeline_outputs()
    keys, dists = pe.collect_predictive_distributions()
    y_true = keys["y_true"].to_numpy(dtype=np.float64)
    by_name = {d.name: d for d in dists}

    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    for name in ("test_predictions.csv", "etas_test_predictions.csv",
                 "calibration_predictions.csv", "model_comparison.csv"):
        (config.OUTPUT_DIR / name).replace(other_dir / name)

    table = tt.tail_test_table_for_directory(other_dir, model_names=("Hybrid_DL_Enhanced",))

    assert set(table["model"]) == {"Hybrid_DL_Enhanced"}
    assert sorted(table["threshold"]) == sorted(tt.THRESHOLDS)
    for _, row in table.iterrows():
        k = int(row["threshold"])
        expected = float(by_name["Hybrid_DL_Enhanced"].exceedance(k).sum())
        assert float(row["expected_count"]) == pytest.approx(expected, abs=1e-9)
        observed = int((y_true >= k).sum())
        assert float(row["observed_count"]) == pytest.approx(observed)


def test_tail_test_table_for_directory_raises_if_a_model_is_missing(isolated_outputs, tmp_path):
    _write_synthetic_pipeline_outputs()
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    for name in ("test_predictions.csv", "etas_test_predictions.csv",
                 "calibration_predictions.csv", "model_comparison.csv"):
        (config.OUTPUT_DIR / name).replace(other_dir / name)

    with pytest.raises(ValueError, match="Does_Not_Exist"):
        tt.tail_test_table_for_directory(other_dir, model_names=("Does_Not_Exist",))


# ── ETAS_Branching_Sim rows ─────────────────────────────────────────────────

def test_etas_simulation_rows_are_empty_when_the_file_is_missing(tmp_path):
    assert tt._etas_simulation_rows(tmp_path, tt.THRESHOLDS) == []


def test_etas_simulation_rows_use_the_simulated_columns(tmp_path):
    sim = pd.DataFrame({
        "cell_id": ["c0", "c0", "c1"],
        "week": ["2021-01-04", "2021-01-11", "2021-01-04"],
        "Y": [0.0, 3.0, 2.0],
        "p_ge2": [0.1, 0.9, 0.8],
        "p_ge3": [0.05, 0.7, 0.2],
    })
    sim.to_csv(tmp_path / "etas_simulation_predictions.csv", index=False)

    rows = tt._etas_simulation_rows(tmp_path, tt.THRESHOLDS)

    by_threshold = {r["threshold"]: r for r in rows}
    assert set(by_threshold) == {2, 3}
    assert by_threshold[2]["model"] == tt.ETAS_SIM_MODEL
    assert by_threshold[2]["observed_count"] == pytest.approx(2.0)  # Y=[0,3,2] >= 2
    assert by_threshold[2]["expected_count"] == pytest.approx(0.1 + 0.9 + 0.8, abs=1e-12)
    assert by_threshold[3]["observed_count"] == pytest.approx(1.0)  # Y=[0,3,2] >= 3


# ── score_power.csv ─────────────────────────────────────────────────────────

def test_score_power_table_matches_hand_computed_values(isolated_outputs):
    comparisons = pd.DataFrame([
        {"model_a": "X", "model_b": "Y", "score": "log_score",
         "mean_diff": 0.02, "dm_stat": 2.5, "diff_lo": 0.0, "diff_hi": 0.0,
         "p_value": 0.01, "n_weeks": 144, "newey_west_lag": 4, "better": "Y"},
        {"model_a": "X", "model_b": "Y", "score": "twcrps_ge5",
         "mean_diff": 1.0, "dm_stat": 1.0, "diff_lo": 0.0, "diff_hi": 0.0,
         "p_value": 0.3, "n_weeks": 144, "newey_west_lag": 4, "better": "Y"},
    ])
    path = config.OUTPUT_DIR / "score_comparisons.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(path, index=False)

    table = tt._score_power_table(path)

    assert len(table) == 1, "only the log_score row must be kept"
    row = table.iloc[0]
    se = 0.02 / 2.5
    z = norm.ppf(0.975) + norm.ppf(0.80)
    min_detectable = z * se
    assert row["se"] == pytest.approx(se, abs=1e-12)
    assert row["min_detectable_diff"] == pytest.approx(min_detectable, abs=1e-12)
    assert row["ratio"] == pytest.approx(abs(0.02) / min_detectable, abs=1e-12)


# ── run_tail_test: the full step, end to end ────────────────────────────────

def test_run_tail_test_writes_both_files_with_the_declared_columns(isolated_outputs):
    _write_synthetic_pipeline_outputs()
    pe.run_probabilistic_evaluation()

    result = tt.run_tail_test()

    tail_df = pd.read_csv(tt.OUTPUT_TAIL_TEST_CSV)
    power_df = pd.read_csv(tt.OUTPUT_SCORE_POWER_CSV)

    assert list(tail_df.columns) == [
        "model", "threshold", "observed_count", "expected_count",
        "count_q025", "count_q975", "p_lower", "p_upper", "p_two_sided",
    ]
    assert list(power_df.columns) == [
        "model_a", "model_b", "mean_diff", "se", "min_detectable_diff", "ratio",
    ]
    assert set(tail_df["threshold"]) == set(tt.THRESHOLDS)
    comparisons_df = pd.read_csv(pe.OUTPUT_COMPARISONS_CSV)
    expected_power_rows = int((comparisons_df["score"] == "log_score").sum())
    assert expected_power_rows > 0
    assert len(power_df) == expected_power_rows
    pd.testing.assert_frame_equal(result["tail_test"], tail_df)
    pd.testing.assert_frame_equal(result["score_power"], power_df)


def test_run_tail_test_includes_the_branching_simulation_when_present(isolated_outputs):
    """The wiring, not just the two pieces separately: if the simulation's
    file is sitting next to the other pipeline outputs, its rows must reach
    the file ``run_tail_test`` actually writes - a step could easily compute
    ``_etas_simulation_rows`` and then forget to fold it into the table.
    """
    _write_synthetic_pipeline_outputs()
    pe.run_probabilistic_evaluation()
    sim = pd.DataFrame({
        "cell_id": ["c0", "c1"], "week": ["2021-01-04", "2021-01-11"],
        "Y": [1.0, 4.0], "p_ge2": [0.2, 0.95], "p_ge3": [0.1, 0.9],
    })
    sim.to_csv(config.OUTPUT_DIR / "etas_simulation_predictions.csv", index=False)

    tt.run_tail_test()

    tail_df = pd.read_csv(tt.OUTPUT_TAIL_TEST_CSV)
    sim_rows = tail_df.loc[tail_df["model"] == tt.ETAS_SIM_MODEL]
    assert sorted(sim_rows["threshold"]) == sorted(tt.THRESHOLDS)
    row_k2 = sim_rows.loc[sim_rows["threshold"] == 2].iloc[0]
    assert row_k2["expected_count"] == pytest.approx(0.2 + 0.95, abs=1e-12)
    assert row_k2["observed_count"] == pytest.approx(1.0)  # Y=[1,4] >= 2
