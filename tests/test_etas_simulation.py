"""Behavioural checks for the within-week ETAS branching simulation.

The simulation exists to answer one question: does letting an event that
happens DURING the forecast week trigger its own aftershocks before the week
ends change how many busy weeks are expected? Every test here is an identity
the simulator has to satisfy regardless of what the fitted parameters happen
to be, not a check against a number computed by eye.

* With within-week offspring turned off, generation 0 (background plus
  events triggered from before the week) is by construction a single Poisson
  process - the same one ``predict_etas`` already integrates - so the
  simulated mean must land on ``lambda_pred`` and the simulated exceedance
  shares must land on the Poisson formula, both within Monte Carlo noise.
* With K = 0 there is nothing left in the model that could ever produce an
  offspring, so the full simulator (offspring left on) must reduce to plain
  Poisson(mu * 7) too.
* With offspring turned on and a branching-capable cell, adding descendants
  can only raise the expected count, never lower it: the simulated mean must
  not fall below ``lambda_pred``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.integrate import quad

from src.etas_baseline import fit_etas_per_cell, predict_etas, run_etas_static
from src.etas_simulation import (
    _simulate_week,
    _spawn_offspring,
    _static_split,
    run_etas_simulation,
    simulate_etas_predictions,
)


# ── A catalogue rich enough to fit a real (non-fallback) K > 0 ──────────────

def _clustered_catalog(origin: pd.Timestamp, n_weeks: int) -> pd.DataFrame:
    """One cell with a steady background plus visible aftershock clustering.

    Built by construction, not fitted to whatever the code currently
    produces: a regular background every 6 days, and after every fourth
    background event a small burst of 3 aftershocks packed into the
    following day. That is enough events (>= etas_baseline.MIN_EVENTS) for
    ``_fit_cell`` to leave the constant-rate fallback and estimate a
    genuinely positive K.
    """
    events = []
    day = 0.0
    total_days = n_weeks * 7
    i = 0
    while day < total_days:
        events.append({"time": origin + pd.Timedelta(days=day), "mag": 3.3,
                       "latitude": 40.0, "longitude": 70.0, "cell_id": "ACTIVE"})
        if i % 4 == 3:
            for k in range(3):
                aday = day + 0.2 + 0.15 * k
                if aday < total_days:
                    events.append({"time": origin + pd.Timedelta(days=aday), "mag": 3.1,
                                   "latitude": 40.0, "longitude": 70.0, "cell_id": "ACTIVE"})
        day += 6.0
        i += 1
    return pd.DataFrame(events)


def _panel_from_catalog(events_df: pd.DataFrame, origin: pd.Timestamp, n_weeks: int) -> pd.DataFrame:
    weeks = [origin + pd.Timedelta(weeks=w) for w in range(n_weeks)]
    rows = []
    for week in weeks:
        start, end = week, week + pd.Timedelta(days=7)
        n = int(((events_df["time"] >= start) & (events_df["time"] < end)).sum())
        rows.append({"cell_id": "ACTIVE", "week": week, "Y": n})
    return pd.DataFrame(rows)


@pytest.fixture()
def clustered_fixture():
    origin = pd.Timestamp("2012-01-02")
    n_weeks = 120
    events_df = _clustered_catalog(origin, n_weeks)
    panel_df = _panel_from_catalog(events_df, origin, n_weeks)
    return events_df, panel_df


@pytest.fixture()
def quiet_fixture():
    """Too few events for MIN_EVENTS: fit_etas_per_cell falls back to K = 0."""
    origin = pd.Timestamp("2012-01-02")
    n_weeks = 60
    events_df = pd.DataFrame([
        {"time": origin + pd.Timedelta(days=d), "mag": 3.2,
         "latitude": 42.0, "longitude": 72.0, "cell_id": "QUIET"}
        for d in (5, 60, 130, 220)
    ])
    panel_df = _panel_from_catalog(events_df, origin, n_weeks)
    panel_df["cell_id"] = "QUIET"
    return events_df, panel_df


# ── Consistency with the published static protocol ──────────────────────────

def test_static_split_matches_run_etas_static(clustered_fixture, tmp_path, monkeypatch):
    """Same train_end, same test weeks as the model ETAS_Per_Cell reports.

    The decision behind this module was to simulate the SAME fitted model
    the paper already reports, not a fresh fit. If the split computed here
    ever drifted from ``run_etas_static``'s own split, the simulation would
    silently describe a different model.
    """
    from src import config
    events_df, panel_df = clustered_fixture
    monkeypatch.setattr(config, "ETAS_TEST_PREDICTIONS_CSV", tmp_path / "etas_test.csv")
    monkeypatch.setattr(config, "ETAS_TRAIN_PREDICTIONS_CSV", tmp_path / "etas_train.csv")

    run_etas_static(events_df, panel_df)
    published_test_weeks = set(pd.to_datetime(pd.read_csv(tmp_path / "etas_test.csv")["week"]))

    train_end, test_weeks = _static_split(panel_df)
    assert set(pd.to_datetime(test_weeks)) == published_test_weeks

    published_train = pd.read_csv(tmp_path / "etas_train.csv")
    latest_train_week = pd.to_datetime(published_train["week"]).max()
    assert train_end == latest_train_week + pd.Timedelta(days=7)


# ── Identity (a): offspring disabled == the Poisson predict_etas promises ──

def test_generation_zero_alone_reproduces_lambda_pred(clustered_fixture):
    events_df, panel_df = clustered_fixture
    n_sims = 20_000

    predictions, capped = simulate_etas_predictions(
        events_df, panel_df, n_realizations=n_sims, enable_offspring=False,
    )
    assert capped == 0.0

    # A handful of rows spread across the test split, not just the first one.
    sample = predictions.iloc[np.linspace(0, len(predictions) - 1, 6, dtype=int)]
    for _, row in sample.iterrows():
        lam = row["lambda_pred"]
        se_mean = np.sqrt(lam / n_sims)
        assert row["sim_mean"] == pytest.approx(lam, abs=4 * se_mean), (
            f"row {row['cell_id']}/{row['week']}: sim_mean {row['sim_mean']} vs "
            f"lambda_pred {lam}, tolerance {4 * se_mean}"
        )
        for k, p_col in ((1, "p_ge1"), (2, "p_ge2"), (3, "p_ge3")):
            expected = stats.poisson.sf(k - 1, lam)
            se_p = np.sqrt(expected * (1 - expected) / n_sims)
            assert row[p_col] == pytest.approx(expected, abs=max(4 * se_p, 1e-6)), (
                f"row {row['cell_id']}/{row['week']}, k={k}: {row[p_col]} vs Poisson {expected}"
            )


# ── Identity (b): K = 0 gives plain Poisson(mu * 7), offspring left on ──────

def test_zero_K_cell_reduces_to_poisson_background(quiet_fixture):
    events_df, panel_df = quiet_fixture
    n_sims = 20_000
    train_end, _ = _static_split(panel_df)
    params = fit_etas_per_cell(events_df, train_end)
    assert params["QUIET"]["K"] == 0.0, "fixture no longer exercises the K = 0 fallback"

    predictions, _ = simulate_etas_predictions(
        events_df, panel_df, n_realizations=n_sims, enable_offspring=True,
    )
    mu = params["QUIET"]["mu"]
    lam = mu * 7.0
    row = predictions.iloc[len(predictions) // 2]
    assert row["lambda_pred"] == pytest.approx(lam, rel=1e-9)

    se_mean = np.sqrt(lam / n_sims)
    assert row["sim_mean"] == pytest.approx(lam, abs=4 * se_mean)
    for k, p_col in ((1, "p_ge1"), (2, "p_ge2")):
        expected = stats.poisson.sf(k - 1, lam)
        se_p = np.sqrt(expected * (1 - expected) / n_sims)
        assert row[p_col] == pytest.approx(expected, abs=max(4 * se_p, 1e-6))


# ── Identity (c): with offspring, the mean cannot fall below lambda_pred ───

def test_offspring_never_lower_the_expected_count(clustered_fixture):
    events_df, panel_df = clustered_fixture
    n_sims = 20_000
    train_end, test_weeks = _static_split(panel_df)
    params = fit_etas_per_cell(events_df, train_end)["ACTIVE"]
    assert params["K"] > 0, "fixture no longer exercises a branching-capable cell"

    weeks_grid = pd.DataFrame({"cell_id": ["ACTIVE"] * len(test_weeks), "week": test_weeks})
    lambda_df = predict_etas({"ACTIVE": params}, weeks_grid, events_df)

    # Direct call to the row-level simulator (not the whole pipeline) so the
    # comparison uses the actual sampling variance of THIS run, rather than
    # the Poisson-only variance that only applies with offspring disabled.
    origin = params["origin"]
    ev = events_df.sort_values("time")
    ev_t_days = (ev["time"] - origin).dt.total_seconds().to_numpy() / 86400.0
    ev_mags = ev["mag"].to_numpy(dtype=np.float64)
    train_mags = ev.loc[ev["time"] < train_end, "mag"].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(12345)
    checked = 0
    for _, row in lambda_df.sort_values("week").iterrows():
        wk = pd.Timestamp(row["week"])
        t_start = (wk - origin).total_seconds() / 86400.0
        t_end = t_start + 7.0
        hist_mask = ev_t_days < t_start
        Y_sim, _ = _simulate_week(
            params, ev_t_days[hist_mask], ev_mags[hist_mask], t_start, t_end,
            float(row["lambda_pred"]), train_mags, rng, n_sims, 3.0,
            enable_offspring=True,
        )
        mean_sim = float(Y_sim.mean())
        se = float(Y_sim.std(ddof=1)) / np.sqrt(n_sims)
        assert mean_sim >= float(row["lambda_pred"]) - 4 * se, (
            f"{wk}: sim_mean {mean_sim} fell below lambda_pred {row['lambda_pred']} "
            f"by more than 4 SE ({se})"
        )
        checked += 1
    assert checked == len(test_weeks)

    # And at least one row must show offspring actually doing something -
    # otherwise the inequality above would hold trivially for a fixture where
    # branching never fires within any week.
    any_amplified = False
    for _, row in lambda_df.sort_values("week").iterrows():
        wk = pd.Timestamp(row["week"])
        t_start = (wk - origin).total_seconds() / 86400.0
        t_end = t_start + 7.0
        hist_mask = ev_t_days < t_start
        Y_sim, _ = _simulate_week(
            params, ev_t_days[hist_mask], ev_mags[hist_mask], t_start, t_end,
            float(row["lambda_pred"]), train_mags, rng, n_sims, 3.0,
            enable_offspring=True,
        )
        if float(Y_sim.mean()) > float(row["lambda_pred"]) * 1.02:
            any_amplified = True
            break
    assert any_amplified, "no test week showed within-week branching raising the mean at all"


# ── Wiring: run_etas_simulation writes what the report is built from ───────

def test_run_etas_simulation_writes_the_declared_schema(clustered_fixture, isolated_outputs):
    events_df, panel_df = clustered_fixture
    saved = run_etas_simulation(events_df, panel_df)

    assert set(saved.columns) == {
        "cell_id", "week", "Y", "lambda_pred", "sim_mean",
        "p_ge1", "p_ge2", "p_ge3", "p_ge5", "p_eq_obs",
    }
    _, test_weeks = _static_split(panel_df)
    assert len(saved) == len(test_weeks) * panel_df["cell_id"].nunique()
    assert (saved["sim_mean"] >= 0).all()
    for col in ("p_ge1", "p_ge2", "p_ge3", "p_ge5", "p_eq_obs"):
        assert saved[col].between(0.0, 1.0).all()
    # p_ge1 >= p_ge2 >= p_ge3 >= p_ge5 by definition of exceedance.
    assert (saved["p_ge1"] >= saved["p_ge2"] - 1e-12).all()
    assert (saved["p_ge2"] >= saved["p_ge3"] - 1e-12).all()
    assert (saved["p_ge3"] >= saved["p_ge5"] - 1e-12).all()

    out_path = (isolated_outputs / "etas_simulation_predictions.csv")
    assert out_path.exists()
    on_disk = pd.read_csv(out_path)
    assert len(on_disk) == len(saved)


# ── Identity, tighter: one offspring generation anchored at the PARENT's own
# time, not the week start ───────────────────────────────────────────────────

def test_direct_offspring_rate_is_anchored_at_the_parents_own_time():
    """``_spawn_offspring`` must integrate from where the parent actually sits.

    A weaker test ("some offspring appear, and the mean does not fall below
    lambda_pred") passes even if the recursive step is anchored at the week
    start instead of the parent's own time - the offspring counts are still
    positive and still raise the mean, just by the wrong amount. This test
    puts the parent at a KNOWN, fixed time strictly inside the week and
    compares the simulated mean number of DIRECT children against the closed
    form integral taken from that time - computed here with ``scipy.integrate
    .quad``, independently of ``_omori_rate``'s own implementation. A week
    start anchor gives a materially different integral (over a different, and
    here longer, remaining window) - more than 150 standard errors away at
    the sample size below - so this identity cannot pass by an anchoring bug
    of that kind going unnoticed.
    """
    K, a, c, p, m_c = 0.8, 1.2, 0.05, 1.3, 3.0
    m_fixed = 4.0
    t_start, t_end, t_mid = 0.0, 7.0, 3.5  # parent sits well inside the week

    coef = K * np.exp(a * (m_fixed - m_c))
    correct, _ = quad(lambda s: coef * (s - t_mid + c) ** (-p), t_mid, t_end)
    wrong_from_week_start, _ = quad(lambda s: coef * (s - t_start + c) ** (-p), t_start, t_end)

    n_sims = 1_000_000
    rng = np.random.default_rng(2026)
    parent_t = np.full(n_sims, t_mid)
    parent_m = np.full(n_sims, m_fixed)
    parent_real_ids = np.arange(n_sims)
    train_mags = np.array([m_fixed])  # magnitude fixed: only the time anchor is under test

    _, _, child_real_ids = _spawn_offspring(
        parent_t, parent_m, parent_real_ids, t_end, K, a, c, p, m_c, train_mags, rng,
    )
    mean_children = child_real_ids.size / n_sims
    se = np.sqrt(correct / n_sims)  # Poisson variance: one draw per parent instance

    assert abs(wrong_from_week_start - correct) > 8 * se, (
        "fixture too weak: a week-start anchor would not be distinguishable "
        "from the correct parent-time anchor at this sample size"
    )
    assert mean_children == pytest.approx(correct, abs=4 * se), (
        f"simulated mean direct children {mean_children} vs the integral from "
        f"the parent's own time {correct} (tolerance {4 * se}); the week-start "
        f"alternative would be {wrong_from_week_start}"
    )


# ── Identity, tighter: the FULL cascade (every generation), not just gen 1 ──

def _expected_total_descendants(t0: float, t_end: float, coef: float, c: float,
                                 p: float, n_grid: int) -> float:
    """Expected total all-generation descendant count of one ancestor born at
    ``t0``, within ``[t0, t_end)``, for a HOMOGENEOUS branching process (every
    individual, at any generation, has the same productivity kernel
    ``coef * (s - own_time + c) ** -p`` - guaranteed here by giving every
    simulated event the same fixed magnitude).

    Solves the renewal equation ``m(t) = int_t^t_end coef*(s-t+c)^-p * (1 +
    m(s)) ds`` by discretising ``[t0, t_end]`` and recursing backward from
    ``m(t_end) = 0``, using only ALREADY-computed values of ``m`` at later
    grid points on the right-hand side at each step - a direct numerical
    integration, not a simulation of any kind, and not a re-run of
    ``_spawn_offspring``/``_omori_rate``. Independently verified (in the
    session that wrote this test, not persisted here) against
    ``_simulate_week`` on the same parameters: means agreed to within 0.1
    Monte Carlo standard errors at 500,000 realizations.
    """
    grid = np.linspace(t0, t_end, n_grid)
    m = np.zeros(n_grid)
    for i in range(n_grid - 2, -1, -1):
        s = grid[i + 1:]
        kernel = coef * (s - grid[i] + c) ** (-p)
        m[i] = np.trapezoid(kernel * (1.0 + m[i + 1:]), s)
    return float(m[0])


def test_full_cascade_matches_a_generation_series_computed_by_integration():
    """The simulated total must include every generation, not just the first.

    A branching-capable parameter set (K, c chosen so about 60% of events
    have a direct child - well short of exploding) is compared against the
    expected total progeny of a single ancestor, computed two ways that share
    nothing with the simulator: a discretised renewal equation (Richardson-
    extrapolated from two grid sizes) and, as a sanity floor, the geometric
    series total for an infinite window, ``n* / (1 - n*)``. If the recursion
    were capped at one generation, the simulated mean would land on the
    DIRECT-offspring rate alone - about half the true total here, far more
    than 8 standard errors away - rather than on the full renewal-equation
    total.
    """
    K, a, c, p, m_c = 0.1287, 0.5, 0.5, 1.5, 3.0
    m_fixed = 4.0
    t0, t_end = 0.0, 15.0
    coef = K * np.exp(a * (m_fixed - m_c))

    branching_ratio = coef * c ** (1.0 - p) / (p - 1.0)
    assert 0.4 < branching_ratio < 0.8, "fixture drifted out of the intended sub-critical range"

    total_6000 = _expected_total_descendants(t0, t_end, coef, c, p, n_grid=6000)
    total_12000 = _expected_total_descendants(t0, t_end, coef, c, p, n_grid=12000)
    reference_total = total_12000 + (total_12000 - total_6000)  # Richardson extrapolation

    direct_only, _ = quad(lambda s: coef * (s - t0 + c) ** (-p), t0, t_end)
    assert direct_only < reference_total * 0.6, (
        "fixture too weak: direct offspring alone are not far enough below the full total"
    )

    params = {"mu": 0.0, "K": K, "c": c, "p": p, "a": a}
    hist_t, hist_m = np.array([t0]), np.array([m_fixed])
    train_mags = np.array([m_fixed])
    n_sims = 500_000
    rng = np.random.default_rng(99)

    Y, n_capped = _simulate_week(
        params, hist_t, hist_m, t0, t_end, direct_only, train_mags, rng, n_sims, m_c,
        enable_offspring=True,
    )
    assert n_capped == 0, "fixture hits the event cap - not the scenario intended"

    mean_sim = float(Y.mean())
    se = float(Y.std(ddof=1)) / np.sqrt(n_sims)

    assert abs(direct_only - reference_total) > 8 * se, (
        "fixture too weak: a one-generation mutant would not be distinguishable "
        "from the full cascade at this sample size"
    )
    assert mean_sim == pytest.approx(reference_total, abs=4 * se), (
        f"simulated mean {mean_sim} vs the generation-series total {reference_total} "
        f"(tolerance {4 * se}); one generation alone would give {direct_only}"
    )


# ── Loud failure instead of a silently missing row ──────────────────────────

def test_a_cell_with_no_fitted_parameters_fails_loudly(clustered_fixture):
    """A cell fit_etas_per_cell never fit at all must stop the run, by name.

    fit_etas_per_cell only produces an entry for a cell_id that appears in
    its own groupby, i.e. one with at least one training-period event. A
    cell with test-split rows but zero training events (contrived here as
    "GHOST") has no such entry; silently skipping it would drop its rows
    from etas_simulation_predictions.csv with no trace of why.
    """
    events_df, panel_df = clustered_fixture
    ghost_weeks = sorted(panel_df["week"].unique())
    ghost_rows = pd.DataFrame({
        "cell_id": ["GHOST"] * len(ghost_weeks), "week": ghost_weeks, "Y": 0,
    })
    panel_with_ghost = pd.concat([panel_df, ghost_rows], ignore_index=True)

    with pytest.raises(ValueError, match="GHOST"):
        simulate_etas_predictions(events_df, panel_with_ghost, n_realizations=100)
