"""The completeness threshold must be one decision in one place.

The catalogue is complete only above some magnitude; below it, a missing event
is missing detection, not absence of an earthquake. mc_estimation.py estimates
that threshold at 4.5 for this catalogue while everything is modelled from 3.0,
so the number has to be a parameter someone can change and test, not two
literals in two files that happen to agree today.

These tests pin the behaviour of that parameter, not its current value: raising
M_C to 4.5 must change what the tests assert about counts, and none of them
should have to be rewritten for that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, etas_baseline
from src.grid_builder import build_spatiotemporal_grid, filter_to_completeness


def _events(mags: list[float]) -> pd.DataFrame:
    origin = pd.Timestamp("2015-01-05", tz="UTC")
    return pd.DataFrame({
        "time": [origin + pd.Timedelta(weeks=i) for i in range(len(mags))],
        "latitude": [39.0] * len(mags),
        "longitude": [66.0] * len(mags),
        "mag": mags,
    })


def test_filter_keeps_events_at_the_threshold_and_drops_those_below():
    """At the threshold means kept - the boundary is inclusive.

    An exclusive boundary would quietly discard every event reported exactly at
    the completeness magnitude, which for a catalogue rounded to one decimal is
    a whole bin.
    """
    df = _events([2.9, 3.0, 3.1, 4.4, 4.5, 4.6])

    kept = filter_to_completeness(df, m_c=4.5)

    assert sorted(kept["mag"]) == [4.5, 4.6]
    assert len(filter_to_completeness(df, m_c=3.0)["mag"]) == 5
    assert len(filter_to_completeness(df, m_c=0.0)) == len(df)


def test_filter_defaults_to_the_configured_threshold(monkeypatch):
    df = _events([3.0, 4.0, 5.0])

    monkeypatch.setattr(config, "M_C", 4.5)
    assert sorted(filter_to_completeness(df)["mag"]) == [5.0]

    monkeypatch.setattr(config, "M_C", 3.0)
    assert len(filter_to_completeness(df)) == 3


def test_etas_reads_the_threshold_when_it_runs_not_when_it_is_imported(monkeypatch):
    """One number, read at call time.

    ETAS's likelihood is derived assuming every event above M_c was observed. If
    its threshold drifted from the one the panel was built with, the baseline
    would be fitted against a catalogue different from the one it is scored on -
    silently, because both would still run.

    Reading config at import time is the subtler version of the same bug: the
    value freezes at whatever was loaded first, so changing the threshold for an
    experiment moves the panel and leaves ETAS behind. The check therefore
    changes config after import and asserts ETAS follows.
    """
    monkeypatch.setattr(config, "M_C", 4.5)
    assert etas_baseline._m_c() == 4.5
    assert etas_baseline._m_c(3.0) == 3.0, "an explicit argument must still win"

    monkeypatch.setattr(config, "M_C", 3.0)
    assert etas_baseline._m_c() == 3.0


def test_etas_intensity_uses_the_configured_threshold(monkeypatch):
    """Behaviour, not just the resolver: the threshold must reach the maths.

    Productivity enters as exp(a * (m - M_c)), so raising M_c lowers every
    event's aftershock contribution. Two identical calls that differ only in the
    configured threshold must therefore give different forecasts.
    """
    origin = pd.Timestamp("2015-01-05")
    events = pd.DataFrame({
        "time": [origin + pd.Timedelta(days=d) for d in (0, 10, 20, 30, 40, 50)],
        "mag": [5.5, 4.0, 4.2, 4.1, 4.3, 4.6],
        "cell_id": ["A"] * 6,
    })
    params = {"A": {"mu": 0.01, "K": 0.5, "c": 0.05, "p": 1.2, "a": 1.0,
                    "origin": origin, "fallback": False}}
    grid = pd.DataFrame({"cell_id": ["A"], "week": [origin + pd.Timedelta(days=60)]})

    monkeypatch.setattr(config, "M_C", 3.0)
    low = etas_baseline.predict_etas(params, grid, events)["lambda_pred"].iloc[0]
    monkeypatch.setattr(config, "M_C", 4.5)
    high = etas_baseline.predict_etas(params, grid, events)["lambda_pred"].iloc[0]

    assert low > high, (
        f"forecast did not react to the threshold: {low} at M_c=3.0 versus "
        f"{high} at 4.5 - the parameter is not reaching the intensity"
    )


def test_raising_the_threshold_actually_thins_the_panel():
    """The parameter has to reach the panel, not just the filter.

    Verified through the builder rather than by counting rows of the filter's
    output: the point is that a raised threshold produces a panel with fewer
    events in it, which is what the completeness experiment depends on.
    """
    origin = pd.Timestamp("2015-01-05", tz="UTC")
    rows = []
    for i in range(40):
        rows.append({"time": origin + pd.Timedelta(weeks=i, days=1),
                     "latitude": 39.0, "longitude": 66.0,
                     "mag": 4.8 if i % 4 == 0 else 3.5})
    df = pd.DataFrame(rows)

    low = build_spatiotemporal_grid(filter_to_completeness(df, m_c=3.0))
    high = build_spatiotemporal_grid(filter_to_completeness(df, m_c=4.5))

    assert int(low["Y"].sum()) > int(high["Y"].sum())
    assert int(high["Y"].sum()) == int((df["mag"] >= 4.5).sum()) - _warmup_events(df, 4.5)


def _warmup_events(df: pd.DataFrame, m_c: float) -> int:
    """Events of the filtered catalogue that fall inside the dropped warm-up."""
    from src.grid_builder import W_MAX

    kept = df.loc[df["mag"] >= m_c]
    weeks = kept["time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("W").dt.start_time
    axis = pd.date_range(weeks.min(), weeks.max(), freq="W-MON")
    return int(weeks.isin(set(axis[:W_MAX])).sum())


def test_the_real_catalogue_is_untouched_at_the_configured_default():
    """The published numbers must stay reproducible.

    The catalogue's smallest event is 3.1 and the configured threshold is 3.0, so
    the filter is a no-op today. This test is what turns that from a coincidence
    into a checked fact - if either number moves, the published panel changes and
    someone has to say so out loud.
    """
    raw = pd.read_csv(config.RAW_DATA_PATH / config.RAW_DATA_FILE)
    assert len(filter_to_completeness(raw)) == len(raw), (
        f"the completeness filter drops events at M_C={config.M_C}; the panel "
        "behind the published numbers would change"
    )


def test_fitting_uses_the_threshold_too_not_only_forecasting(monkeypatch):
    """The threshold has to reach the LIKELIHOOD, not just the forecast.

    Productivity enters the ETAS likelihood as exp(a * (m - M_c)) exactly as it
    enters the intensity, so a fit run at a different threshold must land on
    different parameters. Two paths, two chances to hard-code a 3.0.

    Found by review: the earlier tests pinned the forecasting path only, and a
    literal 3.0 planted inside _negative_log_likelihood passed all of them.
    """
    origin = pd.Timestamp("2015-01-05")
    rng = np.random.default_rng(0)
    days = np.sort(rng.uniform(0, 500, size=40))
    events = pd.DataFrame({
        "time": [origin + pd.Timedelta(days=float(d)) for d in days],
        "mag": rng.uniform(4.0, 6.0, size=len(days)),
        "cell_id": ["A"] * len(days),
    })
    train_end = origin + pd.Timedelta(days=600)

    monkeypatch.setattr(config, "M_C", 3.0)
    low = etas_baseline.fit_etas_per_cell(events, train_end)["A"]
    monkeypatch.setattr(config, "M_C", 4.5)
    high = etas_baseline.fit_etas_per_cell(events, train_end)["A"]

    moved = [k for k in ("mu", "K", "c", "p", "a")
             if not np.isclose(low[k], high[k], rtol=1e-9, atol=1e-12)]
    assert moved, (
        "the fit landed on identical parameters at M_c=3.0 and M_c=4.5 - the "
        "threshold is not reaching the likelihood"
    )


def test_the_likelihood_obeys_the_threshold_scaling_identity():
    """An exact identity, so a threshold hard-coded anywhere breaks it.

    Productivity enters as K * exp(a * (m - M_c)), so raising the threshold by d
    multiplies every aftershock term by exp(-a*d) - which multiplying K by
    exp(a*d) undoes exactly. The likelihood at (K, M_c) and at (K*exp(a*d),
    M_c+d) must therefore be the same number.

    The earlier version of this file only asserted that fits at two thresholds
    differ. Review showed that is too weak: planting a literal 3.0 in one of the
    two places the threshold is used still left the two fits different, via the
    other place, and the test passed. This identity holds only if EVERY use of
    the threshold moves together.
    """
    rng = np.random.default_rng(1)
    t_days = np.sort(rng.uniform(0.0, 400.0, size=40))
    mags = rng.uniform(4.0, 6.0, size=40)
    mu, K, c, p, a = 0.02, 0.4, 0.05, 1.25, 1.1
    shift = 1.5

    base = etas_baseline._negative_log_likelihood(
        np.array([mu, K, c, p, a]), t_days, mags, 3.0
    )
    rescaled = etas_baseline._negative_log_likelihood(
        np.array([mu, K * np.exp(a * shift), c, p, a]), t_days, mags, 3.0 + shift
    )

    assert base == pytest.approx(rescaled, rel=1e-12), (
        f"likelihood {base} at M_c=3.0 does not match {rescaled} at "
        f"M_c={3.0 + shift} with K rescaled by exp(a*d) - some use of the "
        "threshold is not following the parameter"
    )
