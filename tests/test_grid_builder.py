"""Behavioural checks for build_spatiotemporal_grid itself.

tests/test_no_leakage.py compares the committed panel against an independent
recomputation. That catches a panel that has drifted from the raw catalogue, but
it never calls the builder, so a leak introduced into the builder would go
unnoticed until someone rebuilt the panel and committed the result. The folder's
rules make this file a precondition for any rebuild - and a rebuild is exactly
what the completeness work needs.

Every test here calls build_spatiotemporal_grid on a synthetic catalogue and
asserts a property of what comes back, so a defect has to survive the function's
actual behaviour rather than a re-implementation of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.grid_builder import W_MAX, build_spatiotemporal_grid

LAGGED_FEATURES = [
    "Y_lag1",
    "mag_max_lag1",
    "mag_min_lag1",
    "mag_max_roll4",
    "count_roll12",
    "energy_roll8",
    "weeks_since_m45",
]


def _catalog(extra: list[dict] | None = None) -> pd.DataFrame:
    """Two cells inside the configured bbox, 40 weeks, a handful of events.

    The first and last events are fixed so that adding an event in the middle
    cannot move the panel's week range - otherwise every row would shift and the
    comparison below would compare different weeks with each other.
    """
    origin = pd.Timestamp("2015-01-05", tz="UTC")
    rows = [
        {"time": origin, "latitude": 39.0, "longitude": 66.0, "mag": 4.0},
        {"time": origin + pd.Timedelta(weeks=39), "latitude": 39.0, "longitude": 66.0, "mag": 4.0},
        {"time": origin + pd.Timedelta(weeks=39), "latitude": 42.0, "longitude": 69.0, "mag": 4.0},
    ]
    for w in (5, 9, 14, 21, 28):
        rows.append({"time": origin + pd.Timedelta(weeks=w, days=1),
                     "latitude": 39.0, "longitude": 66.0, "mag": 4.2})
    for w in (7, 18, 33):
        rows.append({"time": origin + pd.Timedelta(weeks=w, days=2),
                     "latitude": 42.0, "longitude": 69.0, "mag": 4.1})
    if extra:
        rows.extend(extra)
    return pd.DataFrame(rows)


def _panel(extra: list[dict] | None = None) -> pd.DataFrame:
    return build_spatiotemporal_grid(_catalog(extra)).sort_values(
        ["cell_id", "week"], kind="mergesort"
    ).reset_index(drop=True)


def test_features_do_not_see_their_own_week_but_do_see_the_previous_one():
    """The property the whole panel rests on.

    An event added in week t may change the TARGET at week t - that is what a
    target is - but must not touch a single feature at week t, because every
    feature is supposed to be built from strictly earlier weeks. At week t+1 the
    features must react, otherwise they are not carrying the history at all.
    """
    origin = pd.Timestamp("2015-01-05", tz="UTC")
    injected_week = 20
    extra = [{"time": origin + pd.Timedelta(weeks=injected_week, days=3),
              "latitude": 39.0, "longitude": 66.0, "mag": 5.4}]

    before = _panel()
    after = _panel(extra)

    assert list(before["week"]) == list(after["week"]), "the week axis moved"
    assert list(before["cell_id"]) == list(after["cell_id"])

    week_t = pd.Timestamp(origin + pd.Timedelta(weeks=injected_week)).tz_localize(None).to_period("W").start_time
    cell = "38.00_65.00"
    mask_t = (before["cell_id"] == cell) & (before["week"] == week_t)
    assert mask_t.sum() == 1, "the injected week is missing from the panel"

    assert int(after.loc[mask_t, "Y"].iloc[0]) == int(before.loc[mask_t, "Y"].iloc[0]) + 1

    for col in LAGGED_FEATURES:
        assert after.loc[mask_t, col].iloc[0] == before.loc[mask_t, col].iloc[0], (
            f"{col} at week t changed when the target at week t was tampered with - "
            "the feature is reading its own week"
        )

    next_week = week_t + pd.Timedelta(weeks=1)
    mask_next = (before["cell_id"] == cell) & (before["week"] == next_week)
    assert mask_next.sum() == 1
    changed = [c for c in LAGGED_FEATURES
               if after.loc[mask_next, c].iloc[0] != before.loc[mask_next, c].iloc[0]]
    assert changed, "no feature at week t+1 reacted - the history is not being carried"
    assert "Y_lag1" in changed


def test_quiet_weeks_become_zero_rows_rather_than_gaps():
    panel = _panel()
    counts = panel.groupby("cell_id").size()
    assert counts.nunique() == 1, f"cells have different numbers of weeks: {counts.to_dict()}"
    assert (panel["Y"] >= 0).all()
    assert (panel["Y"] == 0).any(), "a panel of only non-empty weeks is not a complete panel"
    assert not panel[LAGGED_FEATURES].isna().any().any()


def test_warmup_rows_are_dropped_per_cell():
    """The first W_MAX weeks of each cell have no usable history and must go."""
    full_weeks = 40
    panel = _panel()
    per_cell = panel.groupby("cell_id").size().unique()
    assert len(per_cell) == 1
    assert int(per_cell[0]) == full_weeks - W_MAX


def test_weeks_since_major_event_never_sees_the_current_week():
    """A magnitude 4.5+ event must not reset the counter in its own week."""
    origin = pd.Timestamp("2015-01-05", tz="UTC")
    big_week = 25
    extra = [{"time": origin + pd.Timedelta(weeks=big_week, days=1),
              "latitude": 39.0, "longitude": 66.0, "mag": 6.0}]
    panel = _panel(extra)

    cell = "38.00_65.00"
    week_t = pd.Timestamp(origin + pd.Timedelta(weeks=big_week)).tz_localize(None).to_period("W").start_time
    at_t = panel.loc[(panel.cell_id == cell) & (panel.week == week_t), "weeks_since_m45"].iloc[0]
    at_t1 = panel.loc[(panel.cell_id == cell) & (panel.week == week_t + pd.Timedelta(weeks=1)),
                      "weeks_since_m45"].iloc[0]

    assert at_t > at_t1, (
        f"weeks_since_m45 is {at_t} in the event's own week and {at_t1} after it - "
        "the counter reset before the event had happened"
    )
    assert at_t1 == 0.0, f"the week after a major event should read 0, got {at_t1}"


def test_every_event_outside_the_warmup_is_counted_exactly_once():
    """Y must total the events that survive the warm-up cut - no more, no less.

    The expected number is counted straight from the catalogue here, not derived
    from the panel: deriving it would make the assertion true by construction and
    test nothing. A double-count or a dropped event changes only this total,
    while every per-row check above still passes.
    """
    catalog = _catalog()
    panel = _panel()

    weeks = catalog["time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("W").dt.start_time
    all_weeks = pd.date_range(weeks.min(), weeks.max(), freq="W-MON")
    warmup = set(all_weeks[:W_MAX])

    expected = int((~weeks.isin(warmup)).sum())
    assert expected > 0, "the fixture puts every event in the warm-up - it proves nothing"
    assert int(panel["Y"].sum()) == expected


def test_a_cell_that_starts_late_gets_zero_rows_for_the_earlier_weeks():
    """The week axis is global; a quiet cell must be padded, not shifted.

    ``full_weeks`` spans the whole catalogue, not each cell's own history. A cell
    whose first event lands in the middle of the period must still carry rows for
    the weeks before it, filled with zeros. If those rows went missing the cell's
    series would silently start late and every lag would line up against the
    wrong week - which is exactly the failure mode a rebuild at a higher
    magnitude threshold invites, because raising the threshold empties the early
    history of the quiet cells.
    """
    origin = pd.Timestamp("2015-01-05", tz="UTC")
    late_cell_event = {"time": origin + pd.Timedelta(weeks=30, days=1),
                       "latitude": 44.0, "longitude": 78.0, "mag": 4.3}
    panel = _panel([late_cell_event])

    late_cell = "44.00_77.00"
    assert late_cell in set(panel["cell_id"]), "the late-starting cell is missing entirely"

    per_cell = panel.groupby("cell_id").size()
    assert per_cell.nunique() == 1, (
        f"cells carry different numbers of weeks: {per_cell.to_dict()} - "
        "a late-starting cell was not padded to the global week axis"
    )

    late = panel[panel["cell_id"] == late_cell].sort_values("week")
    assert late["Y"].iloc[0] == 0
    assert late["Y"].sum() == 1, "the single event of the late cell was lost or duplicated"
    first_nonzero = late.loc[late["Y"] > 0, "week"].iloc[0]
    assert first_nonzero > late["week"].iloc[0], "the cell's series starts at its first event"


def test_one_row_per_cell_and_week():
    """A merge on the wrong keys duplicates rows without failing anything else."""
    panel = _panel()
    pairs = panel[["cell_id", "week"]]
    assert not pairs.duplicated().any(), "the panel has more than one row for some (cell, week)"
    assert len(panel) == panel["cell_id"].nunique() * panel["week"].nunique()


def test_output_carries_exactly_the_declared_columns():
    """Intermediate columns must not leak into the panel that gets committed."""
    from src.grid_builder import FINAL_COLUMNS

    panel = _panel()
    assert list(panel.columns) == FINAL_COLUMNS
    for leaked in ("mag_max", "mag_min", "energy_sum", "energy"):
        assert leaked not in panel.columns, f"intermediate column {leaked} reached the output"
