"""The switches main.py reads from the environment: offline run, pinned weekly axis, seed.

They exist so that the runs reported in the paper can be repeated from main.py
alone; unset, the pipeline behaves as it did before they were added.
"""
from __future__ import annotations

import pandas as pd
import pytest

import main


def test_an_offline_run_never_downloads(monkeypatch):
    def forbidden() -> None:
        raise AssertionError("SPM_OFFLINE=1 must not download anything")

    monkeypatch.setattr(main, "run_ingestion_pipeline", forbidden)
    monkeypatch.setenv("SPM_OFFLINE", "1")
    main._ingest_unless_offline()


def test_without_the_switch_the_catalogue_is_downloaded(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(main, "run_ingestion_pipeline", lambda: calls.append(1))
    monkeypatch.delenv("SPM_OFFLINE", raising=False)
    main._ingest_unless_offline()
    assert calls == [1]


def test_the_axis_follows_the_events_unless_pinned(monkeypatch):
    monkeypatch.delenv("SPM_PIN_AXIS", raising=False)
    assert main._week_bounds() is None


def test_the_pinned_axis_is_the_weekly_span_of_the_3_0_catalogue(monkeypatch):
    monkeypatch.setenv("SPM_PIN_AXIS", "1")
    assert main._week_bounds() == (pd.Timestamp("2009-12-28"), pd.Timestamp("2023-12-25"))


@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("7", 7), ("2024", 2024)])
def test_the_seed_comes_from_the_environment(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("SPM_SEED", raising=False)
    else:
        monkeypatch.setenv("SPM_SEED", value)
    assert main._training_seed() == expected
