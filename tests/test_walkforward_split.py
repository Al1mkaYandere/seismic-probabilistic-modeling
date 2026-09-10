"""The walk-forward validation block must be the tail of the time axis.

``src/validation.py`` used to cut validation as the last 15% of training ROWS.
The panel on disk is sorted by ``["cell_id", "week"]`` (grid_builder.py), so
that cut carved out whole cells instead of recent weeks: for the 2023 fold only
3 of 17 cells reached validation, two of them absent from training altogether,
leaving their embeddings untrained while the model still had to predict them.

These tests call ``validation.temporal_validation_split`` directly, on the real
panel. An earlier version of this file re-implemented the split arithmetic in
its own fixture; it passed whatever ``validation.py`` did, including with the
defect restored, and therefore tested nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.validation import VAL_FRACTION, temporal_validation_split

PANEL_PATH = config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    """The panel exactly as run_walk_forward reads and orders it."""
    df = pd.read_csv(PANEL_PATH)
    df["week"] = pd.to_datetime(df["week"])
    return df.sort_values("week", kind="mergesort").reset_index(drop=True)


@pytest.mark.parametrize("test_year", [2018, 2019, 2020, 2021, 2022, 2023])
def test_validation_block_is_latest_weeks_and_keeps_every_cell(panel: pd.DataFrame, test_year: int):
    """Two properties, for every walk-forward fold.

    1. Every cell present in the fold's training data is also present in
       validation. Otherwise early stopping is steered by a spatial subset.
    2. Every validation week is strictly later than every training week.
       Otherwise the cut is not chronological at all.
    """
    df_train = panel.loc[panel["week"].dt.year < test_year].reset_index(drop=True)
    weeks = df_train["week"].to_numpy()

    tr_idx, val_idx = temporal_validation_split(weeks, VAL_FRACTION)

    assert len(tr_idx) > 0 and len(val_idx) > 0

    train_cells = set(df_train.loc[tr_idx, "cell_id"])
    val_cells = set(df_train.loc[val_idx, "cell_id"])
    assert train_cells - val_cells == set(), (
        f"{test_year}: cells in training but never validated: "
        f"{sorted(train_cells - val_cells)}"
    )
    assert val_cells - train_cells == set(), (
        f"{test_year}: cells validated but never trained on: "
        f"{sorted(val_cells - train_cells)}"
    )

    assert weeks[val_idx].min() > weeks[tr_idx].max(), (
        f"{test_year}: validation starts at {weeks[val_idx].min()}, which is not "
        f"after the last training week {weeks[tr_idx].max()}"
    )


def test_validation_block_holds_the_requested_share_of_weeks(panel: pd.DataFrame):
    """The block is a share of unique WEEKS, not of rows."""
    df_train = panel.loc[panel["week"].dt.year < 2023].reset_index(drop=True)
    weeks = df_train["week"].to_numpy()
    _, val_idx = temporal_validation_split(weeks, VAL_FRACTION)

    n_unique = len(np.unique(weeks))
    expected = max(1, int(np.floor(VAL_FRACTION * n_unique)))
    assert len(np.unique(weeks[val_idx])) == expected
