"""Check the walk-forward split for the 2023 test year (known issue A3).

There is no need to train the DL models in a unit test — the row slice the
model receives is fully determined by how src/validation.py reads the CSV
and cuts the last 15% of rows (see _dl_train_predict, around lines 101-104,
and run_walk_forward, around line 179). This test reproduces EXACTLY that
arithmetic on the committed panel, without training a network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config

PANEL_PATH = config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE
TEST_YEAR = 2023
VAL_FRACTION = 0.15  # see validation.py::_dl_train_predict, around line 102


@pytest.fixture(scope="module")
def train_val_split() -> dict:
    df = pd.read_csv(PANEL_PATH)  # src/validation.py::run_walk_forward, line 179 — no re-sorting
    df["week"] = pd.to_datetime(df["week"])

    train_mask = df["week"].dt.year < TEST_YEAR
    df_train = df.loc[train_mask].dropna(subset=["Y"]).reset_index(drop=True)

    n_val = max(1, int(np.floor(VAL_FRACTION * len(df_train))))
    val_df = df_train.iloc[-n_val:]

    all_cells = set(df["cell_id"].unique())
    return {"df": df, "df_train": df_train, "val_df": val_df, "all_cells": all_cells, "n_val": n_val}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known issue A3: the slice is taken by row position, while the file "
        "is sorted by cell (grid_builder.py saves "
        "panel.sort_values(['cell_id', 'week'])). The last 15% of "
        "df_train's rows are a handful of cells in full (in cell_id "
        "alphabetical order), not the most recent weeks across all cells. "
        "Verified empirically on the real panel: only 3 of 17 cells end up "
        "in the 2023 validation split, and the slice starts in 2010 rather "
        "than in the last few months of the training period."
    ),
)
def test_validation_slice_has_all_cells_and_is_latest_by_time(train_val_split: dict):
    """Two properties promised by the comment in validation.py ('chronological
    slice — the last 15% of training rows'): (1) validation must include
    every cell of the panel, (2) its weeks must be the most recent ones in
    the training period.
    """
    val_cells = set(train_val_split["val_df"]["cell_id"].unique())
    all_cells = train_val_split["all_cells"]
    assert val_cells == all_cells, (
        f"validation contains {len(val_cells)} of {len(all_cells)} cells: "
        f"missing {sorted(all_cells - val_cells)}"
    )

    df_train = train_val_split["df_train"]
    val_df = train_val_split["val_df"]
    n_val = train_val_split["n_val"]
    true_latest_by_time = df_train.sort_values("week", kind="mergesort").iloc[-n_val:]
    assert val_df["week"].min() == true_latest_by_time["week"].min(), (
        f"the current slice starts at week {val_df['week'].min()}, but the "
        f"actual last {n_val} rows of the training part by time start at "
        f"{true_latest_by_time['week'].min()}"
    )
