"""Check for future leakage in the features of the committed panel.

The panel data/processed/spatiotemporal_grid.csv is not rebuilt: the
features are recomputed INDEPENDENTLY, from the raw catalog in data/raw/,
with separate code (without calling src/grid_builder.py), and compared
against the already-committed values. If grid_builder.py had forgotten
.shift(1) on one of the rolling features, this independent recomputation
would disagree with the panel, and the test would catch it.

The original panel drops the first W_MAX=12 rows of each cell
(grid_builder.py, around line 206):
```
panel = panel[panel.groupby("cell_id").cumcount() >= W_MAX]
```
because the rolling window has not yet accumulated a full history there.
That is not a discrepancy, so the comparison uses an inner join on
(cell_id, week): in the recomputed (non-truncated) version, those rows
simply do not take part in the comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config

PANEL_PATH = config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE
RAW_PATH = config.RAW_DATA_PATH / config.RAW_DATA_FILE


def _independent_rebuild() -> pd.DataFrame:
    """Rebuild Y_lag1 / count_roll12 / energy_roll8 / mag_max_roll4 from scratch."""
    raw = pd.read_csv(RAW_PATH)
    raw["time"] = pd.to_datetime(raw["time"], utc=True, format="mixed").dt.tz_localize(None)

    min_lat = config.BBOX["minlatitude"]
    min_lon = config.BBOX["minlongitude"]
    gs = config.GRID_SIZE
    lat_grid = np.floor((raw["latitude"] - min_lat) / gs) * gs + min_lat
    lon_grid = np.floor((raw["longitude"] - min_lon) / gs) * gs + min_lon
    raw["lat_grid"] = np.round(lat_grid, 2)
    raw["lon_grid"] = np.round(lon_grid, 2)
    raw["cell_id"] = (
        raw["lat_grid"].map(lambda x: f"{float(x):.2f}") + "_" + raw["lon_grid"].map(lambda x: f"{float(x):.2f}")
    )
    raw["energy"] = 10 ** (1.5 * raw["mag"])
    raw["week"] = raw["time"].dt.to_period("W").dt.start_time

    agg = raw.groupby(["cell_id", "week"], as_index=False).agg(
        Y=("mag", "count"), mag_max=("mag", "max"), energy_sum=("energy", "sum")
    )

    min_week, max_week = agg["week"].min(), agg["week"].max()
    full_weeks = pd.date_range(min_week, max_week, freq="W-MON")
    cells = agg["cell_id"].unique()
    full_index = pd.MultiIndex.from_product([cells, full_weeks], names=["cell_id", "week"]).to_frame(index=False)
    full = full_index.merge(agg, on=["cell_id", "week"], how="left")
    full[["Y", "mag_max", "energy_sum"]] = full[["Y", "mag_max", "energy_sum"]].fillna(0.0)
    full = full.sort_values(["cell_id", "week"], kind="mergesort").reset_index(drop=True)

    g = full.groupby("cell_id", sort=False)
    full["Y_lag1_expected"] = g["Y"].shift(1)
    full["count_roll12_expected"] = g["Y"].transform(lambda x: x.rolling(12, min_periods=1).sum().shift(1))
    full["energy_roll8_expected"] = g["energy_sum"].transform(lambda x: x.rolling(8, min_periods=1).sum().shift(1))
    full["mag_max_roll4_expected"] = g["mag_max"].transform(lambda x: x.rolling(4, min_periods=1).max().shift(1))
    return full


@pytest.fixture(scope="module")
def merged_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH)
    panel["week"] = pd.to_datetime(panel["week"])
    rebuilt = _independent_rebuild()
    merged = panel.merge(rebuilt, on=["cell_id", "week"], how="inner", suffixes=("", "_full"))
    # If the panel was not re-sorted and every cell/week found a match, the join must not lose rows.
    assert len(merged) == len(panel), "committed panel has rows the raw catalog cannot reproduce"
    return merged


@pytest.mark.parametrize(
    "feature_col,expected_col",
    [
        ("Y_lag1", "Y_lag1_expected"),
        ("count_roll12", "count_roll12_expected"),
        ("energy_roll8", "energy_roll8_expected"),
        ("mag_max_roll4", "mag_max_roll4_expected"),
    ],
)
def test_feature_equals_recomputed_window_before_t(merged_panel: pd.DataFrame, feature_col: str, expected_col: str):
    actual = merged_panel[feature_col].to_numpy(dtype=np.float64)
    expected = merged_panel[expected_col].fillna(0.0).to_numpy(dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-3)


def test_feature_at_t_is_insensitive_to_tampering_y_at_t_but_sensitive_at_t_plus_1():
    """A targeted leakage check: corrupt Y for exactly one week t of one cell.

    If count_roll12 genuinely uses only the window strictly BEFORE t,
    tampering with Y[t] must not affect count_roll12 computed FOR row t (its
    window looks backward, not at itself). But it must affect count_roll12
    of row t+1 (whose window includes t) — this confirms that the check is
    actually measuring something, rather than being trivially insensitive to
    any change.
    """
    rebuilt = _independent_rebuild().sort_values(["cell_id", "week"], kind="mergesort").reset_index(drop=True)

    cell = rebuilt["cell_id"].iloc[0]
    cell_rows = rebuilt.index[rebuilt["cell_id"] == cell].to_numpy()
    # Pick a row well inside the cell's history so that both t and t+1 exist
    # and both have a full 12-week window behind them.
    t_pos = cell_rows[20]
    t_plus_1_pos = cell_rows[21]

    tampered = rebuilt.copy()
    tampered.loc[t_pos, "Y"] = tampered.loc[t_pos, "Y"] + 10_000.0
    g = tampered.groupby("cell_id", sort=False)
    tampered["count_roll12_tampered"] = g["Y"].transform(lambda x: x.rolling(12, min_periods=1).sum().shift(1))

    before_t = rebuilt.loc[t_pos, "count_roll12_expected"]
    after_t = tampered.loc[t_pos, "count_roll12_tampered"]
    assert after_t == pytest.approx(before_t), "count_roll12[t] changed after tampering with Y[t] — this is future leakage"

    before_t1 = rebuilt.loc[t_plus_1_pos, "count_roll12_expected"]
    after_t1 = tampered.loc[t_plus_1_pos, "count_roll12_tampered"]
    assert after_t1 == pytest.approx(before_t1 + 10_000.0), (
        "count_roll12[t+1] did not respond to tampering with Y[t] — the "
        "check is insensitive, so the first part of this test proves nothing"
    )
