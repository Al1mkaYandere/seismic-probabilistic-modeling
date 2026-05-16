"""Build spatiotemporal panel: grid cells × weeks with lags and zero-filled gaps."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

W_MAX: int = 12

FINAL_COLUMNS: list[str] = [
    "cell_id",
    "week",
    "lat_grid",
    "lon_grid",
    "Y_lag1",
    "mag_max_lag1",
    "mag_min_lag1",
    "mag_max_roll4",
    "count_roll12",
    "energy_roll8",
    "weeks_since_m45",
    "Y",
]


def _weeks_since_major_event(series: pd.Series) -> pd.Series:
    """
    Compute weeks since last major event (mag >= 4.5) with strict lagging.

    The returned feature is shifted by 1 week to prevent look-ahead leakage.
    If a cell has no prior major event, value is filled with 500.0.
    """
    idx = np.arange(series.shape[0], dtype=np.float64)
    major = (series.to_numpy(dtype=np.float64, copy=False) >= 4.5)
    last_major_idx = np.where(major, idx, np.nan)
    last_major_idx = pd.Series(last_major_idx, index=series.index).ffill().to_numpy()
    dist = idx - last_major_idx
    dist = np.where(np.isnan(last_major_idx), np.nan, dist)
    out = pd.Series(dist, index=series.index, dtype=np.float64).shift(1)
    return out.fillna(500.0)


def load_raw_data() -> pd.DataFrame:
    """
    Load cleaned USGS events from the configured raw CSV path.

    Returns
    -------
    pd.DataFrame
        Raw event table with at least time, latitude, longitude, mag.
    """
    path = config.RAW_DATA_PATH / config.RAW_DATA_FILE
    logger.info("Loading raw data from %s", path.resolve())
    df = pd.read_csv(path)
    return df


def build_spatiotemporal_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin events into anchored grid cells and weeks, complete the panel, add lag features.

    Parameters
    ----------
    df : pd.DataFrame
        Event-level data with columns including time, latitude, longitude, mag.

    Returns
    -------
    pd.DataFrame
        Panel with columns in ``FINAL_COLUMNS`` order.
    """
    work = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(work["time"]):
        work["time"] = pd.to_datetime(work["time"], utc=True, format="mixed")
    elif work["time"].dt.tz is None:
        work["time"] = work["time"].dt.tz_localize("UTC")
    else:
        work["time"] = work["time"].dt.tz_convert("UTC")

    min_lat = config.BBOX["minlatitude"]
    min_lon = config.BBOX["minlongitude"]
    gs = config.GRID_SIZE

    lat = work["latitude"].to_numpy(dtype=float, copy=False)
    lon = work["longitude"].to_numpy(dtype=float, copy=False)
    lat_grid = np.floor((lat - min_lat) / gs) * gs + min_lat
    lon_grid = np.floor((lon - min_lon) / gs) * gs + min_lon
    work["lat_grid"] = np.round(lat_grid, 2)
    work["lon_grid"] = np.round(lon_grid, 2)

    work["cell_id"] = (
        work["lat_grid"].map(lambda x: f"{float(x):.2f}")
        + "_"
        + work["lon_grid"].map(lambda x: f"{float(x):.2f}")
    )
    work["energy"] = 10 ** (1.5 * work["mag"])

    t_for_week = work["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    work["week"] = t_for_week.dt.to_period("W").dt.start_time

    agg = (
        work.groupby(["cell_id", "lat_grid", "lon_grid", "week"], as_index=False)
        .agg(
            Y=("mag", "count"),
            mag_max=("mag", "max"),
            mag_min=("mag", "min"),
            energy_sum=("energy", "sum"),
        )
    )

    expected_event_total = int(agg["Y"].sum())
    if expected_event_total != len(work):
        logger.warning(
            "Event count mismatch: agg Y sum=%s vs len(work)=%s",
            expected_event_total,
            len(work),
        )

    min_week = agg["week"].min()
    max_week = agg["week"].max()
    week_tz = getattr(min_week, "tz", None)
    full_weeks = pd.date_range(
        start=min_week,
        end=max_week,
        freq="W-MON",
        tz=week_tz,
    )

    logger.info("Unique weeks in events: %s", agg["week"].nunique())
    logger.info("Unique weeks in grid: %s", len(full_weeks))

    cells = agg[["cell_id", "lat_grid", "lon_grid"]].drop_duplicates(
        subset=["cell_id"],
        keep="first",
    )

    full_index = pd.MultiIndex.from_product(
        [cells["cell_id"].values, full_weeks],
        names=["cell_id", "week"],
    )
    base = full_index.to_frame(index=False)
    base = base.merge(cells, on="cell_id", how="left")

    panel = base.merge(
        agg,
        on=["cell_id", "lat_grid", "lon_grid", "week"],
        how="left",
    )

    panel["Y"] = panel["Y"].fillna(0.0)
    panel["mag_max"] = panel["mag_max"].fillna(0.0)
    panel["mag_min"] = panel["mag_min"].fillna(0.0)
    panel["energy_sum"] = panel["energy_sum"].fillna(0.0)

    y_sum = float(panel["Y"].sum())
    logger.info("Total earthquakes (Y sum) after merge: %s", y_sum)

    if y_sum == 0:
        raise ValueError(
            "Total earthquakes Y sum after merge is 0; check week alignment (e.g. W vs W-MON)."
        )
    if int(y_sum) != expected_event_total:
        raise ValueError(
            f"Total earthquakes Y sum after merge ({int(y_sum)}) must equal "
            f"aggregated event count ({expected_event_total})."
        )

    panel = panel.sort_values(["cell_id", "week"], kind="mergesort").reset_index(
        drop=True
    )

    grp = panel.groupby("cell_id", sort=False)
    panel["mag_max_lag1"] = grp["mag_max"].shift(1)
    panel["mag_min_lag1"] = grp["mag_min"].shift(1)
    panel["Y_lag1"] = grp["Y"].shift(1)
    panel["mag_max_roll4"] = grp["mag_max"].transform(
        lambda x: x.rolling(4, min_periods=1).max().shift(1)
    )
    panel["count_roll12"] = grp["Y"].transform(
        lambda x: x.rolling(12, min_periods=1).sum().shift(1)
    )
    panel["energy_roll8"] = grp["energy_sum"].transform(
        lambda x: x.rolling(8, min_periods=1).sum().shift(1)
    )
    panel["weeks_since_m45"] = grp["mag_max"].transform(_weeks_since_major_event)

    panel["Y_lag1"] = panel["Y_lag1"].fillna(0.0).astype(np.float64)
    panel["mag_max_lag1"] = panel["mag_max_lag1"].fillna(0.0).astype(np.float64)
    panel["mag_min_lag1"] = panel["mag_min_lag1"].fillna(0.0).astype(np.float64)
    panel["mag_max_roll4"] = panel["mag_max_roll4"].fillna(0.0).astype(np.float64)
    panel["count_roll12"] = panel["count_roll12"].fillna(0.0).astype(np.float64)
    panel["energy_roll8"] = panel["energy_roll8"].fillna(0.0).astype(np.float64)
    panel["weeks_since_m45"] = panel["weeks_since_m45"].fillna(500.0).astype(np.float64)
    panel["lat_grid"] = panel["lat_grid"].astype(np.float64)
    panel["lon_grid"] = panel["lon_grid"].astype(np.float64)
    panel["Y"] = panel["Y"].astype(np.int64)

    n_before = len(panel)
    panel = panel[panel.groupby("cell_id").cumcount() >= W_MAX].reset_index(drop=True)
    logger.info(
        "Dropped %d warmup rows (t < %d per cell), %d rows remain",
        n_before - len(panel), W_MAX, len(panel),
    )

    out = panel[FINAL_COLUMNS].copy()
    return out


def save_processed_data(df: pd.DataFrame) -> None:
    """
    Write the processed spatiotemporal grid to the configured processed CSV path.

    Parameters
    ----------
    df : pd.DataFrame
        Final panel to persist.
    """
    config.PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    out_path = config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE
    df.to_csv(out_path, index=False)
    logger.info("Saved processed data to %s", out_path.resolve())
