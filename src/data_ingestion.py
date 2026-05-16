"""USGS earthquake data: download, validate, clean, and persist to CSV."""

from __future__ import annotations

import io
import logging
from typing import Any

import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = [
    "time",
    "latitude",
    "longitude",
    "depth",
    "mag",
    "place",
]


def _build_request_params() -> dict[str, Any]:
    """
    Build query parameters for the USGS FDSN event CSV endpoint.

    Returns
    -------
    dict[str, Any]
        Key-value pairs passed to ``requests.get`` as ``params``.
    """
    params: dict[str, Any] = {
        "format": "csv",
        "starttime": config.START_DATE,
        "endtime": config.END_DATE,
        "minmagnitude": config.MIN_MAGNITUDE,
        "minlatitude": config.BBOX["minlatitude"],
        "maxlatitude": config.BBOX["maxlatitude"],
        "minlongitude": config.BBOX["minlongitude"],
        "maxlongitude": config.BBOX["maxlongitude"],
    }
    return params


def fetch_earthquake_data() -> pd.DataFrame:
    """
    Download earthquake events from USGS as CSV and return a DataFrame.

    Returns
    -------
    pd.DataFrame
        Raw table as returned by the API (CSV parsed by pandas).

    Raises
    ------
    requests.RequestException
        On network errors or non-success HTTP status after ``raise_for_status``.
    """
    params = _build_request_params()
    logger.info("Request URL: %s", config.BASE_URL)
    logger.info("Request parameters: %s", params)

    try:
        response = requests.get(
            config.BASE_URL,
            params=params,
            timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
        logger.info("HTTP status: %s", response.status_code)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.exception("HTTP request failed: %s", exc)
        raise

    text = response.text
    df = pd.read_csv(io.StringIO(text))
    n_rows = len(df)
    logger.info("Number of rows downloaded: %s", n_rows)
    return df


def clean_earthquake_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict columns, drop invalid rows, cast types, and sort by time.

    Parameters
    ----------
    df : pd.DataFrame
        Raw USGS CSV data.

    Returns
    -------
    pd.DataFrame
        Cleaned and sorted data with only required columns.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    out = out.dropna(subset=["time", "latitude", "longitude", "mag"])

    for col in ("latitude", "longitude", "depth", "mag"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude", "mag"])

    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.dropna(subset=["time"])

    out = out.sort_values("time", ascending=True).reset_index(drop=True)

    logger.info("Number of rows after cleaning: %s", len(out))
    return out


def _validate_earthquake_data(df: pd.DataFrame) -> None:
    """
    Validate non-emptiness and geographic/magnitude constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned earthquake data.

    Raises
    ------
    AssertionError
        If any validation check fails.
    """
    assert not df.empty, "DataFrame must not be empty after cleaning"

    lat_min, lat_max = config.BBOX["minlatitude"], config.BBOX["maxlatitude"]
    lon_min, lon_max = config.BBOX["minlongitude"], config.BBOX["maxlongitude"]

    assert df["latitude"].between(lat_min, lat_max, inclusive="both").all(), (
        "All latitudes must be within configured bbox"
    )
    assert df["longitude"].between(lon_min, lon_max, inclusive="both").all(), (
        "All longitudes must be within configured bbox"
    )
    assert (df["mag"] >= config.MIN_MAGNITUDE).all(), (
        "All magnitudes must be >= configured minimum"
    )


def save_data(df: pd.DataFrame) -> None:
    """
    Write the DataFrame to the configured raw CSV path (overwrite if exists).

    Parameters
    ----------
    df : pd.DataFrame
        Data to persist.
    """
    config.RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    out_path = config.RAW_DATA_PATH / config.RAW_DATA_FILE
    df.to_csv(out_path, index=False)
    logger.info("Output file path: %s", out_path.resolve())


def run_ingestion_pipeline() -> None:
    """
    End-to-end: fetch, clean, validate, and save USGS earthquake data.
    """
    logger.info("Starting ingestion pipeline")
    raw = fetch_earthquake_data()
    cleaned = clean_earthquake_data(raw)
    _validate_earthquake_data(cleaned)
    save_data(cleaned)
