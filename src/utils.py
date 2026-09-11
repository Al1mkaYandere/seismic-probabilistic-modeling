"""Project utilities: path resolution, logging, and directory bootstrap."""

import logging
import sys
from pathlib import Path


def get_project_root() -> Path:
    """
    Resolve the project root directory dynamically from this module's location.

    Returns
    -------
    Path
        Absolute path to the repository root (parent of ``src``).
    """
    return Path(__file__).resolve().parent.parent


def setup_logging() -> None:
    """
    Configure root logging: INFO level, timestamps, and logger name in the format.
    """
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def ensure_project_directories() -> None:
    """
    Create the directories the run is configured to use, if they do not exist.

    Every path comes from ``config`` rather than from a literal, so a run
    pointed at another output directory creates THAT directory instead of
    quietly adding an empty one inside the published tree. ``notebooks`` is
    the one directory with no configured location; it is not written to by
    the pipeline and stays at the project root.
    """
    from src import config

    for path in (
        config.RAW_DATA_PATH,
        config.PROCESSED_DATA_PATH,
        get_project_root() / "notebooks",
        config.OUTPUT_DIR,
        config.FIGURES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
