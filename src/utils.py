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
    Create standard project directories if they do not exist.

    Creates ``data/raw``, ``data/processed``, ``notebooks``, and ``outputs/figures``
    under the project root.
    """
    root = get_project_root()
    for relative in (
        Path("data") / "raw",
        Path("data") / "processed",
        Path("notebooks"),
        Path("outputs") / "figures",
    ):
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
