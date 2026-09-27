"""Where the blind test is allowed to write.

This rule lives in its own module so that it can be tested without importing
the driver, which pulls in torch and the whole pipeline: the driver has to
resolve its output directory before those imports happen, because several
pipeline modules bind their own output paths the moment they are imported.
"""
from __future__ import annotations

from pathlib import Path

PUBLISHED_DIRECTORIES: tuple[str, ...] = ("outputs", "outputs-mc45", "data")
"""Directories a blind-test run must never write into: the published tables of
the two main runs, and the committed catalogues and panels."""


def resolve_output_dir(requested: str | None, repo: Path) -> Path:
    """The directory to write into, or ``ValueError`` if it is a published one.

    ``requested`` is the value of ``SPM_BLIND_DIR``; ``None`` means the default,
    ``blind/outputs`` next to the driver.
    """
    out = Path(requested).resolve() if requested else (repo / "blind" / "outputs").resolve()
    for name in PUBLISHED_DIRECTORIES:
        published = (repo / name).resolve()
        if out == published or out.is_relative_to(published):
            raise ValueError(
                f"refusing to write into {out}: it is inside {published}, which "
                "holds published results. Point SPM_BLIND_DIR somewhere else."
            )
    return out
