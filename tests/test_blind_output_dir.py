"""The blind test must not be able to write into a published directory.

The driver rebuilds Section 4.4's table from scratch, and the obvious mistake -
pointing it at ``outputs/`` and overwriting the run every other number in the
paper comes from - is the one this guards. It tests the rule, not the driver:
importing the driver would import torch and the whole pipeline to check one
path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from blind.output_dir import resolve_output_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_a_directory_of_its_own_is_accepted(tmp_path) -> None:
    assert resolve_output_dir(str(tmp_path), REPO_ROOT) == tmp_path.resolve()


def test_the_default_is_outside_every_published_directory() -> None:
    assert resolve_output_dir(None, REPO_ROOT) == (REPO_ROOT / "blind" / "outputs").resolve()


@pytest.mark.parametrize("target", ["outputs", "outputs-mc45", "data",
                                    "outputs/figures", "data/processed", "data/raw"])
def test_a_published_directory_is_refused(target: str) -> None:
    with pytest.raises(ValueError, match="refusing to write"):
        resolve_output_dir(str(REPO_ROOT / target), REPO_ROOT)
