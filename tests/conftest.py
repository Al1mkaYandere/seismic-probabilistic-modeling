"""Shared pytest configuration.

Makes the ``src`` package importable regardless of which directory pytest is
run from and which import mode is in effect (tests/ has no rootdir
__init__.py, so without this explicit insertion pytest may not add the repo
root to sys.path).

It also keeps tests out of the published output directory. That is not a
hypothetical: twice in one session a test wrote into ``outputs/`` - once
overwriting the year-by-year results with stub predictions, once the seed-sweep
table - and both times the damage was found afterwards, by the frozen-numbers
check, rather than at the moment it happened.

The cause is not the individual tests. Several modules bind their output paths at
import time, on purpose (see the docstring in ``src/config.py``: the environment
has to be read before anything else runs), so patching ``config.OUTPUT_DIR`` does
not reach them. Every test calling pipeline code therefore had to remember its own
list of paths to redirect, and a list a person has to remember is a list a person
forgets.

Two things here, deliberately separate:

* ``isolated_outputs`` finds the paths instead of listing them, so a new output
  file is covered the day it is added. Request it in any test that calls code
  which writes.
* ``_published_outputs_untouched`` watches every test and fails the one that wrote
  into the published directory. It changes no behaviour, so tests that legitimately
  READ from ``outputs/`` keep working; it is the net under the fixture, for the
  tests that forget to ask for it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402

PUBLISHED_OUTPUTS: Path = Path(config.OUTPUT_DIR)
"""The output directory as configured when pytest started, before any patching."""


def _fingerprint(root: Path) -> dict[str, tuple[int, int]]:
    """Size and modification time of everything under ``root``.

    Cheaper than hashing on every test, and strictly more sensitive: a write that
    happens to reproduce the same bytes still moves the timestamp, and writing into
    the published directory is the thing being forbidden, not changing it.
    """
    if not root.exists():
        return {}
    out: dict[str, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            out[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return out


def _output_paths_of(namespace) -> list[str]:
    """Names in ``namespace`` holding a Path under the published output directory."""
    found: list[str] = []
    for name, value in list(vars(namespace).items()):
        if isinstance(value, Path):
            try:
                value.relative_to(PUBLISHED_OUTPUTS)
            except ValueError:
                continue
            found.append(name)
    return found


def _src_modules() -> list:
    """Every ``src`` module imported so far.

    Modules the test never imported cannot write anything during it, so walking
    what is loaded is enough - and it avoids importing the rest, one of which
    creates a directory as an import side effect.
    """
    return [m for name, m in list(sys.modules.items())
            if name.startswith("src.") and m is not None]


@pytest.fixture
def isolated_outputs(tmp_path, monkeypatch) -> Path:
    """Send every output path into tmp_path, including those bound at import.

    Returns the directory the run will write into. Paths keep their position
    relative to the published directory, so ``figures/pit_histogram.png`` stays
    under a ``figures`` subdirectory and code that creates parents still works.
    """
    def remap(original: Path) -> Path:
        return tmp_path / original.relative_to(PUBLISHED_OUTPUTS)

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    for name in _output_paths_of(config):
        if name != "OUTPUT_DIR":
            monkeypatch.setattr(config, name, remap(getattr(config, name)))
    for module in _src_modules():
        for name in _output_paths_of(module):
            monkeypatch.setattr(module, name, remap(getattr(module, name)))
    return tmp_path


def changed_files(
    before: dict[str, tuple[int, int]], after: dict[str, tuple[int, int]]
) -> list[str]:
    """Files that were rewritten, added or removed between two fingerprints."""
    return sorted(
        {k for k in before if k in after and before[k] != after[k]}
        | (set(after) - set(before))
        | (set(before) - set(after))
    )


@pytest.fixture(autouse=True)
def _published_outputs_untouched():
    """Fail the test that wrote into the published output directory.

    A net rather than a redirect: it does not change what any test sees, so tests
    that read the published outputs on purpose are unaffected.
    """
    before = _fingerprint(PUBLISHED_OUTPUTS)
    yield
    changed = changed_files(before, _fingerprint(PUBLISHED_OUTPUTS))
    assert not changed, (
        "this test wrote into the published output directory "
        f"{PUBLISHED_OUTPUTS}: {', '.join(changed)}. Request the isolated_outputs "
        "fixture - some output paths are bound at import time, so patching "
        "config.OUTPUT_DIR alone does not reach them."
    )
