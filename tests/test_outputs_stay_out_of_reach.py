"""Tests must not be able to write into the published output directory.

Twice in one session a test did, and both times it was found afterwards rather than
at the moment it happened. The cause is structural: several modules bind their output
paths at import time - deliberately, so that the environment is read before anything
runs - so patching ``config.OUTPUT_DIR`` does not reach them, and every test had to
remember its own list of paths.

Two things are guarded here: that the fixture finds every such path rather than
relying on a list someone maintains, and that the net under it can tell a write
happened at all.
"""
from __future__ import annotations

from pathlib import Path

import conftest
from src import (  # noqa: F401  - imported so the fixture has them to find
    calibration,
    config,
    dl_modeling,
    level_calibration,
    mc_estimation,
    modeling,
    poisson_analysis,
    probabilistic_evaluation,
    spatial_diagnostics,
    tail_metrics,
    validation,
)


def test_the_fixture_finds_every_output_path_including_those_bound_at_import(
    isolated_outputs,
) -> None:
    """No module may still be pointing at the published directory.

    Written as a search rather than a checklist: an output file added tomorrow is
    covered the day it appears, or this fails and says which one.
    """
    leftovers: list[str] = []
    for namespace in [config, *conftest._src_modules()]:
        for name in conftest._output_paths_of(namespace):
            leftovers.append(f"{namespace.__name__}.{name}")

    assert not leftovers, (
        "these still point inside the published output directory while the "
        f"isolation fixture is active: {', '.join(sorted(leftovers))}"
    )
    # Without this the assertion above would pass on a fixture that found nothing
    # because the search itself is broken.
    probe = type("N", (), {"a": conftest.PUBLISHED_OUTPUTS / "x.csv", "b": Path("/tmp/y")})
    assert conftest._output_paths_of(probe) == ["a"], "the search itself does not work"

    # And the writes really go somewhere else.
    assert config.OUTPUT_DIR == isolated_outputs
    assert config.FIGURES_DIR.is_relative_to(isolated_outputs)
    assert dl_modeling.OUTPUT_CALIBRATION_PREDS.is_relative_to(isolated_outputs)
    assert validation.OUTPUT_WF_CSV.is_relative_to(isolated_outputs)


def test_the_net_notices_a_write(tmp_path) -> None:
    """The autouse guard compares fingerprints; this is that comparison."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.csv").write_text("one")
    before = conftest._fingerprint(tmp_path)
    assert before, "fingerprint of a non-empty directory must not be empty"
    assert conftest.changed_files(before, conftest._fingerprint(tmp_path)) == []

    (tmp_path / "sub" / "a.csv").write_text("two different bytes")
    assert conftest.changed_files(before, conftest._fingerprint(tmp_path)) == ["sub/a.csv"]

    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    assert "b.png" in conftest.changed_files(before, conftest._fingerprint(tmp_path))

    (tmp_path / "sub" / "a.csv").unlink()
    assert "sub/a.csv" in conftest.changed_files(before, conftest._fingerprint(tmp_path))


def test_an_absent_directory_is_not_an_error() -> None:
    assert conftest._fingerprint(Path("/nonexistent/outputs")) == {}
