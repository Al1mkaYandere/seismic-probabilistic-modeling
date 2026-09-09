"""Cross-check pipeline numbers against the frozen v1 baseline.

This test does not read or parse baseline/freeze.py — it invokes it as-is
(that is exactly what is being checked: the script prints "zero undeclared
discrepancies" and exits with code 0). It does not perform or touch the
baseline snapshot step (freeze.py save).

baseline/freeze.py is a private comparison tool that lives outside this
repository (it is not published alongside the paper's code). If it is not
present at the expected location, this test is skipped rather than failed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# .../1/seismic-probabilistic-modeling/tests/test_frozen_numbers.py -> .../1
PAPER_ROOT = Path(__file__).resolve().parents[2]
FREEZE_SCRIPT = PAPER_ROOT / "baseline" / "freeze.py"


def test_freeze_check_reports_zero_undeclared_discrepancies():
    if not FREEZE_SCRIPT.exists():
        pytest.skip(
            f"{FREEZE_SCRIPT} not found — this is a private baseline "
            "comparison tool kept outside the repository"
        )

    result = subprocess.run(
        [sys.executable, str(FREEZE_SCRIPT), "check"],
        cwd=PAPER_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        f"baseline/freeze.py check returned exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
