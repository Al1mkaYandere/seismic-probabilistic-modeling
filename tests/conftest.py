"""Shared pytest configuration.

Makes the ``src`` package importable regardless of which directory pytest is
run from and which import mode is in effect (tests/ has no rootdir
__init__.py, so without this explicit insertion pytest may not add the repo
root to sys.path).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
