"""Which distribution a model's per-row predictions are to be read as.

Declared by whoever produced the predictions, in a column, instead of guessed by
every consumer separately. Four different guesses used to coexist over the same
file: a prefix of the model name (``calibration``), a substring of it
(``tail_metrics``), an exact name match (``probabilistic_evaluation``), and the
presence of finite numbers in the dispersion column (``spatial_diagnostics``).

All four happened to agree, which is the dangerous part: a model added tomorrow
breaks three of them silently, and a Poisson network that reports a dispersion it
never fitted was already being scored against a negative binomial variance by one
of them.

Kept in its own module rather than in the one that writes the column: every reader
would otherwise import the training module, and with it torch, to learn the name of
a string.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMN: str = "family"

NB: str = "nb"
"""Negative binomial: the mean plus a fitted dispersion."""

POISSON: str = "poisson"
"""Poisson: the mean alone. A dispersion column may still be present and must be
ignored - see the Poisson network, whose dispersion head receives no gradient."""

KNOWN: frozenset[str] = frozenset({NB, POISSON})


def family_of(rows: pd.DataFrame, model_name: str = "") -> str:
    """The declared family of one model's rows.

    Raises rather than guessing: a file without the column is a stale file, and
    falling back to the model's name is what this module exists to stop.
    """
    label = model_name or "these rows"
    if COLUMN not in rows.columns:
        raise KeyError(
            f"{label}: no {COLUMN!r} column. The predictions were written before the "
            "family was declared; re-run the pipeline rather than inferring it from "
            "the model name."
        )
    declared = rows[COLUMN].dropna().unique()
    if len(declared) != 1:
        raise ValueError(
            f"{label}: expected one declared family, found {sorted(declared)}. One "
            "model's rows cannot be read as two different distributions."
        )
    family = str(declared[0])
    if family not in KNOWN:
        raise ValueError(f"{label}: unknown family {family!r}; expected one of {sorted(KNOWN)}")
    return family


def dispersion_of(rows: pd.DataFrame, model_name: str = "") -> np.ndarray | None:
    """The per-row dispersion, or None when the declared family has none.

    None rather than zero: zero is the Poisson limit of this parameterisation, where
    1/alpha is infinite, so it cannot stand for absence.
    """
    if family_of(rows, model_name) == POISSON:
        return None
    if "alpha_pred" not in rows.columns:
        raise KeyError(f"{model_name or 'these rows'}: declared {NB} but carry no dispersion")
    return rows["alpha_pred"].to_numpy(dtype=np.float64)


def mean_dispersion_of(rows: pd.DataFrame, model_name: str = "") -> float | None:
    """One dispersion for the whole model, or None when it has none.

    Averaged over the rows that have a finite value: a column part of which is
    missing must not lose the values that are there, which a plain mean would do.
    """
    values = dispersion_of(rows, model_name)
    if values is None:
        return None
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else None
