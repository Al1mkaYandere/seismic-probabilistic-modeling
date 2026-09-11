"""Where the validation block is cut out of a training period.

One rule, one place. Both protocols need it - the year-by-year one in
``validation.py`` and the static one in ``dl_modeling.py`` - and
``validation.py`` already imports from ``dl_modeling``, so a shared rule living
in either of them would be an import cycle. It also has a side effect at import
time (it creates a directory), which a module this small has no business
dragging into whoever needs a split.

Keeping it here is not an abstraction over anything: it is the same four lines
that used to sit in one protocol while the other re-implemented them by row
position and cut a week in half.
"""

from __future__ import annotations

import numpy as np


VAL_FRACTION: float = 0.15


def temporal_validation_split(
    weeks_train: np.ndarray,
    val_fraction: float = VAL_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows into (train, validation) indices along the time axis.

    The validation block is the last ``val_fraction`` of the training period's
    UNIQUE WEEKS, not of its rows. The panel on disk is sorted by
    ``["cell_id", "week"]`` (see grid_builder.py), so a row-position cut carves
    out whole cells instead of the tail of the time axis: for the 2023 fold that
    put 3 of 17 cells into validation, two of them absent from training
    altogether, and left their embeddings untrained.

    Kept as a separate function so the property can be tested directly against
    the real panel instead of being re-implemented in a test.

    Parameters
    ----------
    weeks_train : np.ndarray
        Week of each training row, in the row order the caller will use.
    val_fraction : float
        Share of unique weeks to hold out, from the end of the period.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positional indices into ``weeks_train``: training rows, validation rows.
    """
    unique_weeks = np.unique(weeks_train)
    n_val_weeks = max(1, int(np.floor(val_fraction * len(unique_weeks))))
    val_weeks = unique_weeks[-n_val_weeks:]
    val_mask = np.isin(weeks_train, val_weeks)
    return np.where(~val_mask)[0], np.where(val_mask)[0]
