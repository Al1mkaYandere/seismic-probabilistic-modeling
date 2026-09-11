"""The static protocol must hold out whole weeks, like the year-by-year one does.

It used to cut the validation block by row position: ``n_val`` was a tenth of the
ROWS, and the panel holds 17 cells per week, so the cut landed inside a week.
2020-02-17 went 9 cells into training and 8 into validation, and early stopping
was then judged partly on a week the model had already been trained on.

The check spies on the shared split rule rather than reading the source: if the
row-position cut comes back, the rule is never called and the spy stays empty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import dl_modeling, splitting
from src.dl_modeling import FEATURES_BASELINE

# Enough weeks that a tenth of them is several, few enough to keep the suite quick.
TEST_WEEKS = 160


def _short_panel() -> pd.DataFrame:
    full = dl_modeling._load_df()
    weeks = pd.Index(sorted(full["week"].unique()))[-TEST_WEEKS:]
    return full[full["week"].isin(weeks)].reset_index(drop=True)


def test_the_validation_block_is_whole_weeks_from_the_end(monkeypatch) -> None:
    captured: dict[str, object] = {}
    real = splitting.temporal_validation_split

    def spy(weeks_train, val_fraction=splitting.VAL_FRACTION):
        train_idx, val_idx = real(weeks_train, val_fraction)
        captured.update(
            weeks=np.asarray(weeks_train),
            train_idx=train_idx,
            val_idx=val_idx,
            fraction=val_fraction,
        )
        return train_idx, val_idx

    monkeypatch.setattr(dl_modeling, "temporal_validation_split", spy)

    # Hybrid_DL_Baseline writes nothing to disk; the Enhanced name would trigger
    # the alpha audit, which saves a CSV and a figure into the output directory.
    dl_modeling._train_one_variant(
        _short_panel(), FEATURES_BASELINE, "Hybrid_DL_Baseline", loss_kind="nb"
    )

    assert captured, (
        "the static protocol never called the shared split rule - it is cutting the "
        "validation block on its own again, by row position"
    )

    weeks = captured["weeks"]
    train_weeks = set(weeks[captured["train_idx"]].tolist())
    val_weeks = set(weeks[captured["val_idx"]].tolist())

    assert not (train_weeks & val_weeks), (
        f"{len(train_weeks & val_weeks)} week(s) are in training and validation at "
        "once; the cut fell inside a week"
    )
    assert max(train_weeks) < min(val_weeks), (
        "the validation block is not the end of the training period"
    )
    assert len(captured["train_idx"]) + len(captured["val_idx"]) == len(weeks), (
        "rows went missing between training and validation"
    )

    # The relationship, not a literal: a test that hard-codes "57 weeks" starts
    # failing the day the panel changes length, for no defect at all.
    fraction = float(captured["fraction"])
    assert 0.0 < fraction < 1.0, f"implausible validation fraction {fraction}"
    expected = max(1, int(np.floor(fraction * len(set(weeks.tolist())))))
    assert len(val_weeks) == expected, (
        f"held out {len(val_weeks)} weeks, expected {expected} at fraction {fraction}"
    )
