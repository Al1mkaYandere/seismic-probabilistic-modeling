"""The year-by-year baseline must be fitted on the catalogue the panel was built from.

It read the raw file directly while the panel applied the completeness cut, so at
M_c = 4.5 the fit saw all 1607 events and was scored against a target counting only
the 722 above the threshold. ETAS then promised 560 events against the 308 that
happened - the same 561 it promises at M_c = 3.0, as if nothing had been thinned.

At M_c = 3.0 the filter removes nothing, so the defect only ever showed at the
honest threshold. Both thresholds are checked here for exactly that reason: a test
that looked only at 3.0 would have passed on the defect.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")

from src import config, etas_baseline, validation


def _run_and_record(threshold: float, tmp_path, monkeypatch) -> dict[str, object]:
    """Run the year-by-year protocol and report the events ETAS was handed."""
    monkeypatch.setattr(config, "M_C", threshold)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(validation, "OUTPUT_WF_CSV", tmp_path / "walk_forward_results.csv")
    monkeypatch.setattr(validation, "OUTPUT_WF_FIG", tmp_path / "stability.png")

    seen: dict[str, object] = {}

    def fake_fit(events_df, train_end, *a, **k):
        mags = events_df["mag"].to_numpy(dtype=float)
        seen["n"] = int(mags.size)
        seen["min_mag"] = float(mags.min())
        return {}

    def fake_predict(params, weeks_grid, events_df, *a, **k):
        # Prediction reads the history too, and must see the same catalogue.
        seen["n_predict"] = int(len(events_df))
        return pd.DataFrame(columns=["cell_id", "week", "lambda_pred"])

    monkeypatch.setattr(etas_baseline, "fit_etas_per_cell", fake_fit)
    monkeypatch.setattr(etas_baseline, "predict_etas", fake_predict)
    # Training the networks is beside the point here and is the slow part.
    monkeypatch.setattr(
        validation, "_dl_train_predict",
        lambda *args, **kwargs: np.zeros(len(args[4]), dtype=np.float64),
    )
    validation.run_walk_forward()
    return seen


def test_the_baseline_sees_only_events_above_the_completeness_threshold(
    tmp_path, monkeypatch
) -> None:
    # Both thresholds in one test: each run of the protocol costs a few seconds, and
    # the two halves of the property need both. At 4.5 the cut must bite; at 3.0 it
    # must be a no-op, or the published numbers would move.
    honest = _run_and_record(4.5, tmp_path / "honest", monkeypatch)
    published = _run_and_record(3.0, tmp_path / "published", monkeypatch)

    assert honest, "ETAS was never fitted; the test proves nothing"
    assert honest["min_mag"] >= 4.5, (
        f"the fit was handed an event of magnitude {honest['min_mag']}, below the "
        "completeness threshold of 4.5: it is being fitted on a different catalogue "
        "than the panel it is scored against"
    )
    assert honest["n_predict"] == honest["n"], (
        "prediction and fitting were handed different catalogues"
    )

    # Without this the assertion above would pass on a filter that removes nothing.
    assert honest["n"] < published["n"], (
        f"the honest threshold kept {honest['n']} events and the published one "
        f"{published['n']}: the cut is not removing anything"
    )
    raw = pd.read_csv(config.RAW_DATA_PATH / config.RAW_DATA_FILE)
    assert published["n"] == len(raw), (
        f"the published threshold dropped {len(raw) - published['n']} events; at 3.0 "
        "the cut must be a no-op, and anything else moves the published numbers"
    )
    assert honest["n"] == int((raw["mag"] >= 4.5).sum())
