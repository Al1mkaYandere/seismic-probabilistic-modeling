"""A model's numbers must not depend on where it sits in the training queue.

The static protocol used to seed once and then train four models in a row from
one generator stream, so each model drew from whatever the previous one had left.
The same model came out 4.16% apart in Poisson deviance depending on its
position - larger than both effects table 1 was measuring (3.67% and 2.67%), and
enough to reverse their sign.

The property is checked by reordering the queue and demanding identical numbers,
not by looking for ``_set_seed`` in the source: a text check would pass on a
version where the call sits outside the loop again.
"""
from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from src import config, dl_modeling
from src.dl_modeling import FEATURES_BASELINE, FEATURES_ENHANCED

METRICS = ["MAE", "RMSE", "Mean_Poisson_Deviance"]
# Enough weeks for the 80/20 split and the validation block, few enough that two
# full runs of the step stay inside a test suite's patience. The property under
# test does not depend on the length of the panel.
TEST_WEEKS = 160


def _isolate(tmp_path, monkeypatch) -> None:
    """Send every file this step writes into tmp_path, and shorten the panel.

    ``dl_modeling`` binds its output paths at import time, so patching
    ``config`` alone would leave the real ``outputs/`` being written to.
    """
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    for name, filename in (
        ("OUTPUT_COMPARISON", "model_comparison.csv"),
        ("OUTPUT_CALIBRATION_PREDS", "calibration_predictions.csv"),
        ("OUTPUT_ALPHA_SUMMARY", "alpha_audit_summary.csv"),
        ("OUTPUT_ALPHA_HIST", "alpha_distribution.png"),
    ):
        monkeypatch.setattr(dl_modeling, name, tmp_path / filename)

    full = dl_modeling._load_df()
    weeks = pd.Index(sorted(full["week"].unique()))[-TEST_WEEKS:]
    short = full[full["week"].isin(weeks)].reset_index(drop=True)
    monkeypatch.setattr(dl_modeling, "_load_df", lambda: short.copy())


def _run(order, tmp_path, monkeypatch) -> pd.DataFrame:
    monkeypatch.setattr(dl_modeling, "DL_VARIANTS", order)
    return dl_modeling.run_dl_modeling().set_index("model")


def test_reordering_the_queue_leaves_every_model_unchanged(tmp_path, monkeypatch) -> None:
    # All four, not just the two negative-binomial ones: the two Poisson variants
    # draw from the same generator and are moved by a reordering just as much, and
    # "they are symmetrical in the source" is an argument about the code rather
    # than a check on it.
    forward_order = dl_modeling.DL_VARIANTS
    reversed_order = tuple(reversed(forward_order))
    assert len(forward_order) == 4, "the static protocol is expected to train four models"

    _isolate(tmp_path / "a", monkeypatch)
    forward = _run(forward_order, tmp_path / "a", monkeypatch)
    _isolate(tmp_path / "b", monkeypatch)
    backward = _run(reversed_order, tmp_path / "b", monkeypatch)

    for frame, label in ((forward, "forward"), (backward, "reversed")):
        assert (frame["Status"] == "SUCCESS").all(), f"a model failed in the {label} run"

    for _features, model, _loss in forward_order:
        for metric in METRICS:
            assert forward.loc[model, metric] == backward.loc[model, metric], (
                f"{model} moved on {metric} when the training queue was reordered: "
                f"{forward.loc[model, metric]} vs {backward.loc[model, metric]}. "
                "The models are sharing one generator stream again."
            )

    # Without this the test would also pass if every model returned the same
    # number - identical columns compare equal just as well as correct ones, and a
    # defect that collapsed the four models onto one value would slip by.
    deviances = forward["Mean_Poisson_Deviance"].tolist()
    assert len(set(deviances)) == len(deviances), (
        f"two models produced the same deviance ({deviances}); the comparison is vacuous"
    )
