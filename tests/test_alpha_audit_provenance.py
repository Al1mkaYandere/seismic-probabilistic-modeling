"""The alpha audit must describe the model the paper reports, not the last seed.

``_save_alpha_audit`` used to be called from inside ``_train_one_variant`` for any
model named ``Hybrid_DL_Enhanced``, and ``run_alpha_identifiability`` trains exactly
that model once per seed. The file a full run ended with therefore described
whichever seed came last, not the model in table 1 - in v1 too.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src import config, dl_modeling

TEST_WEEKS = 160


def _isolate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    for name, filename in (
        ("OUTPUT_COMPARISON", "model_comparison.csv"),
        ("OUTPUT_CALIBRATION_PREDS", "calibration_predictions.csv"),
        ("OUTPUT_ALPHA_SUMMARY", "alpha_audit_summary.csv"),
        ("OUTPUT_ALPHA_HIST", "alpha_distribution.png"),
    ):
        monkeypatch.setattr(dl_modeling, name, tmp_path / filename)
    # The seed sweep writes through config rather than through module constants.
    monkeypatch.setattr(config, "ALPHA_IDENTIFIABILITY_CSV", tmp_path / "identifiability.csv")
    monkeypatch.setattr(config, "ALPHA_IDENTIFIABILITY_FIG", tmp_path / "identifiability.png")
    # FIGURES_DIR is computed once at import from the real OUTPUT_DIR, so patching
    # OUTPUT_DIR does not reach it and the sweep would mkdir the published figures
    # directory. Harmless while it exists, but a test has no business reaching
    # outside tmp_path at all.
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")

    full = dl_modeling._load_df()
    weeks = pd.Index(sorted(full["week"].unique()))[-TEST_WEEKS:]
    short = full[full["week"].isin(weeks)].reset_index(drop=True)
    monkeypatch.setattr(dl_modeling, "_load_df", lambda: short.copy())


def test_the_audit_describes_the_reported_model_and_the_sweep_leaves_it_alone(
    tmp_path, monkeypatch
) -> None:
    _isolate(tmp_path, monkeypatch)

    dl_modeling.run_dl_modeling()
    audit_path = tmp_path / "alpha_audit_summary.csv"
    assert audit_path.exists(), "the run produced no alpha audit at all"
    audit = pd.read_csv(audit_path).iloc[0]

    calib = pd.read_csv(tmp_path / "calibration_predictions.csv")
    reported = calib.loc[calib["model"] == dl_modeling.AUDITED_MODEL, "alpha_pred"]
    reported = reported.to_numpy(dtype=np.float64)
    reported = reported[np.isfinite(reported)]
    assert reported.size, "the reported model wrote no dispersion to compare against"

    assert audit["n"] == reported.size
    assert audit["median_alpha"] == float(np.median(reported)), (
        "the audit does not describe the model the paper reports: its median is "
        f"{audit['median_alpha']} against {float(np.median(reported))} in the "
        "predictions that model actually wrote"
    )

    before = audit_path.read_bytes()
    # Two seeds are enough: the old defect overwrote the file on the first one.
    dl_modeling.run_alpha_identifiability(seeds=[7, 999])
    assert audit_path.read_bytes() == before, (
        "the five-seed sweep overwrote the audit of the reported model"
    )

    # Without this the test would pass on a sweep that silently produced nothing,
    # in which case it could not have overwritten anything either.
    sweep = pd.read_csv(tmp_path / "identifiability.csv")
    assert sorted(sweep["seed"].tolist()) == [7, 999], "the sweep did not run"
    assert sweep["median_alpha"].nunique() == 2, "the two seeds gave the same model"
