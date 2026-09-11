"""Renaming a model in the declaration must carry through, or fail loudly.

Five independent literals of the same name used to control behaviour inside
dl_modeling: the declaration, the audited model, the set whose per-row predictions
are kept, the model the seed sweep trains, and the rows the comparison table
replaces. A rename in one of them lost the calibration predictions or the alpha
audit without an error - the file simply lacked a model.
"""
from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from src import config, dl_modeling
from src.dl_modeling import FEATURES_BASELINE, FEATURES_ENHANCED

TEST_WEEKS = 160
RENAMED = (
    (FEATURES_BASELINE, "Plain_Net", "nb"),
    (FEATURES_ENHANCED, "Fancy_Net", "nb"),
)


def _isolate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
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


def test_renaming_a_model_carries_through_to_its_predictions(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(dl_modeling, "DL_VARIANTS", RENAMED)
    monkeypatch.setattr(dl_modeling, "AUDITED_MODEL", "Fancy_Net")

    result = dl_modeling.run_dl_modeling()
    assert set(result["model"]) == {"Plain_Net", "Fancy_Net"}

    calib = pd.read_csv(tmp_path / "calibration_predictions.csv")
    assert set(calib["model"]) == {"Fancy_Net"}, (
        f"per-row predictions were written for {sorted(set(calib['model']))}; the "
        "enhanced variant under its new name should be exactly the set, so the rule "
        "is still reading a hand-written list of the old names"
    )
    assert (tmp_path / "alpha_audit_summary.csv").exists(), (
        "the alpha audit was lost when the audited model was renamed"
    )


def test_the_comparison_table_replaces_the_rows_of_the_models_just_trained(
    tmp_path, monkeypatch
) -> None:
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(dl_modeling, "DL_VARIANTS", RENAMED)
    monkeypatch.setattr(dl_modeling, "AUDITED_MODEL", "Fancy_Net")

    # A table that already holds a stale row for one of them, plus a row belonging
    # to another step that must survive untouched.
    pd.DataFrame([
        {"model": "Fancy_Net", "Mean_Poisson_Deviance": 999.0},
        {"model": "ETAS_Per_Cell", "Mean_Poisson_Deviance": 0.5},
    ]).to_csv(tmp_path / "model_comparison.csv", index=False)

    dl_modeling.run_dl_modeling()
    table = pd.read_csv(tmp_path / "model_comparison.csv")

    assert (table["model"] == "Fancy_Net").sum() == 1, (
        "the stale row was not replaced: the merge is matching the old names"
    )
    assert 999.0 not in table["Mean_Poisson_Deviance"].to_numpy()
    assert "ETAS_Per_Cell" in set(table["model"]), "a row from another step was dropped"


def test_an_audited_model_that_is_not_trained_fails_loudly() -> None:
    # The choice of which model the paper reports cannot be derived from the list,
    # so the only protection is to refuse a disagreement instead of losing the audit.
    with pytest.raises(ValueError, match="not among the models actually trained"):
        dl_modeling.check_audited_model(audited="Nonexistent_Net")

    baseline_only = ((FEATURES_BASELINE, "Plain_Net", "nb"),)
    with pytest.raises(ValueError, match="keeps no per-row predictions"):
        dl_modeling.check_audited_model(audited="Plain_Net", variants=baseline_only)

    # And it must accept the real configuration, or the guard is just an obstacle.
    dl_modeling.check_audited_model()


def test_a_rename_stops_the_seed_sweep_loudly(tmp_path, monkeypatch) -> None:
    """Not five log lines saying a seed failed, without saying why.

    Every seed in the sweep is wrapped in ``except Exception`` so that one bad fit
    cannot kill the run. That swallowed the lookup error for a model that does not
    exist, and the sweep finished having trained nothing.
    """
    # The sweep writes its own table and figure through config, read at call time.
    # Without redirecting them a run that gets past the guard - which is exactly what
    # the mutation check makes happen - writes into the published outputs. It did.
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "ALPHA_IDENTIFIABILITY_CSV", tmp_path / "identifiability.csv")
    monkeypatch.setattr(config, "ALPHA_IDENTIFIABILITY_FIG", tmp_path / "identifiability.png")
    monkeypatch.setattr(dl_modeling, "AUDITED_MODEL", "Nonexistent_Net")
    with pytest.raises(ValueError, match="not among the models actually trained"):
        dl_modeling.run_alpha_identifiability(seeds=[42])

    with pytest.raises(ValueError, match="no model named"):
        dl_modeling.variant_for("Nonexistent_Net")
