"""A model's distribution is read from its declaration, not from its name.

Four consumers of calibration_predictions.csv used to guess it four different ways:
a prefix of the model name, a substring of it, an exact name match, and the presence
of finite numbers in the dispersion column. All four agreed, which is what made it
dangerous - a model added tomorrow breaks three of them without a word.

Each test below gives a model a name that says one thing and a declaration that says
another, and demands that the declaration win. A consumer still reading the name
fails; one reading the declaration passes.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from src import calibration, config, dl_modeling, predictive_family
from src.dl_modeling import FEATURES_ENHANCED

TEST_WEEKS = 160


def _short_panel() -> pd.DataFrame:
    full = dl_modeling._load_df()
    weeks = pd.Index(sorted(full["week"].unique()))[-TEST_WEEKS:]
    return full[full["week"].isin(weeks)].reset_index(drop=True)


def test_training_declares_the_family_it_actually_fitted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(dl_modeling, "OUTPUT_ALPHA_SUMMARY", tmp_path / "alpha.csv")
    monkeypatch.setattr(dl_modeling, "OUTPUT_ALPHA_HIST", tmp_path / "alpha.png")
    panel = _short_panel()

    _, nb_rows = dl_modeling._train_one_variant(
        panel, FEATURES_ENHANCED, "Hybrid_DL_Enhanced", loss_kind="nb"
    )
    _, poisson_rows = dl_modeling._train_one_variant(
        panel, FEATURES_ENHANCED, "Neural_Poisson_Enhanced", loss_kind="poisson"
    )

    assert predictive_family.family_of(nb_rows) == predictive_family.NB
    assert predictive_family.family_of(poisson_rows) == predictive_family.POISSON
    assert predictive_family.dispersion_of(poisson_rows) is None
    assert predictive_family.dispersion_of(nb_rows) is not None


def _misleading(name: str, family: str, n: int = 40) -> pd.DataFrame:
    """Rows whose model name contradicts their declared family."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "model": name,
        predictive_family.COLUMN: family,
        "y_true": rng.poisson(0.4, n).astype(float),
        "mu_pred": np.full(n, 0.4),
        "alpha_pred": np.full(n, 3.0),
        "cell_id": [f"c{i % 4}" for i in range(n)],
        "week": pd.date_range("2021-01-04", periods=n, freq="W-MON").astype(str),
    })


def test_the_pit_follows_the_declaration_not_the_name(tmp_path, monkeypatch) -> None:
    # Same rows twice, same name, only the declaration differs. A name-based reader
    # would give the identical answer both times.
    summaries = {}
    for family in (predictive_family.NB, predictive_family.POISSON):
        path = tmp_path / f"{family}.csv"
        _misleading("Hybrid_DL_Enhanced", family).to_csv(path, index=False)
        monkeypatch.setattr(calibration, "CALIBRATION_PREDS", path)
        monkeypatch.setattr(calibration, "OUTPUT_CALIBRATION_SUMMARY", tmp_path / f"s_{family}.csv")
        monkeypatch.setattr(calibration, "OUTPUT_PIT_HIST", tmp_path / f"p_{family}.png")
        summaries[family] = calibration.run_calibration().iloc[0]["pit_mean"]

    assert summaries[predictive_family.NB] != summaries[predictive_family.POISSON], (
        "the PIT came out the same under both declarations, so it is still being "
        "chosen by the model's name"
    )


def test_the_scored_distribution_follows_the_declaration(tmp_path, monkeypatch) -> None:
    """The family a model is scored as comes from its declaration.

    This consumer used to compare the model's name to a literal. A model called
    Hybrid_DL_Enhanced that declares Poisson must be scored as Poisson.
    """
    from src import probabilistic_evaluation as pe

    keys = _misleading("Hybrid_DL_Enhanced", predictive_family.NB)[
        ["cell_id", "week", "y_true"]
    ]
    preds = keys.copy()
    for col in ("pred_naive", "pred_poisson", "pred_nb"):
        preds[col] = 0.4
    preds_path = tmp_path / "test_predictions.csv"
    preds.to_csv(preds_path, index=False)
    monkeypatch.setattr(config, "TEST_PREDICTIONS_CSV", preds_path)

    comparison = tmp_path / "model_comparison.csv"
    pd.DataFrame([{"model": "NB_Enhanced", "alpha_hat": 2.5}]).to_csv(comparison, index=False)
    monkeypatch.setattr(config, "MODEL_COMPARISON_CSV", comparison)

    etas = keys.copy()
    etas["lambda_pred"] = 0.3
    etas_path = tmp_path / "etas_test.csv"
    etas.to_csv(etas_path, index=False)
    monkeypatch.setattr(config, "ETAS_TEST_PREDICTIONS_CSV", etas_path)
    monkeypatch.setattr(config, "ETAS_TRAIN_PREDICTIONS_CSV", tmp_path / "absent.csv")

    families = {}
    for declared in (predictive_family.NB, predictive_family.POISSON):
        calib = pd.concat([
            _misleading("Hybrid_DL_Enhanced", declared),
            _misleading("Neural_Poisson_Enhanced", predictive_family.POISSON),
        ], ignore_index=True)
        calib_path = tmp_path / f"calib_{declared}.csv"
        calib.to_csv(calib_path, index=False)
        monkeypatch.setattr(config, "CALIBRATION_PREDICTIONS_CSV", calib_path)
        _keys, dists = pe.collect_predictive_distributions()
        families[declared] = {d.name: d.family for d in dists}["Hybrid_DL_Enhanced"]

    assert families[predictive_family.NB] == "nb"
    assert families[predictive_family.POISSON] == "poisson", (
        "a model declaring Poisson was still scored as negative binomial, so the "
        "family is coming from its name"
    )


def test_the_tail_metrics_follow_the_declaration(tmp_path, monkeypatch) -> None:
    """Same rows, same name, two declarations: the scores must differ.

    This consumer used to decide by looking for "Hybrid" in the model's name, so it
    gave the identical answer under either declaration.
    """
    from src import tail_metrics

    keys = _misleading("Hybrid_DL_Enhanced", predictive_family.NB)[
        ["cell_id", "week", "y_true"]
    ]
    preds = keys.copy()
    preds["pred_poisson"] = 0.4
    preds_path = tmp_path / "test_predictions.csv"
    preds.to_csv(preds_path, index=False)
    monkeypatch.setattr(tail_metrics, "PREDICTION_STORE", preds_path)
    monkeypatch.setattr(tail_metrics, "OUTPUT_TAIL_CSV", tmp_path / "tail.csv")
    monkeypatch.setattr(tail_metrics, "OUTPUT_TAIL_FIG", tmp_path / "tail.png")
    monkeypatch.setattr(tail_metrics, "MODEL_COMPARISON", tmp_path / "absent.csv")

    for declared in (predictive_family.NB, predictive_family.POISSON):
        rows = _misleading("Hybrid_DL_Enhanced", declared)
        calib_path = tmp_path / f"calib_{declared}.csv"
        rows.to_csv(calib_path, index=False)
        monkeypatch.setattr(tail_metrics, "CALIBRATION_PREDS", calib_path)
        table = tail_metrics.run_tail_metrics()
        row = table[(table["model"] == "Hybrid_DL_Enhanced") & (table["stratum"] == "All")]
        reported = float(row["NLL"].iloc[0])

        # Pinned to the distribution itself, not to "the two runs differ": the
        # dispersion also follows the declaration, so two runs differ even when the
        # distribution is still picked by name.
        y = rows["y_true"].to_numpy(float)
        mu = np.clip(rows["mu_pred"].to_numpy(float), 1e-9, None)
        if declared == predictive_family.POISSON:
            expected = tail_metrics._poisson_nll(y, mu)
        else:
            expected = tail_metrics._nb_nll(y, mu, rows["alpha_pred"].to_numpy(float))
        assert reported == pytest.approx(expected), (
            f"declared {declared}, but the score is not the {declared} one: the "
            "distribution is still being chosen by the model's name"
        )


def test_a_name_that_contradicts_the_declaration_cannot_win() -> None:
    # A model called Hybrid_DL_... that declares Poisson has no dispersion, whatever
    # its alpha column holds.
    poisson_named_nb = _misleading("Hybrid_DL_Enhanced", predictive_family.POISSON)
    assert predictive_family.dispersion_of(poisson_named_nb) is None
    assert predictive_family.mean_dispersion_of(poisson_named_nb) is None

    # And the other way round: a name with no "Hybrid" in it that declares nb keeps
    # its dispersion.
    nb_named_poisson = _misleading("Neural_Poisson_Enhanced", predictive_family.NB)
    assert predictive_family.mean_dispersion_of(nb_named_poisson) == 3.0


def test_the_spatial_residuals_follow_the_declaration(tmp_path, monkeypatch) -> None:
    """Only the declaration is flipped, and Moran's I must move.

    This consumer had no name check at all - it averaged whatever numbers were in
    the dispersion column. A Poisson model that reports an untrained dispersion was
    therefore scored against a negative binomial variance built from it.

    The inputs are built from the committed panel rather than taken from
    ``outputs/``, which is not tracked: a test that skips on a fresh clone guards
    nothing there.
    """
    from src import spatial_diagnostics

    panel = pd.read_csv(config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE)
    keys = panel[["cell_id", "week", "Y"]].tail(17 * 30).reset_index(drop=True)
    keys = keys.rename(columns={"Y": "y_true"}).astype({"y_true": float})

    preds = keys.copy()
    preds["pred_poisson"] = 0.4
    preds_path = tmp_path / "test_predictions.csv"
    preds.to_csv(preds_path, index=False)
    monkeypatch.setattr(spatial_diagnostics, "PREDICTION_STORE", preds_path)
    monkeypatch.setattr(spatial_diagnostics, "MODEL_COMPARISON", tmp_path / "absent.csv")

    results = {}
    for declared in (predictive_family.NB, predictive_family.POISSON):
        calib = keys.copy()
        calib["model"] = "Hybrid_DL_Enhanced"
        calib[predictive_family.COLUMN] = declared
        calib["mu_pred"] = 0.4
        # A dispersion that is really there, so that ignoring the declaration and
        # averaging the column would give a negative binomial variance either way.
        calib["alpha_pred"] = 3.0
        path = tmp_path / f"calib_{declared}.csv"
        calib.to_csv(path, index=False)
        monkeypatch.setattr(spatial_diagnostics, "CALIBRATION_PREDS", path)

        out_dir = tmp_path / declared
        (out_dir / "figures").mkdir(parents=True)
        out = spatial_diagnostics.run_spatial_diagnostics(out_dir=out_dir)
        row = out.loc[out["model"] == "Hybrid_DL_Enhanced"]
        assert not row.empty, "the model is missing from the diagnostics output"
        results[declared] = float(row["moran_I"].iloc[0])

    assert np.isfinite(results[predictive_family.NB])
    assert results[predictive_family.NB] != results[predictive_family.POISSON], (
        "the residuals came out identical under both declarations, so the variance "
        "is still being built from whatever the dispersion column holds"
    )


def test_a_file_without_the_declaration_is_refused_not_guessed() -> None:
    rows = _misleading("Hybrid_DL_Enhanced", predictive_family.NB).drop(
        columns=[predictive_family.COLUMN]
    )
    with pytest.raises(KeyError, match="no 'family' column"):
        predictive_family.family_of(rows, "Hybrid_DL_Enhanced")


def test_contradictory_and_unknown_declarations_are_refused() -> None:
    mixed = _misleading("M", predictive_family.NB)
    mixed.loc[: len(mixed) // 2, predictive_family.COLUMN] = predictive_family.POISSON
    with pytest.raises(ValueError, match="expected one declared family"):
        predictive_family.family_of(mixed, "M")

    with pytest.raises(ValueError, match="unknown family"):
        predictive_family.family_of(_misleading("M", "gaussian"), "M")
