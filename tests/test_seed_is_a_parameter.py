"""Both protocols must take their seed from the caller, not from a literal.

Without this the spread of the numbers over seeds cannot be measured, and that
spread is itself a result: the same model moved 4.16% on the order it was trained
in, which is more than the effects table 1 reports.

The static protocol is checked by running it and comparing numbers. The
year-by-year one is checked by recording the seed that actually reaches every
consumer - it trains 12 networks and fits ETAS six times per run, which is too
slow for a test suite, and "the seed reached every place that uses it" is the
property anyway.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src import config, dl_modeling, validation
from src.dl_modeling import FEATURES_ENHANCED

TEST_WEEKS = 160
ONE_VARIANT = ((FEATURES_ENHANCED, "Hybrid_DL_Enhanced", "nb"),)


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
    monkeypatch.setattr(dl_modeling, "DL_VARIANTS", ONE_VARIANT)
    full = dl_modeling._load_df()
    weeks = pd.Index(sorted(full["week"].unique()))[-TEST_WEEKS:]
    short = full[full["week"].isin(weeks)].reset_index(drop=True)
    monkeypatch.setattr(dl_modeling, "_load_df", lambda: short.copy())


def _deviance(frame: pd.DataFrame) -> float:
    return float(frame.set_index("model").loc["Hybrid_DL_Enhanced", "Mean_Poisson_Deviance"])


def test_the_static_protocol_obeys_the_seed_it_is_given(tmp_path, monkeypatch) -> None:
    _isolate(tmp_path / "a", monkeypatch)
    first = _deviance(dl_modeling.run_dl_modeling(seed=1))
    _isolate(tmp_path / "b", monkeypatch)
    again = _deviance(dl_modeling.run_dl_modeling(seed=1))
    _isolate(tmp_path / "c", monkeypatch)
    other = _deviance(dl_modeling.run_dl_modeling(seed=2))

    assert first == again, (
        f"the same seed gave {first} and {again}: the run is not reproducible, so "
        "a spread measured over seeds would be measuring noise of its own"
    )
    assert first != other, (
        f"seeds 1 and 2 both gave {first}: the seed argument does not reach the "
        "generator, and a five-seed sweep would report five copies of one number"
    )


def test_the_year_by_year_protocol_passes_its_seed_to_every_consumer(
    tmp_path, monkeypatch
) -> None:
    # run_walk_forward writes three files, and two of their paths are bound at
    # import time, so patching config.OUTPUT_DIR alone leaves the published
    # walk-forward results being overwritten - with this test's stub predictions.
    # It happened once; the frozen-numbers test caught it.
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(validation, "OUTPUT_WF_CSV", tmp_path / "walk_forward_results.csv")
    monkeypatch.setattr(validation, "OUTPUT_WF_FIG", tmp_path / "stability.png")

    seeds_set: list[int] = []
    trainer_seeds: list[int] = []
    monkeypatch.setattr(validation, "set_seeds", lambda seed=None: seeds_set.append(seed))

    def fake_trainer(*args, **kwargs):
        trainer_seeds.append(kwargs.get("seed"))
        # Shape only: the caller scores these against the test rows.
        return np.zeros(len(args[4]), dtype=np.float64)

    monkeypatch.setattr(validation, "_dl_train_predict", fake_trainer)

    # ETAS is refitted once per fold and is the slow part of this function; the
    # seed never reaches it, so stubbing it keeps the test about the seed. The
    # module is imported inside the loop, so the patch goes on the module itself.
    from src import etas_baseline
    monkeypatch.setattr(etas_baseline, "fit_etas_per_cell", lambda *a, **k: {})
    monkeypatch.setattr(
        etas_baseline, "predict_etas",
        lambda *a, **k: pd.DataFrame(columns=["cell_id", "week", "lambda_pred"]),
    )

    validation.run_walk_forward(seed=7)

    assert seeds_set == [7], f"set_seeds received {seeds_set} instead of [7]"
    assert trainer_seeds, "the trainer was never called; the test proves nothing"
    assert set(trainer_seeds) == {7}, (
        f"the trainer was called with {sorted(set(trainer_seeds))}: some call site "
        "still relies on the default instead of the seed it was given"
    )
    # Two models per year over six years: a run that trained fewer would let a
    # missed call site hide.
    assert len(trainer_seeds) == 12, f"expected 12 trainings, recorded {len(trainer_seeds)}"


def test_calling_without_a_seed_uses_the_shared_default(tmp_path, monkeypatch) -> None:
    """The default must BE the shared constant, observed rather than read.

    Comparing ``__defaults__`` to the constant would be a check on the text of the
    program: it passes on code that takes the right default and then ignores it.
    Here each entry point is called with no argument and asked what it seeded.
    """
    import torch

    validation.set_seeds()
    assert torch.initial_seed() == dl_modeling.SEED, (
        f"set_seeds() with no argument seeded {torch.initial_seed()} instead of "
        f"{dl_modeling.SEED}"
    )

    used: list[int] = []
    monkeypatch.setattr(dl_modeling, "_set_seed", lambda s: used.append(s))
    # No training needed: the question is only which seed the loop starts from.
    monkeypatch.setattr(dl_modeling, "_load_df", lambda: pd.DataFrame())
    monkeypatch.setattr(
        dl_modeling, "_train_one_variant",
        lambda *a, **k: ({"model": a[2], "Status": "SUCCESS"}, None),
    )
    monkeypatch.setattr(dl_modeling, "DL_VARIANTS", ONE_VARIANT)
    # A path that does not exist: run_dl_modeling merges into the comparison table
    # when it is there, and an empty file is not a readable one.
    monkeypatch.setattr(dl_modeling, "OUTPUT_COMPARISON", tmp_path / "model_comparison.csv")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    dl_modeling.run_dl_modeling()
    assert used == [dl_modeling.SEED], (
        f"run_dl_modeling() with no argument seeded {used} instead of "
        f"[{dl_modeling.SEED}]"
    )
