"""Standard tail hypothesis test and comparison power, on top of the
probabilistic evaluation.

Why this module exists
-----------------------
``probabilistic_evaluation.py`` reports the expected number of weeks with
Y >= k (the sum of every row's P(Y >= k)) against what happened, together
with a bootstrap interval on the OBSERVED side. That interval answers "is the
observed count a typical value for weeks like these", not "is the observed
count typical UNDER the distribution the model itself predicts". This module
answers the second question: for a model's own row-level exceedance
probabilities, what is the full distribution of the number of tail weeks
they imply, and how extreme is what actually happened relative to it?

Row independence
-----------------
The distribution below treats every (cell, week) row's exceedance event as an
independent Bernoulli trial. That is a simplification: weekly counts in the
same cell are weakly autocorrelated (+0.05 one week apart, measured directly
on this panel), so two rows a week apart in the same cell are not really
independent. It matters less for the weeks this test is built around than it
would in general - they sit in different cells with no two of them adjacent
in time within a cell - but the assumption is still doing real work for rows
this test does not single out, and is stated here rather than left implicit.
This is a standard test of the model's declared distribution, not a test
that also validates independence itself.

Exact, not simulated
---------------------
If the rows really are independent Bernoulli trials with per-row probability
p_i = P(Y_i >= k), the number of weeks with Y >= k is a Poisson-binomial
random variable - the sum of independent, non-identically distributed
Bernoulli trials. Its distribution is computed exactly here, by a dynamic-
programming convolution (``poisson_binomial_pmf``), rather than approximated
by simulation: there is no sampling noise to report or to average away over
repeated runs.

Power
------
For two models compared by their mean weekly log score,
``diebold_mariano`` already reports ``dm_stat = mean_diff / se``, so
``se = mean_diff / dm_stat`` needs no new computation. The smallest true mean
difference a two-sided test at the 5% level would detect 80% of the time is
``(z_0.975 + z_0.80) * se`` - the usual normal sample-size formula, solved for
the effect size instead of the sample size. Comparing that to the difference
actually observed says whether "not significant" here means "no difference"
or just "144 weeks are not enough to tell them apart".
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import config
from src import probabilistic_evaluation as pe

logger = logging.getLogger(__name__)

THRESHOLDS: tuple[int, ...] = (2, 3)
"""Tail thresholds this module tests. A subset of ``probabilistic_evaluation
.THRESHOLDS``: the two the paper's tail claim is decided on."""

NEURAL_MODELS: tuple[str, ...] = ("Hybrid_DL_Enhanced", "Neural_Poisson_Enhanced")

ETAS_SIM_MODEL: str = "ETAS_Branching_Sim"
"""Name given to the branching-process simulation's rows, matching the model
naming already used elsewhere (``ETAS_Per_Cell``, ``ETAS_NB``)."""

Z_TWO_SIDED_95: float = float(norm.ppf(0.975))
Z_POWER_80: float = float(norm.ppf(0.80))

OUTPUT_TAIL_TEST_CSV: Path = config.OUTPUT_DIR / "tail_test.csv"
OUTPUT_SCORE_POWER_CSV: Path = config.OUTPUT_DIR / "score_power.csv"


def poisson_binomial_pmf(probabilities: np.ndarray) -> np.ndarray:
    """Exact P(N = 0), ..., P(N = n) for N, the sum of n independent
    Bernoulli(p_i) trials.

    Dynamic-programming convolution: start from a certain zero and fold in
    one row's Bernoulli distribution at a time. O(n^2) time, O(n) space,
    where n is at most 2448 on this panel's test split (17 cells x 144
    weeks) - a fraction of a second, well short of where an FFT-based variant
    would be needed.
    """
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    n = len(p)
    dist = np.zeros(n + 1, dtype=np.float64)
    dist[0] = 1.0
    for i in range(n):
        pi = p[i]
        # dist[1:i+2] uses the OLD dist[0:i+1] and dist[1:i+2] (numpy builds
        # the right-hand side before assigning), so the order of these two
        # lines matters: dist[0] must be updated last, or the row just folded
        # in would be counted twice for the "stayed at 0" outcome.
        dist[1:i + 2] = dist[1:i + 2] * (1.0 - pi) + dist[0:i + 1] * pi
        dist[0] *= (1.0 - pi)
    return dist


def _discrete_quantile(pmf: np.ndarray, q: float) -> int:
    """Smallest n with P(N <= n) >= q - the standard quantile of a discrete
    distribution."""
    cdf = np.cumsum(pmf)
    return int(np.searchsorted(cdf, q, side="left"))


def _tail_test_row(name: str, threshold: int, p: np.ndarray, observed: int) -> dict:
    """One row of ``tail_test.csv`` for one model and one threshold.

    ``expected_count`` is ``p.sum()`` directly, the same expression
    ``probabilistic_evaluation.run_probabilistic_evaluation`` uses for
    ``exceedance_calibration.csv`` - not the mean of ``pmf`` below, which
    would be mathematically identical but would pick up more floating-point
    error from the convolution than a plain sum does.
    """
    p = np.asarray(p, dtype=np.float64)
    expected = float(p.sum())
    pmf = poisson_binomial_pmf(p)
    cdf = np.cumsum(pmf)

    p_lower = float(cdf[observed])                                   # P(N <= observed)
    p_upper = float(1.0 - cdf[observed - 1]) if observed > 0 else 1.0  # P(N >= observed)
    p_two_sided = float(min(1.0, 2.0 * min(p_lower, p_upper)))

    return {
        "model": name,
        "threshold": threshold,
        "observed_count": float(observed),
        "expected_count": expected,
        "count_q025": float(_discrete_quantile(pmf, 0.025)),
        "count_q975": float(_discrete_quantile(pmf, 0.975)),
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_two_sided": p_two_sided,
    }


def _tail_test_rows_for(
    y_true: np.ndarray,
    dists: list[pe.PredictiveDistribution],
    thresholds: tuple[int, ...],
) -> list[dict]:
    return [
        _tail_test_row(dist.name, k, dist.exceedance(k), int((y_true >= k).sum()))
        for k in thresholds
        for dist in dists
    ]


def _etas_simulation_rows(output_dir: Path, thresholds: tuple[int, ...]) -> list[dict]:
    """Tail test rows for the within-week branching simulation, if it ran.

    Its exceedance probabilities are already simulated frequencies
    (``p_ge2``, ``p_ge3`` in ``etas_simulation_predictions.csv``) rather than
    values from a ``PredictiveDistribution`` - there is no closed-form family
    behind them to call ``.exceedance()`` on, only 10,000 realizations per
    row. A missing file (the simulation step guards itself independently in
    ``main.py`` and can fail on its own) means no rows here, not an error.
    """
    path = Path(output_dir) / "etas_simulation_predictions.csv"
    if not path.exists():
        return []
    sim = pd.read_csv(path)
    y_sim = sim["Y"].to_numpy(dtype=np.float64)
    column_by_threshold = {2: "p_ge2", 3: "p_ge3"}
    rows: list[dict] = []
    for k in thresholds:
        column = column_by_threshold.get(k)
        if column is None or column not in sim.columns:
            continue
        rows.append(_tail_test_row(
            ETAS_SIM_MODEL, k, sim[column].to_numpy(dtype=np.float64), int((y_sim >= k).sum()),
        ))
    return rows


def collect_predictive_distributions_from(
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[pe.PredictiveDistribution]]:
    """``collect_predictive_distributions`` for a directory other than this
    process's own ``config.OUTPUT_DIR`` - a seed-sweep run kept in a scratch
    location, for instance, that was never meant to overwrite the published
    one.

    ``output_dir`` defaults to ``config.OUTPUT_DIR``, read at CALL time
    rather than stored as the parameter's default value: a default value is
    bound once, when this function is defined, and would then keep pointing
    at whatever directory was published at IMPORT time even after something
    later overrides ``config.OUTPUT_DIR`` (an environment variable, or a
    test's fixture) - the exact failure mode already documented for the
    module-level paths in ``config.py``.

    ``collect_predictive_distributions`` itself reads its five input paths
    from ``config`` at call time too, so redirecting them to ``output_dir``
    for the length of one call is safe. Nothing here writes; the five
    attributes are put back in a ``finally`` block regardless of how the
    call ends, so no other code in the same process ever observes them
    moved.
    """
    attr_names = (
        "TEST_PREDICTIONS_CSV", "MODEL_COMPARISON_CSV",
        "ETAS_TEST_PREDICTIONS_CSV", "ETAS_TRAIN_PREDICTIONS_CSV",
        "CALIBRATION_PREDICTIONS_CSV",
    )
    output_dir = Path(output_dir) if output_dir is not None else Path(config.OUTPUT_DIR)
    originals = {name: getattr(config, name) for name in attr_names}
    try:
        for name in attr_names:
            setattr(config, name, output_dir / originals[name].name)
        return pe.collect_predictive_distributions()
    finally:
        for name, value in originals.items():
            setattr(config, name, value)


def tail_test_table_for_directory(
    output_dir: Path | None = None,
    model_names: tuple[str, ...] = NEURAL_MODELS,
    thresholds: tuple[int, ...] = THRESHOLDS,
) -> pd.DataFrame:
    """The tail test table for a chosen set of models, read from ``output_dir``.

    Exists so the neural models' p-values can be checked across several
    training seeds without re-running the pipeline once per seed: a seed
    sweep already writes one full output directory per seed, and this reads
    the same files ``run_tail_test`` does, from wherever that directory
    happens to be kept. ``output_dir`` defaults to ``config.OUTPUT_DIR``,
    resolved at call time for the same reason given in
    ``collect_predictive_distributions_from``.
    """
    resolved_dir = Path(output_dir) if output_dir is not None else Path(config.OUTPUT_DIR)
    keys, dists = collect_predictive_distributions_from(resolved_dir)
    y_true = keys["y_true"].to_numpy(dtype=np.float64)
    wanted = [d for d in dists if d.name in model_names]
    missing = set(model_names) - {d.name for d in wanted}
    if missing:
        raise ValueError(f"models not found in {resolved_dir}: {sorted(missing)}")
    return pd.DataFrame(_tail_test_rows_for(y_true, wanted, thresholds))


def _score_power_table(comparisons_path: Path | None = None) -> pd.DataFrame:
    """Minimum detectable log-score difference for every pair already
    compared in ``score_comparisons.csv``.

    Restricted to the ``log_score`` rows: the log score is the whole-
    distribution measure the paper's headline comparisons are decided on,
    and ``twcrps_ge5`` rows would need their own power figure computed the
    same way, which is not part of what was asked for here.

    ``comparisons_path`` defaults to ``probabilistic_evaluation.
    OUTPUT_COMPARISONS_CSV``, resolved at call time for the same reason given
    in ``collect_predictive_distributions_from``.
    """
    if comparisons_path is None:
        comparisons_path = pe.OUTPUT_COMPARISONS_CSV
    comparisons = pd.read_csv(comparisons_path)
    log_rows = comparisons.loc[comparisons["score"] == "log_score"]

    rows: list[dict] = []
    for _, row in log_rows.iterrows():
        mean_diff = float(row["mean_diff"])
        dm_stat = float(row["dm_stat"])
        se = float(abs(mean_diff / dm_stat)) if dm_stat != 0.0 else float("nan")
        min_detectable = (Z_TWO_SIDED_95 + Z_POWER_80) * se
        rows.append({
            "model_a": row["model_a"],
            "model_b": row["model_b"],
            "mean_diff": mean_diff,
            "se": se,
            "min_detectable_diff": min_detectable,
            "ratio": abs(mean_diff) / min_detectable,
        })
    return pd.DataFrame(rows)


def run_tail_test() -> dict[str, pd.DataFrame]:
    """Compute and save the tail test and comparison power tables in the
    current output directory (``config.OUTPUT_DIR``, e.g. via
    ``SPM_OUTPUT_DIR``).

    Meant to run last in the pipeline (see ``main.py``), after both the
    whole-sample probabilistic evaluation and the branching-process
    simulation: it only reads files those steps and the modelling steps
    before them already wrote, and adds two new small tables next to them -
    it cannot move a number any earlier step computed.
    """
    keys, dists = pe.collect_predictive_distributions()
    y_true = keys["y_true"].to_numpy(dtype=np.float64)
    rows = _tail_test_rows_for(y_true, dists, THRESHOLDS)
    rows += _etas_simulation_rows(config.OUTPUT_DIR, THRESHOLDS)
    tail_df = pd.DataFrame(rows)

    power_df = _score_power_table()

    for df, path in ((tail_df, OUTPUT_TAIL_TEST_CSV), (power_df, OUTPUT_SCORE_POWER_CSV)):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %s (%d rows)", path, len(df))
    return {"tail_test": tail_df, "score_power": power_df}
