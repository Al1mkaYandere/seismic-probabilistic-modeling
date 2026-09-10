"""Probabilistic evaluation of every model on the whole test split.

Why this module exists
----------------------
The paper's tail claim was computed on the 6 observations with Y >= 5, four of
which sit in one cell and three of those in three consecutive weeks - one
aftershock sequence. Six points cannot separate models. This module scores the
full predictive distribution of every model on all 2448 test observations
(144 weeks x 17 cells), so that statements about rare weeks rest on the whole
sample rather than on its extreme corner.

The seismological baseline is included. ETAS returns a conditional intensity,
i.e. a complete Poisson predictive distribution, so its exceedance probabilities
are computed directly rather than approximated.

What is scored
--------------
* Exceedance calibration: predicted P(Y >= k) against the observed frequency,
  for k = 1, 2, 3, 5, with a block bootstrap interval on the observed side and a
  Brier score on the predicted side.
* Logarithmic score of the whole distribution, taken from scipy's ``logpmf`` for
  both families so the log(y!) convention cannot drift apart between them - the
  defect fixed in 4f9aafc was exactly such a drift.
* CRPS, unweighted and threshold-weighted (Gneiting & Ranjan 2011). The weighted
  version puts weight only on the part of the distribution above a threshold, so
  it says something about the tail while still using every observation.
* Diebold-Mariano tests on the 144 weekly mean score differences, with a
  Newey-West correction, because weeks are serially dependent after a large event.

All intervals come from a moving block bootstrap over runs of CONSECUTIVE weeks,
five weeks to a block. A week is never split - its 17 cells always move together
- but keeping a week whole is not what the method is for: the measured
correlation between cells within a week is -0.001. What has to be preserved is
the run, because a large event raises counts in the same cell for the weeks that
follow (+0.052 at lag 1). Resampling single weeks independently breaks exactly
that structure and returns an interval narrower than resampling raw rows.

The block length follows the same rule as the Newey-West lag used by the
Diebold-Mariano test, so both ways of handling serial dependence in this module
agree on how far it reaches.

Two consequences worth stating rather than hiding. Blocks of five over 144 weeks
means 29 blocks, i.e. 145 weeks per resample instead of 144; the surplus is 0.7%
of the sample, well inside the bias any moving block bootstrap carries anyway.
And the interval does not always widen relative to independent weeks: where the
weekly differences are negatively autocorrelated, consecutive weeks cancel and
the honest interval is narrower. That is the method reproducing the dependence
that is there, not a failure of it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

from src import config

logger = logging.getLogger(__name__)

THRESHOLDS: tuple[int, ...] = (1, 2, 3, 5)
N_BOOTSTRAP: int = 2000
BOOTSTRAP_SEED: int = 42
CRPS_TAIL_PAD: int = 40

OUTPUT_EXCEEDANCE_CSV: Path = config.OUTPUT_DIR / "exceedance_calibration.csv"
OUTPUT_SCORES_CSV: Path = config.OUTPUT_DIR / "probabilistic_scores.csv"
OUTPUT_COMPARISONS_CSV: Path = config.OUTPUT_DIR / "score_comparisons.csv"

# Pairs worth testing, in the order the paper argues them.
COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (
    ("Hybrid_DL_Enhanced", "ETAS_Per_Cell"),
    ("Hybrid_DL_Enhanced", "Hybrid_DL_as_Poisson"),
    ("Hybrid_DL_Enhanced", "NB_GLM"),
    ("Hybrid_DL_Enhanced", "Neural_Poisson_Enhanced"),
    ("NB_GLM", "ETAS_Per_Cell"),
    ("Neural_Poisson_Enhanced", "ETAS_Per_Cell"),
)


@dataclass(frozen=True)
class PredictiveDistribution:
    """One model's predictive distribution over the test rows.

    ``alpha`` is the negative binomial dispersion, per observation. It is None
    for Poisson models, and must be None rather than zero: alpha = 0 is the
    Poisson limit only as a limit, and 1/alpha overflows on the way there.
    """

    name: str
    family: str  # "poisson" | "nb"
    mu: np.ndarray
    alpha: np.ndarray | None = None

    def _nb_args(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.clip(np.asarray(self.alpha, dtype=np.float64), 1e-9, None)
        r = 1.0 / a
        mu = np.clip(self.mu, 1e-9, None)
        return r, r / (r + mu)

    def exceedance(self, k: int) -> np.ndarray:
        """P(Y >= k) for every observation."""
        if self.family == "poisson":
            return poisson.sf(k - 1, np.clip(self.mu, 1e-9, None))
        r, p = self._nb_args()
        return nbinom.sf(k - 1, n=r, p=p)

    def log_score(self, y: np.ndarray) -> np.ndarray:
        """-log P(Y = y) for every observation, log(y!) included."""
        if self.family == "poisson":
            return -poisson.logpmf(y, np.clip(self.mu, 1e-9, None))
        r, p = self._nb_args()
        return -nbinom.logpmf(y, n=r, p=p)

    def crps_terms(self, y: np.ndarray, k_max: int) -> np.ndarray:
        """(F(k) - 1{y <= k})^2 for every observation and every k <= k_max.

        Returned per-k rather than summed, so that a threshold weight can be
        applied afterwards without recomputing the distribution.
        """
        ks = np.arange(k_max + 1)
        if self.family == "poisson":
            F = poisson.cdf(ks[None, :], np.clip(self.mu, 1e-9, None)[:, None])
        else:
            r, p = self._nb_args()
            F = nbinom.cdf(ks[None, :], n=r[:, None], p=p[:, None])
        return (F - (y[:, None] <= ks[None, :]).astype(np.float64)) ** 2


def collect_predictive_distributions() -> tuple[pd.DataFrame, list[PredictiveDistribution]]:
    """Assemble every model's predictive distribution on the same test rows.

    Returns the key frame (cell_id, week, y_true) and the distributions aligned
    to it row for row. Every source is joined on (cell_id, week) rather than
    concatenated by position: the files are written by different steps and are
    not guaranteed to share a row order, and a silent misalignment here would
    corrupt every number downstream.
    """
    preds = pd.read_csv(config.TEST_PREDICTIONS_CSV)
    keys = preds[["cell_id", "week", "y_true"]].copy()

    comparison = pd.read_csv(config.MODEL_COMPARISON_CSV)
    alpha_row = comparison.loc[comparison["model"] == "NB_Enhanced", "alpha_hat"]
    alpha_glm = float(alpha_row.iloc[0])
    if not np.isfinite(alpha_glm):
        raise ValueError("NB_Enhanced alpha_hat is not finite; cannot score the NB GLM")

    n = len(keys)
    dists: list[PredictiveDistribution] = [
        PredictiveDistribution("Naive_Persistence", "poisson", preds["pred_naive"].to_numpy(float)),
        PredictiveDistribution("Poisson_GLM", "poisson", preds["pred_poisson"].to_numpy(float)),
        PredictiveDistribution("NB_GLM", "nb", preds["pred_nb"].to_numpy(float),
                               np.full(n, alpha_glm)),
    ]

    etas = pd.read_csv(config.ETAS_TEST_PREDICTIONS_CSV)
    merged = keys.merge(etas[["cell_id", "week", "lambda_pred"]], on=["cell_id", "week"], how="left")
    if merged["lambda_pred"].isna().any():
        raise ValueError("ETAS predictions do not cover every test row")
    dists.append(PredictiveDistribution("ETAS_Per_Cell", "poisson",
                                        merged["lambda_pred"].to_numpy(float)))

    calib = pd.read_csv(config.CALIBRATION_PREDICTIONS_CSV)
    hybrid_mu: np.ndarray | None = None
    for name in ("Neural_Poisson_Enhanced", "Hybrid_DL_Enhanced"):
        sub = calib.loc[calib["model"] == name, ["cell_id", "week", "mu_pred", "alpha_pred"]]
        joined = keys.merge(sub, on=["cell_id", "week"], how="left")
        if joined["mu_pred"].isna().any():
            raise ValueError(f"{name} predictions do not cover every test row")
        mu = joined["mu_pred"].to_numpy(float)
        if name == "Hybrid_DL_Enhanced":
            hybrid_mu = mu
            dists.append(PredictiveDistribution(name, "nb", mu,
                                                joined["alpha_pred"].to_numpy(float)))
        else:
            # Neural_Poisson carries an alpha column too, but it is a Poisson
            # model: scoring it as NB would silently give it a second parameter
            # it never used at training time.
            dists.append(PredictiveDistribution(name, "poisson", mu))

    if hybrid_mu is not None:
        # Same mean, Poisson instead of NB. This is the only comparison that
        # isolates the contribution of the DISTRIBUTION: everything else differs
        # in the mean as well, so a win there cannot be attributed to dispersion.
        dists.append(PredictiveDistribution("Hybrid_DL_as_Poisson", "poisson", hybrid_mu))

    return keys, dists


def _week_blocks(weeks: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Row indices of each week, in chronological order.

    Weeks are sorted as strings, which is chronological for ISO dates. The order
    matters: the bootstrap below draws runs of CONSECUTIVE weeks, so a shuffled
    ordering would silently turn a moving-block bootstrap back into a single-week
    one.
    """
    unique = np.unique(weeks)
    return unique, [np.flatnonzero(weeks == w) for w in unique]


def newey_west_lag(n_periods: int) -> int:
    """Standard automatic lag, floor(4 * (T / 100) ** (2/9)); 4 at T = 144."""
    return int(np.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0)))


def block_bootstrap_ci(values: np.ndarray, weeks: np.ndarray,
                       n_boot: int = N_BOOTSTRAP,
                       seed: int = BOOTSTRAP_SEED,
                       block_weeks: int | None = None) -> tuple[float, float]:
    """Percentile interval for the mean of ``values``, resampling runs of weeks.

    A week is the indivisible unit - its 17 cells always move together - and the
    bootstrap draws blocks of ``block_weeks`` CONSECUTIVE weeks.

    Both halves of that matter, for different reasons, and an earlier version of
    this function got the reasoning backwards. Keeping a week whole costs
    nothing but buys nothing either: the measured correlation between cells
    within a week is -0.001, so there is no within-week dependence to protect
    against. The dependence that does exist is temporal - a large event raises
    counts in the same cell for the following weeks, measured at +0.052 at lag 1
    and near zero beyond - and only consecutive runs capture it. Resampling
    single weeks independently destroys exactly the structure that needs
    preserving, and returns an interval narrower than resampling raw rows.

    The block length follows the same rule as the Newey-West lag used by the
    Diebold-Mariano test, so the two ways of handling serial dependence in this
    module agree on how far it reaches: 5 weeks at T = 144.
    """
    unique, blocks = _week_blocks(weeks)
    n_weeks = len(unique)
    if block_weeks is None:
        block_weeks = min(newey_west_lag(n_weeks) + 1, n_weeks)
    block_weeks = max(1, min(block_weeks, n_weeks))

    n_starts = n_weeks - block_weeks + 1
    n_blocks = int(np.ceil(n_weeks / block_weeks))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, n_starts, size=n_blocks)
        picked = np.concatenate([
            blocks[start + offset]
            for start in starts
            for offset in range(block_weeks)
        ])
        means[b] = values[picked].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, weeks: np.ndarray,
                    lag: int | None = None) -> dict[str, float]:
    """Compare two models on their weekly mean loss difference.

    Positive ``mean_diff`` means model A loses more, i.e. B is better. The
    Newey-West correction is applied because a strong event raises the loss of
    every model for several weeks running, which makes the raw variance of the
    difference too small.
    """
    diffs = pd.Series(loss_a - loss_b).groupby(pd.Series(weeks)).mean().to_numpy()
    T = len(diffs)
    if lag is None:
        lag = newey_west_lag(T)
    d_bar = float(diffs.mean())
    u = diffs - d_bar
    var = float(np.mean(u * u))
    for h in range(1, lag + 1):
        var += 2.0 * (1.0 - h / (lag + 1.0)) * float(np.mean(u[h:] * u[:-h]))
    var = max(var, 1e-18)
    stat = d_bar / np.sqrt(var / T)
    return {"n_weeks": T, "lag": lag, "mean_diff": d_bar,
            "dm_stat": float(stat), "p_value": float(2.0 * (1.0 - norm.cdf(abs(stat))))}


def run_probabilistic_evaluation() -> dict[str, pd.DataFrame]:
    """Score every model on the full test split and save the three tables."""
    keys, dists = collect_predictive_distributions()
    y = keys["y_true"].to_numpy(dtype=np.float64)
    weeks = keys["week"].to_numpy()
    k_max = int(y.max()) + CRPS_TAIL_PAD
    n_obs = len(y)

    # ── Exceedance calibration ────────────────────────────────────────────────
    exceedance_rows: list[dict] = []
    for k in THRESHOLDS:
        observed = (y >= k).astype(np.float64)
        obs_lo, obs_hi = block_bootstrap_ci(observed, weeks)
        for dist in dists:
            p = dist.exceedance(k)
            expected = float(p.sum())
            exceedance_rows.append({
                "model": dist.name,
                "threshold": k,
                "predicted_rate": float(p.mean()),
                "observed_rate": float(observed.mean()),
                "observed_rate_lo": obs_lo,
                "observed_rate_hi": obs_hi,
                "expected_count": expected,
                "observed_count": float(observed.sum()),
                # How many times the model under-promises the event. Above 1 the
                # model is too optimistic; below 1 it cries wolf.
                "observed_over_expected": float(observed.sum() / max(expected, 1e-12)),
                "inside_interval": bool(obs_lo * n_obs <= expected <= obs_hi * n_obs),
                "brier": float(np.mean((p - observed) ** 2)),
            })
    exceedance_df = pd.DataFrame(exceedance_rows)

    # ── Scores over the whole distribution ────────────────────────────────────
    log_scores = {d.name: d.log_score(y) for d in dists}
    crps_terms = {d.name: d.crps_terms(y, k_max) for d in dists}
    tail_crps = {
        d.name: {r: crps_terms[d.name][:, r:].sum(axis=1) for r in (0,) + THRESHOLDS}
        for d in dists
    }

    score_rows: list[dict] = []
    for dist in dists:
        ls = log_scores[dist.name]
        ls_lo, ls_hi = block_bootstrap_ci(ls, weeks)
        row = {"model": dist.name, "family": dist.family, "n_obs": n_obs,
               "log_score": float(ls.mean()), "log_score_lo": ls_lo, "log_score_hi": ls_hi}
        for r, values in tail_crps[dist.name].items():
            label = "crps" if r == 0 else f"twcrps_ge{r}"
            row[label] = float(values.mean())
        crps_lo, crps_hi = block_bootstrap_ci(tail_crps[dist.name][0], weeks)
        row["crps_lo"], row["crps_hi"] = crps_lo, crps_hi
        score_rows.append(row)
    scores_df = pd.DataFrame(score_rows).sort_values("log_score").reset_index(drop=True)

    # ── Pairwise tests ────────────────────────────────────────────────────────
    available = {d.name for d in dists}
    comparison_rows: list[dict] = []
    for a, b in COMPARISON_PAIRS:
        if a not in available or b not in available:
            logger.warning("Comparison %s vs %s skipped: model missing", a, b)
            continue
        for score_name, series in (("log_score", log_scores),
                                   ("twcrps_ge5", {d.name: tail_crps[d.name][5] for d in dists})):
            diff = series[a] - series[b]
            dm = diebold_mariano(series[a], series[b], weeks)
            lo, hi = block_bootstrap_ci(diff, weeks)
            comparison_rows.append({
                "model_a": a, "model_b": b, "score": score_name,
                "mean_diff": dm["mean_diff"], "diff_lo": lo, "diff_hi": hi,
                "dm_stat": dm["dm_stat"], "p_value": dm["p_value"],
                "n_weeks": dm["n_weeks"], "newey_west_lag": dm["lag"],
                "better": a if dm["mean_diff"] < 0 else b,
            })
    comparisons_df = pd.DataFrame(comparison_rows)

    for df, path in ((exceedance_df, OUTPUT_EXCEEDANCE_CSV),
                     (scores_df, OUTPUT_SCORES_CSV),
                     (comparisons_df, OUTPUT_COMPARISONS_CSV)):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %s (%d rows)", path, len(df))

    return {"exceedance": exceedance_df, "scores": scores_df, "comparisons": comparisons_df}
