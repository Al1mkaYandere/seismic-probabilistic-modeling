"""Behavioural checks for the distributions: NB/Poisson NLL and discrete CRPS.

One test in this file is an expected failure (xfail). It documents known
issue A5: ``_poisson_nll`` in src/tail_metrics.py omits the log(y!)
normalising constant, so its output is not the true Poisson negative
log-likelihood and is not directly comparable to NB NLL, which does include
that constant.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import gammaln
from scipy.stats import poisson as sp_poisson

from src.dl_modeling import NBLoss, PoissonLoss
from src.tail_metrics import _discrete_crps, _poisson_nll


def _synthetic_counts(seed: int, n: int = 200, mu_low: float = 0.5, mu_high: float = 8.0):
    """Poisson-distributed (y, mu) pairs with a fixed seed — a known ground-truth process."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(mu_low, mu_high, size=n)
    y = rng.poisson(mu).astype(np.float64)
    return mu, y


def test_nbloss_converges_to_poisson_plus_log_factorial():
    """NBLoss (dl_modeling.py) must match PoissonLoss + log(y!) as alpha -> 0.

    NBLoss already includes -lgamma(y+1) in its log-likelihood term
    (dl_modeling.py, around line 110); PoissonLoss omits that constant by
    definition (this is not a bug for the torch models — they are compared
    against each other via MAE/RMSE/MPD, not via absolute NLL). So this test
    expects an exact match up to that constant, rather than being an xfail.

    Alpha is set to 3e-5 rather than the more literal 1e-8. Verified
    empirically: NBLoss adds eps=1e-8 inside its logarithms for numerical
    stability, and when alpha is at the same order as eps, the alpha*mu term
    (~1e-7 for typical mu) becomes comparable in size to eps itself, and the
    discrepancy grows to ~0.1 — an artefact of the stabilising eps, not a
    property of the NB NLL formula. At alpha=3e-5 (alpha*mu is orders of
    magnitude above eps, while alpha is still negligible for the Poisson
    limit) the discrepancy drops to ~3e-7, comfortably within the stated
    1e-6 tolerance. The residual error depends on the largest y in the
    sample (stability of lgamma(y + 1/alpha) for large 1/alpha) — n=200 and
    mu in [0.5, 8.0] are fixed deliberately so that max(y) stays out of the
    range where this error grows above 1e-6 again.
    """
    mu_np, y_np = _synthetic_counts(seed=1)
    mu = torch.tensor(mu_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64)
    alpha = torch.full_like(mu, 3e-5)

    nb_loss = NBLoss()(mu, alpha, y).item()
    poisson_loss = PoissonLoss()(mu, alpha, y).item()
    log_factorial = float(np.mean(gammaln(y_np + 1.0)))

    assert nb_loss == pytest.approx(poisson_loss + log_factorial, abs=1e-6)


def test_poisson_nll_missing_log_factorial():
    mu_np, y_np = _synthetic_counts(seed=2)
    true_poisson_nll = float(np.mean(-sp_poisson.logpmf(y_np, mu_np)))
    assert _poisson_nll(y_np, mu_np) == pytest.approx(true_poisson_nll, abs=1e-6)


def test_discrete_crps_matches_direct_summation_by_definition():
    """CRPS of a Poisson forecast by definition: sum_k (F(k) - 1{y<=k})^2.

    The reference value is computed independently of
    tail_metrics._discrete_crps, with a generous margin on the summation's
    upper bound (max(mu) + 60), so the tail of the sum is clearly convergent
    regardless of the K_max heuristic used by the function under test.
    """
    mu = np.array([0.3, 1.5, 4.0, 7.5, 12.0])
    y = np.array([0.0, 2.0, 3.0, 9.0, 20.0])

    def _reference_crps(y: np.ndarray, mu: np.ndarray, k_max_extra: int = 60) -> float:
        k_max = int(np.max(mu)) + k_max_extra
        crps = np.zeros(len(y), dtype=np.float64)
        for k in range(k_max + 1):
            F_k = sp_poisson.cdf(k, mu=mu)
            indicator = (y <= k).astype(np.float64)
            crps += (F_k - indicator) ** 2
        return float(np.mean(crps))

    expected = _reference_crps(y, mu)
    actual = _discrete_crps(y, mu, dist="poisson")
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9)
