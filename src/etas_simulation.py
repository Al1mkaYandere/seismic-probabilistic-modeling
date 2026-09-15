"""Branching-process simulation of ETAS aftershock chains within a forecast week.

``etas_baseline.predict_etas`` treats the number of events in a forecast week
as Poisson with a mean obtained by integrating the conditional intensity over
that week. That mean already counts the aftershocks a PRE-week event is
expected to trigger during the week, but it has no term for aftershocks
triggered by an event that only happens DURING the week: a shock on Tuesday
can still trigger its own shocks before Sunday, and the static prediction
never sees them, because it stops at the mean of a Poisson distribution and
never asks what a single realization of the process actually contains.

This module simulates the branching process itself, conditional on the same
event history ``predict_etas`` uses, so within-week chains are counted rather
than ignored. It reuses the fitted parameters (``fit_etas_per_cell``) and the
static train/test split of ``etas_baseline.run_etas_static`` - the model
being simulated is the published ETAS_Per_Cell baseline, not a new fit.

Model, one forecast week [week_start, week_start + 7d) at a time
------------------------------------------------------------------
Generation 0 is the union of two independent Poisson processes over the week:
a homogeneous background at rate ``mu``, and the aftershocks triggered by
events that happened BEFORE the week (their expected count over the week is
exactly the Omori integral ``etas_baseline._predict_cell`` already computes,
one term per past event). By the superposition property of independent
Poisson processes, their total count is Poisson with mean
``mu * 7 + sum(per-event integrals)`` - which is ``lambda_pred`` itself - so
generation 0 is drawn as a single Poisson variate per realization, and each
event's origin (background, or a specific past event) is assigned afterwards
with probability proportional to that process's share of the mean; only the
origin's own Omori kernel shapes WHEN in the week the event falls.

Every event from generation 0 onward - background or triggered, at any
generation - can itself trigger further events before the week ends: its own
expected offspring count is the same Omori integral, now taken from its own
(simulated) time to the end of the week. Offspring counts and times are drawn
the same way at every generation, so the recursion has one rule, not one
rule per generation. It stops on its own once a generation produces zero
events - which realistic (sub-critical) parameters make almost certain
within a few generations - or when a realization hits the per-realization
event cap.

Magnitudes of every simulated event, at any generation, are drawn with
replacement from the observed training-catalogue magnitudes (all cells,
m >= M_c, before train_end): the fitted model gives no other information
about how large a hypothetical future event would be.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.etas_baseline import fit_etas_per_cell, predict_etas

logger = logging.getLogger(__name__)

N_REALIZATIONS: int = 10_000
"""Monte Carlo realizations per (cell, week) row."""

MAX_EVENTS_PER_REALIZATION: int = 1_000
"""Safety cap: a realization is truncated (not further simulated) once its
event count reaches this many. Only matters for a runaway (super-critical)
branching ratio; the fraction of realizations that ever hit it is reported by
``run_etas_simulation`` and is expected to be far below the cap's purpose."""

MAX_GENERATIONS: int = 200
"""Depth safety net alongside the event cap, for the same runaway scenario."""

BASE_SEED: int = 42
WEEK_DURATION_DAYS: float = 7.0
CAPPED_FRACTION_WARN: float = 0.001  # 0.1%


def _static_split(panel_df: pd.DataFrame) -> tuple[pd.Timestamp, np.ndarray]:
    """Same 80/20 weekly split as ``etas_baseline.run_etas_static``.

    Returns ``(train_end, test_weeks)``. Reproduced here rather than imported
    because ``run_etas_static`` folds the split into fitting, scoring and
    saving in one function and returns only aggregate metrics - the split
    itself is a dozen lines of arithmetic on ``panel_df["week"]``.
    ``tests/test_etas_simulation.py`` checks this against
    ``etas_test_predictions.csv`` so the two cannot silently drift apart.
    """
    unique_weeks = np.sort(panel_df["week"].dropna().unique())
    split_w = int(np.floor(0.8 * len(unique_weeks)))
    train_weeks = unique_weeks[:split_w]
    test_weeks = unique_weeks[split_w:]
    train_end = pd.Timestamp(train_weeks[-1])
    if train_end.tzinfo is not None:
        train_end = train_end.tz_localize(None)
    train_end = train_end + pd.Timedelta(days=7)
    return train_end, test_weeks


def _omori_rate(
    parent_t: np.ndarray,
    parent_m: np.ndarray,
    t_lo,
    t_end: float,
    K: float,
    a: float,
    c: float,
    p: float,
    m_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected offspring count of each parent within [max(t_lo, parent_t), t_end).

    ``t_lo`` is either a scalar (the week start, for parents from BEFORE the
    week) or an array matching ``parent_t`` (a parent's own time, for
    offspring of an event simulated inside the week - such a parent already
    sits at or after the week start, so ``max`` is a no-op and this reduces to
    the interval [parent_t, t_end)). Same closed-form integral of the Omori
    kernel (s - t_i + c)^-p as ``etas_baseline._predict_cell``, evaluated per
    parent instead of summed, because the simulation needs individual rates
    to draw individual offspring counts and times.
    """
    s0 = np.maximum(t_lo, parent_t)
    dt0 = s0 - parent_t + c
    dt1 = t_end - parent_t + c
    coef = K * np.exp(a * (parent_m - m_c))
    if abs(p - 1.0) < 1e-12:
        integral = np.log(dt1) - np.log(dt0)
    else:
        integral = (dt1 ** (1.0 - p) - dt0 ** (1.0 - p)) / (1.0 - p)
    rate = np.where(s0 < t_end, coef * integral, 0.0)
    return rate, s0


def _sample_omori_times(
    parent_t: np.ndarray,
    s0: np.ndarray,
    t_end: float,
    c: float,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inverse-CDF draw of one event time in [s0, t_end) per parent.

    The Omori kernel (s - parent_t + c)^-p, restricted to [s0, t_end) and
    normalised to that interval, is exactly what ``_omori_rate`` integrates to
    get the expected count; this draws WHERE in that interval each such event
    actually falls.
    """
    dt0 = s0 - parent_t + c
    dt1 = t_end - parent_t + c
    u = rng.uniform(0.0, 1.0, size=parent_t.shape)
    if abs(p - 1.0) < 1e-12:
        y = dt0 * (dt1 / dt0) ** u
    else:
        y = (dt0 ** (1.0 - p) + u * (dt1 ** (1.0 - p) - dt0 ** (1.0 - p))) ** (1.0 / (1.0 - p))
    return parent_t + (y - c)


def _spawn_offspring(
    parent_t: np.ndarray,
    parent_m: np.ndarray,
    parent_real_ids: np.ndarray,
    t_end: float,
    K: float,
    a: float,
    c: float,
    p: float,
    m_c: float,
    train_mags: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One generation of offspring for every given parent, all at once.

    Every parent here is an event already simulated INSIDE the week -
    background, history-triggered, or itself an earlier offspring - so its
    own time already sits at or after the week start, and what is left of its
    productive window is [parent_t, t_end), not [week_start, t_end). This is
    the one rule ``_simulate_week`` recurses with at every generation from
    generation 1 onward; it is split out on its own so a test can check the
    rate it produces against the closed-form Omori integral taken from the
    PARENT's own time, independently of everything else ``_simulate_week``
    does around it.
    """
    rate, _ = _omori_rate(parent_t, parent_m, parent_t, t_end, K, a, c, p, m_c)
    counts = rng.poisson(np.clip(rate, 0.0, None))
    total_children = int(counts.sum())
    if total_children == 0:
        empty_ids_dtype = parent_real_ids.dtype if parent_real_ids.size else np.int64
        return (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=empty_ids_dtype))

    child_real_ids = np.repeat(parent_real_ids, counts)
    child_parent_t = np.repeat(parent_t, counts)
    child_times = _sample_omori_times(child_parent_t, child_parent_t, t_end, c, p, rng)
    child_mags = rng.choice(train_mags, size=total_children, replace=True)
    return child_times, child_mags, child_real_ids


def _simulate_week(
    params: dict,
    hist_t: np.ndarray,
    hist_m: np.ndarray,
    t_start: float,
    t_end: float,
    lambda_pred: float,
    train_mags: np.ndarray,
    rng: np.random.Generator,
    n_sims: int,
    m_c: float,
    enable_offspring: bool = True,
) -> tuple[np.ndarray, int]:
    """Simulate ``Y`` for one (cell, week) row.

    Returns ``(Y, n_capped)``: ``Y`` has one entry per realization, capped at
    ``MAX_EVENTS_PER_REALIZATION``; ``n_capped`` is how many realizations hit
    that cap.

    ``enable_offspring=False`` stops after generation 0 (background plus
    events triggered by pre-week history) - used only by tests, to isolate
    that generation 0 alone reproduces the plain Poisson(lambda_pred)
    ``predict_etas`` already promises, with no within-week chains at all.
    """
    K = float(params.get("K", 0.0))
    c = float(params.get("c", 0.01))
    p = float(params.get("p", 1.1))
    a = float(params.get("a", 1.0))
    mu = float(params["mu"])
    week_dur = t_end - t_start

    n0 = rng.poisson(max(lambda_pred, 0.0), size=n_sims)
    Y = n0.astype(np.int64)

    if not enable_offspring or K <= 0:
        return np.minimum(Y, MAX_EVENTS_PER_REALIZATION), int((Y >= MAX_EVENTS_PER_REALIZATION).sum())

    total0 = int(n0.sum())
    if total0 == 0:
        return Y, 0

    hist_rate = np.zeros(0)
    hist_s0 = np.zeros(0)
    if len(hist_t) > 0:
        hist_rate, hist_s0 = _omori_rate(hist_t, hist_m, t_start, t_end, K, a, c, p, m_c)

    weights = np.clip(np.concatenate(([max(mu * week_dur, 0.0)], hist_rate)), 0.0, None)
    total_w = float(weights.sum())
    if total_w <= 0:
        # lambda_pred was effectively 0, so n0 must have been all zeros too.
        return Y, 0
    probs = weights / total_w
    comp = rng.choice(len(weights), size=total0, p=probs)
    is_bg = comp == 0

    gen_times = np.empty(total0, dtype=np.float64)
    gen_times[is_bg] = t_start + rng.uniform(0.0, week_dur, size=int(is_bg.sum()))
    if len(hist_t) > 0 and (~is_bg).any():
        hidx = comp[~is_bg] - 1
        gen_times[~is_bg] = _sample_omori_times(hist_t[hidx], hist_s0[hidx], t_end, c, p, rng)
    gen_mags = rng.choice(train_mags, size=total0, replace=True)
    gen_real_ids = np.repeat(np.arange(n_sims), n0)

    active = Y < MAX_EVENTS_PER_REALIZATION
    depth = 0
    while gen_times.size > 0 and depth < MAX_GENERATIONS:
        depth += 1
        keep = active[gen_real_ids]
        if not keep.all():
            gen_times = gen_times[keep]
            gen_mags = gen_mags[keep]
            gen_real_ids = gen_real_ids[keep]
        if gen_times.size == 0:
            break

        child_times, child_mags, child_real_ids = _spawn_offspring(
            gen_times, gen_mags, gen_real_ids, t_end, K, a, c, p, m_c, train_mags, rng,
        )
        if child_times.size == 0:
            break

        Y += np.bincount(child_real_ids, minlength=n_sims).astype(np.int64)
        active = Y < MAX_EVENTS_PER_REALIZATION

        gen_times, gen_mags, gen_real_ids = child_times, child_mags, child_real_ids

    n_capped = int((Y >= MAX_EVENTS_PER_REALIZATION).sum())
    return np.minimum(Y, MAX_EVENTS_PER_REALIZATION), n_capped


def simulate_etas_predictions(
    events_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    m_c: float | None = None,
    n_realizations: int = N_REALIZATIONS,
    base_seed: int = BASE_SEED,
    enable_offspring: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Simulate within-week ETAS chains on the static protocol's test split.

    Parameters mirror ``etas_baseline.run_etas_static``: ``events_df`` is the
    full raw catalogue (already completeness-filtered by the caller, as the
    pipeline does), ``panel_df`` the processed (cell_id, week, Y) panel. The
    fit uses exactly the same ``train_end`` and the same events as the
    published ETAS_Per_Cell baseline.

    Returns ``(predictions, capped_fraction)``. ``predictions`` has one row
    per (cell_id, week) test observation with columns ``cell_id``, ``week``,
    ``Y`` (observed count), ``lambda_pred`` (``predict_etas``'s own static
    prediction), ``sim_mean``, ``p_ge1``..``p_ge5`` (simulated
    P(Y_sim >= k)), and ``p_eq_obs`` (simulated P(Y_sim == the observed Y)).
    ``capped_fraction`` is the share of (row, realization) pairs across the
    whole run that hit ``MAX_EVENTS_PER_REALIZATION``.

    Each cell gets its own reproducible random stream, spawned from a single
    ``SeedSequence(base_seed)`` in sorted cell_id order; a cell's own weeks
    are simulated in chronological order from that one stream. The global
    ``numpy.random`` state is never touched.
    """
    m_c = float(config.M_C) if m_c is None else float(m_c)
    train_end, test_weeks = _static_split(panel_df)

    events_work = events_df[["time", "mag", "cell_id"]].copy()
    events_work["time"] = pd.to_datetime(events_work["time"])
    if events_work["time"].dt.tz is not None:
        events_work["time"] = events_work["time"].dt.tz_localize(None)

    params_by_cell = fit_etas_per_cell(events_df, train_end, m_c)

    train_mags = events_work.loc[
        (events_work["time"] < train_end) & (events_work["mag"] >= m_c), "mag"
    ].to_numpy(dtype=np.float64)
    if train_mags.size == 0:
        raise ValueError(
            "no training-catalogue magnitudes at or above M_c before train_end: "
            "simulated events would have nothing to draw a magnitude from"
        )

    test_panel = panel_df[panel_df["week"].isin(test_weeks)][["cell_id", "week", "Y"]].copy()
    test_panel["cell_id"] = test_panel["cell_id"].astype(str)
    test_panel["week"] = pd.to_datetime(test_panel["week"])
    lambda_df = predict_etas(params_by_cell, test_panel[["cell_id", "week"]], events_df, m_c)
    test_panel = test_panel.merge(lambda_df, on=["cell_id", "week"], how="left")
    test_panel["lambda_pred"] = test_panel["lambda_pred"].fillna(0.1).clip(lower=1e-9)

    events_by_cell: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cid, grp in events_work.groupby("cell_id"):
        grp_sorted = grp.sort_values("time")
        events_by_cell[str(cid)] = (
            grp_sorted["time"].to_numpy(),
            grp_sorted["mag"].to_numpy(dtype=np.float64),
        )
    empty_times = np.array([], dtype="datetime64[ns]")
    empty_mags = np.array([], dtype=np.float64)

    cell_ids_sorted = sorted(test_panel["cell_id"].unique())
    missing = [cid for cid in cell_ids_sorted if cid not in params_by_cell]
    if missing:
        raise ValueError(
            f"no fitted ETAS parameters for cell(s) {missing}: predict_etas falls "
            "back to a flat rate for these, but the branching simulation has no "
            "parameters to simulate a chain from. Silently skipping them would "
            "drop their rows from etas_simulation_predictions.csv instead of "
            "failing loudly."
        )

    seed_seq = np.random.SeedSequence(base_seed)
    cell_seeds = dict(zip(cell_ids_sorted, seed_seq.spawn(len(cell_ids_sorted))))

    rows: list[dict] = []
    total_capped = 0
    total_pairs = 0
    for cid in cell_ids_sorted:
        p = params_by_cell[cid]
        origin = p["origin"]
        origin_dt64 = pd.Timestamp(origin).to_datetime64()
        ev_times, ev_mags = events_by_cell.get(cid, (empty_times, empty_mags))
        ev_t_days = (ev_times - origin_dt64) / np.timedelta64(1, "D")
        n_events = len(ev_t_days)

        cell_rows = test_panel[test_panel["cell_id"] == cid].sort_values("week")
        rng = np.random.default_rng(cell_seeds[cid])
        ptr = 0
        for _, row in cell_rows.iterrows():
            wk = pd.Timestamp(row["week"])
            t_start = (wk - origin).total_seconds() / 86400.0
            t_end = t_start + WEEK_DURATION_DAYS
            while ptr < n_events and ev_t_days[ptr] < t_start:
                ptr += 1
            hist_t, hist_m = ev_t_days[:ptr], ev_mags[:ptr]

            Y_sim, n_capped = _simulate_week(
                p, hist_t, hist_m, t_start, t_end, float(row["lambda_pred"]),
                train_mags, rng, n_realizations, m_c, enable_offspring=enable_offspring,
            )
            total_capped += n_capped
            total_pairs += n_realizations

            y_obs = float(row["Y"])
            Y_sim_f = Y_sim.astype(np.float64)
            rows.append({
                "cell_id": cid, "week": wk, "Y": y_obs,
                "lambda_pred": float(row["lambda_pred"]),
                "sim_mean": float(Y_sim_f.mean()),
                "p_ge1": float((Y_sim_f >= 1).mean()),
                "p_ge2": float((Y_sim_f >= 2).mean()),
                "p_ge3": float((Y_sim_f >= 3).mean()),
                "p_ge5": float((Y_sim_f >= 5).mean()),
                "p_eq_obs": float((Y_sim_f == y_obs).mean()),
            })

    capped_fraction = total_capped / total_pairs if total_pairs else 0.0
    return pd.DataFrame(rows), capped_fraction


def run_etas_simulation(
    events_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    m_c: float | None = None,
) -> pd.DataFrame:
    """Simulate and save ``etas_simulation_predictions.csv`` in the current output dir.

    Meant to run last in the pipeline (see ``main.py``): it only reads the
    raw catalogue and the processed panel, and writes one new file, so it
    cannot move any number produced by an earlier step.
    """
    predictions, capped_fraction = simulate_etas_predictions(events_df, panel_df, m_c)

    # Logged at WARNING regardless of outcome (not just when the threshold is
    # crossed) because main.py raises the root logger to WARNING - an INFO
    # line here would never reach the run's own log.
    logger.warning(
        "ETAS simulation: %.4f%% of (row, realization) pairs hit the %d-event cap.",
        100.0 * capped_fraction, MAX_EVENTS_PER_REALIZATION,
    )
    if capped_fraction > CAPPED_FRACTION_WARN:
        logger.warning(
            "ETAS simulation: capped fraction exceeds the 0.1%% reporting threshold; "
            "the tail of the simulated distribution is not trustworthy as reported."
        )

    out_path = config.OUTPUT_DIR / "etas_simulation_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)
    logger.info("Saved ETAS simulation predictions to %s (%d rows)", out_path, len(predictions))
    return predictions
