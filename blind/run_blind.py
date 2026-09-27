#!/usr/bin/env python3
"""The blind test of Section 4.4, in one command.

    python blind/run_blind.py                        # writes into blind/outputs/
    SPM_BLIND_DIR=/somewhere/else python blind/run_blind.py

What it does, in order:

1. combines the two committed catalogues - ``usgs_central_asia_raw.csv``
   (2010-2023, the one every other number in the paper is computed on) and
   ``usgs_2024_2026_raw.csv`` (the period downloaded after the analysis was
   frozen) - and rebuilds the panel on the same weekly axis and the same 17
   cells;
2. fits every model on the rows before 2024-01-01 and on nothing else, and
   forecasts 2024-01-01 .. 2026-09-07 without retraining;
3. scores all of them on those 2397 cell-weeks: level, Poisson deviance with
   and without the level correction, logarithmic score, CRPS, and the exact
   tail test at Y >= 2 and Y >= 3.

The result is ``measures_all.csv`` in the output directory: one row per model,
the table Section 4.4 reports.

Everything is offline - the catalogues are committed, nothing is downloaded -
and the whole run takes well under a minute.

It writes only inside the output directory, and refuses to start if that
directory is, or is inside, a published one - see ``blind/output_dir.py``. The
pipeline modules bind some of their own output paths when they are imported, so
those are redirected into the same place before any of them is imported; see
``tests/conftest.py`` for why that has to happen this early.
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from blind.output_dir import resolve_output_dir  # noqa: E402

try:
    OUT = resolve_output_dir(os.environ.get("SPM_BLIND_DIR"), REPO)
except ValueError as refused:
    raise SystemExit(str(refused))
OUT.mkdir(parents=True, exist_ok=True)

# Import-time output paths of the pipeline modules, pointed somewhere harmless:
# this driver calls their functions and never writes their tables, so the
# directory below stays empty. It exists because several modules build their
# output paths from these two variables the moment they are imported.
os.environ["SPM_OUTPUT_DIR"] = str(OUT / "unused-by-this-driver")
os.environ["SPM_PANEL_FILE"] = "unused-by-this-driver.csv"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import mean_poisson_deviance  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src import config  # noqa: E402
from src.data_ingestion import clean_earthquake_data  # noqa: E402
from src.dl_modeling import (  # noqa: E402
    BATCH_SIZE, EPOCHS, EarthquakeDataset, EarthquakeNet, LR, NBLoss, PATIENCE,
    PoissonLoss, STATIC_VAL_FRACTION, _encode_cells, _set_seed,
)
from src.etas_baseline import fit_etas_per_cell, predict_etas  # noqa: E402
from src.etas_simulation import (  # noqa: E402
    BASE_SEED, N_REALIZATIONS, WEEK_DURATION_DAYS, _simulate_week,
)
from src.grid_builder import build_spatiotemporal_grid, filter_to_completeness  # noqa: E402
from src.probabilistic_evaluation import (  # noqa: E402
    CRPS_TAIL_PAD, PredictiveDistribution, fit_dispersion,
)
from src.splitting import temporal_validation_split  # noqa: E402
from src.tail_test import _tail_test_row  # noqa: E402
from src.validation import _cell_dummy_columns, _fit_nb_mle_glm  # noqa: E402

M_C = 4.5
BLIND_START = pd.Timestamp("2024-01-01")
BLIND_END = pd.Timestamp("2026-09-07")
SEEDS = (42, 7, 123, 2024, 999)
FEATURES = ["Y_lag1", "mag_max_lag1", "mag_min_lag1",
            "mag_max_roll4", "count_roll12", "energy_roll8", "weeks_since_m45"]
MAX_HIST_BIN = 200
DEVICE = torch.device("cpu")

OLD_CATALOGUE = REPO / "data" / "raw" / "usgs_central_asia_raw.csv"
NEW_CATALOGUE = REPO / "data" / "raw" / "usgs_2024_2026_raw.csv"
PUBLISHED_PANEL = REPO / "data" / "processed" / "spatiotemporal_grid_mc45.csv"


# ── 1. catalogue and panel ────────────────────────────────────────────────────

def _add_cell_id(df: pd.DataFrame) -> pd.DataFrame:
    """The cell label used everywhere else, computed the way main.py computes it."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, format="mixed").dt.tz_localize(None)
    for axis, source, low in (("lat_grid", "latitude", config.BBOX["minlatitude"]),
                              ("lon_grid", "longitude", config.BBOX["minlongitude"])):
        df[axis] = (np.floor((df[source] - low) / config.GRID_SIZE) * config.GRID_SIZE
                    + low).round(2)
    df["cell_id"] = (df["lat_grid"].map(lambda x: f"{float(x):.2f}") + "_"
                     + df["lon_grid"].map(lambda x: f"{float(x):.2f}"))
    return df


def build_panels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train rows, blind rows, combined events at M >= 4.5)."""
    old = pd.read_csv(OLD_CATALOGUE)
    new = clean_earthquake_data(pd.read_csv(NEW_CATALOGUE))

    old_time = pd.to_datetime(old["time"], utc=True, format="mixed")
    new_time = pd.to_datetime(new["time"], utc=True, format="mixed")
    assert old_time.max() < new_time.min(), "the two catalogues overlap in time"

    combined = pd.concat([old, new], ignore_index=True)
    duplicates = combined.duplicated(subset=["time", "latitude", "longitude", "mag"]).sum()
    assert duplicates == 0, f"{duplicates} duplicate events between the catalogues"
    print(f"catalogue: {len(combined)} events ({len(old)} through 2023 + {len(new)} after)")

    combined = _add_cell_id(combined)
    events_45 = filter_to_completeness(combined.copy(), m_c=M_C)

    # The weekly axis is pinned on the M >= 3.0 catalogue, exactly as the main
    # pipeline pins it: otherwise the last week of the period would be lost and
    # the two runs would not be scored on the same weeks.
    weeks_30 = (filter_to_completeness(combined.copy(), m_c=3.0)["time"]
                .dt.to_period("W").dt.start_time)
    bounds = (weeks_30.min(), weeks_30.max())
    print(f"weekly axis pinned at 3.0: {bounds[0].date()} .. {bounds[1].date()}")

    panel = build_spatiotemporal_grid(
        events_45[["time", "latitude", "longitude", "mag"]], week_bounds=bounds)

    published_cells = sorted(pd.read_csv(PUBLISHED_PANEL)["cell_id"].unique())
    outside = events_45[(events_45["time"] >= BLIND_START)
                        & (~events_45["cell_id"].isin(published_cells))]
    print(f"cells: {len(published_cells)} published; "
          f"events after 2024 outside them: {len(outside)}")
    panel = panel[panel["cell_id"].isin(published_cells)].reset_index(drop=True)

    train = panel[panel["week"] < BLIND_START].reset_index(drop=True)
    blind = panel[(panel["week"] >= BLIND_START)
                  & (panel["week"] <= BLIND_END)].reset_index(drop=True)
    assert blind["week"].nunique() == 141 and len(blind) == 2397, (
        f"expected 141 weeks and 2397 rows, got {blind['week'].nunique()} and {len(blind)}")
    print(f"train: {len(train)} rows through {train['week'].max().date()}; "
          f"blind: {len(blind)} rows, {int(blind['Y'].sum())} events")
    return train, blind, events_45


# ── 2. linear models and ETAS ─────────────────────────────────────────────────

def fit_linear_and_etas(train: pd.DataFrame, blind: pd.DataFrame,
                        events_45: pd.DataFrame) -> tuple[dict[str, dict], dict]:
    """Returns (model name -> forecasts on train and blind, fitted ETAS parameters)."""
    y_train = train["Y"].astype(float).to_numpy()

    scaler = StandardScaler()
    X_train = sm.add_constant(
        scaler.fit_transform(train[FEATURES].fillna(0.0).astype(float)), has_constant="add")
    X_blind = sm.add_constant(
        scaler.transform(blind[FEATURES].fillna(0.0).astype(float)), has_constant="add")

    models: dict[str, dict] = {}

    poisson_fit = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit(
        disp=False, maxiter=500)
    models["Poisson_GLM"] = dict(
        family="poisson",
        pred_train=np.clip(np.asarray(poisson_fit.predict(X_train), float), 1e-9, None),
        pred_blind=np.clip(np.asarray(poisson_fit.predict(X_blind), float), 1e-9, None),
        alpha_train=None, alpha_blind=None)

    nb_fit, _, nb_alpha = _fit_nb_mle_glm(y_train, X_train)
    assert nb_fit is not None, "the NB GLM profile-likelihood grid failed"
    models["NB_GLM"] = dict(
        family="nb",
        pred_train=np.clip(np.asarray(nb_fit.predict(X_train), float), 1e-9, None),
        pred_blind=np.clip(np.asarray(nb_fit.predict(X_blind), float), 1e-9, None),
        alpha_train=np.full(len(train), nb_alpha), alpha_blind=np.full(len(blind), nb_alpha))

    dummies_train, dummies_blind = _cell_dummy_columns(train["cell_id"], blind["cell_id"])
    X_train_cell = sm.add_constant(np.hstack([X_train[:, 1:], dummies_train]),
                                   has_constant="add")
    X_blind_cell = sm.add_constant(np.hstack([X_blind[:, 1:], dummies_blind]),
                                   has_constant="add")
    cell_fit, _, cell_alpha = _fit_nb_mle_glm(y_train, X_train_cell)
    assert cell_fit is not None, "the NB GLM + cell profile-likelihood grid failed"
    models["NB_Enhanced_MLE_Cell"] = dict(
        family="nb",
        pred_train=np.clip(np.asarray(cell_fit.predict(X_train_cell), float), 1e-9, None),
        pred_blind=np.clip(np.asarray(cell_fit.predict(X_blind_cell), float), 1e-9, None),
        alpha_train=np.full(len(train), cell_alpha),
        alpha_blind=np.full(len(blind), cell_alpha))

    params = fit_etas_per_cell(events_45, BLIND_START, m_c=M_C)
    fallbacks = sum(1 for p in params.values() if p.get("fallback"))
    print(f"ETAS fitted on {len(params)} cells ({fallbacks} fell back to a constant rate)")

    def etas_on(rows: pd.DataFrame) -> np.ndarray:
        merged = rows[["cell_id", "week"]].merge(
            predict_etas(params, rows[["cell_id", "week"]], events_45, m_c=M_C),
            on=["cell_id", "week"], how="left")
        assert merged["lambda_pred"].notna().all(), "ETAS left rows unforecast"
        return np.clip(merged["lambda_pred"].to_numpy(float), 1e-9, None)

    etas_train, etas_blind = etas_on(train), etas_on(blind)
    models["ETAS_Per_Cell"] = dict(family="poisson", pred_train=etas_train,
                                   pred_blind=etas_blind, alpha_train=None, alpha_blind=None)

    # The same intensity with one dispersion parameter, fitted on the training
    # period only - a baseline that had seen the blind period could not lose
    # honestly.
    alpha_etas = fit_dispersion(y_train, etas_train)
    print(f"ETAS dispersion fitted on the training period: alpha = {alpha_etas:.4f}")
    if alpha_etas > 0:
        models["ETAS_NB"] = dict(family="nb", pred_train=etas_train, pred_blind=etas_blind,
                                 alpha_train=np.full(len(train), alpha_etas),
                                 alpha_blind=np.full(len(blind), alpha_etas))
    return models, params


# ── 3. the two networks ───────────────────────────────────────────────────────

def _train_network(train: pd.DataFrame, blind: pd.DataFrame, loss_kind: str, seed: int):
    """One training run, with the static protocol's settings.

    ``dl_modeling._train_one_variant`` cannot be reused directly: it carves its
    own chronological 80/20 split out of whatever panel it is handed, and the
    split needed here is train-through-2023 against the blind period. The loop
    below is that function's loop with the same optimizer, schedule and
    early-stopping rule, over this split instead.
    """
    _set_seed(seed)
    fit_idx, val_idx = temporal_validation_split(
        train["week"].to_numpy(), val_fraction=STATIC_VAL_FRACTION)
    fit_df = train.iloc[fit_idx].reset_index(drop=True)
    val_df = train.iloc[val_idx].reset_index(drop=True)

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(fit_df[FEATURES].fillna(0.0).astype(float)).astype(np.float32)
    X_val = scaler.transform(val_df[FEATURES].fillna(0.0).astype(float)).astype(np.float32)
    X_blind = scaler.transform(blind[FEATURES].fillna(0.0).astype(float)).astype(np.float32)
    X_train_all = scaler.transform(train[FEATURES].fillna(0.0).astype(float)).astype(np.float32)

    y_fit = fit_df["Y"].astype(float).to_numpy(dtype=np.float32)
    y_val = val_df["Y"].astype(float).to_numpy(dtype=np.float32)

    cells_fit, n_cells = _encode_cells(fit_df["cell_id"], fit_df["cell_id"])
    cells_val, _ = _encode_cells(fit_df["cell_id"], val_df["cell_id"])
    cells_blind, _ = _encode_cells(fit_df["cell_id"], blind["cell_id"])
    cells_train_all, _ = _encode_cells(fit_df["cell_id"], train["cell_id"])

    def loader(X, cells, y=None, shuffle=False):
        target = np.zeros(len(cells), dtype=np.float32) if y is None else y
        return DataLoader(EarthquakeDataset(X, cells, target),
                          batch_size=BATCH_SIZE, shuffle=shuffle)

    fit_loader = loader(X_fit, cells_fit, y_fit, shuffle=True)
    val_loader = loader(X_val, cells_val, y_val)

    model = EarthquakeNet(num_cells=n_cells, input_dim=len(FEATURES), emb_dim=8).to(DEVICE)
    criterion = NBLoss() if loss_kind == "nb" else PoissonLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, cb, yb in fit_loader:
            optimiser.zero_grad(set_to_none=True)
            mu, alpha = model(xb.to(DEVICE), cb.to(DEVICE))
            criterion(mu, alpha, yb.to(DEVICE)).backward()
            optimiser.step()

        model.eval()
        total, seen = 0.0, 0
        with torch.no_grad():
            for xb, cb, yb in val_loader:
                mu, alpha = model(xb.to(DEVICE), cb.to(DEVICE))
                loss = criterion(mu, alpha, yb.to(DEVICE))
                total += float(loss.item()) * yb.shape[0]
                seen += yb.shape[0]
        val_loss = total / max(seen, 1)
        schedule.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val, best_state, bad_epochs = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict(X, cells):
        mus, alphas = [], []
        with torch.no_grad():
            for xb, cb, _ in loader(X, cells):
                mu, alpha = model(xb.to(DEVICE), cb.to(DEVICE))
                mus.append(mu.cpu().numpy())
                alphas.append(alpha.cpu().numpy())
        return (np.clip(np.concatenate(mus).astype(np.float64), 1e-9, None),
                np.concatenate(alphas).astype(np.float64))

    return predict(X_train_all, cells_train_all), predict(X_blind, cells_blind)


def run_networks(train: pd.DataFrame, blind: pd.DataFrame) -> dict[str, dict]:
    """Both networks, five seeds each, averaged row by row.

    This is the one place in the paper where the seeds are averaged as
    forecasts rather than as scores (Section 4.1 says so): there is a single
    blind period, so there is no week-by-week score to average first.
    """
    out: dict[str, dict] = {}
    for loss_kind, name in (("nb", "Hybrid_DL_Enhanced"),
                            ("poisson", "Neural_Poisson_Enhanced")):
        train_mu, train_alpha, blind_mu, blind_alpha = [], [], [], []
        for seed in SEEDS:
            started = time.time()
            (mu_t, alpha_t), (mu_b, alpha_b) = _train_network(train, blind, loss_kind, seed)
            train_mu.append(mu_t); train_alpha.append(alpha_t)
            blind_mu.append(mu_b); blind_alpha.append(alpha_b)
            print(f"  {name} seed {seed}: {time.time() - started:.0f}s, "
                  f"mean blind forecast {mu_b.mean():.4f}")
        blind_stack = np.stack(blind_mu)
        out[name] = dict(
            family="nb" if loss_kind == "nb" else "poisson",
            pred_train=np.stack(train_mu).mean(axis=0),
            alpha_train=np.stack(train_alpha).mean(axis=0) if loss_kind == "nb" else None,
            pred_blind=blind_stack.mean(axis=0),
            alpha_blind=np.stack(blind_alpha).mean(axis=0) if loss_kind == "nb" else None,
            pred_seed_std=blind_stack.std(axis=0),
            per_seed=blind_stack,
        )
    return out


# ── 4. within-week branching simulation ───────────────────────────────────────

def simulate(rows: pd.DataFrame, events_45: pd.DataFrame, params: dict,
             keep_histogram: bool) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The ETAS chain simulation of Section 4.2, on the rows it is given.

    ``etas_simulation.simulate_etas_predictions`` makes its own 80/20 split, so
    the per-row simulator is driven here instead, with the module's own
    conventions: one RNG stream per cell spawned from a single SeedSequence in
    sorted cell order, magnitudes drawn from the training catalogue.
    """
    intensity = rows[["cell_id", "week"]].merge(
        predict_etas(params, rows[["cell_id", "week"]], events_45, m_c=M_C),
        on=["cell_id", "week"], how="left")
    assert intensity["lambda_pred"].notna().all()
    work = rows.copy()
    work["lambda_pred"] = intensity["lambda_pred"].clip(lower=1e-9).to_numpy()

    history = {str(cell): (group.sort_values("time")["time"].to_numpy(),
                           group.sort_values("time")["mag"].to_numpy(dtype=np.float64))
               for cell, group in events_45.groupby("cell_id")}
    empty = (np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.float64))
    train_mags = events_45.loc[(events_45["time"] < BLIND_START)
                               & (events_45["mag"] >= M_C), "mag"].to_numpy(dtype=np.float64)
    assert train_mags.size > 0

    cells = sorted(work["cell_id"].unique())
    missing = [c for c in cells if c not in params]
    assert not missing, f"no fitted ETAS parameters for {missing}"
    streams = dict(zip(cells, np.random.SeedSequence(BASE_SEED).spawn(len(cells))))

    summaries, histograms = [], []
    for cell in cells:
        p = params[cell]
        origin = p["origin"]
        times, mags = history.get(cell, empty)
        days = (times - pd.Timestamp(origin).to_datetime64()) / np.timedelta64(1, "D")
        rng = np.random.default_rng(streams[cell])
        pointer = 0
        for _, row in work[work["cell_id"] == cell].sort_values("week").iterrows():
            start = (pd.Timestamp(row["week"]) - origin).total_seconds() / 86400.0
            while pointer < len(days) and days[pointer] < start:
                pointer += 1
            simulated, _ = _simulate_week(
                p, days[:pointer], mags[:pointer], start, start + WEEK_DURATION_DAYS,
                float(row["lambda_pred"]), train_mags, rng, N_REALIZATIONS, M_C,
                enable_offspring=True)
            counts = simulated.astype(np.float64)
            observed = float(row["Y"])
            summaries.append({
                "cell_id": cell, "week": pd.Timestamp(row["week"]), "y_true": observed,
                "lambda_pred": float(row["lambda_pred"]), "sim_mean": float(counts.mean()),
                "p_ge1": float((counts >= 1).mean()), "p_ge2": float((counts >= 2).mean()),
                "p_ge3": float((counts >= 3).mean()), "p_ge5": float((counts >= 5).mean()),
            })
            if keep_histogram:
                histogram = np.bincount(np.minimum(simulated, MAX_HIST_BIN),
                                        minlength=MAX_HIST_BIN + 1)
                histograms.append([cell, pd.Timestamp(row["week"]), observed]
                                  + histogram.tolist())

    summary_df = pd.DataFrame(summaries)
    if not keep_histogram:
        return summary_df, None
    columns = ["cell_id", "week", "y_true"] + [f"bin_{i}" for i in range(MAX_HIST_BIN + 1)]
    return summary_df, pd.DataFrame(histograms, columns=columns)


# ── 5. measures ───────────────────────────────────────────────────────────────

def _measure_row(name: str, family: str, y_blind: np.ndarray, pred_blind: np.ndarray,
                 alpha_blind: np.ndarray | None, y_train: np.ndarray,
                 pred_train: np.ndarray) -> dict:
    correction = float(y_train.sum() / pred_train.sum())
    distribution = PredictiveDistribution(name, family, pred_blind, alpha_blind)
    row = {
        "model": name, "family": family, "n_obs": len(y_blind),
        "level_pred_over_actual": float(pred_blind.sum() / y_blind.sum()),
        "mean_poisson_deviance_raw": float(mean_poisson_deviance(y_blind, pred_blind)),
        "level_correction_factor_train": correction,
        "mean_poisson_deviance_corrected": float(mean_poisson_deviance(
            y_blind, np.clip(pred_blind * correction, 1e-9, None))),
        "log_score": float(distribution.log_score(y_blind).mean()),
        "crps": float(distribution.crps_terms(
            y_blind, int(y_blind.max()) + CRPS_TAIL_PAD).sum(axis=1).mean()),
    }
    for k in (2, 3):
        test = _tail_test_row(name, k, distribution.exceedance(k), int((y_blind >= k).sum()))
        row[f"expected_weeks_Yge{k}"] = test["expected_count"]
        row[f"observed_weeks_Yge{k}"] = test["observed_count"]
        row[f"p_two_sided_Yge{k}"] = test["p_two_sided"]
    return row


def _simulation_row(sim_blind: pd.DataFrame, sim_train: pd.DataFrame,
                    histograms: pd.DataFrame) -> dict:
    """The simulation has no closed-form family: its scores come from the draws.

    The logarithmic score needs P(Y = y) for the observed count, and in 10,000
    draws a rare count can simply never appear. Those rows get a half-count
    floor rather than an infinite score, and how many needed it is reported
    next to the number.
    """
    y = sim_blind["y_true"].to_numpy(float)
    pred = np.clip(sim_blind["sim_mean"].to_numpy(float), 1e-9, None)
    correction = float(sim_train["y_true"].sum() / sim_train["sim_mean"].sum())

    bins = [c for c in histograms.columns if c.startswith("bin_")]
    counts = histograms[bins].to_numpy(dtype=np.float64)
    exact = counts[np.arange(len(y)), np.minimum(y.astype(int), len(bins) - 1)]
    floored = int((exact == 0).sum())
    cdf = np.cumsum(counts, axis=1) / N_REALIZATIONS
    indicator = (y[:, None] <= np.arange(len(bins))[None, :]).astype(np.float64)

    row = {
        "model": "ETAS_Branching_Sim", "family": f"empirical_{N_REALIZATIONS}_sims",
        "n_obs": len(y),
        "level_pred_over_actual": float(pred.sum() / y.sum()),
        "mean_poisson_deviance_raw": float(mean_poisson_deviance(y, pred)),
        "level_correction_factor_train": correction,
        "mean_poisson_deviance_corrected": float(mean_poisson_deviance(
            y, np.clip(pred * correction, 1e-9, None))),
        "log_score": float(np.mean(-np.log(np.maximum(exact, 0.5) / N_REALIZATIONS))),
        "crps": float(((cdf - indicator) ** 2).sum(axis=1).mean()),
        "n_rows_log_score_floored": floored,
    }
    for k in (2, 3):
        test = _tail_test_row("ETAS_Branching_Sim", k, sim_blind[f"p_ge{k}"].to_numpy(float),
                              int((y >= k).sum()))
        row[f"expected_weeks_Yge{k}"] = test["expected_count"]
        row[f"observed_weeks_Yge{k}"] = test["observed_count"]
        row[f"p_two_sided_Yge{k}"] = test["p_two_sided"]
    return row


def main() -> int:
    started = time.time()
    print(f"blind test, output directory {OUT}")

    train, blind, events_45 = build_panels()
    y_train = train["Y"].astype(float).to_numpy()
    y_blind = blind["Y"].astype(float).to_numpy()

    print("\nfitting the linear models and ETAS on data through 2023 ...")
    models, etas_params = fit_linear_and_etas(train, blind, events_45)

    print("\ntraining the networks (five seeds each) ...")
    models.update(run_networks(train, blind))

    print("\nsimulating within-week chains on the blind period ...")
    sim_blind, histograms = simulate(blind, events_45, etas_params, keep_histogram=True)
    print("simulating within-week chains on the training period "
          "(the level-correction factor needs them) ...")
    sim_train, _ = simulate(train, events_45, etas_params, keep_histogram=False)

    print("\nscoring ...")
    rows = [_measure_row(name, m["family"], y_blind, m["pred_blind"], m["alpha_blind"],
                         y_train, m["pred_train"])
            for name, m in models.items()]
    rows.append(_simulation_row(sim_blind, sim_train, histograms))
    measures = pd.DataFrame(rows)

    long = pd.concat([
        pd.DataFrame({"cell_id": blind["cell_id"], "week": blind["week"],
                      "y_true": y_blind, "model": name, "family": m["family"],
                      "pred": m["pred_blind"],
                      "alpha": m["alpha_blind"] if m["alpha_blind"] is not None else np.nan})
        for name, m in models.items()], ignore_index=True)

    blind.to_csv(OUT / "blind_panel.csv", index=False)
    long.to_csv(OUT / "blind_predictions.csv", index=False)
    sim_blind.to_csv(OUT / "etas_sim_blind_predictions.csv", index=False)
    measures.to_csv(OUT / "measures_all.csv", index=False)
    for name in ("Hybrid_DL_Enhanced", "Neural_Poisson_Enhanced"):
        per_seed = pd.DataFrame(models[name]["per_seed"].T,
                                columns=[f"seed_{s}" for s in SEEDS])
        per_seed.insert(0, "cell_id", blind["cell_id"].to_numpy())
        per_seed.insert(1, "week", blind["week"].to_numpy())
        per_seed.insert(2, "y_true", y_blind)
        per_seed.to_csv(OUT / f"dl_{name}_blind_per_seed.csv", index=False)

    pd.set_option("display.width", 200)
    shown = ["model", "level_pred_over_actual", "mean_poisson_deviance_raw",
             "expected_weeks_Yge3", "observed_weeks_Yge3", "p_two_sided_Yge3"]
    print("\n" + measures[shown].to_string(index=False))
    print(f"\nwritten to {OUT} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
