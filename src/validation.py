"""Walk-Forward Validation: NB GLM (MLE alpha), Hybrid DL NB, Neural Poisson, ETAS per-cell.

Walk-Forward protocol:
- For each test year Y in 2018–2023, train on all years < Y, test on year Y.
- DL models use a chronological validation cut (last 15 % of training rows).
- NB GLM uses profile-likelihood MLE for alpha per fold.
"""

from __future__ import annotations

import logging
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.dl_modeling import EarlyStopping, HybridModel, NegativeBinomialLoss, PoissonLoss

config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

logger = logging.getLogger(__name__)

OUTPUT_WF_CSV = config.WALK_FORWARD_RESULTS_CSV
OUTPUT_WF_FIG = config.WALK_FORWARD_STABILITY_FIG


def set_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_poisson_deviance_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(
            y_true > 0.0,
            y_true * np.log((y_true + 1e-9) / y_pred) - (y_true - y_pred),
            y_pred,
        )
    dev = 2.0 * term
    return float(np.nanmean(dev))


def _fit_nb_mle_glm(y: np.ndarray, X_sm: np.ndarray) -> tuple:
    """Profile-likelihood MLE for NB alpha. Returns (result, llf, alpha_hat)."""
    grid = np.logspace(-3, 2, 60)
    best_res, best_llf, best_alpha = None, -np.inf, 1.0
    for a in grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.GLM(
                    y, X_sm, family=sm.families.NegativeBinomial(alpha=float(a))
                ).fit(disp=False, maxiter=500)
            llf = float(res.llf)
            if llf > best_llf:
                best_llf = llf
                best_res = res
                best_alpha = float(a)
        except Exception:
            continue
    return best_res, best_llf, best_alpha


def _dl_train_predict(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    c_train: np.ndarray,
    X_test_scaled: np.ndarray,
    c_test: np.ndarray,
    num_cells: int,
    n_features: int,
    device: torch.device,
    loss_kind: str = "nb",
    seed: int = 42,
) -> np.ndarray:
    """Train a DL model and return test predictions (mu only)."""
    set_seeds(seed)
    # Chronological validation cut — last 15 % of training rows (sorted order is preserved)
    n_val = max(1, int(np.floor(0.15 * len(y_train))))
    tr_idx = np.arange(len(y_train) - n_val)
    val_idx = np.arange(len(y_train) - n_val, len(y_train))

    X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]
    c_tr, c_val = c_train[tr_idx], c_train[val_idx]

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(c_tr, dtype=torch.long),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(c_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(c_test, dtype=torch.long),
        torch.zeros(len(c_test), dtype=torch.float32),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = HybridModel(num_cells=num_cells, embedding_dim=8, input_dim=n_features).to(device)
    criterion: torch.nn.Module = NegativeBinomialLoss() if loss_kind == "nb" else PoissonLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

    for _epoch in range(30):
        model.train()
        for xb, cb, yb in train_loader:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            mu, alpha = model(xb, cb)
            loss = criterion(mu, alpha, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, cb, yb in val_loader:
                xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
                mu, alpha = model(xb, cb)
                loss = criterion(mu, alpha, yb)
                bs = yb.shape[0]
                val_loss_sum += float(loss.item()) * bs
                val_n += bs
        val_loss = val_loss_sum / max(val_n, 1)
        if early_stopping.step(val_loss, model):
            break

    early_stopping.restore_best(model)
    preds_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, cb, _ in test_loader:
            xb, cb = xb.to(device), cb.to(device)
            mu, _ = model(xb, cb)
            preds_parts.append(mu.detach().cpu().numpy())

    preds = np.concatenate(preds_parts).astype(np.float64) if preds_parts else np.zeros(len(c_test))
    return np.clip(preds, 1e-9, None)


def run_walk_forward() -> pd.DataFrame:
    set_seeds(42)

    processed_path = config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE
    df = pd.read_csv(processed_path)
    df["week"] = pd.to_datetime(df["week"])

    FEATURES_ENHANCED = [
        "Y_lag1", "mag_max_lag1", "mag_min_lag1",
        "mag_max_roll4", "count_roll12", "energy_roll8", "weeks_since_m45",
    ]

    df = df.reindex(columns=["cell_id", "week", "Y"] + FEATURES_ENHANCED, fill_value=0.0)
    cell_codes, _ = pd.factorize(df["cell_id"].astype(str), sort=True)
    df["cell_id_idx"] = cell_codes.astype(np.int64)
    num_cells = int(df["cell_id_idx"].nunique())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load raw events for ETAS
    raw_path = config.RAW_DATA_PATH / config.RAW_DATA_FILE
    raw_df: pd.DataFrame | None = None
    try:
        raw_df = pd.read_csv(raw_path)
        raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True, format="mixed").dt.tz_localize(None)
        # Add cell_id to raw_df by spatial binning
        min_lat = config.BBOX["minlatitude"]
        min_lon = config.BBOX["minlongitude"]
        gs = config.GRID_SIZE
        raw_df["lat_grid"] = np.floor((raw_df["latitude"] - min_lat) / gs) * gs + min_lat
        raw_df["lon_grid"] = np.floor((raw_df["longitude"] - min_lon) / gs) * gs + min_lon
        raw_df["lat_grid"] = raw_df["lat_grid"].round(2)
        raw_df["lon_grid"] = raw_df["lon_grid"].round(2)
        raw_df["cell_id"] = (
            raw_df["lat_grid"].map(lambda x: f"{float(x):.2f}")
            + "_"
            + raw_df["lon_grid"].map(lambda x: f"{float(x):.2f}")
        )
    except Exception as exc:
        logger.warning("Could not load raw events for ETAS: %s", exc)
        raw_df = None

    results_list: list[dict] = []

    for test_year in range(2018, 2024):
        train_mask = df["week"].dt.year < test_year
        test_mask = df["week"].dt.year == test_year

        df_train = df.loc[train_mask].dropna(subset=["Y"]).reset_index(drop=True)
        df_test = df.loc[test_mask].dropna(subset=["Y"]).reset_index(drop=True)

        if df_train.empty or df_test.empty:
            logger.info("Year %d skipped: empty train/test split", test_year)
            continue

        X_train = df_train[FEATURES_ENHANCED].fillna(0.0).astype(np.float32)
        X_test = df_test[FEATURES_ENHANCED].fillna(0.0).astype(np.float32)
        y_train = df_train["Y"].astype(np.float32).to_numpy()
        y_test = df_test["Y"].astype(np.float32).to_numpy()
        c_train_arr = df_train["cell_id_idx"].astype(np.int64).to_numpy()
        c_test_arr = df_test["cell_id_idx"].astype(np.int64).to_numpy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        # --- 1. NB GLM with MLE alpha per fold ---
        mae_nb, rmse_nb, dev_nb, alpha_nb = np.nan, np.nan, np.nan, np.nan
        try:
            X_train_sm = sm.add_constant(X_train_scaled, has_constant="add")
            X_test_sm = sm.add_constant(X_test_scaled, has_constant="add")
            nb_result, _, alpha_nb = _fit_nb_mle_glm(y_train, X_train_sm)
            if nb_result is None:
                raise RuntimeError("NB MLE grid failed")
            preds_nb = np.clip(np.asarray(nb_result.predict(X_test_sm), dtype=np.float64), 1e-9, None)
            mae_nb = float(mean_absolute_error(y_test, preds_nb))
            rmse_nb = float(np.sqrt(mean_squared_error(y_test, preds_nb)))
            dev_nb = mean_poisson_deviance_safe(y_test, preds_nb)
        except Exception as exc:
            logger.warning("NB GLM failed for %d: %s", test_year, exc)

        results_list.append({
            "Year": test_year, "Model": "NB_Enhanced_MLE",
            "MAE": mae_nb, "RMSE": rmse_nb, "Mean_Poisson_Deviance": dev_nb,
            "alpha_hat": alpha_nb,
        })

        # --- 2. Hybrid DL Enhanced (NB loss) ---
        mae_dl, rmse_dl, dev_dl = np.nan, np.nan, np.nan
        try:
            preds_dl = _dl_train_predict(
                X_train_scaled, y_train, c_train_arr,
                X_test_scaled, c_test_arr, num_cells,
                len(FEATURES_ENHANCED), device, loss_kind="nb",
            )
            mae_dl = float(mean_absolute_error(y_test, preds_dl))
            rmse_dl = float(np.sqrt(mean_squared_error(y_test, preds_dl)))
            dev_dl = mean_poisson_deviance_safe(y_test, preds_dl)
        except Exception as exc:
            logger.warning("Hybrid DL failed for %d: %s", test_year, exc)

        results_list.append({
            "Year": test_year, "Model": "Hybrid_DL_Enhanced",
            "MAE": mae_dl, "RMSE": rmse_dl, "Mean_Poisson_Deviance": dev_dl,
            "alpha_hat": np.nan,
        })

        # --- 3. Neural Poisson Enhanced ---
        mae_np, rmse_np, dev_np = np.nan, np.nan, np.nan
        try:
            preds_np = _dl_train_predict(
                X_train_scaled, y_train, c_train_arr,
                X_test_scaled, c_test_arr, num_cells,
                len(FEATURES_ENHANCED), device, loss_kind="poisson",
            )
            mae_np = float(mean_absolute_error(y_test, preds_np))
            rmse_np = float(np.sqrt(mean_squared_error(y_test, preds_np)))
            dev_np = mean_poisson_deviance_safe(y_test, preds_np)
        except Exception as exc:
            logger.warning("Neural Poisson failed for %d: %s", test_year, exc)

        results_list.append({
            "Year": test_year, "Model": "Neural_Poisson_Enhanced",
            "MAE": mae_np, "RMSE": rmse_np, "Mean_Poisson_Deviance": dev_np,
            "alpha_hat": np.nan,
        })

        # --- 4. ETAS per-cell ---
        mae_etas, rmse_etas, dev_etas = np.nan, np.nan, np.nan
        if raw_df is not None:
            try:
                from src.etas_baseline import fit_etas_per_cell, predict_etas
                train_end = pd.Timestamp(f"{test_year}-01-01")
                params = fit_etas_per_cell(raw_df, train_end)
                weeks_grid = df_test[["cell_id", "week"]].copy()
                pred_etas_df = predict_etas(params, weeks_grid)
                df_test_etas = df_test.merge(pred_etas_df, on=["cell_id", "week"], how="left")
                df_test_etas["lambda_pred"] = df_test_etas["lambda_pred"].fillna(0.1).clip(lower=1e-9)
                y_etas = df_test_etas["Y"].astype(float).to_numpy()
                preds_etas = df_test_etas["lambda_pred"].to_numpy()
                mae_etas = float(mean_absolute_error(y_etas, preds_etas))
                rmse_etas = float(np.sqrt(mean_squared_error(y_etas, preds_etas)))
                dev_etas = mean_poisson_deviance_safe(y_etas, preds_etas)
            except Exception as exc:
                logger.warning("ETAS failed for %d: %s", test_year, exc)

        results_list.append({
            "Year": test_year, "Model": "ETAS_Per_Cell",
            "MAE": mae_etas, "RMSE": rmse_etas, "Mean_Poisson_Deviance": dev_etas,
            "alpha_hat": np.nan,
        })

        logger.info(
            "Year %d | NB MPD=%.4f | DL MPD=%.4f | NP MPD=%.4f | ETAS MPD=%.4f",
            test_year, dev_nb, dev_dl, dev_np, dev_etas,
        )

    results_df = pd.DataFrame(results_list)
    OUTPUT_WF_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_WF_CSV, index=False)
    logger.info("Walk-forward results saved to %s", OUTPUT_WF_CSV)

    _plot_walk_forward(results_df)
    return results_df


def _plot_walk_forward(results_df: pd.DataFrame) -> None:
    """Plot MPD by year for all 4 WF models."""
    model_map = {
        "NB_Enhanced_MLE": "NB GLM (MLE α)",
        "Hybrid_DL_Enhanced": "Hybrid DL (NB loss)",
        "Neural_Poisson_Enhanced": "Neural Poisson",
        "ETAS_Per_Cell": "ETAS per-cell",
    }
    palette = {
        "NB GLM (MLE α)": "#b0b0b0",
        "Hybrid DL (NB loss)": "#1F4E79",
        "Neural Poisson": "#2980b9",
        "ETAS per-cell": "#e67e22",
    }
    plot_df = results_df.copy()
    plot_df["Model_Display"] = plot_df["Model"].map(model_map).fillna(plot_df["Model"])
    plot_df = plot_df.dropna(subset=["Mean_Poisson_Deviance"])

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    try:
        sns.barplot(
            data=plot_df,
            x="Year", y="Mean_Poisson_Deviance",
            hue="Model_Display",
            palette=palette,
            errorbar=None, ax=ax,
        )
    except TypeError:
        sns.barplot(
            data=plot_df,
            x="Year", y="Mean_Poisson_Deviance",
            hue="Model_Display",
            palette=palette,
            ci=None, ax=ax,
        )
    ax.set_title("Walk-Forward Stability: Mean Poisson Deviance by Year")
    ax.set_xlabel("Test year")
    ax.set_ylabel("Mean Poisson Deviance")
    ax.legend(title="", loc="upper right", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    nb_df = results_df[results_df["Model"] == "NB_Enhanced_MLE"]
    dl_df = results_df[results_df["Model"] == "Hybrid_DL_Enhanced"]
    avg_nb = float(nb_df["Mean_Poisson_Deviance"].mean()) if not nb_df.empty else np.nan
    avg_dl = float(dl_df["Mean_Poisson_Deviance"].mean()) if not dl_df.empty else np.nan
    if np.isfinite(avg_nb) and avg_nb > 0 and np.isfinite(avg_dl):
        avg_improvement = 100.0 * (avg_nb - avg_dl) / avg_nb
        ax.text(
            0.03, 0.95,
            f"DL vs NB GLM: {avg_improvement:+.1f}% Avg.",
            transform=ax.transAxes, va="top",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#9a9a9a"},
        )
    fig.tight_layout()
    OUTPUT_WF_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_WF_FIG, dpi=300)
    plt.close(fig)
    logger.info("Saved walk-forward figure to %s", OUTPUT_WF_FIG)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    run_walk_forward()
