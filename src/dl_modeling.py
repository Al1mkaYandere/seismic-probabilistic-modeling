"""Deep learning ablation: Hybrid NB model on baseline vs enhanced feature sets."""

from __future__ import annotations

import copy
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src import config

logger = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES_BASELINE = ["mag_max_lag1", "mag_min_lag1", "Y_lag1"]
FEATURES_ENHANCED = [
    "mag_max_lag1",
    "mag_min_lag1",
    "Y_lag1",
    "mag_max_roll4",
    "count_roll12",
    "energy_roll8",
    "weeks_since_m45",
]
CAT_COL = "cell_id"
TARGET_COL = "Y"

BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
LR = 1e-3

OUTPUT_COMPARISON: Path = config.MODEL_COMPARISON_CSV
OUTPUT_CALIBRATION_PREDS: Path = config.CALIBRATION_PREDICTIONS_CSV
OUTPUT_ALPHA_SUMMARY: Path = config.ALPHA_AUDIT_SUMMARY_CSV
OUTPUT_ALPHA_HIST: Path = config.ALPHA_DISTRIBUTION_FIG


class EarthquakeDataset(Dataset):
    """Dataset for numeric features + cell_id index + target."""

    def __init__(self, X_num: np.ndarray, cell_idx: np.ndarray, y: np.ndarray) -> None:
        self.X_num = torch.as_tensor(X_num, dtype=torch.float32)
        self.cell_idx = torch.as_tensor(cell_idx, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_num[idx], self.cell_idx[idx], self.y[idx]


class EarthquakeNet(nn.Module):
    """Spatial-embedding + MLP with NB output heads.

    Architecture: Embedding(cell) -> concat(num_features) -> Linear(64) + ReLU + Dropout(0.2)
    -> Linear(32) + ReLU + Dropout(0.2) -> Linear(2) -> Softplus -> (mu, alpha).
    Dropout(p=0.2) is applied after each hidden layer during training only.
    """

    def __init__(self, num_cells: int, input_dim: int, emb_dim: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_cells, embedding_dim=emb_dim)
        self.fc1 = nn.Linear(emb_dim + input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(32, 2)

    def forward(self, X_num: torch.Tensor, cell_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(cell_idx)
        x = torch.cat([emb, X_num], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        raw = self.out(x)
        mu = F.softplus(raw[:, 0]) + 1e-8
        alpha = F.softplus(raw[:, 1]) + 1e-8
        return mu, alpha


class NBLoss(nn.Module):
    """Negative Binomial NLL."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, mu: torch.Tensor, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        mu = torch.clamp(mu, min=eps)
        alpha = torch.clamp(alpha, min=eps)
        y = torch.clamp(y, min=0.0)

        inv_alpha = 1.0 / alpha
        denom = 1.0 + alpha * mu
        log_prob = (
            torch.lgamma(y + inv_alpha)
            - torch.lgamma(y + 1.0)
            - torch.lgamma(inv_alpha)
            + inv_alpha * torch.log(1.0 / (denom + eps))
            + y * torch.log((alpha * mu) / (denom + eps) + eps)
        )
        return -torch.mean(log_prob)


class PoissonLoss(nn.Module):
    """Poisson negative log-likelihood without constant term log(y!)."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, mu: torch.Tensor, _alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        mu = torch.clamp(mu, min=eps)
        y = torch.clamp(y, min=0.0)
        return torch.mean(mu - y * torch.log(mu + eps))


class HybridModel(EarthquakeNet):
    """Compatibility alias used by validation workflow."""

    def __init__(self, num_cells: int, embedding_dim: int, input_dim: int) -> None:
        super().__init__(num_cells=num_cells, input_dim=input_dim, emb_dim=embedding_dim)


class NegativeBinomialLoss(NBLoss):
    """Compatibility alias used by validation workflow."""


class EarlyStopping:
    """Simple early stopping with in-memory best weights."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict[str, torch.Tensor] | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = float(val_loss)
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_df() -> pd.DataFrame:
    path = config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE
    df = pd.read_csv(path)
    df["week"] = pd.to_datetime(df["week"])
    return df


def _encode_cells(train_cells: pd.Series, other_cells: pd.Series) -> tuple[np.ndarray, int]:
    classes = pd.Index(train_cells.astype(str).unique())
    mapping = {c: i for i, c in enumerate(classes)}
    unk = len(classes)
    encoded = other_cells.astype(str).map(mapping).fillna(unk).astype(np.int64).to_numpy()
    return encoded, int(unk + 1)


def _save_alpha_audit(alpha: np.ndarray) -> None:
    valid = np.asarray(alpha, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return
    near_zero_thr = 1e-2
    summary = pd.DataFrame(
        [
            {
                "n": int(valid.size),
                "mean_alpha": float(np.mean(valid)),
                "median_alpha": float(np.median(valid)),
                "q10_alpha": float(np.quantile(valid, 0.10)),
                "q90_alpha": float(np.quantile(valid, 0.90)),
                "share_alpha_lt_1e-2": float(np.mean(valid < near_zero_thr)),
            }
        ]
    )
    OUTPUT_ALPHA_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_ALPHA_SUMMARY, index=False)

    OUTPUT_ALPHA_HIST.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid, bins=40, color="#1F4E79", alpha=0.85, edgecolor="white")
    ax.axvline(near_zero_thr, color="#c0392b", linestyle="--", linewidth=1.5)
    ax.set_title("Predicted alpha distribution (Neural NB)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(OUTPUT_ALPHA_HIST, dpi=250)
    plt.close(fig)


def _train_one_variant(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    loss_kind: str = "nb",
) -> tuple[dict[str, float | str], pd.DataFrame | None]:
    required = ["week", CAT_COL, TARGET_COL] + features
    work = df.dropna(subset=["week", CAT_COL, TARGET_COL]).copy()
    work = work.sort_values("week", kind="mergesort").reset_index(drop=True)

    unique_weeks = np.sort(work["week"].dropna().unique())
    if len(unique_weeks) < 2:
        raise ValueError("Insufficient unique weeks for train/test split")
    split_w = int(np.floor(0.8 * len(unique_weeks)))
    if split_w < 1 or split_w >= len(unique_weeks):
        raise ValueError("Invalid week split index for 80/20 chronological split")
    train_weeks = unique_weeks[:split_w]
    test_weeks = unique_weeks[split_w:]
    train_full = work[work["week"].isin(train_weeks)].reset_index(drop=True).copy()
    test_df = work[work["week"].isin(test_weeks)].reset_index(drop=True).copy()
    if train_full.empty or test_df.empty:
        raise ValueError("Chronological week split produced empty train/test")
    n_val = max(1, int(np.floor(0.1 * len(train_full))))
    train_df = train_full.iloc[:-n_val].reset_index(drop=True).copy()
    val_df = train_full.iloc[-n_val:].reset_index(drop=True).copy()

    X_train = train_df[features].fillna(0.0).astype(float)
    X_val = val_df[features].fillna(0.0).astype(float)
    X_test = test_df[features].fillna(0.0).astype(float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    y_train = train_df[TARGET_COL].astype(float).to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COL].astype(float).to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COL].astype(float).to_numpy(dtype=np.float64)

    c_train, n_cells = _encode_cells(train_df[CAT_COL], train_df[CAT_COL])
    c_val, _ = _encode_cells(train_df[CAT_COL], val_df[CAT_COL])
    c_test, _ = _encode_cells(train_df[CAT_COL], test_df[CAT_COL])

    train_loader = DataLoader(EarthquakeDataset(X_train_s, c_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EarthquakeDataset(X_val_s, c_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(EarthquakeDataset(X_test_s, c_test, y_test.astype(np.float32)), batch_size=BATCH_SIZE, shuffle=False)

    model = EarthquakeNet(num_cells=n_cells, input_dim=len(features), emb_dim=8).to(DEVICE)
    criterion: nn.Module = NBLoss() if loss_kind == "nb" else PoissonLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_sum = 0.0
        tr_n = 0
        for xb, cb, yb in train_loader:
            xb, cb, yb = xb.to(DEVICE), cb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad(set_to_none=True)
            mu, alpha = model(xb, cb)
            loss = criterion(mu, alpha, yb)
            loss.backward()
            optim.step()
            bs = yb.shape[0]
            tr_sum += float(loss.item()) * bs
            tr_n += bs
        tr_loss = tr_sum / max(tr_n, 1)

        model.eval()
        va_sum = 0.0
        va_n = 0
        with torch.no_grad():
            for xb, cb, yb in val_loader:
                xb, cb, yb = xb.to(DEVICE), cb.to(DEVICE), yb.to(DEVICE)
                mu, alpha = model(xb, cb)
                loss = criterion(mu, alpha, yb)
                bs = yb.shape[0]
                va_sum += float(loss.item()) * bs
                va_n += bs
        va_loss = va_sum / max(va_n, 1)
        sched.step(va_loss)

        logger.debug("%s | Epoch %03d | Train Loss: %.6f | Val Loss: %.6f", model_name, epoch, tr_loss, va_loss)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                logger.debug("%s | Early stopping at epoch %d", model_name, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    preds_parts: list[np.ndarray] = []
    alpha_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, cb, _ in test_loader:
            xb, cb = xb.to(DEVICE), cb.to(DEVICE)
            mu, alpha = model(xb, cb)
            preds_parts.append(mu.detach().cpu().numpy())
            alpha_parts.append(alpha.detach().cpu().numpy())
    preds = np.concatenate(preds_parts).astype(np.float64) if preds_parts else np.zeros_like(y_test)
    preds = np.clip(preds, 1e-9, None)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mpd = float(mean_poisson_deviance(y_test, preds))

    logger.debug("%s | MAE: %.6f | RMSE: %.6f | Mean Poisson Deviance: %.6f", model_name, mae, rmse, mpd)

    calib_df: pd.DataFrame | None = None
    if model_name in {"Hybrid_DL_Enhanced", "Neural_Poisson_Enhanced"}:
        alpha_pred = (
            np.concatenate(alpha_parts).astype(np.float64)
            if alpha_parts
            else np.zeros_like(y_test, dtype=np.float64)
        )
        calib_df = pd.DataFrame(
            {
                "model": model_name,
                "y_true": y_test.astype(np.float64),
                "mu_pred": preds.astype(np.float64),
                "alpha_pred": alpha_pred,
            }
        )
        if model_name == "Hybrid_DL_Enhanced":
            _save_alpha_audit(alpha_pred)

    result = {
        "model": model_name,
        "Model_Type": "Deep Learning",
        "MAE": mae,
        "RMSE": rmse,
        "Mean_Poisson_Deviance": mpd,
        "alpha_hat": np.nan,
        "Status": "SUCCESS",
    }
    return result, calib_df


def run_alpha_identifiability(seeds: list[int] | None = None) -> pd.DataFrame:
    """Train Hybrid_DL_Enhanced with multiple seeds and report per-seed alpha statistics.

    Returns
    -------
    pd.DataFrame
        One row per seed with columns: seed, n, mean_alpha, median_alpha, q10_alpha,
        q90_alpha, share_alpha_lt_1e-2.
    """
    if seeds is None:
        seeds = [42, 7, 123, 2024, 999]

    df = _load_df()
    summary_rows: list[dict] = []
    per_cell_alphas: dict[int, dict] = {}

    for seed in seeds:
        _set_seed(seed)
        try:
            _, calib_df = _train_one_variant(df, FEATURES_ENHANCED, "Hybrid_DL_Enhanced", loss_kind="nb")
        except Exception as exc:
            logger.warning("Alpha identifiability: seed %d failed: %s", seed, exc)
            continue
        if calib_df is None:
            continue
        alpha_arr = calib_df["alpha_pred"].to_numpy(dtype=np.float64)
        alpha_arr = alpha_arr[np.isfinite(alpha_arr)]
        if alpha_arr.size == 0:
            continue
        summary_rows.append({
            "seed": seed,
            "n": int(alpha_arr.size),
            "mean_alpha": float(np.mean(alpha_arr)),
            "median_alpha": float(np.median(alpha_arr)),
            "q10_alpha": float(np.quantile(alpha_arr, 0.10)),
            "q90_alpha": float(np.quantile(alpha_arr, 0.90)),
            "share_alpha_lt_1e-2": float(np.mean(alpha_arr < 1e-2)),
        })
        per_cell_alphas[seed] = alpha_arr

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.ALPHA_IDENTIFIABILITY_CSV, index=False)

    if per_cell_alphas and len(per_cell_alphas) > 1:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        data = [v for v in per_cell_alphas.values()]
        labels = [f"seed={s}" for s in per_cell_alphas.keys()]
        ax.boxplot(data, labels=labels, patch_artist=True,
                   boxprops={"facecolor": "#1F4E79", "alpha": 0.7},
                   medianprops={"color": "white", "linewidth": 2})
        ax.set_title("Alpha distribution by seed (Hybrid DL NB Enhanced)")
        ax.set_xlabel("Random seed")
        ax.set_ylabel("Predicted alpha")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(config.ALPHA_IDENTIFIABILITY_FIG, dpi=250)
        plt.close(fig)

    logger.info("Saved alpha identifiability results, %d seeds", len(summary_rows))
    return summary_df


def run_dl_modeling() -> pd.DataFrame:
    """
    Run DL ablation for two feature sets and return metrics DataFrame.

    Returns
    -------
    pd.DataFrame
        Four rows: NB (baseline/enhanced) and Neural Poisson (baseline/enhanced).
    """
    _set_seed(SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_df()
    rows: list[dict[str, float | str]] = []
    calib_parts: list[pd.DataFrame] = []
    for features, model_name, loss_kind in (
        (FEATURES_BASELINE, "Hybrid_DL_Baseline", "nb"),
        (FEATURES_ENHANCED, "Hybrid_DL_Enhanced", "nb"),
        (FEATURES_BASELINE, "Neural_Poisson_Baseline", "poisson"),
        (FEATURES_ENHANCED, "Neural_Poisson_Enhanced", "poisson"),
    ):
        try:
            row, calib_df = _train_one_variant(df, features, model_name, loss_kind=loss_kind)
            rows.append(row)
            if calib_df is not None:
                calib_parts.append(calib_df)
        except Exception as exc:
            logger.warning("DL model %s failed: %s", model_name, exc)
            rows.append(
                {
                    "model": model_name,
                    "Model_Type": "Deep Learning",
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "Mean_Poisson_Deviance": np.nan,
                    "alpha_hat": np.nan,
                    "Status": "FAILED",
                }
            )
    out = pd.DataFrame(
        rows,
        columns=["model", "Model_Type", "MAE", "RMSE", "Mean_Poisson_Deviance", "alpha_hat", "Status"],
    )

    if calib_parts:
        calib_out = pd.concat(calib_parts, ignore_index=True)
        OUTPUT_CALIBRATION_PREDS.parent.mkdir(parents=True, exist_ok=True)
        calib_out.to_csv(OUTPUT_CALIBRATION_PREDS, index=False)

    # Optional persistence in shared comparison table.
    if OUTPUT_COMPARISON.exists():
        prev = pd.read_csv(OUTPUT_COMPARISON)
        if "model" in prev.columns:
            prev = prev.loc[
                ~prev["model"].isin(
                    [
                        "Hybrid_DL_Baseline",
                        "Hybrid_DL_Enhanced",
                        "Neural_Poisson_Baseline",
                        "Neural_Poisson_Enhanced",
                    ]
                )
            ].copy()
        out_all = pd.concat([prev, out], ignore_index=True)
    else:
        out_all = out.copy()
    out_all.to_csv(OUTPUT_COMPARISON, index=False)

    return out
