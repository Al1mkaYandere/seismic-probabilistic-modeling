"""Classical modeling with ablation: baseline vs enhanced feature sets."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
from sklearn.preprocessing import StandardScaler
try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - robust fallback for partial statsmodels installs
    from types import SimpleNamespace
    from statsmodels.genmod import families as _families
    from statsmodels.genmod.generalized_linear_model import GLM as _GLM
    from statsmodels.tools.tools import add_constant as _add_constant

    sm = SimpleNamespace(  # type: ignore[assignment]
        GLM=_GLM,
        families=_families,
        add_constant=_add_constant,
    )

from src import config

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

Y_COL: str = "Y"

OUTPUT_COMPARISON: Path = config.MODEL_COMPARISON_CSV
OUTPUT_FORECAST_FIG: Path = config.FORECAST_COMPARISON_FIG
OUTPUT_FORECAST_FIG_BEAUTIFUL: Path = config.FORECAST_COMPARISON_BEAUTIFUL_FIG
OUTPUT_LR_TEST: Path = config.OVERDISPERSION_LR_TEST_CSV


def _ensure_output_dirs() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_processed_data() -> pd.DataFrame:
    path = config.PROCESSED_DATA_PATH / config.PROCESSED_DATA_FILE
    logger.info("Loading processed data from %s", path.resolve())
    df = pd.read_csv(path)
    df["week"] = pd.to_datetime(df["week"])
    return df


def _prepare_train_test(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = features + [Y_COL]
    clean = df.dropna(subset=cols, how="any").copy()
    clean = clean.sort_values("week", kind="mergesort").reset_index(drop=True)
    n = len(clean)
    if n == 0:
        raise ValueError("No rows left after dropping NaN in features/target")
    n_train = int(np.floor(0.8 * n))
    if n_train < 1 or n - n_train < 1:
        raise ValueError("Insufficient rows for 80/20 chronological split")
    train = clean.iloc[:n_train].copy()
    test = clean.iloc[n_train:].copy()
    return train, test


def _to_float_arrays(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = train[features].astype(float).to_numpy()
    X_test = test[features].astype(float).to_numpy()
    y_train = train[Y_COL].astype(float).to_numpy()
    y_test = test[Y_COL].astype(float).to_numpy()
    return X_train, X_test, y_train, y_test


def _non_negative_predictions(y_pred: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.asarray(y_pred, dtype=float))


def _clip_for_poisson_deviance(y_pred: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_pred, dtype=float), 1e-6, None)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    pred_nn = _non_negative_predictions(y_pred)
    mae = float(mean_absolute_error(y_true, pred_nn))
    rmse = float(np.sqrt(mean_squared_error(y_true, pred_nn)))
    pred_dev = _clip_for_poisson_deviance(pred_nn)
    mpd = float(mean_poisson_deviance(y_true, pred_dev))
    return mae, rmse, mpd


def _build_results_table(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _plot_forecast_comparison(test: pd.DataFrame) -> None:
    totals = test.groupby("cell_id", sort=False)[Y_COL].sum()
    best_cell = str(totals.idxmax())
    sub = test.loc[test["cell_id"] == best_cell].sort_values("week").copy()
    if sub.empty:
        logger.warning("No test rows for cell %s; skipping forecast plot", best_cell)
        return

    fig, ax = plt.subplots(figsize=(11, 5.8))
    weeks = sub["week"].to_numpy()
    observed = sub[Y_COL].astype(float).to_numpy()
    nb_series = sub["pred_nb"].astype(float).to_numpy() if "pred_nb" in sub.columns else None
    # Prefer DL prediction when present; fallback to Poisson if DL is unavailable.
    if "pred_dl" in sub.columns:
        dl_series = sub["pred_dl"].astype(float).to_numpy()
        dl_label = "DL Hybrid"
    elif "pred_poisson" in sub.columns:
        dl_series = sub["pred_poisson"].astype(float).to_numpy()
        dl_label = "Poisson"
        logger.warning("pred_dl missing in forecast data; using pred_poisson for third line")
    else:
        dl_series = None
        dl_label = "DL Hybrid"
        logger.warning("Neither pred_dl nor pred_poisson found; plotting only available series")

    sns.lineplot(
        x=weeks,
        y=observed,
        color="#111111",
        linewidth=2.8,
        linestyle="-",
        label="Observed",
        ax=ax,
    )
    if nb_series is not None:
        sns.lineplot(
            x=weeks,
            y=nb_series,
            color="#E68613",
            linewidth=2.2,
            linestyle="--",
            label="NB Baseline",
            ax=ax,
        )
    if dl_series is not None:
        sns.lineplot(
            x=weeks,
            y=dl_series,
            color="#1F4E79",
            linewidth=2.3,
            linestyle="-",
            label=dl_label,
            ax=ax,
        )
    anomaly_start = pd.Timestamp("2023-01-01")
    ax.axvspan(anomaly_start, sub["week"].max(), color="#bdbdbd", alpha=0.18, lw=0)
    ax.axvline(anomaly_start, color="#777777", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Weekly count $Y$")
    ax.set_title("Walk-Forward Forecast Comparison")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="major", axis="y", alpha=0.3)
    ax.legend(frameon=True, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    OUTPUT_FORECAST_FIG_BEAUTIFUL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FORECAST_FIG_BEAUTIFUL, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_classical_models(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Run ablation study for classical models on baseline vs enhanced features.

    Notes
    -----
    - Signature is intentionally flexible to absorb extra args/kwargs.
    - Chronological split is strict (80/20) with index reset.
    """
    FEATURES_BASELINE = ["Y_lag1", "mag_max_lag1", "mag_min_lag1"]
    FEATURES_ENHANCED = [
        "Y_lag1",
        "mag_max_lag1",
        "mag_min_lag1",
        "mag_max_roll4",
        "count_roll12",
        "energy_roll8",
        "weeks_since_m45",
    ]

    required = ["week", Y_COL, "cell_id"] + FEATURES_ENHANCED
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for ablation: {missing}")

    df_work = df.sort_values("week", kind="mergesort").reset_index(drop=True).copy()
    unique_weeks = np.sort(df_work["week"].dropna().unique())
    if len(unique_weeks) < 2:
        raise ValueError("Not enough unique weeks for chronological split")
    split_w = int(np.floor(0.8 * len(unique_weeks)))
    if split_w < 1 or split_w >= len(unique_weeks):
        raise ValueError("Invalid week split index for 80/20 chronological split")
    train_weeks = unique_weeks[:split_w]
    test_weeks = unique_weeks[split_w:]
    df_train = df_work[df_work["week"].isin(train_weeks)].reset_index(drop=True).copy()
    df_test = df_work[df_work["week"].isin(test_weeks)].reset_index(drop=True).copy()
    if df_train.empty or df_test.empty:
        raise ValueError("Chronological week split produced empty train/test")

    y_train = df_train[Y_COL].astype(float).to_numpy()
    y_test = df_test[Y_COL].astype(float).to_numpy()

    rows: list[dict[str, float | str]] = []
    pred_store: dict[str, np.ndarray] = {}
    ll_store: dict[str, float] = {}

    # Naive persistence
    naive_preds = np.clip(df_test["Y_lag1"].fillna(0.0).astype(float).to_numpy(), 1e-9, None)
    mae_n = float(mean_absolute_error(y_test, naive_preds))
    rmse_n = float(np.sqrt(mean_squared_error(y_test, naive_preds)))
    mpd_n = float(mean_poisson_deviance(y_test, naive_preds))
    rows.append(
        {
            "model": "Naive_Persistence",
            "Model_Type": "Classical",
            "MAE": mae_n,
            "RMSE": rmse_n,
            "Mean_Poisson_Deviance": mpd_n,
            "alpha_hat": np.nan,
            "Status": "SUCCESS",
        }
    )
    pred_store["pred_naive"] = naive_preds

    def _fit_nb_mle_glm(
        y: np.ndarray,
        X_sm: np.ndarray,
        alpha_grid: np.ndarray | None = None,
    ) -> tuple:
        """Profile-likelihood MLE for NB dispersion α.

        Returns (fitted_result, llf, alpha_hat).
        """
        grid = alpha_grid if alpha_grid is not None else np.logspace(-3, 2, 60)
        best_res, best_llf, best_alpha = None, -np.inf, grid[0]
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

    model_specs_poisson = [
        ("Poisson_Baseline", FEATURES_BASELINE),
        ("Poisson_Enhanced", FEATURES_ENHANCED),
    ]
    model_specs_nb = [
        ("NB_Baseline", FEATURES_BASELINE),
        ("NB_Enhanced", FEATURES_ENHANCED),
    ]

    alpha_hat_store: dict[str, float] = {}

    for model_name, features in model_specs_poisson:
        X_train = df_train[features].fillna(0.0).astype(float)
        X_test = df_test[features].fillna(0.0).astype(float)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_sm = sm.add_constant(X_train_scaled, has_constant="add")
        X_test_sm = sm.add_constant(X_test_scaled, has_constant="add")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sm.GLM(
                    y_train, X_train_sm, family=sm.families.Poisson()
                ).fit(disp=False, maxiter=500)
            preds = np.clip(np.asarray(result.predict(X_test_sm), dtype=float), 1e-9, None)
            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mpd = float(mean_poisson_deviance(y_test, preds))
            if model_name == "Poisson_Enhanced":
                pred_store["pred_poisson"] = preds
            ll_store[model_name] = float(result.llf)
            status = "SUCCESS"
        except Exception as exc:
            logger.warning("Model %s failed: %s", model_name, exc)
            mae = rmse = mpd = np.nan
            status = "FAILED"
        rows.append({
            "model": model_name, "Model_Type": "Classical",
            "MAE": mae, "RMSE": rmse, "Mean_Poisson_Deviance": mpd,
            "alpha_hat": np.nan, "Status": status,
        })

    for model_name, features in model_specs_nb:
        X_train = df_train[features].fillna(0.0).astype(float)
        X_test = df_test[features].fillna(0.0).astype(float)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train_sm = sm.add_constant(X_train_scaled, has_constant="add")
        X_test_sm = sm.add_constant(X_test_scaled, has_constant="add")
        try:
            result, llf, alpha_hat = _fit_nb_mle_glm(y_train, X_train_sm)
            if result is None:
                raise RuntimeError("NB MLE grid search failed for all alpha values")
            preds = np.clip(np.asarray(result.predict(X_test_sm), dtype=float), 1e-9, None)
            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mpd = float(mean_poisson_deviance(y_test, preds))
            if model_name == "NB_Enhanced":
                pred_store["pred_nb"] = preds
            ll_store[model_name] = float(llf)
            alpha_hat_store[model_name] = alpha_hat
            status = "SUCCESS"
        except Exception as exc:
            logger.warning("Model %s failed: %s", model_name, exc)
            mae = rmse = mpd = np.nan
            alpha_hat = np.nan
            status = "FAILED"
        rows.append({
            "model": model_name, "Model_Type": "Classical",
            "MAE": mae, "RMSE": rmse, "Mean_Poisson_Deviance": mpd,
            "alpha_hat": alpha_hat_store.get(model_name, np.nan), "Status": status,
        })

    if "pred_poisson" in pred_store and "pred_nb" in pred_store:
        test_out = df_test.copy()
        test_out["pred_naive"] = pred_store["pred_naive"]
        test_out["pred_poisson"] = pred_store["pred_poisson"]
        test_out["pred_nb"] = pred_store["pred_nb"]
        _plot_forecast_comparison(test_out)
        # Save test predictions for tail_metrics.py
        pred_cols = ["cell_id", "week", Y_COL, "pred_naive", "pred_poisson", "pred_nb"]
        avail_cols = [c for c in pred_cols if c in test_out.columns]
        save_path = config.TEST_PREDICTIONS_CSV
        out_preds = test_out[avail_cols].rename(columns={Y_COL: "y_true"})
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out_preds.to_csv(save_path, index=False)
        logger.info("Saved test predictions to %s", save_path.resolve())

    # Likelihood-ratio test with NB alpha selected by profile-likelihood grid search.
    try:
        X_train_lr = df_train[FEATURES_ENHANCED].fillna(0.0).astype(float)
        scaler_lr = StandardScaler()
        X_train_lr_s = scaler_lr.fit_transform(X_train_lr)
        X_train_lr_s = sm.add_constant(X_train_lr_s, has_constant="add")
        y_train_lr = df_train[Y_COL].astype(float).to_numpy()

        pois_res = sm.GLM(y_train_lr, X_train_lr_s, family=sm.families.Poisson()).fit(
            disp=False,
            maxiter=500,
        )
        ll_p = float(pois_res.llf)

        best_alpha = None
        ll_nb = -np.inf
        for alpha in np.logspace(-3, 2, 60):
            nb_res = sm.GLM(
                y_train_lr,
                X_train_lr_s,
                family=sm.families.NegativeBinomial(alpha=float(alpha)),
            ).fit(disp=False, maxiter=500)
            ll_cur = float(nb_res.llf)
            if ll_cur > ll_nb:
                ll_nb = ll_cur
                best_alpha = float(alpha)

        lr_stat = float(max(2.0 * (ll_nb - ll_p), 0.0))
        dof = 1
        p_value_chi2 = float(chi2.sf(lr_stat, df=dof))
        # Boundary correction (Self & Liang 1987): alpha=0 is on the boundary of NB
        # parameter space, so the null distribution is 0.5*delta_0 + 0.5*chi2_1.
        p_value_boundary = 0.5 * p_value_chi2
        lr_df = pd.DataFrame(
            [
                {
                    "comparison": "NB_Enhanced_vs_Poisson_Enhanced",
                    "ll_poisson": ll_p,
                    "ll_nb": ll_nb,
                    "lr_stat": lr_stat,
                    "dof": dof,
                    "p_value_chi2": p_value_chi2,
                    "p_value_boundary": p_value_boundary,
                    "null_dist": "0.5*chi2_0 + 0.5*chi2_1",
                    "sample": "in-sample (train fold)",
                    "alpha_hat_grid": best_alpha,
                    "method": "GLM profile-LR (Poisson vs NB alpha grid MLE)",
                }
            ]
        )
        OUTPUT_LR_TEST.parent.mkdir(parents=True, exist_ok=True)
        lr_df.to_csv(OUTPUT_LR_TEST, index=False)
        logger.info("Saved LR overdispersion test to %s", OUTPUT_LR_TEST.resolve())
    except Exception as exc:
        logger.warning("LR overdispersion test failed: %s", exc)

    results_df = _build_results_table(rows)
    return results_df


def run_modeling() -> pd.DataFrame:
    """Run classical ablation study, persist comparison table, and return metrics."""
    _ensure_output_dirs()
    df = _load_processed_data()
    results = run_classical_models(df)
    OUTPUT_COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_COMPARISON, index=False)
    return results
