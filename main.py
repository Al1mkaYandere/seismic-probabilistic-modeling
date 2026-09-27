"""Entry point: ingestion, spatiotemporal grid, all modelling steps, diagnostics."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_ingestion import run_ingestion_pipeline
from src.grid_builder import (
    build_spatiotemporal_grid,
    filter_to_completeness,
    load_raw_data,
    save_processed_data,
)
from src.modeling import run_modeling
from src.poisson_analysis import run_poisson_analysis
from src.utils import ensure_project_directories, setup_logging
from src.calibration import run_calibration
from src.mc_estimation import run_mc_estimation

logger = logging.getLogger(__name__)


def _switch(name: str) -> bool:
    """True when the environment variable ``name`` is set to "1".

    The switches below let the runs reported in the paper be repeated from this
    file alone. Unset, the pipeline behaves as it always has: it downloads the
    catalogue, derives the weekly axis from the events it keeps, writes the
    panel and trains with the default seed.
    """
    return os.environ.get(name) == "1"


def _ingest_unless_offline() -> None:
    """Download the USGS catalogue, unless SPM_OFFLINE=1 says to use the committed one."""
    if _switch("SPM_OFFLINE"):
        logger.info("SPM_OFFLINE=1: using the committed catalogue, nothing is downloaded")
        return
    run_ingestion_pipeline()


def _week_bounds() -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The weekly axis of the M >= 3.0 catalogue when SPM_PIN_AXIS=1, else None.

    Above 3.0 the last week of the period can hold no event; the axis would then
    lose it and the 80/20 split would move by a week. Pinning keeps runs at
    different thresholds on the same weeks.
    """
    if not _switch("SPM_PIN_AXIS"):
        return None
    raw_30 = filter_to_completeness(load_raw_data(), m_c=3.0)
    times = pd.to_datetime(raw_30["time"], utc=True, format="mixed").dt.tz_localize(None)
    weeks = times.dt.to_period("W").dt.start_time
    return weeks.min(), weeks.max()


def _training_seed() -> int | None:
    """Seed for both training protocols from SPM_SEED; None keeps their default (42)."""
    value = os.environ.get("SPM_SEED", "")
    return int(value) if value.strip() else None


def main() -> int:
    """Run the full pipeline end-to-end.

    Execution order
    ---------------
    1. Data ingestion + grid building (mc_estimation runs on raw)
    2. Poisson analysis
    3. Classical GLM modeling (NB with MLE alpha + boundary-corrected LR)
    4. DL modeling (NB + Neural Poisson)
    5. ETAS static evaluation
    6. Walk-forward validation (NB GLM MLE, Hybrid DL, Neural Poisson, ETAS)
    7. Probabilistic calibration (PIT)
    8. Alpha identifiability (5 seeds)
    9. Tail-conditional evaluation
    10. Spatial diagnostics (Moran's I)
    11. Probabilistic evaluation of every model on all test observations
    """
    setup_logging()
    logging.getLogger().setLevel(logging.WARNING)
    ensure_project_directories()
    all_results: list[pd.DataFrame] = []

    try:
        # ── Step 1: Ingestion + Grid ──────────────────────────────────────────
        _ingest_unless_offline()
        # One completeness cut for the whole run: the panel and the ETAS fit must
        # see the same events, or the baseline is scored against a target built
        # from a different catalogue than the one it was fitted on.
        raw = filter_to_completeness(load_raw_data())
        processed = build_spatiotemporal_grid(raw, week_bounds=_week_bounds())
        if _switch("SPM_KEEP_PANEL"):
            logger.info("SPM_KEEP_PANEL=1: the panel on disk is left as it is")
        else:
            save_processed_data(processed)
        seed = _training_seed()

        # ── Step 1b: Magnitude of completeness ───────────────────────────────
        try:
            run_mc_estimation()
        except Exception as exc:
            logger.warning("M_c estimation failed: %s", exc)

        # ── Step 2: Poisson analysis ──────────────────────────────────────────
        run_poisson_analysis(processed)

        # ── Step 3: Classical GLM (NB MLE alpha, boundary-corrected LR) ─────
        classical_results = run_modeling()
        all_results.append(classical_results)

        # ── Step 4: DL modeling (NB + Neural Poisson) ────────────────────────
        try:
            from src.dl_modeling import run_dl_modeling
            dl_results = run_dl_modeling() if seed is None else run_dl_modeling(seed=seed)
            all_results.append(dl_results)
        except ModuleNotFoundError as exc:
            logger.warning("DL step skipped (missing dependency): %s", exc)
            all_results.append(pd.DataFrame([
                {"model": m, "Model_Type": "Deep Learning",
                 "MAE": pd.NA, "RMSE": pd.NA, "Mean_Poisson_Deviance": pd.NA,
                 "alpha_hat": pd.NA, "Status": "FAILED"}
                for m in ["Hybrid_DL_Baseline", "Hybrid_DL_Enhanced",
                          "Neural_Poisson_Baseline", "Neural_Poisson_Enhanced"]
            ]))

        # ── Step 5: ETAS static evaluation ───────────────────────────────────
        try:
            from src.etas_baseline import run_etas_static
            from src import config as _cfg
            raw_for_etas = filter_to_completeness(
                pd.read_csv(_cfg.RAW_DATA_PATH / _cfg.RAW_DATA_FILE)
            )
            raw_for_etas["time"] = pd.to_datetime(raw_for_etas["time"], utc=True, format="mixed").dt.tz_localize(None)
            raw_for_etas["lat_grid"] = (
                np.floor((raw_for_etas["latitude"] - _cfg.BBOX["minlatitude"]) / _cfg.GRID_SIZE)
                * _cfg.GRID_SIZE + _cfg.BBOX["minlatitude"]
            )
            raw_for_etas["lon_grid"] = (
                np.floor((raw_for_etas["longitude"] - _cfg.BBOX["minlongitude"]) / _cfg.GRID_SIZE)
                * _cfg.GRID_SIZE + _cfg.BBOX["minlongitude"]
            )
            raw_for_etas["lat_grid"] = raw_for_etas["lat_grid"].round(2)
            raw_for_etas["lon_grid"] = raw_for_etas["lon_grid"].round(2)
            raw_for_etas["cell_id"] = (
                raw_for_etas["lat_grid"].map(lambda x: f"{float(x):.2f}")
                + "_"
                + raw_for_etas["lon_grid"].map(lambda x: f"{float(x):.2f}")
            )
            etas_metrics = run_etas_static(raw_for_etas, processed)
            etas_row = pd.DataFrame([{
                "model": "ETAS_Per_Cell",
                "Model_Type": "ETAS",
                **etas_metrics,
            }])
            all_results.append(etas_row)
            # Persist into comparison CSV
            from src.modeling import OUTPUT_COMPARISON
            if OUTPUT_COMPARISON.exists():
                prev = pd.read_csv(OUTPUT_COMPARISON)
                prev = prev[prev["model"] != "ETAS_Per_Cell"].copy()
                out_all = pd.concat([prev, etas_row], ignore_index=True)
            else:
                out_all = etas_row
            out_all.to_csv(OUTPUT_COMPARISON, index=False)
        except Exception as exc:
            logger.warning("ETAS static evaluation failed: %s", exc)

        # ── Step 6: Walk-forward validation ──────────────────────────────────
        try:
            from src.validation import run_walk_forward
            run_walk_forward() if seed is None else run_walk_forward(seed=seed)
        except Exception as exc:
            logger.warning("Walk-forward validation failed: %s", exc)

        # ── Step 6b: Level bias of every forecast, and what survives it ──────
        try:
            from src.level_calibration import run_level_calibration
            run_level_calibration()
        except Exception as exc:
            logger.warning("Level calibration failed: %s", exc)

        # ── Step 7: Probabilistic calibration (PIT) ───────────────────────────
        run_calibration()

        # ── Step 8: Alpha identifiability (5 seeds) ───────────────────────────
        try:
            from src.dl_modeling import run_alpha_identifiability
            run_alpha_identifiability()
        except Exception as exc:
            logger.warning("Alpha identifiability run failed: %s", exc)

        # ── Step 9: Tail-conditional evaluation ──────────────────────────────
        try:
            from src.tail_metrics import run_tail_metrics
            run_tail_metrics()
        except Exception as exc:
            logger.warning("Tail metrics failed: %s", exc)

        # ── Step 10: Spatial diagnostics (Moran's I) ─────────────────────────
        try:
            from src.spatial_diagnostics import run_spatial_diagnostics
            run_spatial_diagnostics()
        except Exception as exc:
            logger.warning("Spatial diagnostics failed: %s", exc)

        # ── Step 11: Probabilistic evaluation on the whole test split ────────
        try:
            from src.probabilistic_evaluation import run_probabilistic_evaluation
            run_probabilistic_evaluation()
        except Exception as exc:
            logger.warning("Probabilistic evaluation failed: %s", exc)

        # ── Step 12: ETAS within-week branching simulation ────────────────────
        # Last on purpose: it only reads raw_for_etas/processed (both already
        # built for step 5) and writes one new file, so it cannot shift
        # anything computed by an earlier step.
        try:
            from src.etas_simulation import run_etas_simulation
            run_etas_simulation(raw_for_etas, processed)
        except Exception as exc:
            logger.warning("ETAS branching simulation failed: %s", exc)

        # ── Step 13: Tail hypothesis test and comparison power ───────────────
        # Last on purpose, after the branching simulation: it only reads files
        # every earlier step already wrote (the whole-sample probabilistic
        # evaluation and the simulation's own predictions) and adds two new
        # small tables, so it cannot move a number computed anywhere else.
        try:
            from src.tail_test import run_tail_test
            run_tail_test()
        except Exception as exc:
            logger.warning("Tail hypothesis test failed: %s", exc)

        n_cells = processed["cell_id"].nunique()
        t_min = processed["week"].min()
        t_max = processed["week"].max()
        logger.info("Pipeline complete: %d cells, %s – %s", n_cells, t_min, t_max)

    except Exception:
        logger.exception("Pipeline failed")
        return 1

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        cols = [c for c in ["model", "Model_Type", "MAE", "RMSE", "Mean_Poisson_Deviance",
                             "alpha_hat", "Status"] if c in final_df.columns]
        print("\n" + "=" * 80)
        print("FINAL MODEL COMPARISON")
        print("=" * 80)
        print(final_df[cols].to_string(index=False, justify="left"))
        print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
