# EarthquakeNet

Probabilistic forecasting of **weekly earthquake counts** on a spatial grid over Central Asia (USGS catalog, 2 January 2010 – 27 December 2023). The pipeline downloads and cleans seismic data (or, with `SPM_OFFLINE=1`, uses the committed catalog), builds a spatiotemporal panel, fits classical GLM baselines (Poisson / Negative Binomial), deep learning models (Hybrid DL NB, Neural Poisson), and an ETAS per-cell baseline, then runs walk-forward validation, calibration diagnostics, tail metrics, and spatial residual tests.

```mermaid
flowchart LR
  USGS[USGS API] --> ingest[data_ingestion]
  ingest --> grid[grid_builder]
  grid --> models[GLM_DL_ETAS]
  models --> eval[validation_calibration]
  eval --> artifacts[outputs CSV and figures]
```



## Quick start (3 steps)

Run all commands from the **repository root** (the folder that contains `main.py`).


| Step                        | Command                           | Notes                                                              |
| --------------------------- | --------------------------------- | ------------------------------------------------------------------ |
| **1. Install dependencies** | `pip install -r requirements.txt` | Python 3.10+ recommended; use a virtual environment                |
| **2. Setup**                | *(not required)*                  | `main.py` creates `outputs/` and required subfolders automatically |
| **3. Run the pipeline**     | `SPM_OFFLINE=1 python main.py`    | Full end-to-end run from the committed catalog; without `SPM_OFFLINE=1` it downloads the catalog again |


Example:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
python main.py
```

After a successful run, results appear under `outputs/` (this folder is **not** tracked in git; only `outputs/.gitkeep` is committed).

## What the pipeline does

`main.py` executes the following stages in order:

1. **Data ingestion** — download and clean the USGS event catalog (`src/data_ingestion.py`)
2. **Spatiotemporal grid** — build a `cell_id × week` panel with lags and rolling features (`src/grid_builder.py`)
3. **Magnitude of completeness** — estimate M_c and b-value (`src/mc_estimation.py`)
4. **Poisson diagnostics** — overdispersion checks and exploratory figures (`src/poisson_analysis.py`)
5. **Classical GLM** — Poisson / NB models with likelihood-ratio tests (`src/modeling.py`)
6. **Deep learning** — Hybrid DL NB and Neural Poisson (`src/dl_modeling.py`)
7. **ETAS baseline** — per-cell temporal ETAS (`src/etas_baseline.py`)
8. **Walk-forward validation** — 2018–2023 out-of-sample evaluation (`src/validation.py`)
9. **Calibration** — randomized PIT and summary tables (`src/calibration.py`)
10. **Alpha identifiability** — multi-seed audit for dispersion parameter \alpha
11. **Tail metrics** — stratum-wise MAE / RMSE / MPD / NLL / CRPS (`src/tail_metrics.py`)
12. **Spatial diagnostics** — Moran's I on residuals (`src/spatial_diagnostics.py`)

## Requirements


| Package      | Version (pinned) | Role                           |
| ------------ | ---------------- | ------------------------------ |
| Python       | 3.10+            | Runtime                        |
| pandas       | 2.2.3            | Data frames                    |
| numpy        | 2.2.5            | Numerics                       |
| requests     | 2.32.3           | USGS API                       |
| scikit-learn | 1.5.2            | Scaling, metrics               |
| statsmodels  | 0.14.6           | GLM baselines                  |
| matplotlib   | 3.9.2            | Figures                        |
| seaborn      | 0.13.2           | Styling                        |
| torch        | 2.8.0            | Deep learning (~2 GB download) |
| folium       | 0.17.0           | Optional maps                  |
| jupyter      | 1.1.1            | Notebooks                      |


If `torch` is missing, the DL step is skipped gracefully and placeholder rows are written to the model comparison table.

## Project layout

```
STATBYA/
├── main.py                 # Single entry point
├── requirements.txt
├── src/                    # Pipeline modules
│   ├── config.py           # Paths, API params, output filenames
│   ├── data_ingestion.py
│   ├── grid_builder.py
│   ├── modeling.py
│   ├── dl_modeling.py
│   └── ...
├── data/
│   ├── raw/                # USGS catalog snapshot (committed)
│   └── processed/          # Spatiotemporal panel (committed)
├── notebooks/
│   └── 01_hybrid_nb_deep_dive.ipynb
└── outputs/                # Generated at runtime (gitignored)
    └── .gitkeep
```

Private research notes and the LaTeX manuscript live in `sandbox/` (gitignored, local only).

## Data

The repository includes a snapshot of the USGS catalog and the processed panel:

- `data/raw/usgs_central_asia_raw.csv` — the catalog the analysis is built on, 2 January 2010 to 27 December 2023
- `data/raw/usgs_2024_2026_raw.csv` — 2024 to 13 September 2026, downloaded only after the analysis was frozen and used by the blind test
- `data/processed/spatiotemporal_grid.csv`

Without `SPM_OFFLINE=1`, `main.py` downloads the catalog from USGS for the bounding box and dates in `src/config.py` and overwrites `data/raw/usgs_central_asia_raw.csv` (network access and the `requests` package are needed only for that). The runs reported in the paper use the committed catalogs.

## Reproducing the paper

All runs are offline. `SPM_OUTPUT_DIR` decides where the tables go; point it elsewhere to keep a run next to the published one.

```bash
# threshold 3.0 (the one preprint v1 used)
SPM_OFFLINE=1 python main.py

# threshold 4.5, the catalogue's completeness magnitude; the weekly axis is pinned
# to the 3.0 catalogue so that both runs are scored on the same weeks
SPM_OFFLINE=1 SPM_M_C=4.5 SPM_PIN_AXIS=1 SPM_PANEL_FILE=spatiotemporal_grid_mc45.csv \
SPM_OUTPUT_DIR=outputs-mc45 python main.py

# one more training seed (the paper uses 42, 7, 123, 2024 and 999; 42 is the default)
SPM_OFFLINE=1 SPM_SEED=7 SPM_OUTPUT_DIR=outputs-seed7 python main.py

# the blind test on 2024 - September 2026
python blind/run_blind.py
```

| Variable | Meaning |
| --- | --- |
| `SPM_OFFLINE=1` | use the committed catalog, download nothing |
| `SPM_M_C` | completeness threshold for modelling (default 3.0) |
| `SPM_PIN_AXIS=1` | take the weekly axis from the M ≥ 3.0 catalog |
| `SPM_PANEL_FILE` | panel file name in `data/processed/` (default `spatiotemporal_grid.csv`) |
| `SPM_KEEP_PANEL=1` | read the panel on disk instead of writing the rebuilt one |
| `SPM_OUTPUT_DIR` | where the tables and figures go (default `outputs/`) |
| `SPM_SEED` | training seed of both protocols (default 42) |

## Blind test

`python blind/run_blind.py` refits every model on data through 2023, forecasts
2024 – September 2026 without retraining, and writes the comparison into
`blind/outputs/`. It runs offline from the two committed catalogs and never
writes into `outputs/`, `outputs-mc45/` or `data/`. Details: `blind/README.md`.

## Outputs (after `python main.py`)


| File                                                   | Description                                |
| ------------------------------------------------------ | ------------------------------------------ |
| `outputs/model_comparison.csv`                         | Static hold-out comparison across models   |
| `outputs/walk_forward_results.csv`                     | Year-by-year walk-forward metrics          |
| `outputs/tail_evaluation.csv`                          | Tail-stratum evaluation                    |
| `outputs/calibration_summary.csv`                      | PIT / calibration summary                  |
| `outputs/moran_residuals.csv`                          | Moran's I per model                        |
| `outputs/figures/beautiful_walk_forward_stability.png` | Walk-forward stability plot                |
| `outputs/figures/*.png`                                | Additional diagnostic and forecast figures |


## Usage examples

**Full pipeline (recommended):**

```bash
python main.py
```

**Exploratory notebook:**

```bash
jupyter notebook notebooks/01_hybrid_nb_deep_dive.ipynb
```

The notebook adds the repository root to `sys.path` automatically.

## Troubleshooting


| Issue                                        | Fix                                                                                          |
| -------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'src'` | Run `python main.py` from the repository root, not from `src/`                               |
| DL step skipped / `FAILED` rows              | Install full dependencies: `pip install -r requirements.txt` (includes `torch`)              |
| Empty or incomplete metrics                  | Check internet for USGS download, date range in `src/config.py`, and that `data/raw/` exists |
| Long first run                               | Walk-forward and DL training are compute-intensive; a GPU speeds up PyTorch if available     |


## License

MIT — see [LICENSE](LICENSE).

## Manuscript

The research paper (`paper.tex`) and internal review notes are kept locally in `sandbox/` and are **not** part of this public repository.