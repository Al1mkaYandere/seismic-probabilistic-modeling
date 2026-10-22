# Evaluating Neural Probabilistic Forecasts of Sparse Earthquake Counts

[![arXiv](https://img.shields.io/badge/arXiv-2605.21437-b31b1b.svg)](https://arxiv.org/abs/2605.21437)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Code and data for the paper *Evaluating Neural Probabilistic Forecasts of Sparse Earthquake Counts:
What Survives a Careful Evaluation* (Igilik Alim, [arXiv:2605.21437](https://arxiv.org/abs/2605.21437)).

The pipeline forecasts **weekly earthquake counts** (M ≥ 4.5) on a 3° × 3° grid over the Tian Shan
and Pamir. It compares a neural negative binomial model with Poisson and negative binomial GLMs and a
per-cell temporal ETAS baseline, and tests each forecast on the weeks that matter most: the rare busy
ones.

## Main findings

- **Poisson forecasts understate busy weeks.** In the 144-week test period, six cell-weeks have three
  or more events. Temporal ETAS expects 0.72 (exact test, p = 0.00009).
- **One dispersion parameter fixes it.** ETAS with dispersion expects 4.45 (p = 0.58).
- **The network's edge over a linear model goes away** once that model has an intercept per cell.
  The remaining 3.4 % of deviance is not significant (p = 0.57) and comes from a single cell-week.
- **A blind test on 2024 – September 2026** confirms the tail result. Every model there is fitted on
  data through 2023.

The paper also measures the evaluation traps met on the way: a biased reference level, a single
seed, too few bootstrap resamples, and a magnitude threshold below completeness.

## Quick start

Run everything from the repository root (the folder with `main.py`).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
SPM_OFFLINE=1 python main.py
```

Results go to `outputs/`, which is not tracked in git.

> **Keep `SPM_OFFLINE=1`.** Without it, `main.py` downloads the USGS catalog again and overwrites
> the committed `data/raw/usgs_central_asia_raw.csv`.

## Reproducing the paper

All runs are offline. `SPM_OUTPUT_DIR` decides where the tables go. Point it elsewhere to keep a run
next to the published one.

```bash
# threshold 4.5, the catalog's completeness magnitude (the paper's main run);
# the weekly axis is pinned to the 3.0 catalog so that both runs are scored on the same weeks
SPM_OFFLINE=1 SPM_M_C=4.5 SPM_PIN_AXIS=1 SPM_PANEL_FILE=spatiotemporal_grid_mc45.csv \
SPM_OUTPUT_DIR=outputs-mc45 python main.py

# threshold 3.0, the one preprint v1 used
SPM_OFFLINE=1 python main.py

# another training seed (the paper uses 42, 7, 123, 2024 and 999; 42 is the default)
SPM_OFFLINE=1 SPM_SEED=7 SPM_OUTPUT_DIR=outputs-seed7 python main.py

# the blind test on 2024 – September 2026
python blind/run_blind.py
```

| Variable | Meaning |
| --- | --- |
| `SPM_OFFLINE=1` | use the committed catalog, download nothing |
| `SPM_M_C` | completeness threshold for modelling (default 3.0) |
| `SPM_PIN_AXIS=1` | take the weekly axis from the M ≥ 3.0 catalog |
| `SPM_PANEL_FILE` | panel file name in `data/processed/` (default `spatiotemporal_grid.csv`) |
| `SPM_KEEP_PANEL=1` | leave the panel on disk as it is instead of writing the rebuilt one |
| `SPM_OUTPUT_DIR` | where the tables and figures go (default `outputs/`) |
| `SPM_SEED` | training seed of both protocols (default 42) |

## Blind test

`python blind/run_blind.py` refits every model on data through 2023 and forecasts 2024 – September
2026 without retraining. It writes the comparison into `blind/outputs/`. It runs offline from the
two committed catalogs and never writes into `outputs/`, `outputs-mc45/` or `data/`. Details are in
[`blind/README.md`](blind/README.md).

## What the pipeline does

```mermaid
flowchart LR
  catalog[USGS catalog] --> grid[cell x week panel]
  grid --> models[GLM, neural nets, ETAS]
  models --> eval[validation and tail test]
  eval --> out[tables and figures]
```

`main.py` runs these stages in order:

1. **Catalog and panel.** Load the catalog, cut it at the completeness magnitude, and build a
   `cell × week` panel with lags and rolling features (`src/data_ingestion.py`, `src/grid_builder.py`).
2. **Magnitude of completeness.** Estimate M_c and the b-value (`src/mc_estimation.py`).
3. **Poisson diagnostics.** Overdispersion checks (`src/poisson_analysis.py`).
4. **GLMs.** Poisson and negative binomial models with a boundary-corrected likelihood-ratio test
   (`src/modeling.py`).
5. **Neural nets.** Hybrid NB network and Neural Poisson (`src/dl_modeling.py`).
6. **ETAS.** Per-cell temporal ETAS (`src/etas_baseline.py`).
7. **Walk-forward validation.** Year by year, 2018–2023 (`src/validation.py`).
8. **Level bias.** How far each forecast's total is from the observed one, and what survives a
   correction (`src/level_calibration.py`).
9. **Calibration.** Randomized PIT (`src/calibration.py`).
10. **Dispersion audit.** How well α is identified across five seeds.
11. **Tail metrics** by count stratum (`src/tail_metrics.py`).
12. **Spatial diagnostics.** Moran's I on residuals (`src/spatial_diagnostics.py`).
13. **Scores on the whole test split** (`src/probabilistic_evaluation.py`).
14. **Within-week aftershock simulation** for ETAS (`src/etas_simulation.py`).
15. **Exact tail test** of the number of busy cell-weeks, and the power of the score comparisons
    (`src/tail_test.py`).

## Data

- `data/raw/usgs_central_asia_raw.csv`: the catalog the analysis is built on, 2 January 2010 to
  27 December 2023.
- `data/raw/usgs_2024_2026_raw.csv`: 2024 to 13 September 2026. Downloaded only after the analysis
  was frozen; used by the blind test.
- `data/processed/spatiotemporal_grid_mc45.csv`: the panel at M ≥ 4.5, used in the paper.
- `data/processed/spatiotemporal_grid.csv`: the panel at M ≥ 3.0, used in preprint v1.

## Main outputs

| File | What it holds |
| --- | --- |
| `tail_test.csv` | Exact test: expected vs. observed busy cell-weeks per model |
| `model_comparison.csv` | Hold-out comparison across models |
| `walk_forward_results.csv` | Year-by-year walk-forward metrics |
| `probabilistic_scores.csv`, `score_comparisons.csv` | Scores and paired comparisons |
| `level_calibration.csv` | Level bias of each forecast |
| `calibration_summary.csv` | PIT summary |
| `tail_evaluation.csv` | Metrics by count stratum |
| `moran_residuals.csv` | Moran's I per model |
| `figures/*.png` | Diagnostic and forecast figures |

## Project layout

```
seismic-probabilistic-modeling/
├── main.py              # single entry point
├── requirements.txt
├── src/                 # pipeline modules; paths and parameters in src/config.py
├── blind/               # blind test on 2024 – September 2026
├── tests/               # pytest suite
├── data/
│   ├── raw/             # USGS catalog snapshots (committed)
│   └── processed/       # cell × week panels (committed)
├── notebooks/
│   └── 01_hybrid_nb_deep_dive.ipynb
└── outputs/             # created at runtime (gitignored)
```

## Tests

```bash
python -m pytest tests/ -q
```

The suite runs offline and takes about a minute and a half on a laptop.

## Requirements

Python 3.10+ and the pinned versions in `requirements.txt`: pandas 2.2.3, numpy 2.2.5,
statsmodels 0.14.6, scikit-learn 1.5.2, torch 2.8.0 (about 2 GB), matplotlib 3.9.2, seaborn 0.13.2.
`requests` is needed only to download the catalog. `folium` and `jupyter` are needed only for the
notebook.

If `torch` is missing, the neural-net step is skipped and placeholder rows go into the comparison
table.

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'src'` | Run `python main.py` from the repository root, not from `src/` |
| Neural-net rows marked `FAILED` | Install the full requirements, including `torch` |
| The catalog was downloaded again | Restore it with `git checkout data/raw/usgs_central_asia_raw.csv` and run with `SPM_OFFLINE=1` |
| The first run is slow | Walk-forward validation and network training take most of the time; a GPU helps PyTorch |

## Citation

```bibtex
@misc{igilik2026sparse,
  title         = {Evaluating Neural Probabilistic Forecasts of Sparse Earthquake Counts:
                   What Survives a Careful Evaluation},
  author        = {Igilik, Alim},
  year          = {2026},
  eprint        = {2605.21437},
  archivePrefix = {arXiv}
}
```

## License

MIT, see [LICENSE](LICENSE).
