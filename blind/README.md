# Blind test on 2024 – September 2026

Section 4.4 of the paper forecasts a period that was downloaded only after the
rest of the analysis had been frozen: the code fixed at one commit, the
architecture, features and training settings written down, together with what
we expected to see. Nothing in this directory was tuned on that period.

```bash
python blind/run_blind.py            # writes into blind/outputs/
```

The run takes well under a minute and needs no network: both catalogues are
committed (`data/raw/usgs_central_asia_raw.csv` for 2010–2023 and
`data/raw/usgs_2024_2026_raw.csv` for the new period). It writes only inside
its output directory and refuses to start if that directory is, or is inside,
`outputs/`, `outputs-mc45/` or `data/` — see `blind/output_dir.py`. Another
location: `SPM_BLIND_DIR=/somewhere python blind/run_blind.py`.

## What it produces

| File | Contents |
| --- | --- |
| `measures_all.csv` | one row per model: level, Poisson deviance with and without the level correction, logarithmic score, CRPS, and the exact tail test at `Y >= 2` and `Y >= 3`. This is Table 2 of the paper. |
| `blind_predictions.csv` | every model's forecast for every one of the 2397 cell-weeks |
| `blind_panel.csv` | the blind period's panel, features included |
| `etas_sim_blind_predictions.csv` | the within-week branching simulation, row by row |
| `dl_*_blind_per_seed.csv` | each network's forecast on each of the five seeds |

## How it differs from the main pipeline

Same models, same modules, one different split. `main.py` divides its panel
into training and test parts itself; here the division is fixed by the calendar
— everything before 2024-01-01 trains, the 141 weeks after it are forecast, and
no model is refitted on the new period.

Three functions of the pipeline make their own split internally
(`dl_modeling._train_one_variant`, `etas_simulation.simulate_etas_predictions`
and `validation.run_walk_forward`), so they cannot be called directly here. The
driver calls the pieces underneath them instead — the same network, the same
loss, the same optimiser and early stopping, the same per-week simulator — and
supplies this split. Everything else is imported unchanged.

The networks are the one place where the five seeds are averaged as forecasts
rather than as scores: there is a single blind period, so there is no sequence
of weekly scores to average first. Section 4.1 of the paper says so as well.
