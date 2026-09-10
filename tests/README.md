# Tests

This test suite covers three areas of the modeling pipeline:

1. **Distribution and scoring-rule properties** (`test_distributions.py`) —
   the Negative Binomial loss converges to the Poisson loss plus the
   `log(y!)` term as the dispersion parameter goes to zero, and the discrete
   CRPS implementation matches a direct summation computed from its
   definition.
2. **No look-ahead leakage in features** (`test_no_leakage.py`) — the
   committed feature panel is cross-checked against an independent
   recomputation from the raw catalog, including a targeted check that a
   feature value at week `t` is unaffected by tampering with the target at
   week `t` itself, while the same feature at week `t+1` does respond to it.
3. **ETAS behaviour, the walk-forward split, and Moran's I** — response of
   the ETAS baseline to a strong event and its run-to-run reproducibility
   (`test_etas_behaviour.py`); whether the walk-forward validation slice
   for the 2023 test year actually contains the most recent data for every
   spatial cell (`test_walkforward_split.py`); and whether Moran's I test
   for spatial autocorrelation of residuals correctly detects/rejects a
   known signal, plus its behaviour on the deep-learning models
   (`test_moran.py`).

There is also `test_frozen_numbers.py`, which cross-checks pipeline output
against a frozen baseline snapshot using a private comparison tool that is
kept outside this repository (see "Known limitation" below).

## Running the tests

```
python -m pytest tests/ -q
```

## Defects these tests were written against

Five tests were originally added as strict `xfail`, each documenting a known
defect in the pipeline. All five defects have since been fixed; no `xfail`
marker remains, and every test below now runs as a normal passing test that
guards against the defect coming back.

| Test | Defect it guards against | Fixed in |
| --- | --- | --- |
| `test_distributions.py::test_poisson_nll_missing_log_factorial` | `_poisson_nll` omitted the `log(y!)` normalising constant, so it did not equal the true Poisson NLL and was not comparable to the NB NLL. | `4f9aafc` |
| `test_etas_behaviour.py::test_fit_etas_per_cell_is_reproducible_across_calls` | ETAS fitting jittered its optimizer's starting point with an unseeded random generator, so two fits on identical data converged to different parameters. | `ae1a9d9` |
| `test_moran.py::test_run_spatial_diagnostics_gives_finite_number_for_dl_models` | Residual diagnostics for the deep-learning models returned `NaN`, because the calibration output they read was missing the columns needed to compute per-cell residuals. | `e3dbbb7` |
| `test_walkforward_split.py::test_validation_block_is_latest_weeks_and_keeps_every_cell` | The walk-forward validation slice was taken by row position from a panel sorted by cell, not by time, so it missed most spatial cells and pulled in old data instead of the most recent months. | `a540fd8` |
| `test_etas_behaviour.py::test_etas_forecast_rises_after_strong_event_inside_forecast_period` | `predict_etas` only saw events up to the training cut-off, so it did not react to a strong event occurring during the forecast period. | `3c66613` |

Each fix was checked by mutation: the defect was reintroduced into the
production code and the corresponding test was confirmed to fail again.

## Known limitation

The no-leakage tests (`test_no_leakage.py`) compare the already-built panel
`data/processed/spatiotemporal_grid.csv` against an independent
recomputation; they do not call the grid-building code
(`build_spatiotemporal_grid`) itself. As a result, they cannot catch a
leakage defect introduced into the grid builder until the panel is rebuilt
and re-committed — they only verify that the panel currently on disk is
internally consistent with the raw catalog.
