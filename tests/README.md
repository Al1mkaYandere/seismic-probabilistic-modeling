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

## Expected failures (xfail)

Five tests are marked `xfail` on purpose: each one documents a known,
already-identified defect in the pipeline rather than an accidental test
failure.

| Test | Known defect it documents |
| --- | --- |
| `test_distributions.py::test_poisson_nll_missing_log_factorial` | `_poisson_nll` omits the `log(y!)` normalising constant, so it does not equal the true Poisson NLL and is not directly comparable to the NB NLL. |
| `test_etas_behaviour.py::test_etas_forecast_rises_after_strong_event_inside_forecast_period` | `predict_etas` only sees events up to the training cut-off, so it does not react to a strong event that occurs during the forecast period. |
| `test_etas_behaviour.py::test_fit_etas_per_cell_is_reproducible_across_calls` | ETAS fitting jitters its optimizer's starting point with an unseeded random generator, so two fits on identical data can converge to different parameters. |
| `test_walkforward_split.py::test_validation_slice_has_all_cells_and_is_latest_by_time` | The walk-forward validation slice is taken by row position from a panel sorted by cell, not by time, so it can miss most spatial cells and pull in old data instead of the most recent months. |
| `test_moran.py::test_run_spatial_diagnostics_gives_finite_number_for_dl_models` | Residual diagnostics for the deep-learning models currently return `NaN` because the calibration output they read is missing the columns needed to compute per-cell residuals. |

All five are marked `strict=True`: once the underlying defect is fixed, the
test will unexpectedly pass (`XPASS`), which `strict=True` turns into a hard
failure — a deliberate signal that the `xfail` marker must be removed at
that point.

## Known limitation

The no-leakage tests (`test_no_leakage.py`) compare the already-built panel
`data/processed/spatiotemporal_grid.csv` against an independent
recomputation; they do not call the grid-building code
(`build_spatiotemporal_grid`) itself. As a result, they cannot catch a
leakage defect introduced into the grid builder until the panel is rebuilt
and re-committed — they only verify that the panel currently on disk is
internally consistent with the raw catalog.
