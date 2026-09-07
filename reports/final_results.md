# Final Results

## Experiment design

- Instrument: SPY, daily observations
- Modeling rows: 8,378 (1993-04-27 to 2026-08-07)
- Target: annualized root-mean-square log return over trading days t+1 through t+20
- Train: 5,694 rows, 1993-04-27 to 2015-12-02
- Validation: 1,239 rows, 2016-01-04 to 2020-12-02
- Test: 1,405 rows, 2021-01-04 to 2026-08-07
- Purge: 20 observations before validation and test boundaries
- Primary selection metric: validation MAE

## Validation comparison

| name | MAE | RMSE | R2 | QLIKE | bias |
| --- | --- | --- | --- | --- | --- |
| ridge | 0.0532 | 0.0990 | 0.3665 | 0.8382 | -0.0041 |
| har_linear | 0.0603 | 0.1039 | 0.3015 | 0.7400 | 0.0059 |
| hist_gradient_boosting | 0.0613 | 0.1078 | 0.2489 | 0.8316 | 0.0098 |
| random_forest | 0.0619 | 0.1063 | 0.2696 | 0.8112 | 0.0097 |
| baseline_persistence_rv20 | 0.0678 | 0.1223 | 0.0330 | 1.0082 | 0.0015 |
| baseline_historical_rv60 | 0.0754 | 0.1317 | -0.1225 | 1.2314 | 0.0103 |
| baseline_train_mean | 0.0842 | 0.1258 | -0.0232 | 1.0282 | 0.0190 |

The ML model selected using validation MAE was **ridge**. The test
period was not used to make this choice.

## Final test performance

| name | MAE | RMSE | R2 | QLIKE | bias |
| --- | --- | --- | --- | --- | --- |
| ridge | 0.0371 | 0.0569 | 0.3478 | 0.2260 | -0.0039 |
| baseline_persistence_rv20 | 0.0458 | 0.0700 | 0.0130 | 0.3474 | -0.0000 |
| baseline_historical_rv60 | 0.0467 | 0.0721 | -0.0488 | 0.3051 | 0.0045 |
| baseline_train_mean | 0.0519 | 0.0708 | -0.0106 | 0.3638 | 0.0072 |

## Selected-model errors by realized-volatility regime

Regime thresholds were estimated from the final pre-test training labels, not
from test labels.

| name | observations | mean_actual | MAE | bias | underprediction_rate |
| --- | --- | --- | --- | --- | --- |
| high | 386.0000 | 0.2396 | 0.0647 | -0.0418 | 0.7358 |
| normal | 665.0000 | 0.1341 | 0.0256 | 0.0019 | 0.5113 |
| low | 354.0000 | 0.0906 | 0.0284 | 0.0267 | 0.1215 |

## Interpretation and limitations

- Overlapping 20-day labels make adjacent errors serially correlated, so the
  number of rows overstates the number of independent forecast episodes.
- SPY alone is one market history. A strong result here need not generalize to
  another asset or future market structure.
- Yahoo Finance data are convenient for education, but they are not an audited
  institutional feed.
- Volatility is highly persistent, making the persistence baseline difficult to
  beat. A complex model is useful only if it improves genuinely out-of-sample.
- Regime-level bias shows whether the selected model smooths sudden volatility
  jumps (underprediction in high-volatility periods) or reacts too slowly after
  them (overprediction as volatility falls).

This project is educational and is not investment advice.
