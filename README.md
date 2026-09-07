# SPY 20-Day Volatility Forecast

An educational, end-to-end machine-learning project modeled on the workflow in
Chapter 2 of *Hands-On Machine Learning*. It forecasts SPY's realized volatility
over the next 20 trading days using only information available at the forecast
date.

## Question and unit of prediction

At the close of trading day `t`, predict the annualized volatility realized by
SPY over trading days `t+1` through `t+20`.

For adjusted-close log return `r[t] = log(P[t] / P[t-1])`, the label is:

```text
forward_rv_20[t] = sqrt(252 / 20 * sum(r[t+i]^2 for i = 1,...,20))
```

The output is a decimal annualized volatility: `0.20` means approximately 20%
annualized volatility. This is a forecasting exercise, not a trading strategy.

## Reproduce the project

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[notebook,dev]"
python -m ipykernel install --user --name market-volatility
```

Select the `market-volatility` kernel in VS Code or Jupyter. Download or refresh
the raw CSV by running notebook 01. Then either run notebooks 02 through 04 in
order, or execute the non-interactive pipeline:

```bash
python scripts/run_pipeline.py
```

Run quality checks with:

```bash
pytest
ruff check .
```

## Experiment design

- Instrument: SPY
- Frequency: daily, forecast made after each close
- Target: next-20-session annualized realized volatility
- Train: observations before 2016, less a 20-session purge
- Validation: 2016–2020, less a 20-session purge
- Test: 2021 onward
- Primary metric: mean absolute error (MAE)
- Additional metrics: RMSE, R², QLIKE, and mean forecast bias
- Baselines: training mean, current 20-day volatility (persistence), current
  60-day historical volatility
- Models: HAR-style linear regression, ridge regression, random forest, and
  histogram gradient boosting

The test period is opened only after choosing the best ML candidate on
validation MAE. Twenty samples are removed before each boundary because every
label looks 20 sessions into the future. This prevents labels in an earlier
partition from consuming returns in the next partition.

## Features and their timing

Every predictor dated `t` is computable after the close on `t`:

- Returns: 1-, 5-, 20-, and 60-day log returns and absolute 1-day return
- Volatility: trailing 5-, 20-, and 60-day realized volatility; downside
  volatility; Parkinson high-low volatility
- Intraday structure: high-low range, overnight return, intraday return
- Volume: one-day log change, ratio to 20-day mean, 60-day log-volume z-score
- Trend: adjusted price relative to 20- and 60-day moving averages

Raw price levels are intentionally omitted from model inputs because levels are
nonstationary and change scale over the sample.

## Project map

```text
data/raw/                      immutable downloaded Yahoo response
data/processed/                generated modeling tables and splits
models/                        generated fitted model (ignored by Git)
notebooks/01_get_and_explore_data.ipynb
notebooks/02_prepare_data.ipynb
notebooks/03_train_models.ipynb
notebooks/04_evaluate_model.ipynb
reports/                       metrics, predictions, figures, conclusions
scripts/run_pipeline.py        reproducible headless run
src/market_volatility/data.py  validation, loading, splitting, persistence
src/market_volatility/features.py target and feature formulas
src/market_volatility/train.py baselines and estimators
src/market_volatility/evaluate.py metrics and failure diagnostics
tests/                         timing, split, and metric checks
```

`data_utils.py` remains as a compatibility layer for earlier notebook imports;
the canonical implementations now live in the package under `src/`.

## Important limitations

- Adjacent labels overlap heavily, so daily forecast errors are not independent.
- One ETF and one historical path do not establish broad generalization.
- Yahoo Finance is convenient educational data, not an audited market feed.
- Volatility jumps are intrinsically difficult to predict from lagged daily
  prices and volume. Persistence is therefore a serious baseline.
- Results are not evidence of profitability and do not include trading costs.

This project is educational and is not investment advice.
