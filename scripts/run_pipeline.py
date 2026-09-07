"""Run preparation, model selection, final evaluation, and reporting."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from market_volatility.data import (
    MODELS_DIR,
    REPORTS_DIR,
    chronological_split,
    final_training_set,
    load_yahoo_data,
    save_modeling_data,
)

from market_volatility.evaluate import (
    errors_by_regime,
    forecast_metrics,
    prediction_frame,
    training_regime_thresholds,
    worst_forecasts,
)

from market_volatility.features import TARGET_COLUMN, make_modeling_dataset
from market_volatility.plotting import save_figure
from market_volatility.train import baseline_predictions, fit_models, predict_models


SYMBOL = "SPY"
VALIDATION_START = "2016-01-01"
TEST_START = "2021-01-01"
PURGE_HORIZON = 20


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    """Create a small Markdown table without an optional tabulate dependency."""

    printable = frame.copy()
    for column in printable.select_dtypes(include="number"):
        printable[column] = printable[column].map(lambda value: f"{value:.{decimals}f}")
    printable.insert(0, "name", printable.index.astype(str))
    header = "| " + " | ".join(printable.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(printable.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in printable.to_numpy()]
    return "\n".join([header, rule, *rows])


def write_report(
    dataset: pd.DataFrame,
    splits,
    validation_metrics: pd.DataFrame,
    selected_model: str,
    test_metrics: pd.DataFrame,
    regime_errors: pd.DataFrame,
    output_path: Path,
) -> None:
    content = f"""# Final Results

## Experiment design

- Instrument: {SYMBOL}, daily observations
- Modeling rows: {len(dataset):,} ({dataset.index.min().date()} to {dataset.index.max().date()})
- Target: annualized root-mean-square log return over trading days t+1 through t+20
- Train: {len(splits.train):,} rows, {splits.train.index.min().date()} to {splits.train.index.max().date()}
- Validation: {len(splits.validation):,} rows, {splits.validation.index.min().date()} to {splits.validation.index.max().date()}
- Test: {len(splits.test):,} rows, {splits.test.index.min().date()} to {splits.test.index.max().date()}
- Purge: 20 observations before validation and test boundaries
- Primary selection metric: validation MAE

## Validation comparison

{markdown_table(validation_metrics)}

The ML model selected using validation MAE was **{selected_model}**. The test
period was not used to make this choice.

## Final test performance

{markdown_table(test_metrics)}

## Selected-model errors by realized-volatility regime

Regime thresholds were estimated from the final pre-test training labels, not
from test labels.

{markdown_table(regime_errors)}

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
"""
    output_path.write_text(content)


def save_summary_figures(
    validation_metrics: pd.DataFrame,
    test_actual: pd.Series,
    test_predictions: pd.DataFrame,
    selected_model: str,
) -> None:
    """Create the three headline figures without requiring notebook execution."""

    fig, ax = plt.subplots(figsize=(11, 5))
    validation_metrics["MAE"].sort_values().plot.bar(ax=ax, color="steelblue")
    ax.set(title="Validation MAE (lower is better)", ylabel="MAE", xlabel="")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "03_validation_mae")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_actual.index, test_actual, label="Actual", color="black")
    ax.plot(
        test_predictions.index,
        test_predictions[selected_model],
        label=selected_model,
    )
    ax.set(
        title="Final test: actual versus forecast volatility",
        ylabel="Annualized volatility",
        xlabel="Date",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, "04_test_actual_vs_forecast")
    plt.close(fig)

    signed_error = test_predictions[selected_model] - test_actual
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].plot(signed_error.index, signed_error, linewidth=0.7, color="darkred")
    axes[0].set(title="Signed error: forecast minus actual", ylabel="Error")
    axes[1].plot(
        signed_error.index,
        signed_error.abs().rolling(60).mean(),
        linewidth=1.0,
        color="purple",
    )
    axes[1].set(title="60-session rolling MAE", ylabel="MAE", xlabel="Date")
    for axis in axes:
        axis.grid(alpha=0.25)
    save_figure(fig, "04_test_error_over_time")
    plt.close(fig)


def main() -> None:
    prices = load_yahoo_data(SYMBOL)
    dataset = make_modeling_dataset(prices)
    splits = chronological_split(
        dataset,
        validation_start=VALIDATION_START,
        test_start=TEST_START,
        purge_horizon=PURGE_HORIZON,
    )
    save_modeling_data(dataset, splits)

    fitted_candidates = fit_models(splits.train)
    validation_predictions = baseline_predictions(
        splits.validation,
        training_target_mean=float(splits.train[TARGET_COLUMN].mean()),
    ).join(predict_models(fitted_candidates, splits.validation))
    validation_metrics = forecast_metrics(
        splits.validation[TARGET_COLUMN], validation_predictions
    )
    validation_prediction_table = prediction_frame(
        splits.validation[TARGET_COLUMN], validation_predictions
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    validation_metrics.to_csv(REPORTS_DIR / "validation_metrics.csv")
    validation_prediction_table.to_csv(
        REPORTS_DIR / "validation_predictions.csv", index_label="Date"
    )

    candidate_names = list(fitted_candidates)
    selected_model = validation_metrics.loc[candidate_names, "MAE"].idxmin()

    final_train = final_training_set(
        dataset, test_start=TEST_START, purge_horizon=PURGE_HORIZON
    )
    final_model = fit_models(final_train, names=[selected_model])[selected_model]
    test_predictions = baseline_predictions(
        splits.test,
        training_target_mean=float(final_train[TARGET_COLUMN].mean()),
    ).join(predict_models({selected_model: final_model}, splits.test))
    test_metrics = forecast_metrics(splits.test[TARGET_COLUMN], test_predictions)
    test_prediction_table = prediction_frame(
        splits.test[TARGET_COLUMN], test_predictions
    )

    thresholds = training_regime_thresholds(final_train[TARGET_COLUMN])
    regime_errors = errors_by_regime(
        splits.test[TARGET_COLUMN], test_predictions[selected_model], thresholds
    )
    worst = worst_forecasts(
        splits.test[TARGET_COLUMN], test_predictions[selected_model]
    )

    test_metrics.to_csv(REPORTS_DIR / "test_metrics.csv")
    test_prediction_table.to_csv(
        REPORTS_DIR / "test_predictions.csv", index_label="Date"
    )
    regime_errors.to_csv(REPORTS_DIR / "test_errors_by_regime.csv")
    worst.to_csv(REPORTS_DIR / "worst_test_forecasts.csv", index_label="Date")

    save_summary_figures(
        validation_metrics,
        splits.test[TARGET_COLUMN],
        test_predictions,
        selected_model,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "selected_model.joblib")
    (MODELS_DIR / "selected_model.txt").write_text(f"{selected_model}\n")

    write_report(
        dataset,
        splits,
        validation_metrics,
        selected_model,
        test_metrics,
        regime_errors,
        REPORTS_DIR / "final_results.md",
    )

    print("Validation results:")
    print(validation_metrics.round(4))
    print(f"\nSelected ML model: {selected_model}")
    print("\nFinal test results:")
    print(test_metrics.round(4))


if __name__ == "__main__":
    main()
