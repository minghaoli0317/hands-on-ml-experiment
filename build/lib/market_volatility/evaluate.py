"""Forecast metrics and diagnostic summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def qlike_loss(y_true: pd.Series, y_pred: pd.Series | np.ndarray) -> float:
    """Return nonnegative QLIKE loss using realized and forecast variance."""

    actual_variance = np.square(np.asarray(y_true, dtype=float))
    forecast_variance = np.square(np.clip(np.asarray(y_pred, dtype=float), 1e-8, None))
    ratio = np.clip(actual_variance / forecast_variance, 1e-12, None)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def forecast_metrics(
    y_true: pd.Series,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compare aligned forecast columns with several complementary losses."""

    rows = []
    for name in predictions.columns:
        y_pred = predictions[name]
        rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
                "R2": r2_score(y_true, y_pred),
                "QLIKE": qlike_loss(y_true, y_pred),
                "bias": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("MAE")


def prediction_frame(
    y_true: pd.Series,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Join actuals, forecasts, signed errors, and absolute errors."""

    result = predictions.copy()
    result.insert(0, "actual", y_true)
    for name in predictions.columns:
        result[f"error__{name}"] = predictions[name] - y_true
        result[f"absolute_error__{name}"] = (predictions[name] - y_true).abs()
    return result


def training_regime_thresholds(y_train: pd.Series) -> tuple[float, float]:
    """Estimate low/normal/high volatility cutoffs from training labels only."""

    low, high = y_train.quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
    return float(low), float(high)


def assign_regime(
    realized_volatility: pd.Series,
    thresholds: tuple[float, float],
) -> pd.Series:
    """Label outcomes using fixed thresholds learned from training data."""

    low, high = thresholds
    labels = np.select(
        [realized_volatility <= low, realized_volatility >= high],
        ["low", "high"],
        default="normal",
    )
    return pd.Series(labels, index=realized_volatility.index, name="regime")


def errors_by_regime(
    actual: pd.Series,
    prediction: pd.Series,
    thresholds: tuple[float, float],
) -> pd.DataFrame:
    """Summarize one model's errors across realized-volatility regimes."""

    analysis = pd.DataFrame(
        {
            "actual": actual,
            "prediction": prediction,
            "regime": assign_regime(actual, thresholds),
        }
    )
    analysis["error"] = analysis["prediction"] - analysis["actual"]
    analysis["absolute_error"] = analysis["error"].abs()
    return analysis.groupby("regime", sort=False).agg(
        observations=("actual", "size"),
        mean_actual=("actual", "mean"),
        MAE=("absolute_error", "mean"),
        bias=("error", "mean"),
        underprediction_rate=("error", lambda values: float((values < 0).mean())),
    )


def worst_forecasts(
    actual: pd.Series,
    prediction: pd.Series,
    count: int = 15,
) -> pd.DataFrame:
    """Return the dates with the largest absolute forecast errors."""

    result = pd.DataFrame({"actual": actual, "prediction": prediction})
    result["error"] = result["prediction"] - result["actual"]
    result["absolute_error"] = result["error"].abs()
    return result.nlargest(count, "absolute_error")
