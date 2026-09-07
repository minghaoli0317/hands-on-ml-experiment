"""Leakage-safe target and feature engineering for daily SPY data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_volatility.data import validate_price_data


TRADING_DAYS_PER_YEAR = 252
FORWARD_HORIZON = 20
TARGET_COLUMN = "target_forward_rv_20d"

FEATURE_COLUMNS = [
    "log_return_1d",
    "return_5d",
    "return_20d",
    "return_60d",
    "abs_return_1d",
    "rv_5",
    "rv_20",
    "rv_60",
    "downside_rv_20",
    "parkinson_rv_20",
    "intraday_range",
    "overnight_return",
    "intraday_return",
    "volume_log_change_1d",
    "volume_ratio_20",
    "volume_zscore_60",
    "price_to_sma_20",
    "price_to_sma_60",
]


def forward_realized_volatility(
    log_returns: pd.Series,
    horizon: int = FORWARD_HORIZON,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    r"""Compute annualized realized volatility over returns t+1,...,t+h.

    The definition is

        sqrt(annualization / h * sum(r[t+i]**2 for i=1,...,h)).

    The rolling statistic is shifted backward by ``horizon`` so the value
    indexed at t contains only the next h returns, never return r[t].
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    future_mean_squared_return = (
        log_returns.pow(2)
        .rolling(window=horizon, min_periods=horizon)
        .mean()
        .shift(-horizon)
    )
    return np.sqrt(annualization * future_mean_squared_return).rename(TARGET_COLUMN)


def make_modeling_dataset(prices: pd.DataFrame) -> pd.DataFrame:
    """Create stationary, financially interpretable predictors and the target.

    Every feature at date t uses data available at or before that day's close.
    The target alone uses future returns and is used only as the supervised label.
    """

    validate_price_data(prices)
    prices = prices.sort_index().copy()

    adjusted_close = prices["Adj Close"]
    close = prices["Close"]
    log_returns = np.log(adjusted_close / adjusted_close.shift(1))
    squared_returns = log_returns.pow(2)

    features = pd.DataFrame(index=prices.index)
    features["log_return_1d"] = log_returns
    features["return_5d"] = log_returns.rolling(5).sum()
    features["return_20d"] = log_returns.rolling(20).sum()
    features["return_60d"] = log_returns.rolling(60).sum()
    features["abs_return_1d"] = log_returns.abs()

    for window in (5, 20, 60):
        features[f"rv_{window}"] = np.sqrt(
            TRADING_DAYS_PER_YEAR * squared_returns.rolling(window).mean()
        )

    downside_squared_returns = log_returns.where(log_returns < 0, 0.0).pow(2)
    features["downside_rv_20"] = np.sqrt(
        TRADING_DAYS_PER_YEAR * downside_squared_returns.rolling(20).mean()
    )

    log_high_low = np.log(prices["High"] / prices["Low"])
    features["parkinson_rv_20"] = np.sqrt(
        TRADING_DAYS_PER_YEAR
        * log_high_low.pow(2).rolling(20).mean()
        / (4.0 * np.log(2.0))
    )
    features["intraday_range"] = log_high_low
    features["overnight_return"] = np.log(prices["Open"] / close.shift(1))
    features["intraday_return"] = np.log(close / prices["Open"])

    log_volume = np.log(prices["Volume"].astype(float))
    volume_mean_20 = prices["Volume"].rolling(20).mean()
    volume_mean_60 = log_volume.rolling(60).mean()
    volume_std_60 = log_volume.rolling(60).std()
    features["volume_log_change_1d"] = log_volume.diff()
    features["volume_ratio_20"] = prices["Volume"] / volume_mean_20
    features["volume_zscore_60"] = (log_volume - volume_mean_60) / volume_std_60

    features["price_to_sma_20"] = adjusted_close / adjusted_close.rolling(20).mean() - 1
    features["price_to_sma_60"] = adjusted_close / adjusted_close.rolling(60).mean() - 1

    target = forward_realized_volatility(log_returns)
    dataset = features.join(target)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    dataset.index.name = "Date"
    return dataset
