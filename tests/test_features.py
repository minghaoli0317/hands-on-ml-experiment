import numpy as np
import pandas as pd

from market_volatility.features import (
    FORWARD_HORIZON,
    TARGET_COLUMN,
    forward_realized_volatility,
    make_modeling_dataset,
)


def synthetic_prices(rows=160):
    index = pd.bdate_range("2010-01-01", periods=rows)
    returns = pd.Series(np.linspace(-0.01, 0.012, rows), index=index)
    adjusted_close = 100 * np.exp(returns.cumsum())
    close = adjusted_close + 10
    return pd.DataFrame(
        {
            "Adj Close": adjusted_close,
            "Close": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Open": close * 1.001,
            "Volume": np.linspace(1_000_000, 2_000_000, rows),
        },
        index=index,
    )


def test_forward_target_uses_the_next_twenty_returns():
    index = pd.bdate_range("2020-01-01", periods=50)
    returns = pd.Series(np.arange(50, dtype=float) / 10_000, index=index)
    target = forward_realized_volatility(returns)
    expected = np.sqrt(252 * np.mean(np.square(returns.iloc[1:21])))
    assert np.isclose(target.iloc[0], expected)


def test_modeling_dataset_is_complete_and_has_expected_target_tail_removed():
    prices = synthetic_prices()
    dataset = make_modeling_dataset(prices)
    assert TARGET_COLUMN in dataset
    assert not dataset.isna().any().any()
    assert dataset.index.max() == prices.index[-FORWARD_HORIZON - 1]


def test_features_at_t_do_not_change_when_future_prices_change():
    prices = synthetic_prices()
    changed = prices.copy()
    cutoff = prices.index[100]
    changed.loc[changed.index > cutoff, "Adj Close"] *= 1.5
    original_features = make_modeling_dataset(prices).drop(columns=TARGET_COLUMN)
    changed_features = make_modeling_dataset(changed).drop(columns=TARGET_COLUMN)
    pd.testing.assert_frame_equal(
        original_features.loc[:cutoff], changed_features.loc[:cutoff]
    )
