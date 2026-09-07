"""Educational package for forecasting daily market volatility."""

from market_volatility.features import (
    FEATURE_COLUMNS,
    FORWARD_HORIZON,
    TARGET_COLUMN,
    make_modeling_dataset,
)

__all__ = [
    "FEATURE_COLUMNS",
    "FORWARD_HORIZON",
    "TARGET_COLUMN",
    "make_modeling_dataset",
]
