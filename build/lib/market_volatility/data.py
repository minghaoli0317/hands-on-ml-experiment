"""Data acquisition, validation, persistence, and chronological splitting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

REQUIRED_PRICE_COLUMNS = {
    "Adj Close",
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
}


@dataclass(frozen=True)
class DatasetSplits:
    """Container for purged chronological train, validation, and test sets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def validate_price_data(prices: pd.DataFrame) -> None:
    """Raise a clear exception when raw daily price data are unsuitable."""

    missing_columns = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if prices.empty:
        raise ValueError("The price data are empty.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("The price data index must be a pandas DatetimeIndex.")
    if prices.index.has_duplicates:
        raise ValueError("The price data contain duplicate dates.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("The price data must be sorted from oldest to newest.")
    if prices[list(REQUIRED_PRICE_COLUMNS)].isna().any().any():
        raise ValueError("Required price columns contain missing values.")
    if (prices[["Adj Close", "Close", "High", "Low", "Open", "Volume"]] <= 0).any().any():
        raise ValueError("Prices and volume must be strictly positive.")
    if (prices["High"] < prices[["Open", "Close", "Low"]].max(axis=1)).any():
        raise ValueError("At least one daily High is below another OHLC value.")
    if (prices["Low"] > prices[["Open", "Close", "High"]].min(axis=1)).any():
        raise ValueError("At least one daily Low is above another OHLC value.")


def fetch_yahoo_data(
    symbol: str,
    raw_dir: Path = RAW_DIR,
    period: str = "max",
    interval: str = "1d",
) -> tuple[pd.DataFrame, Path]:
    """Download one ticker and atomically replace its previous raw CSV."""

    symbol = symbol.upper()
    filename_symbol = symbol.lower()
    raw_dir.mkdir(parents=True, exist_ok=True)

    prices = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=True,
        progress=False,
        multi_level_index=False,
    )
    if prices is None or prices.empty:
        raise RuntimeError(f"No price data were returned for {symbol}.")

    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices = prices.sort_index()
    validate_price_data(prices)

    saved_date = datetime.now().astimezone().strftime("%Y-%m-%d")
    raw_path = raw_dir / f"yahoo_{filename_symbol}_daily_{saved_date}.csv"
    temporary_path = raw_path.with_suffix(".csv.tmp")

    try:
        prices.to_csv(temporary_path, index_label="Date")
        temporary_path.replace(raw_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    old_paths = [
        path
        for path in raw_dir.glob(f"yahoo_{filename_symbol}*daily*.csv")
        if path != raw_path
    ]
    for old_path in old_paths:
        old_path.unlink()

    return prices, raw_path


def load_yahoo_data(symbol: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the single locally saved raw Yahoo Finance CSV for a ticker."""

    filename_symbol = symbol.lower()
    saved_paths = sorted(raw_dir.glob(f"yahoo_{filename_symbol}*daily*.csv"))

    if not saved_paths:
        raise FileNotFoundError(
            f"No saved raw data were found for {symbol.upper()} in {raw_dir}."
        )
    if len(saved_paths) > 1:
        raise RuntimeError(
            f"Expected one saved file for {symbol.upper()}, found {len(saved_paths)}."
        )

    prices = pd.read_csv(saved_paths[0], index_col=0, parse_dates=[0])
    prices.index.name = "Date"
    prices = prices.sort_index()
    validate_price_data(prices)
    return prices


def chronological_split(
    dataset: pd.DataFrame,
    validation_start: str = "2016-01-01",
    test_start: str = "2021-01-01",
    purge_horizon: int = 20,
) -> DatasetSplits:
    """Create chronological splits and purge labels crossing each boundary.

    A target observed at date t uses returns through t + purge_horizon.  Removing
    the last ``purge_horizon`` rows of the earlier partition prevents a training
    or validation label from using returns belonging to the following period.
    """

    if purge_horizon < 1:
        raise ValueError("purge_horizon must be at least 1.")

    ordered = dataset.sort_index().copy()
    if not isinstance(ordered.index, pd.DatetimeIndex):
        raise TypeError("The modeling dataset index must be a DatetimeIndex.")
    if ordered.index.has_duplicates:
        raise ValueError("The modeling dataset contains duplicate dates.")

    validation_date = pd.Timestamp(validation_start)
    test_date = pd.Timestamp(test_start)
    if validation_date >= test_date:
        raise ValueError("validation_start must be earlier than test_start.")

    train_candidates = ordered.loc[ordered.index < validation_date]
    validation_candidates = ordered.loc[
        (ordered.index >= validation_date) & (ordered.index < test_date)
    ]
    test = ordered.loc[ordered.index >= test_date].copy()

    if len(train_candidates) <= purge_horizon:
        raise ValueError("Not enough training rows to apply the purge.")
    if len(validation_candidates) <= purge_horizon:
        raise ValueError("Not enough validation rows to apply the purge.")
    if test.empty:
        raise ValueError("The requested test period is empty.")

    train = train_candidates.iloc[:-purge_horizon].copy()
    validation = validation_candidates.iloc[:-purge_horizon].copy()
    return DatasetSplits(train=train, validation=validation, test=test)


def final_training_set(
    dataset: pd.DataFrame,
    test_start: str = "2021-01-01",
    purge_horizon: int = 20,
) -> pd.DataFrame:
    """Return all pre-test observations, purged before the test boundary."""

    candidates = dataset.sort_index().loc[
        lambda frame: frame.index < pd.Timestamp(test_start)
    ]
    if len(candidates) <= purge_horizon:
        raise ValueError("Not enough pre-test rows to apply the purge.")
    return candidates.iloc[:-purge_horizon].copy()


def save_modeling_data(
    dataset: pd.DataFrame,
    splits: DatasetSplits,
    processed_dir: Path = PROCESSED_DIR,
) -> dict[str, Path]:
    """Save the complete modeling table and each chronological partition."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "modeling_dataset": dataset,
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        path = processed_dir / f"{name}.csv"
        frame.to_csv(path, index_label="Date")
        paths[name] = path
    return paths


def load_modeling_dataset(
    path: Path = PROCESSED_DIR / "modeling_dataset.csv",
) -> pd.DataFrame:
    """Load the prepared modeling table from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")
    return pd.read_csv(path, index_col="Date", parse_dates=["Date"])
