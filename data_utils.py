from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# data_utils.py is located in the project root, so its parent directory
# is always the project root.
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_PATH = PROJECT_ROOT / "reports" / "figures"


# Function to download Yahoo Finance data for one ticker and remove previous raw files for this ticker. 
def fetch_yahoo_data(
    symbol: str,
    raw_dir: Path = RAW_DIR,
    period: str = "max",
    interval: str = "1d",
) -> tuple[pd.DataFrame, Path]:
    """Download Yahoo Finance data and replace the previous raw CSV."""

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
        raise RuntimeError(f"No price data was returned for {symbol}.")

    saved_date = datetime.now().astimezone().strftime("%Y-%m-%d")

    raw_path = (
        raw_dir
        / f"yahoo_{filename_symbol}_daily_{saved_date}.csv"
    )

    temporary_path = raw_path.with_name(f"{raw_path.name}.tmp")

    try:
        prices.to_csv(temporary_path)
        temporary_path.replace(raw_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    old_paths = [
        path
        for path in raw_dir.glob(
            f"yahoo_{filename_symbol}*daily*.csv"
        )
        if path != raw_path
    ]

    for old_path in old_paths:
        old_path.unlink()

    print(f"Downloaded {len(prices):,} daily observations.")
    print(f"First date: {prices.index.min()}")
    print(f"Last date: {prices.index.max()}")
    print(f"Saved raw Yahoo data to: {raw_path}")

    if old_paths:
        print(f"Removed {len(old_paths)} older raw data file(s).")

    return prices, raw_path


# Function to read in downloaded raw CSV data file for one ticker. 
def load_yahoo_data(
    symbol: str,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    """Load the locally saved raw Yahoo Finance CSV."""

    filename_symbol = symbol.lower()

    saved_paths = list(
        raw_dir.glob(f"yahoo_{filename_symbol}*daily*.csv")
    )

    if not saved_paths:
        raise FileNotFoundError(
            f"No saved raw data was found for {symbol.upper()} "
            f"in {raw_dir}."
        )

    if len(saved_paths) > 1:
        raise RuntimeError(
            f"Expected one saved file for {symbol.upper()}, "
            f"but found {len(saved_paths)}."
        )

    return pd.read_csv(
        saved_paths[0],
        index_col=0,
        parse_dates=[0],
    )


# Function to save images in report figures folder. 
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"

    print("Saving figure", fig_id)
    
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)