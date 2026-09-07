"""Backward-compatible imports for the first data-acquisition notebook.

New code should import from ``market_volatility`` modules directly.  This file
keeps the original notebook imports working while there is one canonical
implementation under ``src/market_volatility``.
"""

from market_volatility.data import (
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    fetch_yahoo_data,
    load_yahoo_data,
)
from market_volatility.plotting import save_figure


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    """Compatibility wrapper around :func:`save_figure` for old notebooks."""

    if fig_extension != "png":
        raise ValueError("The compatibility wrapper currently supports PNG only.")

    import matplotlib.pyplot as plt

    figure = plt.gcf()
    if tight_layout:
        figure.tight_layout()
    return save_figure(figure, fig_id, dpi=resolution)


__all__ = [
    "PROJECT_ROOT",
    "RAW_DIR",
    "PROCESSED_DIR",
    "fetch_yahoo_data",
    "load_yahoo_data",
    "save_fig",
]
