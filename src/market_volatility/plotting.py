"""Small plotting helper shared by notebooks."""

from pathlib import Path

import matplotlib.figure

from market_volatility.data import FIGURES_DIR

def save_figure(
    figure: matplotlib.figure.Figure,
    figure_id: str,
    figures_dir: Path = FIGURES_DIR,
    dpi: int = 300,
) -> Path:
    """Save a specific Matplotlib figure and return its path."""

    figures_dir.mkdir(parents = True, exist_ok = True)
    path = figures_dir / f"{figure_id}.png"
    figure.savefig(path, dpi = dpi, bbox_inches = "tight")
    return path
