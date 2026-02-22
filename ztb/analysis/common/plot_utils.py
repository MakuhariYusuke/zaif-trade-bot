#!/usr/bin/env python3
"""
Common plotting utilities for consistent chart generation across the codebase.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Any

def setup_plot_style():
    """Set up consistent matplotlib style for all plots."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.dpi'] = 100

def save_plot(output_path: str | Path, dpi: int = 300, bbox_inches: str = 'tight') -> None:
    """Save plot with consistent settings."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

def create_figure(figsize: Optional[Tuple[int, int]] = None) -> Any:
    """Create figure with consistent settings."""
    if figsize is None:
        figsize = (12, 6)
    return plt.figure(figsize=figsize)