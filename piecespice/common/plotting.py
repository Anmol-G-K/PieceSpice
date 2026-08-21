"""Standardized plotting utility for PieceSpice.

MANDATORY PLOTTING CONSTRAINT:
All plot legends MUST be located on the EAST side (outside plotting area to the right).
"""

from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from piecespice.common.types import CurveData, WaveformData


def setup_style():
    """Configure matplotlib defaults for publication-quality, highly legible plots."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'lines.linewidth': 1.8,
        'figure.autolayout': False,
    })


def plot_curves(
    curves: List[CurveData],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    x_log: bool = False,
    y_log: bool = False,
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a collection of characteristic curves with legend strictly at EAST (outside)."""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    for curve in curves:
        x, y = curve.to_numpy()
        label = curve.name
        if curve.family_param and curve.family_value is not None:
            label += f" ({curve.family_param}={curve.family_value:g})"
        ax.plot(x, y, label=label)

    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')

    if not x_label and curves:
        x_label = f"{curves[0].x_name} [{curves[0].x_unit}]"
    if not y_label and curves:
        y_label = f"{curves[0].y_name} [{curves[0].y_unit}]"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # STRICT RULE: Legend at EAST outside plotting area
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, edgecolor='0.8')
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig, ax


def plot_waveforms(
    waveforms: List[WaveformData],
    title: str,
    time_unit: str = "ns",
    time_scale: float = 1e9,
    figsize: Tuple[float, float] = (8, 5),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot transient time-domain waveforms with legend strictly at EAST (outside)."""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    for wf in waveforms:
        t, v = wf.to_numpy()
        ax.plot(t * time_scale, v, label=f"{wf.name} [{wf.unit}]")

    ax.set_xlabel(f"Time [{time_unit}]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    # STRICT RULE: Legend at EAST outside plotting area
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, edgecolor='0.8')
    fig.tight_layout(rect=[0, 0, 0.82, 1])

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig, ax
