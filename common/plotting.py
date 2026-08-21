"""Standardized plotting utility for PieceSpice.

MANDATORY PLOTTING CONSTRAINT:
All plot legends MUST be located on the EAST side (outside plotting area to the right).
"""

from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from common.types import CurveData, WaveformData


def setup_style():
    """Configure matplotlib defaults for publication-quality, highly legible plots."""
    plt.rcParams.update({
        # Academic typography: serif body + STIX math (journal-style)
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        # Resolution: 300 dpi everywhere (inline render and export)
        'figure.dpi': 300,
        'savefig.dpi': 300,
        # Boxed axes with inward ticks on all four sides (IEEE/journal style)
        'axes.linewidth': 1.0,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        # Subtle dashed grid
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'grid.linewidth': 0.7,
        # Traces
        'lines.linewidth': 2.0,
        'lines.markersize': 5,
        # Legend box
        'legend.framealpha': 1.0,
        'legend.edgecolor': '0.8',
        'figure.autolayout': False,
    })


def plot_curves(
    curves: List[CurveData],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    x_log: bool = False,
    y_log: bool = False,
    figsize: Tuple[float, float] = (12, 6.75),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a collection of characteristic curves with legend strictly at EAST (outside).

    The wide default figure keeps the axes area dominant even with long legend
    labels; tight_layout reserves only the strip the legend actually needs.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    # Compact academic labels: the title already names the study, so a curve
    # family entry only needs its parameter value ("Vgs = 4.5"), not the
    # full study name repeated five times.
    labels: List[str] = []
    for curve in curves:
        if curve.family_param and curve.family_value is not None:
            labels.append(f"{curve.family_param} = {curve.family_value:g}")
        else:
            labels.append(curve.name)

    for curve, label in zip(curves, labels):
        x, y = curve.to_numpy()
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

    # STRICT RULE: Legend at EAST outside plotting area.
    # Compact box; two-pass layout: measure legend width, reserve only that
    # strip, re-measure (axes grew -> legend fraction shrank), converge.
    show_legend = len(curves) > 1 or (curves[0].family_param is not None)
    leg = None
    if show_legend:
        leg = ax.legend(
            loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True,
            handlelength=1.6, handletextpad=0.6, labelspacing=0.4,
            borderpad=0.6,
        )
    if leg is not None:
        for _ in range(2):
            fig.canvas.draw()
            leg_w = leg.get_window_extent().width / fig.canvas.get_width_height()[0]
            fig.tight_layout(rect=[0, 0, max(1.0 - leg_w - 0.02, 0.70), 1])
    else:
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig, ax


def plot_waveforms(
    waveforms: List[WaveformData],
    title: str,
    time_unit: str = "ns",
    time_scale: float = 1e9,
    figsize: Tuple[float, float] = (12, 6.75),
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

    # STRICT RULE: Legend at EAST outside plotting area (two-pass sizing).
    leg = ax.legend(
        loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=True,
        handlelength=1.6, handletextpad=0.6, labelspacing=0.4, borderpad=0.6,
    )
    for _ in range(2):
        fig.canvas.draw()
        leg_w = leg.get_window_extent().width / fig.canvas.get_width_height()[0]
        fig.tight_layout(rect=[0, 0, max(1.0 - leg_w - 0.02, 0.70), 1])

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig, ax
