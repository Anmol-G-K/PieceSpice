"""Standardized plotting utility for PieceSpice.

MANDATORY PLOTTING CONSTRAINT:
All plot legends MUST be located on the EAST side (outside plotting area to the right).

Design goals (publication quality):
- 300 dpi everywhere (display and export).
- Constrained layout: matplotlib reserves space for the outside legend automatically,
  so the axes area stays dominant without manual two-pass tight_layout hacks.
- Engineering-unit tick formatting (nF, pF, mV, kV ...) derived from the curve units.
- Serif/STIX typography, boxed axes, inward minor ticks on all four sides, subtle
  dashed grid.
"""

from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter, LogLocator, NullFormatter

from common.types import CurveData, WaveformData


def setup_style():
    """Configure matplotlib defaults for publication-quality, highly legible plots."""
    plt.rcParams.update({
        # Academic typography: serif body + STIX math (journal-style)
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlepad': 12,
        'axes.labelpad': 8,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 15,
        # Resolution: 300 dpi everywhere (inline render and export)
        'figure.dpi': 300,
        'savefig.dpi': 300,
        # Boxed axes with inward ticks on all four sides (IEEE/journal style)
        'axes.linewidth': 0.9,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.major.size': 4.5,
        'xtick.minor.size': 2.5,
        'ytick.major.size': 4.5,
        'ytick.minor.size': 2.5,
        # Subtle dashed grid (major only; minor grid is visual noise)
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'axes.axisbelow': True,
        # Traces
        'lines.linewidth': 1.8,
        'lines.markersize': 4.5,
        # Legend box
        'legend.framealpha': 1.0,
        'legend.edgecolor': '0.75',
        'legend.fancybox': False,
        'legend.borderpad': 0.5,
        'legend.handlelength': 1.8,
        'legend.handletextpad': 0.6,
        'legend.labelspacing': 0.45,
    })


def _apply_eng_formatter(ax, axis: str, unit: Optional[str]):
    """Attach an engineering-notation tick formatter (and unit) to an axis."""
    if not unit:
        return
    formatter = EngFormatter(unit=unit, sep="")
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_formatter(formatter)
    if axis == "y":
        # On log axes the default minor formatter duplicates the major labels.
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))


def _east_legend(ax):
    """Legend strictly at EAST, outside the axes. Constrained layout reserves room."""
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    leg.set_in_layout(True)
    return leg


def _new_figure(figsize: Tuple[float, float] = (10.5, 5.8)):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    return fig, ax


def plot_curves(
    curves: List[CurveData],
    title: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    x_log: bool = False,
    y_log: bool = False,
    figsize: Tuple[float, float] = (10.5, 5.8),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot characteristic curves with legend strictly at EAST (outside).

    Compact academic labels: a curve family entry shows only its parameter value
    ("Vgs = 4.5"); the title carries the study name. Tick labels use engineering
    notation derived from the curve units (e.g. pF, mV, mOhm).
    """
    setup_style()
    fig, ax = _new_figure(figsize)

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
        x_label = curves[0].x_name
    if not y_label and curves:
        y_label = curves[0].y_name
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if curves:
        _apply_eng_formatter(ax, "x", curves[0].x_unit)
        _apply_eng_formatter(ax, "y", curves[0].y_unit)
    if len(curves) > 1 or (curves and curves[0].family_param is not None):
        _east_legend(ax)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig, ax


def plot_waveforms(
    waveforms: List[WaveformData],
    title: str,
    time_unit: str = "ns",
    time_scale: float = 1e9,
    figsize: Tuple[float, float] = (10.5, 5.8),
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot transient time-domain waveforms with legend strictly at EAST (outside)."""
    setup_style()
    fig, ax = _new_figure(figsize)

    for wf in waveforms:
        t, v = wf.to_numpy()
        ax.plot(t * time_scale, v, label=f"{wf.name} [{wf.unit}]")

    ax.set_xlabel(f"Time [{time_unit}]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    _east_legend(ax)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig, ax


# ---------------------------------------------------------------------------
# Generic analysis plots. These accept plain arrays only - no device-, vendor-
# or notebook-specific logic - so every notebook renders the same geometry.
# ---------------------------------------------------------------------------

def plot_families(
    x,
    families: Dict[str, object],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    xunit: Optional[str] = None,
    yunit: Optional[str] = None,
    x_log: bool = False,
    y_log: bool = False,
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10.5, 5.8),
    save_path=None,
    show: bool = True,
    vlines: Optional[list] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot any number of labelled y-vs-x series on one axes.

    House rules (EAST legend, engineering units, dashed grid, 300 dpi) are
    applied here, never in notebooks. Pass ``ax`` to compose panels; in that
    case saving/displaying is left to the caller.
    """
    setup_style()
    if ax is None:
        fig, ax = _new_figure(figsize)
    else:
        fig = ax.figure
    for i, (label, y) in enumerate(families.items()):
        color = colors[i] if colors and i < len(colors) else None
        ax.plot(x, y, label=label, color=color)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35, ls="--")
    if xunit:
        _apply_eng_formatter(ax, "x", xunit)
    if yunit:
        _apply_eng_formatter(ax, "y", yunit)
    if vlines:
        for xm in vlines:
            ax.axvline(xm, ls="--", color="tab:red", lw=1)
    if len(families) > 1 or any(families):
        _east_legend(ax)
    if save_path is not None and ax is None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show and ax is None:
        plt.show()
    return fig, ax


def plot_xy(x, y, *, label: Optional[str] = None, **kwargs):
    """Single-series convenience wrapper around :func:`plot_families`."""
    return plot_families(x, {label or "": y}, **kwargs)


def plot_fit(
    x, y, x_dense, y_fit, breakpoints, *,
    title: str = "", xlabel: str = "", ylabel: str = "",
    xunit: Optional[str] = None, yunit: Optional[str] = None,
    save_path=None, show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Samples + fitted curve + breakpoint markers (generic PWL diagnostic)."""
    setup_style()
    fig, ax = plot_xy(x, y, label="extracted samples", title=title,
                      xlabel=xlabel, ylabel=ylabel, xunit=xunit, yunit=yunit,
                      show=False)
    ax.plot(x_dense, y_fit, lw=2, label="PWL fit")
    for bp in breakpoints:
        ax.axvline(bp, ls=":", color="gray", lw=1)
    _east_legend(ax)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_compression(
    x, y, xc, yc, *,
    title: str = "", xlabel: str = "", ylabel: str = "",
    xunit: Optional[str] = None, yunit: Optional[str] = None,
    save_path=None, show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Full curve + retained points + chord (generic simplification plot)."""
    setup_style()
    fig, ax = plot_families(
        x, {"full curve": y}, title=title, xlabel=xlabel, ylabel=ylabel,
        xunit=xunit, yunit=yunit, show=False)
    ax.plot(xc, yc, "o", ms=7, mfc="none", color="tab:red", lw=1,
            label=f"retained points ({len(xc)} of {len(x)})")
    ax.plot(xc, yc, "-", lw=1, color="tab:red", alpha=0.5)
    _east_legend(ax)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_charge(
    v, q, v_breaks, q_values, c, *,
    title: str = "Q(V) and C(V)", xlabel: str = "Voltage",
    qylabel: str = "Charge", cylabel: str = "Capacitance",
    qunit: Optional[str] = None, cunit: Optional[str] = None,
    save_path=None, show: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Two panels: integrated Q(V) with its PWL table, and C(V)."""
    setup_style()
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.4), layout="constrained")
    plot_xy(v, q, label="Q(V) integrated", xlabel=xlabel, ylabel=qylabel,
            yunit=qunit, ax=axs[0], show=False)
    plot_xy(v_breaks, q_values, label=f"PWL table ({len(q_values)} pts)",
            xlabel=xlabel, ylabel=qylabel, yunit=qunit,
            ax=axs[0], show=False)
    plot_xy(v, c, label="C(V)", xlabel=xlabel, ylabel=cylabel, yunit=cunit,
            y_log=True, ax=axs[1], show=False)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axs


def plot_knee(
    v_hard, i_hard, v_smooth, i_smooth, *,
    title: str = "Hard PWL vs smoothed knee",
    xlabel: str = "Voltage", ylabel: str = "Current",
    xunit: Optional[str] = None, yunit: Optional[str] = None,
    save_path=None, show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Hard piecewise curve overlaid with its smooth approximation."""
    setup_style()
    fig, ax = plot_families(
        v_hard, {"hard PWL": i_hard}, title=title, xlabel=xlabel,
        ylabel=ylabel, xunit=xunit, yunit=yunit, show=False)
    ax.plot(v_smooth, i_smooth, lw=2, label="smoothed knee")
    _east_legend(ax)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_endpoint_ties(
    x, golden, tier2, *,
    title: str = "Golden vs candidate endpoints",
    xlabel: str = "Gate voltage", ylabel: str = "Current",
    xunit: Optional[str] = None, yunit: Optional[str] = None,
    save_path=None, show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """Paired golden/candidate values with tie-lines whose length is the error."""
    setup_style()
    fig, ax = _new_figure()
    for xv, g, t in zip(x, golden, tier2):
        ax.plot([xv, xv], [g, t], "-", color="gray", alpha=0.5, lw=1)
        ax.plot(xv, t, "o", ms=8, color="tab:red", label="candidate")
        ax.plot(xv, g, "x", ms=10, mew=2, color="tab:blue", label="golden")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35, ls="--")
    if xunit:
        _apply_eng_formatter(ax, "x", xunit)
    if yunit:
        _apply_eng_formatter(ax, "y", yunit)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend([handles[0], handles[1]], [labels[0], labels[1]],
                  loc="center left", bbox_to_anchor=(1.02, 0.5))
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
