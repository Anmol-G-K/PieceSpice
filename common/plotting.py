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

from pathlib import Path
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


# ---------------------------------------------------------------------------
# Tiered figure output. Notebooks pass arrays + metadata; all placement and
# resolution logic stays here (house rule: helpers own figure paths).
# ---------------------------------------------------------------------------
FIGURES_ROOT = Path(__file__).resolve().parents[1] / "figures"
_VALID_TIERS = (None, "tier2", "tier3")


def figure_path(vendor: str, filename: str, tier: Optional[str] = None) -> Path:
    """Resolve ``figures/<vendor>/[tier]/<filename>`` and create its parent.

    ``tier`` selects the comparison subdirectory (``tier2``/``tier3``);
    ``None`` keeps extraction/analysis figures at the vendor root.
    """
    if tier not in _VALID_TIERS:
        raise ValueError(f"tier must be one of {_VALID_TIERS}, got {tier!r}")
    parts = (vendor,) if tier is None else (vendor, tier)
    path = FIGURES_ROOT.joinpath(*parts, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, vendor: str, filename: str,
                tier: Optional[str] = None) -> Path:
    """Save a figure into the tiered layout at 300 dpi; return the path."""
    path = figure_path(vendor, filename, tier)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


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


def plot_waveform_comparison(
    t,
    left_families: Dict[str, object],
    right_families: Dict[str, object],
    *,
    vendor: str,
    filename: str,
    tier: Optional[str] = None,
    left_label: str = "left",
    right_label: str = "right",
    title: str = "",
    xlabel: str = "Time [ns]",
    figsize: Tuple[float, float] = (10.5, 8.0),
    show: bool = False,
) -> Tuple[plt.Figure, Tuple[plt.Axes, Optional[plt.Axes]]]:
    """Two stacked panels overlaying golden/tier families on a shared time axis.

    Generic: accepts only arrays and metadata. Each panel gets an EAST legend
    outside the axes; the figure is saved at 300 dpi under the tier figures
    directory selected by ``tier``.
    """
    setup_style()
    if right_families:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                       layout="constrained")
    else:
        fig, ax1 = plt.subplots(1, 1,
                                figsize=(figsize[0], figsize[1] / 2),
                                layout="constrained")
        ax2 = None
    for label, y in left_families.items():
        ax1.plot(t, y, label=label)
    if ax2 is not None:
        for label, y in right_families.items():
            ax2.plot(t, y, label=label)
    axes = [ax1] if ax2 is None else [ax1, ax2]
    for ax, ylabel in zip(axes, (left_label, right_label)):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35, ls="--")
        _east_legend(ax)
    (ax2 if ax2 is not None else ax1).set_xlabel(xlabel)
    if title:
        fig.suptitle(title)
    save_figure(fig, vendor, filename, tier)
    if show:
        plt.show()
    return fig, (ax1, ax2)


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


def plot_dual_axis(
    x,
    left,
    right,
    *,
    left_label: str = "left",
    right_label: str = "right",
    title: str = "",
    xlabel: str = "Time [ns]",
    colors: Tuple[str, str] = ("tab:blue", "tab:red"),
    save_path=None,
    show: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Two generic y-series on one x axis with a single EAST-outside legend."""
    setup_style()
    fig, ax1 = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
    l1 = ax1.plot(x, left, color=colors[0], label=left_label)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(left_label, color=colors[0])
    ax2 = ax1.twinx()
    l2 = ax2.plot(x, right, color=colors[1], label=right_label)
    ax2.set_ylabel(right_label, color=colors[1])
    ax1.set_title(title)
    ax1.grid(True, alpha=0.35, ls="--")
    ax1.legend(l1 + l2, [left_label, right_label],
               loc="center left", bbox_to_anchor=(1.14, 0.5))
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Converter benchmark. One helper owns the four-panel geometry, the mandatory
# EAST legend and the ``figures/converter/`` output; notebooks pass the result
# dict from simulation.testbenches.buck_converter unchanged.
# ---------------------------------------------------------------------------

def plot_converter_benchmark(
    result: dict,
    *,
    filename: Optional[str] = None,
    save_path=None,
    figsize: Tuple[float, float] = (12.0, 8.8),
    show: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    """Four panels: Vout, inductor current, switch node, power/loss summary.

    ``result`` is the record returned by
    :func:`simulation.testbenches.buck_converter.run_converter_benchmark`.
    All axes use the house style (EAST-outside legends, engineering units,
    dashed grid). If ``save_path`` is omitted the figure is written at 300 dpi
    to ``figures/converter/<vendor>_<tier>_buck.png`` -- a fixed directory,
    never a per-run folder.
    """
    setup_style()
    traces = result.get("traces") or {}
    if "time" not in traces:
        raise ValueError("converter result has no 'time' trace to plot")
    t = np.asarray(traces["time"], dtype=float)
    vout = np.asarray(traces.get("vout_v", []), dtype=float)
    i_l = np.asarray(traces.get("i_l_a", []), dtype=float)
    v_sw = np.asarray(traces.get("v_sw_v", []), dtype=float)
    if vout.size != t.size or i_l.size != t.size or v_sw.size != t.size:
        raise ValueError("converter result traces must share the time axis")

    conditions = result.get("conditions") or {}
    target = conditions.get("vout_target_v")
    window = (result.get("extraction") or {}).get("window_s")
    vendor = str(result.get("vendor") or "unknown")
    tier = str(result.get("tier_key") or result.get("tier") or "model")

    fig, axs = plt.subplots(2, 2, figsize=figsize, layout="constrained")
    ax_v, ax_i = axs[0]
    ax_sw, ax_loss = axs[1]

    ax_v.plot(t, vout, label="Vout")
    if target is not None:
        ax_v.axhline(float(target), ls=":", color="0.35",
                     label=f"target {float(target):g} V")
    if window:
        ax_v.axvline(float(window[0]), ls="--", color="tab:red", lw=1.0,
                     label="measurement window")
    ax_v.set_ylabel("Output voltage")
    ax_v.set_title("Output voltage")
    _apply_eng_formatter(ax_v, "x", "s")
    _apply_eng_formatter(ax_v, "y", "V")
    _east_legend(ax_v)

    ax_i.plot(t, i_l, label="I_L")
    ax_i.set_ylabel("Inductor current")
    ax_i.set_title("Inductor current")
    _apply_eng_formatter(ax_i, "x", "s")
    _apply_eng_formatter(ax_i, "y", "A")
    _east_legend(ax_i)

    ax_sw.plot(t, v_sw, label="V_sw", lw=1.2)
    ax_sw.set_xlabel("Time")
    ax_sw.set_ylabel("Switch-node voltage")
    ax_sw.set_title("Switch node")
    _apply_eng_formatter(ax_sw, "x", "s")
    _apply_eng_formatter(ax_sw, "y", "V")
    _east_legend(ax_sw)

    # Power budget: only measured terms are drawn; N/A terms are named in the
    # title instead of being plotted as a fabricated zero.
    loss = result.get("loss") or {}
    terms = (
        ("P_in", result.get("pin_w")),
        ("P_out", result.get("pout_w")),
        ("HS cond.", loss.get("high_side_conduction_w")),
        ("LS cond.", loss.get("low_side_conduction_w")),
        ("freewheel", loss.get("freewheel_conduction_w")),
        ("L DCR", loss.get("inductor_dcr_w")),
        ("C ESR", loss.get("output_esr_w")),
        ("sw residual", loss.get("switching_residual_w")),
    )
    labels = [label for label, value in terms
              if value is not None and np.isfinite(float(value))]
    values = [float(value) for _, value in terms
              if value is not None and np.isfinite(float(value))]
    if values:
        ax_loss.bar(labels, values, color="tab:blue")
        ax_loss.tick_params(axis="x", labelrotation=20)
    else:
        ax_loss.text(0.5, 0.5, "power terms N/A", ha="center", va="center",
                     transform=ax_loss.transAxes)
        ax_loss.set_xticks([])
    efficiency = result.get("efficiency")
    if efficiency is not None and np.isfinite(float(efficiency)):
        eff_text = f"efficiency = {float(efficiency) * 100.0:.2f} %"
    else:
        reason = result.get("reason") or "unmeasured"
        eff_text = f"efficiency = N/A ({reason})"
    ax_loss.set_title(f"Power budget | {eff_text}", fontsize=12)
    ax_loss.set_ylabel("Power")
    _apply_eng_formatter(ax_loss, "y", "W")

    if save_path is None:
        filename = filename or f"{vendor}_{tier}_buck.png"
        save_path = figure_path("converter", filename)
    else:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axs


def _comparison_run_trace(run, key: str):
    """Return ``(time, values)`` only for a completed, measured trace."""
    if not isinstance(run, dict):
        return None
    if not run.get("completed") or run.get("status") not in (
            "measured", "not_settled"):
        return None
    traces = run.get("traces") or {}
    time = traces.get("time")
    values = traces.get(key)
    if time is None or values is None:
        return None
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size != y.size or t.ndim != 1 or y.ndim != 1:
        return None
    finite = np.isfinite(t) & np.isfinite(y)
    if int(finite.sum()) < 2:
        return None
    return t[finite], y[finite]


def plot_converter_comparison(
    result: dict,
    *,
    filename: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (15.0, 4.8),
) -> Tuple[plt.Figure, np.ndarray]:
    """Golden-vs-tier buck comparison: exactly one row of three panels.

    Panels are Vout, inductor current and capacitor current. Each run is
    plotted only when it completed and carries the trace; an N/A run is
    labelled in the panel, never drawn as a fabricated zero. Fixed output:
    ``figures/converter/<vendor>_<tier>_buck_compare.png`` at 300 dpi with
    EAST-outside legends and engineering-unit axes.
    """
    setup_style()
    golden = result.get("golden") or {}
    tier = result.get("tier") or {}
    vendor = str(result.get("vendor") or "unknown")
    tier_key = str(result.get("tier_key") or "tier")
    golden_label = "golden"
    tier_label = f"tier ({tier_key})"

    fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    specs = (
        ("Output voltage", "vout_v", "V"),
        ("Inductor current", "i_l_a", "A"),
        ("Capacitor current", "i_cap_a", "A"),
    )
    styles = (
        ("golden", golden, golden_label, "tab:blue", "-"),
        ("tier", tier, tier_label, "tab:red", "--"),
    )
    for ax, (title, key, unit) in zip(axes, specs):
        plotted = 0
        n_a: list[str] = []
        for name, run, label, color, linestyle in styles:
            traced = _comparison_run_trace(run, key)
            if traced is None:
                state = run.get("status") or "no trace"
                n_a.append(f"{name}: N/A ({state})")
                continue
            t, y = traced
            ax.plot(t, y, color=color, linestyle=linestyle, label=label)
            plotted += 1
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(title)
        _apply_eng_formatter(ax, "x", "s")
        _apply_eng_formatter(ax, "y", unit)
        if plotted:
            _east_legend(ax)
        for index, text in enumerate(n_a):
            ax.text(0.02, 0.96 - 0.09 * index, text, transform=ax.transAxes,
                    va="top", ha="left", fontsize=9, color="0.3")

    fig.suptitle(f"Buck converter comparison: golden vs {tier_key}")
    filename = filename or f"{vendor}_{tier_key}_buck_compare.png"
    path = figure_path("converter", filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes
