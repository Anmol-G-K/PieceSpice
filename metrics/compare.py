"""Unified model-comparison framework (M3 evaluation metrics).

One place to answer "how good is our reduced model?" - per-curve error
normals, a global scorecard across all extracted curves, and industry-standard
comparison plots used in datasheet app notes / EDA vendor decks:

* Overlay plot          : golden vs Tier-2 on every curve family
* Error-vs-x plot       : signed residual trace per curve (log or linear)
* Bar-chart scorecard   : relative RMSE per curve with the <5% target line
* Parity/scatter plot   : candidate vs golden with 45-degree ideal line
* Speedup summary table : wall-clock + point-count compression statistics

All functions accept plain numpy arrays or CurveData; nothing is device
specific. Plotting follows house rules: EAST legends, 300 dpi, eng units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from common.types import CurveData
from metrics.static import compute_curve_error


# ---------------------------------------------------------------------------
# Per-curve / global metric containers
# ---------------------------------------------------------------------------

@dataclass
class CurveComparison:
    """Error normals between a golden and candidate curve on shared x."""

    name: str
    n_points: int
    rmse: float                    # absolute RMSE [y units]
    max_abs_error: float           # L-inf [y units]
    rel_rmse_pct: float            # RMSE / span(golden) * 100
    max_rel_error_pct: float       # L-inf / span(golden) * 100
    mape_pct: float                # mean |err|/|golden| where |golden|>eps
    r_squared: float
    passed: bool                   # within tolerance?
    x_unit: str = "V"
    y_unit: str = "A"


@dataclass
class ComparisonReport:
    """Global scorecard across many curves."""

    curves: List[CurveComparison] = field(default_factory=list)
    target_rel_rmse_pct: float = 5.0

    @property
    def worst(self) -> Optional[CurveComparison]:
        return max(self.curves, key=lambda c: c.rel_rmse_pct) \
            if self.curves else None

    @property
    def all_passed(self) -> bool:
        return bool(self.curves) and all(c.passed for c in self.curves)

    def summary_line(self) -> str:
        if not self.curves:
            return "no curves compared"
        w = self.worst
        return (f"{sum(c.passed for c in self.curves)}/{len(self.curves)} "
                f"curves within {self.target_rel_rmse_pct:.0f}% "
                f"rel-RMSE target; worst = {w.name} "
                f"({w.rel_rmse_pct:.2f}%)")


def compare_curve(
    x_golden: np.ndarray,
    y_golden: np.ndarray,
    y_candidate: np.ndarray,
    name: str,
    x_unit: str = "V",
    y_unit: str = "A",
    mape_eps: float = 1e-9,
) -> CurveComparison:
    """Compare one golden/candidate pair on a common x grid.

    The candidate is interpolated onto the golden x-grid first, so the two
    curves need not be sampled identically.
    """
    xg = np.asarray(x_golden, dtype=float)
    yg = np.asarray(y_golden, dtype=float)
    xc = np.asarray(x_golden, dtype=float)
    yc_raw = np.asarray(y_candidate, dtype=float)

    order = np.argsort(xc)
    yc = np.interp(np.sort(xg), xc[order], yc_raw[order])
    yg_sorted = yg[np.argsort(xg)]
    xg_sorted = np.sort(xg)

    m = compute_curve_error(yg_sorted, yc)
    span = max(float(np.max(yg_sorted) - np.min(yg_sorted)), 1e-30)

    nz = np.abs(yg_sorted) > mape_eps
    mape = float(np.mean(np.abs((yc[nz] - yg_sorted[nz]) / yg_sorted[nz]))
                 * 100.0) if nz.any() else 0.0

    return CurveComparison(
        name=name,
        n_points=len(xg_sorted),
        rmse=m.rmse,
        max_abs_error=m.max_abs_error,
        rel_rmse_pct=m.relative_rmse_pct,
        max_rel_error_pct=m.max_abs_error / span * 100.0,
        mape_pct=mape,
        r_squared=m.r_squared,
        passed=m.relative_rmse_pct <= 5.0,
        x_unit=x_unit,
        y_unit=y_unit,
    )


def build_report(
    pairs: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    target_rel_rmse_pct: float = 5.0,
    x_unit: str = "V",
    y_unit: str = "A",
) -> ComparisonReport:
    """Build a global report from many (name, x, y_golden, y_cand) tuples."""
    report = ComparisonReport(target_rel_rmse_pct=target_rel_rmse_pct)
    for name, x, yg, yc in pairs:
        clean = name.removesuffix("_v")
        cc = compare_curve(x, yg, yc, name=clean,
                           x_unit=x_unit, y_unit=y_unit)
        cc.passed = cc.rel_rmse_pct <= target_rel_rmse_pct
        report.curves.append(cc)
    return report


# ---------------------------------------------------------------------------
# Industry-style comparison plots
# ---------------------------------------------------------------------------

def _style_ax(ax):
    ax.grid(True, alpha=0.35, ls="--")
    ax.set_axisbelow(True)


def plot_overlays(
    pairs: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    ncols: int = 2,
    figsize_per=(6.4, 4.6),
    cand_label: str = "Tier-2 CPWL",
    gold_label: str = "Vendor SPICE",
    save_path: Optional[str] = None,
):
    """Golden-vs-candidate overlay small-multiples (one panel per curve).

    This is the layout datasheet/app-note figures use: each device curve in
    its own panel, both models overlaid, so shape fidelity is visible per
    bias condition.
    """
    import math
    n = len(pairs)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(figsize_per[0] * ncols,
                                     figsize_per[1] * nrows),
                            layout="constrained", squeeze=False)
    for idx, (name, x, yg, yc) in enumerate(pairs):
        ax = axs[idx // ncols][idx % ncols]
        ax.plot(x, yg, lw=2, color="tab:blue", label=gold_label)
        ax.plot(x, yc, lw=1.6, ls="--", color="tab:red", label=cand_label)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Vds [V]")
        ax.set_ylabel("Id [A]")
        _style_ax(ax)
        ax.legend(loc="lower right", fontsize=9)
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].set_visible(False)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_residuals(
    pairs: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    normalize_to: str = "span",      # 'span' | 'max' | 'none'
    save_path: Optional[str] = None,
):
    """Signed residual traces (candidate - golden) per curve.

    EDA-standard way to show WHERE a compact model errs, not just how much.
    """
    fig, ax = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
    for name, x, yg, yc in pairs:
        resid = yc - yg
        if normalize_to == "span":
            denom = max(float(yg.max() - yg.min()), 1e-30)
            resid = resid / denom * 100.0
            unit = "% of span"
        elif normalize_to == "max":
            denom = max(abs(float(yg.max())), 1e-30)
            resid = resid / denom * 100.0
            unit = "% of max"
        else:
            unit = "A"
        ax.plot(x, resid, lw=1.8, marker="", label=f"{name}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Vds [V]")
    ax.set_ylabel(f"Residual (cand - golden) [{unit}]")
    ax.set_title("Signed Residual Traces")
    _style_ax(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_scorecard_bars(
    report: ComparisonReport,
    save_path: Optional[str] = None,
):
    """Horizontal bar chart of rel-RMSE per curve with the target line.

    The 'scorecard' slide from vendor comparison decks: one bar per curve,
    colored green under target / red over, with a vertical pass/fail line.
    """
    names = [c.name for c in report.curves]
    vals = [c.rel_rmse_pct for c in report.curves]
    colors = ["#2ca02c" if v <= report.target_rel_rmse_pct else "#d62728"
              for v in vals]

    ypos = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(10.5, max(3.2, 0.7 * len(names) + 1.6)),
                           layout="constrained")
    bars = ax.barh(ypos, vals, color=colors, edgecolor="0.25", height=0.62)
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals) * 0.01, y, f"{v:.2f}%",
                va="center", fontsize=10)

    ax.axvline(report.target_rel_rmse_pct, color="crimson", ls="--", lw=1.4,
               label=f"target ≤ {report.target_rel_rmse_pct:.0f}%")
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Relative RMSE [% of span]  (lower is better)")
    ax.set_title("Model Fidelity Scorecard — Relative RMSE by Curve")
    ax.set_xlim(0, max(vals) * 1.18)
    _style_ax(ax)
    ax.legend(loc="lower right", fontsize=10)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_parity(
    pairs: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
):
    """Parity (candidate-vs-golden) scatter with the 45° ideal line.

    Standard regression-style fidelity figure: points hugging the diagonal
    mean perfect agreement; systematic offsets reveal bias regions.
    """
    all_g = np.concatenate([np.asarray(g, float).ravel()
                            for _, _, g, _ in pairs])
    lo, hi = 0.0, float(all_g.max()) * 1.05

    fig, ax = plt.subplots(figsize=(7.2, 7.2), layout="constrained")
    for name, x, yg, yc in pairs:
        xc = np.asarray(x, dtype=float)
        order = np.argsort(xc)
        interp = np.interp(np.sort(xc), xc[order], np.asarray(yc)[order])
        ax.plot(np.sort(np.asarray(yg)[order]), interp, "o", ms=5,
                alpha=0.75, label=name)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="ideal (y = x)")
    ax.set_xlabel("Vendor SPICE Id [A]")
    ax.set_ylabel("Tier-2 CPWL Id [A]")
    ax.set_title("Parity Plot — Model vs Golden")
    _style_ax(ax)
    ax.legend(loc="upper left", fontsize=9)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Summary tables (DataFrame-friendly)
# ---------------------------------------------------------------------------

def report_to_rows(report: ComparisonReport) -> List[Dict]:
    """Flatten a report into dicts for pandas DataFrame construction."""
    return [
        {
            "curve": c.name,
            "points": c.n_points,
            "rmse": round(c.rmse, 6),
            "rel_rmse_pct": round(c.rel_rmse_pct, 3),
            "max_err": round(c.max_abs_error, 6),
            "max_rel_pct": round(c.max_rel_error_pct, 3),
            "mape_pct": round(c.mape_pct, 3),
            "r_squared": round(c.r_squared, 6),
            "pass": c.passed,
        }
        for c in report.curves
    ]
