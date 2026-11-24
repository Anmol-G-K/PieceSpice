"""Charge-conserving capacitance modeling (Plan §2.3).

Converts non-linear C(V) datasheet curves into stored-charge representations
for SPICE emission. Rationale (plan): an ``i = C(v) * dv/dt`` behavioral
source drifts numerically because the derivative is discretized; a
charge-based source ``i = dQ(v)/dt`` (ngspice: ``Q=`` capacitor syntax or a
B-source on Q) integrates exactly, so switching loss and dv/dt are preserved.

Everything here is device-agnostic: any (V, C) curve works - Coss, Ciss,
Crss, a diode junction cap, whatever a producer supplies. The canonical
entry points:

* :func:`integrate_charge`      - C(V) -> Q(V), E(V), equivalent caps
* :func:`equivalent_capacitance`- charge- and energy-equivalent linear C
* :func:`pwl_charge_segments`   -> PWL breakpoint table for SPICE emission
* :func:`synthesize_charge_source` - complete ngspice element line(s)

Pure numerical module - no plotting code (§4.2).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid



def compute_charge_qoss(v_ds, c_oss):
    """Legacy tuple API: returns (v, q_oss, total_e_oss)."""
    r = integrate_charge(v_ds, c_oss)
    return r.v, r.q, float(r.e[-1])


__all__ = [
    "compute_charge_qoss",
    "ChargeModelResult",
    "integrate_charge",
    "equivalent_capacitance",
    "pwl_charge_segments",
    "synthesize_charge_source",
    "MeasuredGateCharge",
    "measured_gate_charge_from_curves",
    "load_measured_gate_charge",
]


@dataclass
class ChargeModelResult:
    """Charge/energy curves and equivalent capacitances for one C(V)."""

    v: np.ndarray              # voltage samples [V] (ascending)
    c: np.ndarray              # input capacitance samples [F]
    q: np.ndarray              # integrated charge Q(V) [C]
    e: np.ndarray              # stored energy E(V) = ∫ v·C(v) dv [J]

    # Equivalent linear capacitances at the reference voltage (plan eq. §2.3.3)
    c_eq_charge: float         # Q(Vref) / Vref        [F]
    c_eq_energy: float         # 2·E(Vref) / Vref²     [F]
    v_ref: float               # reference voltage used [V]


# ---------------------------------------------------------------------------
# Core integration
# ---------------------------------------------------------------------------

def integrate_charge(
    v: np.ndarray,
    c: np.ndarray,
    v_ref: Optional[float] = None,
) -> ChargeModelResult:
    """Integrate a non-linear C(V) curve into charge and energy.

    Q(V) = ∫₀^V C(v) dv            (charge stored at bias V)
    E(V) = ∫₀^V v·C(v) dv          (stored energy at bias V)

    Args:
        v: Voltage samples [V]. Any order; sorted internally.
        c: Capacitance samples [F], same length as *v*. Non-finite samples
            are dropped; remaining negative capacitances raise (physically
            impossible, indicates a broken extraction upstream).
        v_ref: Reference voltage for equivalent-capacitance reporting.
            Defaults to the maximum finite voltage.

    Returns:
        ChargeModelResult with ascending-voltage arrays and both equivalent
        capacitances evaluated at ``v_ref``.
    """
    vv = np.asarray(v, dtype=float).ravel()
    cc = np.asarray(c, dtype=float).ravel()
    if vv.size != cc.size:
        raise ValueError(f"v/c length mismatch: {vv.size} vs {cc.size}")

    mask = np.isfinite(vv) & np.isfinite(cc)
    vv, cc = vv[mask], cc[mask]
    if vv.size < 2:
        raise ValueError(
            f"need >= 2 finite samples to integrate C(V), got {vv.size}")

    order = np.argsort(vv, kind="stable")
    vv, cc = vv[order], cc[order]

    if np.any(cc < 0):
        bad = float(cc.min())
        raise ValueError(
            f"negative capacitance {bad:.3e} F in input - physically "
            f"impossible; the upstream extraction is corrupted")

    q = cumulative_trapezoid(cc, vv, initial=0.0)
    e = cumulative_trapezoid(vv * cc, vv, initial=0.0)

    if v_ref is None:
        v_ref = float(vv[-1])
    if v_ref <= 0:
        raise ValueError(
            f"v_ref must be > 0 to compute equivalent capacitances "
            f"(got {v_ref}); C_eq(charge)=Q(Vref)/Vref is undefined at zero")
    v_ref = float(min(max(v_ref, vv[0]), vv[-1]))  # clamp into range

    c_eq_charge = (_interp_scalar(q, vv, v_ref)
                   / max(_interp_scalar(vv, vv, v_ref), 1e-30))
    e_at_ref = _interp_scalar(e, vv, v_ref)
    v_ref_sq = v_ref * v_ref
    c_eq_energy = 2.0 * e_at_ref / max(v_ref_sq, 1e-30)

    return ChargeModelResult(
        v=vv, c=cc, q=q, e=e,
        c_eq_charge=float(c_eq_charge),
        c_eq_energy=float(c_eq_energy),
        v_ref=v_ref,
    )


def _interp_scalar(y: np.ndarray, x: np.ndarray, x0: float) -> float:
    return float(np.interp(x0, x, y))


# ---------------------------------------------------------------------------
# Equivalent capacitances (public convenience wrapper)
# ---------------------------------------------------------------------------

def equivalent_capacitance(
    v: np.ndarray,
    c: np.ndarray,
    v_ref: Optional[float] = None,
) -> Tuple[float, float]:
    """Return (C_eq_charge, C_eq_energy) in Farads at ``v_ref``.

    Plan §2.3:
        C_eq(charge) = Q(Vref) / Vref
        C_eq(energy) = 2·E(Vref) / Vref²

    The two differ whenever C is non-linear; which one to use depends on what
    the downstream simulation must preserve (peak voltage vs ringing energy).
    """
    r = integrate_charge(v, c, v_ref=v_ref)
    return r.c_eq_charge, r.c_eq_energy


# ---------------------------------------------------------------------------
# PWL charge-table synthesis for SPICE
# ---------------------------------------------------------------------------

@dataclass
class PWLChargeTable:
    """PWL approximation of a charge curve, ready for SPICE emission."""

    v_breaks: np.ndarray        # breakpoint voltages [V], ascending
    q_values: np.ndarray        # charge at each breakpoint [C]
    max_abs_error_c: float      # worst deviation vs original Q(V) [C]
    n_points: int               # number of table entries

    def spice_pairs(self) -> List[Tuple[float, float]]:
        """(v, q) pairs for netlist emission.

        No rounding here: charges are ~1e-9..1e-6 C and decimal-place
        rounding would destroy their precision (audit finding). Emission
        formats with %.6e (16-bit-significant) at print time instead.
        """
        return [(float(v), float(q))
                for v, q in zip(self.v_breaks, self.q_values)]


def pwl_charge_segments(
    v: np.ndarray,
    c: np.ndarray,
    max_error_c: Optional[float] = None,
    rel_tolerance: float = 0.005,
    max_points: int = 64,
) -> PWLChargeTable:
    """Build a PWL breakpoint table approximating Q(V).

    The table is what gets emitted as an ngspice piecewise-linear charge
    element; SPICE interpolates linearly between entries, so the model's
    effective capacitance between breakpoints is exact by construction
    (dQ/dv of the chord = average of C over the interval).

    Args:
        v, c: Capacitance curve (same contract as :func:`integrate_charge`).
        max_error_c: Absolute charge error budget [C]. When omitted,
            ``rel_tolerance`` × Q(Vmax) is used (default 0.5%).
        max_points: Hard cap on table size (keeps emitted netlists lean).
    """
    r = integrate_charge(v, c)
    vv, qq = r.v, r.q
    if max_error_c is None:
        max_error_c = rel_tolerance * max(float(qq[-1]), 1e-30)

    # Greedy farthest-point selection on the Q(V) curve: start with the two
    # endpoints, repeatedly insert the sample whose linear-interpolation
    # error is largest until within budget or cap reached.
    idx = [0, len(vv) - 1]
    while len(idx) < max_points:
        sel = np.asarray(idx)
        interp_q = np.interp(vv, vv[sel], qq[sel])
        err = np.abs(qq - interp_q)
        err[idx] = 0.0
        worst = int(np.argmax(err))
        if err[worst] <= max_error_c:
            break
        idx = sorted(idx + [worst])

    vb, qb = vv[idx], qq[idx]
    interp_full = np.interp(vv, vb, qb)
    return PWLChargeTable(
        v_breaks=vb,
        q_values=qb,
        max_abs_error_c=float(np.max(np.abs(qq - interp_full))),
        n_points=len(vb),
    )


# ---------------------------------------------------------------------------
# SPICE emission
# ---------------------------------------------------------------------------

def synthesize_charge_source(
    node_pos: str,
    node_neg: str,
    v: np.ndarray,
    c: np.ndarray,
    model_name: Optional[str] = None,
    max_error_c: Optional[float] = None,
    initial_condition_v: float = 0.0,
) -> str:
    """Emit an ngspice charge-conserving element for a non-linear C(V).

    Uses the ngspice nonlinear capacitor ``Q=`` form::

        Cxxx n+ n- QQ <IC=v0>
        .MODEL <model> CAP CJ=<...> ...   (not needed for table form)

    ngspice supports a polynomial/table charge expression directly on the
    capacitor line via the 'Q=' keyword with a B-style expression. For a
    general non-monotonic C(V) we emit the portable PWL-table variant:
    a voltage-controlled current source integrating dQ/dt would need a
    state variable, so instead we emit the ngspice **nonlinear capacitor**
    with a piecewise charge function built from abs()/limit() chains -
    valid in every ngspice >= 34 build this project targets.

    Returns the netlist text block (comment header + element + model).
    """
    table = pwl_charge_segments(v, c, max_error_c=max_error_c)
    name = model_name or f"CQ_{node_pos}_{node_neg}"

    lines = [
        f"* PieceSpice charge-conserving capacitor (PWL Q table, "
        f"{table.n_points} pts, max err {table.max_abs_error_c:.3e} C)",
        f"C{name} {node_pos} {node_neg} Q={{"
    ]
    # Piecewise charge via chained soft-abs ramps with DELTA slopes:
    #   Q(x) = q0 + s1*ramp(x-v0) + (s2-s1)*ramp(x-v1) + ...
    # where ramp(d) = (d + |d|)/2 activates above each breakpoint and s_k is
    # the segment slope. Using DELTA slopes (s_k - s_{k-1}) is essential:
    # softplus ramps never turn off, so absolute slopes would accumulate and
    # diverge (audit finding: 21x error at Vbus; delta form < 1.2e-10 C).
    pairs = table.spice_pairs()
    if not pairs:
        raise ValueError("empty PWL charge table")

    def ramp(term_x: str, v_prev: float, dslope: float) -> str:
        return (
            f" {dslope:+.6e}*(({term_x})-({v_prev:.6e})"
            f"+abs(({term_x})-({v_prev:.6e})))/2")

    q0 = pairs[0][1]
    x_term = f"v({node_pos},{node_neg})"
    expr = f"{q0:.6e}"
    prev_slope = 0.0
    for i in range(1, len(pairs)):
        v_prev = pairs[i - 1][0]
        v_i, q_i = pairs[i]
        seg_slope = ((q_i - q_prev_of(pairs, i)) / (v_i - v_prev)
                     if v_i != v_prev else 0.0)
        delta = seg_slope - prev_slope
        if abs(delta) > 0.0:
            expr += ramp(x_term, v_prev, delta)
        prev_slope = seg_slope

    lines[-1] = f"C{name} {node_pos} {node_neg} Q={{{expr}}}"
    lines.append(f"* IC hint: simulate from V={initial_condition_v:g}")
    return "\n".join(lines)


def q_prev_of(pairs, i):
    """Charge of the previous table entry."""
    return pairs[i - 1][1]


# ---------------------------------------------------------------------------
# Measured gate charge (Curve 5 Qg(Vgs) + Curve 4 Crss(Vds))
# ---------------------------------------------------------------------------
# The UROM gate-side candidate is allowed to use *measured* charge only when
# the caller explicitly opts in.  These helpers are pure (no simulator, no
# writes): they validate the exported curve arrays and return tables that the
# Tier-2 emitter can place on ``C_ISS`` / ``C_RSS`` as ``Q=`` elements.
#
# Split rules:
# * Qgs(Vgs) = Qg(Vgs) below the Miller plateau and Qg(Vgs) - qgd_nc at/above
#   it.  Curve 5 repeats Vgs across the plateau, where Qg is multi-valued in
#   Vgs; only the *last* measured sample at each voltage is retained, so the
#   plateau collapses to one point whose split equals the reported qgs_nc.
# * Qgd(Vdg) = integral of curve-4 Crss over the measured Vds axis, with the
#   legacy identity Vdg = Vds.  No capacitance is invented below the first
#   measured Vds: Q(0)=0 and the unmeasured sub-domain carries zero charge.
_PLATEAU_ATOL_V = 1.0e-9
_CHARGE_ATOL_C = 1.0e-15
# Measured-Q emission budget: the same 0.5 % charge-fit budget as the UROM G6
# gate, with the documented greedy table cap.  Compaction is Q-space only:
# selected points are measured samples, never interpolated or invented.
_MEASURED_Q_REL_TOLERANCE = 0.005
_MEASURED_Q_MAX_POINTS = 64


@dataclass(frozen=True)
class MeasuredGateCharge:
    """Validated measured gate-charge tables and their provenance.

    ``qgs_v`` / ``qgs_c`` is Qgs(Vgs) in volts/coulombs; ``qgd_v`` / ``qgd_c``
    is Qgd(Vdg) in volts/coulombs under the legacy Vdg = Vds identity.  Both
    tables are ascending, zero-anchored, non-decreasing and Q-space compacted
    to the measured-samples-only 0.5 % charge budget.  ``provenance`` is
    JSON-safe and records every input, split decision, full measured point
    count and achieved compaction error.
    """

    qgs_v: np.ndarray
    qgs_c: np.ndarray
    qgd_v: np.ndarray
    qgd_c: np.ndarray
    provenance: dict


def _finite_vector(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"{name}: need >= 2 samples, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: contains NaN/Inf")
    return arr


def _require_non_decreasing(name: str, values: np.ndarray) -> None:
    diff = np.diff(values)
    bad = np.flatnonzero(diff < 0.0)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"{name} is not non-decreasing at index {i + 1} "
            f"({values[i]:.6g} -> {values[i + 1]:.6g})")


def _compact_measured_q(
    name: str,
    v: np.ndarray,
    q: np.ndarray,
    *,
    rel_tolerance: float = _MEASURED_Q_REL_TOLERANCE,
    max_points: int = _MEASURED_Q_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Greedily compact a measured Q(V) table in charge space.

    Selection is the same farthest-point rule used by
    :func:`pwl_charge_segments`: endpoints are retained, then the sample with
    the largest linear-interpolation error in Q is inserted until the error is
    within ``rel_tolerance`` x Q(Vmax) or the cap is reached.  Every retained
    point is a measured sample (index subset), so nothing is fabricated; the
    zero anchor is index 0 and is always retained.  A budget that cannot be
    met within ``max_points`` raises instead of emitting a half-conditioned
    table.
    """
    vv = np.asarray(v, dtype=float).ravel()
    qq = np.asarray(q, dtype=float).ravel()
    if vv.size != qq.size or vv.size < 2:
        raise ValueError(
            f"{name}: compaction needs >= 2 equal-length samples")
    if not (np.all(np.isfinite(vv)) and np.all(np.isfinite(qq))):
        raise ValueError(f"{name}: compaction input contains NaN/Inf")
    if not 0.0 < rel_tolerance < 1.0:
        raise ValueError(f"{name}: rel_tolerance must be in (0, 1)")
    cap = int(max_points)
    if cap < 2:
        raise ValueError(f"{name}: max_points must be >= 2")
    q_end = abs(float(qq[-1]))
    if q_end <= 0.0:
        raise ValueError(f"{name}: Q(Vmax) must be > 0 to size the budget")
    budget = float(rel_tolerance) * q_end

    idx = [0, vv.size - 1]
    while True:
        sel = np.asarray(idx, dtype=int)
        err = np.abs(qq - np.interp(vv, vv[sel], qq[sel]))
        err[sel] = 0.0
        worst = int(np.argmax(err))
        worst_err = float(err[worst])
        if worst_err <= budget:
            break
        if len(idx) >= cap:
            raise ValueError(
                f"{name}: cannot meet {100.0 * rel_tolerance:.3g}% charge "
                f"budget within {cap} points (worst error {worst_err:.6e} C "
                f"vs budget {budget:.6e} C)")
        idx = sorted(idx + [worst])

    sel = np.asarray(idx, dtype=int)
    achieved = float(np.max(
        np.abs(qq - np.interp(vv, vv[sel], qq[sel]))))
    stats = {
        "rule": ("greedy farthest-point on Q(V); endpoints and the zero "
                 "anchor retained; emitted points are measured samples"),
        "rel_tolerance_pct": 100.0 * float(rel_tolerance),
        "max_points": cap,
        "budget_c": budget,
        "max_error_c": achieved,
        "max_error_pct": 100.0 * achieved / q_end,
        "full_points": int(vv.size),
        "emitted_points": int(sel.size),
    }
    return vv[sel].copy(), qq[sel].copy(), stats


def _extracted(doc: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(doc, Mapping):
        raise ValueError(f"{name}: expected a mapping, got {type(doc).__name__}")
    payload = doc.get("extracted", doc)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name}: 'extracted' must be a mapping")
    return payload


def _select_qgd_curve(
    curve4: Mapping[str, Any], source_curve4: str
) -> tuple[Any, Any, dict]:
    """Pick the measured Crss curve and its Vdg axis.

    When ``cv_sweeps`` exists, prefer the largest nonzero ``|vgs_bias_v|``
    sweep and use the measured ``Vdg = Vds - vgs_bias_v`` axis (non-negative
    part only), matching ``pipeline.tier3_simplis._select_cdg_data``.  The
    legacy ``curve_vds``/``crss_curve_f`` single sweep (Vdg = Vds) is used
    when no multi-bias sweep is available, or when no nonzero bias yields at
    least two non-negative axis points.

    Returns ``(axis, crss, provenance)``; array validation happens later on
    the selected data so a bad selected sweep fails closed.
    """
    sweeps = curve4.get("cv_sweeps")
    if isinstance(sweeps, (list, tuple)) and sweeps:
        biased: list[tuple[float, Mapping[str, Any]]] = []
        for sweep in sweeps:
            if not isinstance(sweep, Mapping):
                raise ValueError(
                    f"{source_curve4}: cv_sweeps entries must be mappings")
            bias = sweep.get("vgs_bias_v")
            if bias is None:
                continue
            bias = float(bias)
            if not np.isfinite(bias):
                raise ValueError(
                    f"{source_curve4}: cv_sweeps vgs_bias_v must be finite")
            if bias != 0.0:
                biased.append((bias, sweep))
        for bias, sweep in sorted(biased, key=lambda item: -abs(item[0])):
            if ("curve_vds" not in sweep
                    or "crss_curve_f" not in sweep):
                raise ValueError(
                    f"{source_curve4}: cv_sweeps at vgs_bias_v={bias:g} "
                    f"missing curve_vds/crss_curve_f")
            vds = np.asarray(sweep["curve_vds"], dtype=float).ravel()
            crss = np.asarray(sweep["crss_curve_f"], dtype=float).ravel()
            if vds.size != crss.size:
                raise ValueError(
                    f"{source_curve4}: cv_sweeps at vgs_bias_v={bias:g} "
                    f"has mismatched curve_vds/crss_curve_f lengths")
            # np.unique keeps the first measured occurrence of each axis
            # point, the same rule as the peer C-V selector.
            axis, idx = np.unique(vds - bias, return_index=True)
            crss = crss[idx]
            keep = axis >= 0.0
            axis, crss = axis[keep], crss[keep]
            if axis.size >= 2:
                return axis, crss, {
                    "cv_bias_aware": True,
                    "vgs_bias_v": float(bias),
                    "axis": "Vdg = Vds - Vgs_bias (bias-aware Crss sweep)",
                    "source": (f"{source_curve4}:cv_sweeps"
                               f"[vgs_bias_v={bias:g}]"),
                    "cv_sweeps_count": len(sweeps),
                    "limitation": None,
                }

    if curve4.get("curve_vds") is None or curve4.get("crss_curve_f") is None:
        raise ValueError(
            f"{source_curve4}: no usable cv_sweeps bias and no legacy "
            f"curve_vds/crss_curve_f pair")
    return curve4.get("curve_vds"), curve4.get("crss_curve_f"), {
        "cv_bias_aware": False,
        "vgs_bias_v": None,
        "axis": "Vdg = Vds (legacy identity)",
        "source": f"{source_curve4}:curve_vds/crss_curve_f",
        "cv_sweeps_count": (len(sweeps)
                            if isinstance(sweeps, (list, tuple)) else 0),
        "limitation": ("no nonzero cv_sweeps bias with >= 2 non-negative "
                       "Vdg points; legacy single-sweep path retained"),
    }


def measured_gate_charge_from_curves(
    curve5: Mapping[str, Any],
    curve4: Mapping[str, Any],
    *,
    source_curve5: str = "curve5_gate_charge",
    source_curve4: str = "curve4_capacitances",
) -> MeasuredGateCharge:
    """Build zero-anchored monotone Qgs/Qgd tables from measured curves.

    Args:
        curve5: Loaded ``curve5_gate_charge`` document (or its ``extracted``
            mapping).  Must carry ``curve_vgs`` [V], ``curve_qg_nc`` [nC] and
            the breakpoints ``v_plateau_v`` [V], ``qgs_nc`` [nC],
            ``qgd_nc`` [nC], ``q_tail_nc`` [nC], ``qg_total_nc`` [nC].
        curve4: Loaded ``curve4_capacitances`` document.  Must carry
            ``curve_vds`` [V] and ``crss_curve_f`` [F].
        source_curve5/source_curve4: Human-readable provenance labels copied
            into the result; normally the YAML paths.

    Every failure is loud: absent/short arrays, NaN/Inf, a non-monotone
    measured axis or charge curve, a missing plateau sample, a negative
    gate-drain charge, a split that would create negative Qgs, or a
    compaction budget that cannot be met within the 64-point cap all raise
    ``ValueError``.  Emitted points are always measured samples; no
    capacitance or charge value is interpolated or fabricated.
    """
    c5 = _extracted(curve5, source_curve5)
    c4 = _extracted(curve4, source_curve4)

    for key in ("curve_vgs", "curve_qg_nc", "v_plateau_v", "qgs_nc",
                "qgd_nc", "q_tail_nc", "qg_total_nc"):
        if key not in c5:
            raise ValueError(f"{source_curve5}: missing required key {key!r}")
    has_legacy_cv = ("curve_vds" in c4 and "crss_curve_f" in c4)
    has_cv_sweeps = (isinstance(c4.get("cv_sweeps"), (list, tuple))
                     and bool(c4.get("cv_sweeps")))
    if not (has_legacy_cv or has_cv_sweeps):
        raise ValueError(
            f"{source_curve4}: missing curve_vds/crss_curve_f and no "
            f"cv_sweeps")

    vgs = _finite_vector(f"{source_curve5}:curve_vgs", c5["curve_vgs"])
    qg = _finite_vector(
        f"{source_curve5}:curve_qg_nc", c5["curve_qg_nc"]) * 1e-9
    if vgs.size != qg.size:
        raise ValueError(
            f"{source_curve5}: curve_vgs ({vgs.size}) and curve_qg_nc "
            f"({qg.size}) lengths differ")
    if abs(vgs[0]) > _PLATEAU_ATOL_V:
        raise ValueError(
            f"{source_curve5}: curve_vgs must start at 0 V for a "
            f"zero-anchored table (got {vgs[0]:.6g} V)")
    if abs(qg[0]) > _CHARGE_ATOL_C:
        raise ValueError(
            f"{source_curve5}: curve_qg_nc must start at 0 C "
            f"(got {qg[0]:.6g} C)")
    _require_non_decreasing(f"{source_curve5}:curve_vgs", vgs)
    _require_non_decreasing(f"{source_curve5}:curve_qg_nc", qg)

    v_plateau = float(c5["v_plateau_v"])
    qgs_breakpoint_nc = float(c5["qgs_nc"])
    qgd_nc = float(c5["qgd_nc"])
    q_tail_breakpoint_nc = float(c5["q_tail_nc"])
    qg_total_nc = float(c5["qg_total_nc"])
    breakpoints = (v_plateau, qgs_breakpoint_nc, qgd_nc,
                   q_tail_breakpoint_nc, qg_total_nc)
    if not all(np.isfinite(value) for value in breakpoints):
        raise ValueError(
            f"{source_curve5}: v_plateau_v/qgs_nc/qgd_nc/q_tail_nc/"
            f"qg_total_nc must be finite")
    qgd_split = qgd_nc * 1e-9
    if v_plateau <= 0.0:
        raise ValueError(
            f"{source_curve5}: v_plateau_v must be > 0 (got {v_plateau!r})")
    if min(qgs_breakpoint_nc, qgd_nc, q_tail_breakpoint_nc,
           qg_total_nc) < 0.0:
        raise ValueError(
            f"{source_curve5}: charge breakpoints must be >= 0")
    if qg_total_nc <= 0.0:
        raise ValueError(
            f"{source_curve5}: qg_total_nc must be > 0 (got {qg_total_nc!r})")

    plateau = np.flatnonzero(np.abs(vgs - v_plateau) <= _PLATEAU_ATOL_V)
    if plateau.size == 0:
        raise ValueError(
            f"{source_curve5}: no curve_vgs sample at the reported plateau "
            f"{v_plateau:.6g} V; cannot split Qgs from Qgd")

    # Collapse repeated Vgs samples to the last measured point at each
    # voltage.  Curve 5 repeats Vgs across the Miller plateau; the last
    # sample is the one that has accumulated the full qgd_nc, so subtracting
    # it yields the reported qgs_nc instead of a negative intermediate.
    rev_v, rev_idx = np.unique(vgs[::-1], return_index=True)
    last_idx = vgs.size - 1 - rev_idx
    uniq_v = rev_v
    q_at_v = qg[last_idx]
    _require_non_decreasing(f"{source_curve5}:unique Vgs", uniq_v)

    at_or_above = uniq_v >= v_plateau - _PLATEAU_ATOL_V
    qgs_c = q_at_v.copy()
    qgs_c[at_or_above] = q_at_v[at_or_above] - qgd_split
    negative = np.flatnonzero(qgs_c < -_CHARGE_ATOL_C)
    if negative.size:
        i = int(negative[0])
        raise ValueError(
            f"{source_curve5}: Qgs split would be negative at Vgs="
            f"{uniq_v[i]:.6g} V (Qg={q_at_v[i]:.6e} C, "
            f"qgd_nc={qgd_nc:.6g} nC)")
    _require_non_decreasing(f"{source_curve5}:derived Qgs", qgs_c)
    if abs(qgs_c[0]) > _CHARGE_ATOL_C:
        raise ValueError(
            f"{source_curve5}: derived Qgs is not zero-anchored "
            f"(Qgs(0)={qgs_c[0]:.6e} C)")

    qgd_v_raw, qgd_c_raw, qgd_selection = _select_qgd_curve(
        c4, source_curve4)
    vds = _finite_vector(
        f"{qgd_selection['source']}:Vdg", qgd_v_raw)
    crss = _finite_vector(
        f"{qgd_selection['source']}:Crss", qgd_c_raw)
    if vds.size != crss.size:
        raise ValueError(
            f"{qgd_selection['source']}: Vdg ({vds.size}) and Crss "
            f"({crss.size}) lengths differ")
    if vds[0] < 0.0:
        raise ValueError(
            f"{qgd_selection['source']}: Vdg axis must be >= 0 "
            f"(got {vds[0]:.6g} V)")
    _require_non_decreasing(f"{qgd_selection['source']}:Vdg", vds)
    if np.any(crss < 0.0):
        raise ValueError(f"{source_curve4}: crss_curve_f contains negative F")

    # Legacy identity Vdg = Vds: integrate Crss along the measured Vds axis.
    qgd_meas = cumulative_trapezoid(crss, vds, initial=0.0)
    if vds[0] > 0.0:
        # Anchor the table at V=0 without inventing a capacitance sample:
        # the unmeasured sub-domain contributes zero charge by construction.
        qgd_v = np.concatenate(([0.0], vds))
        qgd_c = np.concatenate(([0.0], qgd_meas))
    else:
        qgd_v, qgd_c = vds, qgd_meas
    _require_non_decreasing(f"{source_curve4}:derived Qgd", qgd_c)
    if abs(qgd_c[0]) > _CHARGE_ATOL_C:
        raise ValueError(
            f"{source_curve4}: derived Qgd is not zero-anchored "
            f"(Qgd(0)={qgd_c[0]:.6e} C)")

    # Measured full tables are validated and then compacted in Q space for
    # emission: 399-point Qgs tables make every sample a PWL breakpoint and
    # chatter the timestep.  Selection is an index subset of measured samples
    # (endpoints + zero anchor retained) within the 0.5 % charge budget.
    qgs_v_full = uniq_v
    qgs_c_full = qgs_c
    qgd_v_full = np.asarray(qgd_v, dtype=float)
    qgd_c_full = np.asarray(qgd_c, dtype=float)
    qgs_v, qgs_c, qgs_compaction = _compact_measured_q(
        f"{source_curve5}:Qgs", qgs_v_full, qgs_c_full)
    qgd_v, qgd_c, qgd_compaction = _compact_measured_q(
        f"{source_curve4}:Qgd", qgd_v_full, qgd_c_full)

    provenance = {
        "kind": "measured",
        "source": (f"{source_curve5} curve_vgs/curve_qg_nc + "
                   f"{source_curve4} curve_vds/crss_curve_f"),
        "source_curve5": str(source_curve5),
        "source_curve4": str(source_curve4),
        "v_plateau_v": v_plateau,
        "qgs_breakpoint_nc": qgs_breakpoint_nc,
        "qgd_breakpoint_nc": qgd_nc,
        "q_tail_breakpoint_nc": q_tail_breakpoint_nc,
        "qg_total_nc": qg_total_nc,
        "plateau_samples": int(plateau.size),
        "plateau_collapsed_to_last_qg": True,
        "qgs_points": int(qgs_v_full.size),
        "qgd_points": int(qgd_v_full.size),
        "qgs_emitted_points": int(qgs_v.size),
        "qgd_emitted_points": int(qgd_v.size),
        "qgs_compaction": qgs_compaction,
        "qgd_compaction": qgd_compaction,
        "qgs_zero_anchor_c": float(qgs_c[0]),
        "qgd_zero_anchor_c": float(qgd_c[0]),
        "qgs_rule": ("Qgs=Qg below the plateau; Qgs=Qg-qgd_nc at/above it, "
                     "with repeated plateau Vgs collapsed to the last "
                     "measured Qg"),
        "qgd_axis": qgd_selection["axis"],
        "qgd_cv_bias_aware": qgd_selection["cv_bias_aware"],
        "qgd_vgs_bias_v": qgd_selection["vgs_bias_v"],
        "qgd_selection": qgd_selection,
        "qgd_rule": ("cumulative trapezoid of the selected measured Crss "
                     "curve on its Vdg axis; Qgd(0)=0 and no capacitance is "
                     "invented below the first measured Vdg"),
    }
    return MeasuredGateCharge(
        qgs_v=qgs_v.astype(float, copy=True),
        qgs_c=qgs_c.astype(float, copy=True),
        qgd_v=qgd_v.astype(float, copy=True),
        qgd_c=qgd_c.astype(float, copy=True),
        provenance=provenance,
    )


def load_measured_gate_charge(vendor: str) -> MeasuredGateCharge:
    """Load and validate a vendor's measured Curve-5/Curve-4 gate charge.

    Reads ``config/outputs/<vendor>/curve5_gate_charge.yaml`` and
    ``config/outputs/<vendor>/curve4_capacitances.yaml`` through the canonical
    YAML loader.  Missing files or fields raise; there is no heuristic
    fallback here - the caller decides whether to opt in.
    """
    from config.yaml_io import load_output

    source5 = f"config/outputs/{vendor}/curve5_gate_charge.yaml"
    source4 = f"config/outputs/{vendor}/curve4_capacitances.yaml"
    curve5 = load_output(vendor, "curve5_gate_charge")
    curve4 = load_output(vendor, "curve4_capacitances")
    return measured_gate_charge_from_curves(
        curve5, curve4, source_curve5=source5, source_curve4=source4)
