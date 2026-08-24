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

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
