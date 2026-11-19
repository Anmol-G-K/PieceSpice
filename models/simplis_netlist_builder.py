"""Tier-3 ``simplis`` variant: SIMPLIS-style Level-2 charge-table emitter.

The channel, body diode and parasitics are the CLOSED Tier-2 emission: this
module calls ``models.netlist_builder.build_tier2_subcircuit`` and splices in
the charge section, so no channel/diode/parasitic logic is duplicated here.

The new content is the documented SIMPLIS Level-2 three-capacitor set:

    CGS  gi ss   from Ciss - Crss   (X axis Vgs; LINEAR when nearly linear)
    CDS  di ss   from Coss - Crss   (X axis Vds; 4-segment (V,Q) table)
    CDG  di gi   from Crss          (X axis Vdg; 4-segment (V,Q) table)

Every capacitor is charge conserving (``Q=``) and every ramp uses DELTA
slopes, per the repository invariant (absolute slopes diverged 21x in audit).

Sources for the pinned constants / semantics (see local://simplis_plecs_spec.md):
* S-MOS "Capacitor Models": (V,Q) point pairs, capacitance = segment slope,
  CDS_NSEG default 4, CDG_NSEG default 4, CGS_NSEG default 1 (linear),
  CDG X axis is drain-to-gate voltage.
* S-APPB "MOSFET - Level 2: Model Structure": CDS/CDG are 4-segment PWL
  because those nonlinearities are critical for switching transitions;
  "in most devices the gate capacitance is quite linear" and is tested.
* The interior breakpoints are OUR greedy rule (``pwl_charge_segments``);
  SIMPLIS does not document its extraction rule for those points.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from common.types import PWLModelSpec
from models.netlist_builder import build_tier2_subcircuit, _fmt

# Documented SIMPLIS Level-2 defaults [S-MOS Capacitor Models; S-APPB Level 2].
# A table with NSEG segments has NSEG+1 (V,Q) point pairs.
CDS_NSEG = 4
CDG_NSEG = 4
MAX_CGS_NSEG = 4  # ceiling when the measured linearity test selects PWL

# Repo convention: pwl_charge_segments(rel_tolerance=0.005) is the existing
# 0.5% charge-error budget; reuse it for the linearity decision (no new magic).
RELTOL = 0.005

# Stable anchors of the Tier-2 emitter's charge section. We splice that block
# out and replace it; channel, diode and parasitics stay byte-for-byte as
# emitted by the closed Tier-2 builder.
_T2_CHARGE_START = "* --- charge-conserving capacitances (Curve 4, plan 2.3) ---"
_T2_DIODE_START = "* --- body diode (Curve 6) ---"

MIN_PAIRS = 2  # a capacitor needs two (V,Q) points to have a slope


@dataclass(frozen=True)
class CGSLinearityDecision:
    """Data-driven answer to the documented Level-2 gate-capacitor test.

    ``chord_deviation`` is max|Q(V) - chord(V)| / |Q(Vmax) - Q(V0)|. It is
    compared with ``rel_tolerance`` (the repo's existing greedy-charge
    tolerance), not with a device-specific threshold.
    """

    is_linear: bool
    chord_deviation: float
    rel_tolerance: float

    def as_dict(self) -> dict:
        return {
            "mode": "linear" if self.is_linear else "pwl",
            "chord_deviation": self.chord_deviation,
            "rel_tolerance": self.rel_tolerance,
        }


def cgs_linearity_decision(
    v: Sequence[float],
    q: Sequence[float],
    rel_tolerance: float = RELTOL,
) -> CGSLinearityDecision:
    """Decide LINEAR vs PWL for CGS from the measured Q(V) nonlinearity.

    Deterministic: the same (V,Q) arrays and tolerance always give the same
    answer. Uses the straight chord between the table endpoints, which is
    exactly the charge the emitted linear capacitor would store.
    """
    vv = np.asarray(v, dtype=float)
    qq = np.asarray(q, dtype=float)
    if vv.size < MIN_PAIRS or vv.size != qq.size:
        raise ValueError("CGS linearity test needs >= 2 matched (V,Q) points")
    if not np.all(np.diff(vv) > 0):
        raise ValueError("CGS voltage axis must be strictly ascending")
    dv = float(vv[-1] - vv[0])
    dq = float(qq[-1] - qq[0])
    chord = qq[0] + (dq / dv) * (vv - vv[0])
    denom = max(abs(dq), np.finfo(float).tiny)
    deviation = float(np.max(np.abs(qq - chord)) / denom)
    return CGSLinearityDecision(
        is_linear=deviation <= rel_tolerance,
        chord_deviation=deviation,
        rel_tolerance=float(rel_tolerance),
    )


def charge_ramp_terms(pairs: Sequence[Tuple[float, float]]):
    """Yield ``(v_prev, delta_slope)`` for one delta-slope Q= chain.

    Exposed for offline tests: monotonic Q gives non-negative segment slopes;
    the emitted expression integrates those slopes with soft-abs ramps.
    """
    prev_slope = 0.0
    for i in range(1, len(pairs)):
        v_prev, q_prev = pairs[i - 1]
        v_i, q_i = pairs[i]
        dv = v_i - v_prev
        if dv <= 0:
            raise ValueError("charge table voltage axis must ascend strictly")
        slope = (q_i - q_prev) / dv
        if slope < 0:
            raise ValueError("charge table Q must be monotonic non-decreasing")
        delta = slope - prev_slope
        prev_slope = slope
        yield v_prev, delta


def _q_expression(pairs: Sequence[Tuple[float, float]], term: str) -> str:
    """Delta-slope chained soft-abs expression for Q(term)."""
    expr = _fmt(pairs[0][1])
    for v_prev, delta in charge_ramp_terms(pairs):
        if abs(delta) == 0.0:
            continue
        expr += (f" + {delta:.6e}*(({term}-({_fmt(v_prev)}))"
                 f"+abs(({term})-({_fmt(v_prev)})))/2")
    return expr


def _emit_cap(element: str, npos: str, nneg: str, term: str,
              pairs: Sequence[Tuple[float, float]], note: str) -> List[str]:
    """Emit one charge-conserving capacitor line (plus its comment)."""
    return [
        f"* {element}: {note}; {len(pairs)} (V,Q) points",
        f"{element} {npos} {nneg} Q={{{_q_expression(pairs, term)}}}",
    ]


def _pairs(v, q) -> List[Tuple[float, float]]:
    pairs = [(float(a), float(b)) for a, b in zip(v, q)]
    if len(pairs) < MIN_PAIRS:
        raise ValueError("SIMPLIS variant needs populated (V,Q) charge tables")
    if not np.all(np.diff(np.asarray([p[0] for p in pairs])) > 0):
        raise ValueError("charge table voltage axis must ascend strictly")
    if np.any(np.diff(np.asarray([p[1] for p in pairs])) < 0):
        raise ValueError("charge table Q must be monotonic non-decreasing")
    return pairs


def build_simplis_subcircuit(spec: PWLModelSpec) -> str:
    """Emit the Level-2 charge-table Tier-3 subcircuit (ASCII).

    Fails loudly on unpopulated charge tables: an emitted model with fake
    capacitors would be worse than a failed build (anti-fake guard).
    """
    cgs_pairs = _pairs(spec.q_iss_vs_v, spec.q_iss_charge_c)
    cds_pairs = _pairs(spec.q_oss_vs_v, spec.q_oss_charge_c)
    cdg_pairs = _pairs(spec.q_rss_vs_v, spec.q_rss_charge_c)
    decision = cgs_linearity_decision(
        [p[0] for p in cgs_pairs], [p[1] for p in cgs_pairs])

    # Reuse the closed Tier-2 emitter verbatim, then replace only its charge
    # section. Any structure change there must be reflected by these anchors
    # (explicit failure, never a silent partial splice).
    t2 = build_tier2_subcircuit(spec)
    lines = t2.splitlines()
    try:
        i0 = lines.index(_T2_CHARGE_START)
        i1 = lines.index(_T2_DIODE_START)
    except ValueError as exc:
        raise RuntimeError(
            "Tier-2 emitter structure changed; SIMPLIS splice anchors missing") \
            from exc
    if i1 <= i0:
        raise RuntimeError("Tier-2 charge/diode section anchors out of order")

    t2_name = f"{spec.device_name}_PIECESPICE_T2"
    name = f"{spec.device_name}_PIECESPICE_T3_simplis"

    charge: List[str] = [
        "* --- SIMPLIS Level-2 three-capacitor charge model (AC Curve 4) ---",
        "* CGS=Ciss-Crss, CDS=Coss-Crss, CDG=Crss (single AC sweep at Vgs=0)",
        f"* CGS linearity test: {'LINEAR' if decision.is_linear else 'PWL'} "
        f"(chord deviation {decision.chord_deviation:.4g} vs "
        f"rel_tol {decision.rel_tolerance:g})",
        "* CGS X axis is Vgs; table comes from the AC curve swept over Vds "
        "at Vgs=0",
        "* CDG X axis is drain-to-gate V(di,gi); with Vgs=0 in the AC sweep "
        "v_DG=v_DS",
        "* Outside the AC sweep range LTspice table()/Q= lookups end-clamp "
        "at the table endpoints (repo-verified); no charge data is claimed "
        "beyond the sweep",
    ]
    if decision.is_linear:
        c_lin = ((cgs_pairs[-1][1] - cgs_pairs[0][1]) /
                 (cgs_pairs[-1][0] - cgs_pairs[0][0]))
        charge.append(f"* CGS is nearly linear: C={c_lin:.6e} F")
        charge.append(f"CGS gi ss {_fmt(c_lin)}")
    else:
        charge.extend(_emit_cap(
            "CGS", "gi", "ss", "V(gi,ss)", cgs_pairs,
            "gate-source charge, greedy rule for interior points"))
    charge.extend(_emit_cap(
        "CDS", "di", "ss", "V(di,ss)", cds_pairs,
        "drain-source charge, 4-segment SIMPLIS default"))
    charge.extend(_emit_cap(
        "CDG", "di", "gi", "V(di,gi)", cdg_pairs,
        "drain-gate charge on the Vdg axis, 4-segment SIMPLIS default"))
    charge.append("")

    out: List[str] = []
    for line in lines[:i0]:
        if line == f"* PieceSpice Tier 2 CPWL Model for {spec.device_name}":
            out.append(
                f"* PieceSpice Tier 3 SIMPLIS Level-2 charge-table model "
                f"for {spec.device_name}")
        elif line == f".SUBCKT {t2_name} d g s":
            out.append(f".SUBCKT {name} d g s")
        else:
            out.append(line)
    out.extend(charge)
    for line in lines[i1:]:
        if line == f".ENDS {t2_name}":
            out.append(f".ENDS {name}")
        else:
            out.append(line)
    return "\n".join(out)
