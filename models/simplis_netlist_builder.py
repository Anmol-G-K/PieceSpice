"""Tier-3 ``simplis`` variant: SIMPLIS-style Level-2 charge-table emitter.

The channel, body diode and parasitics are the CLOSED Tier-2 emission: this
module calls ``models.netlist_builder.build_tier2_subcircuit`` and splices in
the charge section, so no channel/diode/parasitic logic is duplicated here.

The new content is the documented SIMPLIS Level-2 three-capacitor set:

    CGS  gi ss   from Ciss - Crss   (X axis Vgs; LINEAR when nearly linear)
    CDS  di ss   from Coss - Crss   (X axis Vds; measured PWL V-Q table)
    CDG  di gi   from Crss          (X axis Vdg; measured PWL V-Q table)

Every capacitor is charge conserving (``Q=``) and every ramp uses DELTA
slopes, per the repository invariant (absolute slopes diverged 21x in audit).

Sources for the pinned constants / semantics (see local://simplis_plecs_spec.md):
* S-MOS "Capacitor Models": (V,Q) point pairs, capacitance = segment slope,
  CDS_NSEG default 4, CDG_NSEG default 4, CGS_NSEG default 1 (linear),
  CDG X axis is drain-to-gate voltage.
* S-APPB "MOSFET - Level 2: Model Structure": CDS/CDG are 4-segment PWL
  because those nonlinearities are critical for switching transitions;
  "in most devices the gate capacitance is quite linear" and is tested.
* Interior breakpoints are OUR greedy rule (``pwl_charge_segments`` in the
  producer); SIMPLIS does not document its extraction rule. The producer
  fits until the repo's 0.5% charge budget is met, not to a segment cap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence, Tuple

import numpy as np

from common.types import PWLModelSpec
from models.netlist_builder import build_tier2_subcircuit, _fmt
from models.ann_netlist_builder import emit_ann_residual

if TYPE_CHECKING:
    from approximation.ann_residual import AnnResidualFit
    from models.hybrid_channel import HybridChannelFit

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

# Official documented Level-2 gate-source bleed: 10 MOhm [S-MOS, Level 0/2
# Model tables; local://simplis_plecs_spec.md line 103]. It is a resistor,
# not a fitted value, and is emitted in both charge candidates.
RGS_LINE = "RGS gi ss 1e7"


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
    v: Sequence[float] | np.ndarray,
    q: Sequence[float] | np.ndarray,
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


def _q_expression(pairs: Sequence[Tuple[float, float]], term: str,
                  clamp_above_last: bool = False) -> str:
    """Delta-slope chained soft-abs expression for Q(term).

    ``clamp_above_last=True`` appends one final delta slope ``-last_slope``
    at the last breakpoint, so the emitted Q is constant above the measured
    range (a physical end segment) instead of extrapolating the last slope.
    """
    expr = _fmt(pairs[0][1])
    last_slope = 0.0
    for v_prev, delta in charge_ramp_terms(pairs):
        last_slope += delta
        if abs(delta) == 0.0:
            continue
        expr += (f" + {delta:.6e}*(({term}-({_fmt(v_prev)}))"
                 f"+abs(({term})-({_fmt(v_prev)})))/2")
    if clamp_above_last and last_slope != 0.0:
        v_last = _fmt(pairs[-1][0])
        expr += (f" + {-last_slope:.6e}*(({term}-({v_last}))"
                 f"+abs(({term})-({v_last})))/2")
    return expr


def _emit_cap(element: str, npos: str, nneg: str, term: str,
              pairs: Sequence[Tuple[float, float]], note: str,
              *, clamp_above_last: bool = False) -> List[str]:
    """Emit one charge-conserving capacitor line (plus its comment)."""
    return [
        f"* {element}: {note}; {len(pairs)} (V,Q) points",
        f"{element} {npos} {nneg} Q="
        f"{{{_q_expression(pairs, term, clamp_above_last)}}}",
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


def _charge_meta_lines(meta: dict) -> List[str]:
    """Header lines for C-V axis / gate-drive / temperature provenance.

    Only emitted when the caller supplies provenance. Missing values are
    reported as ``not_measured``; no axis or drive value is invented.
    """
    if not meta:
        return []

    def _num(value):
        if value is None:
            return "not_measured"
        return f"{float(value):g}"

    cdg_axis = meta.get("cdg_axis") or "not_measured"
    if not isinstance(cdg_axis, str) or not cdg_axis.isascii():
        raise ValueError("charge_meta['cdg_axis'] must be an ASCII string")
    cap_error = meta.get("max_cap_error_pct")
    budget = meta.get("cap_error_budget_pct")
    lines = [
        f"* C-V provenance: cdg_axis={cdg_axis}",
        f"* C-V gate drive: vgs_bias_v={_num(meta.get('vgs_bias_v'))} V; "
        f"temperature_c={_num(meta.get('temperature_c'))} C",
        "* C-V measured error: max_cap_error_pct="
        + ("not_measured" if cap_error is None else f"{float(cap_error):.4g}")
        + "; budget_pct="
        + ("not_measured" if budget is None else f"{float(budget):g}"),
    ]
    limitation = meta.get("data_limitation")
    if limitation:
        if not isinstance(limitation, str) or not limitation.isascii():
            raise ValueError(
                "charge_meta['data_limitation'] must be an ASCII string")
        lines.append(f"* C-V data limitation: {limitation}")
    if not all(line.isascii() for line in lines):
        raise ValueError("charge metadata header is not ASCII")
    return lines


def build_simplis_subcircuit(
    spec: PWLModelSpec, charge_mode: str = "tables", *,
    channel_fit: "HybridChannelFit | None" = None,
    residual_fit: "AnnResidualFit | None" = None,
    cgs_axis: str = "vds",
    name: str | None = None,
    subckt_name: str | None = None,
    clamp_charge_above_last: bool = False,
    charge_meta: dict | None = None,
    temperature_c: float | None = None,
    vgs_bias_v: float | None = None,
) -> str:
    """Emit the Level-2 charge-table Tier-3 subcircuit (ASCII).

    ``charge_mode="tables"`` emits the CGS/CDS/CDG V-Q tables measured by
    the pipeline. ``charge_mode="tier2"`` keeps the closed Tier-2 charge
    block verbatim; channel/diode/parasitics are identical in both and the
    caller measures both candidates before selecting one.

    ``cgs_axis`` declares the provenance of ``q_iss_*``: ``"vds"`` is the
    one Vds-swept AC curve at Vgs=0 used by this pipeline (the default),
    ``"vgs"`` is a genuine gate-voltage sweep. A PWL CGS table is rejected
    on the Vds axis because the emitter would evaluate it on V(gi,ss);
    the LINEAR chord is axis-independent and is unchanged.

    ``channel_fit`` is an optional measured ``HybridChannelFit``: it replaces
    ONLY the single Tier-2 ``G_CH`` line with the fit's ASCII VALUE expression.
    Charge, diode and parasitics stay byte-identical; the emitted header
    records the fit provenance and its measured DPT limitation (a static fit
    gain does not imply transient fidelity).

    ``residual_fit`` is an optional measured ``AnnResidualFit``. It adds ONE
    ``G_ANN di ss VALUE = {I_res}`` line immediately after the existing
    ``G_CH`` line. The C0 channel line, both charge candidates, body diode
    and parasitics stay byte-identical; no fourth node is introduced and no
    ANN charge source replaces the C0 charge tables. ``name`` and
    ``subckt_name`` follow the existing override convention (the latter
    wins), defaulting to the documented simplis subcircuit name.

    ``clamp_charge_above_last`` (default False) appends a physical end
    segment to every emitted ``Q=`` chain so Q stops increasing above the
    last measured breakpoint instead of extrapolating the last slope. The
    default keeps the historical text byte-for-byte.

    ``charge_meta`` (or the explicit ``temperature_c`` / ``vgs_bias_v``)
    adds truthful C-V provenance header lines: the CDG axis, the measured
    gate drive/temperature, the measured C-V error and any data limitation.
    Missing values print ``not_measured``; nothing is invented. With no
    metadata the emitted header is byte-for-byte the historical one.

    Fails loudly on unpopulated charge tables in tables mode: an emitted
    model with fake capacitors would be worse than a failed build
    (anti-fake guard).
    """
    if charge_mode not in ("tables", "tier2"):
        raise ValueError("charge_mode must be 'tables' or 'tier2'")
    if cgs_axis not in ("vds", "vgs"):
        raise ValueError("cgs_axis must be 'vds' or 'vgs'")
    if charge_meta is not None and not isinstance(charge_meta, dict):
        raise ValueError("charge_meta must be a dict or None")
    meta = dict(charge_meta or {})
    if temperature_c is not None:
        meta["temperature_c"] = temperature_c
    if vgs_bias_v is not None:
        meta["vgs_bias_v"] = vgs_bias_v
    provenance = _charge_meta_lines(meta)
    residual_expression = ""
    if residual_fit is not None:
        residual_expression = emit_ann_residual(residual_fit)
        if not isinstance(residual_expression, str) \
                or not residual_expression.isascii():
            raise ValueError("residual expression must be non-empty ASCII")
    if charge_mode == "tables":
        cgs_pairs = _pairs(spec.q_iss_vs_v, spec.q_iss_charge_c)
        cds_pairs = _pairs(spec.q_oss_vs_v, spec.q_oss_charge_c)
        cdg_pairs = _pairs(spec.q_rss_vs_v, spec.q_rss_charge_c)
        decision = cgs_linearity_decision(
            [p[0] for p in cgs_pairs], [p[1] for p in cgs_pairs])
        if not decision.is_linear and cgs_axis != "vgs":
            raise ValueError(
                "PWL CGS on the Vds->Vgs axis mismatch: q_iss_* comes from "
                "the Vds-swept AC curve at Vgs=0 but would be emitted on "
                "V(gi,ss); supply a Vgs-axis table with cgs_axis='vgs' or "
                "keep the LINEAR chord")
    else:
        cgs_pairs = cds_pairs = cdg_pairs = None
        decision = None
    if charge_mode == "tables" and (
            decision is None or cgs_pairs is None
            or cds_pairs is None or cdg_pairs is None):
        raise RuntimeError("tables mode requires CGS/CDS/CDG charge tables")

    # Name overrides follow the existing Tier-2 convention: build the source
    # under the requested name (so the D_BD model suffix tracks it) and map
    # .SUBCKT/.ENDS to the simplis name. With no override this is byte-for-
    # byte the original T2 -> T3_simplis splice.
    override = subckt_name or name
    t2 = build_tier2_subcircuit(spec, name=override)
    t2_name = override or f"{spec.device_name}_PIECESPICE_T2"
    final_name = (subckt_name or name
                  or f"{spec.device_name}_PIECESPICE_T3_simplis")
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

    if charge_mode == "tables":
        assert decision is not None
        assert cgs_pairs is not None
        assert cds_pairs is not None
        assert cdg_pairs is not None
        charge: List[str] = [
            "* --- SIMPLIS Level-2 three-capacitor charge model (AC Curve 4) ---",
            "* CGS=Ciss-Crss, CDS=Coss-Crss, CDG=Crss (single AC sweep at Vgs=0)",
            f"* CGS linearity test: {'LINEAR' if decision.is_linear else 'PWL'} "
            f"(chord deviation {decision.chord_deviation:.4g} vs "
            f"rel_tol {decision.rel_tolerance:g})",
            "* CGS X axis is Vgs; the LINEAR chord from the Vds AC sweep is "
            "axis-independent",
            "* PWL CGS emitted only with a declared Vgs-axis table "
            "(cgs_axis='vgs')",
            "* CDG X axis is drain-to-gate V(di,gi); with Vgs=0 in the AC sweep "
            "v_DG=v_DS",
        ]
        if clamp_charge_above_last:
            charge.append(
                "* Q= chains: constant (C=0) below the first breakpoint; "
                "above the last breakpoint a physical end segment clamps the "
                "slope to zero so Q stays constant (no table() end-clamp)")
        else:
            charge.append(
                "* Q= chains: constant (C=0) below the first breakpoint; above the "
                "last breakpoint they extrapolate linearly with the last segment "
                "slope (no table() end-clamp)")
        charge.extend(provenance)
        if decision.is_linear:
            c_lin = ((cgs_pairs[-1][1] - cgs_pairs[0][1]) /
                     (cgs_pairs[-1][0] - cgs_pairs[0][0]))
            charge.append(f"* CGS is nearly linear: C={c_lin:.6e} F")
            charge.append(f"CGS gi ss {_fmt(c_lin)}")
        else:
            charge.extend(_emit_cap(
                "CGS", "gi", "ss", "V(gi,ss)", cgs_pairs,
                "gate-source charge, greedy rule for interior points",
                clamp_above_last=clamp_charge_above_last))
        charge.extend(_emit_cap(
            "CDS", "di", "ss", "V(di,ss)", cds_pairs,
            "drain-source charge, measured breakpoints",
            clamp_above_last=clamp_charge_above_last))
        charge.extend(_emit_cap(
            "CDG", "di", "gi", "V(di,gi)", cdg_pairs,
            "drain-gate charge on the Vdg axis, measured breakpoints",
            clamp_above_last=clamp_charge_above_last))
        charge.append("* RGS: documented Level-2 gate-source bleed (10 MOhm)")
        charge.append(RGS_LINE)
        charge.append("")
    else:
        # Keep the closed Tier-2 charge block byte-for-byte, adding only the
        # documented Level-2 RGS. This is the measured comparison candidate.
        charge = list(lines[i0:i1])
        j = next(i for i, line in enumerate(charge)
                 if line.startswith("C_RSS "))
        charge[j + 1:j + 1] = [
            "* RGS: documented Level-2 gate-source bleed (10 MOhm)",
            RGS_LINE,
        ]

    # Optional hybrid channel. Replace ONLY the Tier-2 G_CH line; every other
    # line (charge, diode, parasitics) must remain byte-identical.
    hybrid_note: List[str] = []
    if channel_fit is not None:
        expression = channel_fit.expression
        if not isinstance(expression, str) or not expression or not expression.isascii():
            raise ValueError("channel_fit.expression must be a non-empty ASCII string")
        meta = getattr(channel_fit, "meta", {})
        note = [
            "* Hybrid channel: measured hinge_c fit; only the Tier-2 G_CH line is replaced",
            f"* fit meta: kind={meta.get('kind')}, n_knees={meta.get('n_knees')}, "
            f"rel_tolerance={meta.get('rel_tolerance')}, "
            f"achieved_rel_rmse={meta.get('achieved_rel_rmse')}, "
            f"max_abs_fit_error_a={getattr(channel_fit, 'max_abs_fit_error_a', None)}",
            f"* fit stop criterion: {meta.get('stop_criterion')}",
            f"* measured limitation: {meta.get('known_limitation')}",
        ]
        if not all(line.isascii() for line in note):
            raise ValueError("hybrid channel fit meta is not ASCII")
        hybrid_note = note

    # Optional ANN residual. Every pre-existing line is preserved; the one
    # new source is inserted directly below G_CH and uses only di/ss.
    residual_note: List[str] = []
    if residual_fit is not None:
        residual_note = [
            "* ANN residual channel: signed zero-anchored correction; "
            "C0 charge unchanged",
            f"* ANN residual: {residual_fit.neurons} neurons, "
            f"{residual_fit.activation}, v_th={_fmt(residual_fit.v_th)} V",
        ]
        if not all(line.isascii() for line in residual_note):
            raise ValueError("ANN residual fit metadata is not ASCII")

    gch_seen = 0
    out: List[str] = []
    for line in lines[:i0]:
        if line == f"* PieceSpice Tier 2 CPWL Model for {spec.device_name}":
            out.append(
                f"* PieceSpice Tier 3 SIMPLIS Level-2 charge-table model "
                f"for {spec.device_name}")
            out.extend(hybrid_note + residual_note)
        elif line == f".SUBCKT {t2_name} d g s":
            out.append(f".SUBCKT {final_name} d g s")
        elif line.startswith("G_CH "):
            gch_seen += 1
            out.append(f"G_CH di ss VALUE={channel_fit.expression}"
                       if channel_fit is not None else line)
            if residual_fit is not None:
                out.append(f"G_ANN di ss VALUE = {{{residual_expression}}}")
        else:
            out.append(line)
    if gch_seen != 1:
        raise RuntimeError(
            f"Tier-2 emitter has {gch_seen} G_CH lines; expected exactly one")
    out.extend(charge)
    for line in lines[i1:]:
        if line == f".ENDS {t2_name}":
            out.append(f".ENDS {final_name}")
        else:
            out.append(line)
    return "\n".join(out)
