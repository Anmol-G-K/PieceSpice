"""ANN unified-ROM emitter -- task T2 (ASCII, zero new internal nodes).

``emit_ann_channel`` unrolls the fitted one-hidden-layer net into a single
scalar B-expression: VCCS current with the VDS*exp(y) transform, analytic tanh
Vth cutoff and a smooth low-VDS linear guard.  Only the verified emitted
primitive set is used (``tanh``, ``exp``, ``sqrt``/``abs`` algebra and
``Q=`` charge elements); the unverified ``hypot`` primitive is not emitted.

``build_ann_subcircuit`` reuses the *closed* Tier-2 builder verbatim for
parasitics (RG/RD/RS), the D_BODY model, semantic pin order and the three
two-terminal charge element placements.  Only the G_CH expression and the
three Q= expressions change; no fourth node is introduced.  eps defaults to
0 (legacy C0 chain) and is passed to the shared ``delta_ramp_chain``.
"""

from __future__ import annotations

import math
from typing import List

from approximation.ann_channel import AnnChannelFit
from approximation.ann_charge import AnnChargeFit
from approximation.smoother import delta_ramp_chain
from models.netlist_builder import build_tier2_subcircuit

__all__ = ["emit_ann_channel", "build_ann_subcircuit"]

_ACT = ("isru", "tanh", "sigmoid")
_TERMS = {"oss": "V(di,ss)", "iss": "V(gi,ss)", "rss": "V(gi,di)"}
_ELEMENTS = {
    "oss": ("C_QOSS", "di ss"),
    "iss": ("C_ISS", "gi ss"),
    "rss": ("C_RSS", "gi di"),
}


def _fmt(x: float) -> str:
    return f"{float(x):.6e}"


def _act_expr(u: str, activation: str) -> str:
    """ASCII activation; ISRU uses the verified sqrt form, not hypot."""
    if activation == "isru":
        return f"(({u})/sqrt(1+({u})*({u})))"
    if activation == "tanh":
        return f"(tanh({u}))"
    if activation == "sigmoid":
        return f"(1/(1+exp(-({u}))))"
    raise ValueError(f"unsupported activation {activation!r}; "
                     f"expected one of {_ACT}")


def _net_expr(coef: dict, activation: str, ng: str, nd: str) -> str:
    """Unroll one hidden layer: scalar multiplications, no loops/arrays."""
    w1 = coef["w1"]
    b1 = coef["b1"]
    w2 = coef["w2"]
    out = _fmt(coef["b2"])
    for i, wi in enumerate(w2):
        pre = (f"({_fmt(w1[i][0])}*{ng}+{_fmt(w1[i][1])}*{nd}"
               f"+{_fmt(b1[i])})")
        out += f"+{_fmt(wi)}*{_act_expr(pre, activation)}"
    return f"({out})"


def emit_ann_channel(fit: AnnChannelFit, vgs_term: str = "V(gi,ss)",
                     vds_term: str = "V(di,ss)") -> str:
    """Return the guarded ANN channel expression (ASCII, no new nodes).

    ``I = I_lo + w(Vds)*(I_hi - I_lo)`` with
    ``I_hi = Vds*exp(y1)*0.5*(1+tanh(alpha*(Vgs-Vth)))``,
    ``I_lo`` the same with the net evaluated at ``Vds = Vsat`` and
    ``w = 0.5*(1+tanh((Vds-Vsat)/guard_width))``.  At Vds=0 both branches are
    zero; for Vds<Vsat the low-V linear guard dominates; for Vgs<Vth the tanh
    cutoff suppresses current.  ``guard_width`` is the static Curve-1 Vds
    grid width carried in ``fit.coefficients`` (never the scattered DPT grid).
    """
    if not isinstance(fit, AnnChannelFit):
        raise TypeError("fit must be an AnnChannelFit")
    for term in (vgs_term, vds_term):
        if not isinstance(term, str) or not term:
            raise ValueError("terminal expressions must be non-empty strings")
        term.encode("ascii")
    c = fit.coefficients
    activation = fit.activation
    if activation not in _ACT:
        raise ValueError(f"unsupported activation {activation!r}")
    gw = float(c["guard_width"])
    if not math.isfinite(gw) or gw <= 0.0:
        raise ValueError("guard_width must be finite and > 0")

    ng = (f"(({vgs_term})-({_fmt(c['vgs_center'])}))"
          f"/({_fmt(c['vgs_scale'])})")
    nd = (f"(({vds_term})-({_fmt(c['vds_center'])}))"
          f"/({_fmt(c['vds_scale'])})")
    nd_sat = _fmt((fit.v_sat - float(c["vds_center"]))
                  / float(c["vds_scale"]))
    y_hi = _net_expr(c, activation, ng, nd)
    y_sat = _net_expr(c, activation, ng, nd_sat)
    cut = (f"(0.5*(1+tanh({_fmt(c['alpha'])}"
           f"*(({vgs_term})-({_fmt(fit.v_th)})))))")
    hi = f"(({vds_term})*exp({y_hi})*{cut})"
    lo = f"(({vds_term})*exp({y_sat})*{cut})"
    blend = (f"(0.5*(1+tanh((({vds_term})-({_fmt(fit.v_sat)}))"
             f"/({_fmt(gw)}))))")
    expr = f"({lo})+({blend})*(({hi})-({lo}))"
    expr.encode("ascii")
    return expr


def _resolve_eps(c1_eps) -> float:
    if c1_eps is None:
        return 0.0
    eps = float(c1_eps)
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("c1_eps must be finite and >= 0")
    return eps


def build_ann_subcircuit(spec, channel_fit: AnnChannelFit,
                         charge_fit: AnnChargeFit, *, c1_eps=None) -> str:
    """Emit ``.SUBCKT <name>_PIECSPICE_UROM d g s``.

    Parasitics, D_BODY model and pin order are copied from the closed Tier-2
    emitter (single source of truth); G_CH and the three Q= charge elements
    are replaced with the ANN bases.  ``c1_eps=None`` keeps the C0 ramp and is
    byte-identical in form to the legacy charge chain at %.6e precision.
    """
    if not isinstance(channel_fit, AnnChannelFit):
        raise TypeError("channel_fit must be an AnnChannelFit")
    if not isinstance(charge_fit, AnnChargeFit):
        raise TypeError("charge_fit must be an AnnChargeFit")
    eps = _resolve_eps(c1_eps)
    old = f"{spec.device_name}_PIECESPICE_T2"
    new = f"{spec.device_name}_PIECSPICE_UROM"

    t2 = build_tier2_subcircuit(spec)
    lines = t2.splitlines()
    i_sub = next(i for i, l in enumerate(lines) if l.startswith(".SUBCKT "))
    i_ch = next(i for i, l in enumerate(lines) if l.startswith("G_CH "))
    i_q = next(i for i, l in enumerate(lines)
               if "--- charge-conserving capacitances" in l)
    i_bd = next(i for i, l in enumerate(lines)
                if "--- body diode" in l)
    if not (i_sub < i_ch < i_q < i_bd):
        raise RuntimeError("closed Tier-2 builder layout changed; "
                           "ANN emitter cannot place its bases safely")

    channel = emit_ann_channel(channel_fit)
    gw = float(channel_fit.coefficients["guard_width"])
    q_lines: List[str] = ["* --- charge-conserving capacitances (ANN Q nets) ---"]
    for name in ("oss", "iss", "rss"):
        if name not in charge_fit.tables:
            raise ValueError(f"charge_fit is missing the {name!r} table")
        table = charge_fit.tables[name]
        expr = delta_ramp_chain(_TERMS[name], table["v"], table["q"], eps)
        element, nodes = _ELEMENTS[name]
        q_lines.append(f"{element} {nodes} Q = {{{expr}}}")
    q_lines.append("")

    body = [l.replace(old, new) for l in lines[i_bd:]]
    header = [
        f"* PieceSpice unified ROM (T2 ANN candidate) for {spec.device_name}",
        "* Channel: guarded one-hidden-layer ANN; charge: log-scale Q nets",
        "* Parasitics, D_BODY and semantic pins from the closed Tier-2 builder",
        f"* Guard width: {_fmt(gw)} V (static Curve-1 Vds grid, eps_from_grid)",
    ]
    out = header + [lines[i_sub].replace(old, new)] \
        + lines[i_sub + 1:i_ch] \
        + [f"G_CH di ss VALUE = {{({channel})}}", ""] \
        + q_lines + body
    text = "\n".join(out)
    text.encode("ascii")  # ASCII-only emission invariant
    return text
