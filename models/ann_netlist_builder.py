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

``build_hybrid_ann_subcircuit`` copies the closed Tier-2 C0 subcircuit and
adds one signed ``G_ANN`` current source after ``G_CH``; charge tables, body
diode and parasitics remain byte-identical to the closed emitter.
"""

from __future__ import annotations

import math
from typing import List

from approximation.ann_channel import AnnChannelFit
from approximation.ann_charge import AnnChargeFit
from approximation.ann_residual import AnnResidualFit
from approximation.smoother import delta_ramp_chain
from models.netlist_builder import build_tier2_subcircuit

__all__ = ["emit_ann_channel", "build_ann_subcircuit",
           "emit_ann_residual", "build_hybrid_ann_subcircuit", "EXP_CAP"]

_ACT = ("isru", "tanh", "sigmoid")
# LTspice-compatible bound on every emitted exponential.  The fitted nets are
# narrow, but an off-grid evaluation must not overflow the simulator's exp().
EXP_CAP = 50.0
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


def _net_expr(coef: dict, activation: str, ng: str, nd: str,
              nt: str | None = None) -> str:
    """Unroll one hidden layer: scalar multiplications, no loops/arrays.

    ``nt`` is the optional third normalised temperature term; it is added
    only when the fit carries a three-column ``w1``.
    """
    w1 = coef["w1"]
    b1 = coef["b1"]
    w2 = coef["w2"]
    out = _fmt(coef["b2"])
    for i, wi in enumerate(w2):
        pre = (f"({_fmt(w1[i][0])}*{ng}+{_fmt(w1[i][1])}*{nd}")
        if nt is not None:
            if len(w1[i]) != 3:
                raise ValueError("three-input temperature term requires "
                                 "three w1 coefficients per neuron")
            pre += f"+{_fmt(w1[i][2])}*{nt}"
        pre += f"+{_fmt(b1[i])})"
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
    if not math.isfinite(EXP_CAP) or EXP_CAP <= 0.0:
        raise ValueError("EXP_CAP must be finite and > 0")

    w1 = c["w1"]
    widths = {len(row) for row in w1}
    if len(widths) != 1 or not widths.issubset({2, 3}):
        raise ValueError("w1 must have 2 or 3 coefficients per neuron")
    nt = None
    if widths == {3}:
        for key in ("temperature_center", "temperature_scale"):
            if key not in c:
                raise ValueError(f"temperature fit is missing {key}")
        tc = float(c["temperature_center"])
        ts = float(c["temperature_scale"])
        if not math.isfinite(ts) or ts <= 0.0:
            raise ValueError("temperature_scale must be finite and > 0")
        temp = float(c.get("temperature_value", tc))
        nt = f"((({_fmt(temp)})-({_fmt(tc)}))/{_fmt(ts)})"

    ng = (f"(({vgs_term})-({_fmt(c['vgs_center'])}))"
          f"/({_fmt(c['vgs_scale'])})")
    nd = (f"(({vds_term})-({_fmt(c['vds_center'])}))"
          f"/({_fmt(c['vds_scale'])})")
    nd_sat = _fmt((fit.v_sat - float(c["vds_center"]))
                  / float(c["vds_scale"]))
    y_hi = _net_expr(c, activation, ng, nd, nt)
    y_sat = _net_expr(c, activation, ng, nd_sat, nt)
    cut = (f"(0.5*(1+tanh({_fmt(c['alpha'])}"
           f"*(({vgs_term})-({_fmt(fit.v_th)})))))")
    cap = _fmt(EXP_CAP)
    hi = f"(({vds_term})*min(exp({y_hi}),{cap})*{cut})"
    lo = f"(({vds_term})*min(exp({y_sat}),{cap})*{cut})"
    blend = (f"(0.5*(1+tanh((({vds_term})-({_fmt(fit.v_sat)}))"
             f"/({_fmt(gw)}))))")
    expr = f"({lo})+({blend})*(({hi})-({lo}))"
    expr.encode("ascii")
    if isinstance(fit.provenance, dict):
        fit.provenance["exp_cap"] = float(EXP_CAP)
        fit.provenance["exp_cap_rule"] = (
            "min(exp(y), EXP_CAP) in both guarded channel branches; "
            "LTspice-compatible min()")
    return expr


def _resolve_eps(c1_eps) -> float:
    if c1_eps is None:
        return 0.0
    eps = float(c1_eps)
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("c1_eps must be finite and >= 0")
    return eps


def _charge_lines(charge_fit: AnnChargeFit, eps: float,
                  *, header: str) -> List[str]:
    """Monotone Q= charge elements (Q(0)=0) from the ANN charge tables."""
    q_lines: List[str] = [header]
    for table_name in ("oss", "iss", "rss"):
        if table_name not in charge_fit.tables:
            raise ValueError(f"charge_fit is missing the {table_name!r} table")
        table = charge_fit.tables[table_name]
        expr = delta_ramp_chain(_TERMS[table_name], table["v"], table["q"],
                                eps)
        element, nodes = _ELEMENTS[table_name]
        q_lines.append(f"{element} {nodes} Q = {{{expr}}}")
    q_lines.append("")
    return q_lines


def emit_ann_residual(fit: AnnResidualFit, vgs_term: str = "V(gi,ss)",
                      vds_term: str = "V(di,ss)") -> str:
    """Return the signed residual source expression (ASCII, no new nodes).

    ``I_res = Vds * (b2 + sum_i w2_i*phi(w1_i0*Vgs + w1_i1*Vds + b1_i))
    * 0.5*(1 + tanh(alpha*(Vgs - Vth)))``.  The explicit Vds factor pins the
    zero current at Vds=0, the tanh gate supplies threshold cutoff, and the
    signed weights allow either correction direction.
    """
    if not isinstance(fit, AnnResidualFit):
        raise TypeError("fit must be an AnnResidualFit")
    if fit.activation not in _ACT:
        raise ValueError(f"unsupported activation {fit.activation!r}")
    for term in (vgs_term, vds_term):
        if not isinstance(term, str) or not term:
            raise ValueError("terminal expressions must be non-empty strings")
        term.encode("ascii")
    c = fit.coefficients
    w1 = c["w1"]
    b1 = c["b1"]
    w2 = c["w2"]
    if not (len(w1) == len(b1) == len(w2) == fit.neurons):
        raise ValueError("residual coefficients do not match neurons")
    widths = {len(row) for row in w1}
    if len(widths) != 1 or not widths.issubset({2, 3}):
        raise ValueError("residual w1 must have 2 or 3 coefficients")
    nt = None
    if widths == {3}:
        for key in ("temperature_center", "temperature_scale"):
            if key not in c:
                raise ValueError(f"temperature residual is missing {key}")
        tc = float(c["temperature_center"])
        ts = float(c["temperature_scale"])
        if not math.isfinite(ts) or ts <= 0.0:
            raise ValueError("temperature_scale must be finite and > 0")
        temp = float(c.get("temperature_value", tc))
        nt = f"((({_fmt(temp)})-({_fmt(tc)}))/{_fmt(ts)})"
    net = _fmt(c["b2"])
    for i in range(fit.neurons):
        u = (f"({_fmt(w1[i][0])}*({vgs_term})"
             f"+{_fmt(w1[i][1])}*({vds_term})")
        if nt is not None:
            u += f"+{_fmt(w1[i][2])}*{nt}"
        u += f"+{_fmt(b1[i])})"
        net += f"+{_fmt(w2[i])}*{_act_expr(u, fit.activation)}"
    cut = (f"(0.5*(1+tanh({_fmt(fit.alpha)}"
           f"*(({vgs_term})-({_fmt(fit.v_th)})))))")
    expr = f"(({vds_term})*({net})*{cut})"
    expr.encode("ascii")  # ASCII-only emission invariant
    return expr


def build_hybrid_ann_subcircuit(spec, residual_fit: AnnResidualFit, *,
                                name: str | None = None,
                                subckt_name: str | None = None) -> str:
    """Emit ``.SUBCKT <subckt> d g s`` with C0 base + signed ANN residual.

    The closed Tier-2 subcircuit is emitted at the requested name and copied
    unchanged (C0 channel, C0 charge tables, body diode and parasitics);
    only ``G_ANN di ss`` is inserted immediately after ``G_CH``.  The
    residual introduces no fourth internal node.  ``name``/``subckt_name``
    follow the existing override convention (``subckt_name`` wins).
    """
    if not isinstance(residual_fit, AnnResidualFit):
        raise TypeError("residual_fit must be an AnnResidualFit")
    new = (subckt_name or name or f"{spec.device_name}_PIECSPICE_HYBRID")

    # Same name as the hybrid emission: every copied line is byte-identical
    # to the closed C0 builder, including the D_BD model suffix and .ENDS.
    lines = build_tier2_subcircuit(spec, name=new).splitlines()
    i_sub = next(i for i, l in enumerate(lines) if l.startswith(".SUBCKT "))
    i_ch = next(i for i, l in enumerate(lines) if l.startswith("G_CH "))
    if i_sub >= i_ch:
        raise RuntimeError("closed Tier-2 builder layout changed; "
                           "hybrid emitter cannot place its residual safely")

    residual = emit_ann_residual(residual_fit)
    header = [
        f"* PieceSpice hybrid C0+ANN ROM for {spec.device_name}",
        "* Channel: closed Tier-2 G_CH plus signed ANN residual G_ANN",
        "* Charge Q tables, body diode and parasitics: closed Tier-2 C0",
        f"* Residual: {residual_fit.neurons} neurons, "
        f"{residual_fit.activation}, v_th={_fmt(residual_fit.v_th)} V",
    ]
    out = (header + lines[:i_ch + 1]
           + ["", "* --- signed ANN channel residual (learned correction) ---",
              f"G_ANN di ss VALUE = {{{residual}}}"]
           + lines[i_ch + 1:])
    text = "\n".join(out)
    text.encode("ascii")  # ASCII-only emission invariant
    return text


def build_ann_subcircuit(spec, channel_fit: AnnChannelFit,
                         charge_fit: AnnChargeFit, *, c1_eps=None,
                         name: str | None = None,
                         subckt_name: str | None = None) -> str:
    """Emit ``.SUBCKT <subckt> d g s``.

    Parasitics, D_BODY model and pin order are copied from the closed Tier-2
    emitter (single source of truth); G_CH and the three Q= charge elements
    are replaced with the ANN bases.  ``c1_eps=None`` keeps the C0 ramp and is
    byte-identical in form to the legacy charge chain at %.6e precision.

    ``name``/``subckt_name`` override the emitted name (and the D_BD model
    suffix and ``.ENDS`` name); both default to None, which keeps the
    canonical ``<device>_PIECSPICE_UROM`` name.  ``subckt_name`` wins when
    both are supplied, matching the closed Tier-2 builder convention.
    """
    if not isinstance(channel_fit, AnnChannelFit):
        raise TypeError("channel_fit must be an AnnChannelFit")
    if not isinstance(charge_fit, AnnChargeFit):
        raise TypeError("charge_fit must be an AnnChargeFit")
    eps = _resolve_eps(c1_eps)
    # The Tier-2 builder emits its closed T2 name; replacing it across the
    # header and body updates the .SUBCKT, D_BD model suffix and .ENDS.
    old = f"{spec.device_name}_PIECESPICE_T2"
    new = (subckt_name or name or f"{spec.device_name}_PIECSPICE_UROM")

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
    q_lines = _charge_lines(charge_fit, eps,
                            header="* --- charge-conserving capacitances "
                                   "(ANN Q nets) ---")

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
