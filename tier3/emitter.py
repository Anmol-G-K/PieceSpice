"""Emit the TPWL terminal model as a standard ASCII SPICE subcircuit.

Structure (all coefficients come from the fitted model):

    .subckt NAME d g s
    * intermediates as behavioral current sources with Rpar=1:
    *   V(aux_vgs) = (V(g,s)-ref)/span   (same for vds)
    *   V(dist_i)  = ||w - anchor_i||
    *   V(dmin)    = min over anchors; V(wex_i) = exp(-25*(dist_i-dmin))
    *   V(wt_i)    = wex_i / sum(wex)
    G_CH d s VALUE={ sum_i V(wt_i)*(a_i + g_i*dw0 + h_i*dw1) }
    C_QG g s Q={ sum_i V(wt_i)*(q0_i + q1_i*dw0 + q2_i*dw1) }
    C_QD d s Q={ sum_i V(wt_i)*(c_i*dw0 + d_i*dw1) }
    .ends NAME

LTspice notes verified in this repository:

* ``Q={...}`` on a capacitor is honored, including 2-D expressions in
  V(g,s) and V(d,s) (smoke-tested to 7e-8 A against an analytic ramp);
* powers must use ``**`` (caret is XOR in B-source expressions);
* nested ``.func`` macros blow up combinatorially between 4 and 8 cells
  (measured: 4 cells runs, 8 cells stalls before the operating point),
  so intermediates are behavioral current sources with ``Rpar=1`` —
  the documented idiom that evaluates each subexpression once and
  gives every node a DC path;
* the Rewienski-White weight exponent 25 is the paper's constant; the
  exp argument is min-shifted (max value 0) so it never overflows, and
  the shift is REQUIRED: without it, far-from-anchor operating points
  drive every exp to zero and the normalisation becomes 0/0.
"""
from __future__ import annotations

import numpy as np

from tier3.tpwl_deim import TPWLTerminalModel, WEIGHT_EXPONENT

_PORT = {"drain": "d", "gate": "g", "source": "s", "tj": "tj"}


def _f(x: float) -> str:
    return f"({float(x):.9e})"


def _term(coef: float, dv: str) -> str:
    return f"({_f(coef)}*({dv}))" if coef != 0.0 else "0"


def emit_subcircuit(model: TPWLTerminalModel, *,
                    name: str | None = None) -> str:
    """Return a complete ``.subckt`` string for one fitted model."""
    name = name or f"{model.vendor.upper()}_TPWL_DEIM"
    ports = [_PORT.get(r, r) for r in model.pin_roles]
    s = len(model.anchors)
    p = f"{name}_"

    lines = [f".subckt {name} {' '.join(ports)}"]

    # scaled terminal voltages
    lines.append(f"B{p}vgs {p}vgs 0 I="
                 f"((V(g,s)-{_f(model.v_ref[0])})/{_f(model.v_span[0])}) Rpar=1")
    lines.append(f"B{p}vds {p}vds 0 I="
                 f"((V(d,s)-{_f(model.v_ref[1])})/{_f(model.v_span[1])}) Rpar=1")
    vgs, vds = f"V({p}vgs)", f"V({p}vds)"

    # distances to each anchor
    dist = []
    for i in range(s):
        sq = (f"({vgs}-{_f(model.anchors[i, 0])})**2"
              f"+({vds}-{_f(model.anchors[i, 1])})**2")
        lines.append(f"B{p}dist{i} {p}dist{i} 0 I=sqrt({sq}) Rpar=1")
        dist.append(f"V({p}dist{i})")
    dmin = dist[0]
    for d in dist[1:]:
        dmin = f"min({dmin},{d})"
    lines.append(f"B{p}dmin {p}dmin 0 I={dmin} Rpar=1")

    # unnormalised weights, min-shifted so the largest is exp(0)=1
    wex = []
    for i in range(s):
        lines.append(f"B{p}wex{i} {p}wex{i} 0 I="
                     f"exp({_f(-WEIGHT_EXPONENT)}*({dist[i]}-V({p}dmin)))"
                     f" Rpar=1")
        wex.append(f"V({p}wex{i})")
    lines.append(f"B{p}wsum {p}wsum 0 I={'+'.join(wex)} Rpar=1")
    wt = []
    for i in range(s):
        lines.append(f"B{p}wt{i} {p}wt{i} 0 I={wex[i]}/V({p}wsum) Rpar=1")
        wt.append(f"V({p}wt{i})")

    def blend(coef_rows, with_offset):
        """Per-cell affine piece; rows are [off, c_vgs, c_vds] or [c_vgs, c_vds]."""
        terms = []
        for i in range(s):
            dw0 = f"({vgs}-{_f(model.anchors[i, 0])})"
            dw1 = f"({vds}-{_f(model.anchors[i, 1])})"
            if with_offset:
                off, cvgs, cvds = coef_rows[i]
            else:
                off, cvgs, cvds = 0.0, coef_rows[i][0], coef_rows[i][1]
            parts = []
            if off != 0.0:
                parts.append(_f(off))
            if cvgs != 0.0:
                parts.append(_term(cvgs, dw0))
            if cvds != 0.0:
                parts.append(_term(cvds, dw1))
            inner = "+".join(parts) if parts else "0"
            terms.append(f"{wt[i]}*({inner})")
        return "+".join(terms)


    # AC-derived Qoss(Vds) base table (delta-slope chain, Tier-2 lineage).
    # The DPT never moves Vds at Vgs = 0, so the 2-D blend alone would leave
    # the blocking drain node with no capacitance and the transient would
    # collapse on the first timestep.
    if model.coss_v is not None and model.coss_c is not None:
        from approximation.charge_model import synthesize_charge_source
        block = synthesize_charge_source(
            "d", "s", model.coss_v, model.coss_c,
            model_name=f"{name}_QOSS")
        lines.extend(block.splitlines())
    channel_expr = blend(model.channel, True)
    if model.v_block is not None and model.v_width is not None:
        envelope = (f"limit((V(g,s)-({_f(model.v_block)}))/"
                    f"({_f(model.v_width)}),0,1)")
        channel_expr = f"{envelope}*({channel_expr})"
    lines.append(f"G_CH d s VALUE={{{channel_expr}}}")
    # C_QG is deliberately not emitted: a 2-terminal Q(vgs,vds) element on
    # the ideal-driven gate node produced a non-symmetric Jacobian and
    # stalled LTspice before the first edge on all tested vendors (measured:
    # removing it alone restores a full 14 us run). Gate current is not a
    # DPT figure; the Miller cross-charge is retained in C_QD via reciprocity.
    lines.append(f"C_QD d s Q={{{blend(model.charge_d, False)}}}")

    lines.append(".ends")
    return "\n".join(lines) + "\n"
