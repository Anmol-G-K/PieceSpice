"""Emit the TPWL/POD-DEIM reduced model as a standard ASCII SPICE subcircuit.

LTspice notes that shaped this implementation:

* the caret `^` is XOR in B-source expressions, so powers use `**`;
* chains of ideal B voltage sources used only as intermediates produce a
  singular MNA matrix (observed: nodes `bdmin#branch`/`dmin`), so distances
  and weights are emitted as `.func` macros that expand inline;
* `idt(expr, ic)` is a verified state integrator under LTspice 26.

The Rewienski-White weight exponent 25 is the paper's constant.  Every
numeric coefficient comes from the fitted model.
"""
from __future__ import annotations

import numpy as np

from tier3.tpwl_deim import TPWLModel

_PORT = {"drain": "d", "gate": "g", "source": "s", "tj": "tj"}


def _f(x: float) -> str:
    return f"{float(x):.9e}"


def _linear(nodes: list[str], coeffs, const: float = 0.0) -> str:
    """SPICE expression for sum(coeff_i*node_i) + const."""
    parts = [f"({_f(c)}*{n})" for n, c in zip(nodes, coeffs) if c != 0.0]
    if const != 0.0 or not parts:
        parts.append(_f(const))
    return "+".join(parts)


def emit_subcircuit(model: TPWLModel, *, name: str | None = None) -> str:
    """Return a complete ``.subckt`` string for one fitted TPWL model."""
    name = name or f"{model.vendor.upper()}_TPWL_DEIM"
    ports = [_PORT.get(r, r) for r in model.pin_roles]
    r, s = model.r, model.n_points
    un = [f"V(z{k})" for k in range(r)]
    uref = model.u_ref if model.u_ref is not None else np.zeros(2)
    uspan = model.u_span if model.u_span is not None else np.ones(2)
    uu = [f"((V(g,s)-({_f(uref[0])}))/{_f(uspan[0])})",
          f"((V(d,s)-({_f(uref[1])}))/{_f(uspan[1])})"]

    lines = [f".subckt {name} {' '.join(ports)}"]

    # distance macros ||z - z_i|| (no intermediate voltage nodes)
    dist_fn = []
    for i in range(s):
        sq = "+".join(f"({un[k]}-({_f(model.z_points[i, k])}))**2"
                      for k in range(r))
        fname = f"{name}_dist{i}"
        lines.append(f".func {fname}() sqrt({sq})")
        dist_fn.append(f"{fname}()")
    dmin = dist_fn[0]
    for node in dist_fn[1:]:
        dmin = f"min({dmin},{node})"
    wex = []
    for i in range(s):
        fname = f"{name}_wex{i}"
        lines.append(f".func {fname}() exp({_f(-25.0)}*({dist_fn[i]}-{dmin}))")
        wex.append(f"{fname}()")
    wsum = "+".join(wex)
    wt = []
    for i in range(s):
        fname = f"{name}_wt{i}"
        lines.append(f".func {fname}() ({wex[i]})/({wsum})")
        wt.append(f"{fname}()")

    # reduced states: z_j = integral of weighted affine dynamics
    inv_t = _f(1.0 / model.time_scale)
    for j in range(r):
        terms = []
        for i in range(s):
            inner = (_linear(un, model.Ared[i][j, :])
                     + "+" + _linear(uu, model.Bred[i][j, :])
                     + "+" + _f(model.gamma[i][j]))
            terms.append(f"{wt[i]}*({inner})")
        # C=1 F integrator: dv/dt = injected current; 1 F is the
        # integrator's numerical scale, not a device parameter. The leak
        # resistor is the measured trajectory duration (R*C = T).
        lines.append(f"Cz{j} z{j} 0 1 IC={_f(model.z0[j])}")
        lines.append(f"Bz{j} z{j} 0 I=(({'+'.join(terms)})*{inv_t})")
        lines.append(f"Rz{j} z{j} 0 {_f(model.time_scale)}")

    # terminal currents, clamped to the measured training range
    for q, target in enumerate(("d", "g")):
        terms = []
        for i in range(s):
            inner = (_linear(un, model.C[q, :])
                     + "+" + _linear(uu, model.D[q, :])
                     + "+" + _f(model.e[q]))
            terms.append(f"{wt[i]}*({inner})")
        expr = "+".join(terms)
        if model.y_min is not None and model.y_max is not None:
            expr = f"limit(({expr}),({_f(model.y_min[q])}),({_f(model.y_max[q])}))"
        lines.append(f"G{target} {target} s VALUE={{{expr}}}")

    lines.append(".ends")
    return "\n".join(lines) + "\n"
