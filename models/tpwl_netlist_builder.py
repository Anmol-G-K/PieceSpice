"""Emit the TPWL terminal model as an ASCII standard-SPICE subcircuit.

Measured hardening (csd16412q5a, 2026-11-19).  The earlier Rewienski-White
weight blend emitted through B-current-source intermediates failed the DPT:
10 trace points, t = 5.12e-12 s, ``trouble with node "drain"``.  Bisection
showed the same collapse with G_CH alone (tolerance-relaxation loop at
9.87e-7 s), +C_QD (1e-14 s) and +C_QOSS (5.12e-12 s).  A hard Voronoi
``if()`` cascade (32 cells, ~437 KB) timed out before the operating point,
so it was rejected.  The flat ``table()`` channel emitted below completes
the 14 us DPT for all four vendors.

The 2026-11-19 speed/fidelity pass measured these changes on the identical
bench (solver options unchanged):

* ``C_QOSS`` becomes a two-exponential smooth monotone charge law fitted to
  the AC-derived Qoss table (fit RMS 0.09-0.19 %).  LTspice's 160-knob
  PWL charge lookup contributed Jacobian discontinuities at the drain node.
* The channel grid is conditioned per vendor.  A 17 x 21 or 21 x 31 mesh
  cuts nested-``table()`` evaluation cost on IRFS/EPC/CPM2 without a static
  regression; dense meshes were measured to be slower.
* IRFS needs the high-Vds saturation hold: the raw TPWL extrapolation turns
  negative at Vgs = 10 V, Vds = 38.94 V, so ``max(0, ...)`` zeroed the device
  for the entire second pulse (Vds RMSE 39.59 %).  Holding the Curve-1
  saturation value above the static sweep span removes that hard failure.
* TI and CPM2 use a soft (softplus) non-negative clamp; EPC uses it where
  the switching locus rides the clamp.  The clamp keeps the max(0, ...)'
  asymptote but removes its curvature discontinuity.
* ``C_QD`` remains omitted: fitted drain-charge slopes are negative at the
  blocking point (~-2.5 nF on TI) and produced an exponential blow-up at
  t ~ 5 ps.  Gate charge is not emitted; it previously made the Jacobian
  non-symmetric on the ideal driven gate node and stalled LTspice.

LTspice syntax facts verified in this repository: ``**`` for powers (caret
is XOR), nested ``table()`` lookups evaluate correctly, and ``limit()`` /
``max()`` / ``tanh()`` / ``sqrt()`` / ``exp()`` are built in.  Emission stays
ASCII.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from config.yaml_io import load_input, load_output
from approximation.static_curves import load_output_family
from approximation.tpwl_deim import TPWLTerminalModel, paper_weights

_PORT = {"drain": "d", "gate": "g", "source": "s", "tj": "tj"}
_VSAT_DEFAULT = 0.2        # low-Vds saturation scale [V], measured on TI/IRFS/CPM2
_VSAT_BY_VENDOR = {"epc2216": 0.5}  # EPC stalls at 0.2 V; 0.5 V completes
_QOSS_FIT_POINTS = 160
_HIGH_VDS_HEADROOM = 1.10


@dataclass(frozen=True)
class EmissionPolicy:
    """Measured emission settings; defaults reproduce the original emitter."""
    vgs_points: int = 25
    vds_points: int = 41
    smooth_qoss: bool = False
    soft_clamp: bool = False
    soft_eps: float = 1e-3
    high_vds_saturation: bool = False


_BASE = EmissionPolicy()
_POLICY_BY_VENDOR = {
    "csd16412q5a": EmissionPolicy(soft_clamp=True),
    "irfs7730pbf": EmissionPolicy(vgs_points=17, vds_points=21,
                                  smooth_qoss=True,
                                  high_vds_saturation=True),
    "epc2216": EmissionPolicy(vgs_points=19, vds_points=25,
                              smooth_qoss=True, soft_clamp=True,
                              soft_eps=1e-5),
    "cpm2-1200": EmissionPolicy(vgs_points=21, vds_points=31,
                                smooth_qoss=True, soft_clamp=True),
}


def emission_policy(model: TPWLTerminalModel) -> EmissionPolicy:
    """Policy measured for one vendor's emitted channel and Qoss."""
    return _POLICY_BY_VENDOR.get(model.vendor, _BASE)


def saturation_scale(model: TPWLTerminalModel) -> float:
    """Measured low-Vds saturation scale for one vendor's emitted channel."""
    return float(_VSAT_BY_VENDOR.get(model.vendor, _VSAT_DEFAULT))


def _f(x: float) -> str:
    return f"{float(x):.9e}"

def raw_anchors(model: TPWLTerminalModel) -> np.ndarray:
    """Anchor positions in raw volts, shape (2, n_cells)."""
    return model.v_ref[:, None] + model.anchors.T * model.v_span[:, None]


def grid_axes(model: TPWLTerminalModel) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic raw-voltage grid covering DPT, static family, anchors."""
    cfg = load_input(model.vendor)
    dpt = cfg["dpt"]
    levels, vds_grid, _ = load_output_family(model.vendor)
    anchors = raw_anchors(model)
    vgs_lo = min(0.0, float(levels.min()), float(anchors[0].min()))
    vgs_hi = max(float(dpt["vgs_on"]), float(levels.max()),
                 float(anchors[0].max()))
    vds_lo = min(0.0, float(vds_grid.min()), float(anchors[1].min()))
    vds_hi = max(float(dpt["vbus"]), float(vds_grid.max()),
                 float(anchors[1].max()))
    pol = emission_policy(model)
    g = np.linspace(vgs_lo, vgs_hi, pol.vgs_points)
    d = np.linspace(vds_lo, vds_hi, pol.vds_points)
    if pol.high_vds_saturation:
        # The switching locus overshoots the bus; cover it explicitly so
        # table() never flat-extrapolates an unphysical negative endpoint.
        dmax = max(float(d[-1]), _HIGH_VDS_HEADROOM * float(dpt["vbus"]))
        d = np.concatenate([d, np.linspace(d[-1], dmax, 9)[1:]])
    if model.v_block is not None and model.v_width is not None:
        knee = model.v_block + model.v_width * np.array(
            [-2.0, -1.0, 0.0, 1.0, 2.0])
        g = np.unique(np.concatenate([g, knee]))
        g = g[(g >= vgs_lo) & (g <= vgs_hi)]
    return g, d


def blended_channel(model: TPWLTerminalModel, u: np.ndarray) -> np.ndarray:
    """Analytic Rewienski-White blend of the affine channel pieces [A]."""
    ws = model.scaled(u)
    w = paper_weights(model.anchors, ws)
    dw = ws[:, None, :] - model.anchors.T[:, :, None]
    return np.einsum("sk,sk->k", w, model.channel[:, 0][:, None]
                     + model.channel[:, 1][:, None] * dw[0]
                     + model.channel[:, 2][:, None] * dw[1])


def grid_channel(model: TPWLTerminalModel, g: np.ndarray, d: np.ndarray
                 ) -> np.ndarray:
    """Channel values on the emitted (Vgs, Vds) mesh, shape (len(g), len(d))."""
    vgs, vds = np.meshgrid(g, d, indexing="ij")
    table = blended_channel(model, np.vstack([vgs.ravel(), vds.ravel()])).reshape(
        vgs.shape)
    if emission_policy(model).high_vds_saturation:
        _, static_vds, _ = load_output_family(model.vendor)
        j = int(np.searchsorted(d, float(np.max(static_vds))))
        hold = np.maximum(table[:, j], 0.0)
        for k in range(j, len(d)):
            table[:, k] = hold
    return table


def transfer_curve(model: TPWLTerminalModel
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Curve-2 Id(Vgs) normalised to 1 at the first static Vgs level."""
    data = load_output(model.vendor, "curve2_transfer_characteristic")
    v = np.asarray(data["extracted"]["curve_vgs"], dtype=float)
    i = np.asarray(data["extracted"]["curve_id"], dtype=float)
    levels, _, _ = load_output_family(model.vendor)
    above = levels[levels > (model.v_block or 0.0)]
    v_full = float(above.min()) if len(above) else float(levels.max())
    mask = v <= v_full
    v, i = v[mask], i[mask]
    i = np.maximum.accumulate(np.maximum(i, 0.0))
    norm = float(np.interp(v_full, v, i)) or 1.0
    return v, np.clip(i / norm, 0.0, 1.0)


def emitted_channel(model: TPWLTerminalModel, u: np.ndarray) -> np.ndarray:
    """SPICE-emitted channel current on operating points (2, K) [A].

    Mirrors the netlist: 2-D table interpolation, Curve-2 gate factor,
    non-negative clamp (soft where measured) and low-Vds saturation.  Used
    for the static metric so the reported number describes what LTspice
    actually evaluates.
    """
    u = np.atleast_2d(u)
    g, d = grid_axes(model)
    table = grid_channel(model, g, d)
    row = np.array([
        [np.interp(vds, d, table[k]) for vds in u[1]]
        for k in range(len(g))])
    # table() interpolates along Vgs after evaluating every row lookup
    vgs_clamped = np.clip(u[0], g[0], g[-1])
    out = np.array([
        np.interp(vgs_clamped[j], g, row[:, j]) for j in range(len(u[0]))])
    tv, tf = transfer_curve(model)
    gate = np.clip(np.interp(u[0], tv, tf), 0.0, 1.0)
    x = out * gate * u[1] / (u[1] + saturation_scale(model))
    pol = emission_policy(model)
    if pol.soft_clamp:
        return 0.5 * (x + np.sqrt(x * x + pol.soft_eps ** 2))
    return np.maximum(0.0, x)


def emitted_static_rmse(model: TPWLTerminalModel) -> float:
    """Relative RMSE of the emitted channel on the Curve-1 static family."""
    from approximation.static_curves import static_samples
    u, ref, _ = static_samples(model.vendor)
    pred = emitted_channel(model, u)
    span = float(np.ptp(ref)) or 1.0
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / span)


def _qoss_expr(model: TPWLTerminalModel) -> str:
    """Monotone drain charge: smooth two-exponential fit or dense table."""
    from approximation.charge_model import integrate_charge
    charge = integrate_charge(model.coss_v, model.coss_c)
    if emission_policy(model).smooth_qoss:
        v = np.linspace(0.0, float(charge.v[-1]), _QOSS_FIT_POINTS)
        q = np.interp(v, charge.v, charge.q)
        qmax = max(float(np.max(np.abs(q))), 1e-15)

        def resid(p: np.ndarray) -> np.ndarray:
            a, b, l1, l2 = p
            return (a * (1 - np.exp(-v / np.exp(l1)))
                    + b * (1 - np.exp(-v / np.exp(l2))) - q) / qmax

        r = least_squares(
            resid,
            [q[-1] * 0.5, q[-1] * 0.5, np.log(2.0), np.log(20.0)],
            bounds=([0, 0, np.log(1e-4), np.log(1e-4)],
                    [np.inf, np.inf, np.log(1e4), np.log(1e4)]),
            max_nfev=5000)
        a, b, t1, t2 = (float(r.x[0]), float(r.x[1]),
                        float(np.exp(r.x[2])), float(np.exp(r.x[3])))
        eps = 1e-3
        vs = (f"(0.5*(V(d,s)+sqrt(V(d,s)*V(d,s)+{_f(eps * eps)})))")
        return (f"({_f(a)}*(1-exp(-{vs}/{_f(t1)}))"
                f"+{_f(b)}*(1-exp(-{vs}/{_f(t2)})))")
    v = np.linspace(0.0, float(charge.v[-1]), _QOSS_FIT_POINTS)
    q = np.interp(v, charge.v, charge.q)
    pairs = ",".join(f"{_f(x)},{_f(y)}" for x, y in zip(v, q))
    return (f"table(limit(V(d,s),{_f(v[0])},{_f(v[-1])}),{pairs})")


def emit_subcircuit(model: TPWLTerminalModel, *,
                    name: str | None = None) -> str:
    """Return a complete ``.subckt`` string for one fitted model."""
    name = name or f"{model.vendor.upper()}_TPWL_DEIM"
    ports = [_PORT.get(r, r) for r in model.pin_roles]
    g, d = grid_axes(model)
    table = grid_channel(model, g, d)
    tv, tf = transfer_curve(model)

    def rows(vals: np.ndarray) -> str:
        return ",".join(
            f"{_f(vgs)},table(V(d,s)," + ",".join(
                f"{_f(x)},{_f(y)}" for x, y in zip(d, row)) + ")"
            for vgs, row in zip(g, vals))

    channel_table = (f"table(limit(V(g,s),{_f(g[0])},{_f(g[-1])}),"
                     + rows(table) + ")")
    gate_table = (f"table(limit(V(g,s),{_f(tv[0])},{_f(tv[-1])}),"
                  + ",".join(f"{_f(x)},{_f(y)}" for x, y in zip(tv, tf)) + ")")
    vsat = saturation_scale(model)
    x = (f"({gate_table})*({channel_table})"
         f"*V(d,s)/(V(d,s)+{_f(vsat)})")
    pol = emission_policy(model)
    if pol.soft_clamp:
        channel = f"(0.5*({x}+sqrt({x}*{x}+{_f(pol.soft_eps ** 2)})))"
    else:
        channel = f"max(0,{x})"

    lines = [f".subckt {name} {' '.join(ports)}"]
    if model.coss_v is not None and model.coss_c is not None:
        lines.append(f"C{name}_QOSS d s Q={{{_qoss_expr(model)}}}")
    lines.append(f"G_CH d s VALUE={{{channel}}}")
    lines.append(".ends")
    return "\n".join(lines) + "\n"
