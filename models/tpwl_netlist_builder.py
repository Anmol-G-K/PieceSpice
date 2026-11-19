"""Emit the TPWL terminal model as an ASCII standard-SPICE subcircuit.

Measured hardening (csd16412q5a, 2026-11-19).  The earlier Rewienski-White
weight blend emitted through B-current-source intermediates failed the DPT:
10 trace points, t = 5.12e-12 s, ``trouble with node "drain"``.  Bisection
showed the same collapse with G_CH alone (tolerance-relaxation loop at
9.87e-7 s), +C_QD (1e-14 s) and +C_QOSS (5.12e-12 s).  A hard Voronoi
``if()`` cascade (32 cells, ~437 KB) timed out before the operating point,
so it was rejected.  The emitted form below completed the full 14 us DPT:
66240 trace points, 394001 solver iterations, Vds rel-RMSE 2.38 % and Id
rel-RMSE 0.42 % against the captured golden waveform for TI.

Structure (all numbers come from the fitted model or extracted curves):

* ``G_CH`` is a flat 2-D ``table()`` lookup of the blended affine channel
  precomputed on a deterministic (Vgs, Vds) grid.  No extra solver nodes and
  no exponential weight intermediates; LTspice's PWL interpolation is robust.
* The gate factor is a 1-D ``table()`` of the extracted Curve-2 transfer
  characteristic normalised to 1 at the first static Vgs level, so the
  Curve-1 static family is unchanged while sub-knee turn-on is physical.
* The channel is clamped non-negative and multiplied by a smooth low-Vds
  saturation ``Vds/(Vds + Vsat)`` (Vsat = 0.2 V, 0.5 V for EPC2216).  Both
  were required by measurement: without them the drain node stalls at the
  0.1 V Qoss kink on the first gate edge.  The TPWL affine fit extrapolates
  a negative current and an unbounded slope below its training span; the
  factor restores a finite linear-region conductance.
* ``C_QD`` is deliberately omitted.  Its fitted drain-charge slopes are
  negative at the blocking point (dQd/dVds ~ -2.5 nF at Vgs = 0, Vds = 12.5 V
  for TI), which produced a negative drain capacitance and an exponential
  current blow-up at t ~ 5 ps.  Qoss from the AC capacitance curve supplies
  the blocking-state charge and is emitted as a dense monotone ``table()``.
* Gate charge is not emitted: a 2-terminal Q(Vgs,Vds) element on the ideal
  driven gate node made the Jacobian non-symmetric and stalled LTspice.

LTspice syntax facts verified in this repository: ``**`` for powers (caret
is XOR), nested ``table()`` lookups evaluate correctly (smoke-tested), and
``limit()``/``max()``/``tanh()`` are built in.  Emission stays ASCII.
"""
from __future__ import annotations

import numpy as np

from config.yaml_io import load_input, load_output
from approximation.static_curves import load_output_family
from approximation.tpwl_deim import TPWLTerminalModel, paper_weights

_PORT = {"drain": "d", "gate": "g", "source": "s", "tj": "tj"}
_VSAT_DEFAULT = 0.2        # low-Vds saturation scale [V], measured on TI/IRFS/CPM2
_VSAT_BY_VENDOR = {"epc2216": 0.5}  # EPC stalls at 0.2 V; 0.5 V completes
_VGS_POINTS = 25
_VDS_POINTS = 41
_QOSS_POINTS = 160


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
    g = np.linspace(vgs_lo, vgs_hi, _VGS_POINTS)
    d = np.linspace(vds_lo, vds_hi, _VDS_POINTS)
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
    return blended_channel(model, np.vstack([vgs.ravel(), vds.ravel()])).reshape(
        vgs.shape)


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
    non-negative clamp and low-Vds saturation.  Used for the static metric
    so the reported number describes what LTspice actually evaluates.
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
    return np.maximum(0.0, out * gate * u[1] / (u[1] + saturation_scale(model)))


def emitted_static_rmse(model: TPWLTerminalModel) -> float:
    """Relative RMSE of the emitted channel on the Curve-1 static family."""
    from approximation.static_curves import static_samples
    u, ref, _ = static_samples(model.vendor)
    pred = emitted_channel(model, u)
    span = float(np.ptp(ref)) or 1.0
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / span)


def _qoss_expr(model: TPWLTerminalModel) -> str:
    """Monotone Qoss(Vds) lookup from the AC-derived charge table."""
    from approximation.charge_model import integrate_charge
    charge = integrate_charge(model.coss_v, model.coss_c)
    v = np.linspace(0.0, float(charge.v[-1]), _QOSS_POINTS)
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
    channel = (f"max(0,({gate_table})*({channel_table})"
               f"*V(d,s)/(V(d,s)+{_f(vsat)}))")

    lines = [f".subckt {name} {' '.join(ports)}"]
    if model.coss_v is not None and model.coss_c is not None:
        lines.append(f"C{name}_QOSS d s Q={{{_qoss_expr(model)}}}")
    lines.append(f"G_CH d s VALUE={{{channel}}}")
    lines.append(".ends")
    return "\n".join(lines) + "\n"
