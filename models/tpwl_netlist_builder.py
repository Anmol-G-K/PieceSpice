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
  t ~ 5 ps.  The fitted gate-charge emission experiment is measured and
  closed: both the dense Qg form and the 0.5%-compressed per-Vgs-row form
  reached the same 120 s IRFS emitted-DPT / forced-bench timeout guards, so
  the closed v1 emitter intentionally contains no ``C_QG`` element.  No
  accuracy or speed claim is made for the removed experiment.

LTspice syntax facts verified in this repository: ``**`` for powers (caret
is XOR), nested ``table()`` lookups evaluate correctly, and ``limit()`` /
``max()`` / ``tanh()`` / ``sqrt()`` / ``exp()`` are built in.  Emission stays
ASCII.

The opt-in ``V2_POLICY`` emission path adds the internal ``gi/di/ss`` nodes
(``RG``/``RD``/``RS``), the DPT gate-charge table (``C_QG``) and the Tier-2
body diode.  The opt-in compact path (``EmissionPolicy.compact`` /
``COMPACT_POLICY``) prunes only the shared Vds axis by measured interpolation
error; Vgs rows, cell count, clamp/Qoss settings and the ASCII ``table()``
syntax are kept.  The default call (``policy=None``) keeps the closed v1 text
byte-for-byte; neither physical elements nor compact emission are inferred.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from config.yaml_io import load_input, load_output
from approximation.smoother import hyp_expr, hyp_value
from approximation.static_curves import load_output_family
from approximation.tpwl_deim import TPWLTerminalModel, paper_weights

_PORT = {"drain": "d", "gate": "g", "source": "s", "tj": "tj"}
_VSAT_DEFAULT = 0.2        # low-Vds saturation scale [V], measured on TI/IRFS/CPM2
_VSAT_BY_VENDOR = {"epc2216": 0.5}  # EPC stalls at 0.2 V; 0.5 V completes
_QOSS_FIT_POINTS = 160

# V=0 regularisation of the landed two-exponential Qoss guard (not a hinge
# width). C1 hinge widths are data-derived via eps_from_grid and supplied to
# the Tier-2 emitter through c1_eps; no new magic width is introduced here.
_QOSS_V0_EPS = 1e-3
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
    include_gate_charge: bool = False
    include_body_diode: bool = False
    include_parasitics: bool = False
    # Opt-in Vds-axis compression; defaults keep the closed v1 table.
    compact: bool = False
    compact_rel_tol: float = 0.005


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

# Opt-in V2 emission path: baseline grid/clamp settings plus the physical
# elements.  The closed v1 record keeps using its measured vendor policy.
V2_POLICY = EmissionPolicy(include_gate_charge=True,
                           include_body_diode=True,
                           include_parasitics=True)


def compact_policy(base: TPWLTerminalModel | EmissionPolicy, *,
                   rel_tol: float | None = None) -> EmissionPolicy:
    """Vendor emission policy with compact Vds emission enabled.

    Grid, clamp and Qoss settings are inherited from the measured vendor
    policy; only ``compact`` and its tolerance change.  Accepts a model or
    an existing policy so callers cannot lose the vendor's settings.
    """
    pol = base if isinstance(base, EmissionPolicy) else emission_policy(base)
    tol = pol.compact_rel_tol if rel_tol is None else float(rel_tol)
    return replace(pol, compact=True, compact_rel_tol=tol)


# Factory, not a fixed policy: the vendor clamp/grid settings must survive
# the opt-in.  Use COMPACT_POLICY(model) or COMPACT_POLICY(policy).
COMPACT_POLICY = compact_policy


# Documented Tier-2 emitter scalars (pipeline.fitting.build_pwl_spec
# metadata: r_gate/r_drain/r_source).  Reused verbatim, not re-measured.
_R_GATE_OHM = 1.5
_R_DRAIN_OHM = 1e-4
_R_SOURCE_OHM = 1e-4


def _registered_vendor(model: TPWLTerminalModel) -> None:
    """Fail closed: emission settings exist only for measured vendors."""
    if model.vendor not in _POLICY_BY_VENDOR:
        raise ValueError(
            f"{model.vendor}: no measured TPWL emission policy; registered "
            f"vendors: {', '.join(sorted(_POLICY_BY_VENDOR))}")


def emission_policy(model: TPWLTerminalModel) -> EmissionPolicy:
    """Policy measured for one vendor's emitted channel and Qoss."""
    _registered_vendor(model)
    return _POLICY_BY_VENDOR[model.vendor]


def saturation_scale(model: TPWLTerminalModel) -> float:
    """Measured low-Vds saturation scale for one vendor's emitted channel."""
    _registered_vendor(model)
    return float(_VSAT_BY_VENDOR.get(model.vendor, _VSAT_DEFAULT))


def _f(x: float) -> str:
    return f"{float(x):.9e}"

def raw_anchors(model: TPWLTerminalModel) -> np.ndarray:
    """Anchor positions in raw volts, shape (2, n_cells)."""
    return model.v_ref[:, None] + model.anchors.T * model.v_span[:, None]


def grid_axes(model: TPWLTerminalModel, *,
              policy: EmissionPolicy | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic raw-voltage grid covering DPT, static family, anchors."""
    pol = emission_policy(model) if policy is None else policy
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


def grid_channel(model: TPWLTerminalModel, g: np.ndarray, d: np.ndarray, *,
                 policy: EmissionPolicy | None = None
                 ) -> np.ndarray:
    """Channel values on the emitted (Vgs, Vds) mesh, shape (len(g), len(d))."""
    pol = emission_policy(model) if policy is None else policy
    vgs, vds = np.meshgrid(g, d, indexing="ij")
    table = blended_channel(model, np.vstack([vgs.ravel(), vds.ravel()])).reshape(
        vgs.shape)
    if pol.high_vds_saturation:
        _, static_vds, _ = load_output_family(model.vendor)
        j = int(np.searchsorted(d, float(np.max(static_vds))))
        hold = np.maximum(table[:, j], 0.0)
        for k in range(j, len(d)):
            table[:, k] = hold
    return table


def compact_channel_table(model: TPWLTerminalModel, g: np.ndarray,
                          d: np.ndarray, table: np.ndarray,
                          rel_tol: float) -> tuple[np.ndarray, np.ndarray,
                                                   float, float]:
    """Prune the shared Vds axis by greedy farthest-point interpolation.

    Endpoints are always kept.  On each step the original Vds index with the
    largest absolute interpolation error over *all* Vgs rows becomes a knot.
    Stop at ``rel_tol * max|table|`` or when no error remains; the returned
    errors are measured on the retained table, not estimated.
    """
    g = np.asarray(g, dtype=float)
    d = np.asarray(d, dtype=float)
    table = np.asarray(table, dtype=float)
    tol = float(rel_tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(
            f"{model.vendor}: compact_rel_tol must be finite and > 0; "
            f"got {rel_tol!r}")
    if d.ndim != 1 or d.size < 2:
        raise ValueError(
            f"{model.vendor}: compact Vds axis has {d.size} point(s); "
            "need >=2")
    if (g.ndim != 1 or table.ndim != 2
            or table.shape != (g.size, d.size)):
        raise ValueError(
            f"{model.vendor}: channel table shape {table.shape} does not "
            f"match axes {(g.size, d.size)}")
    if not np.all(np.isfinite(d)) or np.any(np.diff(d) <= 0.0):
        raise ValueError(
            f"{model.vendor}: compact Vds axis is not finite and increasing")
    if not np.all(np.isfinite(table)):
        raise ValueError(
            f"{model.vendor}: channel table has non-finite values; "
            "refusing to compress")
    scale = float(np.max(np.abs(table)))
    budget = tol * scale
    keep = [0, d.size - 1]

    def remaining_error(indices: list[int]) -> np.ndarray:
        knots = d[indices]
        interp = np.array([np.interp(d, knots, row[indices])
                           for row in table])
        err = np.max(np.abs(table - interp), axis=0)
        err[indices] = -np.inf          # already a knot, no remaining error
        return err

    err = remaining_error(keep)
    while True:
        worst = float(np.max(err))
        if worst <= 0.0 or worst <= budget:
            break
        keep.append(int(np.argmax(err)))
        keep.sort()
        if len(keep) == d.size:
            break
        err = remaining_error(keep)

    knots_idx = np.asarray(keep, dtype=int)
    knots = d[knots_idx]
    interp = np.array([np.interp(d, knots, row[knots_idx]) for row in table])
    abs_err = float(np.max(np.abs(table - interp)))
    rel_err = 0.0 if scale == 0.0 else abs_err / scale
    if not np.isfinite(abs_err) or rel_err > tol:
        raise ValueError(
            f"{model.vendor}: compact Vds budget {tol:.6g} not met "
            f"(achieved {rel_err:.6g} with {len(keep)} point(s))")
    return knots, table[:, knots_idx], abs_err, rel_err


def _emission_channel(model: TPWLTerminalModel, pol: EmissionPolicy
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Emission mesh and measured metrics for one policy (no simulation)."""
    g, d = grid_axes(model, policy=pol)
    table = grid_channel(model, g, d, policy=pol)
    metrics = {
        "table_points": int(d.size),
        "vgs_points": int(g.size),
        "max_abs_error_a": 0.0,
        "max_rel_error_pct": 0.0,
        "compact": bool(pol.compact),
        "rel_tol": float(pol.compact_rel_tol),
    }
    if pol.compact:
        d, table, abs_err, rel_err = compact_channel_table(
            model, g, d, table, pol.compact_rel_tol)
        metrics.update(table_points=int(d.size),
                       max_abs_error_a=abs_err,
                       max_rel_error_pct=100.0 * rel_err)
    return g, d, table, metrics


def emission_metrics(model: TPWLTerminalModel,
                     policy: EmissionPolicy | None = None) -> dict:
    """Measured Vds compression metrics for one emission policy."""
    pol = emission_policy(model) if policy is None else policy
    return _emission_channel(model, pol)[3]


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


def emitted_channel(model: TPWLTerminalModel, u: np.ndarray, *,
                    policy: EmissionPolicy | None = None) -> np.ndarray:
    """SPICE-emitted channel current on operating points (2, K) [A].

    Mirrors the netlist: 2-D table interpolation, Curve-2 gate factor,
    non-negative clamp (soft where measured) and low-Vds saturation.  Used
    for the static metric so the reported number describes what LTspice
    actually evaluates.
    """
    u = np.atleast_2d(u)
    pol = emission_policy(model) if policy is None else policy
    g, d, table, _ = _emission_channel(model, pol)
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
    if pol.soft_clamp:
        return np.asarray(hyp_value(x, pol.soft_eps), dtype=float)
    return np.maximum(0.0, x)


def emitted_static_rmse(model: TPWLTerminalModel, *,
                        policy: EmissionPolicy | None = None) -> float:
    """Relative RMSE of the emitted channel on the Curve-1 static family."""
    from approximation.static_curves import static_samples
    u, ref, _ = static_samples(model.vendor)
    pred = emitted_channel(model, u, policy=policy)
    span = float(np.ptp(ref)) or 1.0
    return float(np.sqrt(np.mean((pred - ref) ** 2)) / span)


def _qoss_expr(model: TPWLTerminalModel, *,
               policy: EmissionPolicy | None = None,
               vds_expr: str = "V(d,s)") -> str:
    """Monotone drain charge: surface/table source, smooth fit or dense PWL."""
    from approximation.charge_model import integrate_charge
    pol = emission_policy(model) if policy is None else policy
    surface = getattr(model, "qoss_surface", None)
    if surface is not None:
        charge_v, charge_q = surface.qoss_table(
            0.0, temperature=model.temperature_c)
        charge_v = np.asarray(charge_v, dtype=float)
        charge_q = np.asarray(charge_q, dtype=float)
    else:
        coss_v, coss_c = model.coss_v, model.coss_c
        if coss_v is None or coss_c is None:
            raise ValueError(
                f"{model.vendor}: Coss charge table is unavailable")
        charge = integrate_charge(coss_v, coss_c)
        charge_v, charge_q = charge.v, charge.q
    if charge_v.size < 2 or charge_q.size != charge_v.size:
        raise ValueError(
            f"{model.vendor}: Coss charge table has "
            f"{charge_v.size} usable point(s); need >=2")
    if pol.smooth_qoss:
        v = np.linspace(0.0, float(charge_v[-1]), _QOSS_FIT_POINTS)
        q = np.interp(v, charge_v, charge_q)
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
        vs = hyp_expr(vds_expr, _QOSS_V0_EPS)
        return (f"({_f(a)}*(1-exp(-{vs}/{_f(t1)}))"
                f"+{_f(b)}*(1-exp(-{vs}/{_f(t2)})))")
    v = np.linspace(0.0, float(charge_v[-1]), _QOSS_FIT_POINTS)
    q = np.interp(v, charge_v, charge_q)
    pairs = ",".join(f"{_f(x)},{_f(y)}" for x, y in zip(v, q))
    return (f"table(limit({vds_expr},{_f(v[0])},{_f(v[-1])}),{pairs})")


def _tier2_body_diode_lines(model: TPWLTerminalModel) -> list[str]:
    """Reuse the closed Tier-2 body-diode construction verbatim.

    ``pipeline.fitting.build_pwl_spec`` supplies the same Curve-6 samples and
    Rds(on) scalar the Tier-2 producer uses; the netlist builder then applies
    its IS/N/RS extraction exactly once.  The returned lines are lifted from
    that generated subcircuit instead of re-deriving the diode math here.
    """
    from models.netlist_builder import build_tier2_subcircuit
    from pipeline.fitting import build_pwl_spec

    spec = build_pwl_spec(model.vendor)
    text = build_tier2_subcircuit(
        spec, subckt_name=f"T3_{model.vendor.upper()}")
    lines = [ln.strip() for ln in text.splitlines()
             if ln.startswith(".MODEL D_BD_")
             or ln.startswith(".MODEL D_IDEAL")
             or ln.startswith("D_BODY ")]
    if len(lines) < 2:
        raise RuntimeError(
            f"{model.vendor}: Tier-2 body-diode lines unavailable")
    return lines


def _gate_charge_expr(model: TPWLTerminalModel) -> str:
    """Monotone ``C_QG gi ss Q={table(...)}`` from the DPT charge table."""
    qv, qq = model.qg_v, model.qg_q
    if qv is None or qq is None:
        raise ValueError(
            f"{model.vendor}: DPT gate-charge table is unavailable; "
            "refusing to emit C_QG without measured charge")
    qv = np.asarray(qv, dtype=float)
    qq = np.asarray(qq, dtype=float)
    if qv.size < 2 or qq.size != qv.size:
        raise ValueError(
            f"{model.vendor}: gate-charge table has {qv.size} voltage "
            f"point(s) and {qq.size} charge point(s); need >=2 equal")
    if np.any(np.diff(qv) <= 0.0):
        raise ValueError(
            f"{model.vendor}: gate-charge voltage axis is not strictly "
            "increasing")
    if np.any(np.diff(qq) < 0.0):
        raise ValueError(
            f"{model.vendor}: gate-charge table is non-monotone; refusing "
            "to emit a negative capacitance")
    pairs = ",".join(f"{_f(x)},{_f(y)}" for x, y in zip(qv, qq))
    return (f"table(limit(V(gi,ss),{_f(qv[0])},{_f(qv[-1])}),{pairs})")


def emit_subcircuit(model: TPWLTerminalModel, *,
                    name: str | None = None,
                    policy: EmissionPolicy | None = None) -> str:
    """Return a complete ``.subckt`` string for one fitted model.

    ``policy=None`` keeps the closed v1 emission (the measured per-vendor
    channel/Qoss policy, no internal nodes).  ``policy=V2_POLICY`` adds the
    Tier-2 parasitics, body diode and DPT gate charge; the channel and Qoss
    tables then attach to the internal ``gi/ss/di`` nodes.  A compact policy
    prunes the common Vds axis only; every row uses that same axis.
    """
    _registered_vendor(model)
    pol = emission_policy(model) if policy is None else policy
    if ((pol.include_gate_charge or pol.include_body_diode)
            and not pol.include_parasitics):
        raise ValueError(
            f"{model.vendor}: gate charge / body diode need the internal "
            "nodes from include_parasitics")
    name = name or f"{model.vendor.upper()}_TPWL_DEIM"
    ports = [_PORT.get(r, r) for r in model.pin_roles]
    g, d, table, _ = _emission_channel(model, pol)
    tv, tf = transfer_curve(model)
    if tv.size < 2:
        raise ValueError(
            f"{model.vendor}: curve2 transfer characteristic has "
            f"{tv.size} point(s); need >=2")

    d_node = "di" if pol.include_parasitics else "d"
    s_node = "ss" if pol.include_parasitics else "s"
    g_node = "gi" if pol.include_parasitics else "g"
    vds_expr = f"V({d_node},{s_node})"
    vgs_expr = f"V({g_node},{s_node})"

    def rows(vals: np.ndarray) -> str:
        return ",".join(
            f"{_f(vgs)},table({vds_expr}," + ",".join(
                f"{_f(x)},{_f(y)}" for x, y in zip(d, row)) + ")"
            for vgs, row in zip(g, vals))

    channel_table = (f"table(limit({vgs_expr},{_f(g[0])},{_f(g[-1])}),"
                     + rows(table) + ")")
    gate_table = (f"table(limit({vgs_expr},{_f(tv[0])},{_f(tv[-1])}),"
                  + ",".join(f"{_f(x)},{_f(y)}" for x, y in zip(tv, tf)) + ")")
    vsat = saturation_scale(model)
    x = (f"({gate_table})*({channel_table})"
         f"*{vds_expr}/({vds_expr}+{_f(vsat)})")
    if pol.soft_clamp:
        channel = hyp_expr(x, pol.soft_eps)
    else:
        channel = f"max(0,{x})"

    lines = [f".subckt {name} {' '.join(ports)}"]
    if pol.include_parasitics:
        lines.append(f"RG g gi {_f(_R_GATE_OHM)}")
        lines.append(f"RD d di {_f(_R_DRAIN_OHM)}")
        lines.append(f"RS s ss {_f(_R_SOURCE_OHM)}")
    if model.qoss_surface is not None or (
            model.coss_v is not None and model.coss_c is not None):
        lines.append(f"C{name}_QOSS {d_node} {s_node} Q={{"
                     f"{_qoss_expr(model, policy=pol, vds_expr=vds_expr)}}}")
    lines.append(f"G_CH {d_node} {s_node} VALUE={{{channel}}}")
    if pol.include_gate_charge:
        lines.append(f"C{name}_QG gi ss Q={{{_gate_charge_expr(model)}}}")
    if pol.include_body_diode:
        lines.extend(_tier2_body_diode_lines(model))
    lines.append(".ends")
    return "\n".join(lines) + "\n"
