"""M3.6: fitting pipeline stage (LTspice backend).

Consumes extracted-curve outputs from `pipeline.curves`, assembles a
PWLModelSpec, and synthesizes the Tier-2 subcircuit .lib via
models.netlist_builder. Stepped-OP validation runs through the spicelib
engine.

Entry point:
    from pipeline.fitting import fit_pwl_model
    result = fit_pwl_model("csd16412q5a")
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict

import numpy as np

from common.types import (
    PWLModelSpec,
    UNIFIED_TUNING_IDENTITY,
    UNIFIED_TUNING_PROVENANCE,
    trim_trailing_invalid,
)
from config.yaml_io import load_input, load_output
from pipeline.curves import _cap_ref_vds
from simulation.engine import DCDirective, SpiceEngine
from simulation.netlist_text import NetlistBuilder


_LOGGER = logging.getLogger(__name__)

# Fitting tolerances (plan §2.1: typically <1% RMSE; 2% per-curve leaves
# margin for the <5% end-to-end goal).
_FIT_REL_TOL = 0.02
_MAX_SEGMENTS = 6


def _curve(vendor: str, tb: str) -> Dict:
    return load_output(vendor, tb)


def _arr(xs) -> np.ndarray:
    return np.asarray(xs, dtype=float)


def _fit_transfer(vendor: str, cond: dict):
    """Curve 2 -> conducting-region vgs/id + plateau estimate."""
    d = _curve(vendor, "curve2_transfer_characteristic")["extracted"]
    vgs = _arr(d["curve_vgs"])
    i_d = _arr(d["curve_id"])
    if vgs.size != i_d.size or vgs.size == 0:
        raise ValueError(f"{vendor}: curve2 has no usable Vgs/Id samples")
    mask = i_d > max(1e-7, 0.001 * i_d.max())
    if not np.any(mask):
        raise ValueError(f"{vendor}: curve2 has no conducting region")
    return vgs[mask], i_d[mask], float(d.get("v_plateau_v", vgs[mask][0]))


def _tuning_bounds(vds_cap_max: float, c_rss_lin: float) -> Dict[str, tuple]:
    """Per-field theta brackets from source data, not fitted WOL constants.

    The multiplier bracket is a one-octave project decision around the
    mathematical identity (see UNIFIED_TUNING_PROVENANCE). The O_GD bracket
    is capped at one full Q_RSS at the top of the measured Coss sweep,
    ``c_rss_lin * Vds_cap_max``: both source values are read from this
    vendor's curve-4 extraction and config YAML. WOL Table 3 final numbers
    are scale hints used only to check the bracket, never defaults here.
    """
    qgd_ref = max(float(c_rss_lin) * float(vds_cap_max), float(c_rss_lin))
    if not math.isfinite(qgd_ref) or qgd_ref <= 0.0:
        raise ValueError("cannot derive a positive O_GD bound from curve 4")
    return {
        "A_GD": (0.5, 2.0),
        "O_GD": (0.0, qgd_ref),
        "A_GS": (0.5, 2.0),
        "A_RGI": (0.5, 2.0),
    }


def build_pwl_spec(vendor: str) -> PWLModelSpec:
    """Assemble the producer spec (curves -> scalars -> metadata).

    Additive T3 change: the spec now carries the T0 ``theta`` /
    ``theta_bounds`` / ``urom_provenance`` metadata keys. Every curve-derived
    scalar and breakpoint is exactly the closed Tier-2 value, so the emitted
    netlist is unchanged (the builder ignores unknown metadata keys).
    """
    cfg = load_input(vendor)
    device = cfg["device"]
    cond = cfg["conditions"]

    # ---- Curve 1: output family -> shared Vds grid + Id rows ----------------
    c1 = _curve(vendor, "curve1_output_characteristics")["extracted"]
    prefix = "curve_id_vgs_"
    vgs_values = sorted(
        float(k[len(prefix):-2])
        for k in c1 if k.startswith(prefix))
    rows = []
    ref_len = None
    for v in vgs_values:
        key = next(k for k in c1 if k.startswith("curve_id_vgs_")
                   and abs(float(k[len("curve_id_vgs_"):-2]) - v) < 1e-9)
        y = _arr(c1[key])
        xs = np.linspace(0.0, float(cond["Vds_sweep_max"]), len(y))
        xs, y = trim_trailing_invalid(xs, y)
        if ref_len is None:
            ref_len = len(y)
        rows.append(np.interp(
            np.linspace(0, float(cond["Vds_sweep_max"]), ref_len), xs, y))
    if ref_len is None or ref_len < 2:
        raise ValueError(f"{vendor}: curve1 has no usable Vgs family")
    breakpoints_vds = list(np.linspace(
        0.0, float(cond["Vds_sweep_max"]), ref_len))
    breakpoints_id = [list(map(float, r)) for r in rows]

    # ---- Curve 3: Rds(on) scalar ---------------------------------------------
    rds_on = 0.005
    try:
        c3 = _curve(vendor, "curve3_rds_on")["extracted"]
        rx = _arr(c3["curve_id"])
        ry = _arr(c3["curve_rds_mohm"]) * 1e-3
        if len(rx):
            pos = ry[ry > 0]
            if pos.size:
                rds_on = float(np.median(pos))
            else:
                _LOGGER.warning(
                    "%s: curve3 has no positive Rds samples; "
                    "0.005 Ohm fallback kept", vendor)
    except FileNotFoundError:
        _LOGGER.warning("%s: curve3 missing; Rds(on) fallback used", vendor)

    # ---- Curve 4: Qoss table via charge integration ----------------------------
    cq = ce = None
    qoss_v: list = []
    qoss_q: list = []
    try:
        # Q(Vref)/Vref is zero AT the sweep minimum by construction. Use the
        # same reference as curve extraction (configured point or a quarter
        # of the sweep span); otherwise cq/ce collapse to 0 and the metadata
        # silently falls back to hardcoded 4.5 nF / 7.5 pF.
        cap_ref = _cap_ref_vds(cond)
        from simulation.testbenches.capacitance import CapacitanceTestbench
        lib = Path(cfg["library"]).resolve()
        _, coss, _ = CapacitanceTestbench(
            f_meas=cond.get("F_meas")).run(
            lib, cfg["subcircuit"],
            vds_min=cond["Vds_cap_min"], vds_max=cond["Vds_cap_max"],
            points=cond["Cap_points"],
            pin_roles=cfg.get("pins", {}).get("pin_order"))
        from approximation.charge_model import integrate_charge
        cm = integrate_charge(_arr(coss.x_data), _arr(coss.y_data),
                              v_ref=cap_ref)
        qoss_v = cm.v.tolist()
        qoss_q = cm.q.tolist()
        cq, ce = cm.c_eq_charge, cm.c_eq_energy
    except FileNotFoundError:
        _LOGGER.warning("%s: curve4 missing; Qoss table skipped", vendor)

    # ---- Curve 7: Vth -----------------------------------------------------------
    v_th = 0.0
    try:
        v_th = float(load_output(vendor,
                                 "curve7_threshold_voltage")
                     ["extracted"]["vth_v"])
    except FileNotFoundError:
        pass

    # ---- Curve 6: body diode ------------------------------------------------------
    b_vsd: list = []
    b_isd: list = []
    try:
        c6 = load_output(vendor, "curve6_body_diode")["extracted"]
        xv, yv = trim_trailing_invalid(_arr(c6["curve_vsd"]),
                                       np.abs(_arr(c6["curve_isd"])))
        b_vsd, b_isd = xv.tolist(), yv.tolist()
    except FileNotFoundError:
        pass

    c_iss_lin = (cq if cq is not None else 1500e-12) * 3.0
    c_oss_lin = cq if cq is not None else 500e-12
    c_rss_lin = (ce if ce is not None else 50e-12) * 0.15
    theta = dict(UNIFIED_TUNING_IDENTITY)
    meta = {
        "r_gate": 1.5,
        "r_source": 1e-4,
        "r_drain": 1e-4,
        "c_iss_lin": c_iss_lin,
        "c_oss_lin": c_oss_lin,
        "c_rss_lin": c_rss_lin,
        "theta": dict(theta),
        "urom_tuning_theta": dict(theta),
        "theta_bounds": _tuning_bounds(float(cond["Vds_cap_max"]), c_rss_lin),
        "urom_provenance": {
            **UNIFIED_TUNING_PROVENANCE,
            "r_gate": "1.5 Ohm documented emitter fallback "
                      "(models/netlist_builder.py:_META_DEFAULTS); "
                      "project_decision unchanged from closed Tier 2",
            "r_source": "1e-4 Ohm closed Tier-2 producer scalar",
            "r_drain": "1e-4 Ohm closed Tier-2 producer scalar",
            "c_iss_lin": "curve4 c_eq_charge * 3.0; closed Tier-2 rule "
                         "(AGENTS.md Traps), 1500 pF fallback when absent",
            "c_oss_lin": "curve4 c_eq_charge; 500 pF fallback when absent",
            "c_rss_lin": "curve4 c_eq_energy * 0.15; closed Tier-2 rule, "
                         "50 pF fallback when absent",
            "Vds_cap_max": f"config/inputs/{vendor}.yaml:conditions.Vds_cap_max",
        },
    }
    spec = PWLModelSpec(
        device_name=device,
        tier=2,
        breakpoints_vds=breakpoints_vds,
        breakpoints_id=breakpoints_id,
        vgs_values=[float(v) for v in vgs_values],
        q_oss_vs_v=qoss_v,
        q_oss_charge_c=qoss_q,
        rds_on=rds_on,
        v_th=v_th,
        body_diode_vsd=b_vsd,
        body_diode_isd=b_isd,
        metadata=meta,
    )
    return spec


def fit_pwl_model(
    vendor: str,
    rel_tol: float = _FIT_REL_TOL,
    out_dir: str | Path = "reports/models",
) -> Path:
    """Fit a Tier-2 PWL model for *vendor* and emit its subcircuit .lib."""
    spec = build_pwl_spec(vendor)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    from models.netlist_builder import build_tier2_subcircuit
    netlist = build_tier2_subcircuit(spec)

    lib_out = out_path / f"{spec.device_name}_PIECESPICE_T2.lib"
    lib_out.write_text(netlist, encoding="ascii")
    _LOGGER.info("%s: Tier-2 model written to %s", vendor, lib_out)
    return lib_out


def compare_tier2_vs_golden(vendor: str, lib_path: Path | None = None):
    """Endpoint-current comparison at Vds=10 V per Vgs curve (LTspice)."""

    if lib_path is None:
        cfg = load_input(vendor)
        device = cfg["device"]
        lib_path = Path("reports/models") / f"{device}_PIECESPICE_T2.lib"
    lib_path = Path(lib_path)
    name = lib_path.stem

    c1 = load_output(vendor, "curve1_output_characteristics")["extracted"]
    prefix = "curve_id_vgs_"
    pairs = sorted((float(k[len(prefix):-2]), k)
                   for k in c1 if k.startswith(prefix))

    engine = SpiceEngine()
    from simulation.netlist_text import NetlistBuilder
    rows = []
    worst = 0.0
    for vgs_val, key in pairs:
        golden = abs(float(c1[key][-1]))
        nb = NetlistBuilder(f"cmp_{vgs_val}")
        nb.include(lib_path)
        nb.v_dc("ds", "drain", "0", 10.0)
        nb.v_dc("gs", "gate", "0", vgs_val)
        nb.x_instance("m1", name, ["drain", "gate", "0"])
        result = engine.run_op(nb.render())
        i_t2 = abs(float(result.traces["I(Vds)"][0]))
        rel = abs(i_t2 - golden) / max(golden, 1e-9) * 100
        worst = max(worst, rel)
        rows.append((vgs_val, golden, i_t2, rel))
    return rows, worst


def compare_tier2_vs_golden_full(vendor: str, lib_path: Path | None = None,
                                 n_points: int = 41, *,
                                 subckt_name: str | None = None):
    """Full-sweep Tier-2 vs golden comparison across every Vgs curve.

    subckt_name overrides the instance name when the lib stem differs from
    its .SUBCKT header (the unified <vendor>_PIECSPICE_UROM.lib case).
    None keeps the historical lib-stem behaviour.
    """
    cfg = load_input(vendor)
    cond = cfg["conditions"]

    if lib_path is None:
        lib_path = Path("reports/models") / \
            f"{cfg['device']}_PIECESPICE_T2.lib"
    lib_path = Path(lib_path)
    name = subckt_name or lib_path.stem

    c1 = load_output(vendor, "curve1_output_characteristics")["extracted"]
    prefix = "curve_id_vgs_"
    keys = sorted((float(k[len(prefix):-2]), k)
                  for k in c1 if k.startswith(prefix))

    engine = SpiceEngine()
    from simulation.netlist_text import NetlistBuilder
    vds_grid = np.linspace(0.0, float(cond["Vds_sweep_max"]), n_points)

    rows = []
    for vgs_val, key in keys:
        y_golden = np.asarray(c1[key], dtype=float)
        g_x = np.linspace(0.0, float(cond["Vds_sweep_max"]), len(y_golden))
        y_golden_g = np.interp(vds_grid, g_x, y_golden)

        i_t2 = np.zeros(n_points)
        nb = NetlistBuilder(f"t2_{vgs_val:g}")
        nb.include(lib_path)
        nb.v_dc("gs", "gate", "0", vgs_val)
        nb.v_dc("sweep_src", "vdswp", "0", 0.0)
        nb.r("load", "vdswp", "d", 1e-6)   # near-short: d tracks vdswp
        nb.r("rsense", "s", "0", 1e-6)    # give source a DC return path
        nb.x_instance("m1", name, ["d", "gate", "s"])
        directive = DCDirective(source_name="Vsweep_src",
                                start=0.0,
                                stop=float(cond["Vds_sweep_max"]),
                                step=float(cond["Vds_sweep_max"])
                                / (n_points - 1))
        # Rename source for sweep semantics: use a dedicated V element.
        text = nb.render().replace("Vsweep_src", "Vds_sweep_el")
        directive.source_name = "Vds_sweep_el"
        result = engine.run_dc_sweep(text, directive)

        vd_actual = np.asarray(result.get("V(d)")) \
            if "V(d)" in result.traces else \
            np.asarray(result.get(result.sweep_var))
        i_raw = np.asarray(result.get("I(Vds_sweep_el)"))
        i_t2 = np.abs(i_raw)

        order = np.argsort(vd_actual)
        i_t2 = np.interp(vds_grid, vd_actual[order], i_t2[order])

        rows.append((vgs_val, vds_grid.copy(), y_golden_g, i_t2))
    return rows
