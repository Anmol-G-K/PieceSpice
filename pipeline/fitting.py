"""M3.6: fitting pipeline stage.

Consumes the extracted-curve outputs written by `pipeline.curves` for a
vendor, fits PWL parameters from them (Curves 1/2/4/6), assembles a
PWLModelSpec, and synthesizes the Tier-2 subcircuit .lib via
models.netlist_builder.

Entry point:
    from pipeline.fitting import fit_pwl_model
    result = fit_pwl_model("csd16412q5a")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

from common.types import PWLModelSpec, trim_trailing_invalid
from config.yaml_io import load_input, load_output

_LOGGER = logging.getLogger(__name__)

# Fitting tolerances (plan §2.1: typically <1% RMSE; we target 2% per-curve
# to leave margin for the <5% end-to-end goal).
_FIT_REL_TOL = 0.02
_MAX_SEGMENTS = 6


def _curve(vendor: str, tb: str) -> Dict:
    return load_output(vendor, tb)


def _arr(xs) -> np.ndarray:
    return np.asarray(xs, dtype=float)


def _fit_transfer(vendor: str, cond: dict):
    """Curve 2 -> v_th estimate + saturation endpoint currents."""
    d = _curve(vendor, "curve2_transfer_characteristic")["extracted"]
    vgs = _arr(d["curve_vgs"])
    i_d = _arr(d["curve_id"])
    # conducting region only
    mask = i_d > max(1e-7, 0.001 * i_d.max())
    return vgs[mask], i_d[mask], float(d.get("v_plateau_v", vgs[mask][0]))


def fit_pwl_model(
    vendor: str,
    rel_tol: float = _FIT_REL_TOL,
    out_dir: str | Path = "reports/models",
) -> Path:
    """Fit a Tier-2 PWL model for *vendor* and emit its subcircuit .lib.

    Returns the path of the generated .lib file.
    """
    cfg = load_input(vendor)
    device = cfg["device"]
    cond = cfg["conditions"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- Curve 1: output family -> shared Vds breakpoints + Id rows ------
    c1 = _curve(vendor, "curve1_output_characteristics")["extracted"]
    prefix = "curve_id_vgs_"
    vgs_values = sorted(
        float(k[len(prefix):-2])                # strip prefix and '_v'
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
    breakpoints_vds = list(np.linspace(0.0, float(cond["Vds_sweep_max"]), ref_len))
    breakpoints_id = [list(map(float, r)) for r in rows]

    # ---- Curve 3: Rds(on) scalar -----------------------------------------
    rds_on = 0.005
    try:
        c3 = _curve(vendor, "curve3_rds_on")["extracted"]
        rx = _arr(c3["curve_id"])
        ry = _arr(c3["curve_rds_mohm"]) * 1e-3
        if len(rx):
            rds_on = float(np.median(ry[ry > 0]))
    except FileNotFoundError:
        _LOGGER.warning("%s: curve3 missing; Rds(on) fallback used", vendor)

    # ---- Curve 4: Qoss table via charge integration -----------------------
    from approximation.charge_model import integrate_charge
    cq = ce = None
    qoss_v: list = []
    qoss_q: list = []
    try:
        c4 = _curve(vendor, "curve4_capacitances")
        # scalars only in YAML; full curves come from re-running the bench is
        # avoided by storing tables in M3.6 - interim: reconstruct from notes
        # is impossible, so read curve files saved alongside if present.
        cap_ref = float(c4["extracted"].get("cap_ref_vds", cond.get(
            "Cap_ref_Vds", 25.0)))
        # Re-run capacitance bench quickly for the Q table (cheap: ~25 pts)
        from simulation.testbenches.capacitance import CapacitanceTestbench
        from parser.netlist import resolve_pin_connections
        lib = Path(cfg["library"]).resolve()
        roles = resolve_pin_connections(lib, cfg["subcircuit"],
            pin_order_semantic=cfg.get("pins", {}).get("pin_order_semantic"))
        _, coss, _ = CapacitanceTestbench(f_meas=cond.get("F_meas")).run(
            lib, cfg["subcircuit"], vds_min=cond["Vds_cap_min"],
            vds_max=cond["Vds_cap_max"], points=cond["Cap_points"],
            pin_roles=roles)
        cm = integrate_charge(_arr(coss.x_data), _arr(coss.y_data),
                              v_ref=cap_ref)
        qoss_v = cm.v.tolist()
        qoss_q = cm.q.tolist()
        cq, ce = cm.c_eq_charge, cm.c_eq_energy
    except FileNotFoundError:
        _LOGGER.warning("%s: curve4 missing; Qoss table skipped", vendor)

    # ---- Curve 7: Vth ------------------------------------------------------
    v_th = cond.get("Id_th_target", 2.5e-4) and 0.0
    try:
        v_th = float(load_output(vendor,
                                 "curve7_threshold_voltage")["extracted"]["vth_v"])
    except FileNotFoundError:
        pass

    # ---- Curve 6: body diode ----------------------------------------------
    b_vsd: list = []
    b_isd: list = []
    try:
        c6 = load_output(vendor, "curve6_body_diode")["extracted"]
        xv, yv = trim_trailing_invalid(_arr(c6["curve_vsd"]),
                                       np.abs(_arr(c6["curve_isd"])))
        b_vsd, b_isd = xv.tolist(), yv.tolist()
    except FileNotFoundError:
        pass

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
        metadata={
            "r_gate": 1.5,
            "r_source": 1e-4,
            "r_drain": 1e-4,
            "c_iss_lin": (cq or 1500e-12) * 3.0,   # Ciss ~ 3x Coss typical
            "c_oss_lin": (cq if cq else 500e-12),
            "c_rss_lin": (ce if ce else 50e-12) * 0.15,
        },
    )

    from models.netlist_builder import build_tier2_subcircuit
    netlist = build_tier2_subcircuit(spec)

    lib_out = out_path / f"{device}_PIECESPICE_T2.lib"
    lib_out.write_text(netlist, encoding="ascii")
    _LOGGER.info("%s: Tier-2 model written to %s", vendor, lib_out)
    return lib_out


def compare_tier2_vs_golden(vendor: str, lib_path: Path | None = None):
    """Simulate the Tier-2 model at Vds=10V vs golden endpoint currents.

    Library-owned validation so notebooks only plot the returned rows.
    Returns (rows, worst_rel_pct) where rows are
    (vgs, golden_id_a, tier2_id_a, rel_err_pct) tuples.
    """
    from PySpice.Spice.Netlist import Circuit

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

    rows = []
    worst = 0.0
    for vgs_val, key in pairs:
        golden = abs(float(c1[key][-1]))
        circuit = Circuit(f"cmp_{vgs_val}")
        circuit.include(str(lib_path))
        circuit.X("M1", name, "d", "g", "gnd")
        circuit.V("ds", "d", "gnd", 10)
        circuit.V("gs", "g", "gnd", vgs_val)
        op = circuit.simulator(temperature=25,
                               nominal_temperature=25).operating_point()
        i_t2 = abs(float(op.branches["vds"][0]))
        rel = abs(i_t2 - golden) / max(golden, 1e-9) * 100
        worst = max(worst, rel)
        rows.append((vgs_val, golden, i_t2, rel))
    return rows, worst

def compare_tier2_vs_golden_full(vendor: str, lib_path: Path | None = None,
                                 n_points: int = 41):
    """Full-sweep Tier-2 vs golden comparison across every Vgs curve.

    Library-owned stepped-OP sweep (notebooks only plot the result).
    Returns rows of (vgs_val, vds_array, golden_id_array, tier2_id_array).
    """
    from PySpice.Spice.Netlist import Circuit

    cfg = load_input(vendor)
    device = cfg["device"]
    cond = cfg["conditions"]

    if lib_path is None:
        lib_path = Path("reports/models") / f"{device}_PIECESPICE_T2.lib"
    lib_path = Path(lib_path)
    name = lib_path.stem

    c1 = load_output(vendor, "curve1_output_characteristics")["extracted"]
    prefix = "curve_id_vgs_"
    keys = sorted((float(k[len(prefix):-2]), k)
                  for k in c1 if k.startswith(prefix))

    vds_grid = np.linspace(0.0, float(cond["Vds_sweep_max"]), n_points)

    rows = []
    for vgs_val, key in keys:
        y_golden = np.asarray(c1[key], dtype=float)
        g_x = np.linspace(0.0, float(cond["Vds_sweep_max"]), len(y_golden))
        y_golden_g = np.interp(vds_grid, g_x, y_golden)

        i_t2 = np.zeros(n_points)
        for j, vd in enumerate(vds_grid):
            circuit = Circuit(f"t2_{vgs_val:g}_{vd:.3g}")
            circuit.include(str(lib_path))
            circuit.X("M1", name, "d", "g", "gnd")
            circuit.V("ds", "d", "gnd", float(vd))
            circuit.V("gs", "g", "gnd", vgs_val)
            op = circuit.simulator(
                temperature=25, nominal_temperature=25).operating_point()
            i_t2[j] = abs(float(op.branches["vds"][0]))

        rows.append((vgs_val, vds_grid.copy(), y_golden_g, i_t2))
    return rows