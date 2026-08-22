"""Gate charge testbench: Vgs vs Qg for Curve 5.

The raw constant-current transient on the IRFS7730PBF vendor subcircuit is
BLOCKED by a MODPEX sense-and-mirror pathology (see
gate_charge_error_analysis.md): under dynamic drain transitions the EV16/CAP/
FI1/FI2 network pumps amp-level junk current into the gate node and Vgs
collapses to -37 V, making Qg = int(Ig dt) unrecoverable. This is a structural
model defect, not a numerical one.

Instead, this module reconstructs Qg(Vgs) quasi-statically (Option A) from the
already-validated Curve 2 (transfer) + Curve 4 (Ciss/Crss vs Vds) data using
the standard three-segment datasheet model. See `reconstruct_gate_charge`.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

from common.types import CurveData
from parser.preprocessor import normalize_pspice_netlist
from simulation.engine import SpiceEngine
def _interp_curve(curve: CurveData, x: np.ndarray) -> np.ndarray:
    """Interpolate a CurveData family onto x (clamped at the ends)."""
    xs = np.asarray(curve.x_data, dtype=float)
    ys = np.asarray(curve.y_data, dtype=float)
    if xs[0] > xs[-1]:  # np.interp requires ascending x
        xs, ys = xs[::-1], ys[::-1]
    return np.interp(x, xs, ys)


def _crss_integral(crss_curve: CurveData, v_hi: float, v_lo: float,
                   n_points: int = 200) -> tuple[float, np.ndarray, np.ndarray]:
    """Integrate Crss(Vds) dVds from v_hi down to v_lo (descending axis kept).

    Returns (total_charge_C, vds_descending, cumulative_charge_at_each_point).
    """
    vds = np.linspace(v_hi, v_lo, n_points)
    crss = _interp_curve(crss_curve, vds)
    dq = -0.5 * (crss[1:] + crss[:-1]) * np.diff(vds)  # positive as Vds falls
    cum = np.concatenate([[0.0], np.cumsum(dq)])
    return float(cum[-1]), vds, cum


def reconstruct_gate_charge(
    transfer_curve: CurveData,
    ciss_curve: CurveData,
    crss_curve: CurveData,
    v_bus: float = 40.0,
    i_load: float = 50.0,
    v_gs_max: float = 10.0,
    v_ds_on: float = 0.1,
    n_points: int = 200,
) -> Tuple[pd.DataFrame, dict]:
    """Quasi-static gate-charge reconstruction (Curve 5, Option A).

    Builds Qg(Vgs) analytically from validated Curve 2 (transfer) and Curve 4
    (capacitance) data using the standard three-segment model:

      Segment 1 - Qgs: integrate Ciss(Vds = Vbus) over Vgs from 0 to the
                   plateau voltage Vgp, where Id(Vgs) = Iload.
      Segment 2 - Qgd: Miller plateau. Integrate Crss(Vds) over the drain
                   swing from Vbus down to Vds,on while Vgs stays ~Vgp.
      Segment 3 - tail: integrate Ciss(Vds ~= Vds,on) from Vgp to Vgs_max.

    Assumes weak Vgs-dependence of the terminal capacitances (the standard
    industry hand-calculation assumption); requires no transient simulation.

    Args:
        transfer_curve: Curve 2 output (Id vs Vgs at fixed high Vds).
        ciss_curve:     Curve 4 Ciss vs Vds.
        crss_curve:     Curve 4 Crss vs Vds.
        v_bus:          Bus voltage / initial drain voltage [V].
        i_load:         Load current defining the plateau [A].
        v_gs_max:       Final gate voltage [V].
        v_ds_on:        Residual drain voltage after collapse [V].

    Returns:
        df: DataFrame with columns [Qg_nC, Vgs_V] in datasheet format.
        breakpoints: dict with keys 'v_plateau_V', 'q_gs_nC', 'q_gd_nC',
                     'q_tail_nC', 'q_total_nC'.
    """
    # --- Plateau voltage: Vgs where Id(Vgs) crosses Iload ---
    vgs_t = np.asarray(transfer_curve.x_data)
    id_t = np.asarray(transfer_curve.y_data)
    if not (id_t.min() <= i_load <= id_t.max()):
        raise ValueError(
            f"Iload={i_load} A outside transfer-curve range "
            f"[{id_t.min():.3g}, {id_t.max():.3g}] A"
        )
    v_plateau = float(np.interp(i_load, id_t, vgs_t))
    v_plateau = min(v_plateau, v_gs_max)

    # --- Capacitances at operating points ---
    # Pre-plateau: drain still at Vbus; post-plateau: drain collapsed.
    ciss_high = _interp_curve(ciss_curve, np.array([v_bus]))[0]
    ciss_low = _interp_curve(ciss_curve, np.array([v_ds_on]))[0]

    # --- Segment 1: Qgs = int Ciss(vbus) dVgs, 0 -> Vgp ---
    vgs_seg1 = np.linspace(0.0, v_plateau, n_points)
    q_seg1 = ciss_high * vgs_seg1  # [C], constant-C approximation per point

    # --- Segment 2: Qgd = int Crss(Vds) dVds, Vbus -> Vds_on (drain swing) ---
    qgd_total, vds_seg2, cum_qgd = _crss_integral(crss_curve, v_bus, v_ds_on,
                                                  n_points)
    q_seg2 = q_seg1[-1] + cum_qgd

    # --- Segment 3: tail = int Ciss(low-Vds) dVgs, Vgp -> Vgs_max ---
    vgs_seg3 = np.linspace(v_plateau, v_gs_max, n_points)
    q_seg3 = q_seg2[-1] + ciss_low * (vgs_seg3 - v_plateau)

    # --- Assemble datasheet-format table ---
    qg_c = np.concatenate([q_seg1, q_seg2[1:], q_seg3[1:]])
    vgs_full = np.concatenate([vgs_seg1, np.full(n_points - 1, v_plateau),
                               vgs_seg3[1:]])
    df = pd.DataFrame({"Qg_nC": qg_c * 1e9, "Vgs_V": vgs_full})
    # Collapse duplicate plateau rows to one for a clean single-value plateau
    df = df.drop_duplicates(subset="Qg_nC", keep="last").reset_index(drop=True)

    breakpoints = {
        "v_plateau_V": v_plateau,
        "q_gs_nC": q_seg1[-1] * 1e9,
        "q_gd_nC": qgd_total * 1e9,
        "q_tail_nC": (q_seg3[-1] - q_seg2[-1]) * 1e9,
        "q_total_nC": q_seg3[-1] * 1e9,
    }
    return df, breakpoints


def plot_gate_charge(df: pd.DataFrame, breakpoints: dict, save_path=None):
    """Plot the reconstructed Vgs vs Qg curve with the three segments labeled."""
    import matplotlib.pyplot as plt
    from common.plotting import setup_style

    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6.75))

    vp = breakpoints["v_plateau_V"]
    q1, q2 = breakpoints["q_gs_nC"], breakpoints["q_gs_nC"] + breakpoints["q_gd_nC"]

    ax.plot(df["Qg_nC"], df["Vgs_V"], color="tab:blue", lw=2,
            label="Qg(Vgs) quasi-static")
    ax.axhline(vp, color="tab:red", ls="--", lw=1,
               label=f"Miller plateau Vgp = {vp:.2f} V")
    ax.axvline(q1, color="tab:green", ls=":", lw=1,
               label=f"Qgs = {q1:.1f} nC")
    ax.axvline(q2, color="tab:orange", ls=":", lw=1,
               label=f"Qgs+Qgd = {q2:.1f} nC")

    for q, lbl in [(q1, "Qgs"), (q2, "Qgs+Qgd"),
                   (breakpoints["q_total_nC"], f"Qg@10V = {breakpoints['q_total_nC']:.1f} nC")]:
        ax.annotate(lbl, xy=(q, vp), xytext=(q, vp + 0.8),
                    fontsize=11, ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_xlabel("Gate Charge Qg [nC]")
    ax.set_ylabel("Gate-Source Voltage Vgs [V]")
    ax.set_title("Reconstructed Gate Charge Curve (IRFS7730PBF)")
    # STRICT RULE: legend at EAST outside plotting area.
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig, ax


def run_gate_charge(
    lib_path: Path,
    subcircuit_name: str,
    v_bus: float = 40.0,
    i_load: float = 50.0,
    i_gate: float = 5e-3,
    v_gs_max: float = 10.0,
    t_end: float = 120e-6,
    step_time: float = 20e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the gate-charge transient under datasheet conditions.

    A constant-current load (i_load into the drain) plus a rail-clamp diode
    hold the device at its rated load current: an ideal voltage source on the
    drain would let the macromodel conduct thousands of amps, driving its
    internal capacitance-sense network unstable.
    """
    """Run the gate-charge transient.

    Returns:
        q_g: Gate charge array [C]
        v_gs: Gate-source voltage array [V]
        v_ds: Drain-source voltage array [V]
    """
    raw = lib_path.read_text(encoding='utf-8', errors='ignore')
    temp_lib = lib_path.parent / f"_temp_norm_{lib_path.name}"
    temp_lib.write_text(normalize_pspice_netlist(raw))

    try:
        circuit = Circuit("Gate_Charge")
        circuit.include(str(temp_lib.resolve()))
        circuit.X("M1", subcircuit_name, "d", "g", circuit.gnd)
        # Drain rail + clamp: before turn-on the load current charges the
        # drain until the clamp returns it to the rail (~Vbus); after turn-on
        # it flows through the channel, holding Id = i_load during the Miller
        # plateau.
        circuit.V("bus", "vbus", circuit.gnd, float(v_bus)@u_V)
        circuit.model('DCLAMP', 'D', IS=1e-10)
        circuit.D("clamp", "d", "vbus", model='DCLAMP')
        circuit.I("load", circuit.gnd, "d", float(i_load)@u_A)
        circuit.PieceWiseLinearCurrentSource(
            "drive", "g", circuit.gnd,
            values=[
                (0.0@u_s, 0.0@u_A),
                (1e-9@u_s, float(i_gate)@u_A),
                (float(t_end)@u_s, float(i_gate)@u_A),
            ],
        )

        engine = SpiceEngine()
        analysis = engine.run_transient(circuit, step_time=step_time, end_time=t_end)

        time = np.array(analysis.time)
        v_gs = np.array(analysis.nodes['g'])
        v_ds = np.array(analysis.nodes['d'])
        q_g = i_gate * time

        # Truncate at first crossing of v_gs_max
        idx = np.argmax(v_gs >= v_gs_max)
        if v_gs[idx] >= v_gs_max and idx > 0:
            # Linear interpolation onto the exact crossing
            t0, t1 = time[idx - 1], time[idx]
            v0, v1 = v_gs[idx - 1], v_gs[idx]
            frac = (v_gs_max - v0) / max(v1 - v0, 1e-15)
            t_cross = t0 + frac * (t1 - t0)
            keep = idx + 1
            time, v_gs, v_ds, q_g = time[:keep], v_gs[:keep], v_ds[:keep], q_g[:keep]
            q_g[-1] = i_gate * t_cross
            v_gs[-1] = v_gs_max

        return q_g, v_gs, v_ds
    finally:
        if temp_lib.exists():
            temp_lib.unlink()
