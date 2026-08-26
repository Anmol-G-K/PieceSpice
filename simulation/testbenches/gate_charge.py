"""Gate charge testbench: Vgs vs Qg for Curve 5.

Quasi-static reconstruction (Option A) from validated Curve 2 (transfer)
and Curve 4 (Ciss/Crss vs Vds) data. The raw constant-current transient is
structurally broken on MODPEX sense-and-mirror vendor models (VM-02): the
EV16/CAP/FI1/FI2 network pumps amp-level junk into the gate and Vgs
collapses to -37 V, making Qg = int(Ig dt) unrecoverable in any engine.
This module is backend-independent (pure Python over extracted curves);
the spicelib/LTspice migration changes nothing about it.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from common.types import CurveData


def _interp_curve(curve: CurveData, x: np.ndarray) -> np.ndarray:
    """Interpolate a CurveData family onto x (clamped at the ends)."""
    xs = np.asarray(curve.x_data, dtype=float)
    ys = np.asarray(curve.y_data, dtype=float)
    if not np.all(np.diff(xs) >= 0):
        if np.all(np.diff(xs[::-1]) >= 0):
            xs, ys = xs[::-1], ys[::-1]
        else:
            order = np.argsort(xs, kind="stable")
            xs, ys = xs[order], ys[order]
            print(f"[gate_charge] WARNING: curve {curve.name!r} x-axis was "
                  f"not monotonic; points re-sorted before interpolation.")
    return np.interp(x, xs, ys)


def _crss_integral(crss_curve: CurveData, v_hi: float, v_lo: float,
                   n_points: int = 200) -> tuple[float, np.ndarray, np.ndarray]:
    """Integrate Crss(Vds) dVds from v_hi down to v_lo (descending axis)."""
    vds = np.linspace(v_hi, v_lo, n_points)
    crss = _interp_curve(crss_curve, vds)
    dq = -0.5 * (crss[1:] + crss[:-1]) * np.diff(vds)
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
    qgd_cal_nC: float | None = None,
) -> Tuple[pd.DataFrame, dict]:
    """Quasi-static gate-charge reconstruction (Curve 5, Option A).

    Three-segment datasheet model:
      Qgs : integrate Ciss(Vbus) over Vgs from 0 to plateau Vgp where
            Id(Vgs) = Iload.
      Qgd : Miller plateau - integrate Crss(Vds) over the drain swing from
            Vbus down to Vds,on at Vgs ~ Vgp.
      tail: integrate Ciss(Vds,on) from Vgp to Vgs_max.

    qgd_cal_nC rescales the Miller segment to a datasheet target (VM-03:
    MODPEX models match small-signal Crss but under-integrate large-signal
    gate-drain charge); the scale factor is reported as 'qgd_cal_scale'.
    """
    vgs_t = np.asarray(transfer_curve.x_data, dtype=float)
    id_t = np.abs(np.asarray(transfer_curve.y_data, dtype=float))

    if not (id_t.min() <= i_load <= id_t.max()):
        raise ValueError(
            f"Iload={i_load} A outside transfer-curve range "
            f"[{id_t.min():.3g}, {id_t.max():.3g}] A")
    v_plateau = float(np.interp(i_load, id_t, vgs_t))

    plateau_clipped = v_plateau > v_gs_max
    if plateau_clipped:
        print(f"[gate_charge] WARNING: required Miller plateau "
              f"Vgp={v_plateau:.2f} V exceeds v_gs_max={v_gs_max} V; "
              f"folding Miller segment at final gate voltage so Qg(Vgs) "
              f"stays monotonic.")
        v_plateau = v_gs_max

    ciss_high = _interp_curve(ciss_curve, np.array([v_bus]))[0]
    ciss_low = _interp_curve(ciss_curve, np.array([v_ds_on]))[0]

    # Segment 1: Qgs
    vgs_seg1 = np.linspace(0.0, v_plateau, n_points)
    q_seg1 = ciss_high * vgs_seg1

    # Segment 2: Qgd (Miller), with optional datasheet calibration
    qgd_total, _vds_seg2, cum_qgd = _crss_integral(
        crss_curve, v_bus, v_ds_on, n_points)
    if qgd_cal_nC is not None and qgd_cal_nC > 0 and qgd_total > 0:
        cal_scale = (qgd_cal_nC * 1e-9) / qgd_total
        cum_qgd = cum_qgd * cal_scale
        qgd_total *= cal_scale
    else:
        cal_scale = 1.0

    q_seg2 = q_seg1[-1] + cum_qgd
    if plateau_clipped:
        vgs_seg2_v = np.full(n_points, v_gs_max)
    else:
        vgs_seg2_v = np.full(n_points, v_plateau)

    # Segment 3: tail
    if plateau_clipped:
        qg_c = np.concatenate([q_seg1, q_seg2[1:]])
        vgs_full = np.concatenate([vgs_seg1, vgs_seg2_v[1:]])
        q_tail = 0.0
    else:
        vgs_seg3 = np.linspace(v_plateau, v_gs_max, n_points)
        q_seg3 = q_seg2[-1] + ciss_low * (vgs_seg3 - v_plateau)
        qg_c = np.concatenate([q_seg1, q_seg2[1:], q_seg3[1:]])
        vgs_full = np.concatenate([vgs_seg1, vgs_seg2_v[1:], vgs_seg3[1:]])
        q_tail = q_seg3[-1] - q_seg2[-1]

    df = pd.DataFrame({"Qg_nC": qg_c * 1e9, "Vgs_V": vgs_full})
    df = df.drop_duplicates(subset="Qg_nC", keep="last").reset_index(drop=True)

    breakpoints = {
        "v_plateau_V": v_plateau,
        "q_gs_nC": q_seg1[-1] * 1e9,
        "q_gd_nC": qgd_total * 1e9,
        "q_tail_nC": q_tail * 1e9,
        "q_total_nC": float(qg_c[-1]) * 1e9,
        "qgd_cal_scale": cal_scale,
        "plateau_clipped": plateau_clipped,
    }
    return df, breakpoints


def plot_gate_charge(df: pd.DataFrame, breakpoints: dict, save_path=None,
                     device_name: str = ""):
    """Plot reconstructed Vgs vs Qg with segment annotations."""
    import matplotlib.pyplot as plt
    from common.plotting import setup_style

    setup_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8), layout="constrained")

    vp = breakpoints["v_plateau_V"]
    q1 = breakpoints["q_gs_nC"]
    q2 = q1 + breakpoints["q_gd_nC"]

    ax.plot(df["Qg_nC"], df["Vgs_V"], color="tab:blue", lw=2,
            label="Qg(Vgs) quasi-static")
    ax.axhline(vp, color="tab:red", ls="--", lw=1,
               label=f"Miller plateau Vgp = {vp:.2f} V")
    ax.axvline(q1, color="tab:green", ls=":", lw=1,
               label=f"Qgs = {q1:.1f} nC")
    ax.axvline(q2, color="tab:orange", ls=":", lw=1,
               label=f"Qgs+Qgd = {q2:.1f} nC")

    for q, lbl in [(q1, "Qgs"), (q2, "Qgs+Qgd"),
                   (breakpoints["q_total_nC"],
                    f"Qg@max = {breakpoints['q_total_nC']:.1f} nC")]:
        ax.annotate(lbl, xy=(q, vp), xytext=(q, vp + 0.8),
                    fontsize=11, ha="center",
                    arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_xlabel("Gate Charge Qg [nC]")
    ax.set_ylabel("Gate-Source Voltage Vgs [V]")
    ax.set_title(f"{device_name + ' - ' if device_name else ''}"
                 f"Gate Charge (Quasi-Static Reconstruction)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    from matplotlib.ticker import EngFormatter
    ax.xaxis.set_major_formatter(EngFormatter(unit="nC"))
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig, ax
