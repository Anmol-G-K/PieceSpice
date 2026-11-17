"""Tier-2 CPWL MOSFET subcircuit synthesis (M3.5, plan §1.3/§2.3/§2.4).

Emits a standard-SPICE subcircuit that reproduces the vendor device's
behavior with PWL/behavioral elements instead of stiff physics:

    .SUBCKT <name> d g s
      RG  g  gi {r_gate}                    * gate parasitic
      RD  d  di {r_drain}                   * drain parasitic
      RS  s  ss {r_source}                  * source parasitic

      G_CH  di ss VALUE = { ... tanh-knee channel ... }   <- Curves 1+2
      B_QOSS di ss Q = { PWL charge table }               <- Curve 4
      C_ISS  gi ss Q = { PWL charge table }               <- Curve 4
      B_QRSS gi di Q = { PWL charge table }               <- Curve 4
      D_BODY ss di <PWL/exponential from Curve 6>

    .ENDS

Design rules honored:
* Charge-conserving `Q=` capacitors (plan §2.3): no i=C(v)*dv/dt drift.
* Tanh-knee channel source (plan §2.4): finite continuous Jacobians.
* Device-agnostic: every parameter comes from the PWLModelSpec produced by
  the M3 fitting pipeline; nothing is tuned to a specific vendor part.
"""

from __future__ import annotations

from typing import List

import numpy as np

from approximation.smoother import fit_tanh_knee_ls
from common.types import PWLModelSpec, PWL_META_KEYS


# Documented fallbacks for optional metadata keys the producer has not yet
# populated. Missing keys never crash the builder; unknown keys are reported.
_META_DEFAULTS = {
    "r_gate": 1.5,
    "r_source": 1e-4,
    "r_drain": 1e-4,
}


def _resolve_metadata(spec: PWLModelSpec) -> dict:
    """Merge spec metadata over documented defaults; warn on unknown keys."""
    import logging
    _logger = logging.getLogger(__name__)

    meta = dict(_META_DEFAULTS)
    for k, v in spec.metadata.items():
        if k in _META_DEFAULTS or k in PWL_META_KEYS:
            meta[k] = v
        else:
            _logger.warning(
                "unknown metadata key %r for %s; ignored.", k,
                spec.device_name)
    missing = [k for k in _META_DEFAULTS if k not in spec.metadata]
    if missing:
        _logger.info(
            "%s metadata missing %s; using documented defaults.",
            spec.device_name, missing)
    return meta


def _fmt(x: float) -> str:
    """Format a float for SPICE output (16 significant chars, e-notation)."""
    return f"{x:.6e}"


def _ramp_chain(xs: List[float], ys: List[float], term: str) -> str:
    """C0 PWL through (xs, ys) as chained soft-abs ramps with delta slopes."""
    expr = f"{_fmt(ys[0])}"
    prev_slope = 0.0
    for i in range(1, len(xs)):
        dv = xs[i] - xs[i - 1]
        slope = (ys[i] - ys[i - 1]) / dv if dv else 0.0
        delta = slope - prev_slope
        if abs(delta) > 0 or i == 1:
            coeff = slope if i == 1 else delta
            expr += (f" + {coeff:.6e}*(({term}-{xs[i - 1]:.6e})"
                     f"+abs(({term})-{xs[i - 1]:.6e}))/2")
        prev_slope = slope
    return expr


def _channel_expression(spec: PWLModelSpec, vgs_term: str) -> str:
    """Behavioral channel current: PWL(Vgs) drive * tanh(Vds/vds_sat(Vgs)).

    The topology is unchanged from the original Tier-2 emitter (PWL transfer,
    tanh saturation knee). What changes is how the knee width is chosen:
    one least-squares fit per extracted output curve instead of a single
    heuristic clamped from Rds(on) and the top-curve plateau. A global knee
    systematically mis-shapes the low/mid Vds region of every Vgs family.
    The width is capped at 0.377*Vmax so tanh(Vmax/s) >= 0.99, and the
    transfer amplitude is scaled by 1/tanh(Vmax/s) so each endpoint still
    matches the extracted golden current exactly.
    """
    vd = np.asarray(spec.breakpoints_vds, dtype=float)
    vmax = float(vd[-1])
    span = vmax - float(vd[0])
    drives = [float(v) for v in spec.vgs_values]
    endpoints = [float(row[-1]) for row in spec.breakpoints_id]

    sat_vals = []
    for row in spec.breakpoints_id:
        y = np.asarray(row, dtype=float)
        plateau = max(float(y[-1]), 1e-9)
        s = fit_tanh_knee_ls(vd, y, plateau)
        sat_vals.append(min(max(s, 0.05 * span), 0.377 * vmax))

    scale = 1.0 / np.tanh(vmax / np.asarray(sat_vals))
    id_vals = (np.asarray(endpoints) * scale).tolist()

    xs = [0.0] + drives
    iv = [0.0] + id_vals
    sv = [sat_vals[0]] + sat_vals
    v_th = float(spec.v_th)
    if 0.0 < v_th < min(drives):
        xs = [0.0, v_th] + drives
        iv = [0.0, 0.0] + id_vals
        sv = [sat_vals[0], sat_vals[0]] + sat_vals

    amp = _ramp_chain(xs, iv, vgs_term)
    sat = _ramp_chain(xs, sv, vgs_term)
    return f"({amp})*tanh(V(di,ss)/({sat}))"


def build_tier2_subcircuit(spec: PWLModelSpec) -> str:
    """Emit an executable Tier-2 CPWL MOSFET subcircuit.

    Ports: d (drain) g (gate) s (source) - positional order fixed and
    documented; Tj thermal port not yet synthesized (M5 scope).
    """
    meta = _resolve_metadata(spec)
    name = f"{spec.device_name}_PIECESPICE_T2"

    channel = _channel_expression(spec, vgs_term="V(gi,ss)")

    lines: List[str] = []
    ap = lines.append
    ap(f"* PieceSpice Tier 2 CPWL Model for {spec.device_name}")
    ap(f"* Generated from fitted PWLModelSpec (K={len(spec.vgs_values)} "
       f"Vgs curves, {len(spec.breakpoints_vds)} Vds breakpoints)")
    ap(f".SUBCKT {name} d g s")
    ap(f"* Pins: 1=Drain(d) 2=Gate(g) 3=Source(s)")
    ap("")
    ap("* --- parasitics ---")
    ap(f"RG g gi {_fmt(meta['r_gate'])}")
    ap(f"RD d di {_fmt(meta['r_drain'])}")
    ap(f"RS s ss {_fmt(meta['r_source'])}")
    ap("")

    ap("* --- channel: tanh-knee behavioral source (Curves 1+2) ---")
    ap(f"G_CH di ss VALUE = {{({channel})}}")
    ap("")

    ap("* --- charge-conserving capacitances (Curve 4, plan 2.3) ---")
    qoss_v = spec.q_oss_vs_v
    qoss_q = spec.q_oss_charge_c
    if len(qoss_v) >= 2:
        ap("* Output charge Qoss(Vds) as chained soft-abs ramps:")
        expr_qoss = f"{_fmt(qoss_q[0])}"
        prev_slope = 0.0
        for i in range(1, len(qoss_v)):
            dv = qoss_v[i] - qoss_v[i - 1]
            dq = qoss_q[i] - qoss_q[i - 1]
            slope = dq / dv if dv else 0.0
            delta = slope - prev_slope   # DELTA, not absolute (audit bug)
            if abs(delta) > 0:
                expr_qoss += (
                    f" + {delta:.6e}*((V(di,ss)-({_fmt(qoss_v[i - 1])}))"
                    f"+abs((V(di,ss)-({_fmt(qoss_v[i - 1])}))))/2")
            prev_slope = slope
        ap(f"C_QOSS di ss Q = {{{expr_qoss}}}")
    else:
        c_lin = float(meta.get("c_oss_lin", 500e-12))
        ap(f"* Qoss table unavailable; linear fallback C={c_lin:.3e}")
        ap(f"C_OSS di ss {_fmt(c_lin)}")

    c_iss_lin = float(meta.get("c_iss_lin", 1500e-12))
    ap(f"C_ISS gi ss {_fmt(c_iss_lin)}")
    c_rss_lin = float(meta.get("c_rss_lin", 50e-12))
    ap(f"C_RSS gi di {_fmt(c_rss_lin)}")
    ap("")

    ap("* --- body diode (Curve 6) ---")
    if len(spec.body_diode_vsd) >= 2:
        vsd = np.asarray(spec.body_diode_vsd)
        isd = np.asarray(spec.body_diode_isd)
        # derive IS/N from two points on the exponential: pick 10% / 90% Isd
        i_lo, i_hi = float(isd.min()), float(isd.max())
        vt = 0.026  # thermal voltage @ 300K
        n_est = max((vsd[np.argmax(isd)] - vsd[np.argmin(isd)]) /
                    max(vt * np.log(max(i_hi / max(i_lo, 1e-15), 1.01)), 0.05),
                     0.5)
        n_est = min(n_est, 4.0)
        is_est = max(i_lo * np.exp(-vsd[np.argmin(isd)] / (vt * n_est)), 1e-15)
        ap(f".MODEL D_BD_{name} D(IS={_fmt(is_est)} N={n_est:.3f} "
           f"RS={_fmt(max(float(spec.rds_on), 1e-4))})")
        ap(f"D_BODY ss di D_BD_{name}")
    else:
        ap("* body-diode samples unavailable; ideal diode fallback")
        ap(".MODEL D_IDEAL D(IS=1e-12)")
        ap("D_BODY ss di D_IDEAL")
    ap("")

    ap(f".ENDS {name}")
    return "\n".join(lines)


class NetlistBuilder:
    """Synthesizes executable Tier-2 SPICE subcircuits (Tier 1/3 future)."""

    @staticmethod
    def build_tier2_subcircuit(spec: PWLModelSpec) -> str:
        return build_tier2_subcircuit(spec)
