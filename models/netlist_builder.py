"""Tier-2 CPWL MOSFET subcircuit synthesis (M3.5, plan §1.3/§2.3/§2.4).

Emits a standard-SPICE subcircuit that reproduces the vendor device's
behavior with PWL/behavioral elements instead of stiff physics:

    .SUBCKT <name> d g s
      RG  g  gi {r_gate}                    * gate parasitic
      RD  d  di {r_drain}                   * drain parasitic
      RS  s  ss {r_source}                  * source parasitic

      G_CH  di ss VALUE = { ... tanh-knee channel ... }   <- Curves 1+2
      C_QOSS di ss Q = { PWL charge table }               <- Curve 4
      C_ISS  gi ss Q = { PWL charge table }               <- Curve 4
      C_RSS  gi di Q = { PWL charge table }               <- Curve 4
      D_BODY ss di <PWL/exponential from Curve 6>

    .ENDS

Design rules honored:
* Charge-conserving `Q=` capacitors (plan §2.3): no i=C(v)*dv/dt drift.
* Tanh-knee channel source (plan §2.4): finite continuous Jacobians.
* Device-agnostic: every parameter comes from the PWLModelSpec produced by
  the M3 fitting pipeline; nothing is tuned to a specific vendor part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import numpy as np
from scipy.optimize import least_squares
from scipy.special import lambertw

from approximation.smoother import (
    delta_ramp_chain,
    emit_qoss_expr,
    emit_transfer_ramp,
    fit_tanh_knee_ls,
    fit_tanh_plateau,
    smooth_relu_product,
)
from common.types import PWLModelSpec, PWL_META_KEYS


# Documented fallbacks for optional metadata keys the producer has not yet
# populated. Missing keys never crash the builder; unknown keys are reported.
_META_DEFAULTS = {
    "r_gate": 1.5,
    "r_source": 1e-4,
    "r_drain": 1e-4,
}

# Documented unified-ROM metadata that the Tier-2 emitter deliberately does
# not consume.  It is accepted silently (never merged into the emitter's
# parasitic/cap scalars) so a UROM C0 build does not emit false "unknown
# metadata" warnings for keys that the unified layer owns.
_UNIFIED_ONLY_METADATA = frozenset({
    "Vds_cap_max", "theta", "theta_bounds",
    "urom_provenance", "urom_c1_eps", "urom_c1_eps_by_table",
    "urom_emitted_eps_by_table", "urom_channel_basis",
    "urom_c1_variant", "urom_tuning_theta", "urom_method_label",
    "urom_gate_charge",
})

# Fixed, vendor-independent fitting rule (validated by the four-device sweep
# in reports/guarded_blend_sweep.json). A plateau outside this ratio band is
# unreliable (e.g. a near-zero curve) and the forced endpoint is retained.
_PLATEAU_BLEND_W = 0.25
_PLATEAU_GUARD = (0.7, 1.3)


def _resolve_metadata(spec: PWLModelSpec) -> dict:
    """Merge spec metadata over documented defaults; warn on unknown keys."""
    import logging
    _logger = logging.getLogger(__name__)

    meta = dict(_META_DEFAULTS)
    for k, v in spec.metadata.items():
        if k in _META_DEFAULTS or k in PWL_META_KEYS:
            meta[k] = v
        elif k not in _UNIFIED_ONLY_METADATA:
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
    """Format a float for SPICE output (7 significant digits, e-notation)."""
    return f"{x:.6e}"


# A measured gate-charge table is emitted as a nonlinear Q= element, so the
# same zero-anchor/monotonicity contract as Qoss applies.  The tolerance is
# the UROM charge anchor bar (1e-15 C).
_GATE_CHARGE_ZERO_ANCHOR_ABS_C = 1.0e-15


def _validate_gate_charge_table(
    spec: PWLModelSpec, name: str, vs, qs
) -> tuple[np.ndarray, np.ndarray]:
    """Validate an emitted Qgs/Qgd table; raise instead of fabricating."""

    v = np.asarray(vs, dtype=float).ravel()
    q = np.asarray(qs, dtype=float).ravel()
    if v.size < 2 or q.size < 2 or v.size != q.size:
        raise ValueError(
            f"{spec.device_name}: measured {name} needs >= 2 equal-length "
            f"samples (got {v.size} V, {q.size} C)")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(q))):
        raise ValueError(
            f"{spec.device_name}: measured {name} contains NaN/Inf")
    if np.any(np.diff(v) < 0.0):
        raise ValueError(
            f"{spec.device_name}: measured {name} voltage axis is not "
            f"non-decreasing")
    if np.any(np.diff(q) < 0.0):
        raise ValueError(
            f"{spec.device_name}: measured {name} charge table is not "
            f"monotone")
    if abs(float(q[0])) > _GATE_CHARGE_ZERO_ANCHOR_ABS_C:
        raise ValueError(
            f"{spec.device_name}: measured {name} is not zero-anchored "
            f"(Q(0)={float(q[0]):.6e} C)")
    return v, q


def _gate_charge_provenance(spec: PWLModelSpec) -> str:
    """One-line ASCII provenance for the emitted measured charge tables."""
    row = spec.metadata.get("urom_gate_charge")
    source = "spec q_iss/q_rss tables"
    if isinstance(row, dict) and row.get("source"):
        source = str(row["source"])
    elif isinstance(row, dict) and row.get("provenance"):
        prov = row["provenance"]
        if isinstance(prov, dict) and prov.get("source"):
            source = str(prov["source"])
    return " ".join(source.split()).encode("ascii", "replace").decode()


def _channel_ramps(spec: PWLModelSpec) -> tuple[List[float], List[float], List[float]]:
    """Return the (Vgs grid, transfer amplitude, saturation width) ramps.

    Single source of truth for the emitter and the offline numeric mirror:
    both consume exactly the same fitted knee, plateau blend and Vth anchor.
    """
    if (len(spec.breakpoints_vds) < 2 or not spec.vgs_values
            or not spec.breakpoints_id):
        raise ValueError(
            f"{spec.device_name}: channel needs >=2 Vds breakpoints and "
            f">=1 Vgs curve (got {len(spec.breakpoints_vds)} Vds, "
            f"{len(spec.vgs_values)} Vgs)")
    vd = np.asarray(spec.breakpoints_vds, dtype=float)
    vmax = float(vd[-1])
    span = vmax - float(vd[0])
    drives = [float(v) for v in spec.vgs_values]
    endpoints = [float(row[-1]) for row in spec.breakpoints_id]

    sat_vals = []
    amplitudes = []
    for row, end in zip(spec.breakpoints_id, endpoints):
        y = np.asarray(row, dtype=float)
        plateau = max(end, 1e-9)
        s = fit_tanh_knee_ls(vd, y, plateau)
        s = min(max(s, 0.05 * span), vmax)
        sat_vals.append(s)

        # Endpoint is forced by default. Only a well-behaved fitted plateau
        # (ratio inside the fixed guard band) is blended in, by a fixed 25 %.
        amp = end / np.tanh(vmax / s)
        i_fit, _ = fit_tanh_plateau(vd, y, plateau)
        ratio = i_fit / max(end, 1e-12)
        if _PLATEAU_GUARD[0] <= ratio <= _PLATEAU_GUARD[1]:
            amp = (1.0 - _PLATEAU_BLEND_W) * amp + _PLATEAU_BLEND_W * i_fit
        amplitudes.append(amp)

    xs = [0.0] + drives
    iv = [0.0] + amplitudes
    sv = [sat_vals[0]] + sat_vals
    v_th = float(spec.v_th)
    if 0.0 < v_th < min(drives):
        xs = [0.0, v_th] + drives
        iv = [0.0, 0.0] + amplitudes
        sv = [sat_vals[0], sat_vals[0]] + sat_vals
    return xs, iv, sv


def evaluate_tier2_channel(spec: PWLModelSpec, vgs, vds, *,
                           eps: float = 0.0) -> np.ndarray:
    """Numeric mirror of the emitted closed Tier-2 ``G_CH`` expression.

    ``I = PWL(Vgs) * tanh(Vds / PWL_sat(Vgs))``.  The PWL grids, fitted tanh
    knee, plateau blend and Vth anchor come from :func:`_channel_ramps`, the
    one source of truth also used by ``_channel_expression``.  Both ramps are
    evaluated with :func:`smooth_relu_product`, the numpy mirror of
    ``delta_ramp_chain``, so ``eps > 0`` agrees with the emitted C1 chain and
    ``eps = 0`` is the closed C0 channel.  Finite inputs give finite current;
    no simulator is started.
    """
    xs, iv, sv = _channel_ramps(spec)
    e = 0.0 if eps is None else float(eps)
    if not math.isfinite(e) or e < 0.0:
        raise ValueError("eps must be finite and >= 0")
    g = np.asarray(vgs, dtype=float)
    d = np.asarray(vds, dtype=float)
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(d))):
        raise ValueError("vgs and vds must be finite")
    g, d = np.broadcast_arrays(g, d)
    amp = np.asarray(smooth_relu_product(g, xs, iv, e), dtype=float)
    sat = np.asarray(smooth_relu_product(g, xs, sv, e), dtype=float)
    amp = np.reshape(amp, g.shape)
    sat = np.reshape(sat, g.shape)
    if np.any(sat <= 0.0):
        raise ValueError("channel saturation width must be positive")
    out = amp * np.tanh(d / sat)
    if not np.all(np.isfinite(out)):
        raise ValueError("channel evaluator produced non-finite current")
    return out


def _channel_expression(spec: PWLModelSpec, vgs_term: str, *,
                        eps: float = 0.0) -> str:
    """Behavioral channel current: PWL(Vgs) drive * tanh(Vds/vds_sat(Vgs)).

    Topology and fitted scalars are unchanged from the closed Tier-2 emitter;
    see :func:`_channel_ramps` (shared with :func:`evaluate_tier2_channel`)
    for the tanh-knee fit, plateau blend and Vth anchor.  eps is a
    smoothness-only option for the two PWL(Vgs) ramps: it is fed to the
    shared delta-slope chain and leaves the fitted knee width, plateau
    blend and Vth anchor untouched. eps = 0 is the closed Tier-2 emitter.
    """
    xs, iv, sv = _channel_ramps(spec)
    amp = emit_transfer_ramp(xs, iv, eps, term=vgs_term)
    sat = emit_transfer_ramp(xs, sv, eps, term=vgs_term)
    return f"({amp})*tanh(V(di,ss)/({sat}))"


# --- Body diode (Curve 6): measured fit -------------------------------------
# Single thermal-voltage convention shared by the fit and the emitted model.
_VT_300K = 0.026

# The three-parameter diode needs at least three forward samples.
_BODY_DIODE_MIN_FIT_POINTS = 3

# Acceptance budget: RMS of the log10(current) residuals in decades. The fit
# must also beat the closed 2-point/full-span heuristic on the same samples.
_BODY_DIODE_RMS_MAX_DEC = 0.20

# A measured forward sweep bottoms out in a testbench current floor: below the
# conduction knee the current is flat or non-monotone in V and no single diode
# can reproduce it. The fit is therefore evaluated on the most inclusive
# current window that meets the budget (fractions of the peak measured
# current). Candidates that keep fewer than three samples are skipped.
_BODY_DIODE_CURRENT_FRACTIONS = (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)

# Physical bounds. IS>0 and N>=0.5 are the standard SPICE diode limits; RS is
# non-negative and cannot exceed the die on-resistance (the old heuristic's
# point estimate, now only an upper bound: the body-diode path shares the
# drift region with the channel).
_BODY_DIODE_IS_BOUNDS = (1.0e-15, 1.0)   # [A]
_BODY_DIODE_N_BOUNDS = (0.5, 8.0)        # ideality factor
_BODY_DIODE_RS_FLOOR = 1.0e-4            # [Ohm]


@dataclass(frozen=True)
class BodyDiodeFit:
    """Fitted SPICE diode parameters and the residual achieved on the data.

    ``rms_log10`` is the RMS of ``log10(I_fit) - log10(I_measured)`` in
    decades over the ``n_retained`` samples used by the fit; ``n_total`` counts
    the finite, forward, strictly positive samples available. The samples are
    the ones at or above ``current_fraction`` times the peak measured current.
    """

    isat: float
    n: float
    rs: float
    rms_log10: float
    n_retained: int
    n_total: int
    current_fraction: float


def body_diode_current(vsd, isat: float, n: float, rs: float) -> np.ndarray:
    """SPICE diode current for a forward bias, solving the implicit equation.

    Implements ``I = IS*(exp((V - I*RS)/(N*Vt)) - 1)`` with ``Vt=_VT_300K``.
    The RS term couples V and I, so the current is solved per point with the
    Lambert-W form ``I = IS*(W(z)/c - 1)``, ``c = RS*IS/(N*Vt)``,
    ``z = c*exp((V + RS*IS)/(N*Vt))``. ``RS = 0`` reduces to ``IS*expm1``.
    """
    v = np.asarray(vsd, dtype=float)
    a = float(n) * _VT_300K
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("body-diode N*Vt must be finite and positive")
    if not (math.isfinite(float(isat)) and float(isat) > 0.0):
        raise ValueError("body-diode IS must be finite and positive")
    if not (math.isfinite(float(rs)) and float(rs) >= 0.0):
        raise ValueError("body-diode RS must be finite and non-negative")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if rs <= 0.0:
            return float(isat) * np.expm1(v / a)
        c = float(rs) * float(isat) / a
        log_z = math.log(c) + (v + float(rs) * float(isat)) / a
        # exp is clipped at the double range; anything beyond that is already
        # far outside the measured currents and only needs to stay finite.
        z = np.exp(np.clip(log_z, -700.0, 700.0))
        w = np.real(lambertw(z))
        return float(isat) * (w / c - 1.0)


def body_diode_log10_rms(vsd, isd, isat: float, n: float, rs: float) -> float:
    """RMS of log10(current) residuals [decades] for the given parameters."""
    v = np.asarray(vsd, dtype=float)
    i = np.asarray(isd, dtype=float)
    pred = body_diode_current(v, isat, n, rs)
    ok = np.isfinite(pred) & (pred > 0.0) & np.isfinite(i) & (i > 0.0)
    if not np.any(ok):
        return math.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.log10(pred[ok]) - np.log10(i[ok])
    if not np.all(np.isfinite(resid)):
        return math.inf
    return float(np.sqrt(np.mean(resid * resid)))


def _body_diode_forward_samples(vsd, isd, *, min_points: int):
    """Validate a measured forward curve; raise instead of fabricating data."""
    v = np.asarray(vsd, dtype=float).ravel()
    i = np.abs(np.asarray(isd, dtype=float).ravel())
    if v.size != i.size:
        raise ValueError(
            f"body-diode Vsd/Isd length mismatch ({v.size} vs {i.size})")
    if v.size == 0:
        raise ValueError("body-diode curve is empty")
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(i))):
        raise ValueError(
            "body-diode curve contains non-finite Vsd/Isd samples; refusing "
            "to fit a diode to corrupted data")
    if np.any(v < 0.0):
        raise ValueError(
            "body-diode Vsd must be a non-negative forward sweep; got "
            f"Vsd_min={float(v.min()):.6g} V")
    keep = i > 0.0
    if int(np.count_nonzero(keep)) < min_points:
        raise ValueError(
            "body-diode curve has "
            f"{int(np.count_nonzero(keep))} finite positive forward sample(s); "
            f"at least {min_points} are required (Vsd/Isd pairs)"
        )
    return v[keep], i[keep]


def heuristic_body_diode(vsd, isd, rds_on: float) -> BodyDiodeFit:
    """The closed Tier-2 heuristic: full-span N/IS with ``RS = Rds(on)``.

    Kept for exactly two-sample curves (the three-parameter fit is
    underdetermined there) and as the acceptance baseline for the fit. It is
    not a measured fit: it anchors IS at the lowest sample and derives N from
    the full voltage/current span, so a series-resistance-dominated high
    current end biases N upward and the forward error is large.
    """
    v, i = _body_diode_forward_samples(vsd, isd, min_points=2)
    i_lo, i_hi = float(i.min()), float(i.max())
    n_est = max(
        (float(v[np.argmax(i)]) - float(v[np.argmin(i)])) /
        max(_VT_300K * math.log(max(i_hi / max(i_lo, 1e-15), 1.01)), 0.05),
        0.5,
    )
    n_est = min(n_est, 4.0)
    is_est = max(
        i_lo * math.exp(-float(v[np.argmin(i)]) / (_VT_300K * n_est)), 1e-15)
    rs = max(float(rds_on), _BODY_DIODE_RS_FLOOR)
    rms = body_diode_log10_rms(v, i, is_est, n_est, rs)
    return BodyDiodeFit(
        isat=is_est, n=n_est, rs=rs, rms_log10=rms,
        n_retained=int(v.size), n_total=int(v.size), current_fraction=0.0)


def _body_diode_residuals(x, v: np.ndarray, i: np.ndarray) -> np.ndarray:
    """log10(predicted) - log10(measured); ``x = [log10 IS, N, RS]``."""
    pred = body_diode_current(v, 10.0 ** x[0], x[1], x[2])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        log_pred = np.where(
            np.isfinite(pred) & (pred > 0.0),
            np.log10(np.maximum(pred, 1e-300)),
            -300.0,
        )
        return log_pred - np.log10(i)


def _fit_body_diode_params(v: np.ndarray, i: np.ndarray,
                           rs_max: float) -> tuple[float, float, float]:
    """Bounded least squares over log10(IS), N and RS with a small start grid.

    The residual is the log10 current error, so one decade of mismatch costs
    one unit independent of the current magnitude. Multiple deterministic
    starts avoid the flat direction that RS shows on low-current curves.
    """
    lo = np.array([math.log10(_BODY_DIODE_IS_BOUNDS[0]),
                   _BODY_DIODE_N_BOUNDS[0], 0.0])
    hi = np.array([math.log10(_BODY_DIODE_IS_BOUNDS[1]),
                   _BODY_DIODE_N_BOUNDS[1], rs_max])
    is0 = float(np.clip(math.log10(max(float(i.min()), 1e-300)),
                        lo[0], hi[0]))
    best = None
    for n0 in (0.8, 1.2, 2.0, 3.5):
        for rs0 in (0.0, rs_max):
            x0 = np.clip(np.array([is0, n0, rs0]), lo, hi)
            try:
                res = least_squares(
                    _body_diode_residuals, x0, args=(v, i),
                    bounds=(lo, hi), max_nfev=500)
            except (ValueError, FloatingPointError):
                continue
            if not np.all(np.isfinite(res.fun)):
                continue
            if best is None or res.cost < best.cost:
                best = res
    if best is None:
        raise ValueError(
            "body-diode least-squares fit failed for every start point")
    return 10.0 ** float(best.x[0]), float(best.x[1]), float(best.x[2])


@lru_cache(maxsize=64)
def _fit_body_diode_cached(vsd: tuple, isd: tuple,
                           rds_on: float) -> BodyDiodeFit:
    v = np.asarray(vsd, dtype=float)
    i = np.asarray(isd, dtype=float)
    rs_max = max(float(rds_on), _BODY_DIODE_RS_FLOOR)
    heuristic = heuristic_body_diode(v, i, rds_on)

    best_seen = math.inf
    for fraction in _BODY_DIODE_CURRENT_FRACTIONS:
        mask = i >= fraction * float(i.max())
        if int(np.count_nonzero(mask)) < _BODY_DIODE_MIN_FIT_POINTS:
            continue
        isat, n, rs = _fit_body_diode_params(v[mask], i[mask], rs_max)
        rms = body_diode_log10_rms(v[mask], i[mask], isat, n, rs)
        ref = body_diode_log10_rms(v[mask], i[mask],
                                   heuristic.isat, heuristic.n, heuristic.rs)
        if not math.isfinite(rms):
            continue
        best_seen = min(best_seen, rms)
        if rms <= _BODY_DIODE_RMS_MAX_DEC and rms < ref:
            return BodyDiodeFit(
                isat=isat, n=n, rs=rs, rms_log10=rms,
                n_retained=int(np.count_nonzero(mask)),
                n_total=int(v.size), current_fraction=fraction)
    raise ValueError(
        "body-diode fit cannot meet the acceptance budget "
        f"(RMS log10 error <= {_BODY_DIODE_RMS_MAX_DEC:.2f} decades and "
        "strictly better than the closed full-span heuristic); best "
        f"achieved RMS={best_seen:.4f} dec over {int(v.size)} samples. "
        "Refusing to emit the old misfit.")


def fit_body_diode(vsd, isd, *, rds_on: float) -> BodyDiodeFit:
    """Fit the standard SPICE diode to a measured forward (Vsd, Isd) curve.

    Public entry point: validates the samples (finite, non-negative Vsd, at
    least three strictly positive forward currents), then fits IS/N/RS by
    bounded least squares on log10(current) residuals with the physical
    bounds documented above, on the most inclusive current window that meets
    :data:`_BODY_DIODE_RMS_MAX_DEC`. Raises ``ValueError`` when the data are
    missing/short/non-finite or the budget cannot be met - it never falls back
    to the old heuristic silently.
    """
    if not math.isfinite(float(rds_on)) or float(rds_on) <= 0.0:
        raise ValueError(
            f"body-diode fit needs a finite positive Rds(on); got {rds_on!r}")
    v, i = _body_diode_forward_samples(
        vsd, isd, min_points=_BODY_DIODE_MIN_FIT_POINTS)
    return _fit_body_diode_cached(tuple(v.tolist()), tuple(i.tolist()),
                                  float(rds_on))


def build_tier2_subcircuit(spec: PWLModelSpec, *,
                           c1_eps: float | None = None,
                           name: str | None = None,
                           subckt_name: str | None = None,
                           measured_gate_charge: bool = False) -> str:
    """Emit an executable Tier-2 CPWL MOSFET subcircuit.

    Ports: d (drain) g (gate) s (source) - positional order fixed and
    documented; Tj thermal port not yet synthesized (M5 scope).

    c1_eps is an opt-in C1 hinge width [V] in the grid's units: None means 0
    (the closed Tier-2 C0 emitter). Callers supply a data-derived width from
    approximation.smoother.eps_from_grid - this builder has no magic eps. It
    touches only the two ramp sites (channel Vgs and Qoss Vds); D_BODY,
    parasitics and the three charge elements stay unchanged.

    measured_gate_charge is a narrow opt-in: False (default) emits the closed
    Tier-2 linear ``C_ISS``/``C_RSS`` lines byte-for-byte as before.  True
    replaces exactly those two lines with nonlinear ``Q=`` ramps built from
    ``spec.q_iss_*`` (on V(gi,ss)) and ``spec.q_rss_*`` (on V(gi,di)); both
    tables must be finite, non-decreasing and zero-anchored or this raises.
    No linear and nonlinear variants are ever emitted together.

    name/subckt_name override the emitted .SUBCKT name (and the D_BD model
    suffix); both default to None, which keeps the closed Tier-2 name.
    subckt_name wins when both are supplied.
    """
    meta = _resolve_metadata(spec)
    subckt = subckt_name or name or f"{spec.device_name}_PIECESPICE_T2"
    eps = 0.0 if c1_eps is None else float(c1_eps)

    channel = _channel_expression(spec, vgs_term="V(gi,ss)", eps=eps)

    lines: List[str] = []
    ap = lines.append
    ap(f"* PieceSpice Tier 2 CPWL Model for {spec.device_name}")
    ap(f"* Generated from fitted PWLModelSpec (K={len(spec.vgs_values)} "
       f"Vgs curves, {len(spec.breakpoints_vds)} Vds breakpoints)")
    ap(f".SUBCKT {subckt} d g s")
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
        ap("* Output charge Qoss(Vds) as chained delta ramps:")
        # One shared delta-slope chain: Q(0)=0 is anchored by the helper.
        # At eps=0 this is numerically identical to the closed emitter; only
        # redundant parentheses around formatted breakpoints are normalised.
        expr_qoss = emit_qoss_expr(qoss_v, qoss_q, eps)
        ap(f"C_QOSS di ss Q = {{{expr_qoss}}}")
    else:
        c_lin = float(meta.get("c_oss_lin", 500e-12))
        ap(f"* Qoss table unavailable; linear fallback C={c_lin:.3e}")
        ap(f"C_OSS di ss {_fmt(c_lin)}")

    c_iss_lin = float(meta.get("c_iss_lin", 1500e-12))
    c_rss_lin = float(meta.get("c_rss_lin", 50e-12))
    if measured_gate_charge:
        iss_v, iss_q = _validate_gate_charge_table(
            spec, "q_iss", spec.q_iss_vs_v, spec.q_iss_charge_c)
        rss_v, rss_q = _validate_gate_charge_table(
            spec, "q_rss", spec.q_rss_vs_v, spec.q_rss_charge_c)
        ap("* Gate charge: measured nonlinear Qgs(Vgs)/Qgd(Vdg) tables "
           "(opt-in; no linear C_ISS/C_RSS)")
        ap(f"* provenance: {_gate_charge_provenance(spec)}")
        expr_iss = delta_ramp_chain(
            "V(gi,ss)", iss_v.tolist(), iss_q.tolist(), 0.0)
        expr_rss = delta_ramp_chain(
            "V(gi,di)", rss_v.tolist(), rss_q.tolist(), 0.0)
        ap(f"C_ISS gi ss Q = {{{expr_iss}}}")
        ap(f"C_RSS gi di Q = {{{expr_rss}}}")
    else:
        ap(f"C_ISS gi ss {_fmt(c_iss_lin)}")
        ap(f"C_RSS gi di {_fmt(c_rss_lin)}")
    ap("")

    ap("* --- body diode (Curve 6) ---")
    n_bd = len(spec.body_diode_vsd)
    if n_bd == 0:
        ap("* body-diode samples unavailable; ideal diode fallback")
        ap(".MODEL D_IDEAL D(IS=1e-12)")
        ap("D_BODY ss di D_IDEAL")
    elif n_bd == 1:
        raise ValueError(
            f"{spec.device_name}: body-diode curve has 1 sample; at least 2 "
            "are required (3+ for the measured IS/N/RS fit)")
    else:
        vsd = np.asarray(spec.body_diode_vsd, dtype=float)
        isd = np.asarray(spec.body_diode_isd, dtype=float)
        if vsd.size != isd.size:
            raise ValueError(
                f"{spec.device_name}: body-diode Vsd/Isd length mismatch "
                f"({vsd.size} vs {isd.size})")
        if vsd.size == 2:
            # Underdetermined (3 parameters, 2 samples): keep the documented
            # closed heuristic, but say so and report its residual.
            diode = heuristic_body_diode(vsd, isd, spec.rds_on)
            ap("* body-diode provenance: measured Curve 6 forward (Vsd,Isd) "
               "sweep, 2 samples; IS/N/RS least-squares fit is "
               "underdetermined -> closed 2-point heuristic; "
               "I=IS*(exp((V-I*RS)/(N*Vt))-1), Vt=0.026 V; "
               f"IS={_fmt(diode.isat)} N={diode.n:.3f} "
               f"RS={_fmt(diode.rs)}; RMS log10 error="
               f"{diode.rms_log10:.6e} dec (2/2 samples)")
        else:
            diode = fit_body_diode(vsd, isd, rds_on=spec.rds_on)
            if diode.current_fraction == 0.0:
                scope = f"all {diode.n_total} samples"
            else:
                scope = (f"{diode.n_retained}/{diode.n_total} samples, "
                         f">={diode.current_fraction:g}*Imax")
            ap("* body-diode provenance: measured Curve 6 forward (Vsd,Isd) "
               f"sweep (config/outputs/{spec.device_name}/"
               "curve6_body_diode.yaml); "
               "bounded least-squares fit of "
               "I=IS*(exp((V-I*RS)/(N*Vt))-1) on log10(current), "
               "Vt=0.026 V; "
               f"IS={_fmt(diode.isat)} N={diode.n:.3f} "
               f"RS={_fmt(diode.rs)}; RMS log10 error="
               f"{diode.rms_log10:.6e} dec ({scope})")
        ap(f".MODEL D_BD_{subckt} D(IS={_fmt(diode.isat)} "
           f"N={diode.n:.3f} RS={_fmt(diode.rs)})")
        ap(f"D_BODY ss di D_BD_{subckt}")
    ap("")

    ap(f".ENDS {subckt}")
    return "\n".join(lines)


class NetlistBuilder:
    """Synthesizes executable Tier-2 SPICE subcircuits (Tier 1/3 future)."""

    @staticmethod
    def build_tier2_subcircuit(spec: PWLModelSpec, *,
                               c1_eps: float | None = None,
                               name: str | None = None,
                               subckt_name: str | None = None) -> str:
        return build_tier2_subcircuit(
            spec, c1_eps=c1_eps, name=name, subckt_name=subckt_name)
