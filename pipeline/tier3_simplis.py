"""Tier-3 ``simplis`` variant: SIMPLIS-style Level-2 charge tables.

Assembles the three (V,Q) charge tables from ONE full C(V) sweep: the
persisted Curve-4 arrays when available, otherwise the live AC capacitance
re-extraction (with an explicit data-limitation disclosure). The closed
Tier-2 channel/diode/parasitics are untouched, and the same DPT / waveform /
speed benches validate the emitted ASCII subcircuit.

All reported numbers are measured; incomplete simulations stay incomplete.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
import json
import time
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from common.types import PWLModelSpec, trim_trailing_invalid
from config.yaml_io import load_input, load_output
from approximation.charge_model import integrate_charge
from metrics.static import compute_curve_error
from parser.netlist import resolve_pin_connections
from models.simplis_netlist_builder import (
    CGSLinearityDecision,
    RELTOL,
    build_simplis_subcircuit,
    cgs_linearity_decision,
)
from models.hybrid_channel import HybridChannelFit, fit_hybrid_channel
from approximation.ann_residual import AnnResidualFit
from pipeline.ann_fit import hybrid_ann_fit_for_vendor
from pipeline import fitting
from pipeline.curves import _cap_ref_vds
from pipeline.registry import ROOT
from pipeline.tier3 import (
    METRICS,
    _dpt_conditions,
    _device_current,
    _DEVICE_CURRENT,
    _log_iterations,
    _rel_err,
    _waveform_errors,
)
from simulation.engine import SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder
from simulation.testbenches.capacitance import CapacitanceTestbench, F_MEAS_DEFAULT
from simulation.testbenches.double_pulse import DoublePulseRun, extract_dpt_metrics
from simulation.testbenches.dpt_trajectory import capture_trajectory
from simulation.testbenches.speedup_bench import _pulse_train, run_inductive_benchmark


# 1.5 Ohm / 0.1 mOhm / 0.1 mOhm: the Tier-2 emitter's documented metadata
# defaults (models/netlist_builder.py::_META_DEFAULTS). Reused, not retuned.
_PARASITICS = {"r_gate": 1.5, "r_source": 1e-4, "r_drain": 1e-4}

# The emitted simplis subckt header is ALWAYS `.SUBCKT <name> d g s`
# (models/simplis_netlist_builder.build_simplis_subcircuit). Its port order is
# its own, NOT the golden subcircuit's: EPC's golden is gate-first and CPM2's
# golden carries a 4th Tj port. Every X-instance of the emitted model binds
# these roles; the golden side keeps traj.pin_roles.
_EMITTED_PIN_ROLES = ["drain", "gate", "source"]

# Declared project decision: the emitted Q(V) tables must reproduce the
# measured small-signal capacitance to within 10% on the measured grid. This
# is a DIAGNOSTIC, not a charge budget: the existing 0.5% charge budget does
# not bound capacitance error, and the two are reported side by side.
CAP_ERROR_BUDGET_PCT = 10.0

# CDG provenance: the legacy extraction measures Crss on a Vds sweep at
# Vgs=0, where v_DG == v_DS. A bias-aware sweep supplies Cgd on a real Vdg
# axis instead; without it the legacy axis is disclosed, never invented.
CDG_AXIS_VGS0 = "vds_at_vgs0"
CDG_AXIS_VDG = "vdg_at_vgs_bias"
_CDG_VGS0_LIMITATION = (
    "Only a Vgs=0 C(V) sweep is available, so CDG is emitted on V(di,gi) "
    "using the identity Vdg=Vds. No bias-aware Cgd table exists and no bias "
    "axis was synthesized.")
_CV_SINGLE_BIAS_LIMITATION = (
    "Only one persisted C(V) sweep at Vgs=0 is available; no multi-bias "
    "C-V set exists. Charge tables are fitted from that single sweep and no "
    "bias axis was invented from scalar equivalents.")


@dataclass(frozen=True)
class _ChargeFit:
    """Greedy PWL fit of an anchored Q(V) curve."""

    pairs: list[Tuple[float, float]]
    v_full: np.ndarray
    q_full: np.ndarray
    max_abs_error_c: float
    error_budget_c: float
    c_full: np.ndarray
    max_cap_error_pct: float | None = None
    cap_error_budget_pct: float = CAP_ERROR_BUDGET_PCT


def _fit_anchored_charge(
    v, c, rel_tol: float, *, forced_v: float | None = None,
    forbid_below: float | None = None,
) -> _ChargeFit:
    """Fit Q(V) with a (0, 0) anchor and NO max_points cap.

    Same farthest-point rule as ``approximation.charge_model.pwl_charge_segments``
    with three measured-data extensions: Q(0)=0, Q(v_min)=s0*v_min where s0
    is the FIRST MEASURED segment slope (no invented capacitance), and an
    optional DPT-derived breakpoint that is forced into the table while
    interior points below it are forbidden. The loop stops only when the
    repo's absolute charge budget is met, so the achieved error/segments
    are measured rather than truncated.
    """
    measured = integrate_charge(np.asarray(v, dtype=float),
                                np.asarray(c, dtype=float))
    vv, qq = measured.v, measured.q
    s0 = float((qq[1] - qq[0]) / (vv[1] - vv[0]))
    anchored_at_zero = vv[0] == 0.0
    if anchored_at_zero:
        # Already anchored at 0 V: prepending another 0 would duplicate the
        # axis sample (the identity s0*vv[0] = 0 makes it a no-op).
        va, qa = vv, qq
    else:
        va = np.concatenate(([0.0], vv))
        qa = np.concatenate(([0.0], qq + s0 * vv[0]))

    forced_idx = None
    if forced_v is not None and va[0] < forced_v < va[-1]:
        j = int(np.searchsorted(va, forced_v))
        if j >= len(va) or not np.isclose(va[j], forced_v):
            q_forced = float(np.interp(forced_v, va, qa))
            va = np.insert(va, j, forced_v)
            qa = np.insert(qa, j, q_forced)
        forced_idx = int(np.searchsorted(va, forced_v))

    budget = float(rel_tol) * float(qa[-1])
    idx = {0, len(va) - 1}
    if forced_idx is not None:
        idx.add(forced_idx)
    # A table holding every measured point is the perfect-fit limit, so it
    # also bounds the loop. No fit can spin forever on a noisy sweep.
    max_iter = len(va)
    for _ in range(max_iter):
        chosen = sorted(idx)
        # ``idx`` holds positions into va; np.interp's xp must be the
        # corresponding VOLTAGES. Passing the positions themselves silently
        # fits Q(index) and changes every emitted table.
        err = np.abs(qa - np.interp(va, va[chosen], qa[chosen]))
        err[chosen] = 0.0
        if forbid_below is not None:
            err[va < forbid_below] = 0.0
        worst = int(np.argmax(err))
        if err[worst] <= budget:
            break
        idx.add(worst)
    else:
        final = sorted(idx)
        achieved = float(np.max(np.abs(
            qa - np.interp(va, va[final], qa[final]))))
        ratio = achieved / budget if budget > 0 else float("inf")
        raise RuntimeError(
            f"charge fit missed the {float(rel_tol):.3%} budget after "
            f"{max_iter} points: achieved {achieved:.6e} C, budget "
            f"{budget:.6e} C, error ratio {ratio:.4f}")

    chosen = sorted(idx)
    vb, qb = va[chosen], qa[chosen]
    full_err = float(np.max(np.abs(qa - np.interp(va, vb, qb))))
    # c_full aligns with (va, qa): the anchor at V=0 uses the first measured
    # segment slope, exactly the charge the fit stores there.
    c_measured = np.asarray(measured.c, dtype=float)
    c_full = (c_measured if anchored_at_zero else
              np.concatenate(([float(measured.c[0])], c_measured)))
    return _ChargeFit(
        pairs=[(float(x), float(y)) for x, y in zip(vb, qb)],
        v_full=va, q_full=qa,
        max_abs_error_c=full_err, error_budget_c=budget,
        c_full=c_full)


def _cap_error_pct(fit: _ChargeFit,
                   pairs: Sequence[Tuple[float, float]]) -> float | None:
    """Measured C-V error of the emitted table, in percent of local C.

    Finite-differences the emitted (piecewise-linear) Q(V) and compares
    dQ/dV with the measured capacitance on the common measured grid. The
    relative denominator is clipped at 0.1% of the peak measured C so a
    near-zero region does not manufacture a divide-by-zero. Returns ``None``
    when no measured capacitance is available: the diagnostic is measured,
    never fabricated.
    """
    c_meas = np.asarray(fit.c_full, dtype=float) if fit.c_full is not None \
        else None
    if c_meas is None or c_meas.size != fit.v_full.size:
        return None
    vb = np.asarray([p[0] for p in pairs], dtype=float)
    qb = np.asarray([p[1] for p in pairs], dtype=float)
    grid = np.asarray(fit.v_full, dtype=float)
    inside = (grid >= vb[0]) & (grid <= vb[-1])
    if np.count_nonzero(inside) < 2:
        return None
    x = grid[inside]
    q_emit = np.interp(x, vb, qb)
    dq_dv = np.gradient(q_emit, x)
    c_local = c_meas[inside]
    peak = float(np.max(np.abs(c_local))) if c_local.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return None
    denom = np.maximum(np.abs(c_local), 1e-3 * peak)
    rel = np.abs(dq_dv - c_local) / denom
    return float(100.0 * np.max(rel))


def _cap_curve_to_vbus(v, curves, vbus: float):
    """Trim the AC sweep at the DPT bus voltage (interpolated endpoint)."""
    if vbus is None or vbus >= v[-1]:
        return v, curves
    grid = np.concatenate([v[v < vbus], [vbus]])
    return grid, tuple(np.interp(grid, v, c) for c in curves)


# ---- Full Curve-4 C(V) array ingestion (offline-testable) ----------------
_FULL_CV_KEYS = ("curve_vds", "ciss_curve_f", "coss_curve_f", "crss_curve_f")

# Disclosed whenever the full arrays are absent. The live AC re-sweep is real
# measured data, but it is not the persisted Curve-4 sweep; the record must
# say so instead of implying a richness the extracted file does not carry.
_CV_DATA_LIMITATION = (
    "Full Curve-4 C(V) arrays are not persisted for this vendor; charge "
    "tables use the live AC capacitance re-sweep and the recorded curve4 "
    "file carries only scalar Ciss/Coss/Crss equivalents. No full C(V) "
    "sweep was synthesized from scalar equivalents."
)


@dataclass(frozen=True)
class _CVSweep:
    """A validated full C(V) sweep: ascending Vds, finite non-negative C.

    ``vgs_bias_v`` / ``temperature_c`` are optional persisted provenance.
    ``None`` means the record did not carry the value; it is never inferred
    from another sweep or from a scalar equivalent.
    """

    v: np.ndarray
    ciss: np.ndarray
    coss: np.ndarray
    crss: np.ndarray
    vgs_bias_v: float | None = None
    temperature_c: float | None = None


@dataclass(frozen=True)
class _ChargeTables:
    """Anchored Q(V) fits plus the (V,Q) pairs actually emitted."""

    fits: dict
    pairs: dict
    n_clipped: dict
    cgs_decision: CGSLinearityDecision
    cdg_axis: str = CDG_AXIS_VGS0
    temperature_c: float | None = None
    vgs_bias_v: float | None = None
    cap_error_budget_pct: float = CAP_ERROR_BUDGET_PCT


def _optional_float(value, label: str, vendor: str) -> float | None:
    """Parse optional persisted metadata; ``None`` stays ``None``."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{vendor}: persisted {label}={value!r} is not numeric") from exc
    if not np.isfinite(parsed):
        raise ValueError(
            f"{vendor}: persisted {label}={value!r} must be finite")
    return parsed


def _validate_cv_arrays(vendor: str, v, ciss, coss, crss, *,
                        vgs_bias_v: float | None = None,
                        temperature_c: float | None = None) -> _CVSweep:
    """Validate one persisted C(V) set; fail closed on malformed data."""
    v = np.asarray(v, dtype=float).ravel()
    ciss = np.asarray(ciss, dtype=float).ravel()
    coss = np.asarray(coss, dtype=float).ravel()
    crss = np.asarray(crss, dtype=float).ravel()
    if not (v.size == ciss.size == coss.size == crss.size and v.size >= 2):
        raise ValueError(
            f"{vendor}: persisted Curve-4 C(V) arrays need equal lengths "
            f">= 2 (got {v.size}, {ciss.size}, {coss.size}, {crss.size})")
    if not np.all(np.diff(v) > 0):
        raise ValueError(
            f"{vendor}: persisted Curve-4 curve_vds axis must be strictly "
            "ascending")
    for name, arr in (("ciss_curve_f", ciss), ("coss_curve_f", coss),
                      ("crss_curve_f", crss)):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
            raise ValueError(
                f"{vendor}: persisted Curve-4 {name} must be finite and "
                "non-negative")
    return _CVSweep(v=v, ciss=ciss, coss=coss, crss=crss,
                    vgs_bias_v=vgs_bias_v, temperature_c=temperature_c)


def _cv_sweep_from_extracted(extracted: dict, vendor: str) -> _CVSweep | None:
    """Build a validated C(V) sweep from persisted Curve-4 arrays.

    Returns ``None`` when any of the four arrays is absent, so the caller
    keeps the existing live-sweep fallback. A partial or malformed persist
    fails closed: a non-ascending Vds axis, mismatched lengths, or non-finite
    / negative capacitance raise instead of being silently repaired.
    """
    if not isinstance(extracted, dict) or not all(
            extracted.get(k) is not None for k in _FULL_CV_KEYS):
        return None
    return _validate_cv_arrays(
        vendor,
        extracted["curve_vds"], extracted["ciss_curve_f"],
        extracted["coss_curve_f"], extracted["crss_curve_f"],
        vgs_bias_v=_optional_float(
            extracted.get("vgs_bias_v"), "vgs_bias_v", vendor),
        temperature_c=_optional_float(
            extracted.get("temperature_c"), "temperature_c", vendor))


_MULTI_CV_BY_VGS_KEYS = ("ciss_curve_f_by_vgs", "coss_curve_f_by_vgs",
                         "crss_curve_f_by_vgs")


def _bias_label(key, vendor: str, field: str) -> float:
    """Parse a persisted bias label; unlabeled keys fail closed."""
    try:
        value = float(key)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{vendor}: {field} key {key!r} is not a usable gate-bias label; "
            "refusing to invent a bias axis") from exc
    if not np.isfinite(value):
        raise ValueError(
            f"{vendor}: {field} key {key!r} is not a finite gate bias")
    return value


def _cv_sweeps_from_extracted(extracted: dict,
                              vendor: str) -> list[_CVSweep] | None:
    """Build one validated C(V) sweep per persisted gate bias.

    Supported multi-bias layouts (optional; single-sweep persists keep the
    historical path):

    * ``cv_sweeps``: a list of per-sweep dicts, each with the four
      ``_FULL_CV_KEYS`` plus optional ``vgs_bias_v`` / ``temperature_c``.
    * ``curve_vds`` plus ``ciss_curve_f_by_vgs`` /
      ``coss_curve_f_by_vgs`` / ``crss_curve_f_by_vgs``: three mappings
      keyed by gate bias, or three parallel lists with the bias labels in
      ``vgs_bias_v_by_sweep`` (or ``vgs_bias_v_list``). An unlabeled list
      fails closed: no bias axis is invented.

    Returns ``None`` when no full arrays are present. A partial multi-bias
    set fails closed instead of silently dropping to the single sweep.
    """
    if not isinstance(extracted, dict):
        return None
    if extracted.get("cv_sweeps") is not None:
        rows = extracted["cv_sweeps"]
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError(f"{vendor}: cv_sweeps must be a non-empty list")
        sweeps = []
        seen: set[float] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(
                    f"{vendor}: every cv_sweeps entry must be a dict")
            sweep = _cv_sweep_from_extracted(row, vendor)
            if sweep is None:
                raise ValueError(
                    f"{vendor}: cv_sweeps entry is missing one of "
                    f"{list(_FULL_CV_KEYS)}")
            bias = sweep.vgs_bias_v
            if bias is not None:
                if bias in seen:
                    raise ValueError(
                        f"{vendor}: duplicate cv_sweeps gate bias {bias:g} V")
                seen.add(bias)
            sweeps.append(sweep)
        return sorted(sweeps, key=lambda s: (s.vgs_bias_v is None,
                                             s.vgs_bias_v or 0.0))

    present = [k for k in _MULTI_CV_BY_VGS_KEYS if extracted.get(k) is not None]
    if not present:
        sweep = _cv_sweep_from_extracted(extracted, vendor)
        return [sweep] if sweep is not None else None
    if len(present) != len(_MULTI_CV_BY_VGS_KEYS):
        raise ValueError(
            f"{vendor}: multi-bias C(V) persist is partial (has "
            f"{present}); all of {list(_MULTI_CV_BY_VGS_KEYS)} are required")
    if extracted.get("curve_vds") is None:
        raise ValueError(
            f"{vendor}: multi-bias C(V) persist has no shared curve_vds")

    groups = [extracted[k] for k in _MULTI_CV_BY_VGS_KEYS]
    if all(isinstance(g, dict) for g in groups):
        key_sets = [set(g) for g in groups]
        if not (key_sets[0] == key_sets[1] == key_sets[2]):
            raise ValueError(
                f"{vendor}: *_by_vgs mappings use different gate-bias keys")
        labels = sorted(_bias_label(k, vendor, "ciss_curve_f_by_vgs")
                        for k in key_sets[0])
        rows = []
        for label in labels:
            key = next(k for k in key_sets[0]
                       if _bias_label(k, vendor, "ciss_curve_f_by_vgs")
                       == label)
            rows.append((label, [g[key] for g in groups]))
    elif all(isinstance(g, (list, tuple)) for g in groups):
        n = [len(g) for g in groups]
        if not (n[0] == n[1] == n[2]):
            raise ValueError(
                f"{vendor}: *_by_vgs lists have different lengths {n}")
        raw_labels = extracted.get("vgs_bias_v_by_sweep")
        if raw_labels is None:
            raw_labels = extracted.get("vgs_bias_v_list")
        if raw_labels is None:
            raise ValueError(
                f"{vendor}: list-form *_by_vgs has no vgs_bias_v_by_sweep "
                "labels; refusing to invent a bias axis")
        if not isinstance(raw_labels, (list, tuple)) \
                or len(raw_labels) != n[0]:
            raise ValueError(
                f"{vendor}: vgs_bias_v_by_sweep must be a list of "
                f"{n[0]} finite labels")
        labels = [_bias_label(k, vendor, "vgs_bias_v_by_sweep")
                  for k in raw_labels]
        rows = list(zip(labels, zip(*groups)))
    else:
        raise ValueError(
            f"{vendor}: *_by_vgs must be all mappings or all lists")

    if len(set(labels)) != len(labels):
        raise ValueError(f"{vendor}: duplicate multi-bias gate labels {labels}")

    raw_temp = extracted.get("temperature_c_by_sweep")
    if isinstance(raw_temp, dict):
        temps = {_bias_label(k, vendor, "temperature_c_by_sweep"):
                 _optional_float(v, "temperature_c_by_sweep", vendor)
                 for k, v in raw_temp.items()}

        def temp_for(_i, bias, _temps=temps):
            return _temps.get(bias)
    elif isinstance(raw_temp, (list, tuple)):
        if len(raw_temp) != len(rows):
            raise ValueError(
                f"{vendor}: temperature_c_by_sweep length does not match "
                "the bias sweeps")

        def temp_for(i, _bias, _raw=raw_temp):
            return _optional_float(
                _raw[i], "temperature_c_by_sweep", vendor)
    else:
        global_temp = _optional_float(
            extracted.get("temperature_c"), "temperature_c", vendor)

        def temp_for(_i, _bias, _temp=global_temp):
            return _temp

    v = extracted["curve_vds"]
    sweeps = []
    for i, (bias, (ciss, coss, crss)) in enumerate(rows):
        sweeps.append(_validate_cv_arrays(
            vendor, v, ciss, coss, crss,
            vgs_bias_v=bias, temperature_c=temp_for(i, bias)))
    return sorted(sweeps, key=lambda s: (s.vgs_bias_v is None,
                                         s.vgs_bias_v or 0.0))


def _full_cv_from_curve4(vendor: str):
    """Read persisted full Curve-4 arrays.

    Returns one ``_CVSweep`` for a legacy single-sweep persist, a list when
    a multi-bias set is present, or ``None`` when the arrays are absent.
    """
    try:
        extracted = load_output(vendor, "curve4_capacitances")["extracted"]
    except (FileNotFoundError, KeyError, TypeError):
        return None
    sweeps = _cv_sweeps_from_extracted(extracted, vendor)
    if sweeps is None:
        return None
    if len(sweeps) == 1:
        return sweeps[0]
    return sweeps


def _primary_cv_sweep(sweeps: list[_CVSweep]) -> _CVSweep:
    """Pick the legacy-compatible primary: Vgs=0 first, else closest to 0."""
    if not sweeps:
        raise ValueError("no C(V) sweep available")
    for sweep in sweeps:
        if sweep.vgs_bias_v is None or sweep.vgs_bias_v == 0.0:
            return sweep
    return min(sweeps, key=lambda s: abs(float(s.vgs_bias_v)))


def _load_cv_sweeps(vendor: str, cfg: dict, cond: dict, *,
                    vgs_bias: float | None = None):
    """Pick every available C(V) sweep.

    Returns ``(sweeps, source, data_limitation)``. Persisted full arrays win;
    a single persisted Vgs=0 sweep carries an explicit limitation, a genuine
    multi-bias set does not. The live AC re-sweep keeps the historical
    disclosure and is run at ``vgs_bias`` only when the caller supplies it.
    """
    full = _full_cv_from_curve4(vendor)
    if full is not None:
        sweeps = list(full) if isinstance(full, (list, tuple)) else [full]
        if len(sweeps) > 1:
            limitation = None
        elif sweeps[0].vgs_bias_v in (None, 0.0):
            limitation = _CV_SINGLE_BIAS_LIMITATION
        else:
            limitation = (
                "Only one persisted C(V) sweep at Vgs="
                f"{float(sweeps[0].vgs_bias_v):g} V is available; no "
                "multi-bias C-V set exists. No bias axis was invented from "
                "scalar equivalents.")
        return sweeps, "curve4_full_arrays", limitation
    live_bias = 0.0 if vgs_bias is None else float(vgs_bias)
    if not np.isfinite(live_bias):
        raise ValueError(f"{vendor}: vgs_bias must be finite")
    ciss_c, coss_c, crss_c = CapacitanceTestbench(
        f_meas=cond.get("F_meas")).run(
        Path(cfg["library"]).resolve(), cfg["subcircuit"],
        vds_min=cond["Vds_cap_min"], vds_max=cond["Vds_cap_max"],
        points=cond["Cap_points"],
        pin_roles=cfg.get("pins", {}).get("pin_order"),
        vgs_bias=live_bias)
    sweep = _validate_cv_arrays(
        vendor,
        np.asarray(ciss_c.x_data, dtype=float),
        np.asarray(ciss_c.y_data, dtype=float),
        np.asarray(coss_c.y_data, dtype=float),
        np.asarray(crss_c.y_data, dtype=float),
        vgs_bias_v=live_bias, temperature_c=None)
    return [sweep], "live_ac_sweep", _CV_DATA_LIMITATION


def _load_cv_sweep(vendor: str, cfg: dict, cond: dict):
    """Pick the C(V) source and return ``(v, ciss, coss, crss, src, limit)``.

    Backward-compatible single-sweep accessor: the primary sweep (Vgs=0 when
    present) is returned. Persisted full Curve-4 arrays win; otherwise the
    historical live AC re-sweep is retained verbatim with its disclosure.
    The full-array tuple limitation stays ``None`` exactly as before; the
    single-bias disclosure is carried by :func:`_load_cv_sweeps`.
    """
    sweeps, source, limitation = _load_cv_sweeps(vendor, cfg, cond)
    primary = _primary_cv_sweep(sweeps)
    if source == "curve4_full_arrays":
        limitation = None
    return (primary.v, primary.ciss, primary.coss, primary.crss,
            source, limitation)


def _decompose_cv(ciss, coss, crss):
    """Standard three-cap identity on one Vds grid; clipping is counted."""
    cgs = np.maximum(ciss - crss, 0.0)
    cds = np.maximum(coss - crss, 0.0)
    cdg = np.asarray(crss, dtype=float).copy()
    return cgs, cds, cdg, {
        "cgs": int(np.count_nonzero(ciss - crss < 0.0)),
        "cds": int(np.count_nonzero(coss - crss < 0.0)),
    }


def _fit_charge_tables(v, ciss, coss, crss, *, vbus,
                       forced_v=None, forbid_below=None,
                       cdg_v=None, cdg_c=None,
                       cdg_axis: str = CDG_AXIS_VGS0,
                       temperature_c: float | None = None,
                       vgs_bias_v: float | None = None) -> _ChargeTables:
    """Fit zero-anchored, monotone Q(V) tables from a full C(V) sweep.

    Shared by both data sources so the charge contract is identical: no
    max_points cap, the repo's 0.5% budget, and the DPT on-state breakpoint
    forced onto CDS/CDG. ``v`` is always a voltage axis, never an index axis.

    ``cdg_v`` / ``cdg_c`` optionally supply Cgd measured on a real Vdg axis
    (bias-aware sweep). Without them CDG uses the legacy Vgs=0 identity
    (Vdg=Vds) and ``cdg_axis`` records that provenance. Each emitted table
    also gets a measured ``max_cap_error_pct`` diagnostic.
    """
    cgs_c, cds_c, cdg_c_default, n_clipped = _decompose_cv(ciss, coss, crss)
    v_fit, curves = _cap_curve_to_vbus(v, (cgs_c, cds_c, cdg_c_default), vbus)
    cgs_fit = _fit_anchored_charge(v_fit, curves[0], RELTOL)
    cds_fit = _fit_anchored_charge(
        v_fit, curves[1], RELTOL, forced_v=forced_v, forbid_below=forbid_below)
    if cdg_v is not None and cdg_c is not None:
        cdg_axis_v = np.asarray(cdg_v, dtype=float).ravel()
        cdg_curve = np.asarray(cdg_c, dtype=float).ravel()
        if cdg_axis_v.size != cdg_curve.size or cdg_axis_v.size < 2:
            raise ValueError(
                "bias-aware CDG axis/curve need >= 2 matched samples")
        if not np.all(np.diff(cdg_axis_v) > 0):
            raise ValueError("bias-aware CDG axis must be strictly ascending")
        cdg_fit = _fit_anchored_charge(cdg_axis_v, cdg_curve, RELTOL)
    else:
        cdg_fit = _fit_anchored_charge(
            v_fit, curves[2], RELTOL,
            forced_v=forced_v, forbid_below=forbid_below)
    decision = cgs_linearity_decision(cgs_fit.v_full, cgs_fit.q_full, RELTOL)
    if decision.is_linear:
        # CGS_NSEG=1 (linear) [S-MOS]; the emitted C is the endpoint chord.
        cgs_pairs = [cgs_fit.pairs[0], cgs_fit.pairs[-1]]
    else:
        cgs_pairs = cgs_fit.pairs
    fits = {"cgs": cgs_fit, "cds": cds_fit, "cdg": cdg_fit}
    pairs = {"cgs": cgs_pairs, "cds": cds_fit.pairs, "cdg": cdg_fit.pairs}
    # Measured C-V diagnostic on the (V,Q) pairs actually emitted.
    for key, fit in fits.items():
        fits[key] = replace(
            fit, max_cap_error_pct=_cap_error_pct(fit, pairs[key]),
            cap_error_budget_pct=CAP_ERROR_BUDGET_PCT)
    return _ChargeTables(
        fits=fits,
        pairs=pairs,
        n_clipped=n_clipped,
        cgs_decision=decision,
        cdg_axis=cdg_axis,
        temperature_c=temperature_c,
        vgs_bias_v=vgs_bias_v,
        cap_error_budget_pct=CAP_ERROR_BUDGET_PCT,
    )


def _on_state_breakpoint(vgs, vds, vgs_on: float, vbus: float) -> dict:
    """DPT-derived CDS/CDG breakpoint: max Vds at >=95% of configured Vgs_on."""
    gate = np.asarray(vgs, dtype=float)
    drain = np.asarray(vds, dtype=float)
    mask = gate >= 0.95 * float(vgs_on)
    if not np.any(mask):
        raise RuntimeError(
            "golden DPT has no samples at >= 95% of configured Vgs_on")
    measured = float(np.max(drain[mask]))
    used = float(min(measured, float(vbus)))
    return {
        "measured_v": measured,
        "used_v": used,
        "clamped_to_vbus": measured > float(vbus),
        "rule": "max Vds where golden input[0] >= 0.95*vgs_on (from YAML), "
                "clamped to vbus so no breakpoint exceeds the bus",
    }


def _table_error(fit: _ChargeFit, pairs) -> float:
    return float(np.max(np.abs(
        np.interp(fit.v_full, [p[0] for p in pairs],
                  [p[1] for p in pairs]) - fit.q_full)))


def _fit_record(fit: _ChargeFit, pairs) -> dict:
    err = _table_error(fit, pairs)
    budget = fit.error_budget_c
    q_max = float(fit.q_full[-1])
    cap_pct = fit.max_cap_error_pct
    cap_budget = fit.cap_error_budget_pct
    return {
        "n_points": len(pairs),
        "n_segments": len(pairs) - 1,
        "max_abs_error_c": err,
        "error_budget_c": budget,
        "error_ratio": err / budget if budget > 0 else None,
        "tolerance_met": bool(err <= budget),
        "achieved_rel_tolerance": err / q_max if q_max > 0 else None,
        "max_cap_error_pct": cap_pct,
        "cap_error_budget_pct": cap_budget,
        "cap_error_tolerance_met": (
            None if cap_pct is None else bool(cap_pct <= cap_budget)),
        "first_breakpoint_v": float(pairs[0][0]),
        "last_breakpoint_v": float(pairs[-1][0]),
    }


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def _fidelity_metrics(waveform: dict, comparison: dict) -> dict:
    keys = ("tr_ns", "tf_ns", "e_on_uj", "e_off_uj")
    rel = {}
    for key in keys:
        value = comparison.get(key, {}).get("tier3_rel_err_pct")
        rel[key] = abs(float(value)) if _finite(value) else None
    return {
        "vds_rmse_pct": waveform.get("vds_rel_rmse_pct"),
        "id_rmse_pct": waveform.get("id_rel_rmse_pct"),
        "abs_rel_err_pct": rel,
    }


def _select_charge_candidate(candidates: dict) -> tuple[str, str]:
    """Select tier2 only on a strict, all-metric improvement over tables."""
    a = candidates["tables"]["fidelity"]
    b = candidates["tier2"]["fidelity"]
    complete = lambda f: (_finite(f["vds_rmse_pct"])
                          and _finite(f["id_rmse_pct"])
                          and all(_finite(x)
                                  for x in f["abs_rel_err_pct"].values()))
    if not complete(a) or not complete(b):
        return ("tables",
                "at least one candidate has an incomplete guard/failed DPT; "
                "tables retained")
    better = (b["vds_rmse_pct"] < a["vds_rmse_pct"]
              and b["id_rmse_pct"] < a["id_rmse_pct"]
              and all(b["abs_rel_err_pct"][k] < a["abs_rel_err_pct"][k]
                      for k in b["abs_rel_err_pct"]))
    if better:
        cap_a = candidates["tables"].get("max_cap_error_pct")
        cap_b = candidates["tier2"].get("max_cap_error_pct")
        if _finite(cap_a) and _finite(cap_b) and float(cap_b) > float(cap_a):
            return ("tables",
                    "tier2 charge improves every fidelity metric but its "
                    f"measured C-V error ({float(cap_b):.4g} pct) exceeds "
                    f"tables ({float(cap_a):.4g} pct); C-V gate retains "
                    "tables")
        return ("tier2",
                "tier2 charge strictly improves Vds RMSE, Id RMSE and "
                "abs(rel err) of tr/tf/e_on/e_off"
                + ("; C-V gate satisfied" if _finite(cap_a)
                   and _finite(cap_b) else "; C-V error not measured, "
                   "charge-only rule"))
    return ("tables",
            "tier2 charge does not strictly improve every recorded "
            "fidelity metric; tables retained")


_CHANNEL_RULE = (
    "FIDELITY-FIRST ONLY: hybrid is selected ONLY IF, for both charge modes, "
    "it strictly improves Vds RMSE, Id RMSE and abs(rel err) of "
    "tr/tf/e_on/e_off over the corresponding Tier-2-channel candidate on "
    "the same forced-2ns DPT guard. Adaptive speed is NOT measured in this "
    "selection step and is not fabricated: it is deferred until fidelity "
    "passes, and a caller holding measured adaptive results may re-evaluate "
    "with adaptive_results. Else Tier-2 channel."
)


def _select_channel_candidate(
    candidates: dict, *, adaptive_results: dict[str, dict] | None = None
) -> tuple[str, str, dict]:
    """Strict, fidelity-first hybrid gate; Tier-2 channel is the default."""
    speedups = adaptive_results or {}
    pairs = (("tables", "hybrid_tables"), ("tier2", "hybrid_tier2"))
    metrics = ("vds_rmse_pct", "id_rmse_pct")
    fidelity_keys = ("tr_ns", "tf_ns", "e_on_uj", "e_off_uj")
    evidence: dict = {"pairs": {}}
    fidelity_ok = True
    for base, hybrid in pairs:
        if base not in candidates or hybrid not in candidates:
            evidence["pairs"][hybrid] = {
                "compared_with": base, "strict_gain": False,
                "reason": "candidate missing (hybrid fit failed or not measured)",
            }
            fidelity_ok = False
            continue
        a, b = candidates[base]["fidelity"], candidates[hybrid]["fidelity"]
        complete = lambda f: (_finite(f["vds_rmse_pct"])
                              and _finite(f["id_rmse_pct"])
                              and all(_finite(f["abs_rel_err_pct"].get(k))
                                      for k in fidelity_keys))
        if not complete(a) or not complete(b):
            evidence["pairs"][hybrid] = {
                "compared_with": base, "strict_gain": False,
                "reason": "incomplete DPT guard on at least one candidate",
                "tier2_channel_completed": candidates[base].get(
                    "dpt_run", {}).get("completed"),
                "hybrid_completed": candidates[hybrid].get(
                    "dpt_run", {}).get("completed"),
            }
            fidelity_ok = False
            continue
        rows = {}
        for key in metrics:
            rows[key] = {"tier2_channel": a[key], "hybrid": b[key],
                         "strict_gain": b[key] < a[key]}
        for key in fidelity_keys:
            av, bv = a["abs_rel_err_pct"][key], b["abs_rel_err_pct"][key]
            rows[key] = {"tier2_channel": av, "hybrid": bv,
                         "strict_gain": bv < av}
        strict = all(row["strict_gain"] for row in rows.values())
        evidence["pairs"][hybrid] = {
            "compared_with": base, "metrics": rows, "strict_gain": strict,
            "tier2_channel_completed": candidates[base].get(
                "dpt_run", {}).get("completed"),
            "hybrid_completed": candidates[hybrid].get(
                "dpt_run", {}).get("completed"),
        }
        fidelity_ok = fidelity_ok and strict
    if not fidelity_ok:
        failed = []
        for hybrid, row in evidence["pairs"].items():
            if row["strict_gain"]:
                continue
            if "metrics" in row:
                keys = [k for k, v in row["metrics"].items()
                        if not v["strict_gain"]]
                failed.append(f"{hybrid} vs {row['compared_with']}: "
                              f"{','.join(keys)}")
            else:
                failed.append(f"{hybrid}: {row['reason']}")
        return ("tier2",
                "hybrid rejected: strict all-metric DPT fidelity gain not "
                "proven (" + "; ".join(failed) + "); Tier-2 channel retained",
                evidence)
    if not speedups:
        # run_vendor selects before any adaptive bench exists. Say so
        # explicitly instead of reporting a measurement that never happened.
        evidence["adaptive"] = {
            combo: {"status": "adaptive deferred; fidelity-first gate only"}
            for pair in pairs for combo in pair
        }
        return ("tier2",
                "hybrid strict DPT gain measured, but adaptive speed is "
                "deferred by the fidelity-first gate (not measured at "
                "selection time, never fabricated); Tier-2 channel retained",
                evidence)
    adaptive_ok = True
    adaptive_rows = {}
    for base, hybrid in pairs:
        for combo in (base, hybrid):
            row = speedups.get(combo)
            if not row or not row.get("completed") or not all(
                    _finite(row.get(k)) for k in
                    ("solver_speedup", "wall_speedup")):
                adaptive_ok = False
                adaptive_rows[combo] = {
                    "status": "adaptive not measured / incomplete"}
        if adaptive_ok:
            not_worse = all(
                speedups[hybrid][k] >= speedups[base][k]
                for k in ("solver_speedup", "wall_speedup"))
            adaptive_rows[hybrid] = {
                "compared_with": base,
                "tier2_channel": speedups[base],
                "hybrid": speedups[hybrid],
                "not_worse": not_worse,
            }
            adaptive_ok = not_worse
    evidence["adaptive"] = adaptive_rows
    if not adaptive_ok:
        return ("tier2",
                "hybrid strict DPT gain measured, but adaptive speedup is "
                "not measured or not no-worse for both hybrid charge modes; "
                "Tier-2 channel retained",
                evidence)
    return ("hybrid",
            "hybrid strictly improves every DPT fidelity metric in both "
            "charge modes and its measured adaptive speedup is not worse",
            evidence)
_RESIDUAL_RULE = (
    "EVIDENCE-GATED ONLY: tier2+ann_residual is selected only when both the "
    "tier2 and residual DPT runs are complete and the residual strictly "
    "improves Vds RMSE, Id RMSE, abs(rel err) of tr/tf/e_on/e_off and their "
    "mean fidelity score on the SAME selected charge-table emission. A "
    "missing fit, failed/incomplete DPT, tie or regression keeps the tier2 "
    "channel; no adaptive speed is claimed here."
)

_FIDELITY_KEYS = ("vds_rmse_pct", "id_rmse_pct",
                  "tr_ns", "tf_ns", "e_on_uj", "e_off_uj")


def _fidelity_values(fidelity: dict) -> dict:
    """Flatten one fidelity record into the six measured DPT values."""
    errors = fidelity["abs_rel_err_pct"]
    return {key: (fidelity[key] if key in ("vds_rmse_pct", "id_rmse_pct")
                  else errors[key]) for key in _FIDELITY_KEYS}


def _select_residual_candidate(
    candidates: dict, *, charge_mode: str,
    residual_mode: str = "tier2+ann_residual"
) -> tuple[str, str, dict]:
    """Fidelity-first gate for the optional ANN residual channel source."""
    base = charge_mode
    residual = residual_mode
    evidence: dict = {
        "charge_mode": charge_mode,
        "tier2_candidate": base,
        "residual_candidate": residual,
    }
    if base not in candidates or residual not in candidates:
        evidence["reason"] = ("residual candidate missing (fit failed or "
                              "DPT not measured)")
        return ("tier2",
                "ANN residual rejected: residual candidate missing or "
                "unmeasured; tier2 channel retained", evidence)

    a, b = candidates[base], candidates[residual]
    guard = {}
    for name, candidate in ((base, a), (residual, b)):
        info = candidate.get("dpt_run", {})
        guard[name] = {
            "completed": info.get("completed"),
            "trace_points": info.get("trace_points"),
            "iterations": info.get("iterations"),
            "t_last_s": info.get("t_last_s"),
            "error": info.get("error"),
        }
    evidence["completion"] = guard
    if not all(guard[name]["completed"] is True
               and (guard[name]["trace_points"] or 0) > 0
               for name in (base, residual)):
        return ("tier2",
                "ANN residual rejected: at least one DPT run is missing, "
                "failed or incomplete; tier2 channel retained", evidence)

    try:
        va = _fidelity_values(a["fidelity"])
        vb = _fidelity_values(b["fidelity"])
        if not all(_finite(va[k]) and _finite(vb[k]) for k in _FIDELITY_KEYS):
            raise ValueError("non-finite fidelity metric")
    except (KeyError, TypeError, ValueError):
        evidence["reason"] = "incomplete fidelity metric on at least one candidate"
        return ("tier2",
                "ANN residual rejected: incomplete fidelity metrics on a "
                "completed DPT run; tier2 channel retained", evidence)

    score_a = float(np.mean([va[k] for k in _FIDELITY_KEYS]))
    score_b = float(np.mean([vb[k] for k in _FIDELITY_KEYS]))
    rows = {
        key: {"tier2": va[key], "residual": vb[key],
              "strict_gain": vb[key] < va[key]}
        for key in _FIDELITY_KEYS
    }
    strict = all(row["strict_gain"] for row in rows.values())
    strict = strict and score_b < score_a
    evidence["fidelity"] = rows
    evidence["score"] = {"tier2": score_a, "residual": score_b,
                         "strict_gain": score_b < score_a}
    evidence["strict_gain"] = strict
    if strict:
        return (residual,
                "ANN residual selected: strict all-metric DPT fidelity and "
                "mean-score gain on the same charge tables", evidence)
    failed = [key for key, row in rows.items() if not row["strict_gain"]]
    if not score_b < score_a:
        failed.append("mean_fidelity_score")
    evidence["failed_metrics"] = failed
    return ("tier2",
            "ANN residual rejected: no strict DPT fidelity gain "
            f"({', '.join(failed)}); tier2 channel retained", evidence)


def _adaptive_guard_run(lib, subckt, pins, kw: dict, end: float,
                        temp: float, max_step: float | None) -> dict:
    """One netlist-identical run to read t_last (the bench returns medians)."""
    nb = NetlistBuilder(f"SPEEDUP_DPT_{subckt}")
    nb.include(lib)
    nb.v_dc("bus", "vbus", "0", kw["vbus"])
    nb.l("load", "vbus", "drain", kw["l_load"])
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
    nb.raw_line("Vgs gate 0 PWL(" + _pulse_train(
        1e-6, kw["first_pulse_width"], kw["pulse_width"],
        kw["pulse_period"], kw["n_pulses"], kw["edge_time"],
        kw["vgs_on"]) + ")")
    nb.x_instance("dut", subckt, pins, temperature=temp)
    directive = TranDirective(step_time=max_step or kw["edge_time"],
                              end_time=end, max_step=max_step)
    result = SpiceEngine(temperature=temp,
                         timeout_s=kw["timeout_s"]).run_transient(
        nb.render(), directive)
    t = np.asarray(result.get("time"), dtype=float)
    found = re.search(r"Total elapsed time:\s*([\d.]+)", result.log_text)
    return {
        "t_last_s": float(t[-1]) if t.size else None,
        "iterations": _log_iterations(result.log_text),
        "trace_points": int(t.size),
        "solver_elapsed_s": float(found.group(1)) if found else None,
    }


def _run_adaptive_benchmark(vendor: str, cfg: dict, dpt: dict, lib: Path,
                            temp: float) -> dict:
    """20-pulse adaptive bench with a 3-repeat + points/t_last rejection guard."""
    kw = {
        "vbus": dpt["vbus"], "vgs_on": dpt["vgs_on"],
        "i_load": dpt["i_load_target"], "l_load": dpt["l_load"],
        "first_pulse_width": dpt["t1_width"],
        "pulse_width": dpt["t2_width"],
        "pulse_period": dpt["t2_on"] - dpt["t1_on"],
        "n_pulses": 20, "edge_time": dpt["edge_time"],
        "max_step": None, "temperature": temp,
        "repeats": 3, "timeout_s": 90.0,
    }
    end = (1.0e-6 + (kw["n_pulses"] - 1) * kw["pulse_period"]
           + kw["pulse_width"] + 2e-6)
    golden_step = 100e-9 if vendor == "epc2216" else None
    pins_cfg = cfg.get("pins") or {}
    semantic = pins_cfg.get("pin_order_semantic")
    if semantic:
        golden_pins = resolve_pin_connections(
            Path(cfg["library"]).resolve(), cfg["subcircuit"], semantic)
    else:
        golden_pins = list(pins_cfg.get("pin_order") or [])
    sides = (
        ("golden", Path(cfg["library"]).resolve(), cfg["subcircuit"],
         golden_pins, golden_step),
        ("tier3_simplis", lib,
         f"{cfg['device']}_PIECESPICE_T3_simplis",
         _EMITTED_PIN_ROLES, None),
    )
    guard_text = (
        "3/3 repeats must return; median solver/wall/iterations/points > 0; "
        "identical-netlist guard run must reach t_last >= 0.999*end_time_s "
        "with points >= 0.5*median points; timeouts/exceptions fail the side")
    results = {}
    for label, library, subckt, pins, max_step in sides:
        pins = list(pins_cfg.get("pin_order") or []) if pins is None else pins
        bench, guard = {}, {}
        error = None
        t_wall = time.perf_counter()
        try:
            bench = run_inductive_benchmark(library, subckt, pins,
                                            **{**kw, "max_step": max_step})
            guard = _adaptive_guard_run(library, subckt, pins, kw, end,
                                        temp, max_step)
        except Exception as exc:  # failed repeats => no speedup, never faked
            error = f"{type(exc).__name__}: {exc}"[:200]
        median_ok = all(_finite(bench.get(k)) and bench[k] > 0
                        for k in ("wall_clock_s", "solver_elapsed_s",
                                  "iterations", "trace_points"))
        t_last = guard.get("t_last_s")
        guard_ok = (
            _finite(t_last) and t_last >= 0.999 * end
            and (guard.get("trace_points") or 0)
            >= 0.5 * (bench.get("trace_points") or 0)
            and (guard.get("iterations") or 0) > 0
            and bench.get("end_time_s") == end)
        row = {
            "library": str(library),
            "subcircuit": subckt,
            "max_step": max_step,
            "end_time_s": end,
            "repeats": kw["repeats"],
            "bench_wall_s": time.perf_counter() - t_wall,
            "guard_run": guard,
            "completed": bool(error is None and median_ok and guard_ok),
            "anti_fake_guard": guard_text,
        }
        for key in ("wall_clock_s", "solver_elapsed_s", "iterations",
                    "trace_points", "trace_names"):
            row[key] = bench.get(key)
        if error:
            row["error"] = error
        results[label] = row

    both = all(row["completed"] for row in results.values())
    notes = [
        "All timing rows are medians of completed runs only; failed repeats "
        "or guards carry completed=false and no speedup.",
    ]
    if golden_step is not None:
        notes.append(
            f"EPC golden uses a documented {golden_step:.0e} s max_step cap "
            "(fully adaptive golden timed out out-of-band); the ROM stays "
            "fully adaptive (max_step=None).")
    return {
        "conditions": {
            "n_pulses": kw["n_pulses"], "max_step_golden_s": golden_step,
            "max_step_simplis_s": None, "end_time_s": end,
            "repeats": kw["repeats"], "timeout_s": kw["timeout_s"],
            "topology": "clamped-inductive DPT, per-vendor config YAML",
        },
        **results,
        "solver_speedup": (
            results["golden"]["solver_elapsed_s"]
            / results["tier3_simplis"]["solver_elapsed_s"]
            if both else None),
        "wall_speedup": (
            results["golden"]["wall_clock_s"]
            / results["tier3_simplis"]["wall_clock_s"]
            if both else None),
        "epc_golden_fully_adaptive_attempt": None,
        "notes": notes,
    }


def _declared_cv_metadata(cfg: dict) -> tuple[float | None, float | None]:
    """Read optional ``simplis.temperature_c`` / ``vgs_bias_v`` declarations.

    Absent keys stay ``None``: the pipeline never infers a temperature or a
    gate-drive value from unrelated conditions.
    """
    simplis = cfg.get("simplis") or {}
    if not isinstance(simplis, dict):
        raise ValueError("config['simplis'] must be a mapping")
    return (
        _optional_float(simplis.get("temperature_c"),
                        "simplis.temperature_c", "config"),
        _optional_float(simplis.get("vgs_bias_v"),
                        "simplis.vgs_bias_v", "config"),
    )


def _select_cdg_data(sweeps, primary):
    """Pick a measured Cgd table, preferring a genuine bias-aware sweep.

    Returns ``(v, c, axis, source_bias_v, limitation)``. A bias-aware sweep
    supplies Cgd on the measured Vdg = Vds - Vgs axis (non-negative part
    only, because the anchored charge integral starts at 0 V). Without one,
    the legacy Vgs=0 Vds axis is kept and the limitation is returned.
    """
    biased = [s for s in sweeps
              if s.vgs_bias_v is not None and s.vgs_bias_v != 0.0]
    for sweep in sorted(biased, key=lambda s: -abs(float(s.vgs_bias_v))):
        bias = float(sweep.vgs_bias_v)
        axis, idx = np.unique(np.asarray(sweep.v, dtype=float) - bias,
                              return_index=True)
        crss = np.asarray(sweep.crss, dtype=float)[idx]
        keep = axis >= 0.0
        axis, crss = axis[keep], crss[keep]
        if axis.size >= 2:
            return axis, crss, CDG_AXIS_VDG, bias, None
    return (primary.v, primary.crss, CDG_AXIS_VGS0, None,
            _CDG_VGS0_LIMITATION)


def _charge_meta_from_details(details: dict) -> dict:
    """Builder-facing C-V provenance assembled from measured details."""
    cap = details.get("cap_error") or {}
    cvb = details.get("cv_bias") or {}
    limitations = [text for text in (details.get("cdg_data_limitation"),
                                     cvb.get("data_limitation")) if text]
    return {
        "cdg_axis": details.get("cdg_axis"),
        "vgs_bias_v": cvb.get("vgs_bias_v"),
        "temperature_c": cvb.get("temperature_c"),
        "max_cap_error_pct": cap.get("max_cap_error_pct_all"),
        "cap_error_budget_pct": cap.get("budget_pct"),
        "data_limitation": "; ".join(limitations) or None,
    }


def _charge_cap_error_from_details(details: dict) -> float | None:
    """Measured C-V error for table emissions; ``None`` when not recorded.

    A missing diagnostic degrades to ``None`` -- never a fabricated 0 -- so
    the charge gate keeps its strict fidelity-only rule.
    """
    cap_meta = details.get("cap_error") or {}
    return cap_meta.get("max_cap_error_pct_all")


def _build_spec(
    vendor: str, *, dpt_vds=None, dpt_vgs=None,
    vgs_on: float | None = None, vbus: float | None = None,
    temperature_c: float | None = None,
    vgs_bias_v: float | None = None,
) -> tuple[PWLModelSpec, dict, HybridChannelFit | None]:
    """Build the spec from extracted curves plus one full C(V) sweep.

    Curve-1/3/6/7 assembly follows ``pipeline.fitting.fit_pwl_model`` exactly;
    the charge side replaces Tier-2's heuristic C_ISS/C_RSS scalars with the
    documented three-cap decomposition of one C(V) sweep. That sweep is the
    persisted full Curve-4 arrays when present, otherwise the existing live
    AC re-extraction (disclosed as a data limitation). The golden DPT trace
    (already captured by ``run_vendor``) supplies the data-derived on-state
    breakpoint and the bus cap; no extra simulation is needed for the fit.
    """
    cfg = load_input(vendor)
    cond = cfg["conditions"]
    device = cfg["device"]

    # ---- Channel (Curve 1), Rds(on) (Curve 3), Vth (Curve 7), diode (6) ----
    c1 = fitting._curve(vendor, "curve1_output_characteristics")["extracted"]
    prefix = "curve_id_vgs_"
    vgs_values = sorted(
        float(k[len(prefix):-2]) for k in c1 if k.startswith(prefix))
    rows = []
    ref_len = None
    for v in vgs_values:
        key = next(k for k in c1 if k.startswith(prefix)
                   and abs(float(k[len(prefix):-2]) - v) < 1e-9)
        y = fitting._arr(c1[key])
        xs = np.linspace(0.0, float(cond["Vds_sweep_max"]), len(y))
        xs, y = trim_trailing_invalid(xs, y)
        if ref_len is None:
            ref_len = len(y)
        rows.append(np.interp(
            np.linspace(0, float(cond["Vds_sweep_max"]), ref_len), xs, y))
    if ref_len is None:
        raise ValueError(f"{vendor}: Curve 1 has no Vgs family rows")
    breakpoints_vds = list(np.linspace(
        0.0, float(cond["Vds_sweep_max"]), ref_len))
    breakpoints_id = [list(map(float, r)) for r in rows]

    rds_on = 0.005  # fitting.py fallback when curve3 is absent
    try:
        c3 = fitting._curve(vendor, "curve3_rds_on")["extracted"]
        rx = fitting._arr(c3["curve_id"])
        ry = fitting._arr(c3["curve_rds_mohm"]) * 1e-3
        if len(rx):
            rds_on = float(np.median(ry[ry > 0]))
    except FileNotFoundError:
        pass

    v_th = 0.0  # fitting.py fallback when curve7 is absent
    try:
        v_th = float(load_output(vendor, "curve7_threshold_voltage")
                     ["extracted"]["vth_v"])
    except (FileNotFoundError, KeyError, ValueError):
        pass

    b_vsd: list = []
    b_isd: list = []
    try:
        c6 = load_output(vendor, "curve6_body_diode")["extracted"]
        xv, yv = trim_trailing_invalid(fitting._arr(c6["curve_vsd"]),
                                       np.abs(fitting._arr(c6["curve_isd"])))
        b_vsd, b_isd = xv.tolist(), yv.tolist()
    except FileNotFoundError:
        pass

    # ---- C(V) sweeps: persisted multi-bias set, else one Vgs=0 sweep ------
    # Persisted full Curve-4 arrays win. When absent, retain the previous
    # live AC re-sweep and disclose the data limitation; never synthesize a
    # sweep from the scalar-equivalent fields. A genuine multi-bias persist
    # yields one _CVSweep per gate bias; a legacy single Vgs=0 persist keeps
    # the historical single-sweep path and carries an explicit limitation.
    sweeps, cv_source, data_limitation = _load_cv_sweeps(
        vendor, cfg, cond, vgs_bias=vgs_bias_v)
    primary = _primary_cv_sweep(sweeps)
    v, ciss, coss, crss = primary.v, primary.ciss, primary.coss, primary.crss
    if v.size < 2 or not np.all(np.diff(v) > 0):
        raise RuntimeError(f"{vendor}: AC capacitance sweep has no valid grid")
    cdg_v, cdg_c, cdg_axis, cdg_source_bias, cdg_limitation = (
        _select_cdg_data(sweeps, primary))
    effective_temp = (temperature_c if temperature_c is not None
                      else primary.temperature_c)
    effective_bias = primary.vgs_bias_v

    # ---- Documented SIMPLIS Level-2 charge tables -------------------------
    # Interior breakpoints: OUR greedy farthest-point rule (SIMPLIS does not
    # document its selection rule). No max_points cap: fit until the repo's
    # 0.5% charge budget is actually met and record what was achieved.
    if vbus is None:
        raise ValueError(f"{vendor}: simplis spec build requires DPT Vbus")
    on_bp = (_on_state_breakpoint(dpt_vgs, dpt_vds, vgs_on, vbus)
             if dpt_vds is not None and dpt_vgs is not None
             and vgs_on is not None and vbus is not None else None)
    forced = on_bp["used_v"] if on_bp else None
    forbid = forced if on_bp and not on_bp["clamped_to_vbus"] else None
    tables = _fit_charge_tables(
        v, ciss, coss, crss, vbus=vbus,
        forced_v=forced, forbid_below=forbid,
        cdg_v=cdg_v, cdg_c=cdg_c, cdg_axis=cdg_axis,
        temperature_c=effective_temp, vgs_bias_v=effective_bias)
    cgs_fit = tables.fits["cgs"]
    cds_fit = tables.fits["cds"]
    cdg_fit = tables.fits["cdg"]
    decision = tables.cgs_decision
    cgs_pairs = tables.pairs["cgs"]
    cds_pairs = tables.pairs["cds"]
    cdg_pairs = tables.pairs["cdg"]
    n_clipped = tables.n_clipped

    cap_ref = _cap_ref_vds(cond)
    c_eq = {
        name: integrate_charge(v, curve, v_ref=cap_ref).c_eq_charge
        for name, curve in (("ciss", ciss), ("coss", coss), ("crss", crss))
    }
    spec = PWLModelSpec(
        device_name=device,
        tier=3,
        breakpoints_vds=breakpoints_vds,
        breakpoints_id=breakpoints_id,
        vgs_values=[float(x) for x in vgs_values],
        q_oss_vs_v=[a for a, _ in cds_pairs],
        q_oss_charge_c=[b for _, b in cds_pairs],
        q_iss_vs_v=[a for a, _ in cgs_pairs],
        q_iss_charge_c=[b for _, b in cgs_pairs],
        q_rss_vs_v=[a for a, _ in cdg_pairs],
        q_rss_charge_c=[b for _, b in cdg_pairs],
        rds_on=rds_on,
        v_th=v_th,
        body_diode_vsd=b_vsd,
        body_diode_isd=b_isd,
        metadata={
            **_PARASITICS,
            "c_iss_lin": c_eq["ciss"],
            "c_oss_lin": c_eq["coss"],
            "c_rss_lin": c_eq["crss"],
        },
    )

    # Fit the measured hinge_c channel to the SAME Curve-1 rows. Every vendor
    # goes through _build_spec; the fit is a candidate only, never a default.
    try:
        hybrid_fit = fit_hybrid_channel(
            spec.vgs_values, spec.breakpoints_vds, spec.breakpoints_id,
            rel_tolerance=RELTOL)
        # vds_knees is one identical shared-knee row per Vgs family; record
        # that list once and assert the rows agree (no silent divergence).
        knees_v = list(hybrid_fit.vds_knees[0])
        assert all(list(row) == knees_v for row in hybrid_fit.vds_knees), (
            f"{vendor}: hybrid vds_knees rows are not identical")
        hybrid_provenance = {
            "status": "fitted",
            "fit_kind": hybrid_fit.meta.get("kind"),
            "n_knees": hybrid_fit.meta.get("n_knees"),
            "rel_tolerance": hybrid_fit.meta.get("rel_tolerance"),
            "achieved_rel_rmse": hybrid_fit.meta.get("achieved_rel_rmse"),
            "max_abs_fit_error_a": hybrid_fit.max_abs_fit_error_a,
            "stop_criterion": hybrid_fit.meta.get("stop_criterion"),
            "smoothing_width_derivation": hybrid_fit.meta.get(
                "smoothing_width_derivation"),
            "knee_selection": hybrid_fit.meta.get("knee_selection"),
            "knee_scope": hybrid_fit.meta.get("knee_scope"),
            "knees_v": knees_v,
            "smoothing_width_v": hybrid_fit.smoothing_width_v,
            "source_rows": {
                "vgs_values": list(spec.vgs_values),
                "n_vds": len(spec.breakpoints_vds),
                "target": "Curve-1 breakpoints_id (same spec as Tier-2 channel)",
            },
            "expression_ascii": hybrid_fit.expression.isascii(),
            "expression_chars": len(hybrid_fit.expression),
            "known_limitation": hybrid_fit.meta.get("known_limitation"),
        }
    except Exception as exc:
        hybrid_fit = None
        hybrid_provenance = {
            "status": "fit_failed",
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "rel_tolerance": RELTOL,
            "source_rows": {
                "n_vgs": len(spec.vgs_values),
                "n_vds": len(spec.breakpoints_vds),
            },
        }

    fits = {"cgs": cgs_fit, "cds": cds_fit, "cdg": cdg_fit}
    emitted_pairs = {"cgs": cgs_pairs, "cds": cds_pairs, "cdg": cdg_pairs}
    details = {
        "cgs_decision": decision.as_dict(),
        "nseg": {k: len(p) - 1 for k, p in emitted_pairs.items()},
        "table_points": {k: len(p) for k, p in emitted_pairs.items()},
        "max_abs_charge_error_c": {
            k: _table_error(fits[k], emitted_pairs[k]) for k in fits},
        "charge_fit": {
            k: _fit_record(fits[k], emitted_pairs[k]) for k in fits},
        "anchor": {
            "v0_v": 0.0, "q0_c": 0.0,
            "v_min_measured_v": float(v[0]),
            "first_segment_slope_f": {
                k: float((f.q_full[1] - f.q_full[0]) /
                         (f.v_full[1] - f.v_full[0]))
                for k, f in fits.items()},
        },
        "on_state_breakpoint": on_bp,
        "vbus_cap_v": float(vbus) if vbus is not None else None,
        "rel_tolerance": RELTOL,
        "max_points_cap": None,
        "ac_sweep": {"v_min_v": float(v[0]), "v_max_v": float(v[-1]),
                     "points": int(v.size),
                     "f_meas_hz": float(cond.get("F_meas", F_MEAS_DEFAULT))},
        "clipped_delta_points": n_clipped,
        "cap_ref_vds": cap_ref,
        "cap_error": {
            "method": "finite-difference dQ/dV of the emitted (V,Q) pairs "
                      "vs measured C(v) on the common measured grid; relative "
                      "to local C, denominator clipped at 0.1% of peak C",
            "budget_pct": CAP_ERROR_BUDGET_PCT,
            "max_cap_error_pct": {
                k: fits[k].max_cap_error_pct for k in fits},
            "max_cap_error_pct_all": max(
                (v for v in (fits[k].max_cap_error_pct for k in fits)
                 if v is not None), default=None),
            "tolerance_met": {
                k: (None if fits[k].max_cap_error_pct is None else
                    bool(fits[k].max_cap_error_pct
                         <= fits[k].cap_error_budget_pct))
                for k in fits},
        },
        "cdg_axis": cdg_axis,
        "cdg_source_bias_v": cdg_source_bias,
        "cdg_data_limitation": cdg_limitation,
        "cv_bias": {
            "source": cv_source,
            "n_sweeps": len(sweeps),
            "biases_v": [s.vgs_bias_v for s in sweeps],
            "vgs_bias_v": effective_bias,
            "declared_config_vgs_bias_v": vgs_bias_v,
            "temperature_c": effective_temp,
            "declared_config_temperature_c": temperature_c,
            "multi_bias_present": len(sweeps) > 1,
            "data_limitation": data_limitation,
        },
        "curve4_cv": {
            "source": cv_source,
            "full_arrays_present": cv_source == "curve4_full_arrays",
            "array_keys": (list(_FULL_CV_KEYS)
                           if cv_source == "curve4_full_arrays" else []),
            "n_sweeps": len(sweeps),
            "biases_v": [s.vgs_bias_v for s in sweeps],
            "data_limitation": data_limitation,
        },
        "breakpoint_formula": "greedy pwl_charge_segments farthest-point rule "
        "on anchored Q(V); no point cap; CDS/CDG forced DPT on-state "
        "breakpoint; no breakpoint above vbus",
        "hybrid_channel": hybrid_provenance,
    }
    return spec, details, hybrid_fit


def _run_emitted_dpt(lib_path: Path, vendor: str):
    """Run the emitted simplis subcircuit in the vendor's DPT bench.

    Mirrors ``pipeline.tier3._run_emitted_dpt`` (same bench, solver options
    and metrics) but binds the variant-specific subcircuit name.
    """
    dpt = _dpt_conditions(vendor)
    cfg = load_input(vendor)
    temp = float(cfg["conditions"]["temperature"])
    subckt = f"{cfg['device']}_PIECESPICE_T3_simplis"
    nb = NetlistBuilder(f"T3_DPT_{vendor}_simplis")
    nb.include(lib_path)
    nb.v_dc("bus", "vbus", "0", dpt["vbus"])
    nb.l("load", "vbus", "drain", dpt["l_load"])
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
    nb.raw_line("Vgs gate 0 PWL(" + _pulse_train(
        dpt["t1_on"], dpt["t1_width"], dpt["t2_width"],
        dpt["t2_on"] - dpt["t1_on"], 2, dpt["edge_time"],
        dpt["vgs_on"]) + ")")
    nb.x_instance("dut", subckt, _EMITTED_PIN_ROLES, temperature=temp)
    t0 = time.perf_counter()
    res = SpiceEngine(temperature=temp, timeout_s=120.0).run_transient(
        nb.render(), TranDirective(step_time=dpt["max_step"],
                                   end_time=dpt["t_end"],
                                   max_step=dpt["max_step"]))
    wall = time.perf_counter() - t0
    i_d, current_def = _device_current(res)
    run = DoublePulseRun(
        time=np.asarray(res.get("time")),
        v_ds=np.asarray(res.get("V(drain)")),
        i_d=i_d,
        v_gs=np.asarray(res.get("V(gate)")),
        v_bus=dpt["vbus"], i_load=dpt["i_load_target"],
        l_load=dpt["l_load"], vgs_on=dpt["vgs_on"],
        wall_clock_s=float(wall), t2_on=dpt["t2_on"])
    return run, res.log_text, current_def


def _emitted_static_rmse(vendor: str, lib: Path):
    """Measured Curve-1 full-sweep rel-RMSE of the emitted subcircuit."""
    rows = fitting.compare_tier2_vs_golden_full(vendor, lib_path=lib)
    per_curve = [
        float(compute_curve_error(np.asarray(golden), np.asarray(cand))
              .relative_rmse_pct)
        for _, _, golden, cand in rows
    ]
    return {
        "mean_rel_rmse_pct": float(np.mean(per_curve)),
        "per_curve_rel_rmse_pct": per_curve,
        "n_curves": len(per_curve),
        "n_points_per_curve": int(rows[0][1].size) if rows else 0,
    }


def _recorded_tier2_static(vendor: str) -> float | None:
    """Tier-2's recorded mean full-sweep rel-RMSE (never recomputed/faked)."""
    path = ROOT / "reports" / "comparison" / vendor / \
        "full_sweep_comparison.json"
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    return float(np.mean([r["rel_rmse_pct"] for r in rows]))


def _residual_fit_provenance(fit: AnnResidualFit) -> dict:
    """Offline fit evidence only; no DPT value is inferred from this fit."""
    diag = fit.diagnostics
    return {
        "status": "fitted",
        "source": "pipeline.ann_fit.hybrid_ann_fit_for_vendor: residual_fit "
                  "only; its ANN charge fit is not used by simplis",
        "sample_counts": {
            "train": diag.get("train_samples"),
            "curve1": diag.get("curve1_samples"),
            "dpt": diag.get("dpt_samples"),
        },
        "activation": fit.activation,
        "neurons": fit.neurons,
        "v_th_v": fit.v_th,
        "alpha": fit.alpha,
        "fit_values": {
            "residual_rel_rmse": diag.get("residual_rel_rmse"),
            "residual_rms_a": diag.get("residual_rms_a"),
            "target_rms_a": diag.get("target_rms_a"),
            "zero_vds_max_abs_a": diag.get("zero_vds_max_abs_a"),
            "cutoff_attenuation": diag.get("cutoff_attenuation"),
            "analytic_vs_finite_difference": diag.get(
                "analytic_vs_finite_difference"),
        },
        "provenance": fit.provenance,
    }


def _measure_candidate(candidate_lib: Path, vendor: str, traj, dpt: dict,
                       golden: dict, tier2: dict) -> dict:
    """One forced-2ns DPT with the SAME completion guard for every candidate."""
    info = {"completed": False, "trace_points": 0,
            "golden_trace_points": int(len(traj.time)),
            "iterations": None, "t_last_s": None,
            "forced_2ns_wall_s": None,
            "error": None, "waveform": {},
            "current_definition": None}
    wave, rom_metrics, log = {}, {}, ""
    try:
        run, log, current_def = _run_emitted_dpt(candidate_lib, vendor)
        info["current_definition"] = current_def
        info["forced_2ns_wall_s"] = float(run.wall_clock_s)
        n3 = int(len(run.time))
        it3 = _log_iterations(log)
        t_last = float(run.time[-1]) if n3 else 0.0
        complete = (n3 >= 0.5 * len(traj.time)
                    and t_last >= 0.999 * dpt["t_end"])
        info.update(trace_points=n3, iterations=it3, t_last_s=t_last)
        if complete:
            info["completed"] = True
            wave = _waveform_errors(run, traj)
            info["waveform"] = wave
            rom_metrics = extract_dpt_metrics(run)
        else:
            info["error"] = (
                f"incomplete: {n3} trace points, t_last={t_last:.3g} s")
            print(f"[tier3/simplis] {vendor}/{candidate_lib.stem}: DPT "
                  f"incomplete ({n3} points, t_last={t_last:.3g} s)",
                  flush=True)
    except Exception as exc:  # solver failures reported, never fabricated
        log = f"{type(exc).__name__}: {exc}"
        info["error"] = log[:200]
        print(f"[tier3/simplis] {vendor}/{candidate_lib.stem}: DPT failed: "
              f"{type(exc).__name__}", flush=True)

    comparison = {}
    for m in METRICS:
        g, t3 = golden.get(m), rom_metrics.get(m)
        row = {"golden": g, "tier3": t3,
               "tier3_rel_err_pct": _rel_err(t3, g)}
        if m in tier2:
            row["tier2"] = tier2[m]
            row["tier2_rel_err_pct"] = _rel_err(tier2[m], g)
        comparison[m] = row
    return {
        "library": str(candidate_lib.relative_to(ROOT)),
        "dpt_run": info,
        "dpt_comparison": comparison,
        "waveform": wave,
        "fidelity": _fidelity_metrics(wave, comparison),
        "spice_log_tail": "\n".join(log.splitlines()[-6:]),
    }


def run_vendor(vendor: str) -> dict:
    """Build and measure all channel/charge candidates; persist the winner."""
    t0 = time.perf_counter()
    traj = capture_trajectory(vendor)
    capture_wall = time.perf_counter() - t0
    cfg = load_input(vendor)
    dpt = _dpt_conditions(vendor)
    temp = float(cfg["conditions"]["temperature"])
    declared_temp, declared_bias = _declared_cv_metadata(cfg)

    spec, details, hybrid_fit = _build_spec(
        vendor, dpt_vds=traj.input[1], dpt_vgs=traj.input[0],
        vgs_on=dpt["vgs_on"], vbus=dpt["vbus"],
        temperature_c=declared_temp, vgs_bias_v=declared_bias)
    fit_wall = time.perf_counter() - t0 - capture_wall
    charge_meta = _charge_meta_from_details(details)
    # Evidence candidate only: consume the residual fit and nothing else
    # from the hybrid ANN fit (its ANN charge fit is NEVER used here).
    t_ann = time.perf_counter()
    try:
        residual_fit = hybrid_ann_fit_for_vendor(vendor)["residual_fit"]
        if not isinstance(residual_fit, AnnResidualFit):
            raise TypeError("hybrid fit did not return an AnnResidualFit")
        residual_fit_provenance = _residual_fit_provenance(residual_fit)
    except Exception as exc:
        residual_fit = None
        residual_fit_provenance = {
            "status": "fit_failed",
            "reason": f"{type(exc).__name__}: {exc}"[:200],
        }
    ann_fit_wall = time.perf_counter() - t_ann

    lib = ROOT / "reports" / "models" / \
        f"{vendor}_PIECESPICE_T3_simplis.lib"
    lib.parent.mkdir(parents=True, exist_ok=True)
    modes = ("tables", "tier2")
    if hybrid_fit is not None:
        modes += ("hybrid_tables", "hybrid_tier2")
    texts = {
        mode: build_simplis_subcircuit(
            spec, charge_mode=mode.removeprefix("hybrid_"),
            channel_fit=hybrid_fit if mode.startswith("hybrid_") else None,
            clamp_charge_above_last=True,
            charge_meta=charge_meta)
        for mode in modes}
    for mode, text in texts.items():
        if not text.isascii():
            raise RuntimeError(f"{vendor}: {mode} netlist is not ASCII")
    candidate_libs = {
        mode: lib.with_name(f"{lib.stem}_{mode}_charge.lib")
        for mode in texts}
    for mode, text in texts.items():
        candidate_libs[mode].write_text(text, encoding="ascii")

    # trajectory output[0] is the fitted device current I(Lload)-I(Dclamp).
    golden_run = DoublePulseRun(
        time=traj.time, v_ds=traj.input[1],
        i_d=np.abs(np.asarray(traj.output[0], dtype=float)),
        v_gs=traj.input[0], v_bus=dpt["vbus"],
        i_load=dpt["i_load_target"], l_load=dpt["l_load"],
        vgs_on=dpt["vgs_on"], wall_clock_s=capture_wall,
        t2_on=dpt["t2_on"])
    golden = extract_dpt_metrics(golden_run)
    current_defs = {"golden": _DEVICE_CURRENT, "tier3": None}

    # Same forced-2ns DPT bench and completion guard for EVERY candidate.
    recorded = ROOT / "reports" / "comparison" / vendor / \
        "dpt_metrics.json"
    tier2_metrics = json.loads(recorded.read_text(encoding="utf-8"))[
        "tier2_metrics"] if recorded.exists() else {}
    candidates = {
        mode: _measure_candidate(
            candidate_lib, vendor, traj, dpt, golden, tier2_metrics)
        for mode, candidate_lib in candidate_libs.items()
    }
    # Measured C-V diagnostic for the candidates that emit the fitted Q(V)
    # tables. The Tier-2 charge block is a different emission and does not
    # have this diagnostic; it stays None (never fabricated), so the charge
    # gate falls back to its strict fidelity-only rule.
    tables_cap_error = _charge_cap_error_from_details(details)
    for mode, row in candidates.items():
        row["max_cap_error_pct"] = (
            tables_cap_error if mode in ("tables", "hybrid_tables") else None)

    selection, reason = _select_charge_candidate(candidates)
    channel_selection, channel_reason, channel_evidence = (
        _select_channel_candidate(candidates))
    final = (("hybrid_" if channel_selection == "hybrid" else "")
             + selection)
    if final not in candidates:  # never select an unmeasured candidate
        channel_selection = "tier2"
        channel_reason = "hybrid candidate missing; Tier-2 channel retained"
        final = selection

    # The residual is compared ONLY against the existing tier2 channel on
    # the charge emission that the unchanged charge gate already selected.
    residual_mode = "tier2+ann_residual"
    if residual_fit is None:
        residual_selection = "tier2"
        residual_reason = ("ANN residual rejected: "
                           + residual_fit_provenance["reason"])
        residual_evidence = residual_fit_provenance
    elif channel_selection != "tier2":
        residual_selection = "tier2"
        residual_reason = ("ANN residual not compared: an accepted hinge_c "
                           "channel is outside the tier2-only residual gate")
        residual_evidence = {"status": "not_compared"}
    else:
        residual_lib = lib.with_name(
            f"{lib.stem}_hybrid_ann_residual_{selection}_charge.lib")
        residual_text = build_simplis_subcircuit(
            spec, charge_mode=selection, residual_fit=residual_fit,
            clamp_charge_above_last=True, charge_meta=charge_meta)
        if not residual_text.isascii():
            raise RuntimeError(f"{vendor}: residual netlist is not ASCII")
        residual_lib.write_text(residual_text, encoding="ascii")
        texts[residual_mode] = residual_text
        candidate_libs[residual_mode] = residual_lib
        candidates[residual_mode] = _measure_candidate(
            residual_lib, vendor, traj, dpt, golden, tier2_metrics)
        residual_selection, residual_reason, residual_evidence = (
            _select_residual_candidate(candidates, charge_mode=selection))
    if residual_selection == residual_mode:
        final = residual_mode
    lib.write_text(texts[final], encoding="ascii")
    current_defs["tier3"] = candidates[final]["dpt_run"].get(
        "current_definition")

    t1 = time.perf_counter()
    static = _emitted_static_rmse(vendor, lib)
    static_wall = time.perf_counter() - t1
    t2_static = _recorded_tier2_static(vendor)
    static_delta = (static["mean_rel_rmse_pct"] - t2_static
                    if t2_static is not None else None)

    # Forced-2ns runtime: fidelity control, NOT the adaptive speed claim.
    bench_kw = {
        "vbus": dpt["vbus"], "vgs_on": dpt["vgs_on"],
        "i_load": dpt["i_load_target"], "l_load": dpt["l_load"],
        "first_pulse_width": dpt["t1_width"],
        "pulse_width": dpt["t2_width"],
        "pulse_period": dpt["t2_on"] - dpt["t1_on"],
        "n_pulses": 2, "edge_time": dpt["edge_time"],
        "max_step": dpt["max_step"], "temperature": temp,
        "repeats": 3, "timeout_s": 120.0,
    }
    t1 = time.perf_counter()
    golden_time = run_inductive_benchmark(
        Path(cfg["library"]).resolve(), cfg["subcircuit"],
        traj.pin_roles, **bench_kw)
    bench_wall_golden = time.perf_counter() - t1
    t1 = time.perf_counter()
    try:
        rom_time = run_inductive_benchmark(
            lib, f"{cfg['device']}_PIECESPICE_T3_simplis",
            _EMITTED_PIN_ROLES, **bench_kw)
    except Exception as exc:  # reported, never fabricated
        rom_time = {"wall_clock_s": None, "solver_elapsed_s": None,
                    "iterations": None, "trace_points": 0,
                    "error": str(exc)[:200]}
    bench_wall_rom = time.perf_counter() - t1
    bench_complete = ((rom_time.get("iterations") or 0) >=
                      0.5 * (golden_time["iterations"] or 0)
                      and (rom_time.get("trace_points") or 0) >=
                      0.5 * (golden_time["trace_points"] or 0))

    adaptive = _run_adaptive_benchmark(vendor, cfg, dpt, lib, temp)
    adaptive["selected_candidate"] = final
    adaptive["scope"] = ("final selected candidate only; no adaptive runs "
                         "for non-selected candidates; both channel gates "
                         "are fidelity-first, adaptive deferred")

    dpt_fit = {}
    dpt_held_out = {}
    for key in (selection, residual_mode):
        if key not in candidates:
            continue
        row = candidates[key]
        dpt_fit[key] = row["fidelity"]
        dpt_held_out[key] = {
            "waveform": row["waveform"],
            "dpt_comparison": row["dpt_comparison"],
            "dpt_run": {k: row["dpt_run"].get(k) for k in (
                "completed", "trace_points", "iterations", "t_last_s",
                "error")},
        }

    record = {
        "schema": "piecespice.tier3/2.0",
        "vendor": vendor,
        "variant": "simplis",
        "methods": [
            "SIMPLIS Level-2 (CGS/CDS/CDG V-Q tables; S-MOS, S-APPB)",
            "Tier-2 CPWL channel + body diode + parasitics (unchanged)",
        ] + (["Gated hybrid hinge_c channel (measured candidates; default off)"]
             if any(m.startswith("hybrid_") for m in candidates) else [])
        + (["Evidence-gated ANN residual current source; C0 charge tables "
            "unchanged"] if residual_mode in candidates else []),
        "algorithm": {
            **details,
            "channel": (
                "hybrid hinge_c (models.hybrid_channel.fit_hybrid_channel); "
                "fidelity-first gate, adaptive speed deferred"
                if channel_selection == "hybrid" else
                "reused models.netlist_builder._channel_expression "
                "(hybrid rejected by strict gate)"),
            "charge_candidate_selected": selection,
            "channel_candidate_selected": channel_selection,
            "residual_channel_candidate_selected": residual_selection,
            "residual_channel_charge_mode": selection,
        },
        "function_fit": {
            "static_curve_rmse_rel": static["mean_rel_rmse_pct"],
            "emitted_static_curve_rmse_rel": static["mean_rel_rmse_pct"],
            "emitted_static_per_curve_pct": static["per_curve_rel_rmse_pct"],
            "emitted_static_samples": static["n_points_per_curve"],
            "tier2_recorded_mean_rel_rmse_pct": t2_static,
            "static_delta_pp_vs_tier2": static_delta,
            "static_rmse_scope": "channel_only_dc_capacitors_are_open",
        },
        "dpt_run": candidates[final]["dpt_run"],
        "current_definitions": current_defs,
        "dpt_waveform": candidates[final]["waveform"],
        "dpt_comparison": candidates[final]["dpt_comparison"],
        "charge_candidates": {
            "modes": list(candidates),
            **candidates,
            "selection": selection,
            "selection_reason": reason,
            "selection_rule": "tier2 only if it strictly improves Vds RMSE, "
            "Id RMSE and abs(rel err) of tr/tf/e_on/e_off; else tables",
        },
        "channel_candidates": {
            "modes": list(candidates),
            **candidates,
            "selection": channel_selection,
            "selection_reason": channel_reason,
            "selection_rule": _CHANNEL_RULE,
            "evidence": channel_evidence,
        },
        "channel_selection": channel_selection,
        "channel_selection_reason": channel_reason,
        "ann_residual_candidate": {
            **residual_fit_provenance,
            "charge_mode": selection,
            "modes": [selection, residual_mode],
            "dpt_fit": dpt_fit,
            "dpt_held_out": dpt_held_out,
            "selection": residual_selection,
            "selection_reason": residual_reason,
            "selection_rule": _RESIDUAL_RULE,
            "evidence": residual_evidence,
        },
        "runtime": {
            "golden": {k: golden_time.get(k) for k in
                       ("wall_clock_s", "solver_elapsed_s", "iterations",
                        "trace_points")},
            "tier3": {k: rom_time.get(k) for k in
                      ("wall_clock_s", "solver_elapsed_s", "iterations",
                       "trace_points")},
            "tier3_bench_completed": bench_complete,
            "solver_speedup": (
                golden_time["solver_elapsed_s"] / rom_time["solver_elapsed_s"]
                if bench_complete and rom_time.get("solver_elapsed_s")
                else None),
            "bench_wall_s": {"golden": bench_wall_golden,
                             "tier3": bench_wall_rom},
            "bench": "forced 2ns fidelity control; not the speed claim",
        },
        "speed_adaptive_20": adaptive,
        "build_seconds": {"capture": capture_wall, "fit": fit_wall,
                          "ann_fit": ann_fit_wall,
                          "static_sweep": static_wall},
        "spice_log_tail": candidates[final]["spice_log_tail"],
        "emitted_lib": str(lib.relative_to(ROOT)),
    }
    out = ROOT / "reports" / "comparison" / vendor / \
        "tier3_validation_simplis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if out.exists():
        try:
            prior = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prior = {}
    if isinstance(prior, dict) and "speed_adaptive_20" in prior:
        record["speed_adaptive_20_source"] = prior["speed_adaptive_20"]
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
