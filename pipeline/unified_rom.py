"""T4 unified-ROM pipeline: one emitted subcircuit, one validation record.

This module is the orchestration layer only.  It builds the unified lib from
the landed T0/T1/T2/T3 contracts, runs the static / DPT / speed benches with
byte-identical command decks on the golden and ROM arms (equal sandbox),
evaluates gates G0-G6 from the recorded observations, and writes

    reports/comparison/<vendor>/unified_validation.json  (schema piecespice.unified/1.0)

No bench is run at import time and no number is computed by
:func:`compare_unified`: that entry point reads the recorded JSON only.

Provenance rule (spec section 5): every non-source constant below is declared
in ``PROVENANCE`` with a config key, a spec/doc locator, or an explicit
project decision.  Missing provenance is a gate failure, not a silent default.
"""
from __future__ import annotations

import json
import logging
import math
import re
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from common.types import (
    PWLModelSpec,
    _normalize_unified_metadata,
    trim_trailing_invalid,
)
from config.yaml_io import load_input, load_output
from approximation.smoother import (
    _chain_coefficients,
    eps_from_grid,
    fit_tanh_knee_ls,
    hyp_value,
)
from parser.netlist import NetlistParser, resolve_pin_connections
from pipeline import registry
from pipeline.curves import _cap_ref_vds
from simulation.engine import SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder
from simulation.testbenches.speedup_bench import (
    _elapsed as _solver_elapsed,
    _iterations as _log_iterations,
    _pulse_train,
)
from simulation.testbenches.double_pulse import (
    DoublePulseRun,
    extract_dpt_metrics,
)
from simulation.testbenches.dpt_trajectory import capture_trajectory

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "piecespice.unified/1.0"

DEFAULT_VARIANT = registry.UNIFIED_VARIANT
METHODS = ("trap/default", "gear-both")
EMIT_MODES = ("ann", "tier2")
# Explicit opt-in label for the C1 Tier-2 experiment.  The plain "tier2"
# emit mode always uses the closed C0 emitter (c1_eps=None).
C1_TIER2_VARIANT = "tier2-c1"
TUNING_MODES = ("on", "off")
_LOGGER = logging.getLogger(__name__)

# Equal-sandbox bench decisions (spec sections 1.3, 4.4, 6).
SPEED_N_PULSES = 20
SPEED_REPEATS = 3
SPEED_TIMEOUT_S = 90.0
DPT_TIMEOUT_S = 120.0
DPT_T_LAST_FRACTION = 0.999
DPT_TRACE_POINT_FRACTION = 0.5
STATIC_POINTS = 41
G1_BAND_PP = 0.05                 # proposed project-decision band (spec G1)
ENDPOINT_BAR_PCT = 4.13           # measured worst endpoint (spec 1.4)
CHARGE_BUDGET_PCT = 0.5           # measured charge-fit budget (spec 1.4)
TUNING_MAX_EVALS = 80             # bounded derivative-free optimizer budget
TIMING_BAND_REL = 0.05           # explicit project-decision band (spec G4)
# G4 bar column per channel basis (spec 6 fallback decision): the ANN
# candidate is scored against the recorded golden table; the closed C0
# fallback must return to its own recorded Tier-2 table.  No number is
# re-measured or re-baselined here.
TIMING_BAR_COLUMN = {"ann": "golden_metrics", "tier2": "tier2_metrics"}
ZERO_CURRENT_FRAC = 0.01         # anchor tolerance, not a measured current
TUNING_METRICS = (
    "td_on_ns", "tr_ns", "td_off_ns", "tf_ns",
    "e_on_uj", "e_off_uj", "e_oss_uj", "i_peak_a",
    "dv_dt_max_v_per_ns", "di_dt_max_a_per_ns",
)
TUNING_HELD_OUT = ("tf_ns", "e_off_uj", "i_peak_a")

# Traced sources of every constant above and every comparison bar below.
PROVENANCE: dict[str, dict[str, str]] = {
    "static_bars_pct": {
        "source": "local://unified_rom_spec.md sections 1.4 and 6; README measured table",
        "kind": "measured",
    },
    "static_score_estimator": {
        "source": "docs/fidelity_speed_research.md:89-100; notebooks/05_comprehensive_validation_report.ipynb:2245; reports/comparison/<vendor>/full_sweep_comparison.json (rel_rmse_pct == peak_rmse_pct per row)",
        "kind": "measured",
        "rationale": "G1 gates the arithmetic mean of per-curve peak-normalised rel RMSE; median and span values stay in the record as diagnostics",
    },
    "dpt_bars_pct": {
        "source": (
            "local://unified_rom_spec.md section 1.4; "
            "reports/comparison/<vendor>/tier3_validation_simplis.json"
            "#dpt_run.waveform (vds_rel_rmse_pct, id_rel_rmse_pct) on the "
            "algorithm.channel_candidate_selected=tier2 control row; "
            "pairs keyed by method label (trap/default, gear-both), "
            "EPC 100 ns cap on both arms"
        ),
        "kind": "measured",
        "waveform_columns": "vds_rel_rmse_pct, id_rel_rmse_pct",
        "channel_candidate": "algorithm.channel_candidate_selected=tier2",
        "keyed_by": "method label (trap/default, gear-both)",
    },
    "timing_energy_bars": {
        "source": "reports/comparison/<vendor>/dpt_metrics.json#golden_metrics for emit=ann, #tier2_metrics for emit=tier2 (spec 6 fallback decision); README DPT timing and energy table",
        "kind": "measured",
        "ann_column": "golden_metrics",
        "tier2_column": "tier2_metrics",
    },
    "speed_n_pulses": {
        "source": "local://unified_rom_spec.md section 4.6 (speed_adaptive_20)",
        "kind": "project_decision",
        "rationale": "20 pulses is the pinned adaptive row; a 2-pulse default row must never be compared to it",
    },
    "speed_repeats": {
        "source": "local://unified_rom_spec.md section 1.3 item 7 (at least 3 interleaved repeats)",
        "kind": "project_decision",
    },
    "dpt_t_last_fraction": {
        "source": "local://unified_rom_spec.md section 1.3 item 5 (DPT completion guard)",
        "kind": "project_decision",
    },
    "dpt_trace_point_fraction": {
        "source": "local://unified_rom_spec.md section 1.3 item 5 (at least 50 percent of golden points)",
        "kind": "project_decision",
    },
    "g1_band_pp": {
        "source": "local://unified_rom_spec.md section 6 G1 (proposed band)",
        "kind": "project_decision",
    },
    "endpoint_bar_pct": {
        "source": "local://unified_rom_spec.md section 1.4 (worst endpoint <= 4.13 percent)",
        "kind": "measured",
    },
    "charge_budget_pct": {
        "source": "local://unified_rom_spec.md section 1.4 (charge-fit error <= 0.5 percent Q(Vmax))",
        "kind": "measured",
    },
    "tuning_max_evals": {
        "source": "pipeline/unified_rom.py:TUNING_MAX_EVALS",
        "kind": "project_decision",
        "rationale": "bounded derivative-free search over four named fields; not an emitter default",
    },
    "tuning_bounds": {
        "source": "pipeline/fitting.py:_tuning_bounds(c_rss_lin, Vds_cap_max); config/inputs/<vendor>.yaml#conditions.Vds_cap_max; curve4 c_eq_energy -> metadata c_rss_lin",
        "kind": "source_derived",
        "rationale": "O_GD is capped at one full Q_RSS at the top of the measured curve-4 sweep; Qoss is frozen and never used for this gate-side bound",
    },
    "timing_band": {
        "source": "local://unified_rom_spec.md section 6 G4 (no worse than the recorded table); reports/comparison/<vendor>/dpt_metrics.json#golden_metrics (emit=ann) or #tier2_metrics (emit=tier2, spec 6 fallback decision)",
        "kind": "project_decision",
        "rationale": "5 percent band around each recorded timing/energy bar: a much smaller delay is not an improvement when the device never turns on; missing bars stay not_measured",
    },
    "zero_current_anchor": {
        "source": "local://unified_rom_spec.md section 2.3 (VDS=0 and Vth cutoff guards); models/netlist_builder.py:_channel_expression",
        "kind": "project_decision",
        "rationale": "1 percent of the swept peak current is a structural G2 anchor tolerance, not a measured device current",
    },
    "tuning_golden_emp": {
        "source": "reports/comparison/<vendor>/dpt_metrics.json#golden_metrics; reports/comparison/<vendor>/tier3_validation_simplis.json#runtime.golden,dpt_run",
        "kind": "measured",
        "rationale": "golden metrics are cached once; completion uses recorded iterations, trace points and t_last. Non-completing conditions are never scored",
    },
    "tuning_candidate_libs": {
        "source": "pipeline/tuning.py:make_dpt_simulator; spec section 5 (no permanent candidate artifacts)",
        "kind": "project_decision",
        "rationale": "candidate netlists live in a TemporaryDirectory owned by the tuning run and are removed when tuning returns",
    },
    "adaptive_max_step": {
        "source": "config/inputs/<vendor>.yaml#dpt.max_step; local://unified_rom_spec.md section 3.3 (EPC 100 ns cap on both arms)",
        "kind": "source_derived",
    },
    "temperature": {
        "source": "config/inputs/<vendor>.yaml#conditions.temperature",
        "kind": "source_derived",
    },
    "eps_rule": {
        "source": "approximation/smoother.py:eps_from_grid; local://unified_c1_study.md P7",
        "kind": "source_derived",
        "rationale": "per-table widths: q_oss for the Qoss Vds chain, transfer_vgs for the channel Vgs amp/sat ramps; transfer_vds is the Vds-grid diagnostic (the tanh knee is fitted, not a Vds ramp)",
    },
    "tier2_fallbacks": {
        "source": "models/netlist_builder.py _META_DEFAULTS; AGENTS.md Traps (2026-11-18 gate-cap fix)",
        "kind": "disclosed_fallback",
        "rationale": "inherited from the closed Tier-2 emitter; gate-cap equivalents are evaluated at Cap_ref_Vds so the 4.5 nF / 7.5 pF / 1.5 Ohm fallbacks are not selected",
    },
    "external_sources": {
        "status": "conditional_until_url_and_section_pinned",
        "source": "local://unified_rom_spec.md section 2.3 (Yang/Tung figures CONDITIONAL)",
    },
}

# Section 1.4 / G1 bars.  Never used as a default, only as a comparison bar.
STATIC_BARS_PCT: dict[str, float] = {
    "irfs7730pbf": 7.0683,
    "csd16412q5a": 8.4935,
    "epc2216": 2.0394,
    "cpm2-1200": 3.4287,
}

# Section 1.4 DPT full-cycle Vds/Id bars, keyed by method label.
DPT_BARS_PCT: dict[str, dict[str, tuple[float, float]]] = {
    "trap/default": {
        "irfs7730pbf": (8.529, 0.532),
        "csd16412q5a": (5.583, 0.048),
        "epc2216": (4.894, 0.049),
        "cpm2-1200": (2.675, 0.261),
    },
    "gear-both": {
        "irfs7730pbf": (12.771, 0.5699),
        "csd16412q5a": (5.467, 0.0442),
        "epc2216": (5.075, 0.0461),
        "cpm2-1200": (4.122, 0.2176),
    },
}

# The EPC fully-adaptive golden times out; the recorded 100 ns cap is applied
# to BOTH arms and the row is labelled condition-specific (spec section 3.3).
ADAPTIVE_MAX_STEP: dict[str, float | None] = {"epc2216": 1.0e-7}
_HEADER_RE = re.compile(r"^\s*\.SUBCKT\s+(\S+)", re.IGNORECASE | re.MULTILINE)
_TOL_RE = re.compile(
    r"\b(reltol|abstol|vntol|trtol|gmin)\s*=", re.IGNORECASE)

# X-instance DUT thermal source (emitted only when a role list contains 'tj').
_TJ_SOURCE_RE = re.compile(r"^\s*Vtjsrc_\S*\s+", re.IGNORECASE)
# The emitted unified subckt is always three-terminal (d g s), whatever token
# names the vendor header uses; keep this canonical fallback so vendors with
# numeric header tokens (TI/IRFS) still resolve against the emitted header.
_EMITTER_PIN_TOKENS = {"d": "drain", "g": "gate", "s": "source"}


# ---------------------------------------------------------------------------
# record I/O and small numeric helpers
# ---------------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """Non-finite floats become None (explicit N/A in JSON), never 0."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _record_path(vendor: str) -> Path:
    return (registry.ROOT / "reports" / "comparison" / vendor
            / "unified_validation.json")



def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _median(values: Sequence[float | None]) -> float | None:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.median(finite)) if finite else None


def _subckt_name_text(netlist: str) -> str:
    match = _HEADER_RE.search(netlist)
    if not match:
        raise ValueError("no .SUBCKT header in emitted netlist")
    return match.group(1)


def _subckt_name(lib: Path) -> str:
    return _subckt_name_text(Path(lib).read_text(encoding="ascii"))


def _rom_pin_order(lib: Path, subckt_name: str, semantic: dict) -> list[str]:
    """Resolve the emitted subckt's X-instance roles from its own header.

    Never copy the vendor pin order onto the ROM instance: CPM2's fourth `Tj`
    port would give the three-terminal UROM subckt four nodes and abort
    LTspice.  Vendor semantic tokens are merged with the emitter's canonical
    d/g/s tokens so gate-first and numeric-token headers resolve too.
    """
    subs = NetlistParser.parse_subcircuits(Path(lib))
    info = next((s for s in subs
                 if s.name.lower() == subckt_name.lower()), None)
    if info is None:
        raise ValueError(
            f"Subcircuit {subckt_name!r} not found in {Path(lib).name}")
    token_to_role = {str(token).lower(): role
                     for role, token in (semantic or {}).items()}
    for token, role in _EMITTER_PIN_TOKENS.items():
        token_to_role.setdefault(token, role)
    return [token_to_role.get(port.lower(), port.lower())
            for port in info.nodes]


# ---------------------------------------------------------------------------
# spec producer: the Tier-2 scalar contract, rebuilt without mutating fitting
# ---------------------------------------------------------------------------
def _pwl_spec(vendor: str) -> PWLModelSpec:
    """Rebuild the closed Tier-2 spec for the unified emitter.

    Uses the same curve loaders and reference rule as
    ``pipeline.fitting.fit_pwl_model``; no channel constants are re-tuned.
    T5 may consolidate the two producers into one shared helper.
    """
    from approximation.charge_model import integrate_charge
    from pipeline.fitting import _arr, _curve
    from simulation.testbenches.capacitance import CapacitanceTestbench

    cfg = load_input(vendor)
    cond = cfg["conditions"]

    c1 = _curve(vendor, "curve1_output_characteristics")["extracted"]
    prefix = "curve_id_vgs_"
    vgs_values = sorted(
        float(k[len(prefix):-2])
        for k in c1 if k.startswith(prefix))
    rows: list[list[float]] = []
    ref_len: int | None = None
    for v in vgs_values:
        key = next(k for k in c1 if k.startswith(prefix)
                   and abs(float(k[len(prefix):-2]) - v) < 1e-9)
        y = _arr(c1[key])
        xs = np.linspace(0.0, float(cond["Vds_sweep_max"]), len(y))
        xs, y = trim_trailing_invalid(xs, y)
        if ref_len is None:
            ref_len = len(y)
        grid_len = ref_len if ref_len is not None else len(y)
        rows.append([float(z) for z in np.interp(
            np.linspace(0, float(cond["Vds_sweep_max"]), grid_len),
            xs, y)])
    if ref_len is None:
        raise ValueError("unified ROM spec requires at least one Vgs curve")
    breakpoints_vds = [float(z) for z in np.linspace(
        0.0, float(cond["Vds_sweep_max"]), ref_len)]
    breakpoints_id = [list(map(float, r)) for r in rows]

    rds_on = 0.005
    try:
        c3 = _curve(vendor, "curve3_rds_on")["extracted"]
        rx, ry = _arr(c3["curve_id"]), _arr(c3["curve_rds_mohm"]) * 1e-3
        if len(rx):
            pos = ry[ry > 0]
            if pos.size:
                rds_on = float(np.median(pos))
            else:
                _LOGGER.warning(
                    "%s: curve3 has no positive Rds samples; "
                    "0.005 Ohm fallback kept", vendor)
    except (FileNotFoundError, KeyError):
        _LOGGER.warning("%s: curve3 missing; Rds(on) fallback used", vendor)

    qoss_v: list[float] = []
    qoss_q: list[float] = []
    cq: float | None = None
    ce: float | None = None
    try:
        cap_ref = _cap_ref_vds(cond)
        _, coss, _ = CapacitanceTestbench(
            f_meas=cond.get("F_meas")).run(
            Path(cfg["library"]).resolve(), cfg["subcircuit"],
            vds_min=cond["Vds_cap_min"], vds_max=cond["Vds_cap_max"],
            points=cond["Cap_points"],
            pin_roles=cfg.get("pins", {}).get("pin_order"))
        cm = integrate_charge(_arr(coss.x_data), _arr(coss.y_data),
                              v_ref=cap_ref)
        qoss_v, qoss_q = cm.v.tolist(), cm.q.tolist()
        cq, ce = cm.c_eq_charge, cm.c_eq_energy
    except (FileNotFoundError, KeyError):
        pass

    v_th = 0.0
    try:
        v_th = float(load_output(vendor, "curve7_threshold_voltage")
                     ["extracted"]["vth_v"])
    except (FileNotFoundError, KeyError):
        pass

    b_vsd: list[float] = []
    b_isd: list[float] = []
    try:
        c6 = load_output(vendor, "curve6_body_diode")["extracted"]
        xv, yv = trim_trailing_invalid(_arr(c6["curve_vsd"]),
                                       np.abs(_arr(c6["curve_isd"])))
        b_vsd, b_isd = xv.tolist(), yv.tolist()
    except (FileNotFoundError, KeyError):
        pass

    c_iss = (cq if cq is not None else 1500e-12) * 3.0
    c_rss = (ce if ce is not None else 50e-12) * 0.15
    # The closed Tier-2 emitter still uses linear gate-side scalars, but the
    # tuning layer needs explicit monotone tables.  Both are derived from the
    # same curve-4 gate-cap equivalents on the same voltage grid; Q(0)=0.
    gate_v = sorted({0.0, float(cond["Vds_cap_max"])}
                    | {float(v) for v in qoss_v if v > 0.0})
    gate_iss_q = [float(c_iss * v) for v in gate_v]
    gate_rss_q = [float(c_rss * v) for v in gate_v]
    charge_provenance = dict(PROVENANCE)
    charge_provenance["gate_charge_tables"] = {
        "source": "curve4 c_eq_charge*3 / c_eq_energy*0.15 on the curve-4 Vds grid",
        "kind": "source_derived",
        "rationale": "closed Tier-2 linear gate-side scalar, tabulated so apply_scalars can add O_GD*V without a hidden fallback",
    }
    return PWLModelSpec(
        device_name=cfg["device"],
        tier=2,
        breakpoints_vds=breakpoints_vds,
        breakpoints_id=breakpoints_id,
        vgs_values=[float(v) for v in vgs_values],
        q_oss_vs_v=qoss_v,
        q_oss_charge_c=qoss_q,
        q_iss_vs_v=gate_v,
        q_iss_charge_c=gate_iss_q,
        q_rss_vs_v=gate_v,
        q_rss_charge_c=gate_rss_q,
        rds_on=rds_on,
        v_th=v_th,
        body_diode_vsd=b_vsd,
        body_diode_isd=b_isd,
        metadata={
            "r_gate": 1.5,
            "r_source": 1e-4,
            "r_drain": 1e-4,
            "Vds_cap_max": float(cond["Vds_cap_max"]),
            "c_iss_lin": c_iss,
            "c_oss_lin": cq if cq is not None else 500e-12,
            "c_rss_lin": c_rss,
            "urom_provenance": charge_provenance,
        },
    )


def _eps_by_table(spec: PWLModelSpec) -> dict[str, float]:
    """Data-derived C1 widths from the emitted grids (spec P7)."""
    eps: dict[str, float] = {}
    if spec.q_oss_vs_v:
        eps["q_oss"] = float(eps_from_grid(spec.q_oss_vs_v))
    if spec.breakpoints_vds:
        eps["transfer_vds"] = float(eps_from_grid(spec.breakpoints_vds))
    if spec.vgs_values:
        eps["transfer_vgs"] = float(eps_from_grid(spec.vgs_values))
    return eps


def _emitted_eps_by_table(spec: PWLModelSpec) -> dict[str, float]:
    """Widths the emitter actually used; empty means the closed C0 emitter."""
    widths = spec.metadata.get("urom_emitted_eps_by_table")
    if not isinstance(widths, dict):
        return {}
    return {k: float(v) for k, v in widths.items()
            if _num(v) is not None and float(v) > 0.0}


def _merge_provenance(producer: dict | None, *,
                      base: dict | None = None) -> dict:
    """Merge producer provenance into the pinned base; producer keys win.

    ``_pwl_spec`` attaches curve/table provenance under ``urom_provenance``;
    replacing it with the global ``PROVENANCE`` at emission loses those source
    entries (spec section 5: no hidden defaults).
    """
    merged = dict(base if base is not None else PROVENANCE)
    if isinstance(producer, dict):
        merged.update(producer)
    return merged


def _c0_bit_identity(spec: PWLModelSpec, emitted: str, name: str) -> bool:
    """A/B the emitted lib against canonical C0 at the same subckt name."""
    from models.netlist_builder import build_tier2_subcircuit

    return bool(emitted == build_tier2_subcircuit(
        spec, c1_eps=None, name=name))

def _recorded_golden_emp(vendor: str) -> tuple[dict, bool, str]:
    """Cached golden DPT metrics plus their recorded completion evidence.

    No golden waveform is simulated here: the metrics come from the recorded
    vendor run and the completion guard re-checks its recorded iterations,
    point count and final time.  A non-completing golden is returned with
    ``completed=False`` and is never used as an objective.
    """
    metrics_path = (registry.ROOT / "reports" / "comparison" / vendor
                    / "dpt_metrics.json")
    source = str(metrics_path.relative_to(ROOT)).replace("\\", "/")
    if not metrics_path.exists():
        return {}, False, source
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    raw = data.get("golden_metrics") or {}
    emp = {k: v for k in TUNING_METRICS
           if (v := _num(raw.get(k))) is not None}
    if not emp or data.get("golden_failure"):
        return {}, False, source

    val_path = (registry.ROOT / "reports" / "comparison" / vendor
                / "tier3_validation_simplis.json")
    if not val_path.exists():
        return emp, False, source
    val = json.loads(val_path.read_text(encoding="utf-8"))
    golden = (val.get("runtime") or {}).get("golden") or {}
    dpt_run = val.get("dpt_run") or {}
    iterations = _num(golden.get("iterations"))
    points = _num(golden.get("trace_points"))
    t_last = _num(dpt_run.get("t_last_s"))
    dpt = load_input(vendor)["dpt"]
    step = _num(dpt.get("max_step"))
    expected = (math.ceil(float(dpt["t_end"]) / step) + 1
                if step and step > 0.0 else None)
    completed = bool(
        iterations is not None and iterations > 0
        and points is not None and points > 0
        and (expected is None
             or points >= DPT_TRACE_POINT_FRACTION * expected)
        and t_last is not None
        and t_last >= DPT_T_LAST_FRACTION * float(dpt["t_end"]))
    return emp, completed, source


def _tuning_condition_descriptors(vendor: str, plan: dict) -> list[dict]:
    """Two honest metric-split arms of one cached golden DPT run.

    The held-out arm is a disjoint set of recorded golden metrics, not a
    made-up condition: only one golden operating point is on record per
    vendor.  ``emp`` and ``completed`` are taken from that record; the real
    ``simulate`` callable is attached in ``_build_emitted`` once the emitted
    spec exists.
    """
    emp, completed, source = _recorded_golden_emp(vendor)
    fit = {k: v for k, v in emp.items() if k not in TUNING_HELD_OUT}
    held = {k: v for k, v in emp.items() if k in TUNING_HELD_OUT}
    if not fit or not held:
        return []
    condition = {
        "vbus": plan["vbus"], "vgs_on": plan["vgs_on"],
        "i_load_target": plan["i_load"], "l_load": plan["l_load"],
        "temperature": plan["temperature"],
    }
    return [
        {"name": "recorded_golden_fit", "emp": fit, "held_out": False,
         "completed": completed, "condition": dict(condition),
         "emp_source": source,
         "split": "fit metrics of the recorded golden DPT"},
        {"name": "recorded_golden_heldout", "emp": held, "held_out": True,
         "completed": completed, "condition": dict(condition),
         "emp_source": source,
         "split": "held-out metrics of the recorded golden DPT"},
    ]


def _serializable_conditions(conditions: list[dict]) -> list[dict]:
    """Drop the live simulator callable before writing the record."""
    return [{k: v for k, v in cond.items() if k != "simulate"}
            for cond in conditions]


# ---------------------------------------------------------------------------
# build stage (T1 / T2 / T3 imports are lazy: they are separate landing tasks)
# ---------------------------------------------------------------------------
def _theta_bounds(spec: PWLModelSpec,
                  vendor: str | None = None) -> dict[str, tuple[float, float]]:
    """Search envelopes for the four named scalars, sourced from curve 4.

    ``O_GD`` is an additive charge on Q_RSS(V), so its bracket is one full
    ``Q_RSS(Vds_cap_max) = c_rss_lin * Vds_cap_max`` (the same rule as
    ``pipeline.fitting._tuning_bounds``).  Qoss is frozen by spec section 2.2
    and must never size a gate-drain offset.
    """
    from pipeline.fitting import _tuning_bounds

    v_cap = _num(spec.metadata.get("Vds_cap_max"))
    if v_cap is None:
        name = vendor or spec.device_name
        if name not in registry.VENDORS:
            raise ValueError(f"{name}: no sourced Vds_cap_max for theta bounds")
        v_cap = float(load_input(name)["conditions"]["Vds_cap_max"])
    c_rss = _num(spec.metadata.get("c_rss_lin"))
    if c_rss is None or c_rss <= 0.0:
        raise ValueError("c_rss_lin must be a positive source-derived scalar")
    return _tuning_bounds(v_cap, c_rss)


def _emit_tier2_c0(spec: PWLModelSpec, *, name: str) -> str:
    """Emit the closed Tier-2 fallback: canonical C0, ``c1_eps=None``.

    The only difference from ``fit_pwl_model`` is the subcircuit name, which
    the compare bench requires to match the UROM library stem.
    """
    from models.netlist_builder import build_tier2_subcircuit

    return build_tier2_subcircuit(spec, c1_eps=None, name=name)


def _emit_tier2_per_table(spec: PWLModelSpec, eps_by_table: dict,
                          *, name: str) -> str:
    """Opt-in C1 Tier-2 variant, labelled ``tier2-c1`` in the record.

    Never used by the plain ``tier2`` fallback, whose C0 emission is
    bit-identical to canonical ``fit_pwl_model`` (modulo the subckt name).

    The shared public builder takes one width for both ramp sites, so the
    channel (Vgs amp/sat) and Qoss (Vds charge) lines are taken from two
    builder passes at their own derived widths; every other line is identical
    and comes from the Qoss pass. A builder refactor that changes the line
    shape fails loudly instead of silently emitting a mixed-width netlist.
    """
    from models.netlist_builder import build_tier2_subcircuit

    eps_qoss = float(eps_by_table.get("q_oss", 0.0))
    eps_transfer = float(eps_by_table.get("transfer_vgs", eps_qoss))
    charge = build_tier2_subcircuit(spec, c1_eps=eps_qoss, name=name)
    if eps_transfer == eps_qoss:
        return charge
    channel = build_tier2_subcircuit(spec, c1_eps=eps_transfer, name=name)
    charge_lines = charge.splitlines()
    channel_lines = channel.splitlines()
    ch_target = [i for i, line in enumerate(charge_lines)
                 if line.startswith("G_CH ")]
    ch_source = [i for i, line in enumerate(channel_lines)
                 if line.startswith("G_CH ")]
    if len(ch_target) != 1 or len(ch_source) != 1:
        raise ValueError(
            f"{spec.device_name}: ambiguous G_CH line in Tier-2 emission")
    charge_lines[ch_target[0]] = channel_lines[ch_source[0]]
    return "\n".join(charge_lines)

def _build_emitted(vendor: str, *, emit: str, tuning: str,
                   plan: dict, c1_variant: str | None = None
                   ) -> tuple[Path, PWLModelSpec, dict, dict]:
    """Build, optionally tune, and emit one unified lib.  No bench is run.

    ``tier2`` is the closed C0 fallback (``c1_eps=None``); C1 transfer ramps
    require the explicit ``tier2-c1`` variant label.  The ANN charge ramps
    keep their data-derived C1 width.
    """
    from pipeline.tuning import apply_scalars, make_dpt_simulator, tune_scalars

    if c1_variant is not None and c1_variant != C1_TIER2_VARIANT:
        raise ValueError(f"unknown C1 variant {c1_variant!r}; expected "
                         f"{C1_TIER2_VARIANT!r} or None")

    lib = registry.lib_path(vendor, 3, registry.UNIFIED_VARIANT)
    stem = lib.stem
    spec = _pwl_spec(vendor)
    eps_by_table = _eps_by_table(spec)
    eps_qoss = float(eps_by_table.get("q_oss", 0.0))
    # Derived widths are provenance only: the plain tier2 fallback emits C0,
    # so neither the channel nor Qoss may use them without an explicit label.
    emitted_eps = (eps_by_table if emit == "ann"
                   or c1_variant == C1_TIER2_VARIANT else {})
    spec.metadata["urom_c1_eps"] = float(emitted_eps.get("q_oss", 0.0))
    spec.metadata["urom_c1_eps_by_table"] = dict(eps_by_table)
    spec.metadata["urom_emitted_eps_by_table"] = dict(emitted_eps)

    ann_record: dict = {}
    if emit == "ann":
        from models.ann_netlist_builder import build_ann_subcircuit
        from pipeline.ann_fit import ann_fit_for_vendor

        ann_record = ann_fit_for_vendor(vendor) or {}
        channel_fit = ann_record.get("channel_fit")
        charge_fit = ann_record.get("charge_fit")
        if channel_fit is None or charge_fit is None:
            raise RuntimeError(
                f"{vendor}: ann_fit_for_vendor returned no channel/charge fit")
        # The ANN channel has no Vgs ramp chain; eps applies to its charge
        # ramps only, so the Qoss grid width is the applicable one here.
        emitter = lambda s, widths: build_ann_subcircuit(
            s, channel_fit, charge_fit,
            c1_eps=widths.get("q_oss", widths.get("transfer_vgs", 0.0)))
    elif c1_variant == C1_TIER2_VARIANT:
        # Explicitly labelled C1 experiment, never the fallback default.
        emitter = lambda s, widths: _emit_tier2_per_table(
            s, widths, name=stem)
    else:
        # The compare bench instantiates the library stem, so the fallback
        # header must use the same UROM name instead of the closed T2 name.
        # The channel and Qoss ramps are the canonical C0 ramps.
        emitter = lambda s, _widths: _emit_tier2_c0(s, name=stem)

    descriptors = (plan.get("tuning_conditions")
                   or _tuning_condition_descriptors(vendor, plan))
    conditions = _serializable_conditions(descriptors)
    bounds = _theta_bounds(spec, vendor)
    tuning_record: dict = {
        "mode": tuning,
        "theta": {"A_GD": None, "O_GD": None, "A_GS": None, "A_RGI": None},
        "bounds": {k: list(v) for k, v in bounds.items()},
        "weights": {},
        "conditions": conditions,
        "golden_completion_flags": [False] * len(conditions),
        "held_out_scores": {},
        "verdict": "infeasible" if tuning == "off" else "reject",
        "max_evals": TUNING_MAX_EVALS,
        "emp_split": "disjoint recorded golden-metric arms; no proxy condition",
    }
    if tuning == "on" and any(c.get("completed") for c in conditions):
        # Every simulate(theta) writes a candidate lib; keep those inside a
        # run-owned temporary directory so no stale candidate survives.
        with tempfile.TemporaryDirectory(
                prefix=f"urom_tune_{vendor}_") as tmp:
            live: list[dict] = []
            for desc in descriptors:
                cond = dict(desc)
                cond["simulate"] = make_dpt_simulator(
                    vendor, spec, method=plan["method_label"],
                    condition=desc["condition"], out_dir=tmp,
                    timeout_s=DPT_TIMEOUT_S)
                live.append(cond)
            result = tune_scalars(
                vendor, method=plan["method_label"], conditions=live,
                bounds=bounds, weights=None, max_evals=TUNING_MAX_EVALS)
        theta = dict(result.theta)
        completion = dict(result.golden_completion)
        verdict = str(result.verdict)
        tuning_record.update({
            "theta": {k: theta.get(k) for k in
                      ("A_GD", "O_GD", "A_GS", "A_RGI")},
            "golden_completion_flags": [
                name for name, ok in completion.items() if ok],
            "condition_results": dict(result.condition_results),
            "weights": dict(result.weights),
            "held_out_scores": dict(result.held_out_scores),
            "objective_fit": _json_safe(result.objective_fit),
            "objective_held_out": _json_safe(result.objective_held_out),
            "evaluations": int(result.evaluations),
            "optimizer": result.optimizer,
            "verdict": verdict,
        })
        if not any(completion.values()):
            tuning_record["verdict"] = "infeasible"
        if tuning_record["verdict"] == "accept":
            spec = apply_scalars(spec, theta)

    spec.metadata["urom_channel_basis"] = emit
    spec.metadata["urom_c1_variant"] = c1_variant
    spec.metadata["urom_tuning_theta"] = tuning_record["theta"]
    spec.metadata["urom_method_label"] = plan["method_label"]
    spec.metadata["urom_provenance"] = _merge_provenance(
        spec.metadata.get("urom_provenance"))
    # Additive metadata validation from T0; a bad speed row or NaN fails here.
    _normalize_unified_metadata(spec.metadata, vendor)

    netlist = emitter(spec, eps_by_table)
    if not netlist.isascii():
        raise ValueError(f"{vendor}: unified emitter produced non-ASCII text")
    header = _subckt_name_text(netlist)
    if header != stem:
        raise ValueError(f"{vendor}: emitted .SUBCKT {header!r} does not "
                         f"match the UROM library stem {stem!r}")
    lib.parent.mkdir(parents=True, exist_ok=True)
    lib.write_text(netlist, encoding="ascii")
    return lib, spec, ann_record, tuning_record


# ---------------------------------------------------------------------------
# equal-sandbox bench construction
# ---------------------------------------------------------------------------
def _bench_plan(vendor: str, method: str, repeats: int) -> dict:
    cfg = load_input(vendor)
    dpt = {k: float(v) for k, v in cfg["dpt"].items()}
    n_pulses = SPEED_N_PULSES
    period = dpt["t2_on"] - dpt["t1_on"]
    last_width = dpt["t1_width"] if n_pulses == 1 else dpt["t2_width"]
    speed_end = 1.0e-6 + (n_pulses - 1) * period + last_width + 2.0e-6
    max_step = ADAPTIVE_MAX_STEP.get(vendor)
    plan = {
        "method_label": method,
        "options_lines": ([".options method=gear"] if method == "gear-both"
                          else []),
        "temperature": float(cfg["conditions"]["temperature"]),
        "vbus": dpt["vbus"], "vgs_on": dpt["vgs_on"],
        "i_load": dpt["i_load_target"], "l_load": dpt["l_load"],
        "t1_on": dpt["t1_on"], "t1_width": dpt["t1_width"],
        "t2_on": dpt["t2_on"], "t2_width": dpt["t2_width"],
        "edge_time": dpt["edge_time"],
        "dpt_t_end": dpt["t_end"], "dpt_max_step": dpt["max_step"],
        "repeats": int(repeats),
        "speed": {
            "n_pulses": n_pulses, "max_step": max_step,
            "end_time_s": speed_end, "repeats": int(repeats),
            "timeout_s": SPEED_TIMEOUT_S,
        },
    }
    plan["tuning_conditions"] = _tuning_condition_descriptors(vendor, plan)
    return plan


def _arm_specs(vendor: str, rom_lib: Path) -> tuple[dict, dict]:
    cfg = load_input(vendor)
    semantic = cfg["pins"].get("pin_order_semantic") or {}
    golden_lib = Path(cfg["library"]).resolve()
    golden_name = cfg["subcircuit"]
    rom_name = _subckt_name(rom_lib)
    return (
        {
            "library_path": str(golden_lib),
            "subcircuit_name": golden_name,
            # Header-resolved: CPM2 keeps its fourth `tj` role; other vendors
            # resolve to exactly their cfg pin_order.
            "pin_order": resolve_pin_connections(
                golden_lib, golden_name, semantic or None),
        },
        {
            "library_path": str(Path(rom_lib).resolve()),
            "subcircuit_name": rom_name,
            "pin_order": _rom_pin_order(rom_lib, rom_name, semantic),
        },
    )


def _dpt_deck(plan: dict, arm: dict) -> str:
    nb = NetlistBuilder("UROM_DPT")
    nb.include(arm["library_path"])
    nb.v_dc("bus", "vbus", "0", plan["vbus"])
    nb.l("load", "vbus", "drain", plan["l_load"])
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
    nb.raw_line("Vgs gate 0 PWL(" + _pulse_train(
        plan["t1_on"], plan["t1_width"], plan["t2_width"],
        plan["t2_on"] - plan["t1_on"], 2, plan["edge_time"],
        plan["vgs_on"]) + ")")
    nb.x_instance("dut", arm["subcircuit_name"], arm["pin_order"],
                  temperature=plan["temperature"])
    nb.raw_line(f".temp {plan['temperature']:g}")
    for line in plan["options_lines"]:
        nb.raw_line(line)
    return nb.render()


def _speed_deck(plan: dict, arm: dict) -> str:
    sp = plan["speed"]
    nb = NetlistBuilder("UROM_SPEED")
    nb.include(arm["library_path"])
    nb.v_dc("bus", "vbus", "0", plan["vbus"])
    nb.l("load", "vbus", "drain", plan["l_load"])
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)")
    nb.raw_line("Vgs gate 0 PWL(" + _pulse_train(
        1.0e-6, plan["t1_width"], plan["t2_width"],
        plan["t2_on"] - plan["t1_on"], sp["n_pulses"],
        plan["edge_time"], plan["vgs_on"]) + ")")
    nb.x_instance("dut", arm["subcircuit_name"], arm["pin_order"],
                  temperature=plan["temperature"])
    nb.raw_line(f".temp {plan['temperature']:g}")
    for line in plan["options_lines"]:
        nb.raw_line(line)
    return nb.render()


def _deck_identity(golden_deck: str, rom_deck: str,
                   options_lines: list[str], *,
                   expected_temperature: float | None = None) -> dict:
    """Prove from deck text that the DUT is the only changed region.

    Includes, the single X-instance that names the DUT, and the DUT-only
    thermal (Tj) source are the only lines removed.  Options, tolerance cards,
    temperature cards, stimulus and measure lines all remain in the comparison,
    so any change there fails the equality flag instead of being assumed away.
    """
    def strip_dut(text: str) -> str:
        lines = text.splitlines()
        x_lines = [i for i, line in enumerate(lines)
                   if re.match(r"^\s*X\S+\s+", line, re.IGNORECASE)]
        keep = []
        for i, line in enumerate(lines):
            s = line.strip().lower()
            if s.startswith((".include ", ".lib ")):
                continue
            if _TJ_SOURCE_RE.match(line):
                continue
            if x_lines and i == x_lines[-1]:
                continue
            keep.append(line)
        return "\n".join(keep)

    def actual_options(deck: str) -> list[str]:
        return [s.strip() for s in deck.splitlines()
                if s.strip().lower().startswith(".options")]

    def library_evidence(deck: str) -> bool:
        cards = [s.strip().lower() for s in deck.splitlines()]
        return (any(s.startswith((".include ", ".lib ")) for s in cards)
                or any(s.startswith(".subckt ") for s in cards))

    def deck_temperature(deck: str) -> float | None:
        # LTspice takes the last .temp card as effective.
        cards = re.findall(
            r"^\s*\.temp\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
            deck, re.IGNORECASE | re.MULTILINE)
        return _num(cards[-1]) if cards else None

    gold_options = actual_options(golden_deck)
    rom_options = actual_options(rom_deck)
    overrides = sorted({m.group(0).lower()
                        for deck in (golden_deck, rom_deck)
                        for line in deck.splitlines()
                        for m in [_TOL_RE.search(line)] if m})
    gold_temp = deck_temperature(golden_deck)
    rom_temp = deck_temperature(rom_deck)
    expected = _num(expected_temperature)
    if expected is None:
        temp_ok = (gold_temp is not None and rom_temp is not None
                   and abs(gold_temp - rom_temp) <= 1e-9)
    else:
        temp_ok = (gold_temp is not None and rom_temp is not None
                   and abs(gold_temp - expected) <= 1e-9
                   and abs(rom_temp - expected) <= 1e-9)
    options_dump = {
        "golden": gold_options,
        "rom": rom_options,
        "declared": list(options_lines),
    }
    return {
        "deck_diff_only_dut": strip_dut(golden_deck) == strip_dut(rom_deck),
        "effective_options_dumped": gold_options == rom_options,
        "effective_options": options_dump,
        "tolerance_overrides": overrides,
        "libraries_inlined_both_arms": (
            library_evidence(golden_deck) and library_evidence(rom_deck)),
        "temperature_both_arms": gold_temp if temp_ok else None,
        "expected_temperature": expected,
    }


# ---------------------------------------------------------------------------
# live bench runners (entered only when run_unified is called)
# ---------------------------------------------------------------------------
def _one_transient(deck: str, directive: TranDirective,
                   engine: SpiceEngine, *, timeout_s: float) -> dict:
    t0 = time.perf_counter()
    result = engine.run_transient(deck, directive)
    wall = time.perf_counter() - t0
    trace = np.asarray(result.get("time"), dtype=float)
    iterations = _log_iterations(result.log_text)
    return {
        "iterations": iterations,
        "points": int(trace.size),
        "t_last": float(trace[-1]) if trace.size else None,
        "solver_time_s": _solver_elapsed(result.log_text),
        "wall_time_s": float(wall),
        "log_tail": result.log_text[-4000:],
        "result": result,
    }


def _run_interleaved_speed(plan: dict, golden: dict, rom: dict,
                           decks: dict) -> dict:
    """Three interleaved repeats per arm; no aborted row is ever 'fast'."""
    sp = plan["speed"]
    directive = TranDirective(step_time=sp["max_step"] or plan["edge_time"],
                              end_time=sp["end_time_s"],
                              max_step=sp["max_step"])
    engine = SpiceEngine(temperature=plan["temperature"],
                         timeout_s=sp["timeout_s"])
    runs: dict[str, list[dict]] = {"golden": [], "urom": []}
    order = (("golden", "urom"), ("urom", "golden"))
    for i in range(sp["repeats"]):
        for label in order[i % len(order)]:
            runs[label].append(
                _one_transient(decks[label], directive, engine,
                               timeout_s=sp["timeout_s"]))
    out: dict[str, Any] = {"repeats": sp["repeats"],
                           "n_pulses": sp["n_pulses"],
                           "max_step": sp["max_step"],
                           "scheduled_end_s": sp["end_time_s"]}
    for label, arm_runs in runs.items():
        pts = [r["points"] for r in arm_runs]
        med_points = _median([float(p) for p in pts]) or 0.0
        complete = []
        for r in arm_runs:
            complete.append(bool(
                r["iterations"] and r["iterations"] > 0
                and r["points"] > 0
                and r["t_last"] is not None
                and r["t_last"] >= DPT_T_LAST_FRACTION * sp["end_time_s"]
                and r["points"] >= DPT_TRACE_POINT_FRACTION * med_points))
        solver = _median([r["solver_time_s"] for r in arm_runs])
        wall = _median([r["wall_time_s"] for r in arm_runs])
        out[label] = {
            "iterations": _median([float(r["iterations"])
                                   if r["iterations"] is not None else None
                                   for r in arm_runs]),
            "points": med_points,
            "t_last": _median([r["t_last"] for r in arm_runs]),
            "solver_time_s": solver,
            "wall_time_s": wall,
            "solver_min_s": min((r["solver_time_s"] for r in arm_runs
                                 if r["solver_time_s"] is not None),
                                default=None),
            "solver_max_s": max((r["solver_time_s"] for r in arm_runs
                                 if r["solver_time_s"] is not None),
                                default=None),
            "wall_min_s": min((r["wall_time_s"] for r in arm_runs
                               if r["wall_time_s"] is not None),
                              default=None),
            "wall_max_s": max((r["wall_time_s"] for r in arm_runs
                               if r["wall_time_s"] is not None),
                              default=None),
            "completion": all(complete),
            "per_repeat": [{k: v for k, v in r.items()
                            if k not in ("result", "log_tail")}
                           for r in arm_runs],
            "log_tail": arm_runs[-1]["log_tail"],
        }
    return out


def _run_dpt_arm(plan: dict, label: str, deck: str, traj,
                 dpt_cfg: dict) -> dict:
    directive = TranDirective(step_time=plan["dpt_max_step"],
                              end_time=plan["dpt_t_end"],
                              max_step=plan["dpt_max_step"])
    engine = SpiceEngine(temperature=plan["temperature"],
                         timeout_s=DPT_TIMEOUT_S)
    run = _one_transient(deck, directive, engine, timeout_s=DPT_TIMEOUT_S)
    result = run["result"]
    dpt_run = DoublePulseRun(
        time=np.asarray(result.get("time")),
        v_ds=np.asarray(result.get("V(drain)")),
        i_d=np.abs(np.asarray(result.get("I(Lload)"), dtype=float)),
        v_gs=np.asarray(result.get("V(gate)")),
        v_bus=plan["vbus"], i_load=plan["i_load"],
        l_load=plan["l_load"], vgs_on=plan["vgs_on"],
        wall_clock_s=float(run["wall_time_s"]),
        t2_on=dpt_cfg["t2_on"])
    metrics: dict = {}
    metric_error: str | None = None
    try:
        metrics = dict(extract_dpt_metrics(dpt_run))
    except (ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
        metric_error = str(exc)
    mask = dpt_run.time <= traj.time[-1]
    waveform = {"vds_rel_rmse_pct": None, "id_rel_rmse_pct": None,
                "compare_points": 0}
    if np.any(mask):
        t = dpt_run.time[mask]
        vds = np.interp(t, traj.time, traj.input[1])
        ids = np.interp(t, traj.time, np.abs(traj.result.get("I(Lload)")))
        waveform = {
            "vds_rel_rmse_pct": float(100 * np.sqrt(np.mean(
                (dpt_run.v_ds[mask] - vds) ** 2)) /
                (np.ptp(traj.input[1]) or 1)),
            "id_rel_rmse_pct": float(100 * np.sqrt(np.mean(
                (dpt_run.i_d[mask] - ids) ** 2)) /
                (np.ptp(np.abs(traj.result.get("I(Lload)"))) or 1)),
            "compare_points": int(mask.sum()),
        }
    completed = bool(
        run["iterations"] and run["iterations"] > 0
        and run["points"] >= DPT_TRACE_POINT_FRACTION * len(traj.time)
        and run["t_last"] is not None
        and run["t_last"] >= DPT_T_LAST_FRACTION * plan["dpt_t_end"])
    return {
        "completed": completed, "iterations": run["iterations"],
        "trace_points": run["points"],
        "golden_trace_points": len(traj.time),
        "t_last": run["t_last"], "waveform": waveform,
        "metrics": metrics, "log_tail": run["log_tail"],
        "error": metric_error,
    }


def _static_bench(vendor: str, lib: Path) -> tuple[dict, list]:
    from metrics.compare import recorded_static_summary
    from pipeline.fitting import compare_tier2_vs_golden_full

    rows = compare_tier2_vs_golden_full(
        vendor, lib_path=lib, n_points=STATIC_POINTS,
        subckt_name=lib.stem)
    summary = recorded_static_summary(
        [(golden, sim) for _, _, golden, sim in rows])
    return summary, rows


def _channel_transfer_ramp(
        spec: PWLModelSpec, *, eps: float = 0.0
        ) -> tuple[list[float], list[float], list[float], float] | None:
    """Numeric mirror of the emitted channel transfer ramps (C0 and C1).

    Returns ``(xs, iv, sv, cutoff_vgs)`` matching
    ``models.netlist_builder._channel_expression`` for the same spec: the
    Vgs grid ``xs`` (Vgs=0 anchor plus the optional Vth anchor), the
    transfer-amplitude ramp ``iv``, the saturation-width ramp ``sv`` and the
    gate bias where the builder emits the zero-current cutoff anchor (Vth
    only when 0<Vth<min(drives), otherwise Vgs=0).  None when there is no
    conducting family to probe.  No device number is invented here: every
    value comes from the spec or the builder's own fixed fitting rule.
    """
    from approximation.smoother import fit_tanh_plateau
    from models.netlist_builder import _PLATEAU_BLEND_W, _PLATEAU_GUARD

    if (len(spec.breakpoints_vds) < 2 or not spec.vgs_values
            or not spec.breakpoints_id
            or len(spec.vgs_values) != len(spec.breakpoints_id)
            or any(len(row) != len(spec.breakpoints_vds)
                   for row in spec.breakpoints_id)):
        return None
    vd = np.asarray(spec.breakpoints_vds, dtype=float)
    vmax = float(vd[-1])
    span = vmax - float(vd[0])
    drives = [float(v) for v in spec.vgs_values]
    if max(float(row[-1]) for row in spec.breakpoints_id) <= 0.0:
        return None
    sat_vals: list[float] = []
    amplitudes: list[float] = []
    for row in spec.breakpoints_id:
        y = np.asarray(row, dtype=float)
        end = float(y[-1])
        plateau = max(end, 1e-9)
        s = min(max(float(fit_tanh_knee_ls(vd, y, plateau)), 0.05 * span),
                vmax)
        sat_vals.append(s)
        amp = end / np.tanh(vmax / s)
        i_fit, _ = fit_tanh_plateau(vd, y, plateau)
        ratio = i_fit / max(end, 1e-12)
        if _PLATEAU_GUARD[0] <= ratio <= _PLATEAU_GUARD[1]:
            amp = (1.0 - _PLATEAU_BLEND_W) * amp + _PLATEAU_BLEND_W * i_fit
        amplitudes.append(float(amp))
    xs = [0.0] + drives
    iv = [0.0] + amplitudes
    sv = [sat_vals[0]] + sat_vals
    vth = _num(spec.v_th)
    cutoff_vgs = 0.0
    if vth is not None and 0.0 < vth < min(drives):
        xs = [0.0, vth] + drives
        iv = [0.0, 0.0] + amplitudes
        sv = [sat_vals[0], sat_vals[0]] + sat_vals
        cutoff_vgs = vth
    return xs, iv, sv, cutoff_vgs


def _emitted_cutoff_anchor(spec: PWLModelSpec) -> tuple[float, float] | None:
    """Probe the emitted channel current at its gate cutoff anchor.

    The closed builder puts a zero-current anchor at Vth when 0<Vth<min
    (drives), otherwise at Vgs=0.  Return ``(cutoff_vgs, |I_cutoff|)`` from
    the emitted transfer ramp, or None when no conducting family exists;
    None is never treated as a measured pass.
    """
    ramp = _channel_transfer_ramp(spec)
    if ramp is None:
        return None
    xs, iv, _sv, cutoff_vgs = ramp
    # Probe at the width the emitter actually used, not the data-derived C1
    # width: the plain tier2 fallback emits closed C0 (eps=0) even though
    # urom_c1_eps_by_table carries derived widths as provenance only.
    eps = _num(_emitted_eps_by_table(spec).get("transfer_vgs")) or 0.0
    current = _ramp_numeric(xs, iv, eps, np.asarray([cutoff_vgs],
                                                    dtype=float))
    return cutoff_vgs, abs(float(np.max(np.abs(current))))


def _derivative_checks(rows: list, spec: PWLModelSpec, ann_record: dict,
                       emit: str) -> dict:
    """Score the emitted model's own sweep: gm/gds, Q monotonicity, anchors.

    Both channel bases are checked against the emitted DC sweep: finite-
    difference gm/gds arrays, a non-negative Qoss derivative and the Vds=0
    plus gate-cutoff zero-current anchors.  For ``emit="tier2"`` the cutoff
    is probed directly at the anchor the closed builder emits and the
    ANN-only out-of-range probe is ``not_applicable``.  For ``emit="ann"``
    the ANN fit record, its out-of-range probe and the explicit sub-Vth swept
    guard stay mandatory.  Every predicate must be measured and hold: no
    emit mode gets an unconditional pass or an unconditional fail.
    """
    result: dict[str, Any] = {
        "pass": False, "status": "not_measured",
        "reason": "no static sweep rows", "emit": emit,
    }
    if not rows:
        return result
    vgs = np.asarray([r[0] for r in rows], dtype=float)
    sim = np.asarray([r[3] for r in rows], dtype=float)
    vds = np.asarray(rows[0][1], dtype=float)
    gm = np.gradient(sim, vgs, axis=0) if sim.shape[0] > 1 else sim
    gds = np.gradient(sim, vds, axis=1) if sim.shape[1] > 1 else sim
    q = np.asarray(spec.q_oss_charge_c, dtype=float)
    qmono = bool(q.size > 1 and np.all(np.diff(q) >= 0))
    finite = bool(np.all(np.isfinite(gm)) and np.all(np.isfinite(gds)))
    peak = float(np.max(np.abs(sim))) if sim.size else 0.0
    vds_zero = bool(vds.size and abs(float(vds[0])) <= 1e-15
                    and float(np.max(np.abs(sim[:, 0]))) <= ZERO_CURRENT_FRAC
                    * (peak or 1.0))

    record = (ann_record.get("record") or {}) if emit == "ann" else {}
    if emit == "ann":
        out_of_range = record.get("out_of_range_check")
        ann_ok = bool(record) and out_of_range == "pass"
        vth = _num(spec.v_th)
        vth_anchor: bool | None = None
        cutoff_vgs = vth
        if vth is not None and vgs.size:
            below = vgs < vth
            if np.any(below):
                idx = int(np.argmin(np.abs(vgs[below] - vth)))
                vth_anchor = bool(
                    float(np.max(np.abs(sim[below][idx])))
                    <= ZERO_CURRENT_FRAC * (peak or 1.0))
        cutoff_anchor = vth_anchor
    else:
        # The closed C0 fallback has no ANN fit, so its out-of-range probe
        # is not applicable rather than a mandatory fail.  Probe the emitted
        # cutoff directly: a swept sub-Vth row is not required (most curve-1
        # families start above Vth).
        out_of_range = "not_applicable"
        ann_ok = True
        probe = _emitted_cutoff_anchor(spec)
        if probe is None:
            cutoff_vgs = None
            vth_anchor = None
            cutoff_anchor = False
        else:
            cutoff_vgs, cutoff_i = probe
            vth_anchor = bool(cutoff_i <= ZERO_CURRENT_FRAC * (peak or 1.0))
            cutoff_anchor = vth_anchor

    measured = bool(finite and q.size and record) if emit == "ann" else bool(
        finite and q.size)
    failed = []
    if not finite:
        failed.append("non_finite_gm_gds")
    if not qmono:
        failed.append("negative_charge_derivative")
    if not vds_zero:
        failed.append("vds_zero_anchor")
    if cutoff_anchor is not True:
        failed.append("gate_cutoff_anchor")
    if emit == "ann" and not ann_ok:
        failed.append("ann_out_of_range")
    reason = ("; ".join(failed) if failed else
              "finite gm/gds; monotone Q; Vds=0 and gate-cutoff anchors near "
              "zero" + ("; ANN out-of-range pass" if emit == "ann" else
                        "; ANN probe not applicable to closed C0"))
    result.update({
        "pass": bool(finite and qmono and vds_zero and cutoff_anchor is True
                     and ann_ok),
        "status": "measured" if measured else "not_measured",
        "reason": reason,
        "gm_min": _json_safe(float(np.min(gm))),
        "gm_max": _json_safe(float(np.max(gm))),
        "gds_min": _json_safe(float(np.min(gds))),
        "gds_max": _json_safe(float(np.max(gds))),
        "charge_derivative_non_negative": qmono,
        "vds_zero_anchor": vds_zero,
        "vth_zero_anchor": vth_anchor,
        "cutoff_probe_vgs": _json_safe(cutoff_vgs),
        "out_of_range_check": out_of_range,
        "activation": record.get("activation"),
        "neurons": record.get("neurons"),
    })
    return result


def _timing_bars(vendor: str, *, emit: str) -> dict:
    """Recorded timing/energy bars for this channel basis; never a guess.

    ``ann`` is scored against the recorded golden table; the closed C0
    ``tier2`` fallback returns to its own recorded Tier-2 table (spec 6
    fallback decision).  No value is re-measured or re-baselined here.
    """
    if emit not in TIMING_BAR_COLUMN:
        raise ValueError(f"unknown emit mode {emit!r}; expected "
                         f"{tuple(TIMING_BAR_COLUMN)}")
    path = (registry.ROOT / "reports" / "comparison" / vendor
            / "dpt_metrics.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    bars = data.get(TIMING_BAR_COLUMN[emit]) or {}
    return {k: bars.get(k) for k in ("td_on_ns", "tr_ns", "td_off_ns",
                                     "tf_ns", "e_on_uj", "e_off_uj")}


def _timing_bars_source(emit: str) -> str:
    """Provenance label naming the G4 bar column actually used by ``emit``."""
    column = TIMING_BAR_COLUMN[emit]
    return (f"reports/comparison/<vendor>/dpt_metrics.json#{column} "
            f"(spec 6 G4 fallback decision; emit={emit}); "
            "README DPT timing and energy table")


def _timing_band(bar: float | None) -> tuple[float, float] | None:
    """Sourced band around a recorded timing/energy bar (spec G4).

    A much smaller delay is not automatically better: a device that never
    turns on has a tiny delay.  A missing bar stays unmeasured.
    """
    value = _num(bar)
    if value is None:
        return None
    half = abs(value) * TIMING_BAND_REL
    return value - half, value + half


def _ramp_numeric(xs: list[float], ys: list[float], eps: float,
                  query: np.ndarray) -> np.ndarray:
    """Numeric mirror of the emitted anchored delta-slope ramp chain."""
    b, q0, delta = _chain_coefficients(xs, ys)
    t = np.asarray(query, dtype=float)
    e = float(eps)
    if e == 0.0:
        out = np.full_like(t, q0, dtype=float)
        for i in range(1, len(b)):
            if abs(delta[i]) > 0.0 or i == 1:
                out = out + delta[i] * np.maximum(t - b[i - 1], 0.0)
        return out
    # Match approximation.smoother.delta_ramp_chain: the shifted hinge keeps
    # the chain's q0 baseline (the channel sat ramp has a nonzero baseline).
    anchor = q0 - float(np.dot(delta[1:], hyp_value(-b[:-1], e)))
    out = np.full_like(t, anchor, dtype=float)
    for i in range(1, len(b)):
        if abs(delta[i]) > 0.0 or i == 1:
            out = out + delta[i] * hyp_value(t - b[i - 1], e)
    return out


def _channel_sat_ramp(spec: PWLModelSpec) -> tuple[list[float], list[float]] | None:
    """The (Vgs, saturation-width) pair built by ``_channel_expression``.

    Thin view over :func:`_channel_transfer_ramp` so the emitted-gate mirror
    has one source of truth for the Vgs grid, anchor sites and fitted knee.
    """
    ramp = _channel_transfer_ramp(spec)
    if ramp is None:
        return None
    xs, _iv, sv, _cutoff = ramp
    return xs, sv


def _charge_checks(spec: PWLModelSpec, eps_by_table: dict, *,
                   emitted_eps_by_table: dict | None = None) -> dict:
    """Compare emitted ramp chains with their source tables.

    Charge tables are checked at the width actually emitted; the channel
    saturation ramp (whose baseline is nonzero) is checked as well, so a C1
    chain that drops that baseline can no longer pass G6.  Every number is
    computed from the spec/emitted widths; none is assumed.
    """
    effective = (eps_by_table if emitted_eps_by_table is None
                 else emitted_eps_by_table)
    tables = {
        "q_oss": (spec.q_oss_vs_v, spec.q_oss_charge_c),
        "q_iss": (spec.q_iss_vs_v, spec.q_iss_charge_c),
        "q_rss": (spec.q_rss_vs_v, spec.q_rss_charge_c),
    }
    per_table: dict[str, dict] = {}
    for name, (vv, qq) in tables.items():
        v = np.asarray(vv, dtype=float)
        q = np.asarray(qq, dtype=float)
        if v.size < 2 or q.size < 2 or v.size != q.size:
            per_table[name] = {
                "status": "not_measured", "charge_fit_error_pct": None,
                "breakpoint_crossings": None,
            }
            continue
        eps = _num(effective.get(name)) or 0.0
        fitted = _ramp_numeric(vv, qq, eps, v)
        q_max = abs(float(q[-1]))
        error = (100.0 * float(np.max(np.abs(fitted - q))) / q_max
                 if q_max > 0.0 else None)
        _, _, delta = _chain_coefficients(vv, qq)
        signs = np.sign(delta[1:])
        crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
        per_table[name] = {
            "status": "measured" if error is not None else "not_measured",
            "charge_fit_error_pct": _json_safe(error),
            "breakpoint_crossings": crossings,
            "eps": eps, "q_max": _json_safe(q_max),
        }

    channel = _channel_sat_ramp(spec)
    if channel is not None and spec.metadata.get("urom_channel_basis") != "ann":
        xs, sv = channel
        eps = _num(effective.get("transfer_vgs")) or 0.0
        fitted = _ramp_numeric(xs, sv, eps, np.asarray(xs, dtype=float))
        scale = max(abs(sv[0]), abs(sv[-1]))
        error = (100.0 * float(np.max(np.abs(fitted - np.asarray(sv))))
                 / scale if scale > 0.0 else None)
        _, _, delta = _chain_coefficients(xs, sv)
        signs = np.sign(delta[1:])
        crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
        per_table["channel_saturation"] = {
            "status": "measured" if error is not None else "not_measured",
            "charge_fit_error_pct": _json_safe(error),
            "breakpoint_crossings": crossings,
            "eps": eps, "q_max": _json_safe(scale),
            "baseline_v": _json_safe(sv[0]),
        }

    errors = [row["charge_fit_error_pct"] for row in per_table.values()
              if row["charge_fit_error_pct"] is not None]
    status = ("measured" if len(errors) == len(per_table) and errors
              else "not_measured")
    q = np.asarray(spec.q_oss_charge_c, dtype=float)
    v = np.asarray(spec.q_oss_vs_v, dtype=float)
    q0 = float(q[0]) if q.size else None
    return {
        "status": status,
        "monotone_q": bool(q.size > 1 and np.all(np.diff(q) >= 0)),
        "q_zero": _json_safe(q0),
        "q_zero_anchored": bool(q0 is not None and abs(q0) <= 1e-15),
        "charge_fit_error_pct": max(errors) if status == "measured" else None,
        "charge_fit_basis": "numeric mirror of the emitted delta-slope ramp "
                            "chain against the source charge tables and the "
                            "channel saturation-width ramp",
        "q_max": _json_safe(float(q[-1]) if q.size else None),
        "v_max": _json_safe(float(v[-1]) if v.size else None),
        "eps_by_table": eps_by_table,
        "emitted_eps_by_table": effective,
        "eps_rule": "median_positive_grid_spacing_cap_half_min_segment",
        "c0_bit_identical_at_eps0": None,
        "c0_ab_note": "not measured here; run_unified A/Bs the tier2 C0 "
                      "fallback against canonical build_tier2_subcircuit",
        "breakpoint_crossings": {
            name: row["breakpoint_crossings"] for name, row in
            per_table.items()},
        "per_table": per_table,
        "c_ds_scaling_excluded": True,
    }


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def _gate(passed: bool, status: str, detail: dict) -> dict:
    return {"pass": bool(passed), "status": status, "detail": detail}


def _evaluate_gates(record: dict, observations: dict) -> dict:
    vendor = record["vendor"]
    method = record["method"]
    equality = observations["equal_sandbox"]
    speed = observations["speed"]
    dpt = observations["dpt"]
    static = observations["static"]
    charge = record["c1"]
    tuning = record["tuning"]
    tuning_ok = (tuning.get("mode") == "off"
                 or tuning.get("verdict") == "accept")

    g0 = (equality["deck_diff_only_dut"]
          and equality["effective_options_dumped"]
          and not equality["tolerance_overrides"]
          and equality["libraries_inlined_both_arms"]
          and equality["temperature_both_arms"]
          and speed["repeats"] >= 3
          and all(a["completion"] for a in (speed["golden"], speed["urom"])))

    bar = STATIC_BARS_PCT.get(vendor)
    s_rmse = static.get("full_sweep_rel_rmse_pct")
    s_end = static.get("worst_endpoint_rel_error_pct")
    g1_pass = (bar is not None and s_rmse is not None and s_end is not None
               and s_rmse <= bar + G1_BAND_PP and s_end <= ENDPOINT_BAR_PCT)
    g1 = _gate(g1_pass,
               "measured" if s_rmse is not None else "not_measured",
               {"rel_rmse_pct": s_rmse, "bar_pct": bar,
                "estimator": static.get("gated_estimator"),
                "band_pp": G1_BAND_PP, "worst_endpoint_pct": s_end,
                "endpoint_bar_pct": ENDPOINT_BAR_PCT,
                "span_normalised_pct": static.get("span_normalised_rmse_pct"),
                "peak_normalised_pct": static.get("peak_normalised_rmse_pct")})

    derivative = observations.get("derivative") or {}
    g2 = _gate(bool(derivative.get("pass")),
               derivative.get("status", "not_measured"), derivative)

    bars = DPT_BARS_PCT.get(method, {}).get(vendor)
    dpt_rows = {}
    for label in ("golden", "urom"):
        arm = dpt[label]
        if not arm["completed"]:
            dpt_rows[label] = {"status": "infeasible" if label == "golden"
                               else "not_completed", "pass": False}
        else:
            wf = arm["waveform"]
            vds, ids = wf.get("vds_rel_rmse_pct"), wf.get("id_rel_rmse_pct")
            ok = (bars is not None and vds is not None and ids is not None
                  and vds <= bars[0] and ids <= bars[1])
            dpt_rows[label] = {"status": "measured", "pass": ok,
                               "vds_rel_rmse_pct": vds,
                               "id_rel_rmse_pct": ids,
                               "bars_vds_id_pct": bars,
                               "compare_points": wf.get("compare_points")}
    g3 = _gate(all(r["pass"] for r in dpt_rows.values()),
               "measured" if all(r["status"] == "measured"
                                 for r in dpt_rows.values())
               else "infeasible", dpt_rows)

    timed = observations.get("timing_energy") or {}
    g4 = _gate(bool(timed.get("pass")), timed.get("status", "not_measured"),
               timed)

    speed_row = record["speed_adaptive_20"]
    has_counts = all(speed_row.get(k) for k in
                     ("n_pulses", "method_label", "t_last", "iterations",
                      "points"))
    g5 = _gate(bool(has_counts and speed["golden"]["completion"]
                    and speed["urom"]["completion"]),
               "measured" if has_counts else "not_measured",
               {"scoped_row_valid": bool(has_counts),
                "method_label": speed_row.get("method_label"),
                "n_pulses": speed_row.get("n_pulses"),
                "max_step": speed_row.get("max_step"),
                "completion": {"golden": speed["golden"]["completion"],
                               "urom": speed["urom"]["completion"]},
                # Spec 1.3/6: one declared-method row is scoped.  A global
                # claim needs the default AND gear rows (min >= 1.5), so the
                # recorded row carries no gear minimum here.  Surface those
                # recorded fields; never recompute them from imported ratios.
                "global_claim": bool(speed_row.get("global_claim", False)),
                "min_default_gear": speed_row.get("min_default_gear")})

    crossings = charge.get("breakpoint_crossings") or {}
    charge_measured = bool(
        charge.get("status") == "measured" and crossings
        and all(isinstance(v, int) and v >= 0 for v in crossings.values()))
    g6_pass = bool(
        charge_measured and charge["monotone_q"] and charge["q_zero_anchored"]
        and charge["charge_fit_error_pct"] is not None
        and charge["charge_fit_error_pct"] <= CHARGE_BUDGET_PCT
        and charge["c_ds_scaling_excluded"]
        and tuning_ok)
    g6 = _gate(g6_pass, "measured" if charge_measured else "not_measured",
               {"monotone_q": charge["monotone_q"],
                "q_zero_anchored": charge["q_zero_anchored"],
                "charge_fit_error_pct": charge["charge_fit_error_pct"],
                "budget_pct": CHARGE_BUDGET_PCT,
                "breakpoint_crossings": crossings,
                "c_ds_scaling_excluded": charge["c_ds_scaling_excluded"],
                "tuning_verdict": record["tuning"]["verdict"]})

    gates: dict[str, Any] = {"G0": _gate(g0, "measured", equality),
             "G1_static": g1, "G2_derivative": g2, "G3_dpt_waveform": g3,
             "G4_timing_energy": g4, "G5_speed": g5, "G6_charge": g6}
    gates["all_pass"] = all(g["pass"] for k, g in gates.items()
                            if k != "all_pass")
    return gates


def _speed_row(plan: dict, speed: dict) -> dict:
    golden, rom = speed["golden"], speed["urom"]
    counts_ready = golden["completion"] and rom["completion"]
    solver = _num(golden["solver_time_s"] / rom["solver_time_s"]) \
        if counts_ready and rom["solver_time_s"] else None
    wall = _num(golden["wall_time_s"] / rom["wall_time_s"]) \
        if counts_ready and rom["wall_time_s"] else None
    return {
        "n_pulses": plan["speed"]["n_pulses"],
        "max_step": plan["speed"]["max_step"],
        "method_label": plan["method_label"],
        "solver_speedup": solver,
        "wall_speedup": wall,
        "t_last": {"golden": golden["t_last"], "urom": rom["t_last"]},
        "iterations": {"golden": golden["iterations"],
                       "urom": rom["iterations"]},
        "points": {"golden": golden["points"], "urom": rom["points"]},
        "global_claim": False,
        "min_default_gear": None,
        "completion_basis": "iterations > 0; trace points >= 50 percent of "
                            "median; t_last >= 0.999 x scheduled end; "
                            "3 interleaved repeats; aborts are failures",
    }


def _runtime_record(speed: dict) -> dict:
    """Serialize both runtime clocks: medians plus min/max repeat spread.

    ``_run_interleaved_speed`` guards each extreme with ``default=None``, so
    an unmeasured arm stays null here; no zero is substituted.
    """
    golden, rom = speed["golden"], speed["urom"]
    return {
        "iterations": {"golden": golden["iterations"],
                       "urom": rom["iterations"]},
        "points": {"golden": golden["points"], "urom": rom["points"]},
        "t_last": {"golden": golden["t_last"], "urom": rom["t_last"]},
        "repeats": speed["repeats"],
        "solver_time_s": {"golden": golden["solver_time_s"],
                          "urom": rom["solver_time_s"]},
        "wall_time_s": {"golden": golden["wall_time_s"],
                        "urom": rom["wall_time_s"]},
        "median": {"solver": {"golden": golden["solver_time_s"],
                              "urom": rom["solver_time_s"]},
                   "wall": {"golden": golden["wall_time_s"],
                            "urom": rom["wall_time_s"]}},
        "min": {"solver": {"golden": golden["solver_min_s"],
                           "urom": rom["solver_min_s"]},
                "wall": {"golden": golden["wall_min_s"],
                         "urom": rom["wall_min_s"]}},
        "max": {"solver": {"golden": golden["solver_max_s"],
                           "urom": rom["solver_max_s"]},
                "wall": {"golden": golden["wall_max_s"],
                         "urom": rom["wall_max_s"]}},
        "completion": {"golden": golden["completion"],
                       "urom": rom["completion"]},
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def run_unified(vendor: str, *, variant: str = DEFAULT_VARIANT,
                method: str = "trap/default", emit: str = "ann",
                tuning: str = "off", repeats: int = SPEED_REPEATS) -> dict:
    """Build one unified ROM, run the equal-sandbox benches and record gates.

    The caller chooses the channel basis (``ann`` or the closed ``tier2``
    fallback), the declared solver method, and whether the fixed-topology
    scalar tuning layer is applied.  Nothing is fabricated: unmeasured or
    non-completing quantities stay null / N/A and fail their gate.
    """
    if vendor not in registry.VENDORS:
        raise ValueError(f"unknown vendor {vendor!r}; expected one of "
                         f"{registry.VENDORS}")
    if variant != registry.UNIFIED_VARIANT:
        raise ValueError(f"unsupported unified variant {variant!r}; expected "
                         f"'{registry.UNIFIED_VARIANT}'")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected {METHODS}")
    if emit not in EMIT_MODES:
        raise ValueError(f"unknown emit mode {emit!r}; expected {EMIT_MODES}")
    if tuning not in TUNING_MODES:
        raise ValueError(f"unknown tuning mode {tuning!r}; expected "
                         f"{TUNING_MODES}")
    if int(repeats) < 3:
        raise ValueError("repeats must be at least 3 (equal-sandbox G0)")

    plan = _bench_plan(vendor, method, int(repeats))
    lib, spec, ann_record, tuning_record = _build_emitted(
        vendor, emit=emit, tuning=tuning, plan=plan)
    golden, rom = _arm_specs(vendor, lib)

    dpt_decks = {"golden": _dpt_deck(plan, golden),
                 "urom": _dpt_deck(plan, rom)}
    speed_decks = {"golden": _speed_deck(plan, golden),
                   "urom": _speed_deck(plan, rom)}
    equality = _deck_identity(dpt_decks["golden"], dpt_decks["urom"],
                              plan["options_lines"],
                              expected_temperature=plan["temperature"])
    speed_identity = _deck_identity(speed_decks["golden"],
                                    speed_decks["urom"],
                                    plan["options_lines"],
                                    expected_temperature=plan["temperature"])
    equality["deck_diff_only_dut"] = (
        equality["deck_diff_only_dut"] and speed_identity["deck_diff_only_dut"])
    equality["effective_options_dumped"] = (
        equality["effective_options_dumped"]
        and speed_identity["effective_options_dumped"])
    equality["libraries_inlined_both_arms"] = (
        equality["libraries_inlined_both_arms"]
        and speed_identity["libraries_inlined_both_arms"])
    equality["temperature_both_arms"] = (
        equality["temperature_both_arms"]
        if equality["temperature_both_arms"] == speed_identity[
            "temperature_both_arms"] else None)
    equality["tolerance_overrides"] = sorted(set(
        equality["tolerance_overrides"]
        + speed_identity["tolerance_overrides"]))
    equality["speed_deck_effective_options"] = speed_identity[
        "effective_options"]

    speed = _run_interleaved_speed(plan, golden, rom, speed_decks)
    traj = capture_trajectory(vendor)
    dpt_cfg = load_input(vendor)["dpt"]
    dpt = {label: _run_dpt_arm(plan, label, deck, traj, dpt_cfg)
           for label, deck in dpt_decks.items()}
    static, static_rows = _static_bench(vendor, lib)
    derivative = _derivative_checks(static_rows, spec, ann_record, emit)
    timing_energy: dict = {"pass": False, "status": "not_measured",
                           "reason": "no crossing extraction for this method"}
    if dpt["urom"]["completed"] and dpt["golden"]["completed"]:
        reference = _timing_bars(vendor, emit=emit)
        actual = dpt["urom"]["metrics"]
        compared: dict[str, dict] = {}
        ok = True
        for metric in ("td_on_ns", "tr_ns", "td_off_ns", "tf_ns",
                       "e_on_uj", "e_off_uj"):
            value = _num(actual.get(metric))
            bar = _num(reference.get(metric))
            band = _timing_band(bar)
            if value is None or band is None:
                compared[metric] = {"status": "not_measured",
                                    "value": value, "bar": bar,
                                    "band": list(band) if band else None}
                ok = False
                continue
            good = bool(band[0] <= value <= band[1])
            compared[metric] = {"status": "measured", "value": value,
                                "bar": bar, "band": list(band), "pass": good}
            ok = ok and good
        timing_energy = {
            "pass": ok,
            "status": "measured" if len(compared) == 6 else "not_measured",
            "rows": compared,
            "band_source": _timing_bars_source(emit),
            "band_rel": TIMING_BAND_REL,
        }

    eps_by_table = _eps_by_table(spec)
    emitted_eps = _emitted_eps_by_table(spec)
    c1_record = dict(_charge_checks(
        spec, eps_by_table, emitted_eps_by_table=emitted_eps))
    c1_variant = spec.metadata.get("urom_c1_variant")
    if emit == "tier2" and c1_variant is None:
        # The fallback must be the closed canonical C0 emitter, modulo the
        # UROM subckt name.  Measure it instead of claiming it.
        c1_record["c0_bit_identical_at_eps0"] = _c0_bit_identity(
            spec, lib.read_text(encoding="ascii"), lib.stem)
        c1_record["c0_ab_note"] = (
            "A/B against canonical build_tier2_subcircuit(c1_eps=None) "
            "at the same UROM subckt name")
    # ann_fit_for_vendor has no 'c1' key; never let a foreign record replace
    # the residual/count measurements computed above.
    fit_record = (ann_record.get("record") or {}) if emit == "ann" else {}
    if tuning == "off":
        tuning_record["theta"] = {
            k: None for k in ("A_GD", "O_GD", "A_GS", "A_RGI")}

    record: dict[str, Any] = {
        "schema": SCHEMA,
        "vendor": vendor,
        "variant": registry.UNIFIED_VARIANT,
        "emitted_lib": str(lib.relative_to(ROOT)).replace("\\", "/"),
        "channel_basis": emit,
        "c1_variant": c1_variant,
        "method": plan["method_label"],
        "equal_sandbox": equality,
        "data_provenance": {
            "curves": sorted(p.name for p in
                             (registry.ROOT / "config" / "outputs" / vendor)
                             .glob("curve*.yaml")),
            "dpt_high_vi": {
                "source": "dpt_trajectory", "vendor": vendor,
                "conditions": {"config": f"config/inputs/{vendor}.yaml#dpt"},
                "sample_indices": list(
                    (((ann_record.get("provenance") or {})
                      .get("dpt_high_vi") or {}).get("sample_indices")
                     or [])),
            },
            "external_source_status":
                PROVENANCE["external_sources"]["status"],
            "guard_width": (
                dict(((ann_record.get("provenance") or {}).get("guard_width")
                      or {})) if emit == "ann" else None),
        },
        "ann_fit": {
            "activation": fit_record.get("activation"),
            "neurons": fit_record.get("neurons"),
            "loss_weights": dict(fit_record.get("loss_weights") or {}),
            "gm_gds_q_derivative_terms": list(
                fit_record.get("gm_gds_q_derivative_terms") or []),
            "train_range": dict(fit_record.get("train_range") or {}),
            "q0_offset": fit_record.get("q0_offset"),
            "q0_offset_by_table": dict(
                fit_record.get("q0_offset_by_table") or {}),
            "q0_offset_source": fit_record.get("q0_offset_source"),
            "guard_width_v": fit_record.get("guard_width_v"),
            "out_of_range_check": fit_record.get("out_of_range_check"),
            "status": "measured" if fit_record else "not_measured",
        } if emit == "ann" else {
            "activation": None, "neurons": None, "loss_weights": {},
            "gm_gds_q_derivative_terms": [], "train_range": {},
            "q0_offset": None, "q0_offset_by_table": {},
            "q0_offset_source": None, "out_of_range_check": None,
            "guard_width_v": None,
            "status": "not_applicable",
        },
        "c1": c1_record,
        "tuning": tuning_record,
        "gates": {},
        "runtime": _runtime_record(speed),
        "speed_adaptive_20": _speed_row(plan, speed),
        "fidelity_scope": {"waveform_gate_retired": False,
                           "energy_only_claim": False},
        "spice_log_tail": speed["urom"]["log_tail"],
        # Reuse tier3_validation/2.0-style keys where possible.
        "dpt_run": dpt["urom"],
        "dpt_waveform": dpt["urom"]["waveform"],
        "dpt_comparison": dpt,
        "static": static,
        "equal_sandbox_detail": {
            "dpt_decks": equality, "speed_decks": speed_identity,
            "timing_energy_bars_source":
                _timing_bars_source(emit),
        },
        "urom_provenance": spec.metadata["urom_provenance"],
        "verdict": "pending",
    }
    record["gates"] = _evaluate_gates(
        record, {"equal_sandbox": equality, "speed": speed, "dpt": dpt,
                 "static": static, "derivative": derivative,
                 "timing_energy": timing_energy})
    record["verdict"] = ("accept" if record["gates"]["all_pass"] else "reject")
    record = _json_safe(record)

    path = _record_path(vendor)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n",
                    encoding="ascii")
    return record


def compare_unified(vendor: str) -> dict:
    """Read the recorded unified validation JSON; never recompute a number."""
    if vendor not in registry.VENDORS:
        raise ValueError(f"unknown vendor {vendor!r}; expected one of "
                         f"{registry.VENDORS}")
    path = _record_path(vendor)
    if not path.exists():
        return {"present": False, "schema": SCHEMA, "vendor": vendor}
    data = json.loads(path.read_text(encoding="utf-8"))
    data["present"] = True
    return data
