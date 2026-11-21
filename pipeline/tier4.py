"""End-to-end Tier-4 (fast converter tier) build and validation.

Tier-4 is an ideal switch plus a measured loss LUT. Its DPT waveform gate is
EXPLICITLY RETIRED: P-DEMO documents the Miller gate droop, diode reverse
recovery and the finite SPICE transients this ideal-switch tier cannot show;
dv/dt and ringing are [INFERENCE] from instantaneous switching and absent
parasitics, not documented observables on any consulted Plexim page.
The gate recorded here is energy/efficiency against the measured golden DPT
plus a guarded 20-pulse adaptive speed comparison. No tolerance, step or
method knob is changed in either arm (research section 5 gate 1).
A faster reduced model legitimately completes with far fewer solver iterations
and trace points than the golden arm; count similarity is not a gate.

Writes:
  reports/models/<vendor>_PIECESPICE_T4.lib          emitted ASCII subcircuit
  reports/comparison/<vendor>/tier4_validation.json   metrics + provenance

Every number is either read from config/inputs/<vendor>.yaml, from a recorded
measured artifact, or produced by a guarded run. NaN is converted to null
(N/A) on export; it is never reported as zero or as a fast timing.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any
from config.yaml_io import load_input
from approximation.loss_lut import build_lut
from approximation.thermal import ThermalNetwork
from common.provenance import sha256_file
from models.tier4_ideal_switch_builder import (
    build_tier4_subcircuit,
    resolve_r_on,
    resolve_vt,
)
from parser.netlist import resolve_pin_connections
from simulation.engine import SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder
from simulation.testbenches.speedup_bench import (
    _pulse_train,
    run_inductive_benchmark,
)

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "piecespice.tier4/1.0"
WAVEFORM_GATE = "retired_by_design"
ENERGY_METRICS = ("e_on_uj", "e_off_uj")
MEASURED_LUT_STATUS = ("measured", "constant_node", "single_node",
                       "interpolated")

# Finite-node statuses above come from approximation/loss_lut.py: a true 1x1
# grid reports "constant_node", a lone finite node in a larger grid reports
# "single_node", and real all-corner interpolation reports "interpolated".
# None of them implies extrapolation, which the LUT no longer supports.

# Named, labeled decisions -- not new tolerances.  The two DPT guard fractions
# are reused from pipeline/tier3.py via loss_lut_bench; the bench knobs are
# pinned project decisions and are republished under provenance.decisions.
DPT_T_LAST_FRACTION = 0.999
DPT_TRACE_POINT_FRACTION = 0.5
EMITTED_DPT_TIMEOUT_S = 120.0
SPEED_N_PULSES = 20
SPEED_REPEATS = 3
SPEED_TIMEOUT_S = 90.0
CLAMP_DIODE_MODEL = ".model DCLAMP D(IS=1e-12 N=1 RS=1e-3 CJO=0)"

# Completion statement republished verbatim in the JSON completion_basis of
# both measured and N/A speedup rows. A count-ratio fraction is deliberately
# absent: fewer iterations/points are the expected signature of a speedup.
SPEED_COMPLETION_BASIS = (
    "count similarity is not required; lower counts are expected for "
    "faster models; aborted runs are rejected by positive iterations/"
    "points and no error. Both arms must also be full-schedule "
    "run_inductive_benchmark records: end_time_s equals the scheduled "
    "20-pulse .tran stop and repeats == 3. The configured .tran stop is "
    "schedule metadata, not a measured trace endpoint."
)

# Mandatory method label on every Tier-4 speed row (research G0 item 1 / gate
# 1: "method declared in every row"). The recorded benchmark rows use the
# LTspice default/trap integrator on BOTH arms; matched-Gear numbers live only
# as a separate declared diagnostic in the research synthesis and must never
# replace or be blended into these rows. Prose/metadata only: no solver knob.
SPEED_METHOD = "LTspice default/trap (both arms)"

GOLDEN_METRICS_SUFFIX = "#golden_metrics"
ENERGY_GATE_DISCLOSURE = (
    "LUT reproduces its own golden DPT crossing; not an independent "
    "converter validation (no converter bench was built)."
)
THERMAL_LOSS_NOTE = (
    "PLECS P-TC: the dissipated thermal power does not correspond to the "
    "electrical power consumed by the device; thermal losses are injected "
    "into a thermal network and need an explicit output-power definition for "
    "efficiency. The emitted ideal switch has no loss element, so every "
    "efficiency figure here is an offline estimate, not simulated electrical "
    "power."
)

# The 1 / (dpt.t2_on - dpt.t1_on) value is the DPT bench pulse repetition
# interval, not a measured converter switching frequency. Reported only so
# p_switch is a labelled linear scaling diagnostic; no converter is modelled.
F_SW_BASIS_NOTE = (
    "f_sw = 1 / (dpt.t2_on - dpt.t1_on) is an illustrative offline frequency "
    "from the DPT pulse repetition interval (a bench schedule choice), NOT a "
    "measured converter switching frequency; p_switch scales linearly with "
    "it and remains a diagnostic only."
)

OFFICIAL_SOURCES = {
    "P-SEM": "https://www.plexim.com/products/plecs/electrical/semiconductors",
    "P-TC": "https://docs.plexim.com/plecs/5.0/thermal-modeling/thermal-concepts",
    "P-DEMO": "https://docs.plexim.com/plecs-demos/plecs-spice/demo-plecs-spice/",
}

# The CPM2 golden adaptive speed is contract-excluded, not a re-measured
# failure claim: older recorded text cited a golden abort (no raw output /
# rc=1, research section 2), while the current recorded baseline artifact
# holds a completed adaptive golden arm.  Both readings are disclosed in
# vendor_scope_notes at the record root.  Prose/metadata only: no number,
# guard or tolerance.
GOLDEN_SPEED_EXCLUSIONS = {
    "cpm2-1200": (
        "CPM2-1200 golden adaptive speed is contract-excluded (VM-10 vendor "
        "turn-off abort; fidelity_speed_research.md section 2, Tier-4 contract "
        "section 20). Older recorded text cited a golden failure "
        "('no raw output / rc=1'); the current "
        "reports/comparison/speedup_benchmark.json instead holds a completed "
        "adaptive golden arm. That artifact conflict is disclosed in "
        "vendor_scope_notes (record root); no ratio is derived and the status stays N/A."
    ),
}

# Vendor-scope disclosure for that excluded CPM2 speed: VM-10 truncates the
# vendor DPT waveform, so only the golden adaptive speed is N/A.  Energy/
# efficiency stay reported from the finite measured 2-pulse DPT crossing.
VENDOR_SPEED_SCOPE_NOTES = {
    "cpm2-1200": {
        "kind": "vendor_model_defect",
        "defect": "VM-10",
        "reference": "vendor_model_defects.md VM-10 (CPM2 turn-off abort)",
        "note": (
            "VM-10 vendor turn-off abort truncates the CPM2-1200 golden "
            "waveform. The Tier-4 contract (section 20) excludes only this "
            "vendor's golden adaptive speed: speedup.status stays N/A and no "
            "solver/wall ratio is derived. Energy/efficiency remain reported "
            "from the finite measured 2-pulse DPT crossing (a different "
            "observable); the emitted Tier-4 model and all guards are unchanged."
        ),
        "artifact_conflict": (
            "Older recorded text reported a golden adaptive failure "
            "('no raw output / rc=1', fidelity_speed_research.md section 2); "
            "the current reports/comparison/speedup_benchmark.json case "
            "long_adaptive instead holds a completed adaptive golden arm "
            "(golden_completed=true). Both readings are disclosed; the "
            "completed legacy golden/Tier-2 case is context only, never a "
            "Tier-4 speedup, and the contract-pinned N/A outcome is unchanged."
        ),
        "source": (
            "vendor_model_defects.md VM-10; docs/fidelity_speed_research.md "
            "section 2; reports/comparison/speedup_benchmark.json "
            "cases.long_adaptive"
        ),
    },
}

_HEADER_RE = re.compile(r"^\s*\.SUBCKT\s+(\S+)\s+(.+)$",
                        re.IGNORECASE | re.MULTILINE)
_PIN_ALIASES = {"d": "drain", "g": "gate", "s": "source"}

# Convergence root cause (measured, controlled ablations) and the exact
# boundary of the stabilization claim.  Disclosures, not new tolerances or
# waveform-fidelity claims; see local://tier4_convergence_rootcause.md.
SOLVER_STABILIZATION_REF = "local://tier4_convergence_rootcause.md"
SOLVER_STABILIZATION_RATIONALE = (
    "Controlled DPT/adaptive ablations (local://tier4_convergence_rootcause.md) "
    "identify the missing drain output capacitance as the primary, sufficient "
    "cause of Tier-4 solver timeouts: with only the ideal switch, clamp diode "
    "and inductor there is no bounded state variable at the near-step dV/dt, "
    "so LTspice never reaches t_end. One measured linear C_QOSS using the "
    "vendor Curve-4 Coss at cap_ref_Vds makes every previously failing arm "
    "complete; it slows drain dV/dt and adds L-C overshoot. The measured "
    "Vgs=0 off-state Roff removes a secondary, vendor-specific 1e12 "
    "singularity but is not a common fix. Solver stabilization only."
)
SOLVER_STABILIZATION_DISCLOSURE = (
    "C_QOSS is a single measured linear output capacitance (curve4.coss_f at "
    "cap_ref_Vds), added only to keep the drain node bounded and let the "
    "solver reach t_end; it is NOT a voltage-dependent charge model and no "
    "waveform fidelity is claimed. It slows drain dV/dt and adds L-C "
    "overshoot. The DPT waveform gate remains retired_by_design; dv/dt, "
    "ringing and Miller droop stay retired/unclaimed. Voltage-dependent "
    "capacitors and the body diode remain retired; loss tables remain "
    "comment-only metadata."
)
_C_QOSS_RE = re.compile(
    r"^C_QOSS\s+(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE | re.MULTILINE)
_SW_MODEL_RE = re.compile(
    r"^\.MODEL\s+SW_T4\s+SW\((.*)\)\s*$", re.IGNORECASE | re.MULTILINE)
_ROFF_RE = re.compile(r"\bRoff\s*=\s*([^\s)]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# small helpers (offline, no benches)
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively turn non-finite floats into None (explicit N/A in JSON)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (bool, str, int)) or obj is None:
        return obj
    if isinstance(obj, (float,)) or hasattr(obj, "item"):
        try:
            value = float(obj)
        except (TypeError, ValueError):
            return str(obj)
        return value if math.isfinite(value) else None
    return obj


def _num(value) -> float | None:
    """A finite float or None; NaN/Inf are not numbers for a gate."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _parse_emitted_header(text: str) -> tuple[str, list[str]]:
    """Read subcircuit name and positional role list from emitted text."""
    match = _HEADER_RE.search(text)
    if match is None:
        raise ValueError("emitted Tier-4 netlist has no .SUBCKT header")
    name = match.group(1)
    roles: list[str] = []
    for token in match.group(2).split():
        if token.lower().startswith("params:"):
            break
        port = token.split("=")[0].lower()
        roles.append(_PIN_ALIASES.get(port, port))
    return name, roles


def _emitted_stabilization(text: str) -> dict:
    """Read C_QOSS and SW_T4 Roff actually written to the emitted lib."""
    cap = _C_QOSS_RE.search(text)
    model = _SW_MODEL_RE.search(text)
    roff = _ROFF_RE.search(model.group(1)) if model else None
    return {
        "c_qoss_emitted": cap is not None,
        "c_qoss_value_f": _num(cap.group(3)) if cap else None,
        "c_qoss_line": cap.group(0).strip() if cap else None,
        "roff_emitted": roff is not None,
        "roff_ohms": _num(roff.group(1)) if roff else None,
    }


def _solver_stabilization(lut, text: str, emission: dict) -> dict:
    """Measured C_QOSS/Roff provenance plus the convergence root cause.

    The LUT owns the measured values and their source dicts; this block
    republishes them alongside what the emitter actually wrote, so a fallback
    or a missing measurement stays explicit instead of becoming a fabricated
    stabilizing value.
    """
    raw_source = getattr(lut, "source", None)
    source = raw_source if isinstance(raw_source, dict) else {}
    coss_meta = dict(source.get("coss_f") or {})
    roff_meta = dict(source.get("r_off_ohms") or {})
    emitted = _emitted_stabilization(text)
    coss_value = _num(getattr(lut, "coss_f", None))
    roff_value = _num(getattr(lut, "r_off_ohms", None))
    kind = str(roff_meta.get("kind") or "")
    status = ("N/A" if emission["status"] != "emitted" else
              "applied" if emitted["c_qoss_emitted"] and coss_value is not None
              else "not_applied")
    basis = ("N/A" if emission["status"] != "emitted" else
             "measured" if kind == "measured" and roff_value is not None
             else "project_decision_fallback")
    return {
        "status": status,
        "rationale": SOLVER_STABILIZATION_RATIONALE,
        "root_cause_reference": SOLVER_STABILIZATION_REF,
        "waveform_gate": WAVEFORM_GATE,
        "c_qoss": {
            "value_f": coss_value,
            "emitted": emitted["c_qoss_emitted"],
            "emitted_value_f": emitted["c_qoss_value_f"],
            "element": emitted["c_qoss_line"],
            "source": _json_safe(coss_meta),
            "note": ("Measured linear output capacitance used only as a "
                     "bounded drain-node state variable for the solver."),
        },
        "r_off": {
            "value_ohms": roff_value,
            "emitted_ohms": emitted["roff_ohms"],
            "basis": basis,
            "source": _json_safe(roff_meta),
            "note": ("Measured Vgs=0 DC-sweep off-state resistance used in "
                     "SW_T4 when finite; otherwise the emitter falls back to "
                     "the project-decision Roff with a loud comment and this "
                     "disclosure."),
        },
        "disclosure": SOLVER_STABILIZATION_DISCLOSURE,
    }


def _recorded_artifacts(vendor: str) -> tuple[dict, dict]:
    """Recorded measured artifacts used as the golden reference/provenance."""
    dpt = _read_json(ROOT / "reports" / "comparison" / vendor /
                     "dpt_metrics.json")
    speed = _read_json(ROOT / "reports" / "comparison" /
                       "speedup_benchmark.json")
    return dpt, speed


def _speed_policy(vendor: str, speed: dict) -> dict:
    """Derive the golden bench case/step from recorded data, not vendor lore.

    EPC's adaptive stall is recorded as an error on ``long_adaptive`` with a
    completed ``long_100ns`` case; the 100 ns cap is read from that case's
    ``max_step``. An excluded golden (pinned contract) is reported N/A.
    """
    excluded = GOLDEN_SPEED_EXCLUSIONS.get(vendor)
    if excluded:
        return {"case": "long_adaptive", "golden_max_step": None,
                "golden_failure": excluded, "cap_reason": None,
                "decision": "contract-excluded golden speed kept N/A (VM-10; contract section 20; artifact conflict disclosed)"}
    cases = speed.get("cases", {})
    vendor_cases = speed.get("vendors", {}).get(vendor, {})
    adaptive = vendor_cases.get("long_adaptive", {}).get("golden", {})
    if not adaptive.get("error"):
        return {"case": "long_adaptive", "golden_max_step": None,
                "golden_failure": None, "cap_reason": None,
                "decision": "fully adaptive golden (recorded run completed)"}
    capped = vendor_cases.get("long_100ns", {}).get("golden", {})
    if capped.get("error"):
        return {"case": "long_adaptive", "golden_max_step": None,
                "golden_failure": adaptive["error"], "cap_reason": None,
                "decision": "no completed golden case; speed N/A"}
    return {"case": "long_100ns",
            "golden_max_step": cases.get("long_100ns", {}).get("max_step"),
            "golden_failure": None, "cap_reason": adaptive["error"],
            "decision": "documented 100 ns golden cap from recorded long_100ns case"}


def _energy_comparison(lut, dpt_record: dict, vbus: float,
                       i_switch: float, vendor: str = "",
                       temperature_c: float | None = None) -> dict:
    """LUT energy vs the golden crossing that produced the same LUT node.

    The two columns are not independent: the LUT node is filled from this
    vendor's ``dpt_metrics.json#golden_metrics``.  A zero error is therefore
    a source self-consistency check and every row says so explicitly.
    """
    if temperature_c is None:
        predicted = lut.energy_at(vbus, i_switch)
    else:
        predicted = lut.energy_at(vbus, i_switch,
                                  temperature_c=float(temperature_c))
    golden = dpt_record.get("golden_metrics", {})
    golden_ok = bool(golden) and not dpt_record.get("golden_failure")
    crossing_ok = getattr(lut, "source", {}).get("dpt_crossing_evaluable", True)
    golden_ok = golden_ok and bool(crossing_ok)
    shared_source = (
        f"reports/comparison/{vendor}/dpt_metrics.json{GOLDEN_METRICS_SUFFIX}"
        if vendor else f"dpt_metrics.json{GOLDEN_METRICS_SUFFIX}")
    disclosure = {
        "independent_energy_gate": False,
        "source_shared": True,
        "kind": "source_self_consistency",
        "shared_source": shared_source,
        "disclosure": ENERGY_GATE_DISCLOSURE,
    }
    rows: dict = {}
    for metric in ENERGY_METRICS:
        tier4 = _num(predicted.get(metric))
        ref = _num(golden.get(metric))
        evaluable = (golden_ok and tier4 is not None and ref is not None
                     and predicted.get("status") in MEASURED_LUT_STATUS)
        row: dict[str, Any] = {"golden": ref, "tier4": tier4}
        if evaluable and tier4 is not None and ref is not None:
            row["abs_error"] = abs(tier4 - ref)
            row["rel_error_pct"] = (
                100.0 * (tier4 - ref) / abs(ref) if ref else None)
            row["status"] = "measured"
        else:
            row["abs_error"] = None
            row["rel_error_pct"] = None
            row["status"] = "N/A"
            row["reason"] = (
                f"LUT status {predicted.get('status')!r} is not a measured "
                "crossing, or the golden DPT is not evaluable"
            )
        row.update(disclosure)
        row["source"] = predicted.get("source")
        rows[metric] = row
    return rows


def _scheduled_bench_end(bench_kw: dict) -> float:
    """Mirror run_inductive_benchmark's scheduled .tran stop for these knobs.

    The 1 us / 2 us offsets belong to the bench schedule (not new
    tolerances). Recomputing the stop here makes the arm's recorded
    end_time_s a checkable condition proof, not a self-reported claim.
    """
    n_pulses = int(bench_kw["n_pulses"])
    last_width = (bench_kw["first_pulse_width"] if n_pulses == 1
                  else bench_kw["pulse_width"])
    return (1.0e-6 + (n_pulses - 1) * bench_kw["pulse_period"]
            + last_width + 2.0e-6)


def _bench_complete(left: dict, right: dict, *,
                    scheduled_end_s: float) -> bool:
    """Anti-fake guard: each arm must be a complete, full-schedule run.

    ``run_inductive_benchmark`` returns measured iterations, trace_points and
    solver time per arm but discards the per-run time vector, so no measured
    trace endpoint exists. Its ``end_time_s`` is the configured .tran stop:
    here it is only a condition proof (it must equal the independently
    computed full-window stop) and is never a measured trace endpoint.
    Count similarity is deliberately not a gate: a faster model SHOULD use
    fewer iterations/points. Zero counts, a bench error, a wrong schedule or
    a partial repeat set keep speed N/A.
    """
    for arm in (left, right):
        if arm.get("error"):
            return False
        for key in ("solver_elapsed_s", "iterations", "trace_points"):
            value = _num(arm.get(key))
            if value is None or value <= 0.0:
                return False
        end_time_s = _num(arm.get("end_time_s"))
        if (end_time_s is None
                or not math.isclose(end_time_s, scheduled_end_s,
                                    rel_tol=1e-12, abs_tol=0.0)):
            return False
        if _num(arm.get("repeats")) != SPEED_REPEATS:
            return False
    return True


def _thermal_network_from_config(
    cfg: dict,
) -> tuple[ThermalNetwork | None, dict]:
    """Parse optional ``config.thermal`` without inventing any default.

    Absent config is ``not_configured``; a malformed network is
    ``invalid_config`` with a reason.  Neither case fabricates an ambient
    temperature or a junction temperature.
    """
    raw = cfg.get("thermal")
    if raw is None:
        return None, {
            "status": "not_configured",
            "tj_steady_c": None,
            "thermal_network": None,
            "source": None,
            "reason": ("no config.thermal block; junction temperature is N/A "
                       "rather than a fabricated ambient"),
            "rule": ("thermal injection requires explicit r_th_k_per_w, tau_s "
                     "and ambient_c from config; none are defaulted"),
        }
    if not isinstance(raw, dict):
        return None, {
            "status": "invalid_config",
            "tj_steady_c": None,
            "thermal_network": None,
            "source": "config.thermal",
            "reason": "config.thermal must be a mapping",
            "rule": "malformed thermal config is N/A, never defaulted",
        }
    try:
        network = ThermalNetwork.from_mapping(
            raw, name="tier4_thermal",
            source={"config": "config.thermal"})
    except ValueError as exc:
        return None, {
            "status": "invalid_config",
            "tj_steady_c": None,
            "thermal_network": None,
            "source": "config.thermal",
            "reason": str(exc),
            "rule": "malformed thermal config is N/A, never defaulted",
        }
    return network, {
        "status": "configured",
        "tj_steady_c": None,
        "thermal_network": network.to_dict(),
        "source": "config.thermal",
        "reason": None,
        "rule": ("Tj = Tamb + p_loss*sum(R_th) for the explicit network; "
                 "p_loss is the offline LUT thermal estimate (PLECS P-TC), "
                 "not simulated electrical power"),
    }


def _thermal_block(cfg: dict, p_loss_w: float | None
                   ) -> tuple[dict, dict | None]:
    """Thermal record plus normalized network; Tj only from explicit inputs."""
    network, block = _thermal_network_from_config(cfg)
    if network is None:
        return block, None
    power = _num(p_loss_w)
    network_dict = network.to_dict()
    block.update({
        "topology": network.topology,
        "r_th_k_per_w": list(network.r_th_k_per_w),
        "tau_s": list(network.tau_s),
        "ambient_c": network.ambient_c,
        "p_loss_w": power,
        "thermal_network": network_dict,
    })
    if power is None:
        block.update(
            tj_steady_c=None,
            reason=("no finite computed loss power is available; Tj is N/A "
                    "rather than a fabricated temperature"))
    else:
        block["tj_steady_c"] = network.steady_state_tj(power)
    return block, network_dict


def _converter_summary(converter_result: dict | None) -> dict:
    """Only an explicitly completed converter run may supply an output power."""
    if not isinstance(converter_result, dict) or not converter_result:
        return {
            "status": "not_provided",
            "completed": None,
            "pout_w": None, "pin_w": None,
            "efficiency": None,
            "reason": "no explicit converter result was supplied",
        }
    status = str(converter_result.get("status") or "unknown")
    completed = bool(converter_result.get("completed"))
    pout = _num(converter_result.get("pout_w"))
    pin = _num(converter_result.get("pin_w"))
    efficiency = _num(converter_result.get("efficiency"))
    measured = status == "measured" and completed and pout is not None
    return {
        "status": status,
        "completed": completed,
        "pout_w": pout if measured else None,
        "pin_w": pin if measured else None,
        "efficiency": efficiency if measured else None,
        "vout_v": _num(converter_result.get("vout_v")) if measured else None,
        "iout_a": _num(converter_result.get("iout_a")) if measured else None,
        "iin_a": _num(converter_result.get("iin_a")) if measured else None,
        "f_sw_hz": _num(converter_result.get("f_sw_hz")) if measured else None,
        "guard": converter_result.get("guard"),
        "reason": (None if measured else
                   "converter result is incomplete or not measured; "
                   "efficiency stays output_power_required"),
        "rule": ("only an explicitly completed measured converter run may "
                 "supply p_out for the efficiency gate"),
    }


# ---------------------------------------------------------------------------
# live benches (only entered when explicitly requested)
# ---------------------------------------------------------------------------

def _run_emitted_dpt(lib: Path, sub_name: str, pin_roles: list[str],
                     vendor: str):
    """Simulate the emitted ideal switch in the vendor's DPT bench."""
    dpt = {k: float(v) for k, v in load_input(vendor)["dpt"].items()}
    temp = float(load_input(vendor)["conditions"]["temperature"])
    nb = NetlistBuilder(f"T4_DPT_{vendor}")
    nb.include(lib)
    nb.v_dc("bus", "vbus", "0", dpt["vbus"])
    nb.l("load", "vbus", "drain", dpt["l_load"])
    nb.raw_line("Dclamp drain vbus DCLAMP")
    nb.raw_line(CLAMP_DIODE_MODEL)
    nb.raw_line("Vgs gate 0 PWL(" + _pulse_train(
        dpt["t1_on"], dpt["t1_width"], dpt["t2_width"],
        dpt["t2_on"] - dpt["t1_on"], 2, dpt["edge_time"],
        dpt["vgs_on"]) + ")")
    nb.x_instance("dut", sub_name, pin_roles, temperature=temp)
    started = time.perf_counter()
    result = SpiceEngine(timeout_s=EMITTED_DPT_TIMEOUT_S).run_transient(
        nb.render(), TranDirective(step_time=dpt["max_step"],
                                   end_time=dpt["t_end"],
                                   max_step=dpt["max_step"]))
    wall = time.perf_counter() - started
    return result, wall


def _dpt_guard(result, log: str, dpt: dict, golden_points, wall: float) -> dict:
    """Completion guard for the emitted DPT; aborted runs are never complete."""
    time_trace = result.get("time")
    n = len(time_trace)
    t_last = float(time_trace[-1]) if n else None
    iterations = None
    match = re.search(r"Total iterations:\s*(\d+)", log or "")
    if match:
        iterations = int(match.group(1))
    complete = bool(
        n and t_last is not None
        and t_last >= DPT_T_LAST_FRACTION * dpt["t_end"]
        and iterations and iterations > 0
        and (golden_points is None
             or n >= DPT_TRACE_POINT_FRACTION * golden_points)
    )
    return {
        "executed": True,
        "completed": complete,
        "trace_points": n,
        "golden_trace_points": golden_points,
        "iterations": iterations,
        "t_last_s": t_last,
        "t_end_s": dpt["t_end"],
        "wall_clock_s": wall,
        "error": None if complete else (
            f"incomplete: {n} points, t_last={t_last}, iterations={iterations}"),
        "spice_log_tail": "\n".join((log or "").splitlines()[-6:]),
    }


def _speedup_row(golden: dict, tier4: dict, complete: bool,
                 policy: dict) -> dict:
    """Build the wiring-facing speedup row; never fabricate a ratio."""
    counts = {
        "golden_iterations": _num(golden.get("iterations")),
        "tier4_iterations": _num(tier4.get("iterations")),
        "golden_points": _num(golden.get("trace_points")),
        "tier4_points": _num(tier4.get("trace_points")),
    }
    row = {
        "solver_speedup": None,
        "wall_speedup": None,
        **counts,
        "end_time_s": _num(golden.get("end_time_s")) if complete else None,
        "end_time_s_source": (
            "run_inductive_benchmark bench schedule (.tran stop); not a "
            "measured trace endpoint"),
        "t_last_s": None,
        "t_last_source": (
            "run_inductive_benchmark discards per-run time vectors; no "
            "measured trace endpoint is exposed, so t_last is N/A"),
        "measured_trace_endpoint_available": False,
        "completion_basis": SPEED_COMPLETION_BASIS,
        "status": "measured" if complete else "N/A",
        "max_step_s": policy["golden_max_step"],
        "case": policy["case"],
        "method": SPEED_METHOD,
    }
    if complete:
        gs, ts = _num(golden.get("solver_elapsed_s")), _num(tier4.get("solver_elapsed_s"))
        gw, tw = _num(golden.get("wall_clock_s")), _num(tier4.get("wall_clock_s"))
        if gs and ts:
            row["solver_speedup"] = gs / ts
        if gw and tw:
            row["wall_speedup"] = gw / tw
    return row


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def run_vendor(vendor: str, *, live: bool = True,
               converter_result: dict | None = None) -> dict:
    """Build, emit and validate one Tier-4 vendor.

    ``live=False`` constructs the whole flow from recorded measured data and
    deliberately leaves the bench-derived fields N/A; it never invents a
    Tier-4 timing. The CLI/registry call the default (live=True).

    ``converter_result`` is an optional explicit synchronous-buck measurement
    (see ``simulation/testbenches/converter.py``).  Only a completed measured
    run may supply a useful output power; otherwise the efficiency gate stays
    ``output_power_required``.  ``config['thermal']`` is optional too: absent
    or malformed thermal data leaves Tj N/A rather than inventing ambient.
    """
    cfg = load_input(vendor)
    dpt = {k: float(v) for k, v in cfg["dpt"].items()}
    vbus, i_switch = dpt["vbus"], dpt["i_load_target"]
    temperature = float(cfg["conditions"]["temperature"])
    dpt_record, speed_record = _recorded_artifacts(vendor)

    lut = build_lut(vendor, simulate=live)
    lib = ROOT / "reports" / "models" / f"{vendor}_PIECESPICE_T4.lib"
    lib.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = build_tier4_subcircuit(lut, i_load_a=i_switch,
                                      vgs_on=dpt["vgs_on"])
        emission = {
            "status": "emitted", "reason": None,
            "finite_device_values": True, "ascii": text.isascii(),
        }
        if not text.isascii():
            raise ValueError("Tier-4 emission must be ASCII")
        lib.write_text(text, encoding="ascii")
        emission["emitted_lib_sha256"] = sha256_file(lib)
        sub_name, pin_roles = _parse_emitted_header(text)
    except (ValueError, TypeError) as exc:
        # The emitter refuses non-finite Vth/Ron rather than writing literal
        # NaN into .MODEL. Keep the validation record with an explicit N/A
        # guard; never emit or bench an unsimulatable switch.
        reason = f"{type(exc).__name__}: {exc}"
        emission = {"status": "N/A", "reason": reason,
                    "finite_device_values": False, "ascii": None,
                    "emitted_lib_sha256": None}
        text = ""
        sub_name = f"{vendor}_PIECESPICE_T4"
        pin_roles = ["drain", "gate", "source"]

    energy = _energy_comparison(lut, dpt_record, vbus, i_switch, vendor,
                                temperature_c=temperature)
    policy = _speed_policy(vendor, speed_record)
    recorded_case = speed_record.get("vendors", {}).get(
        vendor, {}).get(policy["case"], {})
    golden_baseline = recorded_case.get("golden", {})
    tier2_baseline = recorded_case.get("rom", {})
    baseline_fields = ("wall_clock_s", "solver_elapsed_s", "iterations",
                       "trace_points", "end_time_s", "max_step")

    dpt_guard = {
        "executed": False, "completed": None, "trace_points": None,
        "golden_trace_points": None, "iterations": None, "t_last_s": None,
        "t_end_s": dpt["t_end"], "wall_clock_s": None, "error": None,
        "reason": ("No Tier-4 DPT bench was executed in this offline "
                   "generation; the emitted lib is built but not simulated."),
        "spice_log_tail": "",
    }
    if emission["status"] != "emitted":
        dpt_guard["reason"] = f"emission N/A: {emission['reason']}; no bench run"
        dpt_guard["error"] = emission["reason"]
    elif live:
        try:
            result, wall = _run_emitted_dpt(lib, sub_name, pin_roles, vendor)
            golden_points = _num(
                speed_record.get("vendors", {}).get(vendor, {})
                .get("short_forced", {}).get("golden", {}).get("trace_points"))
            dpt_guard = _dpt_guard(
                result, getattr(result, "log_text", ""), dpt, golden_points,
                wall)
        except Exception as exc:  # solver failures are reported, never fabricated
            dpt_guard["error"] = f"{type(exc).__name__}: {exc}"[:200]

    speedup = {
        "solver_speedup": None, "wall_speedup": None,
        "golden_iterations": None, "tier4_iterations": None,
        "golden_points": None, "tier4_points": None, "t_last_s": None,
        "t_last_source": (
            "no arm was measured; run_inductive_benchmark discards per-run "
            "time vectors, so no measured trace endpoint is exposed and "
            "t_last is N/A"),
        "end_time_s": None,
        "end_time_s_source": (
            "run_inductive_benchmark bench schedule (.tran stop); not a "
            "measured trace endpoint"),
        "measured_trace_endpoint_available": False,
        "completion_basis": SPEED_COMPLETION_BASIS,
        "status": "N/A",
        "max_step_s": policy["golden_max_step"], "case": policy["case"],
        "method": SPEED_METHOD,
    }
    measured_runtime = {"golden": None, "tier4": None}
    if (live and emission["status"] == "emitted"
            and policy["golden_failure"] is None):
        bench_kw = {
            "vbus": vbus, "vgs_on": dpt["vgs_on"],
            "i_load": i_switch, "l_load": dpt["l_load"],
            "pulse_width": dpt["t2_width"],
            "first_pulse_width": dpt["t1_width"],
            "pulse_period": dpt["t2_on"] - dpt["t1_on"],
            "n_pulses": SPEED_N_PULSES, "edge_time": dpt["edge_time"],
            "repeats": SPEED_REPEATS, "timeout_s": SPEED_TIMEOUT_S,
        }
        scheduled_end_s = _scheduled_bench_end(bench_kw)
        try:
            roles = resolve_pin_connections(
                ROOT / cfg["library"], cfg["subcircuit"],
                cfg["pins"]["pin_order_semantic"])
            golden_bench = run_inductive_benchmark(
                ROOT / cfg["library"], cfg["subcircuit"], roles,
                max_step=policy["golden_max_step"], **bench_kw)
            tier4_bench = run_inductive_benchmark(
                lib, sub_name, pin_roles,
                max_step=policy["golden_max_step"], **bench_kw)
            measured_runtime = {"golden": golden_bench, "tier4": tier4_bench}
            speedup = _speedup_row(
                golden_bench, tier4_bench,
                _bench_complete(golden_bench, tier4_bench,
                                scheduled_end_s=scheduled_end_s), policy)
        except Exception as exc:  # a failed bench is N/A, never a fast run
            speedup["error"] = f"{type(exc).__name__}: {exc}"[:200]
    elif policy["golden_failure"]:
        speedup["error"] = policy["golden_failure"]
    else:
        speedup["error"] = (
            "No Tier-4 adaptive bench was executed in this offline generation; "
            "recorded golden baseline is attached under runtime.recorded_baseline")
    if emission["status"] != "emitted":
        speedup["error"] = f"emission N/A: {emission['reason']}"

    # Vbus*Iload is DC bus INPUT power, never converter output power.  Only an
    # explicitly completed converter run may supply p_out; otherwise the LUT
    # returns N/A with an explicit reason and only the separately computed
    # loss terms are reported.  The DPT pulse repetition frequency is an
    # illustrative offline fallback, NOT a measured converter f_sw.
    converter = _converter_summary(converter_result)
    pulse_period = dpt["t2_on"] - dpt["t1_on"]
    p_bus_in_w = vbus * i_switch
    dpt_f_sw_hz = 1.0 / pulse_period if pulse_period > 0 else None
    converter_f_sw = _num(converter.get("f_sw_hz"))
    if converter_f_sw is not None and converter_f_sw > 0.0:
        f_sw_hz = converter_f_sw
        f_sw_source = "explicit converter result"
    else:
        f_sw_hz = dpt_f_sw_hz
        f_sw_source = "DPT pulse repetition interval (offline diagnostic)"
    try:
        efficiency = lut.efficiency_at(
            vbus, i_switch, p_out_w=converter.get("pout_w"),
            f_sw_hz=f_sw_hz, temperature_c=temperature)
    except (AttributeError, TypeError, ValueError):
        efficiency = None
    r_meta = dict(getattr(lut, "source", {}).get("r_on") or {})
    eff_dict = efficiency if isinstance(efficiency, dict) else {}
    lut_efficiency = _num(eff_dict.get("efficiency"))
    converter_measured = (converter.get("status") == "measured"
                          and converter.get("pout_w") is not None)
    efficiency_gate = {
        "status": str(eff_dict.get("status") or "unavailable"),
        "efficiency": (converter.get("efficiency") if converter_measured
                       else lut_efficiency),
        "efficiency_source": ("converter_bench" if converter_measured
                              else "output_power_required"),
        "lut_estimate": lut_efficiency,
        "reason": (None if converter_measured else
                   str(eff_dict.get("reason") or
                       "no measured converter output power is available")),
        "p_out_w": converter.get("pout_w"),
        "p_bus_in_w": p_bus_in_w,
        "p_bus_in_role": ("config-derived DC bus input power; never used as "
                          "p_out or as converter output"),
        "p_cond_w": _num(eff_dict.get("p_cond_w")),
        "p_switch_w": _num(eff_dict.get("p_switch_w")),
        "p_total_w": _num(eff_dict.get("p_total_w")),
        "f_sw_hz": f_sw_hz,
        "f_sw_source": f_sw_source,
        "f_sw_basis": F_SW_BASIS_NOTE,
        "temperature_c": temperature,
        "thermal_note": THERMAL_LOSS_NOTE,
        "converter": _json_safe(converter),
        "source": OFFICIAL_SOURCES["P-TC"],
    }
    thermal_block, thermal_network = _thermal_block(
        cfg, _num(eff_dict.get("p_total_w")))
    if converter_measured:
        efficiency_gate["status"] = "computed"
    if isinstance(efficiency, dict):
        efficiency["converter"] = _json_safe(converter)

    generation = {
        "mode": "live" if live else "offline_recorded",
        "live_benches_executed": bool(live),
        "note": ("Live generation ran the emitted DPT and, where the golden "
                 "case is not contract-excluded, the adaptive speed bench; "
                 "unavailable or incomplete runs stay N/A."
                 if live else
                 "Offline generation uses recorded measured artifacts and "
                 "leaves every bench-derived field N/A; it never reports a "
                 "Tier-4 timing that was not measured."),
    }

    solver_stabilization = _solver_stabilization(lut, text, emission)
    try:
        ron_selection = resolve_r_on(
            lut, i_load_a=i_switch, vgs_on=dpt["vgs_on"])
    except (TypeError, ValueError) as exc:
        ron_selection = {
            "value_ohms": None, "source": "unavailable",
            "rule": f"{type(exc).__name__}: {exc}",
            "vgs_on_v": dpt["vgs_on"], "vgs_points": [],
        }
    try:
        vt_selection = resolve_vt(lut)
    except (TypeError, ValueError) as exc:
        vt_selection = {
            "value_v": None, "source": "unavailable",
            "rule": f"{type(exc).__name__}: {exc}",
        }
    ron_meta = dict(getattr(lut, "source", {}).get("r_on_by_vgs") or {})
    vt_meta = dict(getattr(lut, "source", {}).get("vt_switch") or {})
    vt_rule = vt_selection.get("rule")
    if (vt_selection.get("source") == "curve7_jedec_vth"
            and isinstance(vt_rule, str)
            and "curve7_jedec_vth" not in vt_rule):
        vt_rule += " (source=curve7_jedec_vth)"

    record = {
        "schema": SCHEMA,
        "vendor": vendor,
        "methods": ["PLECS-style ideal switch (P-SEM)",
                    "measured conduction/switching loss LUT (P-TC)"],
        "waveform_gate": WAVEFORM_GATE,
        "generation": generation,
        "emission": emission,
        "emitted_lib": str(lib.relative_to(ROOT)),
        "subcircuit": sub_name,
        "pin_roles": pin_roles,
        "conditions": {
            "vbus_v": vbus, "i_switch_a": i_switch,
            "vgs_on_v": dpt["vgs_on"], "temperature_c": temperature,
            "t_end_s": dpt["t_end"],
            "golden_i_peak_a": _num(
                dpt_record.get("golden_metrics", {}).get("i_peak_a")),
        },
        "energy_comparison": energy,
        "energy_gate_disclosure": ENERGY_GATE_DISCLOSURE,
        "efficiency": _json_safe(efficiency),
        "efficiency_gate": _json_safe(efficiency_gate),
        "thermal": _json_safe(thermal_block),
        "thermal_network": _json_safe(thermal_network),
        "tj_steady_c": thermal_block.get("tj_steady_c"),
        "converter": (_json_safe(converter)
                      if converter_result is not None else None),
        "dpt_run": dpt_guard,
        "speedup": speedup,
        "runtime": {
            "recorded_baseline": {
                "case": policy["case"],
                "golden": {k: _num(golden_baseline.get(k)) for k in baseline_fields},
                "tier2_rom": {k: _num(tier2_baseline.get(k)) for k in baseline_fields},
                "golden_completed": not bool(golden_baseline.get("error")),
                "source": "reports/comparison/speedup_benchmark.json",
                "note": ("Recorded golden/Tier-2 context only; not a Tier-4 "
                         "timing and never used as a Tier-4 speedup."
                         + (" The older recorded artifact shows a completed "
                            "adaptive golden case; the pinned contract keeps "
                            "this vendor's golden speed N/A, so no ratio is "
                            "derived from it."
                            if vendor in GOLDEN_SPEED_EXCLUSIONS else "")),
            },
            "golden_cap": {
                "max_step_s": policy["golden_max_step"],
                "reason": policy["cap_reason"],
                "source": "reports/comparison/speedup_benchmark.json cases.long_100ns",
            },
            "measured": _json_safe(measured_runtime),
        },
        "lut": _json_safe(lut.to_dict()),
        "solver_stabilization": solver_stabilization,
        "provenance": {
            "config": f"config/inputs/{vendor}.yaml",
            "measured_golden_dpt":
                f"reports/comparison/{vendor}/dpt_metrics.json",
            "recorded_speed_baseline": "reports/comparison/speedup_benchmark.json",
            "lut_source": _json_safe(lut.source),
            "official_sources": OFFICIAL_SOURCES,
            "decisions": {
                "golden_speed_policy": policy["decision"],
                "gate_note": ("Energy/efficiency vs the measured golden DPT "
                              "is the Tier-4 gate; DPT Vds/Id and tr/tf are "
                              "retired by design, not silently failed."),
                "shared_source_note": ("The LUT golden crossing and the golden "
                                       "DPT energy column share the same "
                                       "measured DPT; agreement at that point "
                                       "is by construction, not an independent "
                                       "converter-bench validation (M5 unbuilt)."),
                "efficiency_inputs": {
                    "p_out_w": converter.get("pout_w"),
                    "p_out_source": ("explicit converter result"
                                     if converter_measured else None),
                    "p_conv_in_w": converter.get("pin_w"),
                    "p_bus_in_w": p_bus_in_w,
                    "f_sw_hz": f_sw_hz,
                    "f_sw_source": f_sw_source,
                    "rule": (
                        "Vbus*Iload is DC bus input power, not output power. "
                        "Only a completed measured converter run supplies "
                        "p_out; otherwise efficiency is N/A. "
                        + F_SW_BASIS_NOTE),
                    "kind": ("explicit_converter_output" if converter_measured
                             else "config_derived_input_power_only"),
                    "thermal_note": THERMAL_LOSS_NOTE,
                },
                "run_constants": {
                    "kind": "project_decision",
                    "source": ("pipeline/tier4.py named constants; guard "
                               "fractions reused from pipeline/tier3.py via "
                               "loss_lut_bench, bench knobs pinned by the "
                               "recorded speedup_benchmark.json cases"),
                    "values": {
                        "emitted_dpt_t_last_fraction": DPT_T_LAST_FRACTION,
                        "emitted_dpt_trace_point_fraction": DPT_TRACE_POINT_FRACTION,
                        "emitted_dpt_timeout_s": EMITTED_DPT_TIMEOUT_S,
                        "speed_n_pulses": SPEED_N_PULSES,
                        "speed_repeats": SPEED_REPEATS,
                        "speed_timeout_s": SPEED_TIMEOUT_S,
                        "clamp_diode_model": CLAMP_DIODE_MODEL,
                    },
                },
                "speed_completion": {
                    "kind": "project_decision",
                    "basis": SPEED_COMPLETION_BASIS,
                    "schedule_proof": (
                        "each arm: end_time_s equals the scheduled end "
                        "recomputed from the 20-pulse run_inductive_benchmark "
                        "knobs and repeats == 3 (SPEED_REPEATS)"),
                    "source": "pipeline/tier4.py::_bench_complete",
                },
                "r_on_drive_voltage": {
                    "kind": "measured",
                    "gate_drive_v": r_meta.get("gate_drive_v"),
                    "rule": r_meta.get("gate_drive_rule"),
                    "note": ("Curve-3 Rds(on) was extracted at "
                             "conditions.Vgs_fixed; this is NOT dpt.vgs_on. "
                             "The emitted vgs_on_v comment remains separate "
                             "gate-drive metadata."),
                },
                "r_on_selection": {
                    "kind": "measured",
                    "basis": r_meta.get("kind"),
                    "source_file": r_meta.get("source_file"),
                    "measured_key": r_meta.get("measured_key"),
                    "method": r_meta.get("method"),
                    "selected_ohms": _num(ron_selection.get("value_ohms")),
                    "selection_source": ron_selection.get("source"),
                    "vgs_on_v": ron_selection.get("vgs_on_v"),
                    "vgs_selected_v": ron_selection.get("vgs_selected_v"),
                    "vgs_match": ron_selection.get("vgs_match"),
                    "vgs_points": ron_selection.get("vgs_points"),
                    "table_available": bool(ron_meta.get("vgs_points")),
                    "table_source": ron_meta.get("source"),
                    "rule": ron_selection.get("rule"),
                    "note": (
                        "Measured Rds(on) family from config.r_on_by_vgs is "
                        "selected at dpt.vgs_on when present (exact measured "
                        "gate drive preferred, nearest measured otherwise); "
                        "without it the representative measured current point "
                        "is used. No resistance is invented."),
                },
                "vt_selection": {
                    "kind": vt_selection.get("source"),
                    "value_v": _num(vt_selection.get("value_v")),
                    "source": vt_selection.get("source"),
                    "curve7_vth_v": _num(getattr(lut, "vth_v", None)),
                    "config_vt_switch_v": _num(
                        getattr(lut, "vt_switch_v", None)),
                    "source_meta": _json_safe(vt_meta),
                    "rule": vt_rule,
                    "note": (
                        "A finite config/LUT switching threshold wins; "
                        "otherwise the measured Curve-7 JEDEC Vth is used and "
                        "labelled source=curve7_jedec_vth. NaN is never "
                        "emitted into the .MODEL."),
                },
                "thermal": {
                    "status": thermal_block.get("status"),
                    "tj_steady_c": thermal_block.get("tj_steady_c"),
                    "thermal_network": _json_safe(thermal_network),
                    "source": thermal_block.get("source"),
                    "rule": thermal_block.get("rule"),
                },
                "v_hysteresis": {
                    "kind": "project_decision",
                    "source": (
                        "models/tier4_ideal_switch_builder.py: zero hysteresis "
                        "constant, labeled source=project_decision in every lib"),
                    "rule": (
                        "zero hysteresis so the switch tracks the Curve-7 "
                        "threshold; no hysteresis measurement was extracted"),
                    "note": (
                        "Duplicates the vh_v source label already emitted in "
                        "the lib comment; Vh and the .MODEL are unchanged."),
                },
                "data_limitations": list(
                    getattr(lut, "source", {}).get("data_limitations") or []),
                "retired_observables": {
                    "documented_by_p_demo": [
                        "Miller gate droop", "diode reverse recovery",
                        "finite SPICE turn-on/off transients"],
                    "inference": ["dv/dt", "ringing"],
                    "note": ("[INFERENCE] dv/dt and ringing follow from "
                             "instantaneous switching and absent parasitics; "
                             "they are not documented observables on the "
                             "consulted Plexim pages."),
                },
                "no_tolerance_change": True,
            },
        },
        # Vendor-specific scope disclosure (VM-10); a RECORD-ROOT key so it
        # matches the documented pointer in GOLDEN_SPEED_EXCLUSIONS.
        **({"vendor_scope_notes": VENDOR_SPEED_SCOPE_NOTES[vendor]}
           if vendor in VENDOR_SPEED_SCOPE_NOTES else {}),
        "guards": {
            "ascii_emission": bool(emission["ascii"]),
            "emission_valid": emission["status"] == "emitted",
            "subcircuit_header_parsed": True,
            "dpt_completion": bool(dpt_guard["completed"]),
            "speed_completion": bool(speedup["status"] == "measured"),
            "waveform_gate": WAVEFORM_GATE,
        },
    }
    out = (ROOT / "reports" / "comparison" / vendor /
           "tier4_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(record), indent=2, allow_nan=False)
                   + "\n", encoding="utf-8")
    return record
