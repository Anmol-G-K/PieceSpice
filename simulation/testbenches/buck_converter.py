"""Config-driven buck-converter benchmark (Milestone 5).

The bench simulates a real switched-mode buck power stage around an emitted
PieceSpice device model:

* the high-side DUT is selected from ``tier2``, ``tier3`` (v1), ``tier3_simplis``,
  ``urom`` or ``tier4`` through ``config/converter.yaml`` -> ``model_selection``
  and resolved with :func:`pipeline.registry.lib_path`;
* the freewheel leg is either a complementary low-side instance of the same
  model (synchronous) or a diode (non-synchronous);
* gate drives, inductor (with DCR), output capacitor (with ESR), load and all
  probes are emitted through :class:`simulation.netlist_text.NetlistBuilder`.

No vendor-specific voltage, gate drive or operating point is hardcoded here:
every value comes from ``config/converter.yaml``.  A missing library, an
aborted/truncated run or a missing required trace returns explicit N/A
measurement fields (``None``) with a guard record.  Efficiency is computed
only from measured ``Vout``, ``I(Rload)`` and ``I(Vbus)`` means over a settled
window; it is never inferred from the duty schedule or from ``Vbus*Iload``.
"""

from __future__ import annotations

import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from common.provenance import sha256_file, sha256_text
from simulation.engine import SpiceEngine, TranDirective
from simulation.netlist_text import NetlistBuilder
from simulation.testbenches.converter import extract_efficiency

SCHEMA = "piecespice.converter-bench/1.0"
CONFIG_SCHEMA = "piecespice.converter-bench/1.0"
ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "converter.yaml"

T_LAST_FRACTION = 0.999
MIN_TRACE_POINTS = 3
MIN_WINDOW_POINTS = 3

_TIER_KEYS = ("tier2", "tier3", "tier3_simplis", "urom", "tier4")
_ITERATIONS_RE = re.compile(r"Total iterations:\s*(\d+)")

_ROLE_ALIASES = {
    "drain": {"d", "drain", "drainin", "d1"},
    "gate": {"g", "gate", "gatein", "g1"},
    "source": {"s", "source", "sourcein", "s1"},
    "tj": {"tj", "temp", "temperature"},
}


class ConverterConfigError(ValueError):
    """Malformed or incomplete ``config/converter.yaml`` (fails closed)."""


class ConverterModelError(RuntimeError):
    """Selected tier/vendor library or subcircuit cannot be used."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _finite(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConverterConfigError(f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise ConverterConfigError(f"{name} must be finite")
    return out


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    out = _finite(value, name)
    if out < 0.0 or (out == 0.0 and not allow_zero):
        raise ConverterConfigError(
            f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return out


def _int_value(value: Any, name: str) -> int:
    out = _finite(value, name)
    if not float(out).is_integer():
        raise ConverterConfigError(f"{name} must be an integer")
    return int(out)


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConverterConfigError(f"missing {where}.{key}")
    return mapping[key]


def _validate_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConverterConfigError("converter config must be a mapping")
    if raw.get("schema") != CONFIG_SCHEMA:
        raise ConverterConfigError(
            f"schema must be {CONFIG_SCHEMA!r}, got {raw.get('schema')!r}")

    cond = _require(raw, "conditions", "config")
    if not isinstance(cond, dict):
        raise ConverterConfigError("conditions must be a mapping")
    for key in ("vbus_v", "vout_target_v", "f_sw_hz", "l_h", "c_out_f",
                "r_load_ohm", "edge_time_s"):
        cond[key] = _positive(cond.get(key), f"conditions.{key}")
    cond["duty"] = _positive(cond.get("duty"), "conditions.duty")
    if not 0.0 < float(cond["duty"]) < 1.0:
        raise ConverterConfigError("conditions.duty must be between 0 and 1")
    cond["dead_time_s"] = _positive(
        cond.get("dead_time_s"), "conditions.dead_time_s", allow_zero=True)
    cond["inductor_dcr_ohm"] = _positive(
        cond.get("inductor_dcr_ohm"), "conditions.inductor_dcr_ohm",
        allow_zero=True)

    filt = _require(cond, "output_filter", "conditions")
    if not isinstance(filt, dict):
        raise ConverterConfigError("conditions.output_filter must be a mapping")
    filt["esr_ohm"] = _positive(
        filt.get("esr_ohm"), "conditions.output_filter.esr_ohm",
        allow_zero=True)

    sim = _require(cond, "simulation", "conditions")
    if not isinstance(sim, dict):
        raise ConverterConfigError("conditions.simulation must be a mapping")
    sim["cycles"] = _int_value(sim.get("cycles"), "conditions.simulation.cycles")
    sim["settling_cycles"] = _int_value(
        sim.get("settling_cycles"), "conditions.simulation.settling_cycles")
    if sim["cycles"] < 3:
        raise ConverterConfigError("conditions.simulation.cycles must be >= 3")
    if not 0 <= sim["settling_cycles"] <= sim["cycles"] - 2:
        raise ConverterConfigError(
            "conditions.simulation.settling_cycles must leave at least two "
            "measured periods (0 <= settling_cycles <= cycles - 2)")
    sim["max_step_fraction_of_period"] = _positive(
        sim.get("max_step_fraction_of_period"),
        "conditions.simulation.max_step_fraction_of_period")
    if sim["max_step_fraction_of_period"] > 1.0:
        raise ConverterConfigError(
            "conditions.simulation.max_step_fraction_of_period must be <= 1")
    sim["timeout_s"] = _positive(
        sim.get("timeout_s"), "conditions.simulation.timeout_s")
    sim["temperature_c"] = _finite(
        sim.get("temperature_c"), "conditions.simulation.temperature_c")
    sim["settling_rel_tolerance_pct"] = _positive(
        sim.get("settling_rel_tolerance_pct"),
        "conditions.simulation.settling_rel_tolerance_pct")

    fw = _require(raw, "freewheel", "config")
    if not isinstance(fw, dict):
        raise ConverterConfigError("freewheel must be a mapping")
    if fw.get("mode") not in ("synchronous", "diode"):
        raise ConverterConfigError(
            "freewheel.mode must be 'synchronous' or 'diode'")
    diode = _require(fw, "diode", "freewheel")
    if not isinstance(diode, dict):
        raise ConverterConfigError("freewheel.diode must be a mapping")
    diode["saturation_current_a"] = _positive(
        diode.get("saturation_current_a"), "freewheel.diode.saturation_current_a")
    diode["emission_coefficient"] = _positive(
        diode.get("emission_coefficient"), "freewheel.diode.emission_coefficient")
    diode["series_resistance_ohm"] = _positive(
        diode.get("series_resistance_ohm"),
        "freewheel.diode.series_resistance_ohm", allow_zero=True)
    diode["junction_capacitance_f"] = _positive(
        diode.get("junction_capacitance_f"),
        "freewheel.diode.junction_capacitance_f", allow_zero=True)

    selection = _require(raw, "model_selection", "config")
    if not isinstance(selection, dict):
        raise ConverterConfigError("model_selection must be a mapping")
    tiers = _require(selection, "tiers", "model_selection")
    if not isinstance(tiers, dict):
        raise ConverterConfigError("model_selection.tiers must be a mapping")
    for key in _TIER_KEYS:
        entry = _require(tiers, key, "model_selection.tiers")
        if not isinstance(entry, dict):
            raise ConverterConfigError(
                f"model_selection.tiers.{key} must be a mapping")
        tier = _int_value(
            entry.get("tier"), f"model_selection.tiers.{key}.tier")
        if tier not in (2, 3, 4):
            raise ConverterConfigError(
                f"model_selection.tiers.{key}.tier must be 2, 3 or 4")
        variant = entry.get("variant")
        if not isinstance(variant, str) or not variant.strip():
            raise ConverterConfigError(
                f"model_selection.tiers.{key}.variant must be a non-empty string")
        entry["tier"] = tier
        entry["variant"] = variant.strip()

    default = _require(selection, "default", "model_selection")
    if not isinstance(default, dict):
        raise ConverterConfigError("model_selection.default must be a mapping")
    default_vendor = _require(default, "vendor", "model_selection.default")
    default_tier = _require(default, "tier", "model_selection.default")
    if default_vendor not in raw.get("vendors", {}):
        raise ConverterConfigError(
            f"model_selection.default.vendor {default_vendor!r} has no "
            "vendors entry")
    if default_tier not in tiers:
        raise ConverterConfigError(
            f"model_selection.default.tier {default_tier!r} is not in tiers")

    vendors = _require(raw, "vendors", "config")
    if not isinstance(vendors, dict):
        raise ConverterConfigError("vendors must be a mapping")
    from pipeline.registry import VENDORS
    missing = [vendor for vendor in VENDORS if vendor not in vendors]
    if missing:
        raise ConverterConfigError(
            "vendors is missing entries for: " + ", ".join(missing))
    for vendor, entry in vendors.items():
        if not isinstance(entry, dict):
            raise ConverterConfigError(f"vendors.{vendor} must be a mapping")
        entry["vgs_on_v"] = _positive(
            entry.get("vgs_on_v"), f"vendors.{vendor}.vgs_on_v")
        for key in ("vbus_v", "vout_target_v", "f_sw_hz", "duty", "l_h",
                    "c_out_f", "r_load_ohm", "dead_time_s", "edge_time_s"):
            if key in entry:
                entry[key] = _positive(entry[key], f"vendors.{vendor}.{key}")
        if "duty" in entry and not 0.0 < float(entry["duty"]) < 1.0:
            raise ConverterConfigError(
                f"vendors.{vendor}.duty must be between 0 and 1")
    return raw


def load_converter_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate ``config/converter.yaml`` (fail closed)."""
    target = Path(path) if path is not None else CONFIG_PATH
    if not target.is_file():
        raise ConverterConfigError(f"converter config not found: {target}")
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConverterConfigError(f"invalid YAML in {target}: {exc}") from exc
    return _validate_config(raw)


def _operating_point(config: Mapping[str, Any],
                     vendor: str) -> dict[str, Any]:
    """Merge the conditions block with a vendor's optional overrides."""
    vendors = config["vendors"]
    if vendor not in vendors:
        raise ConverterConfigError(
            f"vendor {vendor!r} is not configured in config/converter.yaml")
    cond = config["conditions"]
    vcfg = vendors[vendor]
    op = {
        "vbus_v": float(cond["vbus_v"]),
        "vout_target_v": float(cond["vout_target_v"]),
        "f_sw_hz": float(cond["f_sw_hz"]),
        "duty": float(cond["duty"]),
        "l_h": float(cond["l_h"]),
        "inductor_dcr_ohm": float(cond["inductor_dcr_ohm"]),
        "c_out_f": float(cond["c_out_f"]),
        "r_load_ohm": float(cond["r_load_ohm"]),
        "dead_time_s": float(cond["dead_time_s"]),
        "edge_time_s": float(cond["edge_time_s"]),
        "esr_ohm": float(cond["output_filter"]["esr_ohm"]),
        "vgs_on_v": float(vcfg["vgs_on_v"]),
        "cycles": int(cond["simulation"]["cycles"]),
        "settling_cycles": int(cond["simulation"]["settling_cycles"]),
        "max_step_fraction_of_period": float(
            cond["simulation"]["max_step_fraction_of_period"]),
        "timeout_s": float(cond["simulation"]["timeout_s"]),
        "temperature_c": float(cond["simulation"]["temperature_c"]),
        "settling_rel_tolerance_pct": float(
            cond["simulation"]["settling_rel_tolerance_pct"]),
    }
    for key in ("vbus_v", "vout_target_v", "f_sw_hz", "duty", "l_h",
                "c_out_f", "r_load_ohm", "dead_time_s", "edge_time_s"):
        if key in vcfg:
            op[key] = float(vcfg[key])
    if not 0.0 < op["duty"] < 1.0:
        raise ConverterConfigError(f"merged duty is invalid: {op['duty']}")
    return op


# ---------------------------------------------------------------------------
# Emitted model resolution
# ---------------------------------------------------------------------------

def _role_for_port(port: str) -> str | None:
    token = str(port).strip().lower()
    for role, aliases in _ROLE_ALIASES.items():
        if token in aliases:
            return role
    return None


def _parse_model_header(library: Path, subcircuit_name: str | None
                        ) -> tuple[str, list[str]]:
    from parser.netlist import NetlistParser
    try:
        subcircuits = NetlistParser.parse_subcircuits(library)
    except Exception as exc:  # parser failure is a model failure, not a run
        raise ConverterModelError(
            f"cannot parse subcircuits in {library.name}: {exc}") from exc
    if not subcircuits:
        raise ConverterModelError(f"no .SUBCKT found in {library.name}")
    if subcircuit_name is None:
        info = subcircuits[0]
    else:
        info = next((s for s in subcircuits
                     if s.name.lower() == subcircuit_name.lower()), None)
        if info is None:
            raise ConverterModelError(
                f"subcircuit {subcircuit_name!r} not in {library.name}; "
                f"available: {[s.name for s in subcircuits]}")
    roles: list[str] = []
    for port in info.nodes:
        role = _role_for_port(port)
        if role is None:
            raise ConverterModelError(
                f"cannot map port {port!r} of {info.name!r} to a DUT role; "
                "expected drain/gate/source (optional tj)")
        if role in roles:
            raise ConverterModelError(
                f"duplicate {role!r} port in {info.name!r} header")
        roles.append(role)
    missing = {"drain", "gate", "source"} - set(roles)
    if missing:
        raise ConverterModelError(
            f"{info.name!r} header lacks required role(s): "
            + ", ".join(sorted(missing)))
    return info.name, roles


def resolve_converter_model(
    config: Mapping[str, Any],
    vendor: str,
    tier_key: str,
    *,
    library_path: str | Path | None = None,
    subcircuit_name: str | None = None,
) -> dict[str, Any]:
    """Resolve vendor/tier selection to a usable emitted library and ports.

    ``library_path``/``subcircuit_name`` are explicit overrides for offline
    tests and custom artifacts; otherwise the registry filename for the
    configured tier/variant is used.
    """
    tiers = config["model_selection"]["tiers"]
    if tier_key not in tiers:
        raise ConverterConfigError(
            f"unknown tier key {tier_key!r}; configured: {sorted(tiers)}")
    entry = tiers[tier_key]
    tier = int(entry["tier"])
    variant = str(entry["variant"])

    if library_path is not None:
        library = Path(library_path)
    else:
        from pipeline.registry import lib_path
        library = Path(lib_path(vendor, tier, variant))

    if not library.is_file():
        raise ConverterModelError(
            f"emitted library not found for {vendor}/{tier_key}: {library}")

    name, roles = _parse_model_header(library, subcircuit_name)
    op = _operating_point(config, vendor)
    return {
        "vendor": vendor,
        "tier_key": tier_key,
        "tier": tier,
        "variant": variant,
        "library": str(library),
        "library_sha256": sha256_file(library),
        "subcircuit": name,
        "pin_roles": roles,
        "gate_drive_v": op["vgs_on_v"],
        "operating_point": op,
    }


# ---------------------------------------------------------------------------
# Netlist emission
# ---------------------------------------------------------------------------

def _x_line(reference: str, subcircuit: str, roles: Sequence[str],
            nodes: Mapping[str, str]) -> str:
    conn: list[str] = []
    for role in roles:
        key = str(role).strip().lower()
        if key not in nodes:
            raise ConverterModelError(
                f"no external node mapped for DUT role {role!r}")
        conn.append(nodes[key])
    return f"X{reference} {' '.join(conn)} {subcircuit}"


def _pulse_widths(period: float, duty: float, dead_time: float,
                  edge_time: float, *, synchronous: bool
                  ) -> tuple[float, float | None]:
    high_width = duty * period - 2.0 * edge_time
    if high_width <= 0.0:
        raise ValueError(
            f"duty/edge time leave no high-side pulse width ({high_width:g})")
    if not synchronous:
        return high_width, None
    low_width = (1.0 - duty) * period - 2.0 * dead_time - 2.0 * edge_time
    if low_width <= 0.0:
        raise ValueError(
            f"duty/dead/edge times leave no low-side pulse width "
            f"({low_width:g})")
    return high_width, low_width


def build_converter_netlist(
    model: Mapping[str, Any],
    *,
    freewheel_mode: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Emit the complete buck netlist and its schedule record.

    ``model`` is the dict returned by :func:`resolve_converter_model`.
    """
    cfg = config if config is not None else load_converter_config()
    op = dict(model["operating_point"])
    mode = freewheel_mode or cfg["freewheel"]["mode"]
    if mode not in ("synchronous", "diode"):
        raise ValueError("freewheel mode must be 'synchronous' or 'diode'")

    period = 1.0 / op["f_sw_hz"]
    end = op["cycles"] * period
    window_start = op["settling_cycles"] * period
    dead = op["dead_time_s"]
    edge = op["edge_time_s"]
    high_width, low_width = _pulse_widths(
        period, op["duty"], dead, edge, synchronous=(mode == "synchronous"))
    low_delay = op["duty"] * period + dead

    roles = list(model["pin_roles"])
    subcircuit = str(model["subcircuit"])

    nb = NetlistBuilder(f"BUCK_{model['vendor']}_{subcircuit}")
    nb.include(model["library"])
    nb.v_dc("bus", "bus", "0", op["vbus_v"])
    # Series 0 V source: I(Vhssense) is the high-side branch current.
    nb.v_dc("hssense", "bus", "vin", 0.0)
    if "tj" in roles:
        nb.v_dc("tj", "tj", "0", op["temperature_c"])
    nb.raw_line(_x_line("hs", subcircuit, roles, {
        "drain": "vin", "gate": "ghs", "source": "sw", "tj": "tj"}))
    nb.v_pulse("ghs", "ghs", "sw", v1=0.0, v2=op["vgs_on_v"], td=0.0,
               tr=edge, tf=edge, pw=high_width, period=period)

    # Freewheel/body-diode leg. It is the only return path in non-synchronous
    # mode and the dead-time/negative-current path in synchronous mode (the
    # emitted device models do not all contain an active body diode).
    diode = cfg["freewheel"]["diode"]
    nb.raw_line("Dfw 0 sw DFW")
    nb.raw_line(
        ".model DFW D(IS={IS:g} N={N:g} RS={RS:g} CJO={CJO:g})".format(
            IS=float(diode["saturation_current_a"]),
            N=float(diode["emission_coefficient"]),
            RS=float(diode["series_resistance_ohm"]),
            CJO=float(diode["junction_capacitance_f"])))

    if mode == "synchronous":
        # Series 0 V source between the switch node and the low-side drain.
        nb.v_dc("lssense", "sw", "sw_ls", 0.0)
        nb.raw_line(_x_line("ls", subcircuit, roles, {
            "drain": "sw_ls", "gate": "gls", "source": "0", "tj": "tj"}))
        assert low_width is not None
        nb.v_pulse("gls", "gls", "0", v1=0.0, v2=op["vgs_on_v"],
                   td=low_delay, tr=edge, tf=edge, pw=low_width,
                   period=period)

    dcr = op["inductor_dcr_ohm"]
    if dcr > 0.0:
        nb.l("out", "sw", "lx", op["l_h"])
        nb.r("ldcr", "lx", "vout", dcr)
    else:
        nb.l("out", "sw", "vout", op["l_h"])

    esr = op["esr_ohm"]
    if esr > 0.0:
        nb.c("out", "vout_c", "0", op["c_out_f"])
        nb.r("esr", "vout", "vout_c", esr)
    else:
        nb.c("out", "vout", "0", op["c_out_f"])

    nb.r("load", "vout", "0", op["r_load_ohm"])
    schedule = {
        "period_s": period,
        "t_end_s": end,
        "window_start_s": window_start,
        "window_end_s": end,
        "measured_periods": op["cycles"] - op["settling_cycles"],
        "dead_time_s": dead,
        "edge_time_s": edge,
        "high_pulse_width_s": high_width,
        "low_pulse_width_s": low_width,
        "low_pulse_delay_s": low_delay if mode == "synchronous" else None,
        "freewheel_mode": mode,
        "temperature_c": op["temperature_c"],
        "rule": ("duty/cycles/edge/dead-time schedule; measurement window "
                 "starts after the configured settling cycles"),
    }
    return nb.render(), schedule


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------

def _trace(result: Any, name: str) -> np.ndarray | None:
    try:
        values = np.asarray(result.get(name), dtype=float)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return values if values.ndim == 1 else None


def _iterations(log: str | None) -> int | None:
    match = _ITERATIONS_RE.search(log or "")
    return int(match.group(1)) if match else None


def _integral(y: np.ndarray, t: np.ndarray) -> float:
    if y.size < 2:
        return float("nan")
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(t)))


def _weighted_mean(values: np.ndarray, time: np.ndarray,
                   mask: np.ndarray) -> float | None:
    """Time-weighted mean over *mask* (LTspice samples are non-uniform).

    A point average over an adaptive .raw trace over-weights switching
    instants; the trapezoidal integral divides by elapsed time instead.
    """
    selected = values[mask]
    times = time[mask]
    finite = np.isfinite(selected) & np.isfinite(times)
    selected = selected[finite]
    times = times[finite]
    if selected.size < 2:
        return None
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        return None
    return _integral(selected, times) / duration


def _base_result(model: Mapping[str, Any] | None, vendor: str,
                 tier_key: str, conditions: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "invalid_input",
        "completed": False,
        "settled": False,
        "vendor": vendor,
        "tier_key": tier_key,
        "tier": None if model is None else model.get("tier"),
        "variant": None if model is None else model.get("variant"),
        "model": None if model is None else {
            "library": model.get("library"),
            "library_sha256": model.get("library_sha256"),
            "subcircuit": model.get("subcircuit"),
            "pin_roles": list(model.get("pin_roles") or []),
            "gate_drive_v": model.get("gate_drive_v"),
        },
        "conditions": dict(conditions),
        "schedule": None,
        "netlist_sha256": None,
        "library": {
            "path": None if model is None else model.get("library"),
            "available": model is not None,
            "sha256": None if model is None else model.get("library_sha256"),
        },
        "guard": {"executed": False, "completed": False},
        "settling": None,
        "extraction": None,
        "loss": None,
        "speed": None,
        "traces": None,
        "vout_v": None,
        "iout_a": None,
        "iin_a": None,
        "vbus_v": None if conditions is None else conditions.get("vbus_v"),
        "vout_target_v": None if conditions is None
        else conditions.get("vout_target_v"),
        "vout_error_pct": None,
        "i_l_mean_a": None,
        "i_l_ripple_pp_a": None,
        "i_cap_mean_a": None,
        "v_sw_mean_v": None,
        "v_sw_min_v": None,
        "v_sw_max_v": None,
        "pout_w": None,
        "pin_w": None,
        "efficiency": None,
        "measurement_status": None,
        "reason": None,
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_converter_benchmark(
    vendor: str | None = None,
    tier_key: str | None = None,
    *,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    library_path: str | Path | None = None,
    subcircuit_name: str | None = None,
    freewheel_mode: str | None = None,
    runner: Callable[[str, TranDirective], Any] | None = None,
    resolved_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a multi-cycle buck benchmark and extract measured performance.

    ``resolved_model`` bypasses registry/tier resolution when the caller has
    already resolved an artifact (the golden vendor model in the comparison
    path). ``runner`` is an optional ``(netlist_text, TranDirective) -> SimResult``
    hook for offline tests; the default executes the real ``SpiceEngine``.
    Missing libraries, aborted runs and absent required traces return N/A
    measurement fields with a guard reason -- never a fabricated efficiency.
    """
    cfg = dict(config) if config is not None else load_converter_config(
        config_path)
    selection = cfg["model_selection"]
    vendor = vendor or selection["default"]["vendor"]
    tier_key = tier_key or selection["default"]["tier"]
    if not isinstance(vendor, str) or not isinstance(tier_key, str):
        raise ConverterConfigError("vendor and tier_key must be strings")
    op = _operating_point(cfg, vendor)

    base = _base_result(None, vendor, tier_key, op)
    if resolved_model is not None:
        model = dict(resolved_model)
        model.setdefault("operating_point", op)
        model.setdefault("vendor", vendor)
        model.setdefault("tier_key", tier_key)
    else:
        try:
            model = resolve_converter_model(
                cfg, vendor, tier_key, library_path=library_path,
                subcircuit_name=subcircuit_name)
        except ConverterModelError as exc:
            base.update(status="model_unavailable", reason=str(exc))
            base["library"]["path"] = (
                None if library_path is None else str(library_path))
            return base
        except ConverterConfigError as exc:
            base.update(status="model_unavailable", reason=str(exc))
            return base

    base = _base_result(model, vendor, tier_key, op)
    base["library"] = {
        "path": model["library"], "available": True,
        "sha256": model["library_sha256"]}

    try:
        netlist, schedule = build_converter_netlist(
            model, freewheel_mode=freewheel_mode, config=cfg)
    except (ValueError, ConverterModelError) as exc:
        base.update(status="invalid_input", reason=f"{type(exc).__name__}: {exc}")
        return base

    base["netlist_sha256"] = sha256_text(netlist)
    base["schedule"] = schedule
    step_time = schedule["period_s"] * op["max_step_fraction_of_period"]
    directive = TranDirective(step_time=step_time,
                              end_time=schedule["t_end_s"],
                              max_step=step_time)

    started = time.perf_counter()
    try:
        if runner is None:
            result = SpiceEngine(
                temperature=op["temperature_c"],
                timeout_s=op["timeout_s"],
                keep_files=False).run_transient(netlist, directive)
        else:
            result = runner(netlist, directive)
    except Exception as exc:  # solver/backend failure is N/A, never faked
        base.update(status="error",
                    reason=f"{type(exc).__name__}: {exc}"[:300])
        base["guard"] = {"executed": True, "completed": False,
                         "wall_clock_s": time.perf_counter() - started}
        return base
    wall = time.perf_counter() - started

    time_trace = _trace(result, "time")
    vout_trace = _trace(result, "V(vout)")
    vsw_trace = _trace(result, "V(sw)")
    il_trace = _trace(result, "I(Lout)")
    iin_trace = _trace(result, "I(Vbus)")
    iout_trace = _trace(result, "I(Rload)")
    log = getattr(result, "log_text", "") or ""
    iterations = _iterations(log)
    end = schedule["t_end_s"]
    points = 0 if time_trace is None else int(time_trace.size)
    t_last = (None if time_trace is None or time_trace.size == 0
              else float(time_trace[-1]))
    guard: dict[str, Any] = {
        "executed": True,
        "completed": False,
        "points": points,
        "trace_points": points,
        "t_last_s": t_last,
        "t_last": t_last,
        "t_end_s": end,
        "iterations": iterations,
        "wall_clock_s": wall,
        "rule": (f"t_last >= {T_LAST_FRACTION:g}*t_end, iterations > 0, "
                 f"trace_points >= {MIN_TRACE_POINTS}, and every required "
                 "trace present with matching lengths"),
        "reason": None,
    }

    required = (
        ("time", time_trace), ("V(vout)", vout_trace),
        ("V(sw)", vsw_trace), ("I(Lout)", il_trace),
        ("I(Vbus)", iin_trace), ("I(Rload)", iout_trace))
    missing = [name for name, trace in required if trace is None]
    if missing:
        guard["reason"] = "missing required trace(s): " + ", ".join(missing)
        base.update(status="missing_trace", guard=guard,
                    reason=guard["reason"])
        return base
    assert all(trace is not None for _, trace in required)
    # Explicit local narrowing: Pyright does not propagate the generator
    # assertion above to individual optional traces.
    assert time_trace is not None
    assert vout_trace is not None
    assert vsw_trace is not None
    assert il_trace is not None
    assert iin_trace is not None
    assert iout_trace is not None
    lengths_ok = all(
        trace is not None and trace.size == time_trace.size
        for _, trace in required)
    finite_time = bool(np.all(np.isfinite(time_trace)))
    complete = bool(
        t_last is not None
        and t_last >= T_LAST_FRACTION * end
        and iterations is not None and iterations > 0
        and points >= MIN_TRACE_POINTS
        and lengths_ok
        and finite_time)
    guard["completed"] = complete
    if not complete:
        guard["reason"] = (
            "incomplete converter run: "
            f"t_last={t_last}, iterations={iterations}, points={points}, "
            f"lengths_ok={lengths_ok}, finite_time={finite_time}, "
            f"required t_end={end}")
        base.update(status="incomplete", guard=guard,
                    reason=guard["reason"])
        return base

    # settled measurement window: [settling_cycles * period, t_last]
    base["completed"] = True
    window_start = schedule["window_start_s"]
    mask = (time_trace >= window_start) & np.isfinite(time_trace)
    window_points = int(mask.sum())
    if window_points < MIN_WINDOW_POINTS:
        guard["completed"] = False
        guard["reason"] = "no complete measurement window inside the trace"
        base.update(status="incomplete", completed=False, guard=guard,
                    reason=guard["reason"])
        return base
    t_win = time_trace[mask]
    duration = float(t_win[-1] - t_win[0])
    if duration <= 0.0:
        base.update(status="incomplete", reason="zero-length measurement window")
        return base

    vout = _weighted_mean(vout_trace, time_trace, mask)
    iin_raw = _weighted_mean(iin_trace, time_trace, mask)
    iout = _weighted_mean(iout_trace, time_trace, mask)
    il_mean = _weighted_mean(il_trace, time_trace, mask)
    vsw_mean = _weighted_mean(vsw_trace, time_trace, mask)
    if None in (vout, iin_raw, iout, il_mean, vsw_mean):
        base.update(status="invalid_measurement",
                    reason="one or more measured means are non-finite")
        return base
    assert vout is not None and iin_raw is not None and iout is not None
    assert il_mean is not None and vsw_mean is not None

    # LTspice reports a delivering V-source current as negative.
    iin = -iin_raw if iin_raw < 0.0 else iin_raw
    measured = extract_efficiency(vout, abs(iout), iin, op["vbus_v"])
    base["measurement_status"] = measured["status"]
    base.update({
        "vout_v": measured["vout_v"],
        "iout_a": measured["iout_a"],
        "iin_a": measured["iin_a"],
        "pout_w": measured["pout_w"],
        "pin_w": measured["pin_w"],
        "efficiency": measured["efficiency"],
    })

    # Settling diagnostic: first-half vs second-half Vout drift.
    mid_t = float(t_win[0] + 0.5 * duration)
    first_mask = mask & (time_trace < mid_t)
    second_mask = mask & (time_trace >= mid_t)
    vout_first = _weighted_mean(vout_trace, time_trace, first_mask)
    vout_second = _weighted_mean(vout_trace, time_trace, second_mask)
    tolerance = op["settling_rel_tolerance_pct"]
    if vout_first is None or vout_second is None:
        settled = False
        drift = None
        settling_reason = "no points in one half of the measurement window"
    else:
        drift = abs(vout_second - vout_first) / max(abs(vout_second), 1e-12) \
            * 100.0
        settled = bool(drift <= tolerance)
        settling_reason = None
    base["settled"] = settled
    base["settling"] = {
        "settled": settled,
        "rel_drift_pct": drift,
        "tolerance_pct": tolerance,
        "first_half_vout_v": vout_first,
        "second_half_vout_v": vout_second,
        "window_s": [float(t_win[0]), float(t_win[-1])],
        "points": window_points,
        "rule": ("first-half vs second-half Vout mean drift over the "
                 "post-settling window; startup cycles are excluded"),
        "reason": settling_reason,
    }

    vout_target = op["vout_target_v"]
    base["vout_target_v"] = vout_target
    base["vout_error_pct"] = ((vout - vout_target) / vout_target * 100.0
                              if vout_target > 0.0 else None)
    base["i_l_mean_a"] = il_mean
    il_win = il_trace[mask]
    base["i_l_ripple_pp_a"] = float(np.nanmax(il_win) - np.nanmin(il_win))
    icap_trace = _trace(result, "I(Cout)")
    if icap_trace is None:
        icap_trace = _trace(result, "I(Resr)")
    base["i_cap_mean_a"] = (
        _weighted_mean(icap_trace, time_trace, mask)
        if icap_trace is not None and icap_trace.size == time_trace.size
        else None)
    base["v_sw_mean_v"] = vsw_mean
    vsw_win = vsw_trace[mask]
    base["v_sw_min_v"] = float(np.nanmin(vsw_win))
    base["v_sw_max_v"] = float(np.nanmax(vsw_win))

    base["speed"] = {
        "wall_clock_s": wall,
        "iterations": iterations,
        "trace_points": points,
        "time_steps": points,
        "step_time_s": step_time,
    }
    base["guard"] = guard
    base["extraction"] = {
        "window_s": [float(t_win[0]), float(t_win[-1])],
        "points": window_points,
        "duration_s": duration,
        "iout_source": "I(Rload) mean over the settled window",
        "iin_source": "I(Vbus) mean over the settled window",
        "iin_raw_mean_a": iin_raw,
        "iin_sign_rule": ("LTspice V-source current is negative when "
                          "delivering bus power; iin = -mean when the raw "
                          "mean is negative"),
        "vout_source": "V(vout) mean over the settled window",
        "rule": measured["rule"],
    }

    if measured["status"] != "computed":
        base.update(status="invalid_measurement",
                    reason=measured["reason"])
        return base

    efficiency = measured["efficiency"]
    assert efficiency is not None
    if efficiency > 1.0 + 1.0e-3:
        base.update(status="invalid_measurement",
                    efficiency=None,
                    reason=("measured efficiency exceeds 1.0 by more than "
                            "the numerical tolerance; efficiency is N/A"))
        return base

    base["loss"] = _loss_breakdown(
        result, schedule, op, mask, duration, measured, il_trace)

    base["status"] = "measured" if settled else "not_settled"
    base["completed"] = True
    base["reason"] = settling_reason
    base["traces"] = {
        "time": time_trace,
        "vout_v": vout_trace,
        "i_l_a": il_trace,
        "v_sw_v": vsw_trace,
        "i_in_a": iin_trace,
        "i_out_a": iout_trace,
        "v_ghs_v": _trace(result, "V(ghs)"),
        "v_gls_v": _trace(result, "V(gls)"),
        "i_cap_a": _trace(result, "I(Cout)"),
    }
    return base


def _loss_breakdown(
    result: Any,
    schedule: Mapping[str, Any],
    op: Mapping[str, Any],
    mask: np.ndarray,
    duration: float,
    measured: Mapping[str, Any],
    il_trace: np.ndarray,
) -> dict[str, Any]:
    """Measured conduction terms plus an explicit switching residual.

    Conduction is integrated from the switch-node voltage and the inductor
    current during gate-on intervals (the inductor carries the switch current
    on each on-interval).  DCR/ESR terms use the measured branch RMS currents.
    The residual ``Pin - Pout - known conduction`` is labelled as a residual,
    not as a pure switching number: it also contains gate-drive and any
    unmodelled loss.
    """
    time_trace = _trace(result, "time")
    vsw_trace = _trace(result, "V(sw)")
    vin_trace = _trace(result, "V(vin)")
    vghs = _trace(result, "V(ghs)")
    vgls = _trace(result, "V(gls)")
    icap = _trace(result, "I(Cout)")
    if icap is None:
        icap = _trace(result, "I(Resr)")
    out: dict[str, Any] = {
        "total_loss_w": None,
        "high_side_conduction_w": None,
        "low_side_conduction_w": None,
        "freewheel_conduction_w": None,
        "inductor_dcr_w": None,
        "output_esr_w": None,
        "switching_residual_w": None,
        "switching_residual_status": "not_available",
        "reason": None,
        "rule": ("conduction from integral(v_sw*i_L) over gate-on intervals; "
                 "DCR/ESR from measured RMS currents; residual = Pin - Pout - "
                 "known conduction (includes gate drive/unmodelled loss)"),
    }
    if time_trace is None or vsw_trace is None or vghs is None:
        out["reason"] = "switch-node or gate-drive trace absent; losses N/A"
        return out
    if not (time_trace.size == vsw_trace.size == vghs.size == il_trace.size):
        out["reason"] = "loss traces have inconsistent lengths; losses N/A"
        return out
    t = time_trace[mask]
    vsw = vsw_trace[mask]
    il = il_trace[mask]
    # The high-side gate source is referenced to the switch node, so its
    # gate-source voltage is V(ghs)-V(sw); the low-side source is 0.
    vgs_hs = (vghs - vsw_trace)[mask]
    if t.size < MIN_WINDOW_POINTS:
        out["reason"] = "too few points for loss integration; losses N/A"
        return out
    vin = op["vbus_v"] if vin_trace is None else _weighted_mean(
        vin_trace, time_trace, mask)
    if vin is None:
        out["reason"] = "bus-voltage trace has no finite window; losses N/A"
        return out
    vgs_on = op["vgs_on_v"]
    hs_on = vgs_hs > 0.5 * vgs_on
    ls_on = (vgls[mask] > 0.5 * vgs_on
             if vgls is not None and vgls.size == time_trace.size
             else np.zeros_like(hs_on))

    def conduction(vds: np.ndarray, on: np.ndarray) -> float:
        # Resistive conduction dissipates for either current direction, so the
        # instantaneous product is taken in magnitude. Low-side Vds is
        # negative in the usual source-to-drain conduction direction.
        inst = np.abs(vds * il) * on
        return _integral(inst, t) / duration

    def mean_square(values: np.ndarray) -> float | None:
        return _weighted_mean(values * values, time_trace, mask)

    p_hs = conduction(vin - vsw, hs_on)

    p_ls: float | None
    if schedule["freewheel_mode"] == "synchronous":
        if vgls is None or vgls.size != time_trace.size:
            p_ls = None
            out["reason"] = "low-side gate trace absent; loss breakdown N/A"
        else:
            p_ls = conduction(vsw, ls_on)
    else:
        p_ls = 0.0

    off = ~hs_on & ~ls_on
    p_fw = conduction(-vsw, off)

    il_sq = mean_square(il_trace)
    p_dcr = (None if il_sq is None
             else il_sq * op["inductor_dcr_ohm"])
    p_esr: float | None
    if op["esr_ohm"] == 0.0:
        p_esr = 0.0
    elif icap is None or icap.size != time_trace.size:
        p_esr = None
        out["reason"] = "capacitor-current trace absent; ESR loss N/A"
    else:
        icap_sq = mean_square(icap)
        p_esr = (None if icap_sq is None
                 else icap_sq * op["esr_ohm"])

    pin = measured.get("pin_w")
    pout = measured.get("pout_w")
    total = None if pin is None or pout is None else float(pin - pout)
    out.update({
        "total_loss_w": total,
        "high_side_conduction_w": p_hs,
        "low_side_conduction_w": p_ls,
        "freewheel_conduction_w": p_fw,
        "inductor_dcr_w": p_dcr,
        "output_esr_w": p_esr,
    })
    if total is None or p_ls is None or p_esr is None or p_dcr is None:
        out["switching_residual_status"] = "not_available"
        out.setdefault("reason", "one or more measured loss terms are N/A")
        return out
    known = p_hs + p_ls + p_fw + p_dcr + p_esr
    residual = total - known
    tolerance = max(1.0e-9, 0.02 * abs(total))
    if residual < -tolerance:
        out["switching_residual_status"] = "inconsistent"
        out["reason"] = (
            "measured loss exceeds Pin-Pout by more than 2 %; residual N/A")
        return out
    out["switching_residual_w"] = max(0.0, residual)
    out["switching_residual_status"] = "computed"
    out["reason"] = None
    return out


# ---------------------------------------------------------------------------
# Golden-vs-tier comparison
# ---------------------------------------------------------------------------

def resolve_golden_model(config: Mapping[str, Any],
                         vendor: str) -> dict[str, Any]:
    """Resolve the vendor golden library/subcircuit from config/inputs.

    Unlike :func:`resolve_converter_model`, this path uses the vendor YAML's
    ``library``, ``subcircuit`` and ``pin_order_semantic`` fields so numeric
    port tokens (IR/TI ``1/2/3``) and gate-first headers bind correctly.
    """
    from config.yaml_io import load_input
    from parser.netlist import resolve_pin_connections

    try:
        meta = load_input(vendor)
    except Exception as exc:
        raise ConverterModelError(
            f"vendor input config for {vendor!r} is unavailable: {exc}") from exc
    library = Path(str(meta.get("library") or ""))
    if not library.is_absolute():
        library = ROOT / library
    if not library.is_file():
        raise ConverterModelError(
            f"golden vendor library not found for {vendor}: {library}")
    subcircuit = meta.get("subcircuit")
    if not isinstance(subcircuit, str) or not subcircuit.strip():
        raise ConverterModelError(
            f"golden vendor config for {vendor!r} has no subcircuit")

    pins = meta.get("pins") or {}
    semantic = pins.get("pin_order_semantic")
    try:
        if isinstance(semantic, dict) and semantic:
            roles = resolve_pin_connections(library, subcircuit, semantic)
        else:
            _, roles = _parse_model_header(library, subcircuit)
    except Exception as exc:
        raise ConverterModelError(
            f"cannot resolve golden pins for {vendor}: {exc}") from exc
    missing = {"drain", "gate", "source"} - set(roles)
    if missing:
        raise ConverterModelError(
            f"golden {vendor} pins lack role(s): " + ", ".join(sorted(missing)))

    op = _operating_point(config, vendor)
    return {
        "vendor": vendor,
        "tier_key": "golden",
        "tier": None,
        "variant": "golden",
        "library": str(library),
        "library_sha256": sha256_file(library),
        "subcircuit": subcircuit,
        "pin_roles": list(roles),
        "gate_drive_v": op["vgs_on_v"],
        "operating_point": op,
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _comparison_signal(golden: Any, tier: Any) -> dict[str, Any]:
    g = _finite_or_none(golden)
    t = _finite_or_none(tier)
    entry: dict[str, Any] = {
        "golden": g, "tier": t, "abs_error": None,
        "rel_error_pct": None, "status": "not_available", "reason": None,
    }
    if g is None:
        entry["reason"] = "golden value is N/A"
        return entry
    if t is None:
        entry["reason"] = "tier value is N/A"
        return entry
    entry["abs_error"] = t - g
    if abs(g) > 0.0:
        entry["rel_error_pct"] = (t - g) / abs(g) * 100.0
    entry["status"] = "compared"
    return entry


def _compare_results(golden: Mapping[str, Any],
                     tier: Mapping[str, Any]) -> dict[str, Any]:
    signals = {
        "vout_v": _comparison_signal(
            golden.get("vout_v"), tier.get("vout_v")),
        "i_l_mean_a": _comparison_signal(
            golden.get("i_l_mean_a"), tier.get("i_l_mean_a")),
        "i_cap_mean_a": _comparison_signal(
            golden.get("i_cap_mean_a"), tier.get("i_cap_mean_a")),
    }
    efficiency = _comparison_signal(
        golden.get("efficiency"), tier.get("efficiency"))
    golden_ok = bool(
        golden.get("completed")
        and golden.get("status") in ("measured", "not_settled"))
    tier_ok = bool(
        tier.get("completed")
        and tier.get("status") in ("measured", "not_settled"))
    if not golden_ok:
        status = "golden_unavailable"
    elif not tier_ok:
        status = "tier_unavailable"
    elif all(entry["status"] == "compared" for entry in signals.values()):
        status = "compared"
    else:
        status = "partial"

    reasons: list[str] = []
    if not golden_ok:
        reasons.append(f"golden: {golden.get('reason') or golden.get('status')}")
    if not tier_ok:
        reasons.append(f"tier: {tier.get('reason') or tier.get('status')}")
    if status == "partial":
        reasons.append("one or more comparison signals are N/A")
    return {
        "status": status,
        "reason": "; ".join(reasons) if reasons else None,
        "signals": signals,
        "efficiency": efficiency,
        "golden_status": golden.get("status"),
        "tier_status": tier.get("status"),
        "golden_completed": bool(golden.get("completed")),
        "tier_completed": bool(tier.get("completed")),
        "rule": ("same config/conditions for both runs; settled-window means "
                 "from V(vout), I(Lout), I(Cout); errors are tier minus "
                 "golden and never imputed when a trace is N/A"),
    }


def _run_with_simulator_retry(
    run_once: Callable[[], dict[str, Any]], *, allow_retry: bool
) -> dict[str, Any]:
    result = run_once()
    reason = str(result.get("reason") or "")
    if (allow_retry and result.get("status") == "error"
            and "could not convert string to float" in reason):
        retry = run_once()
        retry["simulator_retry"] = {
            "count": 1,
            "first_reason": reason,
            "rule": ("LTspice 26 ASCII .raw occasionally concatenates two "
                     "values; one retry is recorded, and a second failure "
                     "remains N/A"),
        }
        return retry
    return result


def run_converter_comparison(
    vendor: str | None = None,
    tier_key: str | None = None,
    *,
    config_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    runner: Callable[[str, TranDirective], Any] | None = None,
) -> dict[str, Any]:
    """Run the buck bench twice: vendor golden vs selected emitted tier.

    Both runs use identical conditions and the same optional ``runner`` hook.
    A missing golden artifact or an aborted run is preserved as an N/A result
    with its guard/reason; no comparison value is synthesized.  For live
    runs, a single retry is made only when LTspice emitted a malformed ASCII
    raw number (see ``simulator_retry``); a second failure stays N/A.
    """
    cfg = dict(config) if config is not None else load_converter_config(
        config_path)
    selection = cfg["model_selection"]
    vendor = vendor or selection["default"]["vendor"]
    tier_key = tier_key or selection["default"]["tier"]
    if not isinstance(vendor, str) or not isinstance(tier_key, str):
        raise ConverterConfigError("vendor and tier_key must be strings")
    op = _operating_point(cfg, vendor)
    allow_retry = runner is None

    try:
        golden_model = resolve_golden_model(cfg, vendor)
    except ConverterModelError as exc:
        golden = _base_result(None, vendor, "golden", op)
        golden.update(status="model_unavailable", reason=str(exc))
    else:
        golden = _run_with_simulator_retry(
            lambda: run_converter_benchmark(
                vendor, "golden", config=cfg, runner=runner,
                resolved_model=golden_model),
            allow_retry=allow_retry)

    tier = _run_with_simulator_retry(
        lambda: run_converter_benchmark(
            vendor, tier_key, config=cfg, runner=runner),
        allow_retry=allow_retry)
    return {
        "vendor": vendor,
        "tier_key": tier_key,
        "conditions": op,
        "golden": golden,
        "tier": tier,
        "comparison": _compare_results(golden, tier),
    }
