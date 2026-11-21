"""Tier-4 ideal-switch ASCII emitter (fast converter tier).

This emitter is the electrical half of the dual-tier architecture recorded in
``docs/fidelity_speed_research.md`` section 4 item 1 / section 6: the Tier-2/3
ASCII SPICE ROMs keep the device-level DPT waveform gates, while Tier-4 supplies
a converter-level ideal switch whose energy/efficiency gate is met by an
offline loss LUT.

Electrical form (deliberately small -- PLECS-style ideal switching)::

    .SUBCKT <vendor>_PIECESPICE_T4 d g s
    S_SW d s g s SW_T4
    .MODEL SW_T4 SW(Ron=<Curve-3 rds_on>
                    Roff=<measured Vgs=0 Roff | project-decision fallback>
                    Vt=<Curve-7 Vth> Vh=0)
    C_QOSS d s <measured Curve-4 Coss at cap_ref_Vds>
    .ENDS <vendor>_PIECESPICE_T4

The switching-energy tables (``e_on_uj`` / ``e_off_uj``) are emitted as comment
metadata only.  Tier-4 losses are computed offline by the pipeline at the
measured switching current and applied thermally.  This is analogous to PLECS
keeping conduction ``v_on(i,T)`` and ``E_on(v,i,T)`` / ``E_off(v,i,T)`` tables
separate from the electrical switch [P-TC]; unlike PLECS there is no
in-simulator ``(v,i,T)`` interpolation and no thermal RC network here.  The
emitted switch has no loss element, so these numbers are an offline thermal
estimate, not electrical power consumed by the SPICE model.

Exactly one measured LINEAR capacitor (``C_QOSS``, Curve-4 Coss at
``cap_ref_Vds``) is active.  It is a solver stabilizer, not a voltage-dependent
or waveform-fidelity model: it slows drain dv/dt and adds L-C overshoot, so the
DPT waveform gate remains ``retired_by_design``; it only prevents singular /
no-capacitance solver steps.  Voltage-dependent capacitors and the body-diode
switching waveform remain explicitly retired for this ideal-switch tier.
P-DEMO (in ``local://research_simulator_architectures.md``) documents the
finite turn-on transient and Miller gate droop (Fig. 6), diode reverse recovery
(Fig. 7) and the SPICE-side parasitics that this tier drops; it does not print
``dv/dt`` or ``ringing``, which are marked [INFERENCE] here from instantaneous
switching and absent parasitics.  Plexim also offers optional behavioral
models for reverse recovery / limited di/dt, so the retirement is about our
ideal-switch tier, not about PLECS as a product.

No simulator is called and no number is invented here.  Every value is read
from the ``LossLUT`` produced from config inputs and measured curves.  Off
resistance comes from the measured Vgs=0 DC sweep; if that measurement is
unavailable, the historical project-decision fallback is emitted with an
explicit ``source=project_decision`` disclosure.  Zero hysteresis remains the
one unconditional project-decision constant, also labeled in the metadata.
"""

from __future__ import annotations

import json
import math
from statistics import median
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids an import cycle
    from approximation.loss_lut import LossLUT


# --- modelling constants (project decisions, labeled in emitted metadata) ---
# Fallback Roff is used ONLY when the measured Vgs=0 DC-sweep off resistance
# is unavailable; the emitted metadata labels it source=project_decision.
# Vh=0 removes hysteresis so the switch follows the Curve-7 threshold; that
# zero-hysteresis choice stays an explicit project decision in every emission.
_R_OFF_OHMS_FALLBACK = 1.0e12
_SW_HYSTERESIS_V = 0.0

_SW_MODEL_NAME = "SW_T4"
_RETIRED_GATE = "retired_by_design"
_COSS_SOURCE = "measured_curve4_capacitance"
_ROFF_MEASURED = "measured_vgs0_dc_sweep"
_ROFF_FALLBACK = "project_decision"
_HYSTERESIS_SOURCE = "project_decision"

# The on-resistance curve is indexed by the config-derived ``current_points``
# grid.  A single Ron is required by the SPICE ``SW`` primitive; the selected
# measured point is documented in the emitted metadata (project decision, not
# a hidden constant) so the choice traces back to Curve-3.  An even-length
# axis has a midpoint median that is not itself measured; ties break low.
_RON_SELECTION_RULE = (
    "nearest measured current point to median of config-derived "
    "current_points (even-grid midpoint ties break to lower index)")


def _num(x: Any) -> str:
    """SPICE-safe number: NaN stays the explicit ``NaN`` provenance marker."""
    v = float(x)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "NaN"  # never emit an infinite SPICE value
    return f"{v:.9g}"


def _seq(x: Any) -> list[Any]:
    return [] if x is None else list(x)


def _ascii(text: str) -> str:
    """Enforce the repository ASCII invariant on every emitted library."""
    if not text.isascii():
        raise ValueError("Tier-4 emitter requires ASCII-only netlist text")
    return text


def _finite_spice_value(name: str, value: Any) -> float:
    """Reject non-finite values before they can enter an executable .MODEL.

    A NaN/Inf Ron or Vth would produce an unsimulatable deck, so a missing
    measurement fails loudly here instead of being written as the literal
    ``NaN``.  ``_num`` still writes ``NaN`` in *comment* tables, where it is a
    deliberate provenance marker for an unmeasured grid node.
    """
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"Tier-4 emitter requires a finite {name}; got {value!r}")
    return v


def _measured_coss_f(lut: "LossLUT") -> float:
    """Measured linear output capacitance (F) from Curve-4 at cap_ref_Vds.

    This is a solver stabilizer, not a waveform-fidelity capacitance model.
    The emitter never invents a capacitor value: a missing, non-finite or
    non-positive measurement is a hard failure rather than a fabricated
    stabilizer entering an executable netlist.
    """
    raw = getattr(lut, "coss_f", None)
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = float("nan")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "Tier-4 emitter requires a finite positive measured coss_f "
            f"(Curve-4 Coss at cap_ref_Vds); got {raw!r}; no capacitance "
            "value is fabricated")
    return value


def _measured_r_off(lut: "LossLUT") -> tuple[float, str]:
    """Return ``(Roff ohms, source label)`` from the Vgs=0 DC sweep.

    A finite positive measured value wins.  A failed or unavailable
    measurement (NaN, Inf, non-positive or absent) uses the historical
    project-decision fallback; the caller discloses that fallback with
    ``source=project_decision`` in both the netlist comments and metadata.
    """
    raw = getattr(lut, "r_off_ohms", None)
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = float("nan")
    if math.isfinite(value) and value > 0.0:
        return value, _ROFF_MEASURED
    return _R_OFF_OHMS_FALLBACK, _ROFF_FALLBACK


def _reference_index(lut: "LossLUT") -> int | None:
    """Index of the representative conduction point on the LUT current grid.

    Rule (project decision, labeled in metadata): nearest measured point to the
    median of the config-derived ``current_points`` grid.  For an even-length
    axis ``statistics.median`` returns the midpoint of the two central values,
    which is generally NOT one of the measured currents.  The nearest-point
    comparison below then finds that midpoint equidistant from the two central
    points; ``min(range(...), key=abs)`` resolves the exact tie to the LOWER
    index.  No new numeric condition is introduced.
    """
    currents = _seq(getattr(lut, "current_points", ()))
    if not currents:
        return None
    target = median(float(c) for c in currents)
    # Stable lowest-index tie-break for equidistant measured points.
    return min(range(len(currents)),
               key=lambda k: abs(float(currents[k]) - target))


def _r_on_at_index(lut: "LossLUT", index: int | None) -> float:
    """Curve-3 on-resistance at one measured current-grid index."""
    r_on = _seq(getattr(lut, "r_on_ohms", None))
    if r_on:
        k = 0 if index is None else min(index, len(r_on) - 1)
        return float(r_on[k])

    v_on = _seq(getattr(lut, "v_on_v", None))
    currents = _seq(getattr(lut, "current_points", ()))
    if v_on and currents and index is not None:
        k = min(index, len(currents) - 1)
        i, v = float(currents[k]), float(v_on[min(k, len(v_on) - 1)])
        if i > 0.0:
            return v / i
    raise ValueError("LossLUT provides neither r_on_ohms nor usable v_on_v")


def _resolve_reference_index(lut: "LossLUT",
                             i_load_a: float | None) -> tuple[int | None, str]:
    """Index and rule for the representative conduction current."""
    currents = _seq(getattr(lut, "current_points", ()))
    if not currents:
        return None, _RON_SELECTION_RULE
    if i_load_a is None:
        return _reference_index(lut), _RON_SELECTION_RULE
    k = min(range(len(currents)),
            key=lambda j: abs(float(currents[j]) - float(i_load_a)))
    return k, ("nearest measured current point to "
               f"i_load_a={_num(i_load_a)} A (runtime selection)")


def _measured_gate_drives(lut: "LossLUT") -> tuple[float, ...]:
    table = getattr(lut, "r_on_by_vgs", None)
    if not isinstance(table, dict) or not table:
        return ()
    return tuple(sorted(float(k) for k in table))


def resolve_r_on(lut: "LossLUT", *, i_load_a: float | None = None,
                 vgs_on: float | None = None) -> dict[str, Any]:
    """Select the measured Curve-3 on-resistance and publish its rule.

    When the LUT carries a measured ``r_on_by_vgs`` family and ``vgs_on`` is
    supplied, the row at the actual (or nearest measured) gate drive is used.
    Otherwise the historical representative-current rule is preserved.  No
    number is invented; a missing selected value is returned as-is so the
    emitter's finite guard refuses to write it.
    """
    index, current_rule = _resolve_reference_index(lut, i_load_a)
    currents = _seq(getattr(lut, "current_points", ()))
    selected_current = (None if index is None or index >= len(currents)
                        else float(currents[index]))
    table = getattr(lut, "r_on_by_vgs", None)
    gates = _measured_gate_drives(lut)
    out: dict[str, Any] = {
        "value_ohms": None, "index": index,
        "current_a": selected_current,
        "rule": current_rule,
        "source": "r_on_ohms",
        "vgs_on_v": None if vgs_on is None else float(vgs_on),
        "vgs_points": list(gates),
        "vgs_selected_v": None,
        "vgs_match": None,
        "table_available": bool(gates),
    }
    if (gates and vgs_on is not None
            and math.isfinite(float(vgs_on))):
        gate = float(vgs_on)
        chosen = min(gates, key=lambda value: (abs(value - gate), value))
        row = tuple(table[chosen])  # type: ignore[index]
        value = (float(row[index]) if index is not None and index < len(row)
                 else None)
        exact = chosen == gate
        out.update(
            value_ohms=value,
            source="r_on_by_vgs",
            vgs_selected_v=chosen,
            vgs_match=("exact" if exact else "nearest_measured_gate_drive"),
            rule=(f"measured Curve-3 Rds(on) at vgs_on={_num(gate)} V: "
                  + (f"exact measured gate drive {_num(chosen)} V"
                     if exact else
                     "nearest measured gate drive "
                     f"{_num(chosen)} V (no interpolation)")
                  + (f"; {current_rule}" if index is not None else "")),
        )
        return out
    value = _r_on_at_index(lut, index) if index is not None else None
    out["value_ohms"] = None if value is None else float(value)
    if gates and vgs_on is None:
        out["rule"] = (current_rule + "; no vgs_on supplied, so the measured "
                       "r_on_by_vgs family is not selected")
    return out


def conduction_r_on(lut: "LossLUT",
                    vgs_on: float | None = None) -> float:
    """Return the Curve-3 on-resistance used by the emitted ideal switch.

    Uses the measured ``r_on_by_vgs`` row at ``vgs_on`` when both are present;
    otherwise the representative measured current point (median selection, or
    ``i_load_a`` in :func:`build_tier4_subcircuit`).  If a LUT only carries
    ``v_on_v``, R = V/I is derived at that same measured current -- always a
    measured curve point, never a fabricated datasheet number.
    """
    value = resolve_r_on(lut, vgs_on=vgs_on)["value_ohms"]
    if value is None:
        raise ValueError("LossLUT provides neither r_on_ohms nor usable v_on_v")
    return float(value)


def resolve_vt(lut: "LossLUT") -> dict[str, Any]:
    """Switch threshold: config/LUT value first, else Curve-7 JEDEC Vth.

    A non-finite Curve-7 Vth is a hard failure (the emitter never writes a
    literal NaN); a finite ``vt_switch_v`` always wins and is labelled with
    its own source.
    """
    raw = getattr(lut, "vt_switch_v", None)
    try:
        vt = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        vt = float("nan")
    meta = getattr(lut, "source", None)
    meta = meta.get("vt_switch") if isinstance(meta, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    if math.isfinite(vt) and vt > 0.0:
        return {
            "value_v": vt,
            "source": str(meta.get("kind") or "lut_measured"),
            "kind": "configured",
            "rule": ("config/LUT-provided switching threshold used verbatim; "
                     "Curve-7 Vth is not substituted"),
            "source_meta": meta,
        }
    vth = _finite_spice_value("vth_v", getattr(lut, "vth_v"))
    return {
        "value_v": vth,
        "source": "curve7_jedec_vth",
        "kind": "curve7_jedec_vth",
        "rule": ("no finite config/LUT switching threshold; the emitted "
                 "switch uses the measured Curve-7 JEDEC threshold"),
        "source_meta": meta,
    }


def _table_comment(label: str, rows: Any) -> list[str]:
    """Comment rows of a [vbus][current] table, preserving NaN markers."""
    lines: list[str] = []
    for i, row in enumerate(_seq(rows)):
        values = ", ".join(_num(v) for v in _seq(row))
        lines.append(f"* {label}[{i}] = [{values}]")
    return lines


def _source_comments(source: Any) -> list[str]:
    """Emit provenance as one ASCII JSON comment per top-level key."""
    if not isinstance(source, dict) or not source:
        return ["* source = {}"]
    lines = ["* --- provenance (config/measured only) ---"]
    for key in sorted(source):
        value = json.dumps(source[key], sort_keys=True, default=str)
        lines.append(f"* source.{key} = {value}")
    return lines


def _curve_gate_drive_v(lut: "LossLUT") -> float | None:
    """Curve-3 extraction gate drive, if the LUT records it.

    Never conflate this with ``dpt.vgs_on``: CPM2 extracts Curve-3 at
    ``conditions.Vgs_fixed`` = 20 V but switches at ``dpt.vgs_on`` = 18 V.
    """
    source = getattr(lut, "source", None)
    meta = source.get("r_on") if isinstance(source, dict) else None
    drive = meta.get("gate_drive_v") if isinstance(meta, dict) else None
    if drive is None:
        return None
    value = float(drive)
    return value if math.isfinite(value) else None


def build_tier4_subcircuit(lut: "LossLUT", *,
                           i_load_a: float | None = None,
                           vgs_on: float | None = None) -> str:
    """Return a complete ASCII ``.subckt`` for one Tier-4 ideal-switch LUT.

    Parameters
    ----------
    lut:
        The config/measurement-backed ``LossLUT`` (see
        ``approximation.loss_lut``).
    i_load_a:
        Optional measured switching current.  When supplied by the pipeline,
        it selects the Curve-3 conduction point nearest that operating current
        instead of the documented median-grid fallback; it never creates a new
        resistance value, only selects an existing measured one.
    vgs_on:
        Optional actual gate drive.  When the LUT carries a measured
        ``r_on_by_vgs`` family, the row at this (or the nearest measured)
        gate drive is selected; otherwise the representative-current rule is
        preserved and labelled.
    """
    vendor = str(getattr(lut, "vendor", "UNKNOWN"))
    name = f"{vendor}_PIECESPICE_T4"
    ron_selection = resolve_r_on(lut, i_load_a=i_load_a, vgs_on=vgs_on)
    r_on = ron_selection["value_ohms"]
    selection = ron_selection["rule"]
    coss_f = _measured_coss_f(lut)
    r_off, r_off_source = _measured_r_off(lut)
    vt_selection = resolve_vt(lut)
    vt = vt_selection["value_v"]
    # The comment may carry the literal NaN marker, but only when a finite
    # configured Vt already passed resolve_vt(); the executable .MODEL always
    # uses a finite selected threshold.
    vth = getattr(lut, "vth_v", float("nan"))
    temp_c = float(getattr(lut, "temperature_c", float("nan")))
    lut_vgs_on = float(getattr(lut, "vgs_on_v", float("nan")))
    temperature_points = tuple(
        float(t) for t in (getattr(lut, "temperature_points", ()) or ()))
    has_temperature_tables = getattr(
        lut, "e_on_uj_by_temperature", None) is not None

    # Never emit an unsimulatable .MODEL: a missing measurement is a hard
    # failure here, not a literal "NaN" in Ron/Vt.  Comment tables still use
    # "NaN" as the provenance marker for unmeasured grid nodes.
    if r_on is None:
        raise ValueError("LossLUT provides neither r_on_ohms nor usable v_on_v")
    r_on = _finite_spice_value("r_on_ohms", r_on)

    lines: list[str] = []
    ap = lines.append
    ap(f"* PieceSpice Tier-4 ideal-switch model for {vendor}")
    ap("* Method: standard SPICE ideal voltage-controlled switch + offline LUT")
    ap("* Losses handled offline in thermal post-processing (cf. PLECS P-TC).")
    ap("* Standard SPICE SW primitive; no PLECS/SIMPLIS step-count or speed")
    ap("* behavior is claimed for this emitted model.")
    ap(f"* waveform_gate: {_RETIRED_GATE}")
    ap("* Retired observables: dv/dt [INFERENCE], Miller droop (P-DEMO Fig. 6),")
    ap("*   ringing [INFERENCE], reverse recovery (P-DEMO Fig. 7), voltage-")
    ap("*   dependent capacitances, body-diode switching waveform.  dv/dt and")
    ap("*   ringing are inferred from instantaneous switching and absent")
    ap("*   parasitics; P-DEMO names the transient, Miller droop and recovery.")
    ap("* These do not exist in this ideal-switch tier's electrical solution;")
    ap("* energy/efficiency vs golden is scored offline, not on a converter bench.")
    ap("* One measured LINEAR C_QOSS (Curve-4 Coss at cap_ref_Vds) is active only")
    ap("* as a solver stabilizer: it slows drain dv/dt and adds L-C overshoot, so")
    ap("* the DPT waveform gate stays retired_by_design.  It only prevents")
    ap("* singular / no-capacitance solver steps; it is NOT a voltage-dependent")
    ap("* or waveform-fidelity capacitance model.  Voltage-dependent capacitors")
    ap("* and the body diode remain retired (see above).")
    ap("")
    ap(f".SUBCKT {name} d g s")
    ap("* Pins: d=drain g=gate s=source (positional; gate threshold Vgs=V(g,s))")
    ap("")
    ap("* --- ideal conduction (Curve-3 rds_on, measured Roff, switch Vt) ---")
    ap(f"* temperature_c = {_num(temp_c)} (reference temperature; config conditions.temperature)")
    if has_temperature_tables:
        ap("* data_limitation: E_on/E_off are measured over the opt-in")
        ap(f"*   temperature axis [{', '.join(_num(t) for t in temperature_points)}] C;")
        ap("*   trilinear interpolation is used only when all eight corners of")
        ap("*   adjacent slices are finite. No temperature extrapolation exists.")
        ap(f"*   r_on remains measured at conditions.temperature = {_num(temp_c)} C.")
    else:
        ap("* data_limitation: E_on/E_off and r_on are measured only at this")
        ap(f"*   conditions.temperature = {_num(temp_c)} C; the LUT has no temperature")
        ap("*   axis and is valid only there.  It is a measured data slice, not a")
        ap("*   physical guarantee at other junction temperatures.")
    _grid_provenance = getattr(lut, "grid_provenance", None)
    if callable(_grid_provenance):
        ap(f"*   {_grid_provenance()}")
    else:
        ap("*   Grid provenance unavailable (no interpolation capability).")
    ap("*   Losses are evaluated offline.")
    ap("*   They are not electrical power consumed by this switch (P-TC:")
    ap("*   Conduction Losses / Switching Losses notes).")
    ap(f"* vth_v = {_num(vth)} (Curve-7 JEDEC threshold, "
       "config/outputs/<vendor>/curve7_threshold_voltage.yaml)")
    ap(f"* vt_switch_v = {_num(vt)} (source={vt_selection['source']}; "
       f"{vt_selection['rule']})")
    drive_v = _curve_gate_drive_v(lut)
    if drive_v is None:
        ap(f"* r_on_ohms = {_num(r_on)} (Curve-3 rds_on at unrecorded gate drive; "
           f"selection: {selection})")
    else:
        ap(f"* r_on_ohms = {_num(r_on)} (Curve-3 rds_on at {_num(drive_v)} V, "
           f"conditions.Vgs_fixed; selection: {selection})")
    ap(f"* r_on_source = {ron_selection['source']} "
       f"(vgs_on_v={_num(ron_selection['vgs_on_v']) if ron_selection['vgs_on_v'] is not None else 'N/A'}; "
       f"vgs_points={ron_selection['vgs_points']})")
    ap(f"* vgs_on_v = {_num(lut_vgs_on)} (config dpt.vgs_on; gate-drive metadata only)")
    ap(f"* r_off_ohms = {_num(r_off)} (source={r_off_source}; measured at Vgs=0 "
       "DC sweep as Roff=Vbus/Ioff; full provenance in source.r_off)")
    if r_off_source == _ROFF_FALLBACK:
        ap("*   FALLBACK: the measured off resistance is unavailable "
           "(NaN/absent/non-positive).")
        ap("*   No fabricated measurement is substituted; measured Vgs=0 Roff")
        ap("*   is preferred whenever it is available.")
    ap(f"* vh_v = {_num(_SW_HYSTERESIS_V)} (source={_HYSTERESIS_SOURCE}: zero "
       "hysteresis, switch tracks the selected threshold)")
    ap(f"S_SW d s g s {_SW_MODEL_NAME}")
    ap(f".MODEL {_SW_MODEL_NAME} SW(Ron={_num(r_on)} "
       f"Roff={_num(r_off)} Vt={_num(vt)} "
       f"Vh={_num(_SW_HYSTERESIS_V)})")
    ap("* --- one measured linear output capacitor: solver stabilizer only ---")
    ap(f"* coss_f = {_num(coss_f)} F (source={_COSS_SOURCE}; Curve-4 Coss at "
       "cap_ref_Vds; provenance in source.coss)")
    ap(f"C_QOSS d s {_num(coss_f)}")
    ap("")
    ap("* --- loss lookup data: COMMENT / METADATA ONLY (no active elements) ---")
    ap("* Energy is computed offline by pipeline.tier4 at the measured switching")
    ap("* current and applied thermally; no E/G source consumes these tables.")
    ap(f"* vbus_points = [{', '.join(_num(v) for v in _seq(getattr(lut, 'vbus_points', ())))}]")
    ap(f"* current_points = [{', '.join(_num(v) for v in _seq(getattr(lut, 'current_points', ())))}]")
    ap(f"* temperature_points = [{', '.join(_num(t) for t in temperature_points)}]")
    ap("* e_on_uj[vbus][current], microjoules (NaN = no golden crossing):")
    lines.extend(_table_comment("e_on_uj", getattr(lut, "e_on_uj", ())))
    ap("* e_off_uj[vbus][current], microjoules (NaN = no golden crossing):")
    lines.extend(_table_comment("e_off_uj", getattr(lut, "e_off_uj", ())))
    on_by_temperature = getattr(lut, "e_on_uj_by_temperature", None)
    off_by_temperature = getattr(lut, "e_off_uj_by_temperature", None)
    if on_by_temperature is not None and off_by_temperature is not None:
        for temperature, e_on_matrix, e_off_matrix in zip(
                temperature_points, on_by_temperature, off_by_temperature):
            ap(f"* e_on_uj[T={_num(temperature)}][vbus][current] = "
               "(metadata only)")
            lines.extend(_table_comment(
                f"e_on_uj[T={_num(temperature)}]", e_on_matrix))
            ap(f"* e_off_uj[T={_num(temperature)}][vbus][current] = "
               "(metadata only)")
            lines.extend(_table_comment(
                f"e_off_uj[T={_num(temperature)}]", e_off_matrix))
    ron_curve = ", ".join(_num(v) for v in _seq(getattr(lut, "r_on_ohms", None)))
    von_curve = ", ".join(_num(v) for v in _seq(getattr(lut, "v_on_v", None)))
    ap(f"* r_on_ohms[current] = [{ron_curve}]")
    ap(f"* v_on_v[current] = [{von_curve}]")
    ron_by_vgs = getattr(lut, "r_on_by_vgs", None)
    if isinstance(ron_by_vgs, dict):
        for gate, row in sorted(ron_by_vgs.items()):
            ap(f"* r_on_ohms[vgs={_num(gate)}][current] = "
               f"[{', '.join(_num(v) for v in row)}]")
    lines.extend(_source_comments(getattr(lut, "source", None)))
    stabilization = {
        "coss_f": coss_f,
        "coss_source": _COSS_SOURCE,
        "r_off_ohms": r_off,
        "r_off_source": r_off_source,
        "v_hysteresis_v": _SW_HYSTERESIS_V,
        "v_hysteresis_source": _HYSTERESIS_SOURCE,
        "vt_switch_v": vt,
        "vt_switch_source": vt_selection["source"],
        "ron_selection_source": ron_selection["source"],
        "ron_selected_ohms": r_on,
        "ron_vgs_selected_v": ron_selection["vgs_selected_v"],
        "rationale": (
            "missing measured output capacitance caused Tier-4 DPT "
            "convergence failures; one measured linear C prevents "
            "singular/no-cap steps; DPT waveform gate remains retired; "
            "Vt and Ron are the measured/config-selected values, never NaN"),
    }
    ap(f"* emitter_stabilization = {json.dumps(stabilization, sort_keys=True)}")
    ap("")
    ap(f".ENDS {name}")
    return _ascii("\n".join(lines) + "\n")


def build_tier4_ann_subcircuit(
    spec,
    channel_fit,
    charge_fit,
    lut: "LossLUT",
    *,
    subckt_name: str | None = None,
) -> str:
    """Emit Tier-4 dynamic C-inf ANN subcircuit with LossLUT metadata."""
    from models.ann_netlist_builder import build_ann_subcircuit

    device = getattr(lut, "device", None) or getattr(spec, "device_name", None)
    name = subckt_name or f"{device}_PIECESPICE_T4_ann"
    text = build_ann_subcircuit(spec, channel_fit, charge_fit, subckt_name=name)
    lut_comments = [
        "* --- Tier-4 Loss LUT metadata (offline post-processing) ---",
        *_table_comment("e_on_uj", getattr(lut, "e_on_uj", None)),
        *_table_comment("e_off_uj", getattr(lut, "e_off_uj", None)),
    ]
    lines = text.splitlines()
    ends_idx = next(i for i, l in enumerate(lines) if l.strip().upper().startswith(".ENDS"))
    out = lines[:ends_idx] + lut_comments + lines[ends_idx:]
    return _ascii("\n".join(out) + "\n")
