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


def conduction_r_on(lut: "LossLUT") -> float:
    """Return the Curve-3 on-resistance used by the emitted ideal switch.

    Uses ``r_on_ohms`` (aligned 1:1 to ``current_points``) at the documented
    representative current.  If a LUT only carries ``v_on_v``, R = V/I is
    derived at that same measured current -- always a measured curve point,
    never a fabricated datasheet number.
    """
    return _r_on_at_index(lut, _reference_index(lut))


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
                           i_load_a: float | None = None) -> str:
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
    """
    vendor = str(getattr(lut, "vendor", "UNKNOWN"))
    name = f"{vendor}_PIECESPICE_T4"
    r_on = conduction_r_on(lut)
    coss_f = _measured_coss_f(lut)
    r_off, r_off_source = _measured_r_off(lut)
    vth = _finite_spice_value("vth_v", getattr(lut, "vth_v"))
    temp_c = float(getattr(lut, "temperature_c", float("nan")))
    vgs_on = float(getattr(lut, "vgs_on_v", float("nan")))

    selection = _RON_SELECTION_RULE
    if i_load_a is not None:
        currents = _seq(getattr(lut, "current_points", ()))
        if currents:
            k = min(range(len(currents)),
                    key=lambda j: abs(float(currents[j]) - float(i_load_a)))
            r_on = _r_on_at_index(lut, k)
            selection = ("nearest measured current point to "
                         f"i_load_a={_num(i_load_a)} A (runtime selection)")
    # Never emit an unsimulatable .MODEL: a missing measurement is a hard
    # failure here, not a literal "NaN" in Ron/Vt.  Comment tables still use
    # "NaN" as the provenance marker for unmeasured grid nodes.
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
    ap("* --- ideal conduction (Curve-3 rds_on, measured Roff, Curve-7 Vth) ---")
    ap(f"* temperature_c = {_num(temp_c)} (config conditions.temperature)")
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
    ap(f"* vth_v = {_num(vth)} (config/outputs/<vendor>/curve7_threshold_voltage.yaml)")
    drive_v = _curve_gate_drive_v(lut)
    if drive_v is None:
        ap(f"* r_on_ohms = {_num(r_on)} (Curve-3 rds_on at unrecorded gate drive; "
           f"selection: {selection})")
    else:
        ap(f"* r_on_ohms = {_num(r_on)} (Curve-3 rds_on at {_num(drive_v)} V, "
           f"conditions.Vgs_fixed; selection: {selection})")
    ap(f"* vgs_on_v = {_num(vgs_on)} (config dpt.vgs_on; gate-drive metadata only)")
    ap(f"* r_off_ohms = {_num(r_off)} (source={r_off_source}; measured at Vgs=0 "
       "DC sweep as Roff=Vbus/Ioff; full provenance in source.r_off)")
    if r_off_source == _ROFF_FALLBACK:
        ap("*   FALLBACK: the measured off resistance is unavailable "
           "(NaN/absent/non-positive).")
        ap("*   No fabricated measurement is substituted; measured Vgs=0 Roff")
        ap("*   is preferred whenever it is available.")
    ap(f"* vh_v = {_num(_SW_HYSTERESIS_V)} (source={_HYSTERESIS_SOURCE}: zero "
       "hysteresis, switch tracks the Curve-7 threshold)")
    ap(f"S_SW d s g s {_SW_MODEL_NAME}")
    ap(f".MODEL {_SW_MODEL_NAME} SW(Ron={_num(r_on)} "
       f"Roff={_num(r_off)} Vt={_num(vth)} "
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
    ap("* e_on_uj[vbus][current], microjoules (NaN = no golden crossing):")
    lines.extend(_table_comment("e_on_uj", getattr(lut, "e_on_uj", ())))
    ap("* e_off_uj[vbus][current], microjoules (NaN = no golden crossing):")
    lines.extend(_table_comment("e_off_uj", getattr(lut, "e_off_uj", ())))
    ron_curve = ", ".join(_num(v) for v in _seq(getattr(lut, "r_on_ohms", None)))
    von_curve = ", ".join(_num(v) for v in _seq(getattr(lut, "v_on_v", None)))
    ap(f"* r_on_ohms[current] = [{ron_curve}]")
    ap(f"* v_on_v[current] = [{von_curve}]")
    lines.extend(_source_comments(getattr(lut, "source", None)))
    stabilization = {
        "coss_f": coss_f,
        "coss_source": _COSS_SOURCE,
        "r_off_ohms": r_off,
        "r_off_source": r_off_source,
        "v_hysteresis_v": _SW_HYSTERESIS_V,
        "v_hysteresis_source": _HYSTERESIS_SOURCE,
        "rationale": (
            "missing measured output capacitance caused Tier-4 DPT "
            "convergence failures; one measured linear C prevents "
            "singular/no-cap steps; DPT waveform gate remains retired"),
    }
    ap(f"* emitter_stabilization = {json.dumps(stabilization, sort_keys=True)}")
    ap("")
    ap(f".ENDS {name}")
    return _ascii("\n".join(lines) + "\n")
