"""Offline unit tests for the Tier-4 ideal-switch ASCII emitter.

No simulator is invoked: these tests read only the emitted netlist text and a
synthetic LossLUT assembled from config/curve-shaped values.  The DPT/bench
completion and energy gates belong to the pipeline layer.
"""

from __future__ import annotations

import pytest

from approximation.loss_lut import LossLUT
from models.tier4_ideal_switch_builder import (
    build_tier4_subcircuit,
    conduction_r_on,
)


def _toy_lut(*, vendor: str = "csd16412q5a",
             r_on: tuple[float, ...] | None = (0.0104, 0.0108, 0.0116),
             v_on: tuple[float, ...] | None = None,
             vth: float = 1.88,
             vgs_on: float = 10.0,
             gate_drive_v: float | None = None,
             coss_f: float = 3.2e-9,
             r_off: float = 1.25e6) -> LossLUT:
    """Small synthetic LUT shaped like the canonical dataclass.

    The two pinned measurement fields are attached with ``object.__setattr__``
    so this fixture works whether or not the canonical frozen dataclass has
    declared them yet; the emitter reads them with ``getattr``.
    """
    r_on_source: object = "curve3_rds_on"
    if gate_drive_v is not None:
        r_on_source = {"gate_drive_v": gate_drive_v,
                       "rule": "conditions.Vgs_fixed"}
    lut = LossLUT(
        vendor=vendor,
        temperature_c=25.0,
        vbus_points=(12.5, 20.0),
        current_points=(5.0, 10.0, 20.0),
        e_on_uj=((12.0, 25.0, 48.0), (19.0, 39.0, 75.0)),
        e_off_uj=((3.0, 6.0, 12.0), (5.0, float("nan"), 19.0)),
        r_on_ohms=r_on,
        v_on_v=v_on,
        vth_v=vth,
        vgs_on_v=vgs_on,
        source={
            "method": "golden DPT sweep",
            "grid_rules": {"vbus": "conditions.Vbus",
                           "current": "conditions.Id_load"},
            "vector": "config-derived",
            "r_on": r_on_source,
            "vth": "curve7_threshold_voltage",
            "e_on": "golden_dpt_eon",
            "e_off": "golden_dpt_eoff",
            "coss": {
                "file": "config/outputs/<vendor>/curve4_capacitances.yaml",
                "key": "extracted.coss_f",
                "cap_ref_Vds_v": 20.0,
                "temperature_c": 25.0,
            },
            "r_off": {
                "bench": "Vgs=0 DC sweep at Vbus",
                "vgs_v": 0.0,
                "vbus_v": 20.0,
                "i_off_a": 1.6e-5,
                "config_sha256": "toy-synthetic-sha",
            },
            "provenance": {"vendor": vendor, "temperature_c": 25.0},
        },
    )
    object.__setattr__(lut, "coss_f", coss_f)
    object.__setattr__(lut, "r_off_ohms", r_off)
    return lut


def _with_field(lut: LossLUT, name: str, value: object) -> LossLUT:
    """Override one measured field on a frozen fixture for negative tests."""
    object.__setattr__(lut, name, value)
    return lut


def _active_lines(text: str) -> list[str]:
    """Netlist lines that are not blank and are not SPICE comments."""
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("*")]


def test_emission_is_ascii_with_contract_header():
    text = build_tier4_subcircuit(_toy_lut())
    assert text.isascii()
    assert ".SUBCKT csd16412q5a_PIECESPICE_T4 d g s" in text
    assert ".ENDS csd16412q5a_PIECESPICE_T4" in text
    assert text.endswith("\n")


def test_ideal_switch_uses_curve7_vth_and_curve3_ron():
    text = build_tier4_subcircuit(_toy_lut())
    # median of the config-derived current grid (5, 10, 20) is 10 A -> 0.0108
    assert "Ron=0.0108" in text
    assert "Vt=1.88" in text
    assert "S_SW d s g s SW_T4" in text
    assert ".MODEL SW_T4 SW(" in text
    assert "Roff=1250000" in text
    assert "Vh=0" in text
    assert "C_QOSS d s 3.2e-09" in text


def test_runtime_current_selects_an_existing_measured_ron_point():
    text = build_tier4_subcircuit(_toy_lut(), i_load_a=19.0)
    # nearest measured current is 20 A; no interpolated resistance invented
    assert "Ron=0.0116" in text
    assert "i_load_a=19 A" in text


def test_loss_tables_are_metadata_only_not_active_elements():
    text = build_tier4_subcircuit(_toy_lut())
    active = _active_lines(text)
    # Only the subckt frame, ideal switch, its model and ONE linear stabilizer
    # capacitor are executable; the loss tables remain comment metadata.
    assert active == [
        ".SUBCKT csd16412q5a_PIECESPICE_T4 d g s",
        "S_SW d s g s SW_T4",
        ".MODEL SW_T4 SW(Ron=0.0108 Roff=1250000 Vt=1.88 Vh=0)",
        "C_QOSS d s 3.2e-09",
        ".ENDS csd16412q5a_PIECESPICE_T4",
    ]
    assert "e_on_uj[0] = [12, 25, 48]" in text
    assert "e_off_uj[1] = [5, NaN, 19]" in text
    assert "VALUE" not in text


def test_retired_waveform_semantics_are_explicit():
    text = build_tier4_subcircuit(_toy_lut())
    assert "waveform_gate: retired_by_design" in text
    for token in ("dv/dt", "Miller", "ringing", "reverse recovery"):
        assert token in text
    assert "energy/efficiency vs golden" in text
    # Voltage-dependent waveform-shaping elements remain absent; the single
    # linear C_QOSS is present purely as a solver stabilizer.
    for forbidden in ("C_ISS", "C_RSS", "D_BODY", "D_BD_", "Q={"):
        assert forbidden not in text
    assert "C_QOSS d s 3.2e-09" in text
    assert "solver stabilizer" in text
    assert "slows drain dv/dt and adds L-C overshoot" in text
    assert "waveform gate stays retired_by_design" in text
    assert "singular / no-capacitance solver steps" in text
    assert "Voltage-dependent capacitors" in text
    assert "body diode remain retired" in text


def test_conduction_fallback_derives_r_from_measured_v_on_curve():
    lut = _toy_lut(r_on=None, v_on=(0.05, 0.11, 0.23))
    assert conduction_r_on(lut) == pytest.approx(0.11 / 10.0)
    text = build_tier4_subcircuit(lut)
    assert "Ron=0.011" in text
    assert "r_on_ohms[current] = []" in text
    assert "v_on_v[current] = [0.05, 0.11, 0.23]" in text


def test_provenance_is_emitted_as_ascii_json_comments():
    text = build_tier4_subcircuit(_toy_lut())
    assert '* source.method = "golden DPT sweep"' in text
    assert "* source.grid_rules =" in text
    assert "conditions.Vbus" in text
    assert "* source.provenance =" in text


def test_non_ascii_vendor_is_rejected():
    with pytest.raises(ValueError, match="ASCII"):
        build_tier4_subcircuit(_toy_lut(vendor="c\u00e9sar"))


def test_emission_is_deterministic():
    assert build_tier4_subcircuit(_toy_lut()) == \
        build_tier4_subcircuit(_toy_lut())


def test_emitted_lib_makes_no_plecs_step_count_claim():
    text = build_tier4_subcircuit(_toy_lut())
    assert "two steps" not in text
    assert "Losses handled offline in thermal post-processing" in text
    assert "no PLECS/SIMPLIS step-count or speed" in text
    assert "standard SPICE ideal voltage-controlled switch" in text


def test_single_temperature_and_grid_capability_limits_are_in_lib_comments():
    text = build_tier4_subcircuit(_toy_lut())
    assert "data_limitation:" in text
    assert "conditions.temperature = 25 C" in text
    assert "no temperature" in text
    assert "finite measured node(s)" in text
    assert "no 2x2 interpolation cell has all four corners finite" in text
    assert "off-node queries are NaN" in text
    assert "not electrical power consumed by this switch" in text


def test_inferred_observables_are_labeled_and_documented_ones_cited():
    text = build_tier4_subcircuit(_toy_lut())
    assert "dv/dt [INFERENCE]" in text
    assert "ringing [INFERENCE]" in text
    assert "Miller droop (P-DEMO Fig. 6)" in text
    assert "reverse recovery (P-DEMO Fig. 7)" in text


def test_ron_comment_uses_curve3_extraction_drive_not_dpt_vgs():
    text = build_tier4_subcircuit(_toy_lut(gate_drive_v=20.0, vgs_on=18.0),
                                  i_load_a=10.0)
    assert "Curve-3 rds_on at 20 V, conditions.Vgs_fixed" in text
    assert "Curve-3 rds_on at rated dpt.vgs_on" not in text
    assert "vgs_on_v = 18 (config dpt.vgs_on; gate-drive metadata only)" in text


def test_missing_gate_drive_metadata_is_not_mislabeled():
    text = build_tier4_subcircuit(_toy_lut())
    assert "Curve-3 rds_on at unrecorded gate drive" in text


def test_non_finite_vth_or_ron_is_rejected_before_emission():
    with pytest.raises(ValueError, match="finite vth_v"):
        build_tier4_subcircuit(_toy_lut(vth=float("nan")))
    with pytest.raises(ValueError, match="finite r_on_ohms"):
        build_tier4_subcircuit(_toy_lut(r_on=(0.0104, float("nan"), 0.0116)))
    with pytest.raises(ValueError, match="finite r_on_ohms"):
        build_tier4_subcircuit(_toy_lut(r_on=None,
                                        v_on=(0.05, float("nan"), 0.23)))


def test_executable_model_line_has_no_nan_marker():
    text = build_tier4_subcircuit(_toy_lut())
    model_lines = [ln for ln in text.splitlines() if ln.startswith(".MODEL")]
    assert len(model_lines) == 1
    assert "NaN" not in model_lines[0]
    # Comments still carry NaN as the explicit unmeasured-node marker.
    assert "e_off_uj[1] = [5, NaN, 19]" in text


def test_even_grid_median_ties_break_to_lower_measured_point():
    currents = (5.0, 10.0, 20.0, 40.0)
    n = len(currents)
    lut = LossLUT(
        vendor="cpm2-1200",
        temperature_c=25.0,
        vbus_points=(20.0,),
        current_points=currents,
        e_on_uj=((1.0,) * n,),
        e_off_uj=((0.5,) * n,),
        r_on_ohms=(0.001, 0.002, 0.003, 0.004),
        vth_v=2.0,
        vgs_on_v=18.0,
    )
    # This bare fixture predates the pinned measurement fields; attach them
    # explicitly so the emitter has a measured Coss and Roff to emit.
    lut = _with_field(lut, "coss_f", 3.2e-9)
    lut = _with_field(lut, "r_off_ohms", 1.25e6)
    # median(5, 10, 20, 40) = 15 A; 10 and 20 A are equidistant -> lower index.
    assert conduction_r_on(lut) == pytest.approx(0.002)
    text = build_tier4_subcircuit(lut)
    assert "Ron=0.002" in text
    assert "even-grid midpoint ties break to lower index" in text


def test_coss_is_one_active_measured_linear_capacitor():
    text = build_tier4_subcircuit(_toy_lut())
    qoss = [ln for ln in _active_lines(text) if ln.startswith("C_QOSS ")]
    assert qoss == ["C_QOSS d s 3.2e-09"]
    assert "coss_f = 3.2e-09 F (source=measured_curve4_capacitance" in text
    assert "* source.coss =" in text
    assert "curve4_capacitances.yaml" in text
    assert "extracted.coss_f" in text
    assert "cap_ref_Vds_v" in text


def test_r_off_is_measured_when_available():
    text = build_tier4_subcircuit(_toy_lut())
    assert "Roff=1250000" in text
    assert "r_off_ohms = 1250000 (source=measured_vgs0_dc_sweep" in text
    assert "*   FALLBACK:" not in text
    assert "* source.r_off =" in text
    assert "Vgs=0 DC sweep" in text


@pytest.mark.parametrize("failed", [float("nan"), float("inf"),
                                    None, 0.0, -1.0])
def test_unmeasured_r_off_falls_back_with_project_decision_disclosure(failed):
    lut = _with_field(_toy_lut(), "r_off_ohms", failed)
    text = build_tier4_subcircuit(lut)
    assert "Roff=1e+12" in text
    assert "r_off_ohms = 1e+12 (source=project_decision" in text
    assert "*   FALLBACK:" in text
    assert "No fabricated measurement is substituted" in text
    assert '"r_off_source": "project_decision"' in text


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"),
                                 0.0, -1e-9])
def test_missing_or_nonphysical_coss_is_rejected(bad):
    lut = _with_field(_toy_lut(), "coss_f", bad)
    with pytest.raises(ValueError, match="finite positive measured coss_f"):
        build_tier4_subcircuit(lut)


def test_zero_hysteresis_is_labeled_project_decision():
    text = build_tier4_subcircuit(_toy_lut())
    assert "Vh=0" in text
    assert "vh_v = 0 (source=project_decision" in text
    assert '"v_hysteresis_source": "project_decision"' in text


def test_emitter_stabilization_discloses_provenance_and_rationale():
    text = build_tier4_subcircuit(_toy_lut())
    assert "* emitter_stabilization = {" in text
    assert '"coss_source": "measured_curve4_capacitance"' in text
    assert '"r_off_source": "measured_vgs0_dc_sweep"' in text
    assert "convergence failures" in text
    assert "singular/no-cap steps" in text


def test_no_active_element_carries_nan_marker():
    text = build_tier4_subcircuit(_toy_lut())
    assert all("NaN" not in ln for ln in _active_lines(text))
