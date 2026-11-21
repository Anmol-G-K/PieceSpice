"""Offline tests for the bias-aware SIMPLIS Tier-3 charge improvements.

No LTspice: the capacitance bench is driven by a stub AC result and the
pipeline helpers are exercised on synthetic persisted data. Nothing here
introduces a multi-bias or temperature axis that was not measured/persisted.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

import pipeline.tier3_simplis as sm
from common.types import PWLModelSpec
from models.simplis_netlist_builder import (
    _charge_meta_lines,
    _q_expression,
    build_simplis_subcircuit,
)
from pipeline.tier3_simplis import (
    CAP_ERROR_BUDGET_PCT,
    CDG_AXIS_VDG,
    CDG_AXIS_VGS0,
    _CV_SINGLE_BIAS_LIMITATION,
    _cap_error_pct,
    _charge_cap_error_from_details,
    _cv_sweep_from_extracted,
    _cv_sweeps_from_extracted,
    _declared_cv_metadata,
    _fit_anchored_charge,
    _fit_charge_tables,
    _fit_record,
    _load_cv_sweep,
    _load_cv_sweeps,
    _primary_cv_sweep,
    _select_cdg_data,
    _select_charge_candidate,
    _ChargeTables,
    turn_off_energy_decomposition,
)
from simulation.testbenches.capacitance import CapacitanceTestbench
from simulation.testbenches.double_pulse import DoublePulseRun, extract_dpt_metrics


# The historical emitters, captured before the new options were added. The
# default call must remain byte-for-byte identical.
_DEFAULT_SHA256 = {
    "tables_vgs": "43037f2690c5f8e86a43dda1aeff6099e3f40cf2de95aafa9dd7819ecf173c0d",
    "tables_linear": "ef055221a95b6bcddb0cd413cba5d9bdd06ae8176beda7b75e735c763e93bd0f",
    "tier2": "da980aae3b9b2efdca7d970106f911b18c634465c12b1491bc3e7809e66c5512",
}


def _simplis_spec(*, linear_cgs: bool = False) -> PWLModelSpec:
    v = [0.1, 1.0, 10.0, 25.0]
    cgs_q = ([0.0, 2e-9, 20e-9, 50e-9] if linear_cgs
             else [0.0, 3e-9, 12e-9, 26e-9])
    return PWLModelSpec(
        device_name="TOY",
        tier=3,
        breakpoints_vds=[0.0, 1.0, 10.0],
        breakpoints_id=[[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]],
        vgs_values=[4.0, 10.0],
        q_iss_vs_v=v, q_iss_charge_c=cgs_q,
        q_oss_vs_v=v, q_oss_charge_c=[0.0, 4e-9, 20e-9, 50e-9],
        q_rss_vs_v=v, q_rss_charge_c=[0.0, 1e-9, 5e-9, 12e-9],
        rds_on=0.01, v_th=1.0,
        body_diode_vsd=[0.5, 0.8],
        body_diode_isd=[1e-6, 1e-3],
        metadata={"r_gate": 1.5, "r_source": 1e-4, "r_drain": 1e-4},
    )


def test_default_emission_is_byte_identical_to_historical_form():
    for label, kwargs, linear in (
        ("tables_vgs", {"cgs_axis": "vgs"}, False),
        ("tables_linear", {}, True),
        ("tier2", {"charge_mode": "tier2"}, False),
    ):
        text = build_simplis_subcircuit(_simplis_spec(linear_cgs=linear),
                                        **kwargs)
        digest = hashlib.sha256(text.encode("ascii")).hexdigest()
        assert digest == _DEFAULT_SHA256[label], label
        assert not any(line.startswith("* C-V") for line in text.splitlines())


# ---- 1. bias-aware capacitance sweep ---------------------------------------

class _AcResult:
    def __init__(self, magnitude, phase_deg):
        self.traces = {
            "V(gate):m": magnitude, "V(gate):p": phase_deg,
            "V(drain):m": magnitude, "V(drain):p": phase_deg,
        }


def _biased_bench(result_factory):
    bench = CapacitanceTestbench(engine=SimpleNamespace(temperature=25.0))
    bench._run_ac = lambda text: result_factory()
    return bench


_PIN_ROLES = ["drain", "gate", "source"]


def test_bias_netlist_uses_dc_feed_and_keeps_default_unchanged():
    bench = CapacitanceTestbench(engine=SimpleNamespace(temperature=25.0))
    base = bench._ac_netlist(
        "unused.lib", "TOY", _PIN_ROLES, 5.0, "a")
    default = bench._ac_netlist(
        "unused.lib", "TOY", _PIN_ROLES, 5.0, "a", vgs_bias=0.0)
    biased = bench._ac_netlist(
        "unused.lib", "TOY", _PIN_ROLES, 5.0, "a", vgs_bias=2.0)
    assert default == base
    assert "Vgbias" not in base
    assert "Lgbias vgbus gate" in biased
    assert "Vgbias vgbus 0 2" in biased
    # The 1 A AC probe is still the only AC excitation.
    assert "Iprobe gate 0 DC 0 AC 1" in biased


def test_bias_sweep_rejects_on_state_bias():
    # In-phase (resistive) small-signal response: the channel is on and
    # |V| = 1/(w*C) no longer describes a capacitance.
    bench = _biased_bench(lambda: _AcResult(0.1, 0.0))
    with pytest.raises(ValueError, match="not capacitive|channel"):
        bench.run("unused.lib", "TOY", vds_min=1.0, vds_max=2.0, points=2,
                  pin_roles=_PIN_ROLES, vgs_bias=2.0)


def test_bias_sweep_returns_bias_metadata_when_valid():
    # Capacitive phase (-90 deg), magnitude chosen for a sane C.
    bench = _biased_bench(lambda: _AcResult(1e-2, -90.0))
    out = bench.run("unused.lib", "TOY", vds_min=1.0, vds_max=2.0,
                    points=2, pin_roles=_PIN_ROLES, vgs_bias=2.0)
    assert len(out) == 3
    assert all(c.metadata["vgs_bias_v"] == 2.0 for c in out)
    default = bench.run("unused.lib", "TOY", vds_min=1.0, vds_max=2.0,
                        points=2, pin_roles=_PIN_ROLES)
    assert "vgs_bias_v" not in default[0].metadata


# ---- 2. multi-bias persistence and single-bias disclosure ------------------

_V = [0.1, 1.0, 5.0, 10.0]


def _by_vgs(extracted, key, values):
    extracted[key] = {str(bias): list(values) for bias in (0.0, 5.0)}


def _multi_bias_extracted():
    out = {"curve_vds": list(_V)}
    _by_vgs(out, "ciss_curve_f_by_vgs", [1e-9] * 4)
    _by_vgs(out, "coss_curve_f_by_vgs", [5e-10] * 4)
    _by_vgs(out, "crss_curve_f_by_vgs", [1e-10] * 4)
    # Patch the 5 V rows so the two biases are distinguishable.
    for key in ("ciss_curve_f_by_vgs", "coss_curve_f_by_vgs",
                "crss_curve_f_by_vgs"):
        out[key]["5.0"] = [2e-9] * 4
    out["temperature_c_by_sweep"] = {"0.0": 25.0, "5.0": 100.0}
    return out


def test_multi_bias_persist_builds_one_sweep_per_bias():
    sweeps = _cv_sweeps_from_extracted(_multi_bias_extracted(), "TOY")
    assert [s.vgs_bias_v for s in sweeps] == [0.0, 5.0]
    assert [s.temperature_c for s in sweeps] == [25.0, 100.0]
    assert _primary_cv_sweep(sweeps).vgs_bias_v == 0.0
    # A genuine bias-aware Cgd table switches the axis to measured Vdg.
    axis, _, cdg_axis, bias, limitation = _select_cdg_data(
        sweeps, _primary_cv_sweep(sweeps))
    assert cdg_axis == CDG_AXIS_VDG and bias == 5.0
    assert np.all(np.diff(axis) > 0) and limitation is None


def test_unlabeled_multi_bias_list_fails_closed():
    extracted = {"curve_vds": list(_V)}
    for key in ("ciss_curve_f_by_vgs", "coss_curve_f_by_vgs",
                "crss_curve_f_by_vgs"):
        extracted[key] = [[1e-9] * 4, [2e-9] * 4]
    with pytest.raises(ValueError, match="bias axis"):
        _cv_sweeps_from_extracted(extracted, "TOY")


def test_single_vgs0_persist_keeps_legacy_tuple_and_discloses_limit(
        monkeypatch):
    sweep = _cv_sweep_from_extracted(
        {"curve_vds": list(_V), "ciss_curve_f": [1e-9] * 4,
         "coss_curve_f": [5e-10] * 4, "crss_curve_f": [1e-10] * 4}, "TOY")
    monkeypatch.setattr(
        "pipeline.tier3_simplis._full_cv_from_curve4", lambda vendor: sweep)
    sweeps, source, limitation = _load_cv_sweeps("TOY", {}, {})
    assert source == "curve4_full_arrays" and len(sweeps) == 1
    assert limitation == _CV_SINGLE_BIAS_LIMITATION
    # Backward-compatible accessor: full-array tuple limit stays None and the
    # Vgs=0 CDG axis is kept with an explicit limitation.
    got = _load_cv_sweep("TOY", {}, {})
    assert got[4] == "curve4_full_arrays" and got[5] is None
    axis, _, cdg_axis, bias, cdg_limit = _select_cdg_data(
        sweeps, _primary_cv_sweep(sweeps))
    assert cdg_axis == CDG_AXIS_VGS0 and bias is None
    assert "Vdg=Vds" in cdg_limit
    assert np.allclose(axis, _V)


# ---- 3. measured C-V error and selection gate -------------------------------

def test_cap_error_is_measured_and_recorded():
    v = np.linspace(0.1, 20.0, 60)
    c = 1e-9 * (1.0 + 0.2 / (1.0 + np.exp(-(v - 8.0))))
    fit = _fit_anchored_charge(v, c, 0.005)
    err = _cap_error_pct(fit, fit.pairs)
    assert err is not None and np.isfinite(err) and err >= 0.0
    # A step the charge budget tolerates still shows a large local C error.
    step = np.where(v < 7.0, 1e-9, 6e-9)
    step_fit = _fit_anchored_charge(v, step, 0.05)
    assert _cap_error_pct(step_fit, step_fit.pairs) > CAP_ERROR_BUDGET_PCT

    tables = _fit_charge_tables(
        np.linspace(0.1, 10.0, 20), np.full(20, 1e-9),
        np.full(20, 5e-10), np.full(20, 1e-10), vbus=10.0)
    assert all(t.max_cap_error_pct is not None
               for t in tables.fits.values())
    record = _fit_record(tables.fits["cds"], tables.pairs["cds"])
    assert "max_cap_error_pct" in record
    assert record["cap_error_budget_pct"] == CAP_ERROR_BUDGET_PCT
    assert record["cap_error_tolerance_met"] is not None


def _charge_candidate(vds, cap=None):
    abs_err = {k: 1.0 for k in ("tr_ns", "tf_ns", "e_on_uj", "e_off_uj")}
    if vds < 5.0:  # the better candidate improves every metric strictly
        abs_err = {k: 0.5 for k in abs_err}
    row = {
        "fidelity": {
            "vds_rmse_pct": vds, "id_rmse_pct": vds / 10.0,
            "abs_rel_err_pct": abs_err,
        },
        "dpt_run": {"completed": True, "trace_points": 100},
    }
    if cap is not None:
        row["max_cap_error_pct"] = cap
    return row


def test_selection_uses_cap_error_only_when_both_present():
    tables = _charge_candidate(6.0, cap=5.0)
    worse_cap = _charge_candidate(3.0, cap=50.0)
    assert _select_charge_candidate(
        {"tables": tables, "tier2": worse_cap})[0] == "tables"
    better_cap = _charge_candidate(3.0, cap=1.0)
    assert _select_charge_candidate(
        {"tables": tables, "tier2": better_cap})[0] == "tier2"
    # No measured C-V error -> the historical strict charge-only rule.
    assert _select_charge_candidate(
        {"tables": _charge_candidate(6.0),
         "tier2": _charge_candidate(3.0)})[0] == "tier2"


def test_missing_cap_diagnostic_is_none_and_charge_selection_is_charge_only():
    # Offline run_vendor wiring can hand _build_spec details with no cap_error.
    # The read must degrade to None -- never a fabricated 0, never a KeyError.
    assert _charge_cap_error_from_details({}) is None
    assert _charge_cap_error_from_details({"dummy": 1}) is None
    assert _charge_cap_error_from_details({"cap_error": None}) is None
    assert _charge_cap_error_from_details({"cap_error": {}}) is None
    # A present diagnostic is passed through unchanged (production path).
    assert _charge_cap_error_from_details(
        {"cap_error": {"max_cap_error_pct_all": 7.5}}) == 7.5

    # Feeding that None into the gate keeps its charge-only fallback: tier2
    # still wins on strict fidelity even though no C-V error was measured.
    tables = _charge_candidate(6.0)
    tier2 = _charge_candidate(3.0)
    for row in (tables, tier2):
        row["max_cap_error_pct"] = _charge_cap_error_from_details(
            {"dummy": 1})
    assert all(row["max_cap_error_pct"] is None
               for row in (tables, tier2))
    selection, reason = _select_charge_candidate(
        {"tables": tables, "tier2": tier2})
    assert selection == "tier2"
    assert reason.endswith("charge-only rule")


# ---- 4. physical end segment ------------------------------------------------

def _eval_q(pairs, term, clamp):
    slopes = [(pairs[i][1] - pairs[i - 1][1]) /
              (pairs[i][0] - pairs[i - 1][0])
              for i in range(1, len(pairs))]
    q = pairs[0][1]
    prev = 0.0
    for i, slope in enumerate(slopes, start=1):
        q += (slope - prev) * max(term - pairs[i - 1][0], 0.0)
        prev = slope
    if clamp:
        q += -prev * max(term - pairs[-1][0], 0.0)
    return q


def test_end_segment_clamp_stops_q_above_last_breakpoint():
    pairs = [(0.0, 0.0), (1.0, 1e-9), (3.0, 5e-9)]
    default = _q_expression(pairs, "V(x)")
    clamped = _q_expression(pairs, "V(x)", clamp_above_last=True)
    assert default != clamped
    last_slope = (pairs[-1][1] - pairs[-2][1]) / (pairs[-1][0] - pairs[-2][0])
    assert f"{-last_slope:.6e}" in clamped
    assert np.isclose(_eval_q(pairs, 20.0, True),
                      _eval_q(pairs, pairs[-1][0], True))
    assert _eval_q(pairs, 20.0, False) > _eval_q(pairs, 3.0, False)

    spec = _simplis_spec()
    base = build_simplis_subcircuit(spec, cgs_axis="vgs")
    out = build_simplis_subcircuit(
        spec, cgs_axis="vgs", clamp_charge_above_last=True)
    assert out.isascii()
    base_lines, out_lines = base.splitlines(), out.splitlines()
    changed = [i for i, (a, b) in enumerate(zip(base_lines, out_lines))
               if a != b]
    assert changed
    assert all(("Q={" in base_lines[i] and "Q={" in out_lines[i])
               or base_lines[i].startswith("* Q= chains")
               for i in changed)
    assert "extrapolate linearly" in "\n".join(base_lines)
    assert "clamps the slope to zero" in "\n".join(out_lines)


# ---- 5. temperature / gate-drive metadata contracts -------------------------

def test_declared_metadata_absent_stays_none():
    assert _declared_cv_metadata({}) == (None, None)
    assert _declared_cv_metadata({"simplis": {}}) == (None, None)
    assert _declared_cv_metadata(
        {"simplis": {"temperature_c": 100.0, "vgs_bias_v": 5.0}}
    ) == (100.0, 5.0)
    with pytest.raises(ValueError, match="not numeric"):
        _declared_cv_metadata({"simplis": {"vgs_bias_v": "on"}})


def test_charge_tables_defaults_and_not_measured_header():
    tables = _ChargeTables(fits={}, pairs={}, n_clipped={},
                           cgs_decision=SimpleNamespace())
    assert tables.temperature_c is None
    assert tables.vgs_bias_v is None
    assert tables.cdg_axis == CDG_AXIS_VGS0
    lines = _charge_meta_lines({"cdg_axis": CDG_AXIS_VGS0})
    text = "\n".join(lines)
    assert "vgs_bias_v=not_measured" in text
    assert "temperature_c=not_measured" in text
    assert "max_cap_error_pct=not_measured" in text
    assert all(line.isascii() for line in lines)
    # No metadata -> no added header lines at all.
    assert _charge_meta_lines({}) == []


# ---- 6. turn-off energy decomposition (diagnostic, no correction) -----------

_VBUS = 12.5


def _dpt_waveform(*, ring: bool = True):
    """Synthetic two-edge DPT: 10 A on-state, 20 ns Vds rise at full current.

    ``ring`` adds a zero-mean current oscillation after the 90 % crossing,
    which is what the vendor model's internal network does at turn-off. The
    bench current identity is ``abs``, so that oscillation becomes apparent
    loss without any dissipation.
    """
    t = np.arange(0.0, 201e-9, 0.25e-9)
    vgs = np.where(t < 40e-9, 10.0, np.where(t < 45e-9, 10.0 - (t - 40e-9) / 5e-9 * 10.0, 0.0))
    vds = np.where(t < 30e-9, _VBUS,
                   np.where(t < 32e-9, _VBUS * (1.0 - (t - 30e-9) / 2e-9), 0.12))
    rise = (t - 50e-9) / 1e-9
    vds = np.where(t < 50e-9, vds,
                   np.where(t < 51e-9, 0.12 + (_VBUS - 0.12) * rise, _VBUS))
    signed = np.where(t < 50e-9, 10.0,
                      np.where(t < 51e-9, 10.0 * (1.0 - (t - 50e-9) / 1e-9), 0.0))
    if ring:
        tail = t >= 51e-9
        signed = np.where(
            tail, 1.0 + 2.0 * np.sin(2.0 * np.pi * (t - 51e-9) / 4e-9), signed)
    return t, vds, np.abs(signed), vgs, signed


def _dpt_run(ring: bool = True, *, signed: bool = False) -> DoublePulseRun:
    t, vds, i_abs, vgs, signed_curve = _dpt_waveform(ring=ring)
    return DoublePulseRun(time=t, v_ds=vds, i_d=i_abs, v_gs=vgs,
                          v_bus=_VBUS, i_load=10.0, l_load=1e-6,
                          vgs_on=10.0, wall_clock_s=0.0, t2_on=0.0,
                          i_d_signed=signed_curve if signed else None)


def test_turn_off_decomposition_reproduces_the_scored_e_off():
    run = _dpt_run()
    t, vds, i_abs, vgs, signed = _dpt_waveform()
    out = turn_off_energy_decomposition(
        t, vds, i_abs, vgs, _VBUS, i_d_signed=signed)
    assert out is not None
    assert out["e_off_uj"] == pytest.approx(
        extract_dpt_metrics(run)["e_off_uj"], rel=0.0, abs=0.0)
    # With the signed trace the scored energy IS the signed integral; the
    # decomposition must close on the same window, not on a re-derivation.
    assert out["e_off_signed_uj"] == pytest.approx(
        extract_dpt_metrics(_dpt_run(signed=True))["e_off_uj"],
        rel=0.0, abs=0.0)
    # The integer-segment parts partition that window exactly.
    assert out["parts_sum_uj"] == pytest.approx(
        out["e_off_uj"], rel=0.0, abs=1e-12)
    assert out["parts_sum_uj"] == pytest.approx(
        out["transition_uj"] + out["rise_uj"] + out["tail_uj"],
        rel=0.0, abs=1e-12)
    assert out["tail_uj"] > 0.0 and out["rise_uj"] > 0.0


def test_turn_off_decomposition_exposes_rectified_ringing_tail():
    t, vds, i_abs, vgs, signed = _dpt_waveform()
    out = turn_off_energy_decomposition(
        t, vds, i_abs, vgs, _VBUS, i_d_signed=signed)
    # The signed tail oscillates about zero: abs() turns it into loss.
    assert out["rectified_tail_uj"] > 0.0
    assert out["rectified_tail_uj"] == pytest.approx(
        out["tail_uj"] - out["tail_signed_uj"], rel=0.0, abs=1e-12)
    assert out["tail_negative_fraction"] > 0.0
    assert out["e_off_signed_uj"] < out["e_off_uj"]
    # The signed segments partition the signed total exactly as well.
    assert (out["transition_signed_uj"] + out["rise_signed_uj"]
            + out["tail_signed_uj"]) == pytest.approx(
                out["e_off_signed_uj"], rel=0.0, abs=1e-12)
    # No ringing -> no rectification, and the keys stay measured.
    flat = _dpt_waveform(ring=False)
    no_ring = turn_off_energy_decomposition(
        flat[0], flat[1], flat[2], flat[3], _VBUS, i_d_signed=flat[4])
    assert no_ring["rectified_tail_uj"] == pytest.approx(0.0, abs=1e-12)
    assert no_ring["tail_negative_fraction"] == 0.0


def test_turn_off_decomposition_reports_none_without_signed_current():
    t, vds, i_abs, vgs, _ = _dpt_waveform()
    out = turn_off_energy_decomposition(t, vds, i_abs, vgs, _VBUS)
    assert out["tail_negative_fraction"] is None
    for key in ("rectified_tail_uj", "e_off_signed_uj", "tail_signed_uj"):
        assert key not in out
    assert out["e_off_uj"] > 0.0


def test_turn_off_decomposition_returns_none_when_the_window_is_absent():
    t, vds, i_abs, vgs, _ = _dpt_waveform(ring=False)
    # Vds never reaches 90 % of the bus: there is no turn-off window.
    assert turn_off_energy_decomposition(
        t, vds * 0.5, i_abs, vgs, _VBUS) is None
    # A gate that never drove is not a zero-energy turn-off either.
    assert turn_off_energy_decomposition(
        t, vds, i_abs, np.zeros_like(vgs), _VBUS) is None
    with pytest.raises(ValueError, match="matched waveform arrays"):
        turn_off_energy_decomposition(t[:-1], vds, i_abs, vgs, _VBUS)
    with pytest.raises(ValueError, match="sample grid"):
        turn_off_energy_decomposition(
            t, vds, i_abs, vgs, _VBUS, i_d_signed=np.ones(3))


def test_energy_tail_constant_is_pinned_to_the_scored_metric(monkeypatch):
    run = _dpt_run()
    t, vds, i_abs, vgs, _ = _dpt_waveform()
    monkeypatch.setattr(sm, "_ENERGY_TAIL_S", 150e-9)
    drifted = turn_off_energy_decomposition(t, vds, i_abs, vgs, _VBUS)
    assert drifted["e_off_uj"] != pytest.approx(
        extract_dpt_metrics(run)["e_off_uj"], rel=1e-6)


def test_measure_candidate_attaches_the_measured_decomposition(monkeypatch):
    t, vds, i_abs, vgs, signed = _dpt_waveform()
    run = _dpt_run(signed=True)
    monkeypatch.setattr(
        sm, "_run_emitted_dpt",
        lambda lib, vendor: (run, "Total iterations: 10\n", "tier3"))
    monkeypatch.setattr(
        sm, "_waveform_errors",
        lambda r, traj: {"vds_rel_rmse_pct": 1.0, "id_rel_rmse_pct": 1.0})
    traj = SimpleNamespace(
        time=np.arange(0.0, 201e-9, 0.25e-9),
        input=(vgs, vds), output=(signed,), pin_roles=_PIN_ROLES)
    row = sm._measure_candidate(
        sm.ROOT / "reports" / "models" / "TOY.lib", "TOY", traj,
        {"vbus": _VBUS, "t_end": t[-1]}, {}, {})
    decomposition = row["dpt_run"]["energy_decomposition"]
    assert decomposition is not None
    # The scored energy comes from the signed trace, and the split proves it.
    assert decomposition["e_off_signed_uj"] == pytest.approx(
        row["dpt_comparison"]["e_off_uj"]["tier3"], rel=0.0, abs=1e-12)
    assert decomposition["e_off_uj"] > decomposition["e_off_signed_uj"]
    assert decomposition["rectified_tail_uj"] > 0.0
    assert decomposition["tail_negative_fraction"] > 0.0
    # A run without the signed trace degrades to abs and invents no sign.
    monkeypatch.setattr(
        sm, "_run_emitted_dpt",
        lambda lib, vendor: (_dpt_run(signed=False),
                             "Total iterations: 10\n", "tier3"))
    bare = sm._measure_candidate(
        sm.ROOT / "reports" / "models" / "TOY2.lib", "TOY", traj,
        {"vbus": _VBUS, "t_end": t[-1]}, {}, {})["dpt_run"][
            "energy_decomposition"]
    assert bare["tail_negative_fraction"] is None
    assert "rectified_tail_uj" not in bare
