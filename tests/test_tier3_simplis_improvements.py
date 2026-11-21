"""Offline tests for the bias-aware SIMPLIS Tier-3 charge improvements.

No LTspice: the capacitance bench is driven by a stub AC result and the
pipeline helpers are exercised on synthetic persisted data. Nothing here
introduces a multi-bias or temperature axis that was not measured/persisted.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

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
    CGD_CAP_TARGET_ENV,
    CGD_CAP_TARGET_PCT,
    _CV_SINGLE_BIAS_LIMITATION,
    _cap_error_pct,
    _cgd_cap_target_pct,
    _charge_cap_error_from_details,
    _cv_sweep_from_extracted,
    _cv_sweeps_from_extracted,
    _declared_cv_metadata,
    _extend_cdg_for_cap_target,
    _fit_anchored_charge,
    _fit_charge_tables,
    _fit_record,
    _load_cv_sweep,
    _load_cv_sweeps,
    _primary_cv_sweep,
    _select_cdg_data,
    _select_charge_candidate,
    _ChargeTables,
)
from simulation.testbenches.capacitance import CapacitanceTestbench


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
