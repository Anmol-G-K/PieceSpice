"""T3 fixed-topology scalar tuning tests (spec 4.3 / 4.5 / 6).

Every case is offline and deterministic. The WOL objective fixtures are
analytic functions of theta, so each asserted score is computed from WOL Eq.1
in this file -- no number is copied from a report or a simulation. No LTspice
binary, vendor library, generated artifact or live simulation is used, so no
``pytest.skip`` gate is needed and ``PIECESPICE_LIVE`` is irrelevant here.
"""

from __future__ import annotations

import math

import pytest

from common.types import (
    PWLModelSpec,
    UNIFIED_TUNING_IDENTITY,
    UNIFIED_TUNING_KEYS,
    UNIFIED_TUNING_PROVENANCE,
)
from pipeline.tuning import (
    _INCOMPLETE_PENALTY,
    ConditionResult,
    TuningResult,
    _check_bounds,
    _normalise_bounds,
    _normalise_theta,
    _normalise_weights,
    _relative_error,
    apply_scalars,
    derive_o_gd_bound,
    tune_scalars,
)


def _close(value):
    """Farad-scale values need a relative tolerance, not a 1e-12 floor."""
    return pytest.approx(value, rel=1e-12, abs=0.0)


# Fixture curve-4 scalars; the O_GD bracket is derived from these, never a
# hidden literal (spec 4.3 / UNIFIED_TUNING_PROVENANCE["offset_bound"]).
_C_RSS_LIN = 7.5e-10
_Q_RSS_V_GRID = (0.0, 2.0, 4.0)


def _o_gd_cap():
    """One full Q_RSS at the top of the capacitance sweep: c_rss*Vds_max."""
    return max(_C_RSS_LIN * max(_Q_RSS_V_GRID), _C_RSS_LIN)


def _bounds():
    """Per-field brackets; A_* mirror the one-octave project decision,
    O_GD comes from the curve-4 charge budget, not a magic farad literal."""
    return {
        "A_GD": (0.5, 2.0),
        "O_GD": (0.0, _o_gd_cap()),
        "A_GS": (0.5, 2.0),
        "A_RGI": (0.5, 2.0),
    }


def _spec_kwargs(**meta_overrides):
    metadata = {
        "r_gate": 1.5,
        "r_source": 1e-3,
        "r_drain": 1e-3,
        "c_iss_lin": 3e-9,
        "c_oss_lin": 5e-9,
        "c_rss_lin": 7.5e-10,
        "theta_bounds": _bounds(),
    }
    metadata.update(meta_overrides)
    return dict(
        device_name="T3Dev",
        tier=2,
        breakpoints_vds=[0.0, 1.0, 2.0, 5.0],
        breakpoints_id=[[0.0, 1.0, 2.0, 4.0], [0.0, 2.0, 4.0, 8.0]],
        vgs_values=[5.0, 10.0],
        q_oss_vs_v=[0.0, 10.0, 20.0],
        q_oss_charge_c=[0.0, 1e-8, 1.5e-8],
        q_iss_vs_v=[0.0, 2.0, 4.0],
        q_iss_charge_c=[0.0, 2e-9, 3e-9],
        q_rss_vs_v=[0.0, 2.0, 4.0],
        q_rss_charge_c=[0.0, 1e-9, 1.5e-9],
        rds_on=0.005,
        v_th=2.5,
        body_diode_vsd=[0.4, 0.7],
        body_diode_isd=[0.01, 1.0],
        metadata=metadata,
    )


@pytest.fixture
def base_spec():
    return PWLModelSpec(**_spec_kwargs())


def _identity_condition(name="c", held_out=False):
    return {
        "name": name,
        "emp": {"m": 1.0},
        "held_out": held_out,
        "simulate": lambda theta: ConditionResult(
            metrics={"m": 1.0}, completed=True),
    }


def _wol_j(emp, sim):
    """WOL Eq.1 on evaluable terms only (non-finite and zero refs skipped)."""
    total = 0.0
    for key, reference in emp.items():
        ref, value = float(reference), float(sim[key])
        if not (math.isfinite(ref) and math.isfinite(value)) or ref == 0.0:
            continue
        total += abs(ref - value) / abs(ref)
    return total


# ---------------------------------------------------------------------------
# Named-field order and bounds
# ---------------------------------------------------------------------------

def test_named_field_order_identity_and_wol_hints_are_not_defaults(base_spec):
    assert UNIFIED_TUNING_KEYS == ("A_GD", "O_GD", "A_GS", "A_RGI")
    assert set(UNIFIED_TUNING_IDENTITY) == set(UNIFIED_TUNING_KEYS)
    assert [UNIFIED_TUNING_IDENTITY[k] for k in UNIFIED_TUNING_KEYS] == [
        1.0, 0.0, 1.0, 1.0,
    ]
    # WOL Table 3 values are source scale hints only, never emitter defaults.
    wol_final = {"A_GD": 1.47, "O_GD": 5.8e-12, "A_GS": 0.68, "A_RGI": 0.84}
    assert UNIFIED_TUNING_IDENTITY != wol_final
    for key, literal in (("A_GD", "1.47"), ("O_GD", "5.8 pF"),
                         ("A_GS", "0.68"), ("A_RGI", "0.84")):
        text = UNIFIED_TUNING_PROVENANCE[f"hint_{key}"]
        assert literal in text and "scale hint only" in text
    assert "Wolfspeed" in UNIFIED_TUNING_PROVENANCE["source"]
    assert "C_DS" in UNIFIED_TUNING_PROVENANCE["frozen"]
    assert "theta" not in base_spec.metadata
    assert "urom_tuning_theta" not in base_spec.metadata


def test_named_theta_mapping_is_canonicalised_in_declared_order():
    theta = {"A_RGI": 0.9, "A_GS": 0.8, "O_GD": 2e-12, "A_GD": 1.2}
    out = _normalise_theta(theta)
    assert list(out) == list(UNIFIED_TUNING_KEYS)
    assert out == {"A_GD": 1.2, "O_GD": 2e-12, "A_GS": 0.8, "A_RGI": 0.9}


@pytest.mark.parametrize("bad,exc", [
    ((1.0, 0.0, 1.0, 1.0), TypeError),
    ([1.0, 0.0, 1.0, 1.0], TypeError),
    ({"A_GD": 1.0, "O_GD": 0.0, "A_GS": 1.0, "EXTRA": 1.0}, ValueError),
    ({"A_GD": 1.0, "O_GD": 0.0, "A_GS": 1.0}, ValueError),
    ({"A_GD": True, "O_GD": 0.0, "A_GS": 1.0, "A_RGI": 1.0}, ValueError),
    ({"A_GD": 1.0, "O_GD": float("nan"), "A_GS": 1.0, "A_RGI": 1.0},
     ValueError),
    ({"A_GD": 1.0, "O_GD": -1e-12, "A_GS": 1.0, "A_RGI": 1.0}, ValueError),
    ({"A_GD": 0.0, "O_GD": 0.0, "A_GS": 1.0, "A_RGI": 1.0}, ValueError),
    ({"A_GD": 1.0, "O_GD": 0.0, "A_GS": -0.5, "A_RGI": 1.0}, ValueError),
])
def test_theta_rejects_positional_unknown_and_out_of_domain(bad, exc):
    with pytest.raises(exc):
        _normalise_theta(bad)


def test_public_apis_reject_positional_theta(base_spec):
    with pytest.raises(TypeError, match="named mapping"):
        apply_scalars(base_spec, (1.0, 0.0, 1.0, 1.0))
    with pytest.raises(TypeError, match="named mapping"):
        tune_scalars("x", method="trap/default",
                     conditions=[_identity_condition()],
                     bounds=(0.5, 0.0, 0.5, 0.5), max_evals=2)
    unknown = {"A_GD": 1.0, "O_GD": 0.0, "A_GS": 1.0, "A_RGI": 1.0,
               "C_DS": 1.0}
    with pytest.raises(ValueError, match="unknown fields"):
        apply_scalars(base_spec, unknown)
    with pytest.raises(ValueError, match="unknown fields"):
        tune_scalars("x", method="trap/default",
                     conditions=[_identity_condition()],
                     bounds={**_bounds(), "C_DS": (0.5, 2.0)}, max_evals=2)


@pytest.mark.parametrize("bad,exc", [
    ((0.5, 0.0, 0.5, 0.5), TypeError),
    ({**_bounds(), "C_DS": (0.5, 2.0)}, ValueError),
    ({k: v for k, v in _bounds().items() if k != "A_RGI"}, ValueError),
    ({**_bounds(), "A_GD": 1.0}, ValueError),
    ({**_bounds(), "A_GD": (2.0, 0.5)}, ValueError),
    ({**_bounds(), "A_GS": (0.5, float("nan"))}, ValueError),
    ({**_bounds(), "O_GD": (-1e-12, 1e-11)}, ValueError),
])
def test_bounds_are_named_finite_ordered_and_offset_nonnegative(bad, exc):
    with pytest.raises(exc):
        _normalise_bounds(bad)


def test_bounds_are_enforced_by_both_public_apis(base_spec):
    theta = dict(UNIFIED_TUNING_IDENTITY)
    theta["A_GD"] = 2.5
    with pytest.raises(ValueError, match="A_GD"):
        apply_scalars(base_spec, theta)
    with pytest.raises(ValueError, match="A_GD"):
        _check_bounds(theta, _bounds(), "probe")
    tight = _bounds()
    tight["A_GD"] = (1.1, 2.0)  # identity baseline is out of bracket
    with pytest.raises(ValueError):
        tune_scalars("x", method="trap/default",
                     conditions=[_identity_condition()], bounds=tight,
                     max_evals=2)


@pytest.mark.parametrize("meta", [
    {"theta": (1.0, 0.0, 1.0, 1.0)},
    {"theta": {"A_GD": 1.0, "O_GD": 0.0, "A_GS": 1.0, "BAD": 1.0}},
    {"theta": {"A_GD": 1.0, "O_GD": -1.0, "A_GS": 1.0, "A_RGI": 1.0}},
    {"theta_bounds": {"C_DS": (0.5, 2.0)}},
    {"theta_bounds": {"A_GD": 2.0}},
    {"theta_bounds": {"O_GD": (-1.0, 1.0)}},
])
def test_spec_metadata_rejects_positional_and_unknown_tuning_fields(meta):
    with pytest.raises(ValueError):
        PWLModelSpec(**_spec_kwargs(**meta))


# ---------------------------------------------------------------------------
# apply_scalars: validated mapping and frozen structure
# ---------------------------------------------------------------------------

def test_apply_scalars_updates_only_named_gate_side_scalars(base_spec):
    theta = {"A_GD": 1.2, "O_GD": 2e-9, "A_GS": 0.8, "A_RGI": 0.9}
    before = base_spec.model_dump()
    tuned = apply_scalars(base_spec, theta)
    assert tuned is not base_spec
    assert base_spec.model_dump() == before  # never mutate the caller's spec

    # WOL Eqs.2-4 on Q tables: O_GD*V keeps the V=0 anchor at zero.
    assert tuned.q_rss_charge_c == pytest.approx(
        [0.0, 1.2 * (1e-9 + 2e-9 * 2.0), 1.2 * (1.5e-9 + 2e-9 * 4.0)],
        rel=1e-12, abs=0.0)
    assert tuned.q_iss_charge_c == pytest.approx(
        [0.0, 0.8 * 2e-9, 0.8 * 3e-9], rel=1e-12, abs=0.0)
    # Linear fallbacks mirror the same transform.
    assert tuned.metadata["c_rss_lin"] == _close(1.2 * (7.5e-10 + 2e-9))
    assert tuned.metadata["c_iss_lin"] == _close(0.8 * 3e-9)
    assert tuned.metadata["r_gate"] == _close(0.9 * 1.5)
    assert tuned.metadata["r_gate"] > 0.0
    assert tuned.metadata["theta"] == theta
    assert tuned.metadata["urom_tuning_theta"] == theta
    assert "Qoss frozen" in tuned.metadata["urom_provenance"]["applied_theta"]


def test_identity_theta_is_exact_mathematical_noop(base_spec):
    tuned = apply_scalars(base_spec, dict(UNIFIED_TUNING_IDENTITY))
    assert tuned.q_rss_charge_c == base_spec.q_rss_charge_c
    assert tuned.q_iss_charge_c == base_spec.q_iss_charge_c
    assert tuned.metadata["r_gate"] == base_spec.metadata["r_gate"]
    assert tuned.metadata["c_iss_lin"] == base_spec.metadata["c_iss_lin"]
    assert tuned.metadata["c_rss_lin"] == base_spec.metadata["c_rss_lin"]


@pytest.mark.parametrize("missing", ["r_gate", "c_iss_lin", "c_rss_lin"])
def test_no_hidden_scalar_default_in_tuning_layer(missing):
    kwargs = _spec_kwargs()
    del kwargs["metadata"][missing]
    spec = PWLModelSpec(**kwargs)
    with pytest.raises(ValueError, match=missing):
        apply_scalars(spec, dict(UNIFIED_TUNING_IDENTITY))


@pytest.mark.parametrize("bad_r_gate", [0.0, -1.5, float("nan")])
def test_r_gate_must_be_positive(bad_r_gate):
    kwargs = _spec_kwargs()
    kwargs["metadata"]["r_gate"] = bad_r_gate
    spec = PWLModelSpec(**kwargs)
    with pytest.raises(ValueError, match="r_gate"):
        apply_scalars(spec, dict(UNIFIED_TUNING_IDENTITY))


def test_apply_scalars_rejects_theta_outside_declared_bounds(base_spec):
    theta = dict(UNIFIED_TUNING_IDENTITY)
    theta["O_GD"] = _o_gd_cap() * 1.5  # above the curve-derived O_GD cap
    with pytest.raises(ValueError, match="O_GD"):
        apply_scalars(base_spec, theta)


def test_o_gd_bound_is_curve_derived_not_a_literal_default(base_spec):
    """A drifted theta_bounds literal must not cap a valid curve-derived O_GD."""
    cap = derive_o_gd_bound(base_spec)
    assert cap == _close(_C_RSS_LIN * max(_Q_RSS_V_GRID))
    assert _bounds()["O_GD"] == (0.0, cap)
    assert cap > 2e-9  # the offset exercised by the apply_scalars tests

    drifted = dict(base_spec.metadata)
    drifted["theta_bounds"] = {**_bounds(), "O_GD": (0.0, 1e-11)}
    spec = base_spec.model_copy(update={"metadata": drifted})
    tuned = apply_scalars(spec, {
        "A_GD": 1.0, "O_GD": 2e-9, "A_GS": 1.0, "A_RGI": 1.0,
    })
    assert tuned.q_rss_charge_c[-1] > spec.q_rss_charge_c[-1]


def test_o_gd_bound_keeps_config_cap_when_qrss_grid_is_empty():
    """Producer specs carry the config cap in metadata, not an emitted grid."""
    kwargs = _spec_kwargs()
    kwargs["q_rss_vs_v"] = []
    kwargs["q_rss_charge_c"] = []
    kwargs["metadata"]["theta_bounds"]["O_GD"] = (0.0, 4.5e-8)
    spec = PWLModelSpec(**kwargs)
    assert derive_o_gd_bound(spec) == _close(4.5e-8)


# ---------------------------------------------------------------------------
# Charge-table monotonicity, Q(0)=0 and the Qoss/channel freeze
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("o_gd", [0.0, 2e-9, 1e-11])
def test_charge_tables_stay_monotone_and_zero_anchored(base_spec, o_gd):
    theta = {"A_GD": 1.3, "O_GD": o_gd, "A_GS": 0.7, "A_RGI": 1.0}
    tuned = apply_scalars(base_spec, theta)
    for voltage, charge in ((tuned.q_rss_vs_v, tuned.q_rss_charge_c),
                            (tuned.q_iss_vs_v, tuned.q_iss_charge_c)):
        assert voltage[0] == 0.0 and charge[0] == 0.0
        assert all(b > a for a, b in zip(voltage, voltage[1:]))
        assert all(b >= a for a, b in zip(charge, charge[1:]))
        assert all(math.isfinite(float(q)) for q in charge)
        assert all(float(q) >= 0.0 for q in charge)


@pytest.mark.parametrize("field,bad,token", [
    ("q_rss_vs_v", [0.0, 4.0, 2.0], "ascending"),
    ("q_rss_charge_c", [0.0, 2e-9, 1e-9], "monotone"),
    ("q_rss_charge_c", [0.0, -1e-9, 2e-9], ">= 0"),
    ("q_iss_charge_c", [0.0, float("nan"), 3e-9], "finite"),
    ("q_iss_charge_c", [0.0, 2e-9], "length"),
])
def test_invalid_charge_tables_are_rejected(base_spec, field, bad, token):
    spec = base_spec.model_copy(update={field: bad})
    with pytest.raises(ValueError, match=token):
        apply_scalars(spec, dict(UNIFIED_TUNING_IDENTITY))


def test_qoss_channel_and_static_fields_are_frozen(base_spec):
    tuned = apply_scalars(base_spec, {
        "A_GD": 1.2, "O_GD": 2e-9, "A_GS": 0.8, "A_RGI": 0.9,
    })
    for field in ("device_name", "tier", "breakpoints_vds", "breakpoints_id",
                  "vgs_values", "q_oss_vs_v", "q_oss_charge_c", "rds_on",
                  "v_th", "body_diode_vsd", "body_diode_isd", "v_plateau",
                  "q_rss_vs_v", "q_iss_vs_v"):
        assert getattr(tuned, field) == getattr(base_spec, field)
    for key in ("r_source", "r_drain", "c_oss_lin"):
        assert tuned.metadata[key] == base_spec.metadata[key]


# ---------------------------------------------------------------------------
# WOL Eq.1 objective on separable synthetic data
# ---------------------------------------------------------------------------

def test_tune_scalars_recovers_target_on_separable_synthetic_data():
    bounds = _bounds()
    target = {"A_GD": 1.3, "O_GD": 0.3 * _o_gd_cap(), "A_GS": 0.7,
              "A_RGI": 1.1}
    seen = []

    def simulator(label):
        def simulate(theta):
            seen.append((label, dict(theta)))
            return ConditionResult(
                metrics={k: float(theta[k]) for k in UNIFIED_TUNING_KEYS},
                completed=True, points=100, iterations=50, t_last=14e-6,
            )
        return simulate

    conditions = [
        {"name": "fit", "emp": dict(target), "simulate": simulator("fit")},
        {"name": "holdout", "held_out": True, "emp": dict(target),
         "simulate": simulator("holdout")},
    ]
    budget = 400
    result = tune_scalars("synthetic", method="trap/default",
                          conditions=conditions, bounds=bounds,
                          max_evals=budget)

    assert isinstance(result, TuningResult)
    assert seen[0][1] == dict(UNIFIED_TUNING_IDENTITY)  # identity start
    assert all(set(theta) == set(UNIFIED_TUNING_KEYS) for _, theta in seen)
    assert all(lo - 1e-12 <= theta[k] <= hi + 1e-12
               for _, theta in seen for k, (lo, hi) in bounds.items())
    for key in UNIFIED_TUNING_KEYS:
        assert result.theta[key] == pytest.approx(target[key], rel=0.02,
                                                  abs=0.0)

    # The reported score is exactly WOL Eq.1 evaluated at the reported theta.
    expected = _wol_j(target, result.theta)
    assert result.objective_fit == pytest.approx(expected, rel=1e-9, abs=0.0)
    assert result.objective_held_out == pytest.approx(expected, rel=1e-9,
                                                      abs=0.0)
    baseline = _wol_j(target, UNIFIED_TUNING_IDENTITY)
    assert result.baseline_fit == pytest.approx(baseline, rel=1e-12, abs=0.0)
    assert result.baseline_held_out == pytest.approx(baseline, rel=1e-12,
                                                     abs=0.0)
    assert result.objective_held_out < result.baseline_held_out
    assert result.verdict == "accept"
    assert result.golden_completion == {"fit": True, "holdout": True}
    assert result.held_out_scores["holdout"] == pytest.approx(
        result.objective_held_out, rel=1e-12, abs=0.0)

    assert result.vendor == "synthetic" and result.method == "trap/default"
    assert result.bounds == bounds
    assert "Powell" in result.optimizer
    assert 2 <= result.evaluations <= budget
    payload = result.to_dict()
    assert payload["theta"] == result.theta
    assert "Wolfspeed" in payload["provenance"]["source"]


def test_nonfinite_and_zero_reference_terms_are_skipped():
    emp = {"good": 2.0, "zero": 0.0, "nan": float("nan"),
           "inf": float("inf"), "text": "n/a"}
    sim = ConditionResult(
        metrics={"good": 1.0, "zero": 5.0, "nan": 5.0, "inf": 5.0,
                 "text": 5.0},
        completed=True,
    )
    result = tune_scalars(
        "skip", method="gear-both",
        conditions=[{"name": "c", "emp": emp, "simulate": lambda theta: sim}],
        bounds=_bounds(), max_evals=1,
    )
    assert result.objective_fit == _close(0.5)  # only |2-1|/2 survives
    details = result.condition_results["c"]
    assert details["metrics_used"] == 1
    assert set(details["skipped_metrics"]) == {"zero", "nan", "inf", "text"}
    assert result.method == "gear-both"
    assert result.verdict == "infeasible"  # no held-out arm to certify a gain

    assert _relative_error(2.0, 1.0) == _close(0.5)
    assert _relative_error(0.0, 1.0) is None
    assert _relative_error(float("nan"), 1.0) is None
    assert _relative_error(float("inf"), 1.0) is None
    assert _relative_error("bad", 1.0) is None


def test_declared_metric_weights_scale_wol_terms_exactly():
    weights = {"metrics": {"A_GD": 2.0}}
    result = tune_scalars(
        "weighted", method="trap/default",
        conditions=[{
            "name": "c",
            "emp": {"A_GD": 1.0, "O_GD": 1.0},
            "simulate": lambda theta: ConditionResult(
                metrics={"A_GD": theta["A_GD"], "O_GD": theta["O_GD"]},
                completed=True),
        }],
        bounds=_bounds(), weights=weights, max_evals=1,
    )
    # Identity gives |1-1|/1 = 0 for A_GD and |1-0|/1 = 1 for O_GD;
    # the declared metric weight scales only the A_GD term.
    assert result.objective_fit == _close(2.0 * 0.0 + 1.0)
    assert result.weights == {"A_GD": 2.0}

    metric, condition = _normalise_weights(
        {"metrics": {"m": 2.0}, "conditions": {"c": 3.0}})
    assert metric == {"m": 2.0} and condition == {"c": 3.0}
    with pytest.raises(TypeError):
        _normalise_weights([1.0, 2.0])
    with pytest.raises(ValueError):
        _normalise_weights({"metrics": {"m": -1.0}})
    with pytest.raises(ValueError):
        _normalise_weights({"conditions": {"c": -1.0}})


def test_incomplete_candidate_is_penalised_not_reported_as_fast():
    def aborted(theta):
        return ConditionResult(metrics={"m": 1.0}, completed=False,
                               points=1, iterations=0, t_last=1e-9)

    conditions = [
        {"name": "fit", "emp": {"m": 1.0}, "simulate": aborted},
        {"name": "holdout", "held_out": True, "emp": {"m": 1.0},
         "simulate": aborted},
    ]
    result = tune_scalars("abort", method="trap/default", conditions=conditions,
                          bounds=_bounds(), max_evals=2)
    assert result.golden_completion == {"fit": True, "holdout": True}
    assert result.objective_fit >= _INCOMPLETE_PENALTY
    assert result.verdict == "reject"
    for name in ("fit", "holdout"):
        detail = result.condition_results[name]
        assert detail["completed"] is False and detail["score"] is None
        assert detail["points"] == 1 and detail["iterations"] == 0
        assert detail["t_last"] == 1e-9


def test_noncompleting_golden_condition_is_flagged_and_not_scored():
    result = tune_scalars(
        "vm10", method="trap/default",
        conditions=[{"name": "c", "emp": {"m": 1.0}, "completed": False,
                     "simulate": lambda theta: ConditionResult(
                         metrics={"m": 1.0}, completed=True)}],
        bounds=_bounds(), max_evals=1,
    )
    assert result.golden_completion == {"c": False}
    assert result.condition_results["c"]["skipped"] is True
    assert result.objective_fit == 0.0
    assert result.verdict == "infeasible"


def test_method_label_is_declared_recorded_and_validated():
    for label in ("trap/default", "gear-both"):
        result = tune_scalars(label, method=label,
                              conditions=[_identity_condition()],
                              bounds=_bounds(), max_evals=1)
        assert result.method == label
        assert label in result.to_dict()["provenance"]["method_label"]
    with pytest.raises(ValueError, match="method"):
        tune_scalars("bad", method="gear", conditions=[_identity_condition()],
                     bounds=_bounds(), max_evals=1)


@pytest.mark.parametrize("condition,exc", [
    ("not-a-mapping", TypeError),
    ({"simulate": lambda theta: None}, ValueError),
    ({"emp": {"m": 1.0}}, ValueError),
    ({"emp": {"m": 1.0}, "simulate": 5}, ValueError),
])
def test_conditions_require_named_emp_and_callable_simulator(condition, exc):
    with pytest.raises(exc):
        tune_scalars("bad", method="trap/default", conditions=[condition],
                     bounds=_bounds(), max_evals=2)


def test_budget_and_empty_condition_set_are_rejected():
    with pytest.raises(ValueError, match="max_evals"):
        tune_scalars("bad", method="trap/default",
                     conditions=[_identity_condition()], bounds=_bounds(),
                     max_evals=0)
    with pytest.raises(ValueError, match="at least one calibration condition"):
        tune_scalars("bad", method="trap/default", conditions=[],
                     bounds=_bounds(), max_evals=2)
