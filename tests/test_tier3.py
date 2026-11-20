"""Unit tests for the paper-defined parts of the Tier-3 TPWL terminal model."""
import numpy as np

from config.yaml_io import load_input
from models import tpwl_netlist_builder as emitter
from pipeline.tier3 import _log_iterations
from approximation.tpwl_deim import (
    WEIGHT_EXPONENT,
    TPWLTerminalModel,
    deim_indices,
    paper_weights,
)


def test_deim_indices_are_distinct_and_interpolatory():
    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    idx = deim_indices(basis)
    assert idx.shape == (3,)
    assert len(set(idx.tolist())) == 3
    # DEIM interpolatory projector reproduces selected rows exactly
    P = np.eye(8)[:, idx]
    proj = basis @ np.linalg.solve(P.T @ basis, P.T)
    v = rng.normal(size=8)
    assert np.allclose((proj @ v)[idx], v[idx])


def test_paper_weights_follow_rewienski_white_formula():
    assert WEIGHT_EXPONENT == 25.0
    anchors = np.array([[0.0, 0.0], [1.0, 0.0]])
    mid = paper_weights(anchors, np.array([[0.5], [0.0]]))
    assert np.allclose(mid[:, 0], [0.5, 0.5])
    near = paper_weights(anchors, np.array([[0.1], [0.0]]))
    assert near[0, 0] > near[1, 0]
    assert np.isclose(near[:, 0].sum(), 1.0)
    # far from every anchor the weights stay normalised and finite
    far = paper_weights(anchors, np.array([[50.0], [0.0]]))
    assert np.all(np.isfinite(far[:, 0]))
    assert np.isclose(far[:, 0].sum(), 1.0)


def test_single_cell_model_predicts_its_affine_channel():
    model = TPWLTerminalModel(
        vendor="t", subcircuit="T", pin_roles=["drain", "gate", "source"],
        anchors=np.array([[0.0, 0.0]]),
        v_ref=np.zeros(2), v_span=np.ones(2),
        channel=np.array([[1.0, 2.0, 3.0]]),
        charge_d=np.zeros((1, 2)),
        charge_g=np.zeros((1, 3)), edge_window=3)
    t = np.linspace(0.0, 1e-6, 11)
    u = np.vstack([np.linspace(0.0, 1.0, 11), np.zeros(11)])
    id_pred, qg_pred = model.predict(t, u)
    assert np.allclose(id_pred, 1.0 + 2.0 * u[0])
    assert np.allclose(qg_pred, 0.0)


def _toy_model() -> TPWLTerminalModel:
    """Offline model with a real vendor's config-backed curve files."""
    return TPWLTerminalModel(
        vendor="csd16412q5a", subcircuit="TOY",
        pin_roles=["drain", "gate", "source"],
        anchors=np.array([[0.0, 0.0], [0.5, 0.5]]),
        v_ref=np.array([4.0, 8.0]), v_span=np.array([10.0, 10.0]),
        channel=np.array([[10.0, 1.0, 0.5], [20.0, 2.0, 1.0]]),
        charge_d=np.array([[1e-9, 1e-10], [2e-9, 2e-10]]),
        charge_g=np.zeros((2, 3)), edge_window=3,
        coss_v=np.array([0.1, 5.0, 20.0]),
        coss_c=np.array([1e-9, 5e-10, 2e-10]),
        v_block=1.88, v_width=0.05)


def test_grid_axes_are_deterministic_and_cover_dpt():
    model = _toy_model()
    g1, d1 = emitter.grid_axes(model)
    g2, d2 = emitter.grid_axes(model)
    assert np.array_equal(g1, g2)
    assert np.array_equal(d1, d2)
    assert g1[0] <= 0.0 and g1[-1] >= 10.0
    assert d1[0] <= 0.0 and d1[-1] >= 12.5
    assert np.all(np.diff(g1) > 0) and np.all(np.diff(d1) > 0)


def test_emission_is_ascii_and_uses_flat_tables_without_weights():
    text = emitter.emit_subcircuit(_toy_model(), name="TOY_T3")
    assert text.isascii()
    assert text.startswith(".subckt TOY_T3 d g s\n")
    assert text.rstrip().endswith(".ends")
    assert "G_CH d s VALUE={(0.5*(" in text
    assert "table(V(d,s)," in text
    assert "V(d,s)/(V(d,s)+2.000000000e-01)" in text
    assert "QOSS d s Q={table(limit(V(d,s)" in text
    # No B-source intermediates, no per-cell weight nodes, no C_QD.
    assert "\nB" not in text and "C_QD" not in text
    assert "exp(" not in text and "sqrt(" in text


def test_emitted_channel_clamps_negative_drain_voltage():
    model = _toy_model()
    u = np.vstack([np.array([10.0, 10.0, 10.0]),
                   np.array([-0.1, 0.0, 10.0])])
    current = emitter.emitted_channel(model, u)
    # TI's measured soft clamp approaches zero below Vds = 0 but is C1.
    assert current[0] < 1e-2
    assert current[1] < 1e-2
    assert current[2] > 0.0


def test_emission_policies_are_vendor_specific_and_ascii():
    toy = _toy_model()
    ir = _toy_model()
    ir.vendor = "irfs7730pbf"
    epc = _toy_model()
    epc.vendor = "epc2216"
    text = emitter.emit_subcircuit(ir, name="IR_T3")
    assert text.isascii()
    assert "QOSS d s Q={table(" not in text
    assert "exp(-" in text and "sqrt(" in text
    assert "max(0," in text  # IRFS keeps the hard clamp
    assert emitter.emission_policy(toy).soft_clamp
    assert emitter.emission_policy(ir).high_vds_saturation
    assert emitter.emission_policy(epc).vgs_points == 19


def test_high_vds_saturation_extends_grid_and_floors_channel():
    ir = _toy_model()
    ir.vendor = "irfs7730pbf"
    g, d = emitter.grid_axes(ir)
    cfg = load_input(ir.vendor)["dpt"]
    assert d[-1] >= 1.10 * float(cfg["vbus"])
    table = emitter.grid_channel(ir, g, d)
    tail = table[:, np.searchsorted(d, 10.0):]
    assert np.all(np.isfinite(tail)) and np.all(tail >= 0.0)
    assert np.allclose(tail, tail[:, :1])


def test_emitted_static_rmse_matches_channel_prediction():
    from approximation.static_curves import static_samples
    model = _toy_model()
    u, ref, _ = static_samples(model.vendor)
    pred = emitter.emitted_channel(model, u)
    expected = float(np.sqrt(np.mean((pred - ref) ** 2)) /
                     (np.ptp(ref) or 1.0))
    assert np.isclose(emitter.emitted_static_rmse(model), expected)


def test_log_iterations_parses_completed_and_failed_logs():
    assert _log_iterations("Total iterations: 394001\n") == 394001
    assert _log_iterations("Simulation Failed: Iteration limit reached") is None


def test_saturation_scale_override_is_vendor_specific():
    assert emitter.saturation_scale(_toy_model()) == 0.2
    epc = _toy_model()
    epc.vendor = "epc2216"
    assert emitter.saturation_scale(epc) == 0.5


# ---- SIMPLIS Level-2 charge-table variant (offline, no LTspice) ------------
from common.types import PWLModelSpec
from models.simplis_netlist_builder import (
    build_simplis_subcircuit,
    cgs_linearity_decision,
    charge_ramp_terms,
)


def _simplis_spec(*, linear_cgs: bool = False) -> PWLModelSpec:
    """Small deterministic spec carrying the three (V,Q) charge tables."""
    v = [0.1, 1.0, 10.0, 25.0]
    if linear_cgs:
        cgs_q = [0.0, 2e-9, 20e-9, 50e-9]  # exact chord
    else:
        cgs_q = [0.0, 3e-9, 12e-9, 26e-9]
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


def test_simplis_charge_ramp_terms_are_monotonic():
    """Delta slopes integrate back to a non-decreasing Q at every point."""
    spec = _simplis_spec()
    for name, pairs in (
        ("CGS", list(zip(spec.q_iss_vs_v, spec.q_iss_charge_c))),
        ("CDS", list(zip(spec.q_oss_vs_v, spec.q_oss_charge_c))),
        ("CDG", list(zip(spec.q_rss_vs_v, spec.q_rss_charge_c))),
    ):
        terms = list(charge_ramp_terms(pairs))
        assert len(terms) == len(pairs) - 1, name
        slopes = [d for _, d in terms]
        assert all(d >= 0 or i > 0 for i, d in enumerate(slopes)), name
        rebuilt = [pairs[0][1]]
        for (v_prev, delta), (v_i, q_i) in zip(terms, pairs[1:]):
            prev_q = rebuilt[-1]
            prev_slope = 0.0 if len(rebuilt) == 1 else (
                (pairs[len(rebuilt) - 1][1] - pairs[len(rebuilt) - 2][1])
                / (pairs[len(rebuilt) - 1][0] - pairs[len(rebuilt) - 2][0]))
            rebuilt.append(prev_q + (prev_slope + delta) * (v_i - v_prev))
            assert np.isclose(rebuilt[-1], q_i), name
        assert np.all(np.diff(rebuilt) >= 0), name


def test_simplis_emission_is_ascii_and_ports_match_tier2():
    text = build_simplis_subcircuit(_simplis_spec())
    assert text.isascii()
    lines = text.splitlines()
    assert ".SUBCKT TOY_PIECESPICE_T3_simplis d g s" in lines
    assert ".ENDS TOY_PIECESPICE_T3_simplis" in lines
    for expected in ("RG g gi ", "RD d di ", "RS s ss ",
                     "G_CH di ss VALUE", "D_BODY ss di ",
                     "CGS gi ss ", "CDS di ss Q=", "CDG di gi Q="):
        assert any(line.startswith(expected) for line in lines), expected
    for removed in ("C_QOSS ", "C_ISS ", "C_RSS "):
        assert not any(line.startswith(removed) for line in lines), removed
    charges = [line.split()[0] for line in lines
               if line.split() and line.split()[0] in ("CGS", "CDS", "CDG")]
    assert charges == ["CGS", "CDS", "CDG"]


def test_simplis_cgs_linearity_decision_is_deterministic():
    v = [0.1, 1.0, 10.0, 25.0]
    linear = cgs_linearity_decision(v, [0.0, 2e-9, 20e-9, 50e-9])
    nonlinear = cgs_linearity_decision(v, [0.0, 3e-9, 12e-9, 26e-9])
    assert linear.is_linear and not nonlinear.is_linear
    assert linear == cgs_linearity_decision(v, [0.0, 2e-9, 20e-9, 50e-9])
    assert nonlinear.chord_deviation > linear.chord_deviation
    assert linear.as_dict()["mode"] == "linear"


def test_simplis_linear_cgs_emits_plain_capacitor():
    text = build_simplis_subcircuit(_simplis_spec(linear_cgs=True))
    cgs_lines = [x for x in text.splitlines() if x.startswith("CGS ")]
    assert len(cgs_lines) == 1 and "Q=" not in cgs_lines[0]
    text_pwl = build_simplis_subcircuit(_simplis_spec())
    cgs_pwl = [x for x in text_pwl.splitlines() if x.startswith("CGS ")]
    assert len(cgs_pwl) == 1 and "Q={" in cgs_pwl[0]


def test_simplis_emits_rgs_and_truthful_extrapolation_comment():
    text = build_simplis_subcircuit(_simplis_spec())
    lines = text.splitlines()
    assert len([x for x in lines if x.startswith("RGS gi ss 1e7")]) == 1
    assert "lookups end-clamp" not in " ".join(lines)
    assert any("extrapolate linearly with the last segment" in x
               for x in lines)
    tier2 = build_simplis_subcircuit(_simplis_spec(), charge_mode="tier2")
    t2_lines = tier2.splitlines()
    assert any(x.startswith("C_QOSS ") for x in t2_lines)
    assert any(x.startswith("C_ISS ") for x in t2_lines)
    assert any(x.startswith("C_RSS ") for x in t2_lines)
    assert not any(x.split() and x.split()[0] in ("CGS", "CDS", "CDG")
                   for x in t2_lines)
    assert len([x for x in t2_lines if x.startswith("RGS gi ss 1e7")]) == 1
    try:
        build_simplis_subcircuit(_simplis_spec(), charge_mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown charge_mode must fail")


from pipeline.tier3_simplis import (  # noqa: E402
    _cap_curve_to_vbus,
    _fidelity_metrics,
    _fit_anchored_charge,
    _on_state_breakpoint,
    _select_charge_candidate,
)


def test_simplis_charge_fit_anchors_at_zero_and_meets_budget():
    v = np.linspace(0.1, 25.0, 40)
    c = 1e-9 * (1 + 3 * np.exp(-((v - 5) / 0.5) ** 2)
                + 3 * np.exp(-((v - 12) / 0.4) ** 2))
    fit = _fit_anchored_charge(v, c, 0.005)
    assert fit.pairs[0] == (0.0, 0.0)
    assert np.isclose(fit.q_full[1] / fit.v_full[1], c[0])
    assert np.isclose(fit.pairs[-1][0], v[-1])
    assert fit.max_abs_error_c <= fit.error_budget_c
    assert len(fit.pairs) > 5  # the old 5-point cap would have truncated


def test_simplis_forced_on_state_breakpoint_replaces_lower_points():
    v = np.linspace(0.1, 12.5, 30)
    c = np.full_like(v, 2e-10)
    bp = _on_state_breakpoint(
        [0.0, 10.0, 10.0, 10.0],
        [12.5, 0.5, 0.1725, 0.1], vgs_on=10.0, vbus=12.5)
    assert np.isclose(bp["measured_v"], 0.5)
    assert not bp["clamped_to_vbus"]
    fit = _fit_anchored_charge(
        v, c, 0.005, forced_v=bp["used_v"],
        forbid_below=bp["used_v"])
    volts = [p[0] for p in fit.pairs]
    assert any(np.isclose(x, bp["used_v"]) for x in volts)
    assert min(volts[1:]) >= bp["used_v"] - 1e-12
    assert fit.max_abs_error_c <= fit.error_budget_c

    clamped = _on_state_breakpoint(
        [0.0, 10.0], [38.9, 38.9], vgs_on=10.0, vbus=38.0)
    assert clamped["clamped_to_vbus"] and clamped["used_v"] == 38.0
    grid, curves = _cap_curve_to_vbus(
        np.array([0.1, 10.0, 60.0]), (np.ones(3),), 38.0)
    assert np.isclose(grid[-1], 38.0) and curves[0][-1] == 1.0


def _candidate(vds, id_, tr, tf, e_on, e_off) -> dict:
    comparison = {
        key: {"tier3_rel_err_pct": err}
        for key, err in (("tr_ns", tr), ("tf_ns", tf),
                         ("e_on_uj", e_on), ("e_off_uj", e_off))}
    return {"fidelity": _fidelity_metrics(
        {"vds_rel_rmse_pct": vds, "id_rel_rmse_pct": id_}, comparison),
        "dpt_run": {"completed": True}}


def test_simplis_candidate_selection_requires_strict_all_metric_gain():
    base = _candidate(6.0, 0.05, 29.0, 47.0, 51.0, 1.3)
    better = _candidate(3.8, 0.045, 20.0, 35.2, 29.1, 1.25)
    cands = {"tables": base, "tier2": better}
    assert _select_charge_candidate(cands)[0] == "tier2"
    tied = {**better, "fidelity": {
        **better["fidelity"],
        "abs_rel_err_pct": {**better["fidelity"]["abs_rel_err_pct"],
                            "e_off_uj": 1.3}}}
    assert _select_charge_candidate(
        {"tables": base, "tier2": tied})[0] == "tables"
    failed = _candidate(None, 0.045, 35.6, 35.2, 29.1, 1.25)
    assert _select_charge_candidate(
        {"tables": base, "tier2": failed})[0] == "tables"

from pipeline.tier3_simplis import _select_channel_candidate  # noqa: E402
from models.hybrid_channel import fit_hybrid_channel  # noqa: E402
from types import SimpleNamespace  # noqa: E402


_HYBRID_NOTE_PREFIXES = (
    "* Hybrid channel:", "* fit meta:", "* fit stop criterion:",
    "* measured limitation:",
)


def test_simplis_hybrid_fit_replaces_only_gch_and_keeps_provenance():
    spec = _simplis_spec()
    fit = fit_hybrid_channel(spec.vgs_values, spec.breakpoints_vds,
                             spec.breakpoints_id, rel_tolerance=0.005)
    base = build_simplis_subcircuit(spec, charge_mode="tier2")
    hybrid = build_simplis_subcircuit(
        spec, charge_mode="tier2", channel_fit=fit)
    assert base.isascii() and hybrid.isascii()

    base_lines = base.splitlines()
    hybrid_lines = hybrid.splitlines()
    assert sum(line.startswith(_HYBRID_NOTE_PREFIXES)
               for line in hybrid_lines) == len(_HYBRID_NOTE_PREFIXES)
    body = [line for line in hybrid_lines
            if not line.startswith(_HYBRID_NOTE_PREFIXES)]
    assert len(body) == len(base_lines)
    changed = [(a, b) for a, b in zip(base_lines, body) if a != b]
    assert len(changed) == 1, changed
    old, new = changed[0]
    assert old.startswith("G_CH di ss VALUE")
    assert new == f"G_CH di ss VALUE={fit.expression}"
    for prefix in ("RG ", "RD ", "RS ", "D_BODY ",
                   "C_QOSS ", "C_ISS ", "C_RSS "):
        assert [x for x in base_lines if x.startswith(prefix)] == [
            x for x in body if x.startswith(prefix)]
    header = "\n".join(hybrid_lines[:10])
    assert "hinge_c" in header
    assert "local://hybrid_channel_feasibility.md" in header
    assert str(fit.meta["n_knees"]) in header


def test_simplis_hybrid_rejects_non_ascii_expression():
    bad = SimpleNamespace(expression="V(gi,ss)*\u00e9", meta={},
                          max_abs_fit_error_a=0.0)
    try:
        build_simplis_subcircuit(_simplis_spec(), channel_fit=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("non-ASCII hybrid expression must fail")


def test_simplis_channel_selection_rejects_regressing_hybrid_dpt():
    base_tables = _candidate(6.0, 0.05, 29.0, 47.0, 51.0, 1.3)
    base_tier2 = _candidate(5.0, 0.04, 25.0, 40.0, 45.0, 1.2)
    # Measured feasibility outcome: static fit improves, DPT Vds regresses.
    cands = {
        "tables": base_tables, "tier2": base_tier2,
        "hybrid_tables": _candidate(11.20, 0.17, 30.0, 48.0, 52.0, 1.35),
        "hybrid_tier2": _candidate(7.10, 0.10, 26.0, 41.0, 46.0, 1.25),
    }
    choice, reason, evidence = _select_channel_candidate(cands)
    assert choice == "tier2"
    assert "vds_rmse_pct" in reason and "hybrid_tables" in reason
    for mode in ("hybrid_tables", "hybrid_tier2"):
        row = evidence["pairs"][mode]["metrics"]["vds_rmse_pct"]
        assert row["tier2_channel"] < row["hybrid"]
        assert not row["strict_gain"]
    # Charge selection is unchanged by the extra channel candidates.
    assert _select_charge_candidate(cands)[0] == "tier2"
    # A missing hybrid candidate can never be selected.
    assert _select_channel_candidate(
        {"tables": base_tables, "tier2": base_tier2})[0] == "tier2"

    # Even a strict fidelity win is rejected if adaptive speed was not measured.
    winning = {
        **cands,
        "hybrid_tables": _candidate(3.0, 0.02, 10.0, 20.0, 20.0, 1.0),
        "hybrid_tier2": _candidate(2.5, 0.02, 9.0, 19.0, 19.0, 1.0),
    }
    choice, reason, _ = _select_channel_candidate(winning)
    assert choice == "tier2" and "adaptive" in reason

    # Positive control: fidelity gain plus measured, no-worse adaptive speed.
    speed = {
        "tables": {"completed": True, "solver_speedup": 1.0,
                   "wall_speedup": 1.0},
        "tier2": {"completed": True, "solver_speedup": 1.1,
                  "wall_speedup": 1.05},
        "hybrid_tables": {"completed": True, "solver_speedup": 1.2,
                          "wall_speedup": 1.1},
        "hybrid_tier2": {"completed": True, "solver_speedup": 1.3,
                         "wall_speedup": 1.2},
    }
    assert _select_channel_candidate(
        winning, adaptive_results=speed)[0] == "hybrid"
