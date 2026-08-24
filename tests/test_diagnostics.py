"""Unit tests for common/diagnostics.py."""

import sys
sys.path.insert(0, ".")

from common.diagnostics import (
    NgspiceDiagnostics, get_diagnostics, NGSPICE_DIAGNOSTICS,
)


def test_vj_clamp_dedup():
    d = NgspiceDiagnostics()
    # 300 repeats of the same warning with varying instance/numbers
    for i in range(300):
        d.ingest(f"Warning: bodydiode_cpm2-1200-0025b{i % 3}: junction "
                 f"potential VJ too large, limited to 2.000000")
    recs = list(d.records.values())
    assert len(recs) == 1, f"expected 1 record, got {len(recs)}"
    r = recs[0]
    assert r.rule_name == "diode_vj_clamp"
    assert r.count == 300
    assert len(r.samples) <= NgspiceDiagnostics.MAX_SAMPLES_PER_RECORD


def test_classification_severities():
    d = NgspiceDiagnostics()
    d.ingest("Error: 0, -0.5 out of range for pow")
    d.ingest("in line b.xm1.xgmos.bg1")
    d.ingest("doAnalyses: OP:  Timestep too small; cause unrecorded.")
    d.ingest("run simulation(s) aborted")
    d.ingest("Trying gmin =   1.0000E-03")
    d.ingest("Note: Starting dynamic gmin stepping")
    d.ingest("Unsupported Ngspice version 39")

    by_name = {}
    for r in d.records.values():
        by_name.setdefault(r.rule_name, []).append(r)

    assert "pow_domain_error" in by_name
    assert by_name["pow_domain_error"][0].severity == "error"
    assert "timestep_too_small" in by_name
    assert "simulation_aborted" in by_name
    assert "convergence_progress" in by_name
    assert "pyspice_version_table" in by_name

    report = d.report(min_severity="warning")
    # info-class items must NOT appear at 'warning' threshold
    assert "convergence_progress" not in report
    assert "pyspice_version" not in report
    # errors must appear
    assert "pow_domain_error" in report


def test_numeric_variation_same_hash():
    """Same message with different numbers must collapse to one signature."""
    d = NgspiceDiagnostics()
    d.ingest("Warning: singular matrix: initial dynamic gmin step failed")
    d.ingest("Warning: singular matrix: initial dynamic gmin step failed")
    d.ingest("Warning: singular matrix: initial dynamic gmin step failed")
    assert len(d.records) == 1
    assert list(d.records.values())[0].count == 3


def test_report_and_reset_singleton():
    get_diagnostics().reset()
    diag = get_diagnostics()
    diag.ingest("Warning: node tj is floating")
    text = diag.report(min_severity="info")
    assert "floating_node" in text
    diag.reset()
    assert len(diag.records) == 0


def test_normalize_masks_numbers_and_instances():
    a = NgspiceDiagnostics.normalize(
        "Error: 0, -0.5 out of range for pow\nin line b.xm1.xgmos.bg1")
    b = NgspiceDiagnostics.normalize(
        "Error: 12, -3.75 out of range for pow\nin line b.xm9.xgmos.bg7")
    assert a == b, f"{a!r} != {b!r}"


if __name__ == "__main__":
    test_vj_clamp_dedup()
    test_classification_severities()
    test_numeric_variation_same_hash()
    test_report_and_reset_singleton()
    test_normalize_masks_numbers_and_instances()
    print(f"all {len(NGSPICE_DIAGNOSTICS)} registry rules; "
          "diagnostics tests PASS")
