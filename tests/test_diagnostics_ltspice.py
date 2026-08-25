"""Unit tests for common/diagnostics_ltspice.py (T1.1)."""

import sys
sys.path.insert(0, ".")

from common.diagnostics_ltspice import (
    LTSpiceLogDiagnostics, get_ltspice_diagnostics, LTSPICE_DIAGNOSTICS,
)


def test_singular_matrix_dedup():
    d = LTSpiceLogDiagnostics()
    for i in range(200):
        d.ingest(f"WARNING: Node NODEx{i % 4} is floating - singular "
                 f"matrix at step {i * 1.5}")
    recs = [r for r in d.records.values() if r.rule_name == "singular_matrix"]
    assert len(recs) == 1
    assert recs[0].count == 200
    assert len(recs[0].samples) <= LTSpiceLogDiagnostics.MAX_SAMPLES_PER_RECORD


def test_classification_severities():
    d = LTSpiceLogDiagnostics()
    d.ingest("ERROR: Singular matrix: check node net010")
    d.ingest("Fatal Error: Timestep too small; V1 pulse fallback")
    d.ingest("Unknown subcircuit called: epc2216_v0")
    d.ingest("Error on line 12 : xq1 drain gate 0 epc2216_v0")
    d.ingest("LTspice 26.0.2 for Windows")
    d.ingest("Total elapsed time: 1.234 seconds")

    by_name = {}
    for r in d.records.values():
        by_name.setdefault(r.rule_name, []).append(r)

    assert "singular_matrix" in by_name
    assert by_name["singular_matrix"][0].severity == "error"
    assert "timestep_too_small" in by_name
    assert "parse_error" in by_name
    assert "netlist_line_error" in by_name
    assert "version_banner" in by_name
    assert by_name["version_banner"][0].severity == "info"


def test_log_text_ingestion_and_report():
    d = LTSpiceLogDiagnostics()
    text = ("LTspice 26.0.2 for Windows\n"
            "Circuit: smoke_test\n"
            "ERROR: Singular matrix\n"
            "WARNING: unknown parameter foo clamped\n")
    n = d.parse_log_text(text)
    assert n == 4
    report = d.report(min_severity="warning")
    assert "singular_matrix" in report
    d.reset()
    assert len(d.records) == 0


def test_singleton_contract():
    a = get_ltspice_diagnostics()
    b = get_ltspice_diagnostics()
    assert a is b


def test_floating_node_real_log_phrasing():
    """Validated against a real LTspice 26.0.2 log (T1.1 fix 3)."""
    d = LTSpiceLogDiagnostics()
    d.ingest("WARNING: Node out2 is floating.")
    d.ingest("WARNING: Node mid3 is floating.")
    d.ingest("Direct Newton iteration succeeded in finding operating point.")
    recs = {r.rule_name: r for r in d.records.values()}
    assert "floating_node" in recs
    assert recs["floating_node"].count == 2
    assert recs["floating_node"].severity == "warning"
    assert "op_iteration_banner" in recs


def test_report_and_reset_alias():
    from common.diagnostics_ltspice import report_and_reset_ltspice
    d = get_ltspice_diagnostics()
    d.reset()
    d.ingest("ERROR: Singular matrix")
    text = report_and_reset_ltspice(min_severity="warning")
    assert text and "singular_matrix" in text
    assert len(d.records) == 0
