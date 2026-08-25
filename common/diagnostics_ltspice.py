"""T1.1 — LTspice .log diagnostics layer.

Replaces the ngspice console-capture machinery in common/diagnostics.py
with an LTspice log-file classifier. The SHA-256 masking/dedup scheme and
the record/report API are retained verbatim (they operate on text lines,
independent of simulator).

REMOVED (ngspice/PySpice-only, obsolete out-of-process):
  - ConsoleQuiet (fd dup2 redirect)
  - install_notebook_quiet_console / install_console_suppressor
    (send_char callback hooks)
  - drain_shared_buffer (shared-lib buffer polling)
  - NgspiceLogDedupFilter + install_log_dedup (PySpice logging bridge)

ADDED:
  - LTSPICE_DIAGNOSTICS registry (LTspice log phrasing)
  - parse_log_file() -> ingests a .log into the collector
  - classify_log_text() for in-memory text

This module coexists with the ngspice classes during migration; the
ngspice-only pieces at the bottom are marked LEGACY and will be deleted
at T6.3 cleanup.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LTspice diagnostic registry: known .log message patterns.
# Severity/explanation semantics mirror the legacy ngspice registry.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticRule:
    """One known simulator-log message pattern."""

    pattern: re.Pattern
    severity: str          # 'warning' | 'error' | 'info'
    name: str              # stable machine id
    explanation: str


def _c(text: str) -> re.Pattern:
    return re.compile(text, re.IGNORECASE)


LTSPICE_DIAGNOSTICS: tuple[DiagnosticRule, ...] = (
    # --- solver failures ---
    DiagnosticRule(
        _c(r"singular matrix"),
        "error",
        "singular_matrix",
        "Matrix singular at this bias point: some node has no DC path or "
        "a zero-impedance loop. The analysis result is invalid.",
    ),
    DiagnosticRule(
        _c(r"Timestep too small"),
        "error",
        "timestep_too_small",
        "Transient time-step collapsed to zero (stiff transition the "
        "solver could not converge through). Results after this point are "
        "unreliable.",
    ),
    DiagnosticRule(
        _c(r"(simulation aborted|analysis aborted)"),
        "error",
        "simulation_aborted",
        "The solver aborted the requested analysis entirely.",
    ),
    DiagnosticRule(
        _c(r"Unknown (subcircuit|model|parameter)"),
        "error",
        "parse_error",
        "Netlist referenced an undefined subcircuit/model/parameter - "
        "the simulation did not run correctly.",
    ),
    DiagnosticRule(
        _c(r"Error on line"),
        "error",
        "netlist_line_error",
        "Netlist syntax/semantic error reported by LTspice's parser.",
    ),

    # --- topology warnings (verified against real LTspice 26.0.2 log) ---
    DiagnosticRule(
        _c(r"Node \S+ is floating"),
        "warning",
        "floating_node",
        "A node has no DC path to ground (VM-07 class defect when it is a "
        "thermal or source pin). The solver may gmin its way to a wrong "
        "operating point.",
    ),
    DiagnosticRule(
        _c(r"Direct Newton iteration (succeeded|failed)"),
        "info",
        "op_iteration_banner",
        "Operating-point iteration outcome banner.",
    ),

    # --- parameter clamps & numeric warnings ---
    DiagnosticRule(
        _c(r"(too large|too small).*(clamped|limited|reduced)"),
        "warning",
        "param_clamp",
        "A model parameter was outside the allowed range and was silently "
        "adjusted by the solver.",
    ),
    DiagnosticRule(
        _c(r"Power loss .* exceeds"),
        "warning",
        "power_warning",
        "Dissipated power exceeded a device limit declared in the netlist.",
    ),

    # --- informational banners ---
    DiagnosticRule(
        _c(r"^LTspice \d+"),
        "info",
        "version_banner",
        "LTspice version banner.",
    ),
    DiagnosticRule(
        _c(r"(Total elapsed time|Nperiods requested)"),
        "info",
        "run_stats",
        "Run statistics footer.",
    ),
)

# Masking rules shared with the legacy scheme: volatile tokens replaced so
# equal-class messages hash identically.
_MASK_PATTERNS = (
    (re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?"), "<f>"),
    # letter+digit mixed tokens (NODEx17, xm12, vgs34): \b\d+\b cannot see
    # the digit inside a word, so collapse these explicitly.
    (re.compile(r"\b[A-Za-z_]+\d+\b"), "<tok>"),
    (re.compile(r"\b\d+\b"), "<n>"),
    (re.compile(r"\b\S*(?:bodydiode|_cpm2|xm\d)\S*", re.IGNORECASE), "<dev>"),
    (re.compile(r"[A-Z]:\\[^\s:]+", re.IGNORECASE), "<path>"),
)


@dataclass
class DiagnosticRecord:
    """Aggregated record for one unique diagnostic signature."""

    rule_name: str
    severity: str
    explanation: str
    first_raw: str
    count: int = 0
    samples: list = field(default_factory=list)


class LTSpiceLogDiagnostics:
    """Classify + deduplicate LTspice .log lines.

    Same ingest/report/reset contract as the legacy ngspice collector so
    callers can migrate mechanically.
    """

    MAX_SAMPLES_PER_RECORD = 3

    def __init__(self):
        self.records: dict[str, DiagnosticRecord] = {}
        self.unclassified_count = 0
        self._lock = threading.Lock()

    @staticmethod
    def normalize(raw: str) -> str:
        text = raw.strip().lower()
        for pat, repl in _MASK_PATTERNS:
            text = pat.sub(repl, text)
        return text

    def classify(self, raw: str):
        normalized = self.normalize(raw)
        for rule in LTSPICE_DIAGNOSTICS:
            if rule.pattern.search(raw):
                return rule, normalized
        return None, normalized

    def ingest(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return
        rule, key_text = self.classify(line)
        digest = hashlib.sha256(key_text.encode()).hexdigest()[:16]

        with self._lock:
            if rule is None:
                if re.match(r"^(note|\d|\s*$)", line.lower()):
                    return
                self.unclassified_count += 1
                rec = self.records.get(digest)
                if rec is None:
                    rec = DiagnosticRecord("unclassified", "info", "", line)
                    self.records[digest] = rec
                rec.count += 1
                return

            rec = self.records.get(digest)
            if rec is None:
                rec = DiagnosticRecord(rule.name, rule.severity,
                                       rule.explanation, line)
                self.records[digest] = rec
            rec.count += 1
            if (len(rec.samples) < self.MAX_SAMPLES_PER_RECORD
                    and line not in rec.samples):
                rec.samples.append(line)

    # -- file/text entry points ---------------------------------------------

    def parse_log_text(self, text: str) -> int:
        """Ingest every line of an in-memory log. Returns lines ingested."""
        n = 0
        for line in text.splitlines():
            self.ingest(line)
            n += 1
        return n

    def parse_log_file(self, log_path: Path | str) -> int:
        """Ingest an LTspice .log file from disk."""
        path = Path(log_path)
        if not path.exists():
            _LOGGER.debug("log file missing: %s", path)
            return 0
        return self.parse_log_text(
            path.read_text(encoding="utf-8", errors="ignore"))

    # -- reporting ------------------------------------------------------------

    def report(self, min_severity: str = "info") -> str:
        order = {"error": 0, "warning": 1, "info": 2}
        threshold = order[min_severity]
        selected = [r for r in self.records.values()
                    if order.get(r.severity, 2) <= threshold]
        selected.sort(key=lambda r: (order.get(r.severity, 2), -r.count))

        blocks: list[str] = []
        totals: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
        for rec in selected:
            totals[rec.severity] = totals.get(rec.severity, 0) + rec.count
            head = f"[{rec.severity.upper()}] {rec.rule_name} x{rec.count}"
            body = f"    {rec.explanation}" if rec.explanation else ""
            sample = f'    e.g. "{rec.first_raw[:120]}"'
            extra = ("    (%d distinct variants seen)" % len(rec.samples)
                     if len(rec.samples) > 1 else "")
            blocks.append("\n".join(x for x in (head, body, sample, extra)
                                    if x))
        summary = (f"ltspice diagnostics: {totals['error']} error(s), "
                   f"{totals['warning']} warning(s), {totals['info']} info "
                   f"({len(self.records)} unique signatures)")
        return "\n".join([summary] + [""] + blocks) if blocks else summary

    def reset(self) -> None:
        with self._lock:
            self.records.clear()
            self.unclassified_count = 0


# Module-level singleton (mirrors get_diagnostics contract).
_default_ltspice_diagnostics: LTSpiceLogDiagnostics | None = None


def get_ltspice_diagnostics() -> LTSpiceLogDiagnostics:
    global _default_ltspice_diagnostics
    if _default_ltspice_diagnostics is None:
        _default_ltspice_diagnostics = LTSpiceLogDiagnostics()
    return _default_ltspice_diagnostics


def report_and_reset(min_severity: str = "warning") -> str | None:
    """Report accumulated diagnostics and clear them.

    Named identically to the legacy ngspice helper so callers migrate by
    changing only the import. (The old module dies at T6.3, freeing the
    name permanently.)
    """
    global _default_ltspice_diagnostics
    if _default_ltspice_diagnostics is None:
        return None
    text = _default_ltspice_diagnostics.report(min_severity=min_severity)
    had_content = len(_default_ltspice_diagnostics.records) > 0
    _default_ltspice_diagnostics.reset()
    return text if had_content else None


def report_and_reset_ltspice(min_severity: str = "warning") -> str | None:
    """Back-compat alias of :func:`report_and_reset`."""
    return report_and_reset(min_severity)



# ===========================================================================
# LEGACY NGSPICE SECTION — still imported by the not-yet-migrated PySpice
# code path (engine.py, common/__init__.py). Deleted at T6.3 cleanup once
# the spicelib migration completes. Do NOT extend anything below this line.
# ===========================================================================
