"""Simulator log diagnostics for the spicelib/LTspice backend.

Single source of truth for diagnostic capture/classification after the
platform migration. There is deliberately NO ngspice/PySpice machinery
left in this module:

- Capture: out-of-process LTspice writes a .log file per run; callers feed
  it via parse_log_file()/parse_log_text(). No C-level stdout hooks exist
  because there is no in-process simulator.
- Classification: regex registry keyed to real LTspice 26.x phrasing,
  validated against actual failing runs (floating-node circuit).
- Deduplication: SHA-256 over masked text so repeated messages collapse
  into one record with a repetition count.
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
# Registry: known LTspice .log message patterns.
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

    # --- topology warnings (validated against real LTspice 26.0.2 logs) ---
    DiagnosticRule(
        _c(r"Node \S+ is floating"),
        "warning",
        "floating_node",
        "A node has no DC path to ground (VM-07 class defect when it is a "
        "thermal or source pin). The solver may gmin its way to a wrong "
        "operating point instead of reporting failure.",
    ),

    # --- numeric warnings ---
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
        _c(r"Direct Newton iteration (succeeded|failed)"),
        "info",
        "op_iteration_banner",
        "Operating-point iteration outcome banner.",
    ),
    DiagnosticRule(
        _c(r"(Total elapsed time|Nperiods requested)"),
        "info",
        "run_stats",
        "Run statistics footer.",
    ),
)

# Masking applied before hashing so equal-class messages hash identically:
_MASK_PATTERNS = (
    (re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?"), "<f>"),
    # letter+digit tokens (NODEx17): \b\d+\b cannot see digits inside words
    (re.compile(r"\b[A-Za-z_]+\d+\b"), "<tok>"),
    (re.compile(r"\b\d+\b"), "<n>"),
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


class SimulatorLogDiagnostics:
    """Classify + deduplicate simulator log lines."""

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

    # -- entry points --------------------------------------------------------

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

    # -- reporting -----------------------------------------------------------

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


# Module-level singleton.
_default_diagnostics: SimulatorLogDiagnostics | None = None


def get_diagnostics() -> SimulatorLogDiagnostics:
    """Return the process-wide diagnostics collector (creating if needed)."""
    global _default_diagnostics
    if _default_diagnostics is None:
        _default_diagnostics = SimulatorLogDiagnostics()
    return _default_diagnostics


def report_and_reset(min_severity: str = "warning") -> str | None:
    """Report accumulated diagnostics and clear them.

    Returns the report string, or None when nothing at/above the threshold
    was recorded.
    """
    global _default_diagnostics
    if _default_diagnostics is None:
        return None
    text = _default_diagnostics.report(min_severity=min_severity)
    had_content = len(_default_diagnostics.records) > 0
    _default_diagnostics.reset()
    return text if had_content else None
