"""ngspice stderr diagnostics: capture, classify, and deduplicate.

Vendor behavioral models emit large volumes of repetitive ngspice warnings
(e.g. ``Warning: bodydiode_cpm2-1200-0025b: junction potential VJ too large,
limited to 2.000000`` — printed once per Newton iteration, hundreds of times
per sweep). This module:

1. CAPTURES the ngspice shared-library stdout/stderr streams.
2. CLASSIFIES each line against a registry of known warning/error patterns
   (NGSPICE_DIAGNOSTICS below) with severity and a human explanation.
3. DEDUPLICATES by content hash (SHA-256 over the *normalized* message,
   where model-instance numbers and floating-point values are masked), so a
   warning repeated 500 times is reported once with a repetition count.

Usage:
    from common.diagnostics import NgspiceDiagnostics

    diag = NgspiceDiagnostics()
    diag.attach()          # start capturing ngspice output
    ... run simulations ...
    diag.detach()          # stop capturing
    diag.report()          # log one line per unique diagnostic + counts
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostic registry: every known ngspice message pattern we care about.
# Each entry maps a regex -> (severity, short name, explanation).
# Unmatched lines fall through as 'other' at info level.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticRule:
    """One known ngspice message pattern."""

    pattern: re.Pattern
    severity: str          # 'warning' | 'error' | 'info'
    name: str              # stable machine id for the class of message
    explanation: str


def _c(text: str) -> re.Pattern:
    return re.compile(text, re.IGNORECASE)


NGSPICE_DIAGNOSTICS: tuple[DiagnosticRule, ...] = (
    # --- device-model parameter clamps (vendor model limitations) ---
    DiagnosticRule(
        _c(r"warning:\s*\S*(bodydiode|diode)\S*: junction potential \S+ too "
           r"large, limited to ([\d.]+)"),
        "warning",
        "diode_vj_clamp",
        "A body-diode .MODEL declares VJ above ngspice's 2 V internal limit; "
        "ngspice silently clamps it. Vendor-model parameter out of range - "
        "harmless for DC extraction but the diode knee is softer than the "
        "vendor intended.",
    ),
    DiagnosticRule(
        _c(r"warning:\s*\S+: (.*) too large, limited to ([\d.]+)"),
        "warning",
        "model_param_clamp",
        "A device-model parameter exceeded ngspice's allowed range and was "
        "clamped (generic form).",
    ),

    # --- convergence machinery (expected during hard OPs) ---
    DiagnosticRule(
        _c(r"(trying gmin =|starting dynamic gmin stepping|"
           r"true gmin stepping|source stepping|transient op( started| failed)|"
           r"further gmin increment|last gmin step failed|"
           r"dynamic gmin stepping failed|true gmin stepping failed|"
           r"source stepping failed|supplies reduced to)"),
        "info",
        "convergence_progress",
        "Solver fallback ladder progress messages (gmin/source stepping). "
        "Expected for behavioral models with extreme dynamic range; only an "
        "issue when followed by a hard abort.",
    ),
    DiagnosticRule(
        _c(r"warning: (?:further|last) gmin increment"),
        "info",
        "convergence_progress",
        "See convergence_progress.",
    ),
    DiagnosticRule(
        _c(r"warning: singular matrix: initial dynamic gmin step failed"),
        "warning",
        "singular_matrix",
        "The matrix was singular at the initial operating point; ngspice "
        "retries via dynamic gmin stepping. Usually recoverable.",
    ),
    DiagnosticRule(
        _c(r"doanalyses:.*timestep too small"),
        "error",
        "timestep_too_small",
        "Operating point / transient failed: time-step collapsed to zero. "
        "Extraction for this bias point is unreliable.",
    ),
    DiagnosticRule(
        _c(r"run simulation\(s\) aborted"),
        "error",
        "simulation_aborted",
        "ngspice aborted the requested analysis entirely.",
    ),
    DiagnosticRule(
        _c(r"warning: node (\S+) is floating"),
        "warning",
        "floating_node",
        "A circuit node has no DC path to ground - check X-instance wiring "
        "(VM-07 class defect when it is a thermal or source pin).",
    ),

    # --- expression evaluation ---
    DiagnosticRule(
        _c(r"error: .*out of range for pow"),
        "error",
        "pow_domain_error",
        "Fractional power evaluated with an out-of-domain base (ngspice >= 36 "
        "hard-errors where 34 silently returned garbage - see VM-08). The "
        "containing bias point's results are invalid.",
    ),
    DiagnosticRule(
        _c(r"error: unknown subckt|unknown parameter|no such function"),
        "error",
        "parse_error",
        "Netlist parse/evaluation error - the simulation did not run "
        "correctly.",
    ),

    # --- PySpice bridge noise (not ngspice itself) ---
    DiagnosticRule(
        _c(r"unsupported ngspice version"),
        "info",
        "pyspice_version_table",
        "PySpice's internal version table predates this ngspice build. "
        "Cosmetic - the API works regardless.",
    ),
    DiagnosticRule(
        _c(r"doing analysis at temp"),
        "info",
        "analysis_banner",
        "Standard ngspice analysis banner (temperature echo).",
    ),
    DiagnosticRule(
        _c(r"no\. of data rows"),
        "info",
        "data_rows",
        "Number of result rows returned by the analysis.",
    ),
)


# Normalization masks applied before hashing so that per-iteration numeric
# noise does not create new hashes:
_MASK_PATTERNS = (
    # floats (incl. scientific notation)
    (re.compile(r"-?\d+\.\d+(?:[eE][+-]?\d+)?"), "<f>"),
    (re.compile(r"\b\d+\b"), "<n>"),
    # instance paths / names: b.xm1.xgmos.bg1 -> <inst>
    (re.compile(r"\bin line \S+"), "in line <inst>"),
    (re.compile(r"\b\S*(?:bodydiode|_cpm2|xm\d)\S*", re.IGNORECASE), "<dev>"),
)


@dataclass
class DiagnosticRecord:
    """Aggregated record for one unique diagnostic signature."""

    rule_name: str
    severity: str
    explanation: str
    first_raw: str           # first raw occurrence (for context)
    count: int = 0
    samples: list = field(default_factory=list)   # up to N raw variants


class NgspiceDiagnostics:
    """Captures and deduplicates ngspice shared-library console output.

    The PySpice Shared binding exposes the raw console buffer; we poll it on
    demand (after each analysis) rather than hooking callbacks, keeping this
    decoupled from the simulator lifecycle.
    """

    MAX_SAMPLES_PER_RECORD = 3

    def __init__(self):
        self.records: dict[str, DiagnosticRecord] = {}
        self.unclassified_count = 0
        self._lock = threading.Lock()

    # -- classification ----------------------------------------------------

    @staticmethod
    def normalize(raw: str) -> str:
        """Mask volatile tokens so equal-class messages hash equally."""
        text = raw.strip().lower()
        for pat, repl in _MASK_PATTERNS:
            text = pat.sub(repl, text)
        return text

    @staticmethod
    def classify(raw: str) -> tuple[DiagnosticRule | None, str]:
        """Return (matching rule or None, normalized text)."""
        normalized = NgspiceDiagnostics.normalize(raw)
        for rule in NGSPICE_DIAGNOSTICS:
            if rule.pattern.search(raw):
                return rule, normalized
        return None, normalized

    def ingest(self, raw_line: str) -> None:
        """Classify one console line and update its aggregate record."""
        line = raw_line.strip()
        if not line:
            return
        rule, key_text = self.classify(line)
        digest = hashlib.sha256(key_text.encode()).hexdigest()[:16]

        with self._lock:
            if rule is None:
                # ignore pure-progress noise but count it
                if re.match(r"^(note|note:|\d|\s*$)", line.lower()):
                    return
                self.unclassified_count += 1
                rec = self.records.get(digest)
                if rec is None:
                    rec = DiagnosticRecord(
                        "unclassified", "info", "", line)
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

    # -- capture bridge ------------------------------------------------------

    def drain_shared_buffer(self, ngspice_shared) -> int:
        """Pull pending console lines from the ngspice shared library.

        Returns the number of lines ingested. Safe to call after every
        analysis; cheap when the buffer is empty.
        """
        total = 0
        for buf_attr in ("_stdout", "_stderr"):
            buf = getattr(ngspice_shared, buf_attr, None)
            if not buf:
                continue
            try:
                pending = list(buf)
            except TypeError:
                continue
            if pending:
                buf.clear()
                for line in pending:
                    self.ingest(str(line))
                    total += 1
        return total

    # -- reporting -------------------------------------------------------------

    def report(self, min_severity: str = "info") -> str:
        """Render one block per unique diagnostic, with counts.

        Order: errors, then warnings, then info/unclassified.
        """
        order = {"error": 0, "warning": 1, "info": 2}
        threshold = {"error": 0, "warning": 1, "info": 2}[min_severity]

        selected = [r for r in self.records.values()
                    if order.get(r.severity, 2) <= threshold]
        selected.sort(key=lambda r: (order.get(r.severity, 2), -r.count))

        blocks: list[str] = []
        totals = {"error": 0, "warning": 0, "info": 0}
        for rec in selected:
            totals[rec.severity] = totals.get(rec.severity, 0) + rec.count
            head = (f"[{rec.severity.upper()}] {rec.rule_name} "
                    f"x{rec.count}")
            body = f"    {rec.explanation}" if rec.explanation else ""
            sample = f'    e.g. "{rec.first_raw[:120]}"'
            extra = ""
            if len(rec.samples) > 1:
                extra = (f"    ({len(rec.samples)} distinct variants seen)")
            blocks.append("\n".join(x for x in (head, body, sample, extra)
                                    if x))

        summary = (f"ngspice diagnostics: "
                   f"{totals.get('error', 0)} error(s), "
                   f"{totals.get('warning', 0)} warning(s), "
                   f"{totals.get('info', 0)} info "
                   f"({len(self.records)} unique signatures)")

        return "\n".join([summary] + [""] + blocks) if blocks else summary

    def reset(self) -> None:
        """Clear all accumulated records."""
        with self._lock:
            self.records.clear()
            self.unclassified_count = 0


# Module-level singleton used by the engine.
_default_diagnostics: NgspiceDiagnostics | None = None


def get_diagnostics() -> NgspiceDiagnostics:
    """Return the process-wide diagnostics collector (creating if needed)."""
    global _default_diagnostics
    if _default_diagnostics is None:
        _default_diagnostics = NgspiceDiagnostics()
    return _default_diagnostics


def report_and_reset(min_severity: str = "warning") -> str | None:
    """Convenience: report accumulated diagnostics and clear them.

    Returns the report string, or None when nothing at/above the threshold
    was recorded (so callers can skip noisy logging).
    """
    global _default_diagnostics
    if _default_diagnostics is None:
        return None
    text = _default_diagnostics.report(min_severity=min_severity)
    had_content = len(_default_diagnostics.records) > 0
    _default_diagnostics.reset()
    return text if had_content else None


class NgspiceLogDedupFilter(logging.Filter):
    """Logging filter that drops repeated ngspice warning/error messages.

    PySpice forwards ngspice console output through Python logging (logger
    ``PySpice.Spice.NgSpice.Shared``). Vendor models can emit the same
    warning hundreds of times per sweep. This filter keeps a bounded LRU of
    recent message signatures (normalized + hashed, same scheme as
    NgspiceDiagnostics) and drops repeats within the same engine run.

    Install once at package import:
        logging.getLogger("PySpice").addFilter(NgspiceLogDedupFilter())
    """

    MAX_TRACKED = 256

    def __init__(self):
        super().__init__()
        self._seen: dict[str, int] = {}     # hash -> count
        self._order: list[str] = []         # insertion order for LRU cap

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        normalized = NgspiceDiagnostics.normalize(msg)
        digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        # Only dedup ngspice-originated noise; let everything else through.
        if not any(rule.pattern.search(msg) for rule in NGSPICE_DIAGNOSTICS):
            return True

        if digest in self._seen:
            self._seen[digest] += 1
            return False   # drop the repeat
        self._seen[digest] = 1
        self._order.append(digest)
        if len(self._order) > self.MAX_TRACKED:
            oldest = self._order.pop(0)
            self._seen.pop(oldest, None)
        return True        # first occurrence passes

    def counts(self) -> dict[str, int]:
        """Signature-hash -> suppressed-repeat counts (diagnostics)."""
        return dict(self._seen)


def install_log_dedup(*logger_names: str) -> NgspiceLogDedupFilter:
    """Attach the dedup filter directly to the emitting loggers.

    Logging filters only apply at the point of emission, so attaching to a
    parent ('PySpice') does not affect child-logger records. We attach to
    every known PySpice ngspice-emitting logger explicitly.
    """
    names = logger_names or (
        "PySpice.Spice.NgSpice.Shared",
        "PySpice.Spice.NgSpice.Shared.NgSpiceShared",
        "PySpice",
    )
    filt = NgspiceLogDedupFilter()
    for name in names:
        logger = logging.getLogger(name)
        if not any(isinstance(f, NgspiceLogDedupFilter)
                   for f in logger.filters):
            logger.addFilter(filt)
    return filt


def install_console_suppressor() -> None:
    """Suppress ngspice's own C-level console printing for duplicate messages.

    ngspice prints each console line to the real process stdout unless its
    SendChar callback returns 1. PySpice's base ``send_char`` always returns
    0 (print everything). We wrap it: known-diagnostic messages that we have
    already seen once return 1 (suppress); new messages pass through so the
    user still sees the first occurrence.
    """
    from PySpice.Spice.NgSpice import Shared as _shared

    if getattr(_shared.NgSpiceShared.send_char,
               "_piece_spice_suppressor", False):
        return

    original = _shared.NgSpiceShared.send_char

    def send_char(self, message, ngspice_id):
        try:
            normalized = NgspiceDiagnostics.normalize(message)
            digest = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            if any(rule.pattern.search(message)
                   for rule in NGSPICE_DIAGNOSTICS):
                if digest in self.__dict__.setdefault("_seen_diags", set()):
                    return 1          # duplicate: suppress console output
                self._seen_diags.add(digest)
                return 0              # first occurrence: let it print
        except Exception:
            pass
        return original(self, message, ngspice_id)

    send_char._piece_spice_suppressor = True
    _shared.NgSpiceShared.send_char = send_char


class ConsoleQuiet:
    """Temporarily silence C-level stdout/stderr writes (ngspice printf).

    ngspice's C library writes diagnostics directly to the process stdout,
    which no Python logging filter can intercept. This context manager
    redirects the OS file descriptors 1 and 2 to os.devnull for the duration
    of the block, then restores them. Python ``sys.stdout/err`` objects are
    untouched, so normal library logging/printing continues via whatever
    handlers they use — but raw C writes vanish.

    Use around ngspice analysis calls in the engine; the deduplicating
    NgspiceDiagnostics collector still receives every line through the
    PySpice callback, so nothing diagnostic is lost.
    """

    def __init__(self):
        self._saved = None

    def __enter__(self):
        import os, sys
        sys.stdout.flush()
        self._saved = (os.dup(1), os.dup(2))
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
        finally:
            os.close(devnull)
        return self

    def __exit__(self, *exc):
        import os
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        os.close(self._saved[0])
        os.close(self._saved[1])
        return False


def install_notebook_quiet_console() -> None:
    """Suppress ALL ngspice console echo at the SendChar callback level.

    Under Jupyter, stdout is a captured pipe and each of ngspice's ~90k
    per-time-step messages pays a Python-callback + pipe-write cost; total
    runtime balloons 10x+. This wrapper returns 1 (suppress) for EVERY
    message - nothing reaches any pipe - while still feeding each line to
    the NgspiceDiagnostics collector, so warnings/errors are classified and
    deduplicated exactly as before. Real failures still surface through the
    engine's exception path and diag.report().

    Idempotent; safe to call at import time or before any simulation.
    """
    from PySpice.Spice.NgSpice import Shared as _shared

    if getattr(_shared.NgSpiceShared.send_char,
               "_piece_spice_full_quiet", False):
        return

    collector = get_diagnostics()

    def send_char(self, message, ngspice_id, _collector=collector):
        try:
            _collector.ingest(message)
        except Exception:
            pass
        return 1                      # never echo into captured pipes

    send_char._piece_spice_suppressor = True   # implies earlier wrapper
    send_char._piece_spice_full_quiet = True
    _shared.NgSpiceShared.send_char = send_char
