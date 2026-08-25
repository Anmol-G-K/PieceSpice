"""T0.3 — Draft the new SpiceEngine interface contract for review.

This file is a PROPOSAL ONLY: it defines the target API surface that
`simulation/engine.py` will implement on spicelib/LTspice. It is NOT wired
into any testbench yet and imports nothing from spicelib at module level.

Review point for the user before implementation begins (T1.x/T2.x).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """One completed LTspice run.

    Attributes:
        traces: mapping of trace name -> 1-D numpy array. Names follow the
            raw-file convention exactly as LTspice writes them, e.g.
            "V(drain)", "V(gate)", "I(Rload)", "time"/"frequency" for the
            sweep variable. No sign rewriting is applied at this layer -
            callers own polarity interpretation (mirrors today's contract
            where benches negate PySpice source currents themselves).
        sweep_var: "time" for .tran/.dc-style runs, "frequency" for .ac.
        log_text: full LTspice .log contents (fed to the diagnostics layer).
        raw_path: path of the parsed .raw file (kept for debugging).
        warnings: deduplicated warning lines extracted from the log by the
            diagnostics registry (same SHA-256 masking as today).
    """

    traces: dict
    sweep_var: str
    log_text: str = ""
    raw_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def get(self, name: str) -> "np.ndarray":
        """Trace accessor with a clear error for missing names."""
        try:
            return self.traces[name]
        except KeyError:
            raise KeyError(
                f"trace {name!r} not in result; available: "
                f"{sorted(self.traces)})")


# ---------------------------------------------------------------------------
# Analysis directive objects (structured, not text blobs)
# ---------------------------------------------------------------------------

@dataclass
class TranDirective:
    """.tran - transient analysis."""
    step_time: float
    end_time: float
    start_time: float = 0.0
    max_step: float | None = None     # LTspice "maximum timestep"


@dataclass
class DCDirective:
    """.dc - DC transfer curve. Sweeps one independent source by name."""
    source_name: str                  # e.g. "V1" (the element name in text)
    start: float
    stop: float
    step: float


@dataclass
class ACDirective:
    """.ac - small-signal frequency sweep."""
    start_freq: float
    stop_freq: float
    n_points: int = 1
    variation: str = "lin"            # "lin" | "oct" | "dec"
    # NOTE: Curve-4 parity requires COMPLEX output. RawRead must deliver
    # magnitude+phase so the engine can rebuild complex voltages. This is
    # the single highest-technical-risk item; verify FIRST (see T2.4).


@dataclass
class OPDirective:
    """`.op` - single operating point (stepped-OP validators use N of these,
    or one netlist with .step)."""
    pass


# ---------------------------------------------------------------------------
# Target engine interface (replaces the PySpice-shaped SpiceEngine)
# ---------------------------------------------------------------------------

class LTSpiceEngine:
    """Out-of-process LTspice execution via spicelib.

    Contract decisions embedded here (review these):

    1. INPUT IS NETLIST TEXT. Callers (testbenches) generate complete
       SPICE netlist strings via the shared text-builder utility (T2.2).
       The engine never parses or edits the netlist - it writes bytes to a
       temp file and executes it. This keeps engine responsibilities to:
       run / capture / parse / classify.

    2. ONE METHOD PER ANALYSIS TYPE, mirroring today's API shape so testbench
       migration stays mechanical:
           run_transient(netlist_text, TranDirective)  -> SimResult
           run_dc_sweep(netlist_text, DCDirective)     -> SimResult
           run_ac_sweep(netlist_text, ACDirective)     -> SimResult
           run_op(netlist_text)                        -> SimResult
       The directive object is appended to the netlist if not already
       present (engine checks for an existing directive line to avoid
       duplication).

    3. ERRORS ARE EXCEPTIONS, NOT EXIT CODES. Non-zero solver exit, missing
       .raw, or fatal log lines raise `LTSpiceError` carrying the tail of
       the log. HOST-01 (segfault kills Python) is gone by construction:
       LTspice runs out-of-process.

    4. TIMEOUTS ARE MANDATORY. Every run takes `timeout_s` (default 300).
       A hung solver (EPC-01 class failure) becomes a TimeoutExpired error
       plus the partial log - never a stuck pipeline.

    5. TEMPERATURE: emitted as a `.temp <value>` card appended by the
       engine when temperature != 27 (LTspice default). Per-device temp
       parameters stay the caller's business.

    6. WORKDIR HYGIENE: every run gets its own temp directory
       (netlist + .raw + .log + .op.raw sidecar); cleaned after parsing
       unless keep_files=True for debugging.

    7. DIAGNOSTICS: the returned SimResult carries log text AND classified
       warnings; the global collector keeps its aggregate role unchanged.
    """

    def __init__(self, ltspice_exe: str | None = None,
                 timeout_s: float = 300.0, temperature: float = 25.0,
                 keep_files: bool = False):
        """Resolve exe path: explicit arg > LTSPICE_EXE env var > known
        install locations (%LOCALAPPDATA%/Programs/ADI/LTspice, Program Files).
        Raises immediately if not found (fail fast, per T0.1 finding that
        MSYS-style paths are rejected: always store/resolve native Windows
        paths)."""
        raise NotImplementedError("proposal only")

    def run_transient(self, netlist_text: str, directive: TranDirective,
                      *, timeout_s: float | None = None) -> SimResult:
        raise NotImplementedError("proposal only")

    def run_dc_sweep(self, netlist_text: str, directive: DCDirective,
                     *, timeout_s: float | None = None) -> SimResult:
        raise NotImplementedError("proposal only")

    def run_ac_sweep(self, netlist_text: str, directive: ACDirective,
                     *, timeout_s: float | None = None) -> SimResult:
        raise NotImplementedError("proposal only")

    def run_op(self, netlist_text: str,
               *, timeout_s: float | None = None) -> SimResult:
        raise NotImplementedError("proposal only")


class LTSpiceError(RuntimeError):
    """Solver failed: carries exit code + log tail for diagnostics."""
