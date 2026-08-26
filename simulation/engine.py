"""Simulation execution engine for the spicelib + LTspice backend.

Replaces the retired PySpice/ngspice in-process engine. Architecture:

- Netlist TEXT goes in (testbenches generate complete SPICE netlists via
  the shared text-builder utilities; the engine never parses circuits).
- LTspice executes out-of-process via spicelib.SimRunner: solver crashes
  are subprocess failures, never Python faults.
- Waveforms come back as a SimResult: {trace_name: ndarray} parsed from
  the .raw file with spicelib.RawRead, plus the classified .log contents.

All analysis methods share one execution path (_execute) so timeout,
workdir hygiene, diagnostics, and error translation behave identically
for transient, DC, AC and OP runs.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.diagnostics import get_diagnostics

_DEFAULT_TIMEOUT_S = 300.0


# ---------------------------------------------------------------------------
# Analysis directives (structured; the engine renders the dot-card)
# ---------------------------------------------------------------------------

@dataclass
class TranDirective:
    """.tran - transient analysis."""
    step_time: float
    end_time: float
    start_time: float = 0.0
    max_step: float | None = None


@dataclass
class DCDirective:
    """.dc - sweep one independent source. The source element must exist in
    the netlist text under the given name (e.g. 'V1')."""
    source_name: str
    start: float
    stop: float
    step: float


@dataclass
class ACDirective:
    """.ac - small-signal frequency sweep."""
    start_freq: float
    stop_freq: float
    n_points: int = 1
    variation: str = "lin"          # lin | oct | dec


@dataclass
class OPDirective:
    """`.op` - single operating point."""
    pass


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def render_directive(d) -> str:
    """Render an analysis directive object as its SPICE dot-card line."""
    if isinstance(d, TranDirective):
        card = (f".tran 0 {_fmt(d.end_time)} {_fmt(d.start_time)}")
        if d.max_step is not None:
            card += f" {_fmt(d.max_step)}"
        return card
    if isinstance(d, DCDirective):
        return (f".dc {d.source_name} {_fmt(d.start)} {_fmt(d.stop)} "
                f"{_fmt(d.step)}")
    if isinstance(d, ACDirective):
        return (f".ac {d.variation} {d.n_points} {_fmt(d.start_freq)} "
                f"_ {_fmt(d.stop_freq)}".replace(" _ ", " "))
    if isinstance(d, OPDirective):
        return ".op"
    raise TypeError(f"unsupported directive type: {type(d)!r}")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """One completed LTspice run.

    traces      : trace name -> 1-D ndarray, names exactly as LTspice writes
                  them ("V(drain)", "I(Rload)", ...). The sweep variable is
                  included under its raw-file name ("time"/"frequency"/the
                  swept source's trace). No sign rewriting happens here -
                  callers own polarity interpretation.
    sweep_var   : name of the independent-variable trace.
    log_text    : full .log contents.
    warnings    : deduplicated warning records from the diagnostics layer.
    netlist_path/raw_path/log_path : artifact paths (temp dir unless kept).
    """

    traces: dict
    sweep_var: str
    log_text: str = ""
    warnings: list[str] = field(default_factory=list)
    netlist_path: Path | None = None
    raw_path: Path | None = None
    log_path: Path | None = None

    def get(self, name: str):
        try:
            return self.traces[name]
        except KeyError:
            raise KeyError(
                f"trace {name!r} not found; available: {sorted(self.traces)}")


class LTSpiceError(RuntimeError):
    """Solver failure: carries exit status plus the log tail."""

    def __init__(self, message: str, *, log_tail: str = "",
                 returncode: int | None = None):
        super().__init__(message)
        self.log_tail = log_tail
        self.returncode = returncode


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SpiceEngine:
    """Out-of-process LTspice execution via spicelib.

    Resolution order for the executable: explicit argument > LTSPICE_EXE
    environment variable > standard install locations. Paths are stored as
    native Windows strings (MSYS-style paths are rejected by the exe).
    """

    _KNOWN_LOCATIONS = (
        r"C:\Program Files\ADI\LTspice\LTspice.exe",
        r"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe",
    )

    def __init__(self, ltspice_exe: str | None = None,
                 timeout_s: float = _DEFAULT_TIMEOUT_S,
                 temperature: float = 25.0,
                 keep_files: bool = True):
        self.ltspice_exe = self._resolve_exe(ltspice_exe)
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.keep_files = bool(keep_files)

    @classmethod
    def _resolve_exe(cls, explicit: str | None) -> str:
        if explicit:
            cand = Path(explicit)
            if not cand.exists():
                raise FileNotFoundError(f"LTspice not found at {cand}")
            return str(cand)

        env = os.environ.get("LTSPICE_EXE")
        if env and Path(env).exists():
            return str(Path(env))

        local = Path(os.environ.get("LOCALAPPDATA", "")) / \
            "Programs/ADI/LTspice/LTspice.exe"
        if local.exists():
            return str(local)

        for loc in cls._KNOWN_LOCATIONS:
            if Path(loc).exists():
                return loc

        raise FileNotFoundError(
            "LTspice executable not found. Install LTspice or set the "
            "LTSPICE_EXE environment variable.")

    # -- public analysis API -------------------------------------------------

    def run_transient(self, netlist_text: str, d: TranDirective, *,
                      timeout_s: float | None = None) -> SimResult:
        card = render_directive(d)
        if ".tran" not in netlist_text.lower():
            netlist_text = self._append_card(netlist_text, card)
        return self._execute(netlist_text, "tran", timeout_s)

    def run_dc_sweep(self, netlist_text: str, d: DCDirective, *,
                     timeout_s: float | None = None) -> SimResult:
        card = render_directive(d)
        if ".dc" not in netlist_text.lower():
            netlist_text = self._append_card(netlist_text, card)
        return self._execute(netlist_text, "dc", timeout_s)

    def run_ac_sweep(self, netlist_text: str, d: ACDirective, *,
                     timeout_s: float | None = None) -> SimResult:
        card = render_directive(d)
        if ".ac" not in netlist_text.lower():
            netlist_text = self._append_card(netlist_text, card)
        return self._execute(netlist_text, "ac", timeout_s)

    def run_op(self, netlist_text: str, *,
               timeout_s: float | None = None) -> SimResult:
        if ".op" not in netlist_text.lower():
            netlist_text = self._append_card(netlist_text, ".op")
        return self._execute(netlist_text, "op", timeout_s)

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _append_card(netlist_text: str, card_line: str) -> str:
        """Insert the analysis card before .end (or append if absent)."""
        lines = netlist_text.rstrip().splitlines()
        out, inserted = [], False
        for ln in lines:
            if ln.strip().lower() == ".end" and not inserted:
                out.append(card_line)
                inserted = True
            out.append(ln)
        if not inserted:
            out.append(card_line)
        return "\n".join(out) + "\n"

    def _temperature_cards(self) -> list[str]:
        """LTspice default temp is 27 C; emit a card when we differ."""
        return [f".temp {self.temperature:g}"] \
            if abs(self.temperature - 27.0) > 1e-9 else []

    def _prepare_workdir(self, netlist_text: str, tag: str) -> tuple[Path, Path]:
        workdir = Path(tempfile.mkdtemp(prefix=f"piecespice_{tag}_"))
        netlist_path = workdir / f"{tag}.net"
        body = netlist_text.rstrip()
        cards = "\n".join(self._temperature_cards())
        if cards:
            # temperature card must precede .end
            lines_ = body.splitlines()
            out, inserted = [], False
            for ln in lines_:
                if ln.strip().lower() == ".end" and not inserted:
                    out.append(cards)
                    inserted = True
                out.append(ln)
            if not inserted:
                out.append(cards)
            body = "\n".join(out)
        netlist_path.write_text(body + "\n", encoding="ascii", errors="replace")
        return workdir, netlist_path

    def _execute(self, netlist_text: str, tag: str,
                 timeout_s: float | None) -> SimResult:
        """Shared path: write netlist -> run LTspice -> parse raw + log."""
        from spicelib import RawRead

        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        workdir, netlist_path = self._prepare_workdir(netlist_text, tag)
        raw_path = netlist_path.with_suffix(".raw")
        log_path = netlist_path.with_suffix(".log")

        cmd = [self.ltspice_exe, "-b", "-ascii", "-Run",
               str(netlist_path)]
        try:
            completed = subprocess_run(cmd, timeout=timeout, cwd=str(workdir))
            rc = completed.returncode
        except TimeoutExpired as exc:
            raise LTSpiceError(
                f"LTspice timed out after {timeout:g}s ({tag})",
                log_tail=self._log_tail(log_path), returncode=None) from exc

        log_text = (log_path.read_text(encoding="utf-8", errors="ignore")
                    if log_path.exists() else "")

        diag = get_diagnostics()
        diag.parse_log_text(log_text)

        if not raw_path.exists():
            tail = "\n".join(log_text.splitlines()[-12:])
            raise LTSpiceError(
                f"LTspice produced no raw output for '{tag}' "
                f"(rc={rc}). Solver likely failed.",
                log_tail=tail, returncode=rc)

        raw = RawRead(str(raw_path))
        traces = {}
        sweep_var = None
        for name in raw.get_trace_names():
            wave = np.asarray(raw.get_trace(name).get_wave())
            traces[name] = wave
            if name.lower() in ("time", "frequency") or sweep_var is None:
                if name.lower() in ("time", "frequency"):
                    sweep_var = name
        if sweep_var is None:
            first = next(iter(traces))
            sweep_var = first

        warning_records = [
            r.rule_name for r in diag.records.values()
            if r.severity in ("error", "warning")]

        result = SimResult(
            traces=traces, sweep_var=sweep_var, log_text=log_text,
            warnings=warning_records,
            netlist_path=Path(netlist_path).resolve(),
            raw_path=raw_path, log_path=log_path)

        if not self.keep_files:
            shutil.rmtree(workdir, ignore_errors=True)
        return result

    @staticmethod
    def _log_tail(log_path: Path, n: int = 12) -> str:
        if not log_path.exists():
            return ""
        lines = log_path.read_text(encoding="utf-8",
                                   errors="ignore").splitlines()
        return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# subprocess helper (kept separate so tests can monkeypatch it)
# ---------------------------------------------------------------------------

class TimeoutExpired(TimeoutError):
    pass


def subprocess_run(cmd: list[str], timeout: float, cwd: str):
    """Thin wrapper around subprocess.run with LTspice-appropriate flags.

    LTspice returns 0 even on some solver errors (it writes diagnostics to
    the log instead); callers must therefore check for .raw existence, not
    just return code.
    """
    import subprocess as _sp

    try:
        return _sp.run(cmd, capture_output=True, text=True,
                       timeout=timeout, cwd=cwd)
    except _sp.TimeoutExpired as exc:
        raise TimeoutExpired(str(exc)) from exc
