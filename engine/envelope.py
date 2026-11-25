"""Supported modelling envelope for the PieceSpice custom solver.

The custom solver is **not** a general-purpose SPICE engine and does not try
to become one.  It supports exactly the element set and analyses that the
emitted PieceSpice ROMs and this project's own benches use.  Everything else
raises, and the router turns that into an external SPICE fallback
(``docs/research/custom-engine-feasibility.md`` section 4: compatibility is
the harder problem; the fallback is a routing decision, not a solver
workstream).

Why an explicit envelope: a solver that silently approximates an element it
does not model returns numbers that look measured and are not.  In a project
whose every published figure is a recorded measurement, failing closed is
cheaper than a wrong record.

Two failure modes are kept distinct on purpose:

* :class:`UnsupportedConstruct` -- this element/analysis will never be in the
  fast path (it belongs to the external engine).
* :class:`NotYetImplemented` -- the envelope includes it, but the kernel does
  not implement it yet.  Callers must not treat this as a licence to
  approximate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: Envelope tag for the charge-defined capacitor form ``C.. n1 n2 Q = {..}``.
#: Distinct from a linear ``C`` because the two have different state: a linear
#: capacitor stores a voltage, a charge-defined one stores q(V).
CHARGE_CAP_TAG: Final[str] = "C:Q"

#: Envelope tag for the behavioural current source ``G.. VALUE = {..}``.
BEHAVIOURAL_SOURCE_TAG: Final[str] = "G:VALUE"


class EnvelopeError(RuntimeError):
    """Base class for envelope violations."""


class UnsupportedConstruct(EnvelopeError):
    """The construct is outside the fast path; route it to the SPICE fallback."""

    def __init__(self, tag: str, reason: str = "") -> None:
        self.tag = tag
        detail = f" ({reason})" if reason else ""
        super().__init__(
            f"{tag!r} is outside the PieceSpice solver envelope{detail}; "
            "route the circuit to the external SPICE engine"
        )


class NotYetImplemented(EnvelopeError):
    """The construct is in the envelope but the kernel has not implemented it."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        super().__init__(
            f"{tag!r} is declared in the solver envelope but is not implemented "
            "yet; do not approximate it"
        )


@dataclass(frozen=True)
class Support:
    """One supported (or planned) construct.

    ``available`` is the single switch a stage flips when its kernel lands;
    error messages and documentation both read it, so the envelope cannot
    drift from the implementation silently.
    """

    tag: str
    description: str
    available: bool
    note: str = ""


#: Element and device constructs the fast path knows about.
#:
#: The set is deliberately closed.  ``S_SW`` (``SW(Ron Roff Vt Vh)``) is
#: piecewise-linear by construction: two conductances and one threshold, which
#: is exactly the form a PWL engine wants.  A behavioural ``G.. VALUE={..}``
#: is in the envelope only because it can be lowered to ROM tables; the solver
#: never evaluates a SPICE expression.
SUPPORTED: Final[dict[str, Support]] = {
    "R": Support("R", "linear resistor", True),
    "C": Support("C", "linear capacitor", True),
    CHARGE_CAP_TAG: Support(
        CHARGE_CAP_TAG,
        "charge-defined capacitor C.. Q={q(V)}",
        False,
        "charge is a state variable; the table arrives with the ROM contract",
    ),
    "L": Support("L", "linear inductor (branch unknown)", True),
    "V": Support(
        "V",
        "independent voltage source: DC, PWL(...), PULSE(...)",
        True,
        "sources are evaluated at the step time, never symbolically",
    ),
    "I": Support("I", "independent current source: DC, PWL(...), PULSE(...)", True),
    "D": Support(
        "D",
        "diode with a .model D(IS N RS) declaration",
        False,
        "evaluated as a piecewise-linear region set, not Newton on exp()",
    ),
    "S": Support(
        "S",
        "voltage-controlled switch .model SW(Ron Roff Vt Vh)",
        False,
        "Ron/Roff/Vt/Vh are already piecewise-linear",
    ),
    BEHAVIOURAL_SOURCE_TAG: Support(
        BEHAVIOURAL_SOURCE_TAG,
        "behavioural current source lowered to ROM tables",
        False,
        "never evaluated as a SPICE expression; lowered from the ROM artifact",
    ),
    "X": Support(
        "X",
        "subcircuit instance, flattened at parse time",
        True,
        "flattening keeps one flat MNA system; no hierarchical solve",
    ),
}

#: Analysis types the fast path will serve.
ANALYSES: Final[dict[str, Support]] = {
    "op": Support("op", "DC operating point", True),
    "tran": Support("tran", "transient integration", True),
    "dc": Support("dc", "DC sweep", False),
    "ac": Support("ac", "small-signal AC", False, "fallback only"),
    "noise": Support("noise", "noise analysis", False, "fallback only"),
}


def require(analysis: str) -> Support:
    """Return the support record for *analysis*, or raise the right error."""
    support = ANALYSES.get(analysis.lower().lstrip("."))
    if support is None:
        raise UnsupportedConstruct(analysis)
    if not support.available:
        raise NotYetImplemented(analysis)
    return support


def check(tag: str) -> Support:
    """Return the support record for element *tag*, or raise the right error."""
    support = SUPPORTED.get(tag)
    if support is None:
        raise UnsupportedConstruct(tag)
    if not support.available:
        raise NotYetImplemented(tag)
    return support


def missing() -> list[str]:
    """Envelope tags declared but not implemented yet (for docs and tests)."""
    return [s.tag for s in SUPPORTED.values() if not s.available]
