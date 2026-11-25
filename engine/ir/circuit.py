"""Circuit representation consumed by the PieceSpice custom solver.

This is deliberately a small, closed data model rather than a general SPICE
AST.  Two properties matter more than completeness:

* **No expression tree.**  A behavioural source is stored as its raw text and
  rejected by the kernel.  Evaluating SPICE expressions at runtime is the
  entry ramp to being a simulator clone -- the ROM contract lowers tables
  instead (``engine.ROM_ARTIFACT_SCHEMA``).
* **Flat.**  Subcircuits are expanded at parse time, so an element's ``nodes``
  are always top-level node names and one MNA system describes the whole
  circuit.  There is no hierarchical solve.

Names are preserved from the deck so traces and provenance match what LTspice
reports, with one exception: **node names are lower-cased**, because SPICE
treats them case-insensitively and keeping both ``V(Drain)`` and ``V(drain)``
would silently create two nodes.  Internals of an instance are prefixed with
the instance's element name, which is deterministic and collision-free without
depending on LTspice's own ``Xdut:gi`` convention.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise

#: Node name LTspice treats as the reference node.
GROUND: str = "0"

_VALUE_RE = re.compile(
    r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z]*)$")
_SPICE_SUFFIX = {"t": 1e12, "g": 1e9, "k": 1e3, "m": 1e-3, "u": 1e-6,
                 "n": 1e-9, "p": 1e-12, "f": 1e-15}


def parse_value(text: str) -> float:
    """A SPICE numeric literal, including suffixes.

    This lives with the IR rather than the reader because it defines what a
    stored value *is*.  ``meg`` is mega and ``m`` is milli -- the reason this
    is a function and not ``float()``: ``1m`` read as 1 would be a 1000x error
    in a resistance.  Letters trailing a recognised suffix are ignored, so
    ``10kohm`` parses like ``10k``.
    """
    s = text.strip()
    m = _VALUE_RE.match(s)
    if m is None:
        raise ValueError(f"not a SPICE number: {text!r}")
    number = float(m.group(1))
    suffix = m.group(2).lower()
    if not suffix:
        return number
    if suffix.startswith("meg"):
        return number * 1e6
    if suffix.startswith("mil"):
        return number * 25.4e-6
    scale = _SPICE_SUFFIX.get(suffix[0])
    if scale is None:
        raise ValueError(f"unknown SPICE suffix {suffix!r} in {text!r}")
    return number * scale


class ElementKind(str, Enum):
    """The element kinds the IR can represent.

    Kinds are the *envelope* vocabulary (see :mod:`engine.envelope`), not the
    set the stage-2 kernel can evaluate yet: a kind can be representable and
    still raise ``NotYetImplemented`` when the solver tries to build its
    companion model.  Keeping representation separate from evaluation is what
    lets the parser be verified against every emitted ROM while the kernel
    grows.
    """

    RESISTOR = "R"
    CAPACITOR = "C"
    CHARGE_CAPACITOR = "C:Q"
    PWL_CAP = "PWL_CAP"
    INDUCTOR = "L"
    VSOURCE = "V"
    ISOURCE = "I"
    DIODE = "D"
    SWITCH = "S"
    PWL_CHANNEL = "PWL_CHANNEL"
    BEHAVIOURAL_SOURCE = "G:VALUE"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True, slots=True)
class Waveform:
    """Time behaviour of an independent source.

    ``kind`` is one of ``"dc"``, ``"pwl"`` or ``"pulse"``.  The evaluation
    semantics mirror LTspice, because both engines are fed the same deck and a
    systematic offset in a source would masquerade as solver error:

    * PWL -- linear between points, first value before the first point, last
      value after the last.  Times must be strictly increasing.
    * PULSE -- ``PULSE(v1 v2 td tr tf pw per)``; a non-positive period means a
      single pulse.
    """

    kind: str
    dc: float = 0.0
    points: tuple[tuple[float, float], ...] = ()
    pulse: tuple[float, float, float, float, float, float, float] | None = None

    # -- constructors ------------------------------------------------------

    @classmethod
    def constant(cls, value: float) -> Waveform:
        return cls(kind="dc", dc=float(value))

    @classmethod
    def pwl(cls, points: Iterable[tuple[float, float]]) -> Waveform:
        pts = tuple((float(t), float(v)) for t, v in points)
        if len(pts) < 2:
            raise ValueError("PWL needs at least two points")
        times = [p[0] for p in pts]
        if any(b <= a for a, b in pairwise(times)):
            raise ValueError("PWL times must be strictly increasing")
        return cls(kind="pwl", points=pts, dc=pts[0][1])

    @classmethod
    def pulse_wave(
        cls,
        v1: float,
        v2: float,
        td: float = 0.0,
        tr: float = 0.0,
        tf: float = 0.0,
        pw: float = 0.0,
        period: float = 0.0,
    ) -> Waveform:
        return cls(
            kind="pulse",
            dc=float(v1),
            pulse=(float(v1), float(v2), float(td), float(tr), float(tf),
                   float(pw), float(period)),
        )

    # -- evaluation --------------------------------------------------------

    @property
    def is_time_dependent(self) -> bool:
        return self.kind != "dc"

    def value(self, t: float) -> float:
        """Source value at time *t* seconds."""
        if self.kind == "dc":
            return self.dc
        if self.kind == "pwl":
            return _pwl_at(self.points, t)
        if self.kind == "pulse":
            assert self.pulse is not None
            return _pulse_at(self.pulse, t)
        raise ValueError(f"unknown waveform kind {self.kind!r}")


def _pwl_at(points: tuple[tuple[float, float], ...], t: float) -> float:
    if t <= points[0][0]:
        return points[0][1]
    if t >= points[-1][0]:
        return points[-1][1]
    lo, hi = 0, len(points) - 1
    while hi - lo > 1:                      # invariant: points[lo].t <= t < points[hi].t
        mid = (lo + hi) // 2
        if points[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, v0 = points[lo]
    t1, v1 = points[hi]
    if t1 == t0:                            # guarded by the constructor
        return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def _pulse_at(p: tuple[float, ...], t: float) -> float:
    v1, v2, td, tr, tf, pw, period = p
    u = t - td
    if u < 0.0:
        return v1
    if period > 0.0:
        u = math.fmod(u, period)
        if u < 0.0:
            u += period
    if u < tr:
        return v1 if tr <= 0.0 else v1 + (v2 - v1) * u / tr
    u -= tr
    if u < pw:
        return v2
    u -= pw
    if u < tf:
        return v2 if tf <= 0.0 else v2 + (v1 - v2) * u / tf
    return v1


@dataclass(frozen=True, slots=True)
class DeviceModel:
    """A ``.MODEL`` declaration, kept as typed parameters.

    ``family`` is the model type token from the deck (``D``, ``SW``).
    Parameter text is preserved verbatim rather than coerced: a kernel that
    does not read a parameter can still report exactly what the deck said, and
    a coercion at parse time would destroy evidence.
    """

    name: str
    family: str
    params: Mapping[str, str]

    def text(self, key: str) -> str | None:
        for k, v in self.params.items():
            if k.upper() == key.upper():
                return v
        return None

    def get(self, key: str, default: float | None = None) -> float | None:
        """Numeric parameter, or *default* when absent.

        A present-but-non-numeric value raises: silently falling back to a
        default would substitute an invented device parameter.
        """
        raw = self.text(key)
        if raw is None:
            return default
        return parse_value(raw)


@dataclass(frozen=True, slots=True)
class Element:
    """One flat circuit element.

    ``nodes`` is the positional connection list exactly as SPICE expects:
    two nodes for a two-terminal device, four for a voltage-controlled switch
    (out+, out-, control+, control-), three for an inductor's worth of nothing
    else.  ``model`` names a :class:`DeviceModel`.  ``expression`` holds the
    raw brace expression of a behavioural/charge form -- stored for
    provenance, never evaluated here.  ``data`` carries a device payload (a
    charge table, a channel table) for the ROM-level kinds; the IR does not
    inspect it, and the solver device that consumes it validates it.
    """

    name: str
    kind: ElementKind
    nodes: tuple[str, ...]
    value: float | None = None
    waveform: Waveform | None = None
    model: str | None = None
    expression: str | None = None
    data: object | None = None
    source_line: int = 0
    origin: str = ""

    def __post_init__(self) -> None:
        expected = {
            ElementKind.RESISTOR: 2,
            ElementKind.CAPACITOR: 2,
            ElementKind.CHARGE_CAPACITOR: 2,
            ElementKind.PWL_CAP: 2,
            ElementKind.INDUCTOR: 2,
            ElementKind.VSOURCE: 2,
            ElementKind.ISOURCE: 2,
            ElementKind.DIODE: 2,
            ElementKind.SWITCH: 4,
            ElementKind.PWL_CHANNEL: 3,
            ElementKind.BEHAVIOURAL_SOURCE: 2,
        }[self.kind]
        if len(self.nodes) != expected:
            raise ValueError(
                f"{self.name}: {self.kind.value} needs {expected} nodes, "
                f"got {len(self.nodes)} ({self.nodes})")
        if self.kind in (ElementKind.RESISTOR, ElementKind.CAPACITOR,
                         ElementKind.INDUCTOR) and self.value is None:
            raise ValueError(f"{self.name}: {self.kind.value} needs a value")
        if self.kind in (ElementKind.VSOURCE, ElementKind.ISOURCE) \
                and self.waveform is None:
            raise ValueError(f"{self.name}: {self.kind.value} needs a waveform")
        if self.kind in (ElementKind.DIODE, ElementKind.SWITCH) \
                and self.model is None and self.data is None:
            raise ValueError(
                f"{self.name}: {self.kind.value} needs a .model (parametric) "
                "or a table payload (ROM-level)")


@dataclass(frozen=True, slots=True)
class AnalysisDirective:
    """A parsed analysis card.  ``raw`` is kept for provenance."""

    kind: str
    raw: str
    end_time: float | None = None
    start_time: float = 0.0
    max_step: float | None = None
    step_time: float | None = None


@dataclass(frozen=True, slots=True)
class NodeIndex:
    """Deterministic node numbering.  Ground has no unknown.

    ``names`` is ordered by first appearance in the deck, so the numbering is
    reproducible from the netlist text alone -- two runs of the same deck must
    produce identical matrices, otherwise a solver-vs-solver comparison is
    meaningless.
    """

    names: tuple[str, ...]
    ground: str = GROUND
    _lookup: Mapping[str, int] = field(default_factory=dict, repr=False,
                                      compare=False)

    @classmethod
    def build(cls, names: Iterable[str], ground: str = GROUND) -> NodeIndex:
        ordered = tuple(dict.fromkeys(n for n in names if n != ground))
        return cls(names=ordered, ground=ground,
                   _lookup={n: i for i, n in enumerate(ordered)})

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self._lookup

    def index(self, name: str) -> int | None:
        """Row/column for *name*, or ``None`` for the reference node."""
        if name == self.ground:
            return None
        try:
            return self._lookup[name]
        except KeyError:
            raise KeyError(f"unknown node {name!r}") from None

    def name_of(self, index: int) -> str:
        return self.names[index]


@dataclass(frozen=True, slots=True)
class Circuit:
    """A flat circuit plus everything needed to integrate it once.

    ``directives`` carries the analysis cards; the solver never infers a stop
    time, because an invented end time is an invented result.
    """

    title: str
    elements: tuple[Element, ...]
    models: Mapping[str, DeviceModel]
    node_index: NodeIndex
    directives: tuple[AnalysisDirective, ...] = ()
    options: tuple[str, ...] = ()
    expanded_subcircuits: tuple[str, ...] = ()
    temperature_c: float | None = None

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for element in self.elements:
            if element.name in seen:
                raise ValueError(f"duplicate element name {element.name!r}")
            seen.add(element.name)
            for node in element.nodes:
                if node != self.node_index.ground and node not in self.node_index:
                    raise ValueError(
                        f"{element.name}: node {node!r} is not in the node index")
            if element.model is not None and element.model not in self.models:
                raise ValueError(
                    f"{element.name}: undefined .model {element.model!r}")

    def __len__(self) -> int:
        return len(self.elements)

    def of_kind(self, *kinds: ElementKind) -> tuple[Element, ...]:
        wanted = set(kinds)
        return tuple(e for e in self.elements if e.kind in wanted)

    def element(self, name: str) -> Element:
        for element in self.elements:
            if element.name == name:
                return element
        raise KeyError(f"no element named {name!r}")

    def analysis(self, kind: str) -> AnalysisDirective | None:
        for directive in self.directives:
            if directive.kind == kind:
                return directive
        return None
