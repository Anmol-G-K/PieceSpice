"""Restricted SPICE reader for the decks this project emits.

Scope is deliberately narrow: the reader understands what
``simulation.netlist_text`` and ``models.*_netlist_builder`` write, plus the
small subset of dot-cards those decks carry.  It is **not** a vendor-model
front end -- dialect quirks, ``PARAMS:`` headers, ``.param`` expansion and
behavioural expressions all raise, and stage 8 routes such netlists to the
external SPICE engine.

Three things are handled exactly, because a systematic difference between the
two engines would masquerade as solver error:

* PWL/PULSE source semantics (see :class:`engine.ir.circuit.Waveform`),
* subcircuit expansion (pins by position, internals prefixed),
* SPICE numeric suffixes (``1k``, ``2n``, ``1meg``; ``m`` is milli, not mega).

Anything that changes the solution and cannot be honoured -- ``.ic``,
``.nodeset``, ``.param``, ``.global`` -- raises instead of being ignored.  A
silently dropped assertion would make a comparison look better than it is.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from engine.envelope import UnsupportedConstruct
from engine.ir.circuit import (
    GROUND,
    AnalysisDirective,
    Circuit,
    DeviceModel,
    Element,
    ElementKind,
    NodeIndex,
    Waveform,
    parse_value,
)

#: Elements the reader can represent.  ``K`` (coupled inductors), ``E``/``F``/
#: ``H`` (controlled sources other than a behavioural ``G``), ``B``, ``J``,
#: ``M``, ``Q`` and ``T`` are outside the fast path.
_READABLE = {
    "R": ElementKind.RESISTOR,
    "C": ElementKind.CAPACITOR,
    "L": ElementKind.INDUCTOR,
    "V": ElementKind.VSOURCE,
    "I": ElementKind.ISOURCE,
    "D": ElementKind.DIODE,
    "S": ElementKind.SWITCH,
    "G": ElementKind.BEHAVIOURAL_SOURCE,
}

#: Dot-cards that only affect what the simulator writes out, never the
#: solution.  Ignoring them is safe.
_OUTPUT_CARDS = {
    ".save", ".meas", ".measure", ".four", ".plot", ".print", ".probe",
    ".backanno", ".width", ".unprotect",
}

#: Dot-cards that change the solve and are not implemented.  Raising beats a
#: silent difference in the answer.
_REJECTED_CARDS = {
    ".param", ".ic", ".nodeset", ".global", ".func", ".step", ".loadbias",
    ".tstab", ".noise", ".tf", ".sens", ".pz", ".disto",
}

_MAX_INCLUDE_DEPTH = 16
_MAX_SUBCKT_DEPTH = 16

#: Separators inside an argument list: whitespace or comma (`PWL(0,0 1u,5)`).
_ARG_SPLIT_RE = re.compile(r"[,\s]+")


class NetlistError(ValueError):
    """The deck is outside what the reader represents."""


@dataclass(frozen=True, slots=True)
class Subcircuit:
    """A ``.SUBCKT`` body, kept unexpanded so it can be instantiated twice."""

    name: str
    pins: tuple[str, ...]
    body: tuple[str, ...]
    origin: str


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Both halves of a parse: the flattened circuit and its library."""

    circuit: Circuit
    subcircuits: Mapping[str, Subcircuit]
    source: str = "<netlist>"


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> list[tuple[int, str]]:
    """Logical lines with comments removed and ``+`` continuations joined."""
    out: list[tuple[int, str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        stripped = line.lstrip()
        if not stripped or stripped[0] in "*;":
            continue
        if ";" in line:                     # inline comment
            line = line.split(";", 1)[0].rstrip()
        if line.lstrip().startswith("+"):    # continuation of the previous card
            if not out:
                raise NetlistError(f"line {lineno}: continuation with no card")
            prev_lineno, prev = out[-1]
            out[-1] = (prev_lineno, f"{prev} {line.lstrip()[1:].strip()}")
            continue
        out.append((lineno, line.strip()))
    return out


_INCLUDE_RE = re.compile(r"^\s*\.(?:include|inc|lib)\s+(\S+)\s*$", re.IGNORECASE)


def _inline_includes(text: str, base_dir: Path, depth: int = 0) -> str:
    """Recursively inline ``.include``/``.lib`` files.

    Mirrors ``simulation.engine._inline_includes`` for the same reason: the
    engine has to inline before LTspice sees the deck, so a reader that did not
    would parse a different circuit than the one that was simulated.
    """
    if depth > _MAX_INCLUDE_DEPTH:
        raise NetlistError("include nesting is too deep (cycle?)")
    lines: list[str] = []
    for raw in text.splitlines():
        match = _INCLUDE_RE.match(raw)
        if match is None:
            lines.append(raw)
            continue
        target = Path(match.group(1).strip('"'))
        if not target.is_absolute():
            target = (base_dir / target)
        if not target.is_file():
            raise NetlistError(f"included file not found: {target}")
        body = target.read_text(encoding="utf-8", errors="replace")
        lines.append(_inline_includes(body, target.parent, depth + 1))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _balanced(text: str, open_ch: str, close_ch: str) -> tuple[str, str]:
    """Split ``prefix(inner)suffix`` at the first balanced group."""
    start = text.find(open_ch)
    if start < 0:
        raise NetlistError(f"missing {open_ch!r} in {text[:60]!r}")
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[:start], text[start + 1:i]
    raise NetlistError(f"unbalanced {open_ch!r} in {text[:60]!r}")


def _numbers(text: str, where: str = "numbers") -> list[float]:
    """SPICE argument list as floats, suffixes included.

    Tokenising and then calling :func:`parse_value` -- rather than regex-
    scraping digits -- is deliberate: ``.tran 0 200u 0 0.1u`` must become
    200 us, not 200 s.  A regex that ignores the suffix would parse a deck the
    emitter wrote as plain floats but misread a perfectly legal hand-written
    one, and the resulting run would look plausible while being wrong.
    """
    tokens = [t for t in _ARG_SPLIT_RE.split(text.strip()) if t]
    values: list[float] = []
    for token in tokens:
        if token.upper() == "UIC":
            raise UnsupportedConstruct(
                "tran:uic", f"{where}: UIC skips the operating point and is "
                             "not modelled")
        try:
            values.append(parse_value(token))
        except ValueError:
            raise NetlistError(f"{where}: not a numeric argument {token!r}") from None
    return values


def _parse_source_waveform(rest: str, where: str) -> Waveform:
    """Waveform of a ``V``/``I`` card from the text after its two nodes."""
    head = rest.strip()
    upper = head.upper()
    if upper.startswith("PWL"):
        _, inner = _balanced(head, "(", ")")
        nums = _numbers(inner, where)
        if len(nums) < 4 or len(nums) % 2:
            raise NetlistError(f"{where}: PWL needs (t, v) pairs, got {inner!r}")
        return Waveform.pwl(zip(nums[0::2], nums[1::2]))
    if upper.startswith("PULSE"):
        _, inner = _balanced(head, "(", ")")
        nums = _numbers(inner, where)
        if not 2 <= len(nums) <= 7:
            raise NetlistError(
                f"{where}: PULSE takes v1 v2 [td tr tf pw per], got {inner!r}")
        return Waveform.pulse_wave(*nums)
    if upper.startswith("DC"):
        parts = head.split()
        if len(parts) < 2:
            raise NetlistError(f"{where}: DC needs a value")
        return Waveform.constant(parse_value(parts[1]))
    if upper.startswith("AC"):
        # Small-signal amplitude; the fast path does not run .ac, and the
        # transient solution is unchanged by it.
        return Waveform.constant(0.0)
    parts = head.split()
    if not parts:
        raise NetlistError(f"{where}: source has no value")
    return Waveform.constant(parse_value(parts[0]))


def _parse_model(rest: str, where: str) -> DeviceModel:
    """``<name> <family>(<k=v ...>)``."""
    name, _, tail = rest.partition(" ")
    tail = tail.strip()
    if not tail:
        raise NetlistError(f"{where}: .model {name!r} has no family")
    family, inner = _balanced(tail, "(", ")")
    family = family.strip()
    params: dict[str, str] = {}
    for token in inner.replace(",", " ").split():
        if "=" not in token:
            continue
        key, _, val = token.partition("=")
        params[key.strip().upper()] = val.strip()
    return DeviceModel(name=name, family=family.upper(), params=params)


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------

class _Reader:
    def __init__(self, source: str) -> None:
        self.source = source
        self.subcircuits: dict[str, Subcircuit] = {}
        self.models: dict[str, DeviceModel] = {}
        self.elements: list[Element] = []
        self.directives: list[AnalysisDirective] = []
        self.options: list[str] = []
        self.temperature: float | None = None
        self.expanded: list[str] = []

    # -- top level ---------------------------------------------------------

    def read(self, lines: list[tuple[int, str]]) -> Circuit:
        current: Subcircuit | None = None
        body: list[str] = []
        body_start = 0
        for lineno, line in lines:
            lower = line.lower()
            if lower.startswith(".subckt"):
                if current is not None:
                    raise NetlistError(
                        f"{self.source}:{lineno}: nested .subckt is not supported")
                name, pins = self._subckt_header(line, lineno)
                current = Subcircuit(name=name, pins=pins, body=(), origin=self.source)
                body = []
                body_start = lineno
                continue
            if lower.startswith(".ends"):
                if current is None:
                    raise NetlistError(f"{self.source}:{lineno}: stray .ends")
                self.subcircuits[current.name] = Subcircuit(
                    name=current.name, pins=current.pins,
                    body=tuple(body), origin=self.source)
                current = None
                continue
            if current is not None:
                body.append(line)
                continue
            if lower.startswith("."):
                if self._dot_card(line, lineno):
                    break
                continue
            if line[0].upper() == "X":
                self._flatten(line, lineno, node_map={}, prefix="")
                continue
            self.elements.append(
                self._element(line, lineno, node_map={}, prefix=""))

        if current is not None:
            raise NetlistError(
                f"{self.source}:{body_start}: .subckt {current.name!r} "
                "is never closed")

        nodes: list[str] = []
        for element in self.elements:
            nodes.extend(element.nodes)
        index = NodeIndex.build(nodes, ground=GROUND)
        circuit = Circuit(
            title=self.source,
            elements=tuple(self.elements),
            models=dict(self.models),
            node_index=index,
            directives=tuple(self.directives),
            options=tuple(self.options),
            expanded_subcircuits=tuple(dict.fromkeys(self.expanded)),
            temperature_c=self.temperature,
        )
        return circuit

    @staticmethod
    def _subckt_header(line: str, lineno: int) -> tuple[str, tuple[str, ...]]:
        tokens = line.split()[1:]
        if not tokens:
            raise NetlistError(f"line {lineno}: .subckt without a name")
        name, pins = tokens[0], tokens[1:]
        for token in pins:
            if token.upper() == "PARAMS:" or "=" in token:
                raise UnsupportedConstruct(
                    "subckt:PARAMS",
                    f"line {lineno}: parameterised subcircuits need a front end")
        if not pins:
            raise NetlistError(f"line {lineno}: .subckt {name!r} has no pins")
        return name, tuple(p.lower() for p in pins)

    def _dot_card(self, line: str, lineno: int) -> bool:
        """Handle one dot-card.  Returns True when it ends the deck."""
        head = line.split()[0].lower()
        rest = line[len(head):].strip()
        if head == ".end":
            return True
        if head == ".model":
            model = _parse_model(rest, f"{self.source}:{lineno}")
            existing = self.models.get(model.name)
            if existing is not None and existing != model:
                raise NetlistError(
                    f"{self.source}:{lineno}: .model {model.name!r} redefined "
                    "with different parameters")
            self.models[model.name] = model
            return False
        if head == ".tran":
            values = _numbers(rest, f"{self.source}:{lineno}")
            if len(values) < 2:
                raise NetlistError(f"line {lineno}: .tran needs a stop time")
            self.directives.append(AnalysisDirective(
                kind="tran", raw=line,
                step_time=values[0], end_time=values[1],
                start_time=values[2] if len(values) > 2 else 0.0,
                max_step=values[3] if len(values) > 3 else None))
            return False
        if head == ".op":
            self.directives.append(AnalysisDirective(kind="op", raw=line))
            return False
        if head == ".dc":
            self.directives.append(AnalysisDirective(kind="dc", raw=line))
            return False
        if head == ".temp":
            values = _numbers(rest, f"{self.source}:{lineno}")
            if not values:
                raise NetlistError(f"line {lineno}: .temp needs a value")
            self.temperature = values[0]
            return False
        if head == ".options" or head == ".option":
            self.options.append(line)
            return False
        if head == ".title":
            return False
        if head in _OUTPUT_CARDS:
            return False
        if head in _REJECTED_CARDS:
            raise UnsupportedConstruct(
                head, f"line {lineno}: changes the solution and is not modelled")
        if head in (".include", ".inc", ".lib"):
            raise NetlistError(
                f"line {lineno}: unresolved {head} (use read_deck to inline)")
        raise UnsupportedConstruct(head, f"line {lineno}: unknown dot-card")

    # -- elements ----------------------------------------------------------

    def _element(self, line: str, lineno: int, *,
                 node_map: Mapping[str, str], prefix: str) -> Element:
        name = line.split()[0]
        if not name[0].isalpha():
            raise NetlistError(f"{self.source}:{lineno}: bad element name {name!r}")
        letter = name[0].upper()
        kind = _READABLE.get(letter)
        if kind is None:
            raise UnsupportedConstruct(f"element:{letter}",
                                       f"{self.source}:{lineno}: {name}")
        rest = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""

        if kind is ElementKind.BEHAVIOURAL_SOURCE:
            element = self._behavioural(name, rest, lineno, prefix)
        elif kind in (ElementKind.DIODE, ElementKind.SWITCH):
            element = self._device(name, kind, rest, lineno, prefix)
        elif kind in (ElementKind.VSOURCE, ElementKind.ISOURCE):
            element = self._source(name, kind, rest, lineno, prefix)
        else:
            element = self._passive(name, kind, rest, lineno, prefix)
        return _remap_nodes(element, element.nodes, node_map, prefix, lineno)

    def _passive(self, name, kind, rest, lineno, prefix) -> Element:
        tokens = rest.split()
        if len(tokens) < 3:
            raise NetlistError(f"{self.source}:{lineno}: {name}: needs two nodes "
                               "and a value")
        a, b = tokens[0], tokens[1]
        tail = " ".join(tokens[2:])
        upper = tail.upper()
        if upper.startswith("Q"):
            expr = _split_assignment(tail)
            return Element(name=prefix + name, kind=ElementKind.CHARGE_CAPACITOR,
                           nodes=(a, b), expression=expr, source_line=lineno,
                           origin=self.source)
        if tail.startswith("{"):
            raise UnsupportedConstruct(
                "value:expression",
                f"{self.source}:{lineno}: {name} uses a parameterised value")
        value = parse_value(tokens[2])
        if len(tokens) > 3:
            raise UnsupportedConstruct(
                "value:qualifier",
                f"{self.source}:{lineno}: {name} has extra tokens {tokens[3:]}")
        return Element(name=prefix + name, kind=kind, nodes=(a, b), value=value,
                       source_line=lineno, origin=self.source)

    def _source(self, name, kind, rest, lineno, prefix) -> Element:
        tokens = rest.split(None, 2)
        if len(tokens) < 3:
            raise NetlistError(f"{self.source}:{lineno}: {name}: needs two nodes "
                               "and a waveform")
        a, b, tail = tokens
        return Element(name=prefix + name, kind=kind, nodes=(a, b),
                       waveform=_parse_source_waveform(
                           tail, f"{self.source}:{lineno}"),
                       source_line=lineno, origin=self.source)

    def _device(self, name, kind, rest, lineno, prefix) -> Element:
        tokens = rest.split()
        wanted = 2 if kind is ElementKind.DIODE else 4
        if len(tokens) != wanted + 1:
            raise NetlistError(
                f"{self.source}:{lineno}: {name}: {kind.value} needs {wanted} "
                f"nodes and a model, got {tokens}")
        return Element(name=prefix + name, kind=kind, nodes=tuple(tokens[:wanted]),
                       model=tokens[wanted], source_line=lineno, origin=self.source)

    def _behavioural(self, name, rest, lineno, prefix) -> Element:
        # Both spellings occur in the emitted libs: ``VALUE = {..}`` (the Tier-2
        # emitter) and ``VALUE={..}`` (the Tier-3 emitter).  Split the nodes
        # first, then require the tail to start with VALUE so a controlled
        # source (``G1 a b c d 1m``) is rejected instead of misread.
        tokens = rest.split(None, 2)
        if len(tokens) < 3 or not tokens[2].upper().startswith("VALUE"):
            raise UnsupportedConstruct(
                "element:G",
                f"{self.source}:{lineno}: only G.. VALUE={{..}} is represented")
        a, b, tail = tokens
        expr = _split_assignment(tail)
        return Element(name=prefix + name, kind=ElementKind.BEHAVIOURAL_SOURCE,
                       nodes=(a, b), expression=expr, source_line=lineno,
                       origin=self.source)

    # -- subcircuits -------------------------------------------------------

    def _flatten(self, line: str, lineno: int, *,
                 node_map: Mapping[str, str], prefix: str) -> None:
        tokens = line.split()[1:]
        if len(tokens) < 2:
            raise NetlistError(f"{self.source}:{lineno}: {line.split()[0]}: "
                               "instance needs nodes and a subcircuit name")
        if any("=" in t for t in tokens):
            raise UnsupportedConstruct(
                "subckt:params",
                f"{self.source}:{lineno}: parameter overrides need a front end")
        subckt_name = tokens[-1]
        outer_nodes = [node_map.get(n, n) for n in tokens[:-1]]
        sub = self.subcircuits.get(subckt_name)
        if sub is None:
            raise NetlistError(
                f"{self.source}:{lineno}: undefined subcircuit {subckt_name!r}")
        if len(outer_nodes) != len(sub.pins):
            raise NetlistError(
                f"{self.source}:{lineno}: {subckt_name} takes {len(sub.pins)} "
                f"pins, instance supplies {len(outer_nodes)}")
        if len(prefix.split(".")) - 1 > _MAX_SUBCKT_DEPTH:
            raise NetlistError(f"{self.source}:{lineno}: subcircuit nesting too deep")
        ref = line.split()[0]
        inner_prefix = f"{prefix}{ref}."
        pin_map = {pin: node for pin, node in zip(sub.pins, outer_nodes)}
        for inner_line in sub.body:
            if inner_line.lstrip().startswith("."):
                self._body_dot_card(inner_line, lineno)
                continue
            if inner_line[0].upper() == "X":
                self._flatten(inner_line, lineno, node_map=pin_map,
                              prefix=inner_prefix)
                continue
            self.elements.append(
                self._element(inner_line, lineno, node_map=pin_map,
                              prefix=inner_prefix))

    def _body_dot_card(self, line: str, lineno: int) -> None:
        """A dot-card inside a ``.SUBCKT`` body.

        The emitted ROMs declare their ``.MODEL`` cards inside the subcircuit,
        so the body is not purely elements.  LTspice treats those models as
        global; registering them here keeps the flattened circuit faithful,
        and the same-parameters check makes a second instance harmless.
        """
        head = line.split()[0].lower()
        if head != ".model":
            raise UnsupportedConstruct(
                head, f"{self.source}:{lineno}: dot-card inside .subckt")
        model = _parse_model(line[len(line.split()[0]):].strip(),
                             f"{self.source}:{lineno}")
        existing = self.models.get(model.name)
        if existing is not None and existing != model:
            raise NetlistError(
                f"{self.source}:{lineno}: .model {model.name!r} redefined "
                "with different parameters")
        self.models[model.name] = model


def _split_assignment(text: str) -> str:
    """``Q = {expr}`` / ``VALUE=(expr)`` -> the verbatim expression text.

    The expression is kept as written.  It is provenance for a construct the
    kernel lowers from tables, never something it evaluates.

    Both grouping forms occur in this project's own emitters -- the Tier-2 and
    UROM builders write ``VALUE = {...}``, the simplis hybrid candidates write
    ``VALUE=(...)`` -- so both are accepted and stored verbatim.
    """
    if "=" not in text:
        raise NetlistError(f"expected '=' in {text[:60]!r}")
    _, _, tail = text.partition("=")
    tail = tail.strip()
    if tail.startswith("{"):
        _balanced(tail, "{", "}")
    elif tail.startswith("("):
        _balanced(tail, "(", ")")
    else:
        raise UnsupportedConstruct(
            "value:expression",
            f"expected a braced or parenthesised expression, got {tail[:40]!r}")
    return tail


def _remap_nodes(element: Element, nodes: Iterable[str],
                 node_map: Mapping[str, str], prefix: str,
                 lineno: int) -> Element:
    """Resolve subcircuit-local node names against the instance pin map."""
    resolved = tuple(node_map.get(n, f"{prefix}{n}").lower() for n in nodes)
    return Element(name=element.name, kind=element.kind, nodes=resolved,
                   value=element.value, waveform=element.waveform,
                   model=element.model, expression=element.expression,
                   source_line=lineno, origin=element.origin)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def parse(text: str, *, source: str = "<netlist>") -> ParseResult:
    """Parse netlist *text*.  ``.include`` must already be inlined.

    **The first line is the title and is never parsed.**  That is the SPICE
    convention, and LTspice follows it, so honouring it is not pedantry: a deck
    without a title line would otherwise be read as a *different circuit* than
    the one LTspice simulated, and every accuracy and speed comparison would be
    measuring the difference between two circuits rather than between two
    engines.  Every deck this project emits starts with a ``*`` title, so
    nothing changes for them.
    """
    reader = _Reader(source)
    physical = text.splitlines()
    body = "\n".join(physical[1:]) if physical else ""
    circuit = reader.read(_strip_comments(body))
    if not circuit.elements and not reader.subcircuits:
        raise NetlistError(
            f"{source}: nothing was read -- no elements and no subcircuits. "
            "SPICE treats the first line as the title, so a netlist must start "
            "with a title or comment line")
    return ParseResult(circuit=circuit, subcircuits=dict(reader.subcircuits),
                       source=source)


def read_deck(path: str | Path) -> ParseResult:
    """Parse a netlist file, inlining ``.include``/``.lib`` exactly as the
    LTspice backend does before it hands the deck to the simulator."""
    file = Path(path)
    text = file.read_text(encoding="utf-8", errors="replace")
    return parse(_inline_includes(text, file.parent), source=str(file))


def parse_circuit(text: str, *, source: str = "<netlist>") -> Circuit:
    """Convenience: the flattened circuit only."""
    return parse(text, source=source).circuit
