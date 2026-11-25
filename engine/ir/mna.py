"""Structural modified nodal analysis.

Only the *structure* is built here: which unknowns exist and which matrix
positions each element writes to.  Numbers are the companion models' job
(:mod:`engine.solver.companion`).  Keeping the two apart has one concrete
payoff -- the structure can be verified once against hand-derived matrices and
then trusted by every time step, instead of being re-derived per stamp.

Unknown ordering is deterministic and frozen at assembly time:

1. node voltages, in the order nodes first appear in the deck;
2. branch currents for voltage sources and inductors, in element order.

Determinism matters here more than anywhere else in the engine: two runs of the
same deck must produce bit-identical matrices, otherwise a solver-vs-solver
comparison measures ordering noise.

Ground (``0``) has no unknown.  A two-terminal element with one grounded node
therefore contributes only the rows/columns of its other node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.ir.circuit import Circuit, Element, ElementKind, NodeIndex

#: Elements that need a branch current unknown.
_BRANCHED: frozenset[ElementKind] = frozenset(
    {ElementKind.VSOURCE, ElementKind.INDUCTOR})

#: Elements represented by a two-terminal conductance (possibly nonlinear in
#: value, but structurally a conductance).
_CONDUCTIVE: frozenset[ElementKind] = frozenset(
    {
        ElementKind.RESISTOR,
        ElementKind.CAPACITOR,
        ElementKind.CHARGE_CAPACITOR,
        ElementKind.DIODE,
    })


@dataclass(frozen=True, slots=True)
class Stamp:
    """One element's resolved position in the MNA system.

    ``nodes`` are unknown indices in the element's own node order, with
    ``None`` for ground.  ``branch`` is the row/column of its branch current,
    for elements that need one.  ``control`` holds a switch's control pair.
    """

    element: Element
    nodes: tuple[int | None, ...]
    branch: int | None = None
    control: tuple[int | None, ...] = ()

    @property
    def kind(self) -> ElementKind:
        return self.element.kind

    @property
    def name(self) -> str:
        return self.element.name


@dataclass(frozen=True, slots=True)
class MnaStructure:
    """The MNA system's shape, and where each element writes.

    ``dense`` lists behavioural sources: their expression may reference any
    node, so their structural sparsity is genuinely unbounded.  They are named
    rather than expanded so :meth:`pattern` stays a truthful statement about
    the elements whose connectivity *is* known.
    """

    node_index: NodeIndex
    branches: tuple[tuple[str, int], ...]
    stamps: tuple[Stamp, ...]
    size: int

    @property
    def dense(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.stamps
                     if s.kind is ElementKind.BEHAVIOURAL_SOURCE)

    def stamp(self, name: str) -> Stamp:
        for stamp in self.stamps:
            if stamp.name == name:
                return stamp
        raise KeyError(f"no stamp for element {name!r}")

    def branch_of(self, name: str) -> int:
        for element_name, index in self.branches:
            if element_name == name:
                return index
        raise KeyError(f"{name!r} has no branch unknown")

    def unknown_count(self) -> tuple[int, int]:
        """``(nodes, branches)`` -- for asserting against a hand-derived count."""
        return len(self.node_index), len(self.branches)

    def pattern(self, *, include_dense: bool = False) -> np.ndarray:
        """Structural non-zero count per matrix entry.

        Exact for every element whose connectivity is known.  Behavioural
        sources are skipped unless *include_dense*, in which case they fill the
        whole matrix -- which is what "may reference any node" means.
        """
        matrix = np.zeros((self.size, self.size), dtype=int)
        for stamp in self.stamps:
            if stamp.kind is ElementKind.BEHAVIOURAL_SOURCE:
                if include_dense:
                    matrix += 1
                continue
            for row, col in _entries(stamp):
                matrix[row, col] += 1
        return matrix


def _entries(stamp: Stamp) -> list[tuple[int, int]]:
    """Matrix positions one element contributes to (ground entries dropped)."""
    nodes = stamp.nodes
    kind = stamp.kind
    out: list[tuple[int, int]] = []

    def add(row: int | None, col: int | None) -> None:
        if row is not None and col is not None:
            out.append((row, col))

    if kind in _CONDUCTIVE:
        a, b = nodes[0], nodes[1]
        for row in (a, b):
            for col in (a, b):
                add(row, col)
        return out

    if kind is ElementKind.SWITCH:
        # A voltage-controlled switch draws no current at its control terminals,
        # so the control pair selects the conductance but stamps nothing.
        a, b = nodes[0], nodes[1]
        for row in (a, b):
            for col in (a, b):
                add(row, col)
        return out

    if kind is ElementKind.INDUCTOR:
        a, b = nodes[0], nodes[1]
        k = stamp.branch
        assert k is not None
        add(a, k)
        add(k, a)
        add(b, k)
        add(k, b)
        add(k, k)                      # -L di/dt on the branch equation
        return out

    if kind is ElementKind.VSOURCE:
        a, b = nodes[0], nodes[1]
        k = stamp.branch
        assert k is not None
        add(a, k)
        add(k, a)
        add(b, k)
        add(k, b)
        return out

    if kind is ElementKind.ISOURCE:
        return out                      # right-hand side only

    raise ValueError(f"no structural stamp for {kind!r}")


def build(circuit: Circuit) -> MnaStructure:
    """Resolve every element to matrix positions.

    A ``NotYetImplemented``/``UnsupportedConstruct`` is *not* raised here: the
    structure is a description of the deck, and a deck can be described before
    the kernel can evaluate it.  Capability is checked when companions are
    built.
    """
    branches: list[tuple[str, int]] = []
    for element in circuit.elements:
        if element.kind in _BRANCHED:
            branches.append((element.name, len(branches)))
    offset = len(circuit.node_index)
    branch_of = dict(branches)

    stamps: list[Stamp] = []
    for element in circuit.elements:
        nodes = tuple(circuit.node_index.index(n) for n in element.nodes)
        control: tuple[int | None, ...] = ()
        if element.kind is ElementKind.SWITCH:
            control = (nodes[2], nodes[3])
        stamps.append(Stamp(
            element=element,
            nodes=nodes,
            branch=(offset + branch_of[element.name]
                    if element.name in branch_of else None),
            control=control,
        ))

    return MnaStructure(
        node_index=circuit.node_index,
        branches=tuple((name, offset + index)
                       for name, index in branches),
        stamps=tuple(stamps),
        size=offset + len(branches),
    )
