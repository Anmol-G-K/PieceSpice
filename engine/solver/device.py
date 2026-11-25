"""Piecewise-linear devices: the reason the fast path can exist.

Each device here is a *finite set of linear regions*.  Inside a region the
device stamps exactly like a linear element, so a step costs one linear solve
and no Jacobian.  Two rules follow from that and they shape every class below:

**1. Regions are selected, never iterated on.** A device exposes
``region(x)`` -- the region implied by a solution vector.  The solver solves,
re-reads the regions, and re-solves only if the set changed.  That is a
fixed-point over a *finite discrete* set, so it terminates or is declared
chattering; it is not Newton and it cannot diverge.

**2. Only discontinuous devices truncate a step.**  A switch turning on is a
step change in conductance: the instant it happens is a real event, the state
jumps, and an edge-timing measurement depends on locating it exactly.  A diode
crossing a table breakpoint is *not* discontinuous -- the current is
continuous across it -- so those crossings are absorbed by the region
fixed-point instead of triggering a truncated step.  Localising every table
breakpoint would multiply the solve count for no accuracy gain.

The one approximation that is unavoidable, and is reported rather than hidden:
a diode is piecewise-linear, so it cannot be exponential.  ``PWL_DIODE``'s
segment count is a knob, and the resulting deviation from the Shockley curve
is measured in the tests instead of being assumed small.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

#: Boltzmann constant times electron charge, V/K (k_B / q).
_KB_OVER_Q = 8.617333262e-5

#: Default simulator temperature when the deck does not carry a `.temp` card
#: (LTspice's own default).
DEFAULT_TEMPERATURE_C = 27.0


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PwlTable:
    """A monotone piecewise-linear function ``y(x)``.

    Stored as points so the table is self-describing; ``region``/``slope``/
    ``intercept`` are the solver-facing view.  ``x`` must be strictly
    ascending, and the table extends with zero slope beyond its ends -- the
    "physical end segment" rule the shipped ROMs already use, so a transient
    that overshoots the fitted span saturates instead of extrapolating.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[float, float]]) -> PwlTable:
        pts = sorted((float(a), float(b)) for a, b in pairs)
        xs = tuple(p[0] for p in pts)
        if len(xs) < 2:
            raise ValueError("a PWL table needs at least two points")
        if any(b <= a for a, b in pairwise(xs)):
            raise ValueError("table x values must be strictly ascending")
        return cls(x=xs, y=tuple(p[1] for p in pts))

    def __len__(self) -> int:
        return len(self.x) - 1

    def region(self, value: float) -> int:
        """Index of the segment containing *value*, clamped to the table.

        The bracket is over point indices (``0 .. len(x)-1``), then clamped to
        the segment count: bracketing over segments directly loses the last one
        for any value past the second-to-last point.
        """
        if value <= self.x[0]:
            return 0
        if value >= self.x[-1]:
            return len(self) - 1
        lo, hi = 0, len(self.x) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.x[mid] <= value:
                lo = mid
            else:
                hi = mid
        return min(lo, len(self) - 1)

    def bounds(self, region: int) -> tuple[float, float]:
        return self.x[region], self.x[region + 1]

    def slope(self, region: int) -> float:
        x0, x1 = self.bounds(region)
        y0, y1 = self.y[region], self.y[region + 1]
        return (y1 - y0) / (x1 - x0)

    def intercept(self, region: int) -> float:
        x0, y0 = self.x[region], self.y[region]
        return y0 - self.slope(region) * x0

    def at(self, region: int, value: float) -> float:
        return self.slope(region) * value + self.intercept(region)

    def is_monotone_increasing(self) -> bool:
        return all(b >= a for a, b in pairwise(self.y))


@dataclass(frozen=True, slots=True)
class ChargeTable:
    """Q(V) for a charge-defined capacitor.  ``q`` must be non-decreasing.

    Non-monotone Q would mean a negative capacitance somewhere, which is not a
    physical device -- the shipped ROMs already assert monotone charge, and the
    solver refuses to invent one.
    """

    table: PwlTable

    def __post_init__(self) -> None:
        if not self.table.is_monotone_increasing():
            raise ValueError(
                "charge table must be non-decreasing in Q (negative "
                "incremental capacitance is not physical)")
        if self.table.y[0] != 0.0:
            raise ValueError(
                "charge table must be zero-anchored (Q(0) = 0) so switching "
                "loss is not invented by an absolute offset")


# ---------------------------------------------------------------------------
# Device protocol
# ---------------------------------------------------------------------------

class PwlDevice:
    """Base class: a finite set of linear regions over the node voltages.

    Subclasses implement :meth:`region`, :meth:`stamp` and, for devices whose
    region change is a genuine discontinuity, :meth:`control` +
    ``thresholds``/``hysteresis`` so the event finder can locate the instant.
    """

    #: True when a region change is a step discontinuity in conductance.
    discontinuous: bool = False

    def __init__(self, name: str, nodes: tuple[int | None, ...]) -> None:
        self.name = name
        self.nodes = nodes

    def voltage(self, x: np.ndarray) -> float:
        a, b = self.nodes[0], self.nodes[1]
        va = 0.0 if a is None else float(x[a])
        vb = 0.0 if b is None else float(x[b])
        return va - vb

    def region(self, x: np.ndarray) -> int:            # pragma: no cover - abstract
        raise NotImplementedError

    def stamp(self, region: int, *, G: np.ndarray, rhs: np.ndarray,
              h: float) -> None:                        # pragma: no cover - abstract
        raise NotImplementedError

    def current(self, x: np.ndarray) -> float:      # pragma: no cover - abstract
        """Terminal current from the first node to the second, in amperes."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear per-device history (charge, previous region)."""

    def update(self, x: np.ndarray) -> None:
        """Store state after an accepted step."""


# ---------------------------------------------------------------------------
# Diode
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DiodeModel:
    """``.model D(IS N RS)`` plus the PWL table built from it."""

    isat: float = 1e-14
    n: float = 1.0
    rs: float = 0.0
    table: PwlTable = field(default_factory=lambda: PwlTable((0.0, 1.0), (0.0, 0.0)))

    @staticmethod
    def build(*, isat: float = 1e-14, n: float = 1.0, rs: float = 0.0,
              temperature_c: float = DEFAULT_TEMPERATURE_C,
              decades: int = 6, per_decade: int = 6,
              i_max: float = 1e3) -> DiodeModel:
        """Tabulate Shockley + series resistance onto a PWL grid.

        The grid is chosen in *current* decades: a diode conducts over many
        decades and a uniform voltage grid would put every segment in either
        the flat reverse region or deep forward conduction, missing the knee
        where the switching actually happens.
        """
        if isat <= 0.0:
            raise ValueError("diode IS must be positive")
        vt = _KB_OVER_Q * (temperature_c + 273.15)
        nvt = max(n, 1e-6) * vt
        i_start = isat * 10.0
        i_stop = max(i_max, i_start * 10.0)
        counts = decades * per_decade + 1
        currents = np.geomspace(i_start, i_stop, counts)
        points: list[tuple[float, float]] = [
            (-1.0, -isat),                      # deep reverse: flat
            (0.0, 0.0),
        ]
        for i in currents:
            vd = nvt * math.log1p(i / isat)
            points.append((vd + i * rs, float(i)))
        return DiodeModel(isat=isat, n=n, rs=rs,
                          table=PwlTable.from_pairs(points))


class DiodeDevice(PwlDevice):
    """A diode as a PWL current source: ``i = g_region * v + b_region``.

    Two sources of table: the parametric ``.model D(IS N RS)`` of a testbench,
    and the measured Curve-6 ``(Vsd, Isd)`` points a ROM carries as its body
    diode.  Both are the same element once tabulated, which is the point.
    """

    discontinuous = False

    def __init__(self, name: str, nodes: tuple[int | None, ...],
                 table: PwlTable) -> None:
        super().__init__(name, nodes)
        self.table = table

    @classmethod
    def from_model(cls, name: str, nodes: tuple[int | None, ...],
                   model: DiodeModel) -> DiodeDevice:
        return cls(name, nodes, model.table)

    def region(self, x: np.ndarray) -> int:
        return self.table.region(self.voltage(x))

    def current(self, x: np.ndarray) -> float:
        region = self.region(x)
        return (self.table.slope(region) * self.voltage(x)
                + self.table.intercept(region))

    def stamp(self, region: int, *, G: np.ndarray, rhs: np.ndarray,
              h: float) -> None:
        table = self.table
        g = table.slope(region)
        b = table.intercept(region)
        a, c = self.nodes[0], self.nodes[1]
        if a is not None:
            G[a, a] += g
            if c is not None:
                G[a, c] -= g
        if c is not None:
            G[c, c] += g
            if a is not None:
                G[c, a] -= g
        # Constant part of the segment current, flowing a -> c.
        if a is not None:
            rhs[a] -= b
        if c is not None:
            rhs[c] += b


# ---------------------------------------------------------------------------
# Voltage-controlled switch
# ---------------------------------------------------------------------------

class SwitchDevice(PwlDevice):
    """``.model SW(Ron Roff Vt Vh)`` -- two conductances and one threshold.

    This is the one device whose region change is a genuine discontinuity, so
    it is the one that truncates a step.  Hysteresis (``Vh``) is what keeps a
    gate hovering at the threshold from generating an event on every step;
    without it the solver chatters, which is the failure mode the feasibility
    review names explicitly.
    """

    discontinuous = True

    def __init__(self, name: str, nodes: tuple[int | None, ...], *,
                 ron: float, roff: float, vt: float, vh: float = 0.0) -> None:
        super().__init__(name, nodes)
        if ron <= 0.0 or roff <= 0.0:
            raise ValueError(f"{name}: Ron/Roff must be positive")
        self.ron = ron
        self.roff = roff
        self.vt = vt
        self.vh = max(vh, 0.0)
        self._on = False

    @classmethod
    def from_model(cls, name: str, nodes: tuple[int | None, ...],
                   params: dict[str, float]) -> SwitchDevice:
        return cls(name, nodes,
                   ron=float(params.get("ron", 1.0)),
                   roff=float(params.get("roff", 1e12)),
                   vt=float(params.get("vt", 0.0)),
                   vh=float(params.get("vh", 0.0)))

    def control(self, x: np.ndarray) -> float:
        """Gate-source voltage the threshold is compared against."""
        c = self.nodes[2]
        d = self.nodes[3]
        vc = 0.0 if c is None else float(x[c])
        vd = 0.0 if d is None else float(x[d])
        return vc - vd

    def region(self, x: np.ndarray) -> int:
        vgs = self.control(x)
        # Schmit trigger: a wider band separates ON from OFF.
        if self._on:
            if vgs < self.vt - 0.5 * self.vh:
                self._on = False
        else:
            if vgs > self.vt + 0.5 * self.vh:
                self._on = True
        return int(self._on)

    def force_region(self, region: int) -> None:
        self._on = bool(region)

    def event_threshold(self) -> float:
        """Control level the state is about to cross.

        Turning on means crossing ``Vt + Vh/2`` upwards; turning off means
        crossing ``Vt - Vh/2`` downwards.  The event finder needs the level for
        the *current* state, not the model's nominal ``Vt``, otherwise the
        located instant sits half the hysteresis band off.
        """
        return self.vt - 0.5 * self.vh if self._on else self.vt + 0.5 * self.vh

    def current(self, x: np.ndarray) -> float:
        g = 1.0 / (self.ron if self.region(x) else self.roff)
        return g * self.voltage(x)

    def stamp(self, region: int, *, G: np.ndarray, rhs: np.ndarray,
              h: float) -> None:
        g = 1.0 / (self.ron if region else self.roff)
        a, c = self.nodes[0], self.nodes[1]
        if a is not None:
            G[a, a] += g
            if c is not None:
                G[a, c] -= g
        if c is not None:
            G[c, c] += g
            if a is not None:
                G[c, a] -= g

    def reset(self) -> None:
        self._on = False


# ---------------------------------------------------------------------------
# Charge-defined capacitor
# ---------------------------------------------------------------------------

class ChargeCapDevice(PwlDevice):
    """``C .. Q = {q(V)}`` as a table, not an expression.

    The companion uses the *charge* form directly: ``i = (Q(Vn) - Q_prev) / h``.
    Inside a Q segment that is ``i = (c/h) Vn + (q0 - Q_prev)/h``, so it stamps
    as a conductance ``c/h`` plus a source -- still linear, still no derivative
    to re-evaluate.  Working from charge rather than capacitance is what keeps
    switching loss physical under a large drain swing.
    """

    discontinuous = False

    def __init__(self, name: str, nodes: tuple[int | None, ...],
                 charge: ChargeTable) -> None:
        super().__init__(name, nodes)
        self.charge = charge
        self._q_prev = 0.0

    def region(self, x: np.ndarray) -> int:
        return self.charge.table.region(self.voltage(x))

    def stamp(self, region: int, *, G: np.ndarray, rhs: np.ndarray,
              h: float) -> None:
        table = self.charge.table
        c = table.slope(region)                 # dQ/dV inside this segment
        q0 = table.intercept(region)            # Q = c*V + q0
        g = c / h
        b = (q0 - self._q_prev) / h             # constant current part
        a, cnode = self.nodes[0], self.nodes[1]
        if a is not None:
            G[a, a] += g
            rhs[a] -= b
            if cnode is not None:
                G[a, cnode] -= g
        if cnode is not None:
            G[cnode, cnode] += g
            rhs[cnode] += b
            if a is not None:
                G[cnode, a] -= g

    def current(self, x: np.ndarray) -> float:
        # Pointwise value is not meaningful without the previous charge; the
        # kernel derives it from consecutive samples instead.
        raise NotImplementedError(
            "charge-capacitor current comes from the charge history")

    def update(self, x: np.ndarray) -> None:
        region = self.region(x)
        self._q_prev = self.charge.table.at(region, self.voltage(x))

    def reset(self) -> None:
        self._q_prev = 0.0

    @property
    def charge_prev(self) -> float:
        return self._q_prev
