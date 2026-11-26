"""The 2-D piecewise-linear channel: how a ROM's Id table becomes solvable.

The Tier-2 model is *emitted* for SPICE as ``G_CH VALUE={ ... tanh(Vds/s) ... }``
-- smooth, nonlinear, and therefore still Newton work for whoever solves it.
For this engine the same fit is carried as the table it was fitted from
(``PWLModelSpec.breakpoints_id`` on its ``vgs_values`` x ``breakpoints_vds``
grid).  Same spec, two emitters: that is the whole point of the ROM contract.

Each grid cell is split along its diagonal into two triangles.  Within a
triangle the current is a **plane** -- ``Id = a0 + a1*Vgs + a2*Vds`` -- so it
stamps as a linear controlled source with no iteration.  A bilinear
interpolation would carry a ``Vgs*Vds`` cross term and be nonlinear; the
triangulation is what makes the surface piecewise *linear* rather than merely
piecewise smooth, which is the difference between a PWL engine and a Newton
one.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from engine.solver.device import PwlDevice


@dataclass(frozen=True, slots=True)
class ChannelTable:
    """``Id(Vgs, Vds)`` on a rectangular grid, planar per triangle."""

    vgs: tuple[float, ...]
    vds: tuple[float, ...]
    id: tuple[tuple[float, ...], ...]
    #: Per-node transconductance and output conductance, when the basis can
    #: supply them.  With them the triangle's plane uses the *true local slope*
    #: instead of the chord through three corners, which is the difference
    #: between a secant and a tangent over the same cell.
    gm: tuple[tuple[float, ...], ...] = ()
    gds: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if len(self.vgs) < 2 or len(self.vds) < 2:
            raise ValueError("a channel table needs at least a 2x2 grid")
        if any(b <= a for a, b in zip(self.vgs, self.vgs[1:])):
            raise ValueError("Vgs axis must be strictly ascending")
        if any(b <= a for a, b in zip(self.vds, self.vds[1:])):
            raise ValueError("Vds axis must be strictly ascending")
        if len(self.id) != len(self.vgs):
            raise ValueError("Id rows must match the Vgs axis")
        if any(len(row) != len(self.vds) for row in self.id):
            raise ValueError("every Id row must match the Vds axis")

    @classmethod
    def build(cls, vgs: Sequence[float], vds: Sequence[float],
              id_rows: Sequence[Sequence[float]],
              gm_rows: Sequence[Sequence[float]] = (),
              gds_rows: Sequence[Sequence[float]] = ()) -> ChannelTable:
        return cls(
            vgs=tuple(float(v) for v in vgs),
            vds=tuple(float(v) for v in vds),
            id=tuple(tuple(float(i) for i in row) for row in id_rows),
            gm=tuple(tuple(float(i) for i in row) for row in gm_rows),
            gds=tuple(tuple(float(i) for i in row) for row in gds_rows),
        )

    @property
    def n_cells(self) -> int:
        return (len(self.vgs) - 1) * (len(self.vds) - 1)

    # -- location ----------------------------------------------------------

    def cell(self, vgs: float, vds: float) -> tuple[int, int, int]:
        """``(i, j, triangle)`` for a point, clamped to the grid.

        Outside the fitted span the point is clamped to the edge cell -- the
        same "physical end segment" rule the emitted ROMs use, so an
        overshooting transient saturates instead of extrapolating a plane into
        an unmeasured region.
        """
        i = min(max(bisect.bisect_right(self.vgs, vgs) - 1, 0), len(self.vgs) - 2)
        j = min(max(bisect.bisect_right(self.vds, vds) - 1, 0), len(self.vds) - 2)
        g0, g1 = self.vgs[i], self.vgs[i + 1]
        d0, d1 = self.vds[j], self.vds[j + 1]
        u = 0.0 if vgs <= g0 else 1.0 if vgs >= g1 else (vgs - g0) / (g1 - g0)
        v = 0.0 if vds <= d0 else 1.0 if vds >= d1 else (vds - d0) / (d1 - d0)
        # Split along the (i, j) -> (i+1, j+1) diagonal: triangle 0 is the
        # region above the diagonal in index space, triangle 1 below it.
        triangle = 1 if v > u else 0
        return i, j, triangle

    def region(self, vgs: float, vds: float) -> int:
        i, j, triangle = self.cell(vgs, vds)
        return (i * (len(self.vds) - 1) + j) * 2 + triangle

    # -- the plane ---------------------------------------------------------

    def plane(self, region: int) -> tuple[float, float, float]:
        """``(i0, dI/dVgs, dI/dVds)`` of the region's plane.

        The triangle corners carry the interpolation; two corners differ in one
        index only, so the gradient in index space is a plain difference and
        the conversion to volts is a division by the axis spacing.
        """
        per_axis = len(self.vds) - 1
        triangle = region % 2
        cell = region // 2
        i, j = divmod(cell, per_axis)
        dg = self.vgs[i + 1] - self.vgs[i]
        dd = self.vds[j + 1] - self.vds[j]
        i00 = self.id[i][j]
        i10 = self.id[i + 1][j]
        i01 = self.id[i][j + 1]
        i11 = self.id[i + 1][j + 1]
        if self.gm and self.gds:
            # Corner-value plane replaced by a tangent-matched plane: the slope
            # is the mean of the three corner gradients and the value is pinned
            # at the centroid.  Same storage, same runtime shape -- only the
            # numbers change.
            if triangle == 0:
                corners = ((i, j), (i + 1, j), (i + 1, j + 1))
            else:
                corners = ((i, j), (i, j + 1), (i + 1, j + 1))
            a1 = sum(self.gm[a][b] for a, b in corners) / 3.0
            a2 = sum(self.gds[a][b] for a, b in corners) / 3.0
            vg_mean = sum(self.vgs[a] for a, _ in corners) / 3.0
            vd_mean = sum(self.vds[b] for _, b in corners) / 3.0
            i_mean = sum(self.id[a][b] for a, b in corners) / 3.0
            return (i_mean - a1 * vg_mean - a2 * vd_mean, a1, a2)
        if triangle == 0:
            # corners (0,0), (1,0), (1,1)
            d_i_du = i10 - i00
            d_i_dv = i11 - i10
        else:
            # corners (0,0), (0,1), (1,1)
            d_i_du = i11 - i01
            d_i_dv = i01 - i00
        a1 = d_i_du / dg
        a2 = d_i_dv / dd
        a0 = i00 - a1 * self.vgs[i] - a2 * self.vds[j]
        return a0, a1, a2

    def clamp(self, vgs: float, vds: float) -> tuple[float, float, bool, bool]:
        """Clamped point plus whether each axis was inside the fitted span.

        The emitted ROM saturates beyond its last breakpoint instead of
        extrapolating, and a solver table must do the same: a plane extended
        past 15 V into a 100 V overshoot produces a current the device never
        showed, and it looks perfectly smooth while doing it.
        """
        vgs_c = min(max(vgs, self.vgs[0]), self.vgs[-1])
        vds_c = min(max(vds, self.vds[0]), self.vds[-1])
        return vgs_c, vds_c, vgs_c == vgs, vds_c == vds

    def current(self, vgs: float, vds: float) -> float:
        vgs_c, vds_c, _, _ = self.clamp(vgs, vds)
        a0, a1, a2 = self.plane(self.region(vgs_c, vds_c))
        return a0 + a1 * vgs_c + a2 * vds_c

    def is_monotone_in_vds(self) -> bool:
        """Non-decreasing drain current along every Vds row."""
        return all(b >= a for row in self.id for a, b in pairwise(row))


class ChannelDevice(PwlDevice):
    """The ROM channel as a controlled current source, drain -> source.

    ``nodes`` are ``(drain, source, gate)``; the control voltage is measured
    gate-to-source, matching the emitted subcircuit's ``V(gi,ss)``.  The stamp
    is asymmetric (a controlled source, not a conductance), which is why it is
    written out rather than reusing the conductance helper.
    """

    discontinuous = False

    def __init__(self, name: str, nodes: tuple[int | None, ...],
                 table: ChannelTable) -> None:
        super().__init__(name, nodes)
        self.table = table
        # Last point the region was chosen at; the stamp needs it to decide
        # whether an axis is inside the fitted span (gradient) or beyond it
        # (saturated, zero gradient).
        self._last: tuple[float, float] = (table.vgs[0], table.vds[0])

    def _control(self, x: np.ndarray) -> tuple[float, float]:
        d, s, g = self.nodes[0], self.nodes[1], self.nodes[2]
        vd = 0.0 if d is None else float(x[d])
        vs = 0.0 if s is None else float(x[s])
        vg = 0.0 if g is None else float(x[g])
        return vg - vs, vd - vs

    def region(self, x: np.ndarray) -> int:
        vgs, vds = self._control(x)
        self._last = (vgs, vds)
        return self.table.region(vgs, vds)

    def current(self, x: np.ndarray) -> float:
        vgs, vds = self._control(x)
        return self.table.current(vgs, vds)

    def stamp(self, region: int, *, G: np.ndarray, rhs: np.ndarray,
              h: float) -> None:
        a0, a1, a2 = self.table.plane(region)
        vgs, vds, g_inside, d_inside = self.table.clamp(*self._last)
        if not g_inside:                       # beyond the Vgs span: saturate
            a0 += a1 * vgs
            a1 = 0.0
        if not d_inside:                       # beyond the Vds span: saturate
            a0 += a2 * vds
            a2 = 0.0
        d, s, g = self.nodes[0], self.nodes[1], self.nodes[2]
        # Id = a0 + a1*(Vg - Vs) + a2*(Vd - Vs), flowing d -> s.
        def add(row: int | None, col: int | None, value: float) -> None:
            if row is not None and col is not None:
                G[row, col] += value

        for row, sign in ((d, 1.0), (s, -1.0)):
            if row is None:
                continue
            if g is not None:
                add(row, g, sign * a1)
            if d is not None:
                add(row, d, sign * a2)
            if s is not None:
                add(row, s, -sign * (a1 + a2))
            rhs[row] -= sign * a0
