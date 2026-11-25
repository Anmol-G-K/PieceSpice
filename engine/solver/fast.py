"""Compiled step kernel: the interpreter cost was the whole problem.

The reference implementation in :mod:`engine.solver.transient` is readable and
slow: one step is ~230 Python calls doing microseconds of arithmetic.  This
module keeps the identical model and replaces the *execution*: the circuit is
flattened once into flat arrays, and numba-compiled functions evaluate the
sources, stamp the matrix, solve it and re-read the device regions.

Four properties are deliberate:

* **Same physics, same numbers.**  The companion models are unchanged; a test
  asserts the compiled path and the reference path agree to round-off.
* **Regions are cached, not looked up.**  Every piecewise-linear region's
  coefficients (conductance + offset, or the plane of a channel triangle) are
  computed once at plan time into a flat table, so a step indexes a float array
  instead of calling ``slope()``/``intercept()`` a dozen times.
* **Device currents are produced where the state already exists**, not replayed
  afterwards through the Python model.
* **The step is chosen by the error, not by the deck.**  Step doubling gives a
  local-truncation estimate, so the solver takes small steps only where the
  solution is actually moving.  ``max_step`` becomes an optional ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from engine.ir.circuit import Circuit, ElementKind
from engine.solver.channel import ChannelTable
from engine.solver.companions import LinearSystem
from engine.solver.device import ChargeCapDevice, DiodeDevice, SwitchDevice
from engine.solver.result import Event, SolverResult

# Waveform kinds (must match engine.ir.circuit.Waveform.kind).
_W_DC = 0
_W_PWL = 1
_W_PULSE = 2

# Device kinds.
_D_DIODE = 0
_D_SWITCH = 1
_D_CHARGECAP = 2
_D_CHANNEL = 3

#: Region fixed-point limit inside one step.
MAX_REGION_ITERATIONS = 12


# ---------------------------------------------------------------------------
# Compiled primitives
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def source_value(kind, off, cnt, wt, wv, t):
    if kind == _W_DC:
        return wv[off]
    if kind == _W_PWL:
        if t <= wt[off]:
            return wv[off]
        if t >= wt[off + cnt - 1]:
            return wv[off + cnt - 1]
        lo = 0
        hi = cnt - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if wt[off + mid] <= t:
                lo = mid
            else:
                hi = mid
        t0 = wt[off + lo]
        t1 = wt[off + hi]
        v0 = wv[off + lo]
        v1 = wv[off + hi]
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    # pulse: v1 v2 td tr tf pw per
    v1 = wv[off]
    v2 = wv[off + 1]
    td = wv[off + 2]
    tr = wv[off + 3]
    tf = wv[off + 4]
    pw = wv[off + 5]
    per = wv[off + 6]
    u = t - td
    if u < 0.0:
        return v1
    if per > 0.0:
        u = u % per
    if u < tr:
        return v1 if tr <= 0.0 else v1 + (v2 - v1) * u / tr
    u -= tr
    if u < pw:
        return v2
    u -= pw
    if u < tf:
        return v2 if tf <= 0.0 else v2 + (v1 - v2) * u / tf
    return v1


@nb.njit(cache=True)
def _segment(xs, off, cnt, value):
    """Index of the segment containing ``value``, clamped to the table ends."""
    if value <= xs[off]:
        return 0
    if value >= xs[off + cnt - 1]:
        return cnt - 2
    lo = 0
    hi = cnt - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[off + mid] <= value:
            lo = mid
        else:
            hi = mid
    if lo > cnt - 2:
        return cnt - 2
    return lo


@nb.njit(cache=True)
def _solve_dense(a, b, x, work):
    """Dense LU with partial pivoting; False when the matrix is singular."""
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            work[i, j] = a[i, j]
        work[i, n] = b[i]
    for c in range(n):
        pivot = c
        best = abs(work[c, c])
        for r in range(c + 1, n):
            candidate = abs(work[r, c])
            if candidate > best:
                best = candidate
                pivot = r
        if best == 0.0:
            return False
        if pivot != c:
            for j in range(n + 1):
                work[c, j], work[pivot, j] = work[pivot, j], work[c, j]
        piv = work[c, c]
        for r in range(c + 1, n):
            factor = work[r, c] / piv
            if factor != 0.0:
                for j in range(c, n + 1):
                    work[r, j] -= factor * work[c, j]
    for i in range(n - 1, -1, -1):
        acc = work[i, n]
        for j in range(i + 1, n):
            acc -= work[i, j] * x[j]
        x[i] = acc / work[i, i]
    return True


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def assemble_solve(P, S, t, h, method, G, rhs, x, work, trial):
    """Stamp with the trial regions and solve.  Mutates only the scratch arrays."""
    (r_a, r_b, r_g,
     c_a, c_b, c_idx, c_val,
     l_a, l_b, l_k, l_idx, l_val,
     v_a, v_b, v_k, v_kind, v_off, v_cnt,
     i_a, i_b, i_kind, i_off, i_cnt,
     d_kind, d_a, d_b, d_c, d_c2, d_off,
     d_ax, d_axc, d_ax2, d_ax2c, d_vt, d_vh, d_tab, ax, wt, wv) = P
    vc = S[0]
    ic = S[1]
    il = S[2]
    vl = S[3]
    q_prev = S[6]

    n = G.shape[0]
    for i in range(n):
        rhs[i] = 0.0
        for j in range(n):
            G[i, j] = 0.0

    for e in range(r_g.size):
        g = r_g[e]
        a = r_a[e]
        b = r_b[e]
        if a >= 0:
            G[a, a] += g
            if b >= 0:
                G[a, b] -= g
        if b >= 0:
            G[b, b] += g
            if a >= 0:
                G[b, a] -= g

    for e in range(c_val.size):
        c = c_val[e]
        k = c_idx[e]
        if method == 1:
            g_eq = 2.0 * c / h
            i_eq = g_eq * vc[k] + ic[k]
        else:
            g_eq = c / h
            i_eq = g_eq * vc[k]
        a = c_a[e]
        b = c_b[e]
        if a >= 0:
            G[a, a] += g_eq
            rhs[a] += i_eq
            if b >= 0:
                G[a, b] -= g_eq
        if b >= 0:
            G[b, b] += g_eq
            rhs[b] -= i_eq
            if a >= 0:
                G[b, a] -= g_eq

    for e in range(l_val.size):
        k = l_k[e]
        idx = l_idx[e]
        r_eq = 2.0 * l_val[e] / h if method == 1 else l_val[e] / h
        v_eq = -r_eq * il[idx] - (vl[idx] if method == 1 else 0.0)
        a = l_a[e]
        b = l_b[e]
        if a >= 0:
            G[a, k] += 1.0
            G[k, a] += 1.0
        if b >= 0:
            G[b, k] -= 1.0
            G[k, b] -= 1.0
        G[k, k] -= r_eq
        rhs[k] = v_eq

    for e in range(v_k.size):
        k = v_k[e]
        value = source_value(v_kind[e], v_off[e], v_cnt[e], wt, wv, t)
        a = v_a[e]
        b = v_b[e]
        if a >= 0:
            G[a, k] += 1.0
            G[k, a] += 1.0
        if b >= 0:
            G[b, k] -= 1.0
            G[k, b] -= 1.0
        rhs[k] = value
    for e in range(i_kind.size):
        value = source_value(i_kind[e], i_off[e], i_cnt[e], wt, wv, t)
        a = i_a[e]
        b = i_b[e]
        if a >= 0:
            rhs[a] -= value
        if b >= 0:
            rhs[b] += value

    for e in range(d_kind.size):
        kind = d_kind[e]
        region = trial[e]
        off = d_off[e]
        a = d_a[e]
        b = d_b[e]
        if kind == _D_DIODE:
            g = d_tab[off + 2 * region]
            cur = d_tab[off + 2 * region + 1]
            if a >= 0:
                G[a, a] += g
                rhs[a] -= cur
                if b >= 0:
                    G[a, b] -= g
            if b >= 0:
                G[b, b] += g
                rhs[b] += cur
                if a >= 0:
                    G[b, a] -= g
        elif kind == _D_SWITCH:
            g = d_tab[off + region]
            if a >= 0:
                G[a, a] += g
                if b >= 0:
                    G[a, b] -= g
            if b >= 0:
                G[b, b] += g
                if a >= 0:
                    G[b, a] -= g
        elif kind == _D_CHARGECAP:
            c = d_tab[off + 2 * region]
            q0 = d_tab[off + 2 * region + 1]
            g = c / h
            cur = (q0 - q_prev[e]) / h
            if a >= 0:
                G[a, a] += g
                rhs[a] -= cur
                if b >= 0:
                    G[a, b] -= g
            if b >= 0:
                G[b, b] += g
                rhs[b] += cur
                if a >= 0:
                    G[b, a] -= g
        else:
            a0 = d_tab[off + 3 * region]
            a1 = d_tab[off + 3 * region + 1]
            a2 = d_tab[off + 3 * region + 2]
            gate = d_c[e]
            for row in (a, b):
                if row >= 0:
                    sign = 1.0 if row == a else -1.0
                    if gate >= 0:
                        G[row, gate] += sign * a1
                    if a >= 0:
                        G[row, a] += sign * a2
                    if b >= 0:
                        G[row, b] -= sign * (a1 + a2)
                    rhs[row] -= sign * a0

    return _solve_dense(G, rhs, x, work)


@nb.njit(cache=True)
def read_regions(P, S, x, trial):
    """Re-read every device region from the solved state.  True when it moved."""
    (r_a, r_b, r_g,
     c_a, c_b, c_idx, c_val,
     l_a, l_b, l_k, l_idx, l_val,
     v_a, v_b, v_k, v_kind, v_off, v_cnt,
     i_a, i_b, i_kind, i_off, i_cnt,
     d_kind, d_a, d_b, d_c, d_c2, d_off,
     d_ax, d_axc, d_ax2, d_ax2c, d_vt, d_vh, d_tab, ax, wt, wv) = P
    changed = False
    for e in range(d_kind.size):
        kind = d_kind[e]
        a = d_a[e]
        b = d_b[e]
        vb = x[b] if b >= 0 else 0.0
        va = x[a] if a >= 0 else 0.0
        off = d_off[e]
        if kind == _D_DIODE or kind == _D_CHARGECAP:
            new = _segment(ax, d_ax[e], d_axc[e], va - vb)
            if new != trial[e]:
                trial[e] = new
                changed = True
        elif kind == _D_SWITCH:
            control = (x[d_c[e]] if d_c[e] >= 0 else 0.0) \
                - (x[d_c2[e]] if d_c2[e] >= 0 else 0.0)
            on = trial[e]
            if on == 1:
                if control < d_vt[e] - 0.5 * d_vh[e]:
                    trial[e] = 0
                    changed = True
            else:
                if control > d_vt[e] + 0.5 * d_vh[e]:
                    trial[e] = 1
                    changed = True
        else:
            vgs = (x[d_c[e]] if d_c[e] >= 0 else 0.0) - vb
            vds = va - vb
            i = _segment(ax, d_ax[e], d_axc[e], vgs)
            j = _segment(ax, d_ax2[e], d_ax2c[e], vds)
            g0 = ax[d_ax[e] + i]
            g1 = ax[d_ax[e] + i + 1]
            d0 = ax[d_ax2[e] + j]
            d1 = ax[d_ax2[e] + j + 1]
            u = 0.0 if vgs <= g0 else (1.0 if vgs >= g1 else (vgs - g0) / (g1 - g0))
            w = 0.0 if vds <= d0 else (1.0 if vds >= d1 else (vds - d0) / (d1 - d0))
            new = (i * (d_ax2c[e] - 1) + j) * 2 + (1 if w > u else 0)
            if new != trial[e]:
                trial[e] = new
                changed = True
    return changed


@nb.njit(cache=True)
def commit(P, S, x, h, method, trial):
    """Accept a step: roll history forward and adopt the settled regions."""
    (r_a, r_b, r_g,
     c_a, c_b, c_idx, c_val,
     l_a, l_b, l_k, l_idx, l_val,
     v_a, v_b, v_k, v_kind, v_off, v_cnt,
     i_a, i_b, i_kind, i_off, i_cnt,
     d_kind, d_a, d_b, d_c, d_c2, d_off,
     d_ax, d_axc, d_ax2, d_ax2c, d_vt, d_vh, d_tab, ax, wt, wv) = P
    vc = S[0]
    ic = S[1]
    il = S[2]
    vl = S[3]
    regions = S[4]
    q_prev = S[6]
    for e in range(c_val.size):
        c = c_val[e]
        k = c_idx[e]
        a = c_a[e]
        b = c_b[e]
        v_now = (x[a] if a >= 0 else 0.0) - (x[b] if b >= 0 else 0.0)
        if method == 1:
            g_eq = 2.0 * c / h
            i_eq = g_eq * vc[k] + ic[k]
        else:
            g_eq = c / h
            i_eq = g_eq * vc[k]
        ic[k] = g_eq * v_now - i_eq
        vc[k] = v_now
    for e in range(l_val.size):
        k = l_k[e]
        idx = l_idx[e]
        il[idx] = x[k]
        a = l_a[e]
        b = l_b[e]
        vl[idx] = (x[a] if a >= 0 else 0.0) - (x[b] if b >= 0 else 0.0)
    for e in range(d_kind.size):
        if d_kind[e] == _D_CHARGECAP:
            region = trial[e]
            off = d_off[e]
            c = d_tab[off + 2 * region]
            q0 = d_tab[off + 2 * region + 1]
            a = d_a[e]
            b = d_b[e]
            v = (x[a] if a >= 0 else 0.0) - (x[b] if b >= 0 else 0.0)
            q_prev[e] = c * v + q0
        regions[e] = trial[e]


@nb.njit(cache=True)
def device_currents(P, S, x, h, trial, out):
    """Terminal current of every device, from the state that produced it."""
    (r_a, r_b, r_g,
     c_a, c_b, c_idx, c_val,
     l_a, l_b, l_k, l_idx, l_val,
     v_a, v_b, v_k, v_kind, v_off, v_cnt,
     i_a, i_b, i_kind, i_off, i_cnt,
     d_kind, d_a, d_b, d_c, d_c2, d_off,
     d_ax, d_axc, d_ax2, d_ax2c, d_vt, d_vh, d_tab, ax, wt, wv) = P
    q_prev = S[6]
    for e in range(d_kind.size):
        kind = d_kind[e]
        a = d_a[e]
        b = d_b[e]
        va = x[a] if a >= 0 else 0.0
        vb = x[b] if b >= 0 else 0.0
        off = d_off[e]
        region = trial[e]
        if kind == _D_DIODE:
            out[e] = d_tab[off + 2 * region] * (va - vb) + d_tab[off + 2 * region + 1]
        elif kind == _D_SWITCH:
            out[e] = d_tab[off + region] * (va - vb)
        elif kind == _D_CHARGECAP:
            c = d_tab[off + 2 * region]
            q0 = d_tab[off + 2 * region + 1]
            out[e] = (c * (va - vb) + q0 - q_prev[e]) / h
        else:
            vgs = (x[d_c[e]] if d_c[e] >= 0 else 0.0) - vb
            out[e] = (d_tab[off + 3 * region] + d_tab[off + 3 * region + 1] * vgs
                      + d_tab[off + 3 * region + 2] * (va - vb))


@nb.njit(cache=True)
def switch_control(P, x, index):
    """Control voltage of one switch, for the event finder."""
    d_c = P[26]
    d_c2 = P[27]
    return (x[d_c[index]] if d_c[index] >= 0 else 0.0) \
        - (x[d_c2[index]] if d_c2[index] >= 0 else 0.0)


@nb.njit(cache=True)
def switch_threshold(P, index, on):
    d_vt = P[33]
    d_vh = P[34]
    if on == 1:
        return d_vt[index] - 0.5 * d_vh[index]
    return d_vt[index] + 0.5 * d_vh[index]


# ---------------------------------------------------------------------------
# The plan: flat arrays built once per circuit
# ---------------------------------------------------------------------------

def _waveform_entry(waveform) -> tuple[int, np.ndarray, np.ndarray]:
    kind = waveform.kind
    if kind == "dc":
        return _W_DC, np.zeros(1), np.array([waveform.dc])
    if kind == "pwl":
        times = np.array([p[0] for p in waveform.points], dtype=float)
        values = np.array([p[1] for p in waveform.points], dtype=float)
        return _W_PWL, times, values
    if kind == "pulse":
        assert waveform.pulse is not None
        return _W_PULSE, np.zeros(1), np.array(waveform.pulse, dtype=float)
    raise ValueError(f"unknown waveform kind {kind!r}")


@dataclass
class Plan:
    """A circuit flattened into the arrays the compiled kernel reads."""

    P: tuple
    S: tuple
    G: np.ndarray
    rhs: np.ndarray
    x: np.ndarray
    work: np.ndarray
    dev_i: np.ndarray
    node_names: tuple[str, ...]
    branch_stamps: tuple
    device_names: tuple[str, ...]
    switch_index: tuple[int, ...]
    size: int

    # -- construction ----------------------------------------------------

    @classmethod
    def build(cls, system: LinearSystem) -> Plan:
        node = system.structure.node_index

        def idx(name: str) -> int:
            value = node.index(name)
            return -1 if value is None else value

        lists: dict[str, list] = {k: [] for k in (
            "r_a", "r_b", "r_g", "c_a", "c_b", "c_idx", "c_val",
            "l_a", "l_b", "l_k", "l_idx", "l_val",
            "v_a", "v_b", "v_k", "v_kind", "v_off", "v_cnt",
            "i_a", "i_b", "i_kind", "i_off", "i_cnt")}
        wt_pool: list[float] = []
        wv_pool: list[float] = []

        def add_waveform(waveform) -> tuple[int, int, int]:
            kind, times, values = _waveform_entry(waveform)
            offset = len(wv_pool)
            wv_pool.extend(values.tolist())
            if kind == _W_PWL:
                wt_pool.extend(times.tolist())
            else:
                wt_pool.extend([0.0] * len(values))
            return kind, offset, len(values)

        for stamp in system.structure.stamps:
            element = stamp.element
            nodes = stamp.nodes
            a = -1 if nodes[0] is None else nodes[0]
            b = -1 if nodes[1] is None else nodes[1]
            if element.kind is ElementKind.RESISTOR:
                assert element.value is not None
                lists["r_a"].append(a)
                lists["r_b"].append(b)
                lists["r_g"].append(1.0 / element.value)
            elif element.kind is ElementKind.CAPACITOR:
                assert element.value is not None
                lists["c_a"].append(a)
                lists["c_b"].append(b)
                lists["c_idx"].append(len(lists["c_val"]))
                lists["c_val"].append(element.value)
            elif element.kind is ElementKind.INDUCTOR:
                assert element.value is not None and stamp.branch is not None
                lists["l_a"].append(a)
                lists["l_b"].append(b)
                lists["l_k"].append(stamp.branch)
                lists["l_idx"].append(len(lists["l_val"]))
                lists["l_val"].append(element.value)
            elif element.kind is ElementKind.VSOURCE:
                assert stamp.branch is not None and element.waveform is not None
                kind, offset, count = add_waveform(element.waveform)
                lists["v_a"].append(a)
                lists["v_b"].append(b)
                lists["v_k"].append(stamp.branch)
                lists["v_kind"].append(kind)
                lists["v_off"].append(offset)
                lists["v_cnt"].append(count)
            elif element.kind is ElementKind.ISOURCE:
                assert element.waveform is not None
                kind, offset, count = add_waveform(element.waveform)
                lists["i_a"].append(a)
                lists["i_b"].append(b)
                lists["i_kind"].append(kind)
                lists["i_off"].append(offset)
                lists["i_cnt"].append(count)

        d_kind, d_a, d_b, d_c, d_c2, d_off = [], [], [], [], [], []
        d_ax, d_axc, d_ax2, d_ax2c, d_vt, d_vh = [], [], [], [], [], []
        tab: list[float] = []
        ax: list[float] = []
        regions0: list[int] = []
        switch_index: list[int] = []
        for stamp in system.devices:
            device = system.device(stamp.name)
            a = -1 if stamp.nodes[0] is None else stamp.nodes[0]
            b = -1 if stamp.nodes[1] is None else stamp.nodes[1]
            d_a.append(a)
            d_b.append(b)
            d_off.append(len(tab))
            d_ax.append(0)
            d_axc.append(0)
            d_ax2.append(0)
            d_ax2c.append(0)
            d_vt.append(0.0)
            d_vh.append(0.0)
            d_c.append(-1)
            d_c2.append(-1)
            if isinstance(device, DiodeDevice):
                d_kind.append(_D_DIODE)
                table = device.table
                _push_axis(table, ax, d_ax, d_axc, len(tab))
                for region in range(len(table)):
                    tab.extend([table.slope(region), table.intercept(region)])
                regions0.append(table.region(0.0))
            elif isinstance(device, SwitchDevice):
                d_kind.append(_D_SWITCH)
                tab.extend([1.0 / device.roff, 1.0 / device.ron])
                d_vt[-1] = device.vt
                d_vh[-1] = device.vh
                d_c[-1] = -1 if stamp.nodes[2] is None else stamp.nodes[2]
                d_c2[-1] = -1 if stamp.nodes[3] is None else stamp.nodes[3]
                switch_index.append(len(d_kind) - 1)
                regions0.append(0)
            elif isinstance(device, ChargeCapDevice):
                d_kind.append(_D_CHARGECAP)
                table = device.charge.table
                _push_axis(table, ax, d_ax, d_axc, len(tab))
                for region in range(len(table)):
                    tab.extend([table.slope(region), table.intercept(region)])
                regions0.append(table.region(0.0))
            else:
                d_kind.append(_D_CHANNEL)
                table: ChannelTable = device.table
                off = len(ax)
                ax.extend(table.vgs)
                d_ax[-1] = off
                d_axc[-1] = len(table.vgs)
                off2 = len(ax)
                ax.extend(table.vds)
                d_ax2[-1] = off2
                d_ax2c[-1] = len(table.vds)
                for region in range(2 * table.n_cells):
                    tab.extend(table.plane(region))
                d_c[-1] = -1 if stamp.nodes[2] is None else stamp.nodes[2]
                d_c2[-1] = b
                regions0.append(table.region(0.0, 0.0))

        size = system.size
        P = (
            _i(lists["r_a"]), _i(lists["r_b"]), _f(lists["r_g"]),
            _i(lists["c_a"]), _i(lists["c_b"]), _i(lists["c_idx"]),
            _f(lists["c_val"]),
            _i(lists["l_a"]), _i(lists["l_b"]), _i(lists["l_k"]),
            _i(lists["l_idx"]), _f(lists["l_val"]),
            _i(lists["v_a"]), _i(lists["v_b"]), _i(lists["v_k"]),
            _i(lists["v_kind"]), _i(lists["v_off"]), _i(lists["v_cnt"]),
            _i(lists["i_a"]), _i(lists["i_b"]), _i(lists["i_kind"]),
            _i(lists["i_off"]), _i(lists["i_cnt"]),
            _i(d_kind), _i(d_a), _i(d_b), _i(d_c), _i(d_c2), _i(d_off),
            _i(d_ax), _i(d_axc), _i(d_ax2), _i(d_ax2c), _f(d_vt), _f(d_vh),
            _f(tab), _f(ax), _f(wt_pool), _f(wv_pool),
        )
        S = (
            np.zeros(len(lists["c_val"])),          # vc
            np.zeros(len(lists["c_val"])),          # ic
            np.zeros(len(lists["l_val"])),          # il
            np.zeros(len(lists["l_val"])),          # vl
            np.array(regions0, dtype=np.int64),     # regions
            np.array(regions0, dtype=np.int64),     # switches on/off
            np.zeros(len(d_kind)),                  # q_prev
        )
        branch_stamps = tuple(
            s for s in system.structure.stamps if s.branch is not None)
        return cls(
            P=P, S=S,
            G=np.zeros((size, size)),
            rhs=np.zeros(size),
            x=np.zeros(size),
            work=np.zeros((size, size + 1)),
            dev_i=np.zeros(len(d_kind)),
            node_names=system.structure.node_index.names,
            branch_stamps=branch_stamps,
            device_names=tuple(d.name for d in system.devices),
            switch_index=tuple(switch_index),
            size=size,
        )

    # -- driving ---------------------------------------------------------

    def snapshot(self) -> list[np.ndarray]:
        return [a.copy() for a in self.S]

    def restore(self, snapshot: list[np.ndarray]) -> None:
        for target, saved in zip(self.S, snapshot):
            target[:] = saved

    def advance(self, t: float, h: float, method: int) -> tuple[bool, np.ndarray]:
        """Solve one step, iterating the region set.  Does not mutate history."""
        trial = self.S[4].copy()
        for _ in range(MAX_REGION_ITERATIONS):
            if not assemble_solve(self.P, self.S, t, h, method, self.G,
                                  self.rhs, self.x, self.work, trial):
                return False, trial
            if not read_regions(self.P, self.S, self.x, trial):
                return True, trial
        return True, trial

    def commit(self, h: float, method: int, trial: np.ndarray) -> None:
        commit(self.P, self.S, self.x, h, method, trial)

    def currents(self, h: float, trial: np.ndarray) -> np.ndarray:
        device_currents(self.P, self.S, self.x, h, trial, self.dev_i)
        return self.dev_i.copy()

    def control(self, index: int) -> float:
        return switch_control(self.P, self.x, index)

    def threshold(self, index: int, on: int) -> float:
        return switch_threshold(self.P, index, on)


def _i(values: list) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def _f(values: list) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _push_axis(table, ax: list, offsets: list, counts: list,
               _tab_len: int) -> None:
    offsets[-1] = len(ax)
    ax.extend(table.x)
    counts[-1] = len(table.x)


# ---------------------------------------------------------------------------
# The compiled transient loop
# ---------------------------------------------------------------------------

def _initial_state(plan: Plan, x: np.ndarray) -> None:
    """Seed the plan's history from a DC operating point."""
    P = plan.P
    c_a, c_b = P[3], P[4]
    l_a, l_b, l_k = P[7], P[8], P[9]
    d_kind, d_a, d_b, d_off, d_tab = P[23], P[24], P[25], P[28], P[35]

    def node(index, arr):
        return np.where(index >= 0, x[np.maximum(index, 0)], 0.0)

    vc = node(c_a, x) - node(c_b, x)
    plan.S[0][:] = vc
    plan.S[1][:] = 0.0
    plan.S[2][:] = x[l_k] if l_k.size else 0.0
    plan.S[3][:] = (node(l_a, x) - node(l_b, x)) if l_k.size else 0.0

    trial = plan.S[4].copy()
    read_regions(plan.P, plan.S, x, trial)
    plan.S[4][:] = trial
    plan.S[5][:] = trial
    for e in range(d_kind.size):
        if d_kind[e] == _D_CHARGECAP:
            region = trial[e]
            off = d_off[e]
            c = d_tab[off + 2 * region]
            q0 = d_tab[off + 2 * region + 1]
            v = (x[d_a[e]] if d_a[e] >= 0 else 0.0) -                 (x[d_b[e]] if d_b[e] >= 0 else 0.0)
            plan.S[6][e] = c * v + q0
