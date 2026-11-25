"""Where a solver step's time goes, and what is removable.

Run it before optimising anything:

    uv run python scripts/profile_solver_step.py epc2216

It answers three questions with numbers instead of intuition:

1. how long one step takes in the real loop;
2. how that time splits between matrix stamping, the linear solve, device
   region/table lookups and the surrounding loop;
3. how much of it is *removable* -- by timing the same numerical core as a
   compiled loop, which is what the interpreter overhead is paying for.
"""

from __future__ import annotations

import time

import numba as nb
import numpy as np

from engine.rom import SolverRom
from engine.rom.bench import DptConditions, dpt
from engine.solver.companions import LinearSystem, assemble
from engine.solver.transient import (
    _advance,
    _device_current_traces,
    _initial_state,
    solve_op,
    solve_transient,
)
from pipeline.fitting import build_pwl_spec


@nb.njit(cache=True)
def _compiled_core(rows, cols, values, count, rhs, size):
    """Stamps plus a dense solve, in one compiled pass (the numerical core)."""
    g = np.zeros((size, size))
    for k in range(count):
        g[rows[k], cols[k]] += values[k]
    m = np.empty((size, size + 1))
    for i in range(size):
        for j in range(size):
            m[i, j] = g[i, j]
        m[i, size] = rhs[i]
    for c in range(size):
        p = c
        best = abs(m[c, c])
        for r in range(c + 1, size):
            if abs(m[r, c]) > best:
                best = abs(m[r, c])
                p = r
        if p != c:
            for j in range(size + 1):
                m[c, j], m[p, j] = m[p, j], m[c, j]
        piv = m[c, c]
        for r in range(c + 1, size):
            f = m[r, c] / piv
            for j in range(c, size + 1):
                m[r, j] -= f * m[c, j]
    x = np.empty(size)
    for i in range(size - 1, -1, -1):
        s = m[i, size]
        for j in range(i + 1, size):
            s -= m[i, j] * x[j]
        x[i] = s / m[i, i]
    return x


def _time(fn, repeats: int) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats * 1e6


def main(vendor: str = "epc2216") -> int:
    rom = SolverRom.from_spec(build_pwl_spec(vendor))
    circuit = dpt(rom, DptConditions())
    solve_transient(circuit)
    started = time.perf_counter()
    result = solve_transient(circuit)
    wall = time.perf_counter() - started
    steps = result.steps
    print(f"{vendor}: {wall:.3f} s over {steps} steps = {wall / steps * 1e6:.1f} us/step")

    system = LinearSystem.compile(circuit)
    x = solve_op(circuit)
    regions = {d.name: d.region(x) for d in system.devices}
    state = _initial_state(system, x, regions)
    state.regions = dict(regions)
    G = np.zeros((system.size, system.size))
    rhs = np.zeros(system.size)

    parts = {
        "assemble": _time(lambda: assemble(system, state, t=1e-6, h=2e-9,
                                          method="trap", G=G, rhs=rhs), steps),
        "np.linalg.solve": _time(lambda: np.linalg.solve(G, rhs), steps),
        "_advance": _time(lambda: _advance(system, state, x, h=2e-9,
                                           method="trap"), steps),
        "region scan": _time(
            lambda: {d.name: d.region(x) for d in system.devices}, steps),
        "trace snapshot": _time(
            lambda: np.array(x, dtype=float, copy=True), steps),
    }
    for name, value in parts.items():
        print(f"  {name:16s} {value:7.2f} us/step")
    print(f"  {'sum':16s} {sum(parts.values()):7.2f} us/step")

    matrix = np.column_stack([result.get(k) for k in result.traces if k != "time"])
    replay = _time(lambda: _device_current_traces(system, matrix, result.get("time")), 1)
    print(f"  {'device-current replay':16s} {replay:7.1f} us total "
          f"({replay / steps:.1f} us/step, {replay / 1e6 / wall * 100:.0f}% of the run)")

    # Removable interpreter overhead: the same work compiled.
    n = system.size
    values = np.zeros(n * n)
    rows = np.zeros(n * n, dtype=np.int64)
    cols = np.zeros(n * n, dtype=np.int64)
    count = 0
    for i in range(n):
        for j in range(n):
            rows[count], cols[count] = i, j
            values[count] = G[i, j]
            count += 1
    b = rhs.copy()
    _compiled_core(rows, cols, values, count, b, n)
    t_np = _time(lambda: (np.add.at(G, (rows, cols), values),
                          np.linalg.solve(G, b)), 2000)
    t_nb = _time(lambda: _compiled_core(rows, cols, values, count, b, n), 2000)
    print(f"\nnumerical core ({n}x{n})  numpy+python {t_np:6.2f} us   "
          f"compiled {t_nb:6.2f} us   removable {(1 - t_nb / t_np) * 100:4.0f}%")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(*(sys.argv[1:2] or [])))
