"""Is the edge error a time shift or a shape difference?

Time-delay estimation borrows the classic signal-processing estimator (Knapp &
Carter): find the lag that minimises the least-squares misfit between two
waveforms, then report how much of the error that lag explains.  A residual that
collapses under one shift is a *timing* finding -- event localisation.  One that
survives is a *shape* finding -- the element model.  Level-crossing times at
10/50/90 % are the switching-time view a device engineer would take
(IEC 60747 style), and they say the same thing in physical units.
"""
from __future__ import annotations

import sys
import numpy as np

from engine.harness.comparison import (_conditions, _load_config, _ltspice_arm,
                                       _rom_subcircuit, _solver_arm, ROOT)
from engine.rom import solver_rom

DT = 10e-12          # 10 ps sampling for the shift search
MAX_SHIFT = 3e-9     # +/- 3 ns


def crossings(t: np.ndarray, y: np.ndarray, fracs=(0.1, 0.5, 0.9)):
    lo, hi = float(np.min(y)), float(np.max(y))
    span = hi - lo
    out = {}
    for f in fracs:
        level = lo + f * span
        sign = np.sign(y - level)
        idx = np.flatnonzero(np.diff(sign) != 0)
        times = []
        for i in idx:
            y0, y1 = y[i], y[i + 1]
            if y1 == y0:
                continue
            times.append(t[i] + (level - y0) * (t[i + 1] - t[i]) / (y1 - y0))
        out[f] = times
    return out


def best_shift(tb, yb, tc, yc, lo, hi):
    grid = np.arange(lo, hi, DT)
    b = np.interp(grid, tb, yb)
    c = np.interp(grid, tc, yc)
    n = grid.size
    span = float(np.ptp(b)) or 1.0
    raw = float(np.sqrt(np.mean((c - b) ** 2)))
    best_r, best_k = raw, 0
    kmax = int(MAX_SHIFT / DT)
    for k in range(-kmax, kmax + 1):
        if k >= 0:
            bb, cc = b[k:], c[:n - k]
        else:
            bb, cc = b[:n + k], c[-k:]
        if bb.size < 200:
            continue
        r = float(np.sqrt(np.mean((cc - bb) ** 2)))
        if r < best_r:
            best_r, best_k = r, k
    return raw / span * 100, best_r / span * 100, best_k * DT * 1e12


def run(vendor: str, basis: str = "t4ann") -> None:
    cfg = _load_config(vendor)
    cond = _conditions(cfg)
    lib = ROOT / "reports" / "models" / f"{vendor}_PIECESPICE_T4_ann.lib"
    name, pins = _rom_subcircuit(lib)
    b = _ltspice_arm("rom_ltspice", lib, name, pins, cond, 1)
    c = _solver_arm(solver_rom(vendor, basis), cond, 1)

    lo_g, hi_g = float(np.min(b.v_gs)), float(np.max(b.v_gs))
    thr = lo_g + 0.5 * (hi_g - lo_g)
    rising = np.flatnonzero((b.v_gs[:-1] < thr) & (b.v_gs[1:] >= thr))
    falling = np.flatnonzero((b.v_gs[:-1] >= thr) & (b.v_gs[1:] < thr))
    t_on = float(b.time[rising[0]])
    t_off = float(b.time[falling[-1]])
    t_end = float(b.time[-1])
    print(f"\n{'='*74}\n{vendor}   turn-on {t_on*1e6:.3f} us   turn-off "
          f"{t_off*1e6:.3f} us")

    for field, label, unit in (("v_gs", "Vgs", "V"), ("v_ds", "Vds", "V"),
                               ("i_d", "Id", "A")):
        tb, yb = b.time, getattr(b, field)
        tc, yc = c.time, getattr(c, field)
        for tag, centre in (("turn-on ", t_on), ("turn-off", t_off)):
            lo = max(centre - 100e-9, tb[0], tc[0])
            hi = min(centre + 100e-9, tb[-1], tc[-1], t_end)
            raw, aligned, shift = best_shift(tb, yb, tc, yc, lo, hi)
            cb = crossings(tb[(tb >= lo) & (tb <= hi)],
                           yb[(tb >= lo) & (tb <= hi)])
            cc = crossings(tc[(tc >= lo) & (tc <= hi)],
                           yc[(tc >= lo) & (tc <= hi)])
            deltas = []
            for f in (0.1, 0.5, 0.9):
                if cb[f] and cc[f]:
                    deltas.append((cc[f][0] - cb[f][0]) * 1e12)
            dtxt = "  ".join(f"{f*100:.0f}%{d:+7.0f}ps"
                             for f, d in zip((0.1, 0.5, 0.9), deltas))
            print(f"  {label:3s} {tag}  raw {raw:5.2f} %  after {shift:+6.0f} ps "
                  f"shift {aligned:5.2f} %  explained "
                  f"{(1 - (aligned/raw)**2)*100:5.1f} %   [{dtxt}]")


if __name__ == "__main__":
    for v in sys.argv[1:] or ["csd16412q5a", "irfs7730pbf"]:
        run(v)
