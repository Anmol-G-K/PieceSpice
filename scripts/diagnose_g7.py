"""Where does the G7 engine error live?  Windowed B-vs-C comparison."""
from __future__ import annotations
import sys
import numpy as np
from engine.harness.comparison import (_conditions, _load_config, _ltspice_arm,
                                       _rom_subcircuit, _solver_arm, ROOT, _on)
from engine.rom import solver_rom


def run(vendor: str, basis: str = "t4ann") -> None:
    cfg = _load_config(vendor)
    cond = _conditions(cfg)
    lib = ROOT / "reports" / "models" / f"{vendor}_PIECESPICE_T4_ann.lib"
    name, pins = _rom_subcircuit(lib)
    b = _ltspice_arm("rom_ltspice", lib, name, pins, cond, 1)
    c = _solver_arm(solver_rom(vendor, basis), cond, 1)

    lo, hi = float(np.min(b.v_gs)), float(np.max(b.v_gs))
    thr = lo + 0.5 * (hi - lo)
    rising = np.flatnonzero((b.v_gs[:-1] < thr) & (b.v_gs[1:] >= thr))
    falling = np.flatnonzero((b.v_gs[:-1] >= thr) & (b.v_gs[1:] < thr))
    t_on = float(b.time[rising[0]]) if rising.size else float(b.time[-1] * 0.4)
    t_off = float(b.time[falling[-1]]) if falling.size else float(b.time[-1] * 0.6)
    print(f"\n=== {vendor}: on {t_on*1e6:.3f} us, off {t_off*1e6:.3f} us")

    for field, label, unit in (("v_ds", "Vds", "V"), ("i_d", "Id", "A")):
        rb, rc = _on(b, c, field)
        grid = b.time[(b.time >= max(b.time[0], c.time[0]))
                      & (b.time <= min(b.time[-1], c.time[-1]))]
        err = rc - rb
        span = float(np.ptp(rb)) or 1.0
        print(f"  {label}: span {span:.3f} {unit}  overall "
              f"{np.sqrt(np.mean(err**2))/span*100:5.2f} %   "
              f"bias(mean) {err.mean():+.4f} {unit}")
        for name, m in (
            ("pre-edge ", grid < t_on - 20e-9),
            ("turn-on  ", (grid >= t_on - 20e-9) & (grid <= t_on + 60e-9)),
            ("on-state ", (grid > t_on + 60e-9) & (grid < t_off - 20e-9)),
            ("turn-off ", (grid >= t_off - 20e-9) & (grid <= t_off + 60e-9)),
            ("post     ", grid > t_off + 60e-9),
        ):
            if m.sum() < 3:
                continue
            print(f"    {name} n={m.sum():5d} "
                  f"rms {np.sqrt(np.mean(err[m]**2))/span*100:5.2f} %  "
                  f"bias {err[m].mean():+8.4f}  "
                  f"SSE share {np.sum(err[m]**2)/np.sum(err**2)*100:5.1f} %  "
                  f"B_mean {rb[m].mean():8.3f}  C_mean {rc[m].mean():8.3f}")

    on = (b.time > t_on + 60e-9) & (b.time < t_off - 20e-9)
    if on.sum() > 5:
        ron_b = float(np.mean(b.v_ds[on] / np.maximum(b.i_d[on], 1e-9)))
        con = (c.time > t_on + 60e-9) & (c.time < t_off - 20e-9)
        ron_c = float(np.mean(c.v_ds[con] / np.maximum(c.i_d[con], 1e-9)))
        print(f"  on-state Ron: B {ron_b*1e3:.3f} mOhm   C {ron_c*1e3:.3f} mOhm"
              f"   ({(ron_c/ron_b-1)*100:+.2f} %)")


if __name__ == "__main__":
    for v in sys.argv[1:] or ["csd16412q5a", "irfs7730pbf"]:
        run(v)
