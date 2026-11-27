"""Both benches, verified end to end, in one command.

  uv run python scripts/verify_benches.py

Buck: builds the converter netlist for the configured model, integrates it with
the compiled solver at the deck's own maximum step (a ceiling, SPICE's meaning),
and reports Vout, I_L and efficiency measured as *time-weighted* averages over
the bench's measurement window -- a sample average over-weights the sub-
picosecond samples the step control takes through a switch closure and reports a
nonsense efficiency.

DPT: reports the G7 engine error for the four vendors against the declared bar.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from engine.ir.parse import read_deck
from engine.solver.transient import solve_transient
from simulation.testbenches.buck_converter import (build_converter_netlist,
                                                   load_converter_config,
                                                   resolve_converter_model)

ROOT = Path(__file__).resolve().parent.parent


def buck(vendor: str = "epc2216", tier: str = "tier4") -> dict:
    cfg = load_converter_config()
    model = resolve_converter_model(cfg, vendor, tier)
    text, _ = build_converter_netlist(model, config=cfg)
    path = ROOT / "reports" / "models" / f"_verify_buck_{vendor}.cir"
    path.write_text(text, encoding="ascii")
    circ = read_deck(path).circuit
    cond = cfg["conditions"]
    period = 1.0 / float(cond["f_sw_hz"])
    t_end = int(cond["simulation"]["cycles"]) * period
    max_step = float(cond["simulation"]["max_step_fraction_of_period"]) * period
    result = solve_transient(circ, engine="compiled", end_time=t_end,
                             max_step=max_step, adaptive=True)
    path.unlink(missing_ok=True)

    y = {k: np.asarray(v, dtype=float) for k, v in result.traces.items()}
    t = y["time"]
    window = t >= int(cond["simulation"]["settling_cycles"]) * period
    tw = t[window]

    def wmean(series: np.ndarray) -> float:
        s = series[window]
        return float(np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(tw))
                     / (tw[-1] - tw[0]))

    vout = wmean(y["V(vout)"])
    pin = wmean(-y["V(bus)"] * y["I(Vbus)"])
    pout = vout * vout / float(cond["r_load_ohm"])
    return {"vendor": vendor, "vout": vout, "il": wmean(np.abs(y["I(Lout)"])),
            "pin": pin, "eff": 100.0 * pout / pin, "steps": result.steps}


def dpt(bar: float = 2.0, basis: str = "t4ann") -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "reports" / "comparison" / "*" /
                                  f"solver_validation_{basis}.json"))):
        d = json.loads(Path(f).read_text())
        e = d["comparison"]["solver_error_C_minus_B_capped"]
        vds, idd = e["vds"]["rel_rmse_pct"], e["id"]["rel_rmse_pct"]
        rows.append({"vendor": d["vendor"], "vds": vds, "id": idd,
                     "pass": max(vds, idd) <= bar})
    return rows


if __name__ == "__main__":
    b = buck()
    print(f"BUCK  {b['vendor']}: Vout {b['vout']:.4f} V  I_L {b['il']:.4f} A  "
          f"efficiency {b['eff']:.2f} %  ({b['steps']} steps)")
    print("  recorded: golden 5.3077 V / 97.55 %   Tier-2 5.3053 V / 96.91 %")
    print()
    print("DPT   G7 engine error against the same ROM in LTspice (bar 2 %):")
    for r in dpt():
        print(f"  {r['vendor']:13s} Vds {r['vds']:5.2f} %  Id {r['id']:5.2f} %  "
              f"{'PASS' if r['pass'] else 'FAIL'}")
