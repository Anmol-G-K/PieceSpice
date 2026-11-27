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

    truncated = [w for w in result.warnings if "sample buffer full" in w]
    if truncated:
        raise RuntimeError("run truncated: " + truncated[0])
    y = {k: np.asarray(v, dtype=float) for k, v in result.traces.items()}
    t = y["time"]
    window = t >= int(cond["simulation"]["settling_cycles"]) * period
    tw = t[window]
    if tw.size < 10:
        raise RuntimeError("measurement window not reached: run ended at "
                           f"{t[-1]:.6g} s")

    def wmean(series: np.ndarray) -> float:
        s = series[window]
        return float(np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(tw))
                     / (tw[-1] - tw[0]))

    vout = wmean(y["V(vout)"])
    pin = wmean(-y["V(bus)"] * y["I(Vbus)"])
    pout = vout * vout / float(cond["r_load_ohm"])
    return {"vendor": vendor, "vout": vout, "il": wmean(np.abs(y["I(Lout)"])),
            "pin": pin, "eff": 100.0 * pout / pin, "steps": result.steps}


def dpt(basis: str = "t4ann") -> list[dict]:
    """The gate's own verdict, so this script cannot disagree with G7."""
    from pipeline.solver_gate import summarise

    rows = []
    for vendor, gate in sorted(summarise(basis).items()):
        detail = gate["detail"]
        rows.append({"vendor": vendor,
                     "vds": detail["engine_vds_pct"],
                     "id": detail["engine_id_pct"],
                     "speed": detail.get("solver_speedup"),
                     "bar": detail["equivalence_bar_pct"],
                     "relaxed": detail.get("relaxation_applied"),
                     "pass": gate["pass"]})
    return rows


VENDORS = ("epc2216", "csd16412q5a", "cpm2-1200", "irfs7730pbf")


def buck_all() -> list[dict]:
    """The buck for every vendor.  A failure is recorded as N/A with its
    reason, never imputed and never skipped silently."""
    rows = []
    for vendor in VENDORS:
        try:
            rows.append(buck(vendor))
        except Exception as exc:                      # noqa: BLE001
            rows.append({"vendor": vendor,
                         "error": f"{type(exc).__name__}: {exc}"[:240]})
    return rows


if __name__ == "__main__":
    print("BUCK  every vendor, compiled engine, adaptive at the deck ceiling")
    for b in buck_all():
        if "error" in b:
            print(f"  {b['vendor']:13s} N/A - {b['error']}")
        else:
            print(f"  {b['vendor']:13s} Vout {b['vout']:7.4f} V  "
                  f"I_L {b['il']:7.4f} A  efficiency {b['eff']:6.2f} %  "
                  f"({b['steps']} steps)")
    print("  recorded EPC2216: golden 5.3077 V / 97.55 %   Tier-2 5.3053 V / 96.91 %")
    print()
    print("DPT   G7, both arms resolved, against the policy bar:")
    for r in dpt():
        tag = (f"relaxed {r['relaxed']['equivalence_rel_pct_max']:.1f} %"
               if r["relaxed"] else f"bar {r['bar']:.1f} %")
        speed = f"{r['speed']:.2f}x" if r["speed"] else "n/a"
        print(f"  {r['vendor']:13s} Vds {r['vds']:5.2f} %  Id {r['id']:5.2f} %  "
              f"speed {speed:>8s}  {tag:>12s}  "
              f"{'PASS' if r['pass'] else 'FAIL'}")
