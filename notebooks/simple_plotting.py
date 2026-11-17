"""Simple switching-plot runner (notebook 04a companion).

Reads the verified golden waveform CSV (data/m4_subset_<vendor>_golden.csv),
simulates the Tier-2 model live, and produces:
  - Images/<vendor>_switching_3cycle.png   (300 dpi, full-run overlay)
  - Images/<vendor>_switching_edge_zoom.png (300 dpi, rising+falling edges)
  - reporting/<vendor>_switching_metrics.json

Usage: python notebooks/simple_plotting.py [vendor]   (default: epc2216)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from common.plotting import setup_style
from metrics.static import compute_curve_error
from metrics.loss import compute_switching_energy
from metrics.switching import extract_switching_times
from metrics.fitness import infer_drive_conditions
from simulation.testbenches.simple_switching import (
    run_simple_switching)

IMG_DIR = REPO_ROOT / "Images"
REPORT_DIR = REPO_ROOT / "reports" / "switching"
DATA_DIR = REPO_ROOT / "data"
F_SW = 50_000.0


def _legend_east(ax):
    return ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))


def load_golden_csv(vendor: str) -> dict:
    path = DATA_DIR / f"{vendor}_golden.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Capture the golden waveform first: 6 "
            f"simulation for this vendor first.")
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return {"time": d[:, 0], "v_ds": d[:, 1], "i_d": d[:, 2],
            "v_gs": d[:, 3]}


def run_tier2(vendor: str, cfg: dict, vbus: float, vgs_on: float,
              i_load: float, temp: float) -> dict:
    t2_path = REPO_ROOT / "reports/models" / f"{vendor}_PIECESPICE_T2.lib"
    if not t2_path.exists():
        raise FileNotFoundError(
            f"Tier-2 model for '{vendor}' not found at {t2_path}. Run "
            f"notebooks/02_piecewise_approximation.ipynb first.")
    # Tier-2 subckt pin order is (d g s) - fixed by netlist_builder.
    run = run_simple_switching(
        t2_path, t2_path.stem, ["drain", "gate", "source"],
        vbus=vbus, vgs_on=vgs_on, i_load_target=i_load, temperature=temp)
    return {"time": run.time, "v_ds": run.v_ds, "i_d": run.i_d,
            "v_gs": run.v_gs}


def find_vds_edges(time: np.ndarray, v_ds: np.ndarray, v_bus: float):
    """Locate the LAST falling edge (turn-on) and rising edge (turn-off)
    of v_ds between the 90% and 10% bus levels. Returns (t_on_mid,
    t_off_mid) center times used to window the zooms."""
    hi, lo = 0.9 * v_bus, 0.1 * v_bus
    falling = np.where((v_ds[:-1] > lo) & (v_ds[:-1] > hi * 0.5) &
                       (v_ds[1:] <= lo))[0]
    rising = np.where((v_ds[:-1] < hi) & (v_ds[1:] >= hi))[0]
    # restrict to second half of the run (steady state)
    mid = len(v_ds) // 2
    falling = falling[falling >= mid]
    rising = rising[rising >= mid]
    t_fall = float(time[falling[-1]]) if len(falling) else None
    t_rise = float(time[rising[-1]]) if len(rising) else None
    return t_rise, t_fall


def metrics_of(run: dict, vbus: float, i_load: float) -> dict:
    sw = extract_switching_times(
        time=run["time"], v_ds=run["v_ds"], i_d=run["i_d"],
        v_gs=run["v_gs"], v_bus=run["v_ds"].max(), i_load=i_load)
    plateau = float(np.percentile(run["v_gs"], 95))
    rise = np.where((run["v_gs"][:-1] < .5 * plateau)
                    & (run["v_gs"][1:] >= .5 * plateau))[0]
    fall = np.where((run["v_gs"][:-1] > .5 * plateau)
                    & (run["v_gs"][1:] <= .5 * plateau))[0]

    def energy(idx):
        if len(idx) == 0:
            return 0.0
        i = int(idx[-1])
        lo, hi = max(0, i - 200), min(len(run["time"]) - 1, i + 200)
        return compute_switching_energy(run["time"], run["v_ds"],
                                        run["i_d"], run["time"][lo],
                                        run["time"][hi])

    on = run["v_gs"] > 0.9 * plateau
    return {
        "td_on_ns": sw.delay_on_ns, "tr_ns": sw.rise_time_ns,
        "td_off_ns": sw.delay_off_ns, "tf_ns": sw.fall_time_ns,
        "e_on_uj": energy(rise), "e_off_uj": energy(fall),
        "vds_on_mv": (float(np.median(run["v_ds"][on])) * 1e3
                      if on.any() else 0.0),
        "i_peak_a": float(np.max(run["i_d"])),
    }


def main(vendor: str) -> int:
    setup_style()
    IMG_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    cfg = yaml.safe_load(
        (REPO_ROOT / f"config/inputs/{vendor}.yaml").read_text(encoding="utf-8"))
    # Drive conditions come from the captured waveform, not today's YAML:
    # several golden CSVs predate condition changes in config/inputs/.
    i_load = 1.0
    temp = float(cfg["conditions"].get("temperature", 25.0))

    print(f"[plot] golden <- {DATA_DIR / f'{vendor}_golden.csv'}")
    g = load_golden_csv(vendor)
    vbus, vgs_on = infer_drive_conditions(g)
    print(f"[plot] drive inferred from golden: Vbus={vbus:g} V, "
          f"Vgs_on={vgs_on:g} V")
    print(f"[plot]   {len(g['time'])} pts | "
          f"vds [{g['v_ds'].min():.3f}, {g['v_ds'].max():.3f}] V")

    print("[plot] simulating Tier-2 ...")
    t0 = time.perf_counter()
    t = run_tier2(vendor, cfg, vbus, vgs_on, i_load, temp)
    print(f"[plot] tier-2: {time.perf_counter()-t0:.1f}s | "
          f"{len(t['time'])} pts")

    # ---- plot 1: full-run overlay (3 cycles) ----
    fig, ax = plt.subplots(figsize=(11.5, 6.0), layout="constrained")
    ax.plot(g["time"] * 1e6, g["v_ds"], lw=1.6, color="tab:blue",
            label="Vendor golden")
    ax.plot(t["time"] * 1e6, t["v_ds"], lw=1.4, ls="--", color="tab:red",
            alpha=0.9, label="Tier-2 CPWL")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Vds [V]")
    ax.set_title(f"Switching Transient — {vendor} "
                 f"(50 kHz, 50% duty, resistive {i_load:.0f} A)")
    _legend_east(ax)
    p1 = IMG_DIR / f"{vendor}_switching_3cycle.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"[plot] saved {p1.name}")

    # ---- plot 2: zoomed turn-on and turn-off edges of Vds (dynamic) ----
    def edge_window(t, vd, kind):
        """Find [lo, hi] time window tightly around a Vds transition.

        Detects where vd crosses from the 90% level toward the 10% level
        (fall: turn-on) or reverse (rise: turn-off), scanning through
        intermediate samples. Window = crossing span padded, so the edge
        fills the zoom for any vendor/edge speed.
        """
        v_hi = float(np.percentile(vd, 98))
        below = vd < 0.5 * abs(v_hi)
        v_lo = float(np.percentile(vd[below], 50)) if below.any() else 0.0
        span = max(v_hi - v_lo, 1e-9)
        hi_lvl = v_lo + 0.9 * span
        lo_lvl = v_lo + 0.1 * span

        mid = len(vd) // 3

        if kind == "fall":
            leave = np.where((vd[:-1] > hi_lvl) & (vd[1:] < hi_lvl))[0]
            settle = lo_lvl
            start_lvl_test = vd >= hi_lvl
        else:
            leave = np.where((vd[:-1] < lo_lvl) & (vd[1:] > lo_lvl))[0]
            settle = hi_lvl
            start_lvl_test = vd <= lo_lvl
        leave = leave[leave >= mid]
        if len(leave) == 0:
            return None
        i = int(leave[-1])

        # walk forward until settled at target level (or limit)
        limit = min(len(vd) - 1, i + len(vd) // 10)
        k = i + 1
        while k < limit:
            if kind == "fall" and vd[k] <= settle:
                break
            if kind == "rise" and vd[k] >= settle:
                break
            k += 1
        # walk backward to where it left the start level
        j = i
        while j > 0:
            if kind == "fall" and vd[j] >= hi_lvl:
                break
            if kind == "rise" and vd[j] <= lo_lvl:
                break
            j -= 1
        t0, t1 = float(t[j]), float(t[min(k + 2, len(vd) - 1)])
        dur = max(t1 - t0, 8 * float(np.mean(np.diff(t))))
        pad = 0.5 * dur
        return t0 - pad, t1 + pad


    fig, axs = plt.subplots(2, 1, figsize=(11.5, 8.0), layout="constrained")

    for axp, kind, title in ((axs[0], "fall",
                              "Zoom: Turn-On Edge (Vds falling)"),
                             (axs[1], "rise",
                              "Zoom: Turn-Off Edge (Vds rising)")):
        wg = edge_window(g["time"], g["v_ds"], kind)
        wt = edge_window(t["time"], t["v_ds"], kind)
        drew = []
        if wg:
            lo_g, hi_g = wg
            mgw = (g["time"] >= lo_g) & (g["time"] <= hi_g)
            if mgw.any():
                axp.plot((g["time"][mgw] - wg[0]) * 1e9, g["v_ds"][mgw],
                         lw=1.8, color="tab:blue", label="Vendor golden")
                drew.append(True)
        if wt:
            lo_t, hi_t = wt
            mtw = (t["time"] >= lo_t) & (t["time"] <= hi_t)
            if mtw.any():
                axp.plot((t["time"][mtw] - wt[0]) * 1e9, t["v_ds"][mtw],
                         lw=1.6, ls="--", color="tab:red",
                         label="Tier-2 CPWL")
                drew.append(True)
        if not drew:
            axp.text(0.5, 0.5, "no edge detected", ha="center")
        axp.set_xlabel("Time from edge start [ns]")
        axp.set_title(title)
        _legend_east(axp)

    p2 = IMG_DIR / f"{vendor}_switching_edge_zoom.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"[plot] saved {p2.name}")

    # ---- fitness metrics ----
    mg, mt = metrics_of(g, vbus, i_load), metrics_of(t, vbus, i_load)

    def rel(a, b):
        return round((a - b) / max(abs(b), 1e-12) * 100, 2)

    errors = {"td_on": rel(mt["td_on_ns"], mg["td_on_ns"]),
              "tr": rel(mt["tr_ns"], mg["tr_ns"]),
              "td_off": rel(mt["td_off_ns"], mg["td_off_ns"]),
              "tf": rel(mt["tf_ns"], mg["tf_ns"]),
              "e_on": rel(mt["e_on_uj"], mg["e_on_uj"]),
              "e_off": rel(mt["e_off_uj"], mg["e_off_uj"])}

    tg = np.linspace(max(g["time"][0], t["time"][0]),
                     min(g["time"][-1], t["time"][-1]), 2000)
    err = compute_curve_error(np.interp(tg, g["time"], g["v_ds"]),
                              np.interp(tg, t["time"], t["v_ds"]))
    wave_pct = round(err.relative_rmse_pct, 3)

    passed = (all(abs(v) <= 20 for k, v in errors.items()
                  if k in ("td_on", "tr", "td_off", "tf"))
              and all(abs(v) <= 15 for k, v in errors.items()
                      if k in ("e_on", "e_off"))
              and wave_pct <= 5.0)
    verdict = ("PASS - Tier-2 switches like the vendor model" if passed
               else "FAIL - see errors_pct")

    payload = {
        "schema": "piecespice.switching.fitness/1.0",
        "backend": "golden=LTspice capture; tier2=venv shared-lib transient",
        "vendor": vendor,
        "device": cfg.get("device", vendor),
        "conditions": {"f_sw_hz": F_SW, "duty": 0.5, "vbus_v": vbus,
                       "vgs_on_v": vgs_on, "i_load_a": i_load,
                       "temperature_c": temp},
        "golden": {k: round(v, 5) for k, v in mg.items()},
        "tier2": {k: round(v, 5) for k, v in mt.items()},
        "errors_pct": errors,
        "waveform_rel_rmse_pct": wave_pct,
        "tolerances_pct": {"timing": 20.0, "energy": 15.0, "waveform": 5.0},
        "passed": passed,
        "verdict": verdict,
    }
    jpath = REPORT_DIR / f"{vendor}_switching_metrics.json"
    jpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[plot] saved {jpath.name}")

    rows = [("td_on [ns]", mg["td_on_ns"], mt["td_on_ns"], errors["td_on"]),
            ("tr [ns]", mg["tr_ns"], mt["tr_ns"], errors["tr"]),
            ("td_off [ns]", mg["td_off_ns"], mt["td_off_ns"],
             errors["td_off"]),
            ("tf [ns]", mg["tf_ns"], mt["tf_ns"], errors["tf"]),
            ("E_on [uJ]", mg["e_on_uj"], mt["e_on_uj"], errors["e_on"]),
            ("E_off [uJ]", mg["e_off_uj"], mt["e_off_uj"], errors["e_off"]),
            ("Vds on [mV]", mg["vds_on_mv"], mt["vds_on_mv"], None),
            ("Id peak [A]", mg["i_peak_a"], mt["i_peak_a"], None),
            ("waveform RMSE %", np.nan, wave_pct, None)]
    print(f"\n{'metric':<18}{'golden':>12}{'tier-2':>12}{'err %':>10}")
    print("-" * 52)
    for name, a, b, e in rows:
        es = f"{e:+.2f}" if e is not None else "-"
        print(f"{name:<18}{a:>12.4f}{b:>12.4f}{es:>10}")
    print(f"\nwaveform rel-RMSE: {wave_pct}% (<=5%)")
    print("VERDICT:", verdict)
    return 0


if __name__ == "__main__":
    vendor_arg = sys.argv[1] if len(sys.argv) > 1 else "epc2216"
    sys.exit(main(vendor_arg))