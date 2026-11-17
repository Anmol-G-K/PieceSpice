"""M4-subset fitness evaluation: golden vs Tier-2 switching comparison.

Library-level so notebooks only plot. Reuses metrics/switching.py +
metrics/compare.py - no new metric math here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from approximation.charge_model import integrate_charge  # noqa: F401
from metrics.loss import SwitchingLossMetrics, compute_switching_energy
from metrics.static import compute_curve_error
from metrics.switching import extract_switching_times
from simulation.testbenches.simple_switching import (
    F_SW_HZ, SwitchingRun, run_simple_switching, steady_state_slice)



def infer_drive_conditions(golden: SwitchingRun | dict) -> tuple[float, float]:
    """Recover (vbus, vgs_on) actually used by a captured golden run.

    Config YAML may have changed since a golden CSV was recorded (e.g. CPM2
    Vbus 800 V vs a 10 V capture). Driving the Tier-2 model with today's
    config then compares two different experiments. The off-state drain
    plateau and the gate plateau are recoverable from the waveform itself.
    """
    vds = getattr(golden, "v_ds", None)
    vgs = getattr(golden, "v_gs", None)
    if vds is None:                      # load_golden_csv() returns arrays
        vds, vgs = golden["v_ds"], golden["v_gs"]
    vbus = float(np.percentile(vds, 95))
    vgs_on = float(np.percentile(vgs, 95))
    return vbus, vgs_on

_LOGGER = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    """All fitness-relevant numbers for ONE switching run."""

    td_on_ns: float
    tr_ns: float
    td_off_ns: float
    tf_ns: float
    e_on_uj: float
    e_off_uj: float
    vds_on_mean_v: float       # conduction drop during on-window
    i_peak_a: float            # turn-on current overshoot


@dataclass
class SwitchFitness:
    """Golden-vs-Tier-2 comparison result."""

    golden: RunMetrics
    tier2: RunMetrics
    timing_rel_err_pct: dict      # per-timing-metric relative error
    energy_rel_err_pct: dict
    waveform_rel_rmse_pct: float  # Vds(t) full-cycle relative RMSE
    passed: bool                  # all gates within tolerance
    evaluable: bool = True             # False when crossings/spans prevent a fair gate
    verdict: str = ""

    def summary(self) -> str:
        lines = [
            "=== M4-subset Switching Fitness ===",
            f"waveform rel-RMSE (Vds, steady cycle): "
            f"{self.waveform_rel_rmse_pct:.2f}%",
        ]
        for k, v in self.timing_rel_err_pct.items():
            lines.append(f"{k:>8}: {v:+7.2f}%")
        for k, v in self.energy_rel_err_pct.items():
            lines.append(f"{k:>8}: {v:+7.2f}%")
        lines.append(f"VERDICT: {self.verdict}")
        return "\n".join(lines)


def _metrics_from_run(run: SwitchingRun) -> RunMetrics:
    """Extract timing + loss + conduction metrics from one run."""
    sw = extract_switching_times(
        time=run.time, v_ds=run.v_ds, i_d=run.i_d, v_gs=run.v_gs,
        v_bus=run.v_bus, i_load=run.i_load)

    # switching-energy windows: around the drain edges of the LAST cycle.
    # On/off windows bracket each gate edge.
    # Match metrics.switching: the 95th percentile is duty-cycle agnostic.
    # median(last 3/4) is 0 V for a 50% duty cycle captured from t=0.
    plateau = float(np.percentile(run.v_gs, 95))
    rising = np.where((run.v_gs[:-1] < 0.5 * plateau)
                      & (run.v_gs[1:] >= 0.5 * plateau))[0]
    falling = np.where((run.v_gs[:-1] > 0.5 * plateau)
                       & (run.v_gs[1:] <= 0.5 * plateau))[0]

    def _energy_around(idx_arr):
        if len(idx_arr) == 0:
            return float("nan")
        i = int(idx_arr[-1])
        lo = max(0, i - 200)
        hi = min(len(run.time) - 1, i + 200)
        return compute_switching_energy(run.time, run.v_ds, run.i_d,
                                        float(run.time[lo]),
                                        float(run.time[hi]))

    e_on = _energy_around(rising)
    e_off = _energy_around(falling)

    on_mask = run.v_gs > 0.9 * plateau
    vds_on_mean = float(np.mean(run.v_ds[on_mask])) if on_mask.any() else 0.0

    return RunMetrics(
        td_on_ns=float(sw.delay_on_ns),
        tr_ns=float(sw.rise_time_ns),
        td_off_ns=float(sw.delay_off_ns),
        tf_ns=float(sw.fall_time_ns),
        e_on_uj=e_on,
        e_off_uj=e_off,
        vds_on_mean_v=vds_on_mean,
        i_peak_a=float(np.max(run.i_d)),
    )


def _rel_err_pct(a: float, b: float) -> float:
    """Signed relative error; NaN for a missing or near-zero reference."""
    if not (np.isfinite(a) and np.isfinite(b)) or abs(b) < 1e-12:
        return float("nan")
    return (a - b) / abs(b) * 100.0


def compare_switching_fitness(
    vendor_lib: Path | None = None,
    vendor_subckt: str = "",
    vendor_pins: list[str] | None = None,
    tier2_lib: Path | None = None,
    tier2_subckt: str = "",
    *,
    vbus: float,
    vgs_on: float,
    i_load_target: float = 1.0,
    temperature: float = 25.0,
    timing_tol_pct: float = 20.0,
    energy_tol_pct: float = 15.0,
    waveform_tol_pct: float = 5.0,
    golden_run: SwitchingRun | None = None,
    tier2_run: SwitchingRun | None = None,
) -> SwitchFitness:
    """Run both models under identical conditions and score the Tier-2 one.

    If `golden_run` / `tier2_run` (raw SwitchingRun from
    simple_switching.run_simple_switching) are supplied, they are reused and
    no simulation happens - callers that already ran both models pass them
    here to avoid duplicate work.

    Tolerances per M4-subset plan: timing ±20%, energies ±15%,
    waveform rel-RMSE ±5%.
    """
    if golden_run is not None and tier2_run is not None:
        for attr in ("v_bus", "vgs_on"):
            gv, tv = getattr(golden_run, attr), getattr(tier2_run, attr)
            if not np.isclose(gv, tv, rtol=1e-6, atol=1e-9):
                raise ValueError(
                    f"golden and Tier-2 runs use different {attr} "
                    f"({gv:g} vs {tv:g}); infer the drive from the golden "
                    "with infer_drive_conditions() before comparing")
        period = 1.0 / F_SW_HZ
        if golden_run.time[-1] - golden_run.time[0] < 0.9 * period:
            # VM-10: the CPM2 golden aborts before turn-off. A last-period
            # slice would compare golden t=0..14us with Tier-2 t=40..60us.
            # Keep both full runs; the waveform window below uses the common
            # leading span, and turn-off metrics stay NaN (not evaluable).
            run_g, run_t = golden_run, tier2_run
        else:
            run_g = steady_state_slice(golden_run)
            run_t = steady_state_slice(tier2_run)
    else:
        if vendor_lib is None or tier2_lib is None:
            raise ValueError(
                "provide either both runs or both library paths")
        common = dict(vbus=vbus, vgs_on=vgs_on, i_load_target=i_load_target,
                      temperature=temperature)
        run_g = steady_state_slice(run_simple_switching(
            Path(vendor_lib), vendor_subckt, vendor_pins, **common))
        run_t = steady_state_slice(run_simple_switching(
            Path(tier2_lib), tier2_subckt, ["drain", "gate", "source"],
            **common))

    mg = _metrics_from_run(run_g)
    mt = _metrics_from_run(run_t)

    timing_errs = {
        "td_on": _rel_err_pct(mt.td_on_ns, mg.td_on_ns),
        "tr": _rel_err_pct(mt.tr_ns, mg.tr_ns),
        "td_off": _rel_err_pct(mt.td_off_ns, mg.td_off_ns),
        "tf": _rel_err_pct(mt.tf_ns, mg.tf_ns),
    }
    energy_errs = {
        "e_on": _rel_err_pct(mt.e_on_uj, mg.e_on_uj),
        "e_off": _rel_err_pct(mt.e_off_uj, mg.e_off_uj),
    }

    # align time grids before comparing waveforms
    t_common = np.linspace(run_g.time[0], run_g.time[-1],
                           min(len(run_g.time), len(run_t.time)))
    t2_on_grid = np.interp(t_common, run_t.time, run_t.v_ds)
    g_on_grid = np.interp(t_common, run_g.time, run_g.v_ds)
    err = compute_curve_error(g_on_grid, t2_on_grid)

    finite_timing = [v for v in timing_errs.values() if np.isfinite(v)]
    finite_energy = [v for v in energy_errs.values() if np.isfinite(v)]
    ok_timing = all(abs(v) <= timing_tol_pct for v in finite_timing)
    ok_energy = all(abs(v) <= energy_tol_pct for v in finite_energy)
    ok_wave = err.relative_rmse_pct <= waveform_tol_pct
    passed = ok_timing and ok_energy and ok_wave

    evaluable = (len(finite_timing) == len(timing_errs)
                 and len(finite_energy) == len(energy_errs))
    if not evaluable:
        missing = [k for k, v in {**timing_errs, **energy_errs}.items()
                   if not np.isfinite(v)]
        verdict = ("N/A - level crossing missing for: " + ",".join(missing))
        passed = False
    elif passed:
        verdict = "PASS - Tier-2 switches like the vendor model"
    else:
        fails = []
        if not ok_wave:
            fails.append(f"waveform {err.relative_rmse_pct:.1f}%")
        if not ok_timing:
            bad = [k for k, v in timing_errs.items()
                   if abs(v) > timing_tol_pct]
            fails.append("timing " + ",".join(bad))
        if not ok_energy:
            bad = [k for k, v in energy_errs.items()
                   if abs(v) > energy_tol_pct]
            fails.append("energy " + ",".join(bad))
        verdict = "FAIL - " + "; ".join(fails)

    return SwitchFitness(
        evaluable=evaluable,
        golden=mg, tier2=mt,
        timing_rel_err_pct=timing_errs,
        energy_rel_err_pct=energy_errs,
        waveform_rel_rmse_pct=err.relative_rmse_pct,
        passed=passed,
        verdict=verdict,
    )
