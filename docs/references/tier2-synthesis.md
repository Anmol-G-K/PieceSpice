# Tier-2 subcircuit synthesis & fitting pipeline (M3.5–M3.8, Aug 2026)

Patterns from building the Tier-2 CPWL MOSFET synthesis stage: fitting
pipeline (`pipeline/fitting.py::fit_pwl_model`), subcircuit emitter
(`models/netlist_builder.py`), and the validating notebook (02).

## Emission bugs — all found only by evaluating the emitted expression

Text inspection is NOT enough; every emitted SPICE expression must be
evaluated numerically against its source table. Three distinct bugs slipped
past visual review in one function:

1. **Absolute-slope soft-abs ramps accumulate** (see
   `pwl-approximation-pitfalls.md`). The channel transfer and the Qoss
   capacitor each independently reintroduced this bug even after it was
   fixed once — any new chained-ramp emitter must use delta slopes
   (`s_k − s_{k−1}`) with the first ramp at the first data point.
2. **Rounding at table level destroys small charges.** `round(q, 9)` on
   charges of ~1e-8 C loses ~10% per entry; the emitted expression then
   deviates between breakpoints even with correct slopes. Never round
   physical values at data-structure level; format only at print time
   (`%.6e` keeps 7 significant digits).
3. **Non-ASCII characters in comment lines break ngspice parsing on cp1252
   consoles** (a `§` in a comment). Emit ASCII-only netlists; assert
   `text.encode("ascii")` in tests.

## Element-syntax gotchas (ngspice)

- Charge-form capacitors are **C elements with `Q={expr}`**, NOT B-sources —
  `Bxxx ... Q=` is invalid.
- There is no `u()` step function in ngspice expressions; use the soft-abs
  form `(d + |d|)/2` for ramps.
- Behavioral G-source: `Gxxx n+ n- VALUE = {expr}`.

## Verification sequence for a synthesized model

1. Evaluate emitted expression at table breakpoints AND segment midpoints vs
   the true integrated curve (midpoints catch divergence that breakpoint-
   only checks miss).
2. Instantiate the subckt in an actual PySpice circuit (X-instance!) — a
   bare include without an X line silently simulates nothing and reports
   Id=0 everywhere. If currents read zero, first check the X line exists.
3. Compare endpoint currents vs vendor golden per Vgs curve; worst error
   < 0.6% achieved on real TI data (target < 5%).

## Fitting pipeline structure (generalizes)

`fit_pwl_model(vendor)`: load outputs YAML → derive shared breakpoints +
per-curve rows → scalar params (Rds median, Vth, body-diode IS/N two-point
exponential fit with N clamped [0.5,4]) → charge tables via integrate_charge
→ PWLModelSpec → builder → write .lib to `reports/models/`.

## Vgs-family key formatting — truncation corrupts the axis

`f"curve_id_vgs_{int(v)}_v"` stores a 4.5 V curve under key `..._4_v`,
mislabeling its axis in every downstream consumer (the fitting stage then
builds the transfer PWL with a shifted breakpoint — internally consistent,
so acceptance tests PASS while wrong). Fix: `%g` formatter
(`curve_id_vgs_{v:g}_v`) plus a regex-based validator for family keys.
Lesson: **internally consistent test output does not prove axis labels are
correct**; check labels against config values.

## Notebook architecture (user-mandated)

Notebooks must ONLY call library functions and plot their outputs — no
derivation math inline ("the notebook should not bluff"). When a cell needs
a derivation (e.g. tanh knee parameters from a compressed curve), add a
library function (`smoother.fit_tanh_knee(v, i) -> TanhKneeFit`) and shrink
the cell to call+plot.

Fabrication trap: demo cells inventing parameters (`gm_est*3`, arbitrary
vds_sat) make the smoothed curve diverge 7× from the hard PWL — always
derive demo parameters FROM the same data being plotted (e.g. Vds_sat where
the PWL crosses tanh(1)=76% of Id_sat), so curves coincide away from the
knee by construction.

Also: when rewriting shared type modules, re-check that every public class
they used to export still exists — dropping CurveData broke half the
codebase's imports at collection time.

## Plotting

Reuse `common.plotting.plot_curves()` (EAST legend, EngFormatter, setup_style)
instead of hand-building axes; keep notebook plotting to thin wrappers.
