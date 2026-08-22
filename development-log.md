# PieceSpice Development Log

Timestamped record of all development work. Newest entries at the top.
All timestamps are IST (UTC+05:30).

---

## 2026-08-22 15:21 IST — Root cause analysis: Id = 1.5–1.7 kA vs datasheet ID = 246 A (new defect VM-04)

### Trigger

User flagged Curve 1 output characteristics showing Id ≈ 1.5–1.7 kA at Vgs = 10 V against
the IRFS7730PBF Absolute Maximum Ratings (ID = 246 A @ TC=25 °C silicon-limited, 195 A
package-limited, IDM 984 A pulsed).

### Analysis performed

- Re-ran Curve 1 and Curve 2 sweeps; measured Id @ Vgs=10 V, Vds=10 V = **1668.1 A**.
- Hand-solved the LEVEL=1 core equations with the model's own parameters
  (KP=465.97, VTO=4.34579, LAMBDA=0.0605061, RS=2.03 mΩ, RD=0.1 mΩ):
  analytic Id_sat @ 10 V/10 V = **1727 A** — matches the measurement to ~4%.
- Verified the fitter had no free parameter left: predicted Rds(on) =
  1/(β·Vov) + RS + RD = 2.509 mΩ vs extracted 2.52 mΩ ✅; gfs @ 100 A ≈ 215 S vs
  datasheet 249 S (−14%, expected for LEVEL=1).
- Thermal cross-check of the datasheet rating: ID = 246 A is "Silicon Limited" —
  P = 246² × 2.52 mΩ ≈ 152 W < PD max 375 W. At 1668 A, P ≈ 7 kW = 18.7× rated PD.

### Verdict

Not an extraction bug. The vendor model *does* conduct kA-level current by design:
LEVEL=1 has no maximum-current parameter, no mobility degradation, no velocity
saturation, no thermal feedback — so Id grows as Vov² without limit once KP and VTO are
pinned by Rds(on) and the transfer shape. The datasheet's 246 A is a thermal/package
limit that a purely electrical macromodel cannot encode. The model's realistic operating
point is Id = 246 A at Vgs ≈ 5.66 V (interpolated from Curve 2).

Recorded as defect **VM-04** in the new `vendor_model_defects.md` catalog, alongside
VM-01 (no subthreshold / fake-high Vth), VM-02 (sense-and-mirror transient pathology),
VM-03 (charge-inaccurate Cgd), and VM-05 (corrupted t=0 operating point).

### Operating rules adopted for this device model

1. Never drive beyond ~250–350 A continuous in testbenches.
2. Derive operating points by transfer-curve interpolation at the target load current,
   never from "Vgs = 10 V".
3. Tier 2 synthesis should consider a soft current clamp near datasheet ID.

---

## 2026-08-22 15:40 IST — Plotting overhaul + notebook 01 updated with gate-charge study

### Motivation

User review: existing plots were "not professional enough — bad layout and placement."
All notebook figures flow through `common/plotting.py`, so the fix was made centrally.
300 dpi resolution retained per project constraint (plan §4.3); everything else overhauled.

### Changes

**`common/plotting.py` (rewritten)**
- Replaced the fragile two-pass `tight_layout` legend-width hack with
  **constrained layout** (`layout="constrained"`) — matplotlib now reserves exactly the
  space the EAST legend needs; no clipped legends, no dead whitespace, deterministic.
- Added engineering-unit tick formatting: `EngFormatter` derived from each curve's
  unit field (A -> kA on high-current plots, F -> pF/nF, V, mOhm...), so tick labels
  read "250 A", "13 nF" instead of raw floats/scientific notation.
- On log axes: minor formatter set to NullFormatter + LogLocator(subs='auto') to kill
  duplicate minor labels.
- Typography scale retuned (12/13/14 pt), title padding, thinner spines (0.9),
  major/minor tick sizes, grid drawn below data (`axes.axisbelow`), minor grid off,
  legend styling (square frame, edgecolor 0.75).
- Default figure size 10.5 x 5.8 (16:9-ish, better for notebooks/reports than the old
  12 x 6.75 that shrank the axes when a long legend was attached).
- `plot_waveforms` updated to the same machinery.

**`simulation/testbenches/gate_charge.py`**
- `plot_gate_charge` migrated to constrained layout + EngFormatter("nC") x-axis;
  removed `fig.tight_layout()`.

**`notebooks/01_datasheet_curves_extraction.ipynb` (rebuilt and re-executed)**
- Added **Study 5 - Gate Charge (quasi-static reconstruction)** using
  `reconstruct_gate_charge` at datasheet conditions (Id = 100 A, Vbus = 38 V), with
  printed breakpoint comparison vs datasheet typ values (Qgs 55 / Qgd 79 / Qg 271 nC)
  and the annotated three-segment plot. The intro table's "gate charge excluded" note
  replaced by an explanation of the quasi-static approach.
- All cells re-executed via nbconvert (inline backend): 7 figures embedded at 300 dpi,
  execution counts 1-9, no errors.
- Fixed LIB path to be repo-root-relative-resolved instead of hardcoded absolute.

### Verification

- Visual QA of exported figures (output-characteristics family and gate-charge plot):
  legend fully outside right with zero overlap, no clipped text, engineering-unit ticks
  (kA / nC), annotations readable, Miller plateau line visible, no layout flaws.
- Gate-charge numbers in-notebook: Vgp 5.07 V, Qgs 67.3 nC, Qgd 36.0 nC, Qg 183.2 nC.
- Full pytest suite: **8 passed** — no regressions.

---

## 2026-08-22 14:47 IST — Curve 5 (Gate Charge) delivered via quasi-static reconstruction (Option A); datasheet cross-validation analysis

### Context

Curve 5 (Vgs vs Qg under constant-current gate drive) was blocked on the IRFS7730PBF
vendor subcircuit (`SpiceModels/irfs7730pbf.spi` → `irfb_s_sl7730pbf`) by a MODPEX
sense-and-mirror network pathology: under dynamic drain transitions the EV16/CAP/FI1/FI2
network enters high conduction and pumps amp-level junk current into the internal gate
node, driving Vgs to -37 V. Confirmed structural (Gear/reltol/timestep changes have no
effect). Full root-cause analysis in `gate_charge_error_analysis.md`.

Decision: do NOT attempt to fix the transient testbench; implement Option A —
quasi-static Qg reconstruction from already-validated Curve 2 (transfer) and
Curve 4 (Ciss/Coss/Crss vs Vds) data.

### Changes made

**File: `simulation/testbenches/gate_charge.py`**

1. Module docstring rewritten to document the vendor-model blockage and the Option A
   approach, cross-referencing `gate_charge_error_analysis.md`.
2. Added imports: `pandas`, `common.types.CurveData`.
3. Added `_interp_curve(curve, x)` — interpolates a `CurveData` onto arbitrary x,
   handling both ascending and descending x-axes (np.interp requires ascending;
   CapacitanceTestbench returns log-spaced Vds which is ascending, but defensive
   handling added after an initial bug where descending axes produced negative Qgd).
4. Added `_crss_integral(crss_curve, v_hi, v_lo, n_points=200)` — integrates Crss(Vds)
   over the drain swing from v_hi down to v_lo on a descending axis so dQ is positive
   as the drain collapses. Returns (total charge C, descending Vds array, cumulative
   charge array).
5. Added **`reconstruct_gate_charge(transfer_curve, ciss_curve, crss_curve, v_bus=40.0,
   i_load=50.0, v_gs_max=10.0, v_ds_on=0.1, n_points=200)`** — the core three-segment
   quasi-static model:
   - Segment 1 (Qgs): plateau voltage Vgp found by interpolating the transfer curve at
     Id = Iload; Qgs = Ciss(Vds = Vbus) x Vgp.
   - Segment 2 (Qgd / Miller plateau): Qgd = integral of Crss(Vds) dVds over the drain
     swing Vbus -> Vds,on, expressed on a cumulative axis while Vgs stays flat at Vgp.
   - Segment 3 (tail): Ciss(Vds ~ Vds,on) integrated from Vgp -> Vgs_max.
   - Returns `(pd.DataFrame([Qg_nC, Vgs_V]), breakpoints_dict)` where breakpoints dict
     has keys `v_plateau_V`, `q_gs_nC`, `q_gd_nC`, `q_tail_nC`, `q_total_nC`.
6. Added **`plot_gate_charge(df, breakpoints, save_path=None)`** — plots Vgs vs Qg with:
   - dashed horizontal line at the Miller plateau voltage,
   - vertical markers + arrow annotations at Qgs, Qgs+Qgd, and total Qg @ 10 V,
   - EAST-outside legend (`loc='center left', bbox_to_anchor=(1.02, 0.5)`) per the
     project's mandatory plotting constraint (plan §4.3).
7. The blocked transient routine `run_gate_charge(...)` is retained unchanged in the
   module for reference only — no code path calls it.

No changes to any other module. No mutation of the vendor `.spi` file. No new transient
simulation attempted.

### Verification results (real device: irfs7730pbf.spi)

Pipeline run: StaticSweepTestbench.run_transfer_characteristics +
CapacitanceTestbench.run -> reconstruct_gate_charge.

Initial run (v_bus=40 V, i_load=50 A):
- Plateau voltage Vgp = 4.70 V
- Qgs = 62.30 nC | Qgd = 36.99 nC | tail = 85.89 nC | **Qg total = 185.18 nC**
- Within expected 150-200 nC range. Monotonicity check passed.
- Plot visually confirmed (vision check): pre-plateau rise, flat Miller plateau at
  4.7 V, tail rise to 10 V, annotations present, legend EAST outside.

Full pytest suite: **8 passed** — no regressions.

Artifacts produced:
- `reports/gate_charge_curve5.csv` — [Qg_nC, Vgs_V] table
- `reports/gate_charge_reconstructed.png` — annotated three-segment plot

Bug fixed during verification: first run produced Qgd = -36.99 nC because Crss curve
x-axis ordering interacted badly with np.interp over a descending sweep axis; resolved
by `_crss_integral` keeping the physically meaningful descending-Vds integration order.

---

## 2026-08-22 14:47 IST — Datasheet cross-validation: why extracted values deviate from the IRFS7730PBF datasheet

Datasheet dynamic/static tables (Tj = 25 °C) were compared against model-extracted
values, re-run under exact datasheet conditions (Id = 100 A, Vds = 38 V).

### Agreement (extraction machinery validated)

| Quantity | Model-extracted | Datasheet typ | Error |
|---|---|---|---|
| Ciss @ Vds=25 V | 13 381 pF | 13 660 pF | −2% |
| Coss @ Vds=25 V | 1 108 pF | 1 120 pF | −1% |
| Crss @ Vds=25 V | 685 pF | 690 pF | −0.7% |
| Rds(on) @ Vgs=10 V | 2.52 mΩ | 2.2 (max 2.6) mΩ | within spec |

Conclusion: Curve 2/4 testbenches measure the vendor model faithfully.

### Root cause of deviations (traced to specific lines of irfs7730pbf.spi)

Line 21-23 of the vendor file:

```spice
M1 9 7 8 8 MM L=100u W=100u
.MODEL MM NMOS LEVEL=1 IS=1e-32
+VTO=4.34579 LAMBDA=0.0605061 KP=465.97
```

1. **Vth deviation (extracted 4.30 V vs datasheet window 2.1–3.7 V)** — structural:
   - `IS=1e-32` eliminates subthreshold conduction entirely. A LEVEL=1 SPICE MOSFET
     conducts zero current until VGS > VTO, whereas a real device reaches the JEDEC
     Vth definition current (250 uA) gradually below threshold.
   - Consequently the MODPEX fitter could not fit VTO to the datasheet Vth; it fitted
     an *effective strong-inversion onset* (VTO = 4.35 V) tuned jointly with KP and
     LAMBDA to match the high-current transfer characteristic. The datasheet's Vth
     quantity simply does not exist inside this macromodel.
   - Same trade-off visible in gfs: model ~350 S near high Id vs datasheet 249 S — a
     single square-law LEVEL=1 core cannot match Vth, gfs, and Rds(on) simultaneously.

2. **Qgs too high (67 nC @ 100 A conditions vs datasheet 55 nC)** — consequence of the
   inflated effective threshold: the reconstructed plateau sits at ~5.07 V instead of
   the real device's lower plateau, so segment 1 absorbs excess charge
   (Qgs ~ Ciss x Vplateau).

3. **Qgd too low (~36 nC vs datasheet 79 nC) and Qg total low (184 vs 271 nC, −32%)** —
   the MODPEX sense-and-mirror network reproduces small-signal Crss accurately at fixed
   bias (hence the perfect Crss @ 25 V match), but its large-signal charge delivery
   across the full 38 V -> 0 drain swing under-integrates versus the real device's Cgd
   bulge during collapse. Extracted Crss does rise from 685 pF @ 25 V toward ~3.4 nF at
   low Vds, but not steeply enough in integrated charge.

### Datasheet-condition reconstruction summary (Id=100 A, Vbus=38 V)

| Segment | Reconstructed | Datasheet typ |
|---|---|---|
| Plateau voltage | 5.07 V | — |
| Qgs | 67.25 nC | 55 nC |
| Qgd | 35.95 nC | 79 nC |
| Tail | 80.57 nC | — |
| Qg total @ 10 V | 183.78 nC | 271 nC |

### Implication for PieceSpice

Every observed deviation is a documented defect of the golden vendor model itself, not
of the extraction pipeline. This supports the framework premise: vendor "golden" models
carry systematic errors (behavioral DC core without subthreshold; AC-calibrated but not
charge-accurate capacitance emulation). Validation targets must be chosen deliberately
(datasheet values vs vendor-model values), and this should inform Tier 2 model targets.

TODO carried forward: consider appending this cross-validation as §8 of
`gate_charge_error_analysis.md`.

---

## 2026-08-22 (morning) — Project indexing report (pre-work baseline)

Prior to implementation, a full repository index was performed:

- Walked full directory tree; read every source module.
- Cross-referenced `piecespice_plan.md` against disk state.
- Verified `uv run pytest -q`: 8 passed.
- Key findings recorded:
  - Milestones 1 and static portion of 2 solidly done (Curves 1, 2, 3, 4, 6 working).
  - Milestones 3–5 skeletal: netlist_builder ignores fitted breakpoints; no
    double_pulse.py or buck_converter.py; metrics/switching.py returns hardcoded
    timing placeholders; notebooks 02–05 are unexecuted stubs.
  - Missing vs plan: parser/extractor.py, approximation/smoother.py,
    simulation/testbenches/double_pulse.py, simulation/testbenches/buck_converter.py.
  - Known blocker documented in gate_charge_error_analysis.md (Curve 5 transient).
  - Plan checkboxes stale in both directions; README.md empty; unused deps in
    pyproject.toml (polars, shapely, rdp, seaborn, pandas, hypothesis,
    pytest-benchmark).
