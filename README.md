<div align="center">

# ⚡ PieceSpice

**High-Fidelity Piecewise SPICE Modeling Framework**

*PLECS-class speed (target). SPICE-class resolution. Standard SPICE netlists.*

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LTspice](https://img.shields.io/badge/LTspice-26-orange)](https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html)
[![spicelib](https://img.shields.io/badge/spicelib-powered-yellow)](https://pypi.org/project/spicelib/)
[![Tests](https://img.shields.io/badge/tests-118%20passing%20%C2%B7%204%20skipped-brightgreen?logo=pytest&logoColor=white)](#testing)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()
[![uv](https://img.shields.io/badge/managed%20with-uv-dead00?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

[![Infineon](https://img.shields.io/badge/supported-Infineon%20%2F%20IR-0A8276?labelColor=1a1a1a)](#validated-devices) [![TI](https://img.shields.io/badge/supported-Texas%20Instruments-CC0000?labelColor=1a1a1a)](#validated-devices) [![EPC](https://img.shields.io/badge/supported-EPC%20GaN-006FBA?labelColor=1a1a1a)](#validated-devices) [![Wolfspeed](https://img.shields.io/badge/supported-Wolfspeed%20SiC-58238C?labelColor=1a1a1a)](#validated-devices)

*Vendor `.lib` in → validated Tier-2 CPWL **or** Tier-3 TPWL macromodel out*

</div>

---

## 📖 Overview

**PieceSpice** bridges the simulation gap between full physics SPICE models
(high fidelity, hours of compute, convergence failures) and system-level
ideal-switch tools (instant, but no ringing/losses). It automatically
distills vendor MOSFET subcircuits into one of **three selectable model options**:

- **Tier-2 — CPWL macromodel.** Continuous piecewise-linear channel with a
  `tanh` saturation knee plus charge-defined capacitors, emitted as a small
  standard-SPICE subcircuit. Method closed; fidelity maintained at recorded
  levels, with no significant measured speedup at matched conditions.
- **Tier-3 — TPWL terminal model, two variants.** Trajectory piecewise-linear
  fit with POD-DEIM anchors captured from the device's own double-pulse
  waveform, emitted as its own subcircuit. `v1` has better static fit on three
  of four devices but is slower than golden today; `simplis` reuses the closed
  Tier-2 channel with SIMPLIS-style PWL charge tables, holding Tier-2 static
  parity with mixed waveform fidelity (TI Vds REGRESSED vs `v1`). Neither is a
  global speed tier: the guarded 20-pulse adaptive ≥5× result is TI-only.

| | Fidelity | Speed | Robustness |
|---|---|---|---|
| Vendor SPICE | ●●●●● | ●○○○○ | ○○○○○ |
| **PieceSpice Tier-2 (CPWL)** | **●●●●○ (2.04–8.49 % static rel-RMSE; ≤ 4.13 % endpoint)** | **●●○○○ (~1× at matched conditions; not significant)** | **●●●●● (C⁰ PWL + tanh knee)** |
| **PieceSpice Tier-3 (TPWL)** | **●●●●○ (0.61–4.24 % static fit rel-RMSE)** | **●●○○○ (0.0988–0.570× measured; slower than golden)** | **●●●○○ (anchored on captured trajectory)** |
| **PieceSpice Tier-3 `simplis` (PWL charge)** | **●●●○○ (2.04–8.49 % static rel-RMSE, Tier-2 parity; DPT Vds 2.79–11.26 %; TI Vds REGRESSED vs v1)** | **●●○○○ (20-pulse adaptive solver 0.961–5.079×; ≥5× TI-only)** | **●●●○○ (guarded 20-pulse runs; four completed DPTs)** |
| PLECS / Ideal | ●○○○○ | ●●●●● | ●●●●● |

**Measured status — honest:** Three options are selectable: Tier-2 CPWL,
Tier-3 v1 TPWL and Tier-3 `simplis` (PWL charge tables). At matched 2-pulse,
forced 2 ns-step DPT conditions, neither established tier shows a significant
speedup against the vendor golden: Tier-2 solver 0.875–1.173×, Tier-3 v1
0.0988–0.570× (IRFS / TI / EPC / CPM2); Tier-3 v1 is slower on every device.
The `simplis` variant holds Tier-2 static parity within ±0.0002 pp, has mixed
waveform fidelity (TI Vds REGRESSED +3.699 pp vs v1), and its guarded
20-pulse adaptive solver / wall speedups are 0.961× / 0.941×, 5.079× / 3.070×,
1.080× / 1.059× and 1.124× / 1.066× (same vendor order). Only TI exceeds 5×,
driven by the golden's stiffness; the other three goldens already ride the
adaptive-step ceiling. Tier-3 v1 speed was audited closed on 2026-11-20;
all three options hold their recorded fidelity. See
[Measured Results](#-measured-results) for the tables and conditions of validity.

Everything is emitted as a **standard SPICE subcircuit** — no proprietary
solver, no special runtime. Drop the generated `.lib` into any SPICE flow
that supports behavioral sources and charge-defined capacitors (validated on
LTspice 26) and simulate against the golden model.

---

## ✨ How It Works

```text
                 ┌──────────────────────────────────────────────┐
 vendor .lib ──▶ │ M1/M2: EXTRACTION (LTspice testbenches)      │
                 │  7 datasheet curves per device               │
                 └──────────────────┬───────────────────────────┘
                                    ▼  config/outputs/<vendor>/
                 ┌──────────────────────────────────────────────┐
                 │ M3: APPROXIMATION ENGINES                    │
                 │  Tier-2: adaptive PWL regression → Vth, gfs* │
                 │          RDP compression, charge integration │
                 │          C¹ knee smoothing (tanh kernels)    │
                 │  Tier-3: TPWL terminal model + POD-DEIM      │
                 │          anchors from the captured DPT       │
                 └──────────────────┬───────────────────────────┘
                                    ▼
          Tier-2 → reports/models/<device>_PIECESPICE_T2.lib
          Tier-3 → reports/models/<device>_PIECESPICE_T3[_<variant>].lib
```

`*` analysis-only (notebook 02 / `metrics.compare`); the Tier-2 emitted model
is built from the extracted curve rows plus the charge integration, not from
the PWLF/RDP fits.

### The Seven Datasheet Curves

| # | Curve | Extracted From |
|---|-------|----------------|
| 1 | Output Characteristics — Id vs Vds family | static sweep |
| 2 | Transfer Characteristic — Id vs Vgs | static sweep |
| 3 | On-State Resistance — Rds(on) vs Id | static sweep |
| 4 | Capacitances — Ciss/Coss/Crss vs Vds | small-signal AC |
| 5 | Gate Charge — Vgs vs Qg | quasi-static reconstruction |
| 6 | Body Diode — Isd vs Vsd | reverse-conduction sweep |
| 7 | Threshold Voltage — JEDEC constant-current | diode-connected sweep |

---

## 🧠 Modelling Approach — Behavioral Reduced-Order Modelling

PieceSpice is a **behavioral reduced-order modelling** framework: instead
of projecting the vendor model's internal state equations (POD/Krylov-style
MOR), it fits compact *behavioral* elements to the device's own extracted
response. The reduction target is structural — hundreds of internal nodes
and stiff nonlinear devices become a ~10-element subcircuit — while the
input-output behavior at the terminals is preserved against a fidelity
target measured directly on waveforms and curves.

Each macromodel is built from the device's own extracted behavior, using
the simplest formulation that preserves the physics that matter:

**Channel conduction (Tier-2).** The drain current is a continuous
piecewise-linear function of Vgs with one segment per extracted Vgs family,
anchored at the origin, multiplied by a `tanh` saturation knee in Vds. The
breakpoints carry the physical meaning: threshold sits at the knee,
transconductance is the post-knee slope, so the reduced parameters remain
interpretable.

**Charge behavior.** Nonlinear capacitances are integrated into stored
charge Q(V) rather than used as C(v)·dv/dt — charge-conserving formulation,
so switching-loss and dv/dt predictions stay physical even under large
signal swings. The output charge Qoss is emitted as a PWL charge source;
gate capacitances are currently linear equivalents at the operating point
(a documented extension point).

**Saturation smoothing.** A `tanh` knee bounds the channel current in
saturation, keeping derivatives continuous through the transition — this is
what buys Newton robustness without sacrificing the ohmic region.

**Off-state & body diode.** The origin anchor guarantees a true blocking
state below threshold; the body diode is synthesized directly from extracted
reverse-conduction data (IS/N/RS).

**Tier-3 — TPWL terminal model, `v1` and `simplis`.** Where Tier-2 fits
static curves, Tier-3 captures a double-pulse trajectory from the golden
device, selects POD-DEIM anchors and builds a trajectory piecewise-linear
terminal model, emitted as its own SPICE subcircuit. `v1` targets terminal
waveform fidelity rather than internal-state projection: its static fit error
is lower than Tier-2 on three of four devices, but its emitted DPT runs
slower than golden. The `simplis` variant reuses the closed Tier-2 channel and
adds SIMPLIS-style PWL charge tables; its fidelity is mixed (TI Vds REGRESSED
vs `v1`) and its speed is reported per vendor on the 20-pulse adaptive bench.

The target: PLECS-class evaluation speed inside a plain SPICE subcircuit,
with error measured and bounded against the vendor golden model rather than
assumed. Measured results are below; the ≥5× solver result is reached only by
`simplis` on TI under the 20-pulse adaptive bench (3.070× wall), and no
global speedup is claimed.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- LTspice 26 (system install; `LTSPICE_EXE` overrides auto-discovery)

### Installation

```bash
git clone <repo-url> PieceSpice
cd PieceSpice
uv sync
```

### Run the Full Pipeline

```bash
# 1) Extract all 7 datasheet curves from a vendor model
uv run python -c "from pipeline.curves import run_all_curves; run_all_curves('csd16412q5a')"

# 2) Fit + synthesize the Tier-2 macromodel
uv run python -c "from pipeline.fitting import fit_pwl_model; fit_pwl_model('csd16412q5a')"
```

Or via the CLI (no console script is installed, so run the module):

```bash
uv run python -m cli.main curves csd16412q5a     # all 7 curves
uv run python -m cli.main parse SpiceModels/EPC2216_V0_PSPICE.net
```

### Select the Solver Tier

The `fit`, `validate` and `compare` commands choose the tier explicitly:

```bash
uv run python -m cli.main fit csd16412q5a --tier 2                 # Tier-2 CPWL (default)
uv run python -m cli.main fit csd16412q5a --tier 3                 # Tier-3 TPWL, default variant
uv run python -m cli.main fit csd16412q5a --tier 3 --variant alt   # named Tier-3 variant
uv run python -m cli.main validate csd16412q5a --tier 2            # golden vs Tier-2
uv run python -m cli.main validate csd16412q5a --tier 3             # golden vs Tier-3
uv run python -m cli.main compare csd16412q5a                       # golden vs Tier-2 vs Tier-3
```

Tier-2 emits `reports/models/<vendor>_PIECESPICE_T2.lib`; Tier-3 emits
`<vendor>_PIECESPICE_T3.lib` for the default variant and
`<vendor>_PIECESPICE_T3_<variant>.lib` for named variants. Multiple Tier-3
variants per vendor are the norm, and the registry lists them.

Outputs land in:

```text
config/outputs/<vendor>/        extracted curve data (YAML)
reports/models/                 synthesized Tier-2 (.T2) and Tier-3 (.T3) .lib files
reports/comparison/<vendor>/    golden-vs-model metrics and validation JSONs
figures/<vendor>/tier2|tier3/   golden-vs-tier DPT figures
```

### Adding a New Vendor Model

1. Drop the vendor `.lib/.spi` into `SpiceModels/`
2. Create `config/inputs/<vendor>.yaml` (copy an existing config; set
   `library`, `subcircuit`, `pin_order_semantic`, and bias conditions)
3. `uv run python -c "from pipeline.curves import run_all_curves; run_all_curves('<vendor>')"`

Pin order is resolved semantically from the subckt header — IR/TI (D,G,S),
EPC (G,D,S), and Wolfspeed 4-port (D,G,S,Tj) layouts are handled
automatically from the YAML pin map.

---

## 🏭 Validated Devices

| Vendor | Device | Technology | Status |
|---|---|---|---|
| Infineon / IR | IRFS7730PBF | Si MODPEX LEVEL=1 | ✅ 7/7 curves |
| Texas Instruments | CSD16412Q5A | Si PSpice LEVEL=7 | ✅ 7/7 curves |
| EPC | EPC2216 | GaN eGaN (behavioral) | ✅ 7/7 curves |
| Wolfspeed | CPM2-1200-0025B | SiC bare die (Tj port) | ✅ 7/7 curves |

All three selectable options are exercised on all four devices; measured figures follow.

---

## 📊 Measured Results

> **AES-STD100 — vendor-model fidelity caveat.** "Manufacturer datasheets
> were checked against extracted currents at the rated gate drive:
> IRFS7730PBF 246 A continuous / 984 A pulsed vs 1671 A extracted at Vgs=10 V;
> CSD16412Q5A 52 A / 91 A vs 305 A; EPC2216 3.4 A / 28 A vs 61.7 A at
> Vgs=6 V; CPM2-1200 81 A / 200 A vs 342 A at Vgs=20 V. These excesses are
> properties of the vendor subcircuits (VM-04/VM-06). Matching the vendor
> golden is not the same as matching the datasheet; fidelity claims against
> the vendor must not be read as physical device ratings."

**This caveat applies to every table in this section.** All figures below are
measured from the JSON exports in `reports/comparison/<vendor>/`; none are
estimates or imputations.

**Conditions of validity.** Every figure below is measured on the
clamped-inductive double-pulse bench configured per vendor YAML (two pulses,
forced 2 ns maximum step for the matched-condition runs), with the vendor
golden subcircuit at default settings as the sole reference. The `simplis`
rows use the same benches: emitted ROM vs golden, static via the live LTspice
DC sweep, DPT waveform with both sides on `|I(Lload)|`, and a separate
20-pulse adaptive bench (3 repeats, 176 us, medians, guard on every row;
EPC golden alone at the documented 100 ns cap). Fidelity means matching the
golden's terminal curves and waveforms, not datasheet ratings (the caveat
above); only quantities that were actually measured are reported, and the
guard returns N/A whenever a bench does not complete.

**Three-option measured status (2026-11-20).** Neither Tier-2 nor Tier-3 v1
delivers a significant speedup at matched conditions. Tier-2 solver speedups
are 0.875× / 1.173× / 1.011× / 0.945× and Tier-3 v1 solver speedups 0.570× /
0.0988× / 0.399× / 0.175× (IRFS / TI / EPC / CPM2), each measured as golden
solver time over ROM solver time. Tier-3 v1 is slower than golden on every
device. The measured Tier-2 win is the 20-pulse adaptive TI case — 4.86×
solver / 3.39× wall — with IRFS and CPM2 near 1× and EPC's golden stalling
under an adaptive step (it needs the 100 ns step cap). The 2026-11-20 audit
closed the Tier-3 v1 speed question: no fidelity-preserving speedup exists in
this emission family or through the documented solver knobs. `simplis` adds a
third selectable option with Tier-2 static parity, mixed DPT fidelity (TI Vds
REGRESSED +3.699 pp vs T3-v1), and guarded 20-pulse adaptive solver speedups
0.961× / 5.079× / 1.080× / 1.124× (wall 0.941× / 3.070× / 1.059× / 1.066×,
same vendor order). Only TI exceeds 5×, and only in solver time, because
TI's golden is globally stiff; the other goldens already ride the adaptive
step ceiling. Fidelity is maintained at the recorded levels in the tables
below, with the edge-timing limit stated after the timing table.

**Static Curve-1 relative RMSE** — Tier-2 (closed, full sweep) vs Tier-3
(`v1` function fit / emitted netlist; `simplis` emitted netlist). Tier-3 v1's
fit error is lower on 3 of 4 devices; EPC is the exception. `simplis` reuses
the Tier-2 channel unchanged, so the same live LTspice DC-sweep methodology
applies to its row.

| Device | Tier-2 rel-RMSE | T3-v1 fit rel-RMSE | T3-v1 emitted rel-RMSE | T3 simplis rel-RMSE | simplis delta vs T2 (pp) |
|---|---|---|---|---|---|
| IRFS7730PBF | 7.07 % | 4.24 % | 4.15 % | 7.0683 % | -0.00014 |
| CSD16412Q5A (TI) | 8.49 % | 1.54 % | 2.15 % | 8.4935 % | +0.00011 |
| EPC2216 | 2.04 % | 2.37 % | 7.22 % | 2.0394 % | -0.00008 |
| CPM2-1200 | 3.43 % | 0.61 % | 0.97 % | 3.4287 % | +0.00020 |

Tier-2 worst endpoint error ≤ 4.13 %. *AES-STD100 caveat above applies.*
`simplis` holds Tier-2 parity on all four devices (|delta| ≤ 0.0002 pp);
its REGRESSED flags are against the lower T3-v1 emitted bar, not against T2.

**DPT waveform completion and terminal waveform error** — golden points →
T3-v1 / `simplis` points, with relative RMSE against golden. Device current
is like-with-like (both sides use `|I(Lload)|` interpolated onto the ROM
grid); the completion guard requires ≥50 % of golden points and
`t_last ≥ 0.999×14 us`. All runs completed; `simplis` reaches 1.4e-5 s on all
four devices.

| Device | Golden points | T3-v1 points | T3-v1 Vds | T3-v1 Id | Simplis points / iterations | Simplis Vds | Simplis Id |
|---|---|---|---|---|---|---|---|
| IRFS7730PBF | 7114 | 7376 | 16.96 % | 0.51 % | 7171 / 18602 | 11.260 % | 0.537 % |
| CSD16412Q5A | 10189 | 65788 | 2.36 % | 0.41 % | 7197 / 18265 | 6.063 % | 0.047 % |
| EPC2216 | 7418 | 9434 | 4.83 % | 0.79 % | 7198 / 18449 | 5.250 % | 0.049 % |
| CPM2-1200 | 7154 | 32487 | 4.09 % | 0.29 % | 7127 / 18669 | 2.786 % | 0.261 % |

Against T3-v1, `simplis` improves CPM2 Vds (2.786 vs 4.09 %) and IRFS Vds
(11.260 vs 16.96 %), but EPC Vds REGRESSES (+0.422 pp) and TI Vds REGRESSES
(+3.699 pp: 6.063 vs 2.36 %) while TI Id improves. IRFS Id REGRESSES
(+0.0264 pp) and carries the catalogued VM-02 vendor tail. These are
golden-vs-ROM figures, not datasheet ratings. *AES-STD100 caveat above applies.*

**Solver speed vs golden at matched 2-pulse, forced 2 ns-step conditions**
(vendor golden at default settings as the sole reference):

| Device | Tier-2 solver | Tier-3 solver |
|---|---|---|
| IRFS7730PBF | 0.875× | 0.570× |
| CSD16412Q5A (TI) | 1.173× | 0.0988× |
| EPC2216 | 1.011× | 0.399× |
| CPM2-1200 | 0.945× | 0.175× |

Neither tier shows a significant speedup at matched conditions; Tier-3 v1 is
slower on every device. The 20-pulse adaptive TI case (4.86× solver / 3.39×
wall) is the measured Tier-2 exception; `simplis` measures 5.079× solver /
3.070× wall there (table below). The guard reports N/A unless both benches
complete.
*AES-STD100 caveat above applies.*

**Tier-3 `simplis` — 20-pulse adaptive solver and wall speed vs golden.**
Medians of 3 completed runs; 176 us end time, adaptive step on both sides
except EPC golden at the documented 100 ns cap, 90 s timeout. Every row is
guarded (t_last = 1.76e-4 s on both sides, iterations and trace points > 0);
speedup = golden solver (wall) time over `simplis` solver (wall) time, with
the vendor golden at default settings as the sole reference.

| Device | Golden solver s (iters) | Simplis solver s (iters) | Solver speedup | Wall speedup | Ceiling context |
|---|---|---|---|---|---|
| IRFS7730PBF | 0.321 (12,718) | 0.334 (14,299) | 0.961× | 0.941× | golden 12.96 ns inter-pulse / 80.94 ns plateau; 5× needs ≤0.0642 s, measured PWL floor 0.300 s |
| CSD16412Q5A (TI) | 1.737 (131,558) | 0.342 (12,250) | **5.079×** | 3.070× | only globally stiff golden (0.175 ns median step); 5× budget ≤0.3474 s |
| EPC2216 (golden 100 ns cap) | 0.350 (16,514) | 0.324 (12,665) | 1.080× | 1.059× | fully-adaptive golden failed (30 s timeout, no timing); capped golden at 100 ns, 5× needs ≤0.0700 s (simplis 4.63× over budget), ROM plateau 171.875 ns |
| CPM2-1200 | 0.344 (12,492) | 0.306 (10,380) | 1.124× | 1.066× | golden plateau / inter-pulse already 171.875 ns; 5× needs ≤0.0688 s |

The 171.875 ns = 176 us / 1024 step is the plateau ceiling the ROMs reach;
its origin as an LTspice setting is an inference from the equality, not a
documented setting. Only TI's golden is stiff; the other three goldens
already ride that ceiling (EPC at the capped 100 ns), which bounds a
waveform-resolving ROM to roughly 1.1–1.23× in the feasibility controls —
`simplis` measures 0.961× / 1.080× / 1.124× there. TI is the sole ≥5× solver
result and is golden-side driven; its wall speedup is 3.070×. No global
speedup is claimed. Full four-vendor tables, REGRESSED flags and the D2
methodology note are in
[§15 of the candidates doc](docs/plans/tier3_method_candidates.md#15-characterization-results-2026-11-20).
*AES-STD100 caveat above applies.*

**DPT timing and energy — golden / Tier-3 / Tier-2.** `tr`, `tf` in ns;
`e_on`, `e_off` in µJ. All Tier-3 crossings now exist on every device
(the earlier IRFS N/A cells were a second-pulse turn-on failure, fixed
2026-11-19); values are never imputed.

| Device | tr (ns) G / T3 / T2 | tf (ns) G / T3 / T2 | e_on (µJ) G / T3 / T2 | e_off (µJ) G / T3 / T2 |
|---|---|---|---|---|
| IRFS7730PBF | 25.84 / 0.836 / 2.893 | 26.54 / 1.410 / 2.529 | 157.4 / 25.10 / 71.09 | 597.0 / 535.3 / 533.4 |
| CSD16412Q5A | 1.096 / 0.106 / 1.486 | 1.662 / 0.622 / 2.246 | 1.063 / 0.938 / 0.753 | 17.97 / 17.56 / 17.73 |
| EPC2216 | 0.846 / 1.485 / 0.838 | 1.227 / 2.255 / 1.343 | 1.091 / 1.260 / 1.149 | 15.63 / 15.26 / 15.41 |
| CPM2-1200 | 4.703 / 0.850 / 4.123 | 4.139 / 2.528 / 8.789 | 5.619 / 4.440 / 6.109 | 56.70 / 55.46 / 57.04 |

*AES-STD100 caveat above applies.*

**Fidelity scope — edge timing.** The small full-waveform rel-RMSE above does
not imply accurate switching edges. `tr`/`tf` deviate measurably on several
devices: CPM2 Tier-3 `tr` 0.850 ns vs golden 4.703 ns, and IRFS Tier-3 `tr`
0.836 ns vs 25.84 ns. These ROMs hold terminal waveform shape at the recorded
levels, not switching-edge timing. The device-current comparison plotted in
notebook 03 (corrected 2026-11-20 to compare like-with-like `I(Lload)-I(Dclamp)`
on both sides; it previously plotted golden device current against ROM
inductor current, a 52-64 % artifact) is full-waveform 1.5-11.0 % across
tiers and vendors, with 70-100 % of that error inside the +/-50 ns edge
windows; outside them the deviation is 0.03-0.4 % except IRFS ~5.6 %
(the catalogued VM-02 vendor-defect tail). Tier-2 full-waveform adherence,
computed at the matched bench in the 2026-11-19 analysis (not stored in a
JSON), is Vds 12.31 / 3.81 / 4.29 / 3.65 % and Id 0.57 / 0.04 / 0.05 /
0.21 % (IRFS / TI / EPC / CPM2); the same edge-timing limit applies.

**Test suite.**

| Command | Passed | Skipped |
|---|---|---|
| `uv run pytest` | 124 | 4 |

*AES-STD100 caveat above applies to the fidelity figures these tests
protect, not to the test process itself.*

---

## 📂 Project Structure

```text
PieceSpice/
├── approximation/        # Core algorithms: pwlf_fit, rdp_compress,
│                         # charge_model, smoother, static_curves,
│                         # tpwl_deim (Tier-3 TPWL + POD-DEIM)
├── cli/                  # Typer CLI entrypoint (--tier / --variant)
├── common/               # Shared types, roles registry, plotting,
│                         # diagnostics (stderr dedup)
├── config/
│   ├── inputs/           # Per-vendor simulation conditions (YAML)
│   └── outputs/          # Extracted curve data (YAML)
├── data/                 # Captured golden switching waveforms (CSV)
├── docs/                 # Reports, plans, reference notes
├── figures/
│   └── <vendor>/
│       ├── 01_*, 02_*    # extraction / analysis figures
│       ├── tier2/        # golden-vs-Tier-2 DPT figures
│       └── tier3/        # golden-vs-Tier-3 DPT figures
├── metrics/              # Static/switching/loss/comparison metrics
├── models/               # netlist_builder (Tier-2) +
│                         # tpwl_netlist_builder (Tier-3)
├── notebooks/            # 01 extraction · 02 approximation · 03 DPT · 05 validation
├── parser/               # Subcircuit/pin parser (dialect transpiler retired)
├── pipeline/             # run_all_curves · fit_pwl_model · tier3 ·
│                         # registry (tier registry for the CLI)
├── reporting/            # Single-file HTML report generator
├── simulation/           # LTspice engine + benches (static/cap/gate-charge/
│                         # switching/dpt_trajectory)
├── SpiceModels/          # Vendor libraries (read-only inputs)
├── tests/                # pytest suite
└── scripts/              # Utility scripts (incl. run_tier3)
```

The former `tier3/` package has been folded into the paths above — there are
no re-export shims.

---

## 🧪 Testing

```bash
uv run pytest                # 124 passed, 4 skipped
uv run pytest tests/test_pwlf_fit.py -v    # single module
```

The suite covers the parser, every approximation engine (including the
Tier-3 TPWL/DEIM path), metric math, netlist emission, YAML I/O and the
LTspice log-diagnostics layer, plus live LTspice engine tests (transient,
DC sweep, timeout, temperature). The synthesized Tier-2 subcircuit is
exercised end-to-end through PySpice/ngspice at an operating point — that is
the only remaining PySpice use, and it is scheduled for rewrite in
`docs/migration_todo.md` T6.2a together with four skipped tests. Tier-3 has
its own test module, `tests/test_tier3.py`.

---

## 📓 Notebooks

| Notebook | Purpose |
|---|---|
| [`01_datasheet_curves_extraction`](notebooks/01_datasheet_curves_extraction.ipynb) | Extract the 7 datasheet curves from any configured vendor |
| [`02_piecewise_approximation`](notebooks/02_piecewise_approximation.ipynb) | Fit → compress → charge-integrate → smooth → synthesize → validate |
| [`04a_simple_switching_subset`](notebooks/04a_simple_switching_subset.ipynb) | Golden-vs-Tier-2 resistive-load switching comparison (M4 subset) |
| [`03_double_pulse_transients`](notebooks/03_double_pulse_transients.ipynb) | Clamped-inductive DPT: golden vs Tier-2 vs Tier-3 waveforms, timing and energy; writes `figures/<vendor>/tier2/` and `tier3/` |
| [`04_buck_converter_benchmark`](notebooks/04_buck_converter_benchmark.ipynb) | *placeholder — buck bench not implemented; no speedup numbers* |
| [`05_comprehensive_validation_report`](notebooks/05_comprehensive_validation_report.ipynb) | Rewritten — Tier-2 **and** Tier-3 validation, measured numbers only, per-vendor table |

---

## 🗺️ Roadmap

- [x] **M1** — Environment, Parser & SPICE Dialect Normalization
- [x] **M2** — Automated Datasheet Curves Simulation (7 curves × 4 vendors)
- [x] **M3** — Piecewise Approximation & Charge Integration Engines
      + Tier-2 synthesis + golden validation
      **Method closed.** Final mean full-sweep rel-RMSE: IRFS 7.07 %,
      TI 8.49 %, EPC 2.04 %, CPM2 3.43 %; worst endpoint error ≤4.13 %.
      The remaining TI/IRFS deviations are limits of this model form —
      further parameter ablations were measured and did not improve them.
- [x] **Tier-3 — TPWL terminal model** *(two variants selectable alongside Tier-2)*
      Static Curve-1 fit rel-RMSE beats Tier-2 on 3 of 4 devices
      (IRFS 4.24 vs 7.07 %, TI 1.54 vs 8.49 %, CPM2 0.61 vs 3.43 %;
      EPC 2.37 vs 2.04 % is the exception). All four DPTs completed.
      **No significant speedup** — matched-condition solver speed
      0.0988–0.570×, slower than golden on all four devices; the
      2026-11-20 speed audit closed this line of work.
      (2026-11-19 emission rework: smooth two-exponential Qoss,
      conditioned grids, soft-plus clamps, IRFS second-pulse fix);
      the guard reports N/A if a bench does not complete.
      The `simplis` variant (Level-2 PWL charge tables) holds Tier-2 static
      parity and reaches 5.079× solver / 3.070× wall on TI in the 20-pulse
      adaptive bench; the other three sit at the measured golden ceiling
      (~1×), with TI Vds REGRESSED vs v1.
- [x] **M4** — Switching Dynamics & Double Pulse Testbench
      *(metrics, resistive-load subset and inductive DPT)*
- [ ] **M5** — Switched-Mode Converter Benchmark & Speedup Evaluation
      *(DPT-topology runtime benchmark only; no converter benchmark exists.
      Neither established tier shows a significant matched-condition speedup;
      the 20-pulse adaptive TI win is 4.86× solver / 3.39× wall for Tier-2
      and 5.079× / 3.070× for `simplis` (TI-only), and Tier-3 v1 speed was
      audited closed on 2026-11-20 — see Measured Results.)*
- [x] **M6** — Automated HTML Reporting & CLI Polish
      *(the report renders only metrics it is given)*

Backend: **LTspice 26 + spicelib**, out-of-process; the PySpice/ngspice path
is retired. Known limitations and vendor-model quirks are tracked in
[`vendor_model_defects.md`](vendor_model_defects.md) (VM-01…VM-12,
HOST-01/HOST-02).

---

## 🤝 Contributing Notes

- Every code change must be recorded in
  [`development-log.md`](development-log.md) (newest first, IST timestamps).
- Vendor libraries under `SpiceModels/` are **read-only inputs** — never edit.
- Plotting follows house rules: legends EAST outside the axes, 300 dpi.
- See [`piecespice_plan.md`](piecespice_plan.md) for the architecture and
  algorithm formulations.

---

<div align="center">
<sub>Built for power-electronics engineers who need both accuracy and speed.</sub>
</div>
