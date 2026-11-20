<div align="center">

# ⚡ PieceSpice

**High-Fidelity Piecewise SPICE Modeling Framework**

*PLECS-class speed (target). SPICE-class resolution. Standard SPICE netlists.*

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LTspice](https://img.shields.io/badge/LTspice-26-orange)](https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html)
[![spicelib](https://img.shields.io/badge/spicelib-powered-yellow)](https://pypi.org/project/spicelib/)
[![Tests](https://img.shields.io/badge/tests-531%20passed%20%C2%B7%204%20skipped%20%C2%B7%201%20warning-brightgreen?logo=pytest&logoColor=white)](#testing)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()
[![uv](https://img.shields.io/badge/managed%20with-uv-dead00?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

[![Infineon](https://img.shields.io/badge/supported-Infineon%20%2F%20IR-0A8276?labelColor=1a1a1a)](#validated-devices) [![TI](https://img.shields.io/badge/supported-Texas%20Instruments-CC0000?labelColor=1a1a1a)](#validated-devices) [![EPC](https://img.shields.io/badge/supported-EPC%20GaN-006FBA?labelColor=1a1a1a)](#validated-devices) [![Wolfspeed](https://img.shields.io/badge/supported-Wolfspeed%20SiC-58238C?labelColor=1a1a1a)](#validated-devices)

*Vendor `.lib` in → validated Tier-2 CPWL, Tier-3 TPWL / unified one-ROM (urom) or Tier-4 ideal-switch + LUT out*

</div>

---

## 📖 Overview

**PieceSpice** bridges the simulation gap between full physics SPICE models
(high fidelity, hours of compute, convergence failures) and system-level
ideal-switch tools (instant, but no ringing/losses). It automatically
distills vendor MOSFET subcircuits into four selectable configurations —
Tier-2 CPWL, Tier-3 `v1`, Tier-3 `simplis` and Tier-4 ideal-switch + LUT —
plus the registered Tier-3 `urom` variant (`--tier 3 --variant urom`), which
is not a fifth tier.

- **Tier-2 — CPWL macromodel.** Continuous piecewise-linear channel with a
  `tanh` saturation knee plus charge-defined capacitors, emitted as a small
  standard-SPICE subcircuit. Method closed; fidelity maintained at recorded
  levels, with no significant measured speedup at matched conditions.
- **Tier-3 — TPWL terminal model, `v1` (default) and `simplis`.** Trajectory
  piecewise-linear fit with POD-DEIM anchors captured from the device's own
  double-pulse waveform. `v1` has better static fit on three of four devices
  but is slower than golden today; its Qg-emission experiment was measured,
  closed and reverted, and the shipped emitter is the restored four-line
  closed form. `simplis` reuses the closed Tier-2 channel with SIMPLIS-style
  PWL voltage-axis charge tables, holding Tier-2 static parity with mixed
  waveform fidelity on the device-current metric (TI Vds higher than `v1`).
  Neither is a global speed tier: the guarded 20-pulse adaptive ≥5× result is
  TI-only.
- **Tier-3 `urom` variant — `--tier 3 --variant urom`.** The canonical basis
  is C0 (the closed Tier-2 channel/charge emitter, `channel_basis=tier2`),
  emitted as ASCII `<vendor>_PIECSPICE_UROM.lib`. In the final equal-sandbox
  validation it is **accepted on TI and EPC** and **rejected on IRFS and
  CPM2** on measured common-grid DPT waveform bars. ANN is a basis selected
  by `--basis ann`, not a variant; its explicit keyed records reject on all
  four. A fixed-topology tuning API exists; the final run reports
  `tuning.verdict="not_run"` (tuning off).
- **Tier-4 — ideal-switch fast tier (dual-tier addition).** Standard-SPICE
  ideal switch plus a measured conduction/switching loss LUT kept as comments
  for offline post-processing. The DPT waveform gate is retired by design:
  no dv/dt, Miller droop, ringing or reverse-recovery fidelity is claimed.
  Solver stabilization uses a measured linear Curve-4 Coss and a measured
  Vgs=0 off-state Roff; energy rows are source self-consistency and
  efficiency is N/A without an explicit converter output power. Speed is
  measured on the existing inductive bench only; no converter bench exists.

| | Fidelity | Speed | Robustness |
|---|---|---|---|
| Vendor SPICE | ●●●●● | ●○○○○ | ○○○○○ |
| **PieceSpice Tier-2 (CPWL)** | **●●●●○ (2.04–8.49 % static rel-RMSE; ≤ 4.13 % endpoint)** | **●●○○○ (~1× at matched conditions; not significant)** | **●●●●● (C⁰ PWL + tanh knee)** |
| **PieceSpice Tier-3 `v1` (TPWL)** | **●●●●○ (0.61–4.24 % static fit rel-RMSE)** | **●●○○○ (0.0953–0.5661× measured; slower than golden)** | **●●●○○ (anchored on captured trajectory)** |
| **PieceSpice Tier-3 `simplis` (PWL charge)** | **●●●○○ (2.04–8.49 % static rel-RMSE, Tier-2 parity; DPT Vds 3.032–8.704 %; TI Vds higher than `v1`)** | **●●○○○ (20-pulse adaptive solver 1.0561–5.3765×, CPM2 N/A; ≥5× TI-only)** | **●●●○○ (guarded 20-pulse runs; four completed 2-pulse DPTs)** |
| **PieceSpice Tier-3 `urom` (canonical C0)** | **●●●●○ (static rel-RMSE at Tier-2 bars 2.0394–8.4935 %; G3 rejects IRFS/CPM2)** | **●●○○○ (TI 5.3172× solver / 3.1286× wall; EPC 1.0057× / 1.0237×; scoped, not global)** | **●●●○○ (one declared trap/default method, per-vendor verdicts; keyed ANN rejects 4/4)** |
| **PieceSpice Tier-4 (ideal switch + LUT)** | **●●○○○ (DPT waveform gate retired by design)** | **●●●○○ (guarded inductive-bench solver 1.1896–6.3540×; EPC golden 100 ns cap; CPM2 N/A)** | **●●●○○ (DPT 4/4 complete, speed 3/4 measured)** |
| PLECS / Ideal | ●○○○○ | ●●●●● | ●●●●● |

**Measured status — honest.** Four configurations are selectable: Tier-2 CPWL,
Tier-3 `v1` TPWL, Tier-3 `simplis` (voltage-axis PWL charge tables) and
Tier-4 (ideal switch plus loss LUT); the Tier-3 `urom` variant is a fifth
selectable model, not a fifth tier. At matched 2-pulse, forced 2 ns-step DPT
conditions, neither established waveform-resolving configuration shows a
significant speedup against the vendor golden: Tier-2 solver 0.875–1.173×,
Tier-3 `v1` 0.0953–0.5661× (IRFS / TI / EPC / CPM2); `v1` is slower on every
device. The `simplis` variant holds Tier-2 static parity within ±0.0002 pp
and has mixed waveform fidelity on the same device-current metric; its
guarded 20-pulse adaptive solver / wall speedups are 1.0561× / 1.0392×,
5.3765× / 3.1496×, 1.1108× / 1.0272× and N/A / N/A (IRFS / TI / EPC / CPM2;
CPM2 golden aborts). Only TI exceeds 5×, driven by the golden's stiffness;
the other completed goldens already ride the adaptive-step ceiling. Tier-3
`v1` speed was audited closed on 2026-11-20; Tier-4 is a separate, hardened
fast tier whose inductive-bench speed and honest scope are tabulated below.
All four configurations hold their recorded scope (Tier-4's waveform gate is
retired by design, not passed). See [Measured Results](#-measured-results)
for the tables and conditions of validity.

The canonical `--tier 3 --variant urom` record (`channel_basis=tier2`, C0)
closes that split only partially: accepted on TI/EPC and rejected on
IRFS/CPM2 on measured common-grid DPT bars, so one emitted ROM holds fidelity
and scoped speed on two of four devices. The keyed `--basis ann` records and
libs are separate, non-default artifacts and reject on all four. The goal is
partially achieved, not universal, and no global or portable speed claim is
made.

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
          Tier-3 --variant urom → reports/models/<device>_PIECSPICE_UROM.lib (C0)
          Tier-3 --variant urom --basis ann → ..._PIECSPICE_UROM_ann_trap-default.lib
          Tier-4 → reports/models/<device>_PIECSPICE_T4.lib
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
slower than golden. The Qg-emission experiment was measured and closed, then
reverted: the shipped `v1` emitter is the restored four-line closed form
(`.subckt`, one `C_QOSS`, one `G_CH`, `.ends`) with no emitted gate-charge
element. The `simplis` variant reuses the closed Tier-2 channel and adds
SIMPLIS-style PWL charge tables fitted on the **voltage axis**; both golden
and ROM now use device current `I(Lload)-I(Dclamp)`. Its Id values changed
because the old inductor-current-only metric hid clamp current — a metric
correction, not a newly fabricated model improvement or regression. Its
fidelity is mixed on the like-for-like metric and its speed is reported per
vendor on the 20-pulse adaptive bench (`method=trap/default`).

The target: PLECS-class evaluation speed inside a plain SPICE subcircuit,
with error measured and bounded against the vendor golden model rather than
assumed. Measured results are below; the ≥5× solver result is reached only by
`simplis` on TI under the 20-pulse adaptive bench (3.1496× wall), and no
global or portable speedup is claimed.

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

`fit` and `validate` choose tier and variant explicitly; `compare` renders
all recorded rows and takes `--basis` for the UROM row (`--tier 4` is
variant-free; `--variant` is ignored there). Registered Tier-3 variants are
`v1` (default), `simplis` and `urom`; `--variant alt` is invalid. ANN is a
basis, not a variant.

```bash
uv run python -m cli.main fit csd16412q5a --tier 2                 # Tier-2 CPWL (default)
uv run python -m cli.main fit csd16412q5a --tier 3                 # Tier-3 v1, default variant
uv run python -m cli.main fit csd16412q5a --tier 3 --variant simplis # PWL charge variant
uv run python -m cli.main fit csd16412q5a --tier 3 --variant urom  # canonical C0/tier2 basis
uv run python -m cli.main validate csd16412q5a --tier 3 --variant urom --basis ann  # keyed ANN candidate
uv run python -m cli.main validate csd16412q5a --tier 3 --variant urom  # G0-G6 + scoped speed
uv run python -m cli.main validate csd16412q5a --tier 2            # golden vs Tier-2
uv run python -m cli.main validate csd16412q5a --tier 3             # golden vs Tier-3 v1
uv run python -m cli.main fit csd16412q5a --tier 4                  # Tier-4 build + record
uv run python -m cli.main validate csd16412q5a --tier 4             # Tier-4 energy/speed summary
uv run python -m cli.main compare csd16412q5a                       # golden vs T2 vs T3 + T4 rows
```

Tier-2 emits `reports/models/<vendor>_PIECESPICE_T2.lib`; Tier-3 emits
`<vendor>_PIECESPICE_T3.lib` for the default `v1` variant and
`<vendor>_PIECESPICE_T3_<variant>.lib` for `simplis`. Tier-4 emits
`reports/models/<vendor>_PIECSPICE_T4.lib`; its summary and `compare` rows
print DPT-source energy self-consistency and guarded inductive-bench speed,
with `N/A` wherever a bench did not complete — never DPT waveform fidelity.

The `urom` variant emits the canonical ASCII
`reports/models/<vendor>_PIECSPICE_UROM.lib` (`channel_basis=tier2`, C0) with
the four-vendor G0-G6 record in
`reports/comparison/<vendor>/unified_validation.json`. Selecting `--basis ann`
does **not** overwrite it: the keyed, non-default candidate lives at
`reports/models/<vendor>_PIECSPICE_UROM_ann_trap-default.lib` with record
`reports/comparison/<vendor>/unified_ann_trap-default.json`. Each record's
`emitted_lib_sha256` hashes its own on-disk lib. These final generated libs
are un-ignored/tracked; experimental or candidate libs (for example a Qg
experiment) and vendor `SpiceModels/**` remain ignored.

Outputs land in:

```text
config/outputs/<vendor>/        extracted curve data (YAML)
reports/models/                 final .T2 / .T3 / .T3_simplis / .T4 / UROM (C0 + keyed ANN) .lib files (tracked)
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

All four selectable configurations plus the Tier-3 `urom` variant are exercised on all four devices; measured figures follow.

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
golden subcircuit at default settings as the sole reference. Golden and ROM
share the simulator, bench, `.tran`/`.options`, tolerances, `trap/default`
method and 25 °C; an aborted run is a failure, never a speedup. The `simplis`
rows use the same benches: emitted ROM vs golden, static via the live LTspice
41-point channel-only DC sweep (capacitors open at DC) with estimator
`mean(per_curve_peak_rel_rmse_pct)`, and DPT waveform on **device current**
`I(Lload)-I(Dclamp)` on both sides, interpolated onto the golden grid; the
separate 20-pulse adaptive bench uses 3 repeats, 176 µs, medians and a guard
on every row (EPC golden alone at the documented 100 ns cap; CPM2 golden
adaptive aborts, so its ratio is N/A). Fidelity means matching the golden's
terminal curves and waveforms, not datasheet ratings (the caveat above);
only quantities that were actually measured are reported, N/A is returned
whenever a bench does not complete, and no global or portable speed claim is
made. Speed rows are `trap/default` and are not comparable to Gear rows.

**Measured status (2026-11-20 regeneration).** Neither Tier-2 nor Tier-3 `v1`
delivers a significant speedup at matched conditions. Tier-2 solver speedups
are 0.875× / 1.173× / 1.011× / 0.945× and Tier-3 `v1` solver speedups 0.5661× /
0.0953× / 0.3907× / 0.1659× (IRFS / TI / EPC / CPM2; `runtime.method`
`default/trap`), each measured as golden solver time over ROM solver time.
Tier-3 `v1` is slower than golden on every device. The measured Tier-2 win is
the 20-pulse adaptive TI case — 4.86× solver / 3.39× wall — with IRFS and
CPM2 near 1× and EPC's golden stalling under an adaptive step (it needs the
100 ns step cap). The Tier-3 `v1` speed question was audited closed; its Qg
emission experiment was likewise measured, closed and reverted, leaving the
restored four-line emitter. `simplis` adds a second Tier-3 configuration with
Tier-2 static parity and mixed DPT fidelity on the regenerated device-current
metric (TI Vds 5.140 vs `v1` 2.925 %, +2.216 pp), with guarded 20-pulse
adaptive solver speedups 1.0561× / 5.3765× / 1.1108× / N/A (wall
1.0392× / 3.1496× / 1.0272× / N/A; IRFS / TI / EPC / CPM2, whose golden
adaptive run aborts). Only TI exceeds 5×, and only in solver time, because
TI's golden is globally stiff; the other completed goldens already ride the
adaptive step ceiling. Fidelity is maintained at the recorded levels in the
tables below, with the edge-timing limit stated after the timing table.

**Static Curve-1 relative RMSE** — Tier-2 (closed, full sweep) vs Tier-3
(`v1` function fit / emitted netlist; `simplis` emitted netlist). Tier-3 `v1`'s
fit error is lower on 3 of 4 devices; EPC is the exception. `simplis` reuses
the Tier-2 channel unchanged, so the same 41-point live LTspice DC-sweep
methodology applies to its row (capacitors open at DC).

| Device | Tier-2 rel-RMSE | T3-v1 fit rel-RMSE | T3-v1 emitted rel-RMSE | T3 simplis rel-RMSE | simplis delta vs T2 (pp) |
|---|---|---|---|---|---|
| IRFS7730PBF | 7.07 % | 4.24 % | 4.15 % | 7.0683 % | -0.00014 |
| CSD16412Q5A (TI) | 8.49 % | 1.54 % | 2.15 % | 8.4935 % | +0.00010 |
| EPC2216 | 2.04 % | 2.37 % | 7.22 % | 2.0394 % | -0.00009 |
| CPM2-1200 | 3.43 % | 0.61 % | 0.97 % | 3.4287 % | +0.00019 |

Tier-2 worst endpoint error ≤ 4.13 %. *AES-STD100 caveat above applies.*
`simplis` holds Tier-2 parity on all four devices (|delta| ≤ 0.0002 pp);
its REGRESSED flags are against the lower T3-`v1` emitted bar, not against T2.

**DPT waveform completion and terminal waveform error** — golden points →
T3-`v1` / `simplis` points, with relative RMSE against golden. Both sides now
use device current `I(Lload)-I(Dclamp)` on the golden grid; the completion
guard requires ≥50 % of golden points and `t_last ≥ 0.999×14 µs`. All runs
completed; both arms reach 1.4e-5 s on all four devices. The Id columns are
**not comparable to the earlier inductor-current-only values**: the old
metric hid clamp current, so the jump is a metric correction, not a newly
fabricated model improvement or regression.

| Device | Golden points | T3-v1 points / iterations | T3-v1 Vds / Id | Simplis points / iterations | Simplis Vds / Id |
|---|---|---|---|---|---|
| IRFS7730PBF | 7114 | 7376 / 20,004 | 12.69 % / 11.00 % | 7094 / 16,834 | 8.704 % / 8.517 % |
| CSD16412Q5A | 10189 | 64,665 / 380,226 | 2.92 % / 4.90 % | 7169 / 18,149 | 5.140 % / 4.216 % |
| EPC2216 | 7418 | 9440 / 33,785 | 2.85 % / 1.47 % | 7177 / 18,332 | 3.032 % / 1.774 % |
| CPM2-1200 | 7154 | 32,250 / 205,154 | 4.38 % / 5.19 % | 7151 / 18,308 | 3.851 % / 4.615 % |

Against `v1` on the same device-current metric, `simplis` Vds is lower on IRFS
(8.704 vs 12.69 %) and CPM2 (3.851 vs 4.38 %), but higher on TI (+2.216 pp:
5.140 vs 2.92 %) and EPC (+0.185 pp: 3.032 vs 2.85 %); Id is lower on IRFS,
TI and CPM2 and higher on EPC (+0.305 pp). TI Vds is the only sizable
like-for-like REGRESSION; the rest are within the corrected metric's measured
spread. IRFS carries the catalogued VM-02 vendor tail. These are
golden-vs-ROM figures, not datasheet ratings. *AES-STD100 caveat above
applies.*

**Solver speed vs golden at matched 2-pulse, forced 2 ns-step conditions**
(vendor golden at default settings as the sole reference):

| Device | Tier-2 solver (`speedup_benchmark.json#short_forced`) | Tier-3 `v1` solver (`tier3_validation.json#runtime`, `default/trap`) |
|---|---|---|
| IRFS7730PBF | 0.875× | 0.5661× |
| CSD16412Q5A (TI) | 1.173× | 0.0953× |
| EPC2216 | 1.011× | 0.3907× |
| CPM2-1200 | 0.945× | 0.1659× |

Neither tier shows a significant speedup at matched conditions; Tier-3 `v1` is
slower on every device. The 20-pulse adaptive TI case (4.86× solver / 3.39×
wall) is the measured Tier-2 exception; `simplis` measures 5.3765× solver /
3.1496× wall there (table below; `method=trap/default`). The guard reports N/A
unless both benches complete. *AES-STD100 caveat above applies.*

**Tier-3 `simplis` — 20-pulse adaptive solver and wall speed vs golden
(`method=trap/default`; source: `tier3_validation_simplis.json#speed_adaptive_20`).**
Medians of 3 completed runs; 176 µs end time, adaptive step on both sides
except EPC golden at the documented 100 ns cap, 90 s timeout. Every completed
row is guarded (`t_last = 1.76e-4 s`, iterations and trace points > 0);
CPM2's golden arm aborted (no raw output, rc=1), so that row is N/A;
speedup = golden solver (wall) time over `simplis` solver (wall) time, with
the vendor golden at default settings as the sole reference. No global or
portable speedup is claimed.

| Device | Golden solver s (iters) | Simplis solver s (iters) | Solver speedup | Wall speedup | Ceiling context |
|---|---|---|---|---|---|
| IRFS7730PBF | 0.320 (12,718) | 0.303 (8,440) | 1.0561× | 1.0392× | 5× would need solver ≤0.064 s; measured solver floor 0.303 s |
| CSD16412Q5A (TI) | 1.742 (131,558) | 0.324 (12,073) | **5.3765×** | 3.1496× | only globally stiff golden; 5× budget ≤0.348 s, both arms fully adaptive |
| EPC2216 (golden 100 ns cap) | 0.351 (16,514) | 0.316 (12,123) | 1.1108× | 1.0272× | fully-adaptive golden timed out out-of-band; capped golden at 100 ns, 5× would need ≤0.070 s (ROM ~4.5× over budget), ROM plateau 171.875 ns |
| CPM2-1200 | N/A (golden aborted, rc=1) | 0.306 (10,592) | N/A | N/A | golden adaptive produced no raw output; guard returns N/A, no 5× budget |

The 171.875 ns = 176 µs / 1024 step is the plateau ceiling the ROMs reach;
its origin as an LTspice setting is an inference from the equality, not a
documented setting. Only TI's golden is stiff; the completed non-TI goldens
already ride that ceiling (EPC at the capped 100 ns), which bounds a
waveform-resolving ROM to near-unity in the feasibility controls: `simplis`
measures 1.0561× (IRFS) and 1.1108× (EPC) there; CPM2's golden adaptive run
aborts, so its ratio is N/A. TI is the sole ≥5× solver result and is
golden-side driven; its wall speedup is 3.1496×. No global speedup is claimed.

**Provenance — SUPERSEDED baselines.** The former README set (solver / wall:
IRFS 0.961× / 0.941×, TI 5.079× / 3.070×, EPC 1.080× / 1.059×, CPM2 1.124× /
1.066×) and `reports/comparison/simplis_char_speed_summary.json` are
superseded by the final per-vendor records
`reports/comparison/<vendor>/tier3_validation_simplis.json`
(`speed_adaptive_20`). Also discarded: IRFS 1.074× and TI 4.858× (Tier-2
benchmark rows), TI 5.299× / 3.069× (earlier docs), EPC error/stall and CPM2
0.977× (benchmark rows). See §5.1 of
[the deep report](docs/deep_spice_fidelity_speed_research.md). Full
four-vendor tables, REGRESSED flags and the D2 methodology note are in
[§15 of the candidates doc](docs/plans/tier3_method_candidates.md#15-characterization-results-2026-11-20);
quote the per-vendor JSON for every measured number. *AES-STD100 caveat above
applies.*

**Tier-4 — hardened fast converter tier (dual-tier addition; measured
2026-11-20 regeneration).** The emitted ASCII subcircuit is a standard-SPICE
ideal switch (`S_SW` + `.MODEL SW_T4`) with measured Curve-3 `Ron` and
Curve-7 `Vth`; the conduction/switching loss LUT is **comment-only metadata
for offline post-processing** (PLECS P-TC role), not an active electrical
loss element or thermal RC network. The DPT waveform gate is **retired by
design**: no dv/dt, Miller droop, ringing or reverse-recovery fidelity is
claimed. One measured linear Curve-4 Coss at `cap_ref_Vds` plus a measured
Vgs=0 DC-sweep `Roff` are emitted as solver stabilizers only — they bound the
drain node so the solver reaches `t_end`, but they slow drain dV/dt and add
L-C overshoot; they do not restore waveform fidelity. No tolerance,
step-size or simulator change is claimed, and no PLECS/SIMPLIS step-count
speed is attributed to the emitted switch. Speed rows carry
`speedup.method="LTspice default/trap (both arms)"` on all four vendors.

**Conditions of validity (Tier-4).** The vendor golden at LTspice defaults is
the sole reference. Energy nodes come from each vendor's recorded golden
2-pulse DPT crossing (2 ns forced, 14 µs). Speed is the existing 20-pulse,
176 µs adaptive clamped-inductive bench (median of 3 repeats, golden vs
Tier-4, `method=LTspice default/trap (both arms)`), with EPC's golden alone
at its documented 100 ns cap; N/A is reported whenever a bench does not
complete. Completion guards: emitted DPT on **4/4** devices
(`t_last = 1.4e-5 s`, positive iterations, ≥50 % of golden trace points);
speed on **3/4** (no error, positive iterations/points, full-schedule
`.tran` stop, 3 repeats). CPM2 speed is contract-excluded under VM-10: the
vendor notes at the record root disclose that older text reported a golden
adaptive failure (`no raw output`, `rc=1`) while the legacy
`speedup_benchmark.json#cases.long_adaptive` arm holds a completed golden
(`golden_completed=true`); both readings are disclosed and neither yields a
Tier-4 ratio. The bench discards per-run time vectors, so speed `t_last` is
N/A and `end_time_s` is schedule metadata, not a measured trace endpoint.
*AES-STD100 caveat above applies: golden-vs-model figures, not datasheet
ratings.*

| Device (Vbus / Id) | Solver speedup | Wall speedup | Golden → Tier-4 iterations | Golden → Tier-4 points | Case |
|---|---:|---:|---|---|---|
| IRFS7730PBF (38 V / 100 A) | 1.1896× | 1.0952× | 12,718 → 6,165 | 3,432 → 2,247 | long_adaptive |
| CSD16412Q5A / TI (12.5 V / 10 A) | **6.3540×** | 3.3328× | 131,558 → 6,201 | 44,050 → 2,296 | long_adaptive |
| EPC2216 (11 V / 10 A) | 1.2474× | 1.0608× | 16,514 → 6,296 | 4,047 → 2,597 | long_100ns (golden 100 ns cap) |
| CPM2-1200 (20 V / 20 A) | N/A | N/A | N/A | N/A | VM-10 contract-excluded golden speed; no ratio derived |

**Tier-4 energy/efficiency scope.** E_on/E_off relative error is 0 % **by
construction**: the LUT value and the compared "golden" value share the same
`dpt_metrics.json#golden_metrics` crossing. The flags live in
`tier4_validation.json#energy_comparison.e_on_uj` and
`#energy_comparison.e_off_uj`: `source_shared: true`,
`independent_energy_gate: false`, `kind: source_self_consistency`. Those rows
are source self-consistency, not an independent converter validation — no
converter bench was built. Efficiency is N/A on all four devices
(`output_power_required`): E_on/E_off/r_on are evaluated offline, and
Vbus×Iload is DC bus input power, not converter output power. The loss LUT's
finite-node provenance is vendor-specific: IRFS and TI have one `constant_node`
(`1×1` grid), EPC has three finite nodes with `grid_interpolation`, and CPM2
has one `single_node` on a `2×2` grid; no cell has all four corners, so
off-node queries are NaN and no bilinear interpolation is exercised. TI's
6.3540× comes from the stiff golden (131,558 golden vs 6,201 Tier-4
iterations) and is not presented as a portable speed class. No unmeasured
superlative is claimed.

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

**Tier-3 `urom` — final equal-sandbox result (`--tier 3 --variant urom`,
regenerated records).** The top-level canonical record in
`reports/comparison/<vendor>/unified_validation.json` selects the C0 basis
(`channel_basis="tier2"`, `emit="tier2"`) and emits ASCII
`reports/models/<vendor>_PIECSPICE_UROM.lib`; its header prints
`basis=tier2 method=trap/default tuning=off`. ANN is a separate basis chosen
by `--basis ann`, not a variant: the keyed non-default records
`reports/comparison/<vendor>/unified_ann_trap-default.json` and libs
`reports/models/<vendor>_PIECSPICE_UROM_ann_trap-default.lib` carry
`channel_basis="ann"`, `ann_fit` measured (`isru`, 10 neurons) and reject on
all four vendors. Each record's `emitted_lib_sha256` hash-matches its own
on-disk lib; the canonical and keyed paths never clobber each other. A
fixed-topology tuning API exists, but the final run reports
`tuning.verdict="not_run"` (`tuning.mode="off"`).

**Conditions of validity (unified one-ROM).** Golden and ROM share the same
LTspice 26 core, bench, `.tran`/`.options`, tolerances, declared
`trap/default` method and 25 °C; only the DUT differs
(`deck_diff_only_dut=true`, `tolerance_overrides=[]`, libraries inlined on
both arms). Fidelity gates use the recorded forced-2 ns / 14 µs DPT against
the closed Tier-2 control bars; G3 now compares golden and ROM on an equal
7000-point common grid (`window_equal=true`, `common_grid_guard=true`), so
the historical unequal-window artifact is gone. Speed is the 20-pulse,
176 µs adaptive clamped-inductive bench, median of 3 interleaved repeats,
with `t_last = 1.76e-4 s` on both arms, positive iterations/points and a
≥50 % point-completion guard; EPC uses a 100 ns cap on both arms. An aborted
run is a failure, never a speedup. Figures below are golden-vs-model
measurements at these conditions, not datasheet ratings, and are bound to
this one declared method and emitted basis (`global_claim=false`,
`min_default_gear=null`, no portability screen).

| Device (Vbus / Id) | Verdict | Static rel-RMSE | Solver / wall speedup | Iterations golden → urom | Points golden → urom | G3 DPT Vds / Id vs bar |
|---|---|---|---|---|---|---|
| CSD16412Q5A / TI (12.5 V / 10 A) | **accept** (G0-G6 all pass) | 8.4935 % | **5.3172× / 3.1286×** | 131,558 → 10,195 | 44,050 → 2,868 | 1.984 / 0.045 vs 5.583 / 0.048 |
| EPC2216 (11 V / 10 A; 100 ns cap both arms) | **accept** (G0-G6 all pass) | 2.0394 % | 1.0057× / 1.0237× | 16,514 → 12,206 | 4,047 → 3,550 | 0.704 / 0.046 vs 4.894 / 0.049 |
| IRFS7730PBF (38 V / 100 A) | reject (G3 only) | 7.0683 % | 1.0444× / 1.0262× | 12,718 → 9,274 | 3,432 → 2,556 | **9.999 / 0.575 vs 8.529 / 0.532 (both breach)** |
| CPM2-1200 (20 V / 20 A) | reject (G3 only) | 3.4287 % | 1.1299× / 1.0293× | 12,492 → 9,214 | 3,092 → 2,523 | **2.893 / 0.215 vs 2.675 / 0.261 (Vds breach only)** |

Static rel-RMSE returns to the recorded Tier-2 bars
(7.0683 / 8.4935 / 2.0394 / 3.4287 %, IRFS / TI / EPC / CPM2; estimator
`mean(per_curve_peak_rel_rmse_pct)`) and the worst endpoint stays ≤4.13 % on
every C0 row. All C0 rows pass G0, G1, G2, G4, G5 and G6; only IRFS and CPM2
fail G3, on the measured common-grid bars above. The keyed ANN rows fail G1
and G4 on all four and fail G3 on all four (EPC also fails G2); no ANN row is
accepted and EPC's earlier 2-vs-7 compare-window artifact is absent from the
regenerated records. `fidelity_scope` reports `waveform_gate_retired=false`
and `energy_only_claim=false` — this row still uses the waveform gate,
unlike Tier-4. No global or portable claim is made: `urom` is one declared
method, verdicts are per vendor and `global_claim=false`. One emitted ROM
therefore holds fidelity and scoped speed only on TI/EPC — the project goal
is partially achieved, not universal across all four vendors. TI's gain is
a golden-stiffness artifact of this scoped bench; EPC's solver gain is
marginal (≈0.6 %), and IRFS/CPM2 remain rejects.

**Test suite (final executed run).** Final result: **531 passed, 4 skipped,
1 warning** from `uv run pytest`; no collected-count figure is claimed.

| Command | Passed | Skipped | Warnings |
|---|---|---|---|
| `uv run pytest` | 531 | 4 | 1 |

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
├── models/               # netlist_builder (Tier-2), tpwl_netlist_builder
│                         # (Tier-3), tier4_ideal_switch_builder (Tier-4)
├── notebooks/            # 01 extraction · 02 approximation · 03 DPT · 05 validation
├── parser/               # Subcircuit/pin parser (dialect transpiler retired)
├── pipeline/             # run_all_curves · fit_pwl_model · tier3 · tier4 ·
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

Tier-4 is variant-free: `models/tier4_ideal_switch_builder.py` emits
`reports/models/<vendor>_PIECESPICE_T4.lib` and the guarded record lands in
`reports/comparison/<vendor>/tier4_validation.json`. Its loss LUT is comment
metadata only; no converter bench or T4 figures directory exists.

---

## 🧪 Testing

```bash
uv run pytest                # final: 531 passed, 4 skipped, 1 warning
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
- [x] **Tier-3 — TPWL terminal model** *(`v1` default and `simplis` variants
      alongside Tier-2)*
      Static Curve-1 fit rel-RMSE beats Tier-2 on 3 of 4 devices
      (IRFS 4.24 vs 7.07 %, TI 1.54 vs 8.49 %, CPM2 0.61 vs 3.43 %;
      EPC 2.37 vs 2.04 % is the exception). All four DPTs completed on the
      common golden grid with device current `I(Lload)-I(Dclamp)`.
      **No significant speedup** — matched-condition solver speed
      0.0953–0.5661×, slower than golden on all four devices; the
      2026-11-20 speed audit closed this line of work. The Qg-emission
      experiment was measured, closed and reverted to the four-line emitter.
      The `simplis` variant (Level-2 PWL charge tables fitted on voltage
      axes) holds Tier-2 static parity and reaches 5.3765× solver / 3.1496×
      wall on TI in the 20-pulse adaptive bench; IRFS (1.0561×) and EPC
      (1.1108×) sit at the measured golden ceiling, CPM2's golden adaptive
      run aborts (N/A). Its Id values changed because the old
      inductor-current-only metric hid clamp current — a metric correction,
      not a new model improvement or regression; TI Vds remains higher than
      `v1` on the like-for-like metric.
- [x] **Tier-4 — ideal-switch + loss-LUT fast tier** *(hardened dual-tier addition)*
      DPT completion **4/4**; guarded inductive-bench speed **3/4**
      (`method=LTspice default/trap (both arms)`; solver/wall:
      IRFS 1.1896× / 1.0952×, TI 6.3540× / 3.3328×,
      EPC 1.2474× / 1.0608× at the golden 100 ns cap; CPM2 N/A on VM-10
      contract exclusion, with the legacy completed-golden conflict
      disclosed). Waveform gate retired by design; energy rows are
      DPT-source self-consistency, efficiency N/A without converter output
      power, and no converter bench was built.
- [x] **Tier-3 `urom` — `--tier 3 --variant urom`** *(final equal-sandbox
      result, regenerated records)*: canonical top-level C0/tier2 basis,
      ASCII `<vendor>_PIECSPICE_UROM.lib` with hash-matched
      `unified_validation.json`. `accept` on TI (5.3172× solver /
      3.1286× wall) and EPC (1.0057× / 1.0237×, 100 ns cap both arms);
      `reject` on IRFS and CPM2 on common-grid G3 DPT bars only. Static
      rel-RMSE returns to the Tier-2 bars; the keyed `--basis ann` records
      and libs are separate, non-default and reject on all four; tuning is
      `not_run`. Per-vendor scope with `global_claim=false`: one emitted ROM
      holds fidelity and scoped speed on TI/EPC only — partially achieved,
      not universal.
- [x] **M4** — Switching Dynamics & Double Pulse Testbench
      *(metrics, resistive-load subset and inductive DPT)*
- [ ] **M5** — Switched-Mode Converter Benchmark & Speedup Evaluation
      *(partial, not complete: Tier-4 adds measured inductive-bench speed —
      IRFS 1.1896×, TI 6.3540×, EPC 1.2474× solver; CPM2 N/A — but **no
      converter bench exists**. Tier-4 energy rows are DPT-source
      self-consistency and efficiency is N/A without an explicit converter
      output power. The established tiers' matched-condition and adaptive
      numbers are in Measured Results; Tier-3 `v1` speed was audited closed
      2026-11-20.)*
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
