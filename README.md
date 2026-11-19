<div align="center">

# ⚡ PieceSpice

**High-Fidelity Piecewise SPICE Modeling Framework**

*PLECS-class speed. SPICE-class resolution. Standard SPICE netlists.*

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
distills vendor MOSFET subcircuits into one of **two selectable solver tiers**:

- **Tier-2 — CPWL macromodel.** Continuous piecewise-linear channel with a
  `tanh` saturation knee plus charge-defined capacitors, emitted as a small
  standard-SPICE subcircuit. Method closed; robust and fast.
- **Tier-3 — TPWL terminal model.** Trajectory piecewise-linear fit with
  POD-DEIM anchors captured from the device's own double-pulse waveform,
  emitted as its own subcircuit. Better static fit, but slower today.

| | Fidelity | Speed | Robustness |
|---|---|---|---|
| Vendor SPICE | ●●●●● | ●○○○○ | ○○○○○ |
| **PieceSpice Tier-2 (CPWL)** | **●●●●○ (2.04–8.49 % static rel-RMSE; ≤ 4.13 % endpoint)** | **●●○○○ (no speedup claimed)** | **●●●●● (C⁰ PWL + tanh knee)** |
| **PieceSpice Tier-3 (TPWL)** | **●●●●○ (0.61–4.24 % static fit rel-RMSE)** | **●●○○○ (0.10–0.57× measured)** | **●●●○○ (anchored on captured trajectory)** |
| PLECS / Ideal | ●○○○○ | ●●●●● | ●●●●● |

Everything is emitted as a **standard SPICE subcircuit** — no proprietary
solver, no special runtime. Drop the generated `.lib` into any SPICE flow
that supports behavioral sources and charge-defined capacitors (validated on
LTspice 26) and simulate against the golden model.

**Measured status — honest:** Tier-3 does **not** currently deliver a speedup.
Measured solver speedups are 0.570× (IRFS), 0.0988× (TI), 0.399× (EPC) and
0.175× (CPM2) — that is **0.10–0.57×** (rounded), slower than the golden in
every case.
The earlier aspirational speedup claim is superseded by these measurements.
The pipeline guard reports speedup as **N/A whenever either DPT bench fails
to complete**, so an incomplete run is never counted as a win. See
[Measured Results](#-measured-results) for the full tables.

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

**Tier-3 — TPWL terminal model.** Where Tier-2 fits static curves, Tier-3
captures a double-pulse trajectory from the golden device, selects POD-DEIM
anchors and builds a trajectory piecewise-linear terminal model, emitted as
its own SPICE subcircuit. It targets terminal waveform fidelity rather than
internal-state projection. Its static fit error is lower than Tier-2 on
three of four devices, but its emitted DPT runs slower than golden — select
Tier-3 for fidelity studies, not for speed today.

The target: PLECS-class evaluation speed inside a plain SPICE subcircuit,
with error measured and bounded against the vendor golden model rather than
assumed. Measured results are below; the speed target is **not** met by
either tier in the current DPT runs.

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

Both tiers are exercised on all four devices; measured figures follow.

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

**Static Curve-1 relative RMSE** — Tier-2 (closed, full sweep) vs Tier-3
(function fit / emitted netlist). Tier-3's fit error is lower on 3 of 4
devices; EPC is the exception.

| Device | Tier-2 rel-RMSE | Tier-3 fit rel-RMSE | Tier-3 emitted rel-RMSE |
|---|---|---|---|
| IRFS7730PBF | 7.07 % | 4.24 % | 4.15 % |
| CSD16412Q5A (TI) | 8.49 % | 1.54 % | 2.15 % |
| EPC2216 | 2.04 % | 2.37 % | 7.22 % |
| CPM2-1200 | 3.43 % | 0.61 % | 0.98 % |

Tier-2 worst endpoint error ≤ 4.13 %. *AES-STD100 caveat above applies.*

**DPT waveform completion and terminal waveform error** — golden points →
Tier-3 points, with Tier-3 relative RMSE against golden. All four Tier-3 DPT
runs completed.

| Device | Golden points | Tier-3 points | Vds rel-RMSE | Id rel-RMSE |
|---|---|---|---|---|
| IRFS7730PBF | 7114 | 7210 | 16.96 % | 0.51 % |
| CSD16412Q5A | 10189 | 66240 | 2.36 % | 0.41 % |
| EPC2216 | 7418 | 9301 | 4.83 % | 0.79 % |
| CPM2-1200 | 7154 | 31436 | 4.09 % | 0.29 % |

*AES-STD100 caveat above applies.*

**Solver speed (Tier-3 vs golden):** 0.570× / 0.0988× / 0.399× / 0.175×
(IRFS / TI / EPC / CPM2) — **no speedup**. The pipeline guard reports N/A
unless both benches complete. *AES-STD100 caveat above applies.*

**DPT timing and energy — golden / Tier-3 / Tier-2.** `tr`, `tf` in ns;
`e_on`, `e_off` in µJ. All Tier-3 crossings now exist on every device
(the earlier IRFS N/A cells were a second-pulse turn-on failure, fixed
2026-11-19); values are never imputed.

| Device | tr (ns) G / T3 / T2 | tf (ns) G / T3 / T2 | e_on (µJ) G / T3 / T2 | e_off (µJ) G / T3 / T2 |
|---|---|---|---|---|
| IRFS7730PBF | 25.84 / 0.836 / 2.893 | 26.54 / 1.410 / 2.529 | 157.4 / 25.10 / 71.09 | 597.0 / 535.3 / 533.4 |
| CSD16412Q5A | 1.096 / 0.106 / 1.486 | 1.662 / 0.622 / 2.246 | 1.063 / 0.938 / 0.753 | 17.97 / 17.56 / 17.73 |
| EPC2216 | 0.846 / 1.485 / 0.838 | 1.227 / 2.255 / 1.343 | 1.091 / 1.260 / 1.149 | 15.63 / 15.26 / 15.41 |
| CPM2-1200 | 4.703 / 0.851 / 4.123 | 4.139 / 2.528 / 8.789 | 5.619 / 4.440 / 6.109 | 56.70 / 55.46 / 57.04 |

*AES-STD100 caveat above applies.*

**Test suite.**

| Command | Passed | Skipped |
|---|---|---|
| `uv run pytest` | 118 | 4 |

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
uv run pytest                # 118 passed, 4 skipped
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
- [x] **Tier-3 — TPWL terminal model** *(selectable alongside Tier-2)*
      Static Curve-1 fit rel-RMSE beats Tier-2 on 3 of 4 devices
      (IRFS 4.24 vs 7.07 %, TI 1.54 vs 8.49 %, CPM2 0.61 vs 3.43 %;
      EPC 2.37 vs 2.04 % is the exception). All four DPTs completed.
      **No speedup claimed** — measured solver speed 0.10–0.57×
      (2026-11-19 emission rework: smooth two-exponential Qoss,
      conditioned grids, soft-plus clamps, IRFS second-pulse fix);
      the guard reports N/A if a bench does not complete.
- [x] **M4** — Switching Dynamics & Double Pulse Testbench
      *(metrics, resistive-load subset and inductive DPT)*
- [ ] **M5** — Switched-Mode Converter Benchmark & Speedup Evaluation
      *(DPT-topology runtime benchmark only; no converter benchmark exists.
      Tier-3 adds no measured speedup — see Measured Results.)*
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
