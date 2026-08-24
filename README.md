<div align="center">

# ⚡ PieceSpice

**High-Fidelity Piecewise SPICE Modeling Framework**

*PLECS-class speed. SPICE-class resolution. Standard SPICE netlists.*

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![ngspice](https://img.shields.io/badge/ngspice-34--shared-orange)](https://ngspice.sourceforge.io/)
[![PySpice](https://img.shields.io/badge/PySpice-powered-yellow)](https://pyspice.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-97%20passing-brightgreen?logo=pytest&logoColor=white)](#testing)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()
[![uv](https://img.shields.io/badge/managed%20with-uv-dead00?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

[![Infineon](https://img.shields.io/badge/supported-Infineon%20%2F%20IR-0A8276?labelColor=1a1a1a)](#validated-devices) [![TI](https://img.shields.io/badge/supported-Texas%20Instruments-CC0000?labelColor=1a1a1a)](#validated-devices) [![EPC](https://img.shields.io/badge/supported-EPC%20GaN-006FBA?labelColor=1a1a1a)](#validated-devices) [![Wolfspeed](https://img.shields.io/badge/supported-Wolfspeed%20SiC-58238C?labelColor=1a1a1a)](#validated-devices)

*Vendor `.lib` in → validated Tier-2 CPWL macromodel out*

</div>

---

## 📖 Overview

**PieceSpice** bridges the simulation gap between full physics SPICE models
(high fidelity, hours of compute, convergence failures) and system-level
ideal-switch tools (instant, but no ringing/losses). It automatically
distills vendor MOSFET subcircuits into **Tier-2 Continuous Piecewise-Linear
(CPWL) macromodels** that:

| | Fidelity | Speed | Robustness |
|---|---|---|---|
| Vendor SPICE | ●●●●● | ●○○○○ | ○○○○○ |
| **PieceSpice Tier-2** | **●●●●○ (<5% err)** | **●●●●○ (10–100×)** | **●●●●● (C¹ Jacobians)** |
| PLECS / Ideal | ●○○○○ | ●●●●● | ●●●●● |

Everything is emitted as a **standard SPICE subcircuit** — no proprietary
solver, no special runtime. Drop the generated `.lib` into any ngspice-based
flow and simulate 10–100× faster with <5% error against the golden model.

---

## ✨ How It Works

```text
                 ┌──────────────────────────────────────────────┐
 vendor .lib ──▶ │ M1/M2: EXTRACTION (ngspice testbenches)      │
                 │  7 datasheet curves per device               │
                 └──────────────────┬───────────────────────────┘
                                    ▼  config/outputs/<vendor>/
                 ┌──────────────────────────────────────────────┐
                 │ M3: APPROXIMATION ENGINES                    │
                 │  • Adaptive PWL regression  → Vth, gfs       │
                 │  • RDP compression          → lean curves    │
                 │  • Charge integration       → Q(V) tables    │
                 │  • C¹ knee smoothing        → tanh kernels   │
                 └──────────────────┬───────────────────────────┘
                                    ▼  PWLModelSpec
                 ┌──────────────────────────────────────────────┐
                 │ SYNTHESIS: netlist_builder                   │
                 │  tanh-knee G_CH + C_QOSS + D_BODY            │
                 └──────────────────┬───────────────────────────┘
                                    ▼
                     reports/models/<device>_PIECESPICE_T2.lib
```

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

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- ngspice shared library (bundled via PySpice)

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

Or via the CLI:

```bash
uv run piece-spice curves csd16412q5a     # all 7 curves
uv run piece-spice parse SpiceModels/foo.lib
```

Outputs land in:

```text
config/outputs/<vendor>/        extracted curve data (YAML)
reports/models/                 synthesized Tier-2 .lib files
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

**Golden-vs-Tier-2 validation** (TI CSD16412Q5A, Vds = 10 V):

| Vgs | Vendor Golden | Tier-2 CPWL | Error |
|---|---|---|---|
| 4.5 V | 76.5 A | 76.5 A | 0.07 % |
| 5.0 V | 98.9 A | 98.4 A | 0.51 % |
| 6.0 V | 144.7 A | 144.0 A | 0.52 % |
| 8.0 V | 236.4 A | 235.1 A | 0.53 % |
| 10.0 V | 325.5 A | 323.8 A | 0.51 % |

---

## 📂 Project Structure

```text
PieceSpice/
├── approximation/        # Core algorithms: pwlf_fit, rdp_compress,
│                         # charge_model, smoother
├── cli/                  # Typer CLI entrypoint
├── common/               # Shared types, roles registry, plotting,
│                         # diagnostics (stderr dedup), plotting utils
├── config/
│   ├── inputs/           # Per-vendor simulation conditions (YAML)
│   └── outputs/          # Extracted curve data (YAML)
├── docs/                 # Reports, plans, reference notes
├── metrics/              # Static/switching/loss/comparison metrics
├── models/               # Tier-2 netlist synthesis
├── notebooks/            # 01 extraction · 02 approximation & validation
├── parser/               # PSpice/LTspice dialect transpiler + netlist parser
├── pipeline/             # run_all_curves() · fit_pwl_model(vendor)
├── reporting/            # Self-contained HTML report generator
├── simulation/           # Engine + testbenches (static/cap/gate-charge)
├── SpiceModels/          # Vendor libraries (read-only inputs)
├── tests/                # pytest suite
└── scripts/              # Utility scripts
```

---

## 🧪 Testing

```bash
uv run pytest                # 97 tests
uv run pytest tests/test_pwlf_fit.py -v    # single module
```

The suite covers the parser/transpiler, every approximation engine, metric
math, netlist emission (including end-to-end ngspice simulation of the
synthesized Tier-2 model), YAML I/O contracts, and the stderr-diagnostics
deduplicator.

---

## 📓 Notebooks

| Notebook | Purpose |
|---|---|
| [`01_datasheet_curves_extraction`](notebooks/01_datasheet_curves_extraction.ipynb) | Extract the 7 datasheet curves from any configured vendor |
| [`02_piecewise_approximation`](notebooks/02_piecewise_approximation.ipynb) | Fit → compress → charge-integrate → smooth → synthesize → validate |
| `03_double_pulse_transients` | *(planned — Milestone 4)* |
| `04_buck_converter_benchmark` | *(planned — Milestone 5)* |
| `05_comprehensive_validation_report` | *(planned — Milestone 6)* |

---

## 🗺️ Roadmap

- [x] **M1** — Environment, Parser & SPICE Dialect Normalization
- [x] **M2** — Automated Datasheet Curves Simulation (7 curves × 4 vendors)
- [x] **M3** — Piecewise Approximation & Charge Integration Engines
      + Tier-2 synthesis + golden validation
- [ ] **M4** — Switching Dynamics & Double Pulse Testbench (metrics done)
- [ ] **M5** — Switched-Mode Converter Benchmark & Speedup Evaluation
- [x] **M6** — Automated HTML Reporting & CLI Polish

Known limitations and vendor-model quirks are tracked in
[`vendor_model_defects.md`](vendor_model_defects.md) (VM-01…VM-08).

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
