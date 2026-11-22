<div align="center">

# ⚡ PieceSpice

**High-Fidelity Piecewise SPICE Modeling Framework**

*PLECS-class speed (target). SPICE-class resolution. Standard SPICE netlists.*

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LTspice](https://img.shields.io/badge/LTspice-26-orange)](https://www.analog.com/en/resources/design-tools-and-calculators/ltspice-simulator.html)
[![spicelib](https://img.shields.io/badge/spicelib-powered-yellow)](https://pypi.org/project/spicelib/)
[![Tests](https://img.shields.io/badge/tests-830%20passed%20%C2%B7%200%20skipped%20%C2%B7%202%20warnings-brightgreen?logo=pytest&logoColor=white)](#testing)
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
  validation the overall verdict is **`reject` on all four vendors**: the
  independent G4 timing/energy gate rejects every row, so no row is accepted
  even though fidelity passes on three. Fidelity accepts TI / EPC / CPM2 and
  rejects IRFS on G3 device-current Id (**9.0127 % > 5 %**); speed accepts
  TI (5.3021× solver / 3.0086× wall) and CPM2 (1.1023× / 1.0583×) but rejects
  IRFS (1.0073× / 0.9860×) and EPC (0.9863× / 1.0169×). ANN is a basis
  selected by `--basis ann`, not a variant; its explicit keyed records reject
  on all four. A fixed-topology tuning API exists; the final run reports
  `tuning.verdict="not_run"` (tuning off). No global or portable speed claim
  is made.
- **Tier-4 — ideal-switch fast tier (dual-tier addition).** Standard-SPICE
  ideal switch plus a measured conduction/switching loss LUT kept as comments
  for offline post-processing. The DPT waveform gate is retired by design:
  no dv/dt, Miller droop, ringing or reverse-recovery fidelity is claimed.
  Solver stabilization uses a measured linear Curve-4 Coss and a measured
  Vgs=0 off-state Roff; DPT/LUT energy rows are source self-consistency and
  the recorded efficiency stays N/A until converter output power is measured.
  Speed stays scoped to the inductive bench. A measured buck bench and
  notebook 04 now exist: EPC2216 Tier-2 completes at Vout 5.3053 V /
  96.91 % versus golden 5.3077 V / 97.55 %. Missing or aborted arms remain
  N/A (Tier-3 `v1` timed out at the 180 s bench guard); no every-tier result
  or portable speedup is claimed.

| | Fidelity | Speed | Robustness |
|---|---|---|---|
| Vendor SPICE | ●●●●● | ●○○○○ | ○○○○○ |
| **PieceSpice Tier-2 (CPWL)** | **●●●●○ (2.04–8.49 % static rel-RMSE; ≤ 4.13 % endpoint)** | **●●○○○ (~1× at matched conditions; not significant)** | **●●●●● (C⁰ PWL + tanh knee)** |
| **PieceSpice Tier-3 `v1` (TPWL)** | **●●●●○ (emitted static rel-RMSE 0.97–7.22 %; fit 0.61–2.37 %)** | **●●○○○ (forced solver 0.0946–0.5827× at matched conditions; slower than golden)** | **●●●○○ (anchored on captured trajectory)** |
| **PieceSpice Tier-3 `simplis` (PWL charge)** | **●●●○○ (static 2.0394–8.4935 %; DPT Vds 2.942–8.704 %, Id 1.764–8.517 %)** | **●●○○○ (20-pulse adaptive solver 1.0705–5.3592×, CPM2 1.1004×; ≥5× TI-only)** | **●●●○○ (guarded 20-pulse runs; four completed 2-pulse DPTs)** |
| **PieceSpice Tier-3 `urom` (canonical C0)** | **●●●○○ (static 2.0394–8.4935 %; fidelity accept 3/4, IRFS G3 Id 9.0127 %)** | **●●○○○ (solver/wall: IRFS 1.0073/0.9860×, TI 5.3021/3.0086×, EPC 0.9863/1.0169×, CPM2 1.1023/1.0583×; speed accept 2/4)** | **●●○○○ (one declared trap/default method; overall reject 4/4 on G4 timing/energy; keyed ANN rejects 4/4)** |
| **PieceSpice Tier-4 (ideal switch + LUT)** | **●●○○○ (DPT waveform gate retired by design)** | **●●●○○ (guarded inductive-bench solver 1.2212–6.4730×; EPC golden 100 ns cap; CPM2 1.3618×)** | **●●●●○ (DPT 4/4 complete, speed 4/4 measured)** |
| PLECS / Ideal | ●○○○○ | ●●●●● | ●●●●● |

**Measured status — honest.** Four configurations are selectable: Tier-2 CPWL,
Tier-3 `v1` TPWL, Tier-3 `simplis` (voltage-axis PWL charge tables) and
Tier-4 (ideal switch plus loss LUT); the Tier-3 `urom` variant is a fifth
selectable model, not a fifth tier. At matched 2-pulse, forced 2 ns-step DPT
conditions, neither established waveform-resolving configuration shows a
significant speedup against the vendor golden: Tier-2 solver 0.875–1.173×,
Tier-3 `v1` 0.0946–0.5827× (IRFS / TI / EPC / CPM2); `v1` is slower on every
device. The `simplis` variant holds Tier-2 static parity at the quoted
precision and has mixed waveform fidelity on the same device-current metric;
its guarded 20-pulse adaptive solver / wall speedups are 1.07054× / 1.04492×,
5.14545× / 2.92373×, 1.14729× / 1.06856× and 1.10037× / 1.00592× (IRFS / TI / EPC / CPM2;
CPM2 Tj port now driven). Only TI exceeds 5×, driven by the golden's stiffness;
the other completed goldens already ride the adaptive-step ceiling. Tier-3
`v1` speed was audited closed on 2026-11-20; Tier-4 is a separate, hardened
fast tier whose inductive-bench speed and honest scope are tabulated below.
All four configurations hold their recorded scope (Tier-4's waveform gate is
retired by design, not passed). See [Measured Results](#-measured-results)
for the tables and conditions of validity.

The canonical `--tier 3 --variant urom` record (`channel_basis=tier2`, C0)
now returns **overall `reject` on all four vendors**: its fidelity verdict is
`accept` for TI / EPC / CPM2 and `reject` for IRFS (G3 device-current Id
9.0127 % > 5 %), its timing/energy verdict is `reject` on all four under the
independent G4 golden gate, and its speed verdict is `accept` only for TI and
CPM2. There is no global claim. The keyed `--basis ann` records and libs are
separate, non-default artifacts and reject on all four. TI's speed gain is
driven by its stiff golden; EPC is slower than golden on both clocks.

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
assumed. Measured results are below; ≥5× solver results occur only on TI
(`simplis` 5.14545× solver / 2.92373× wall; canonical C0 5.30208× /
3.00864× wall) on the 20-pulse adaptive bench, and no global or portable
speedup is claimed.

### PLECS speed semantics — why imported SPICE is not an ideal switch

"PLECS-class speed" here is a workflow target, not a claim that an imported
netlist inherits ideal-switch speed. In PLECS 5.0, standard electrical circuits
are solved with the fast piecewise-linear **state-space** formulation, while a
circuit containing a Netlist / Netlist File Reference component is solved as a
physical SPICE model with **Modified Nodal Analysis (MNA)**, producing a
differential-algebraic equation (DAE) system integrated by third-order implicit
Runge-Kutta (SDIRK/ESDIRK). That portion is not an ideal switch. Workflow speed
is preserved by **isolation**: a Configurable Subsystem toggles between ideal
PLECS switches and the detailed SPICE netlist, so only the refined device
circuit pays the MNA/DAE cost while controls and the rest of the system stay in
state-space (Plexim, *PLECS 5.0 Documentation* — "How PLECS Spice Works" /
"Using PLECS Spice"; Plexim flyer, *Bridging the Gap Between PLECS and SPICE*,
2026-01-27). Tier-4 is the ideal-switch/system tier; Tier-2/Tier-3/UROM are the
detailed-device tiers. Converting a detailed SPICE model into an ideal switch
is not how that speed is achieved.

### Hybrid C0+ANN framework (Phase VI research direction — implemented, non-default, not a shipped final artifact)

The researched direction is a physics-first **hybrid residual**, not a pure ANN
replacement: keep the closed C0 channel/charge tables as the physical base and
add only a small, signed, zero-anchored ANN residual **current** source
(`I = I_C0 + ΔI(Vgs,Vds)`) fitted to golden-minus-C0 on Curve-1 IV families plus
recorded DPT samples only; the emitted hybrid
netlist keeps the closed C0 charge model unchanged — there is no `ΔQ(V)` term,
no ANN charge source and no replacement of the C0 charge tables. It stays an
ASCII SPICE subcircuit with no new nodes and preserves Vds=0 zero current and
sub-threshold cutoff; the preserved C0 charge stays monotone and zero-anchored
(Q(0)=0). The hybrid fit trains no ANN charge term and uses no gate-charge
data. The **full Curve-4 C(V) sweep** is persisted only for simplis charge
fitting and as a data-richness diagnostic, not hybrid training. This is
source-backed: Yang et al. (*IEEE TPEL* 2022, DOI 10.1109/TPEL.2021.3101713)
report ~30 % less computation with ANN IV/CV plus physical guards; Tung,
Salahuddin & Hu (*SISPAD* 2024, DOI 10.1109/SISPAD62626.2024.10733248) report
~40× with separate IV/QV networks; Gong et al. (*IEEE TCAD* 2026) constrain the
ANN with monotonicity/symmetry; Ward & Dutton (1978) motivate the
charge-conserving base. These are the cited authors' results under their own
conditions, **not** PieceSpice measurements. The final acceptance policy is
declared in `config/acceptance.yaml` (schema `piecespice.acceptance/1.0`):
G3 scores the device-current identity `I(Lload)-I(Dclamp)` on the equal
common grid against the recorded per-vendor Vds baselines and a declared
**Id 5 %** application requirement, and G4 scores timing/energy against the
independent golden DPT rather than the old Tier-2 self-comparison. The former
Id bars (0.60 / 0.048 / 0.049 / 0.261 %) were measured on the old
inductor-current metric and are retired; the Id change is a metric
correction, not a relaxed model result. Under this policy the canonical C0
basis has fidelity `accept` for TI / EPC / CPM2 and `reject` for IRFS
(9.0127 % Id), but **overall `reject` 4/4 because G4 timing/energy rejects
every row** (see [Measured Results](#-measured-results)). Hybrid timing and
energy are scored against those same independent golden metrics. The hybrid
residual emitter and its fail-closed probe are implemented, but the hybrid
basis is **non-default and is not an accepted or shipped final UROM
artifact**; no provisional hybrid fidelity or timing number is presented as
a final result.

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
completes with its 4th `Tj` port driven, 1.10037× / 1.00592×). Fidelity means matching the golden's
terminal curves and waveforms, not datasheet ratings (the caveat above);
only quantities that were actually measured are reported, N/A is returned
whenever a bench does not complete, and no global or portable speed claim is
made. Speed rows are `trap/default` and are not comparable to Gear rows.
The gate values are declared in `config/acceptance.yaml` (`schema
piecespice.acceptance/1.0`; recorded baselines vs project decisions are
labelled per value), and the canonical UROM records store an
`artifact_manifest` whose input and emitted-lib SHA-256 hashes are verified
(`artifact_manifest_verified.ok=true`). Static "worst endpoint" is the
relative error at the configured `Vds_sweep_max`, not an interior best case.

**Measured status (final regeneration).** Neither Tier-2 nor Tier-3 `v1`
delivers a significant speedup at matched conditions. Tier-2 solver speedups
are 0.875× / 1.173× / 1.011× / 0.945× and Tier-3 `v1` solver speedups 0.5827× /
0.0946× / 0.4202× / 0.1788× (IRFS / TI / EPC / CPM2; `runtime.method`
`default/trap`), each measured as golden solver time over ROM solver time.
Tier-3 `v1` is slower than golden on every device. The measured Tier-2 win is
the 20-pulse adaptive TI case — 4.86× solver / 3.39× wall — with IRFS and
CPM2 near 1× and EPC's golden stalling under an adaptive step (it needs the
100 ns step cap). The Tier-3 `v1` speed question was audited closed; its Qg
emission experiment was likewise measured, closed and reverted, leaving the
restored four-line emitter. `simplis` adds a second Tier-3 configuration with
Tier-2 static parity and mixed DPT fidelity on the regenerated device-current
metric (TI Vds 5.116 % vs `v1` 2.925 %, +2.191 pp), with guarded 20-pulse
adaptive solver speedups 1.07054× / 5.14545× / 1.14729× / 1.10037× (wall
1.04492× / 2.92373× / 1.06856× / 1.00592×; IRFS / TI / EPC / CPM2, with
CPM2's golden completed with its 4th `Tj` port driven). Only TI exceeds 5×, and only in solver time, because
TI's golden is globally stiff; the other completed goldens already ride the
adaptive step ceiling. Fidelity is maintained at the recorded levels in the
tables below, with the edge-timing limit stated after the timing table.

**Static Curve-1 relative RMSE** — Tier-2 (closed, full sweep) vs Tier-3
(`v1` function fit / emitted netlist; `simplis` emitted netlist). Tier-3 `v1`'s
fit error is lower on 3 of 4 devices; EPC is the exception. `simplis` reuses
the Tier-2 channel unchanged, so the same 41-point live LTspice DC-sweep
methodology applies to its row (capacitors open at DC).

| Device | Tier-2 rel-RMSE | T3-v1 fit rel-RMSE | T3-v1 emitted rel-RMSE | T3 simplis rel-RMSE |
|---|---|---|---|---|
| IRFS7730PBF | 7.07 % | 2.16 % | 1.70 % | 7.068260 % |
| CSD16412Q5A (TI) | 8.49 % | 1.54 % | 2.15 % | 8.493502 % |
| EPC2216 | 2.04 % | 2.37 % | 7.22 % | 2.039408 % |
| CPM2-1200 | 3.43 % | 0.61 % | 0.97 % | 3.428692 % |

Tier-2 worst endpoint error ≤ 4.13 %; the measured endpoint worst errors at
the configured `Vds_sweep_max` are 3.9601 / 4.1248 / 0.5572 / 1.4568 %
(IRFS / TI / EPC / CPM2). *AES-STD100 caveat above applies.*
`simplis` reproduces the recorded Tier-2 static values at the quoted
precision; its REGRESSED flags are against the lower T3-`v1` emitted bar, not
against T2.

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
| IRFS7730PBF | 7114 | 7453 / 20,840 | 12.70 % / 10.96 % | 7094 / 16,834 | 8.704415 % / 8.516744 % |
| CSD16412Q5A | 10189 | 64,665 / 380,226 | 2.92 % / 4.90 % | 7164 / 18,232 | 5.116158 % / 4.212055 % |
| EPC2216 | 7418 | 9440 / 33,785 | 2.85 % / 1.47 % | 7190 / 18,405 | 2.948953 % / 1.763691 % |
| CPM2-1200 | 7154 | 32,250 / 205,154 | 4.38 % / 5.19 % | 7133 / 18,129 | 3.852518 % / 4.558134 % |

Against `v1` on the same device-current metric, `simplis` Vds is lower on IRFS
(8.704 vs 12.70 %) and CPM2 (3.853 vs 4.38 %), but higher on TI (+2.191 pp:
5.116 vs 2.92 %) and EPC (+0.103 pp: 2.949 vs 2.85 %); Id is lower on IRFS,
TI and CPM2 and higher on EPC (+0.294 pp). TI Vds is the only sizable
like-for-like REGRESSION; the rest are within the corrected metric's measured
spread. IRFS carries the catalogued VM-02 vendor tail. These are
golden-vs-ROM figures, not datasheet ratings. *AES-STD100 caveat above
applies.*

**Solver speed vs golden at matched 2-pulse, forced 2 ns-step conditions**
(vendor golden at default settings as the sole reference):

| Device | Tier-2 solver (`speedup_benchmark.json#short_forced`) | Tier-3 `v1` solver (`tier3_validation.json#runtime`, `default/trap`) |
|---|---|---|
| IRFS7730PBF | 0.875× | 0.5827× |
| CSD16412Q5A (TI) | 1.173× | 0.0946× |
| EPC2216 | 1.011× | 0.4202× |
| CPM2-1200 | 0.945× | 0.1788× |

Neither tier shows a significant speedup at matched conditions; Tier-3 `v1` is
slower on every device. The 20-pulse adaptive TI case (4.86× solver / 3.39×
wall) is the measured Tier-2 exception; `simplis` measures 5.14545× solver /
2.92373× wall there (table below; `method=trap/default`). The guard reports N/A
unless both benches complete. *AES-STD100 caveat above applies.*

**Tier-3 `simplis` — 20-pulse adaptive solver and wall speed vs golden
(`method=trap/default`; source: `tier3_validation_simplis.json#speed_adaptive_20`).**
Medians of 3 completed runs; 176 µs end time, adaptive step on both sides
except EPC golden at the documented 100 ns cap, 90 s timeout. Every completed
row is guarded (`t_last = 1.76e-4 s`, iterations and trace points > 0);
CPM2's 4th `Tj` port must be driven; with header-resolved pins the golden
completes and the measured row is 1.10037× / 1.00592×;
speedup = golden solver (wall) time over `simplis` solver (wall) time, with
the vendor golden at default settings as the sole reference. No global or
portable speedup is claimed.

| Device | Golden solver s (iters) | Simplis solver s (iters) | Solver speedup | Wall speedup | Ceiling context |
|---|---|---|---|---|---|
| IRFS7730PBF | 0.258 (12,718) | 0.241 (8,440) | 1.07054× | 1.04492× | 5× would need solver ≤0.052 s; measured solver floor 0.241 s |
| CSD16412Q5A (TI) | 1.415 (131,558) | 0.275 (12,228) | **5.14545×** | 2.92373× | only globally stiff golden; 5× budget ≤0.283 s, both arms fully adaptive |
| EPC2216 (golden 100 ns cap) | 0.296 (16,514) | 0.258 (12,221) | 1.14729× | 1.06856× | fully-adaptive golden timed out out-of-band; capped golden at 100 ns, 5× would need ≤0.059 s; ROM plateau 171.875 ns |
| CPM2-1200 | 0.283 (12,492) | 0.259 (10,621) | **1.10037×** | 1.00592× | Tj port driven; both arms fully adaptive |

The 171.875 ns = 176 µs / 1024 step is the plateau ceiling the ROMs reach;
its origin as an LTspice setting is an inference from the equality, not a
documented setting. Only TI's golden is stiff; the completed non-TI goldens
already ride that ceiling (EPC at the capped 100 ns), which bounds a
waveform-resolving ROM to near-unity in the feasibility controls: `simplis`
measures 1.07054× (IRFS) and 1.14729× (EPC) there; CPM2, with its Tj port
driven, measures 1.10037×. TI is the sole ≥5× solver result and is
golden-side driven; its wall speedup is 2.92373×. No global speedup is claimed.

**`simplis` charge provenance and candidate selection.** The charge tables are
fitted on the **voltage axis** from the persisted full Curve-4 arrays
(`curve4_full_arrays`, `full_arrays_present=true` on all four). The record
carries the Cgd axis provenance: a genuine multi-bias sweep supplies Cgd on
`Vdg=Vds−Vgs` and is preferred when present, but the current four-vendor
records hold only the single Vgs=0 sweep, so `cdg_axis=vds_at_vgs0` with
`multi_bias_present=false` and an explicit disclosure — no bias axis is
synthesized. A measured C-V diagnostic finite-differences the emitted Q(V)
pairs and compares dQ/dV with the measured C(v) against a declared 10 %
diagnostic budget (separate from the 0.5 % charge budget); the measured Cdg
errors are 15.825 / 13.224 / 6.806 / 50.692 % and Cgs stays ≤1.249 %
(IRFS / TI / EPC / CPM2), so the diagnostic is reported, not silently
ignored. Every emitted table appends a **physical end segment**
(`clamp_charge_above_last=True`): above the last breakpoint Q stays constant
instead of extrapolating the last slope. Candidate selection is
fidelity-first: the channel is `tier2` on all four; charge mode is `tier2`
for IRFS and `tables` for TI / EPC / CPM2; the ANN residual is compared on
the selected charge emission and rejected on all four (`tier2` retained).

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
2026-11-21 regeneration).** The emitted ASCII subcircuit is a standard-SPICE
ideal switch (`S_SW` + `.MODEL SW_T4`) with measured Curve-3 `Ron` and
Curve-7 `Vth`; the conduction/switching loss LUT is **comment-only metadata
for offline post-processing** (PLECS P-TC role), not an active electrical
loss element or thermal RC network. The LUT itself supports a real measured
(Vbus, Id) grid with bilinear interpolation only on complete cells, an
optional temperature axis, and a caller-configured thermal network; the
current records are a single 25 °C slice with `thermal.status =
not_configured`, so those capabilities are present but not exercised. The DPT
waveform gate is **retired by design**: no dv/dt, Miller droop, ringing or
reverse-recovery fidelity is claimed. One measured linear Curve-4 Coss at
`cap_ref_Vds` plus a measured Vgs=0 DC-sweep `Roff` are emitted as solver
stabilizers only — they bound the drain node so the solver reaches `t_end`,
but they slow drain dV/dt and add L-C overshoot; they do not restore waveform
fidelity. No tolerance, step-size or simulator change is claimed, and no
PLECS/SIMPLIS step-count speed is attributed to the emitted switch. Speed
rows carry `speedup.method="LTspice default/trap (both arms)"` on all four
vendors.

**Conditions of validity (Tier-4).** The vendor golden at LTspice defaults is
the sole reference. Energy nodes come from each vendor's recorded golden
2-pulse DPT crossing (2 ns forced, 14 µs). Speed is the existing 20-pulse,
176 µs adaptive clamped-inductive bench (median of 3 repeats, golden vs
Tier-4, `method=LTspice default/trap (both arms)`), with EPC's golden alone
at its documented 100 ns cap; N/A is reported whenever a bench does not
complete. Completion guards: emitted DPT on **4/4** devices
(`t_last = 1.4e-5 s`, positive iterations, ≥50 % of golden trace points);
speed on **4/4** (no error, positive iterations/points, full-schedule
`.tran` stop, 3 repeats). CPM2's 4th `Tj` port is driven in both arms, so its
adaptive golden completes and Tier-4 measures 1.3618× solver / 1.1586× wall
(12,492 → 5,320 iterations); VM-10 is a 3-pin harness bug, not a contract
exclusion. The bench discards per-run time vectors, so speed `t_last` is
N/A and `end_time_s` is schedule metadata, not a measured trace endpoint.
*AES-STD100 caveat above applies: golden-vs-model figures, not datasheet
ratings.*

| Device (Vbus / Id) | Solver speedup | Wall speedup | Golden → Tier-4 iterations | Golden → Tier-4 points | Case |
|---|---:|---:|---|---|---|
| IRFS7730PBF (38 V / 100 A) | 1.2212× | 1.0994× | 12,718 → 6,165 | 3,432 → 2,247 | long_adaptive |
| CSD16412Q5A / TI (12.5 V / 10 A) | **6.4730×** | 3.2734× | 131,558 → 6,201 | 44,050 → 2,296 | long_adaptive |
| EPC2216 (11 V / 10 A) | 1.2440× | 1.1019× | 16,514 → 6,296 | 4,047 → 2,597 | long_100ns (golden 100 ns cap) |
| CPM2-1200 (20 V / 20 A) | 1.3618× | 1.1586× | 12,492 → 5,320 | 3,092 → 1,973 | long_adaptive (Tj driven) |

**Tier-4 energy/efficiency scope.** E_on/E_off relative error is 0 % **by
construction**: the LUT value and the compared "golden" value share the same
`dpt_metrics.json#golden_metrics` crossing. The flags live in
`tier4_validation.json#energy_comparison.e_on_uj` and
`#energy_comparison.e_off_uj`: `source_shared: true`,
`independent_energy_gate: false`, `kind: source_self_consistency`. Those rows
are source self-consistency, not an independent converter validation. The
recorded Tier-4 efficiency gate remains `N/A` (`output_power_required`):
E_on/E_off/r_on are evaluated offline, and Vbus×Iload is DC bus input power,
not converter output power. A separate measured buck bench now supplies
converter output power: EPC2216 Tier-4 measures 99.87 % there when the run
completes, while incomplete arms keep their guard reason.

**Tier-4 ANN (`run_vendor_ann`, `*_PIECESPICE_T4_ann.lib`, `tier4_validation_ann.json`).**
Positive-conductance `C-inf` ANN channel (trainable `alpha`, `vth_eff`,
per-curve peak normalization, `r_source_ohm=None`) + rational `Q_dep(V)` +
Curve-5 `Vdg < 0` `Q_gd` excess + calibrated dynamic parasitics, with package
inductances (`LG`/`LS`/`LD`) opt-in via `package_inductance: true/false`
(default `false`, `Rpar=20` damping when enabled). Speedup reports both raw
`solver_speedup` and `net_solver_speedup` above the measured LTspice internal
floor (`LTSPICE_INTERNAL_FLOOR_S = 0.075 s` on a trivial 100-point `.tran 1u 100u`):

| Device | Vds / Id RMSE (%) | td_on / tr / td_off / tf err (%) | E_on / E_off err (%) | n=20 raw / net speedup | n=200 raw / net speedup |
|---|---|---|---|---|---|
| `irfs7730pbf` | 0.47 / 0.66 | +0.03 / −3.92 / +2.53 / −3.50 | +2.95 / −11.82 | 0.96× / 0.95× | 0.96× / 0.95× |
| `csd16412q5a` | 6.32 / 4.60 | −68.04 / +26.31 / +11.38 / −7.25 | −21.66 / −55.09 | 5.63× / 7.45× | 19.33× / 21.43× |
| `epc2216` | 2.68 / 0.83 | +1.22 / +1.24 / +0.82 / −3.15 | +27.69 / +41.24 | 1.10× / 1.14× | 2.25× / 2.43× |
| `cpm2-1200` | 2.50 / 2.12 | −13.69 / −12.44 / −5.15 / −0.67 | +20.94 / +6.26 | 1.21× / 1.29× | 1.46× / 1.53× |

Tier lock summary: Tier-2, Tier-3 `v1`, and Tier-3 `simplis` (`C^0` PWL ceiling:
EPC `0.47%/0.18%`, CPM2 `0.59%/1.05%`, TI `5.59×`) are locked; `irfs7730pbf`
golden is a 3-element `LEVEL=1` MOSFET (~1× speedup ceiling), and `epc2216`
sub-nanosecond edge (`0.85 ns`) caps switching energy alignment.

The loss LUT's
finite-node provenance is vendor-specific: IRFS and TI have one `constant_node`
(`1×1` grid), EPC and CPM2 have `grid_interpolation` with 3 and 4 finite
nodes respectively; off-node queries are NaN and no unmeasured corner is
invented. TI's
6.4730× comes from the stiff golden (131,558 golden vs 6,201 Tier-4
iterations) and is not presented as a portable speed class. No unmeasured
superlative is claimed.

**DPT timing and energy — golden / Tier-3 / Tier-2.** `tr`, `tf` in ns;
`e_on`, `e_off` in µJ. Switching energy integrates the **signed** device
current `I(Lload)-I(Dclamp)` on both golden and ROM arms
(`energy_current_definition` in every record); `abs()` applies only to G3,
timing and peak comparisons. Rectifying the current turns zero-mean
post-turn-off ringing into apparent loss: the retired abs-integrated golden
e_off was 0.953449 µJ (TI) and 1.652084 µJ (CPM2), versus the signed values
below. Every energy row carries a signed/abs label; a mismatch is
`definition_mismatch`, never a silently scored numeric error. All Tier-3
crossings now exist on every device (the earlier IRFS N/A cells were a
second-pulse turn-on failure, fixed 2026-11-19); values are never imputed.

| Device | tr (ns) G / T3 / T2 | tf (ns) G / T3 / T2 | e_on (µJ) G / T3 / T2 | e_off (µJ) G / T3 / T2 |
|---|---|---|---|---|
| IRFS7730PBF | 25.84 / 1.184 / 2.893 | 26.54 / 1.473 / 2.529 | 45.25 / 7.938 / 12.88 | 77.72 / 3.316 / 12.87 |
| CSD16412Q5A | 1.096 / 0.106 / 1.486 | 1.662 / 0.624 / 2.246 | 0.379 / 0.336 / 0.265 | 0.370 / 0.098 / 0.230 |
| EPC2216 | 0.846 / 1.485 / 0.838 | 1.227 / 2.255 / 1.343 | 0.398 / 0.616 / 0.542 | 0.231 / 0.367 / 0.388 |
| CPM2-1200 | 4.703 / 0.850 / 4.123 | 4.139 / 2.429 / 8.789 | 2.760 / 1.811 / 3.214 | 1.420 / 0.698 / 3.314 |

*AES-STD100 caveat above applies.*

**Fidelity scope — edge timing.** The small full-waveform rel-RMSE above does
not imply accurate switching edges. `tr`/`tf` deviate measurably on several
devices: CPM2 Tier-3 `tr` 0.850 ns vs golden 4.703 ns, and IRFS Tier-3 `tr`
1.184 ns vs 25.84 ns. These ROMs hold terminal waveform shape at the recorded
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

**Tier-3 `urom` — final equal-sandbox result: overall `reject` 4/4
(`--tier 3 --variant urom`, regenerated records).** The top-level canonical
record in `reports/comparison/<vendor>/unified_validation.json` selects the
C0 basis (`channel_basis="tier2"`, `emit="tier2"`) and emits ASCII
`reports/models/<vendor>_PIECSPICE_UROM.lib`; its header prints
`basis=tier2 method=trap/default tuning=off`. The final verdict is per
vendor and is **`reject` on all four**. Fidelity is `accept` for TI / EPC /
CPM2 and `reject` for IRFS; the independent G4 timing/energy gate is
`reject` on all four; speed is `accept` only for TI and CPM2. No global or
portable claim is made (`global_claim=false`, no portability screen).

**G4 is now an independent golden gate.** Timing and energy are scored
against each vendor's own golden DPT crossing
(`timing_bar_source=tier3_validation.json#dpt_comparison.<metric>.golden`),
not against the old Tier-2 self-comparison. That old comparison is retained
separately as `G4_tier2_regression`, labelled "Tier-2 self-comparison
(diagnostic only; never scored)", and it passes on all four while the
independent gate fails on all four. This is why fidelity acceptance cannot
be read as an overall acceptance.

**G3 uses the device-current identity.** Golden and ROM both use
`I(Lload)-I(Dclamp)` on the equal 7000-point common grid
(`window_equal=true`, `common_grid_guard=true`). The G3 bars are the recorded
per-vendor Vds baselines plus a declared Id 5 % application requirement:
IRFS 10.0 / 5.0, TI 5.583 / 5.0, EPC 4.894 / 5.0 and CPM2 3.0 / 5.0
(Vds / Id, %). IRFS fails on Id 9.0127 %; TI, EPC and CPM2 pass. The former
Id bars (0.60 / 0.048 / 0.049 / 0.261 %) came from the old inductor-current
metric and are retired.

| Device (Vbus / Id) | Fidelity | G4 timing/energy | G5 speed | Overall | Static rel-RMSE | Solver / wall speedup | Iterations golden → urom | G3 Vds / Id vs bar |
|---|---|---|---|---|---|---|---|---|
| IRFS7730PBF (38 V / 100 A) | **reject** (G3 Id 9.0127 %) | reject | **reject** | **reject** | 7.0683 % | 1.0073× / 0.9860× | 12,718 → 9,274 | 9.999 / 9.013 vs 10.0 / 5.0 |
| CSD16412Q5A / TI (12.5 V / 10 A) | **accept** | reject | **accept** | **reject** | 8.4935 % | **5.3021× / 3.0086×** | 131,558 → 10,195 | 1.984 / 1.259 vs 5.583 / 5.0 |
| EPC2216 (11 V / 10 A; 100 ns cap both arms) | **accept** | reject | **reject** | **reject** | 2.0394 % | 0.9863× / 1.0169× | 16,514 → 12,206 | 0.704 / 0.662 vs 4.894 / 5.0 |
| CPM2-1200 (20 V / 20 A) | **accept** | reject | **accept** | **reject** | 3.4287 % | 1.1023× / 1.0583× | 12,492 → 9,214 | 2.893 / 3.209 vs 3.0 / 5.0 |

Static rel-RMSE returns to the recorded Tier-2 bars
(7.068256 / 8.493510 / 2.039414 / 3.428701 %, IRFS / TI / EPC / CPM2; estimator
`mean(per_curve_peak_rel_rmse_pct)`) and the worst endpoint at the configured
`Vds_sweep_max` is 3.9605 / 4.1273 / 0.5589 / 1.4597 % — all under the 4.13 %
bar. G0, G1, G2 and G6 pass on every C0 row. G3 passes for TI / EPC / CPM2 and
fails for IRFS on Id; G4 fails 4/4; G5 passes only TI / CPM2. The keyed ANN
rows fail G1, G3 and G4 on all four (EPC also fails G2); no ANN row is
accepted. `fidelity_scope` reports `waveform_gate_retired=false` and
`energy_only_claim=false` — these rows still use the waveform gate, unlike
Tier-4. Speed is scoped to this bench alone: TI's 5.3021× solver / 3.0086×
wall is golden-stiffness driven, EPC is slower than golden on solver
(0.9863×), and IRFS is at unity on solver (1.0073×) with wall below 1×
(0.9860×). CPM2's 1.1023× / 1.0583× is a small, bench-scoped gain.

**ANN basis — fitted charge net + `positive_conductance` channel (`fidelity_all_pass` 4/4, `verdict=reject` 4/4).**
ANN is a separate basis chosen by `--basis ann`, not a variant: the keyed
non-default records `reports/comparison/<vendor>/unified_ann_trap-default.json`
and libs `reports/models/<vendor>_PIECSPICE_UROM_ann_trap-default.lib` carry
`channel_basis="ann"` using the `positive_conductance` (`tanh`, 6 neurons,
`hard_guarantee=true` for `gds > 0`) channel and fitted rational/log charge net.
All four vendors pass **G0, G1_static (1.00–2.07 %, worst endpoint 1.29–4.02 %),
G2_derivative (`gm/gds` negative fraction `0.0`), G3_dpt_waveform
(0.42–2.24 % Vds, 0.40–1.43 % Id), and G6_charge (`monotone_q=true`,
`q_zero_anchored=true`, 0.5 % budget)** (`fidelity_all_pass=true` 4/4), while
overall `verdict` remains `reject` 4/4 on G4/G5. The fit path carries hold-out
fields (`holdout_id_rms_rel`, `holdout_gm_rms_rel`, `holdout_gds_rms_rel`,
`holdout_n`) and reports them as `not_measured` when no hold-out is reserved.
Each canonical record's `artifact_manifest` and every record's
`emitted_lib_sha256` match the on-disk lib and inputs; the canonical and keyed
paths never clobber each other. A fixed-topology tuning API exists, but the
final run reports `tuning.verdict="not_run"` (`tuning.mode="off"`).

**Test suite (measured).** Full live suite: **830 passed, 0 skipped,
2 warnings** (`.venv\Scripts\python.exe -m pytest -q`, 396.77 s). Local LTspice-free
CI (`PIECESPICE_LIVE=0`): **819 passed, 10 skipped, 1 deselected, 1 warning**
(737.00 s). Previous GitHub Actions CI (run **44934384493**, commit `4402640`,
both jobs success): **806 passed,
15 skipped, 1 deselected, 1 warning**. The runner has no `SpiceModels/`, so
vendor-gated tests skip more than locally (`test_vendor_ti` 5 skipped; buck
19 passed / 4 skipped, versus 3 passed / 2 skipped and 21 passed / 2 skipped
locally); the 15 CI skips comprise the `PIECESPICE_LIVE` gate and those
vendor-dependent skips. The live engine module alone reports **19 passed**.
T6.2a closed the four former skips on the LTspice backend —
`test_vendor_ti`, `test_netlist_builder.py::test_fit_pwl_model_generates_lib`,
`test_parser.py::test_normalize_forward_referenced_params` and
`test_parser.py::test_normalize_safe_functions` (29 passed / 0 skipped
targeted). T6.3 removed `pyspice` from `pyproject.toml` and `uv.lock`
(16 passed targeted); no PySpice import or dependency remains. The earlier
713-pass CI result and its run hash are historical (pre-T6.2a) and are not
current evidence.

| Command | Passed | Skipped | Deselected | Warnings |
|---|---|---|---|---|
| `uv run pytest` (full live suite, 396.77 s) | 830 | 0 | — | 2 |
| `PIECESPICE_LIVE=0 .venv\Scripts\python.exe -m pytest -q --deselect tests/test_netlist_builder.py::test_tier2_simulates_and_matches` (local CI) | 819 | 10 | 1 | 1 |
| `PIECESPICE_LIVE=0 uv run pytest -q --deselect tests/test_netlist_builder.py::test_tier2_simulates_and_matches` (GitHub CI, run 44934384493) | 806 | 15 | 1 | 1 |
| Live engine module (`tests/test_engine_ltspice.py`) | 19 | — | — | — |

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
│   ├── acceptance.yaml   # Numeric gate policy (recorded vs project values)
│   ├── converter.yaml    # Buck bench operating point + model selection
│   └── outputs/          # Extracted curve data (YAML)
├── data/                 # Captured golden switching waveforms (CSV)
├── docs/                 # Reports, plans, reference notes
├── figures/
│   ├── <vendor>/
│   │   ├── 01_*, 02_*    # extraction / analysis figures
│   │   ├── tier2/        # golden-vs-Tier-2 DPT figures
│   │   └── tier3/        # golden-vs-Tier-3 DPT figures
│   └── converter/        # Buck waveforms + fixed-path 1x3 golden-vs-tier plots
├── metrics/              # Static/switching/loss/comparison metrics
├── models/               # netlist_builder (Tier-2), tpwl_netlist_builder
│                         # (Tier-3), tier4_ideal_switch_builder (Tier-4)
├── notebooks/            # 01 extraction · 02 approximation · 03 DPT · 04 buck · 05 validation
├── parser/               # Subcircuit/pin parser (dialect transpiler retired)
├── pipeline/             # run_all_curves · fit_pwl_model · tier3 · tier4 ·
│                         # registry (tier registry for the CLI)
├── reporting/            # Single-file HTML report generator
├── simulation/           # LTspice engine + benches (static/cap/gate-charge/
│                         # switching/dpt_trajectory/buck_converter)
├── SpiceModels/          # Vendor libraries (read-only inputs)
├── tests/                # pytest suite
└── scripts/              # Utility scripts (incl. run_tier3)
```

The former `tier3/` package has been folded into the paths above — there are
no re-export shims.

Tier-4 is variant-free: `models/tier4_ideal_switch_builder.py` emits
`reports/models/<vendor>_PIECESPICE_T4.lib` and the guarded record lands in
`reports/comparison/<vendor>/tier4_validation.json`. Its loss LUT is comment
metadata only; the recorded DPT efficiency stays N/A (`output_power_required`),
while `simulation/testbenches/buck_converter.py` measures completed arms and
writes fixed `figures/converter/` plots.

---

## 🧪 Testing

```bash
uv run pytest                # full live suite: 830 passed, 0 skipped, 2 warnings (396.77 s)
PIECESPICE_LIVE=0 uv run pytest -q \
  --deselect tests/test_netlist_builder.py::test_tier2_simulates_and_matches
                             # local CI: 819 passed, 10 skipped, 1 deselected, 1 warning (GitHub run 44934384493: 806/15/1/1)
uv run pytest tests/test_engine_ltspice.py -q   # live engine module: 19 passed
uv run pytest tests/test_pwlf_fit.py -v    # single module
```

The suite covers the parser, every approximation engine (including the
Tier-3 TPWL/DEIM path), metric math, netlist emission, YAML I/O and the
LTspice log-diagnostics layer, plus live LTspice engine tests (transient,
DC sweep, timeout, temperature). The four formerly skipped tests were
rewritten on the LTspice backend under T6.2a with their PySpice imports
removed, and T6.3 purged the `pyspice` dependency from `pyproject.toml` and
`uv.lock`. The LTspice-free CI command deselects the one Tier-2 live match
test and gates the live class (`PIECESPICE_LIVE=0`); its 15 skips include
vendor-dependent tests on the runner without `SpiceModels/`, while the
full-live run has zero skips. GitHub run **44934384493** on commit
`4402640`: both jobs success. Tier-3 has its own test module,
`tests/test_tier3.py`.

---

## 📓 Notebooks

| Notebook | Purpose |
|---|---|
| [`01_datasheet_curves_extraction`](notebooks/01_datasheet_curves_extraction.ipynb) | Extract the 7 datasheet curves from any configured vendor |
| [`02_piecewise_approximation`](notebooks/02_piecewise_approximation.ipynb) | Fit → compress → charge-integrate → smooth → synthesize → validate |
| [`04a_simple_switching_subset`](notebooks/04a_simple_switching_subset.ipynb) | Golden-vs-Tier-2 resistive-load switching comparison (M4 subset) |
| [`03_double_pulse_transients`](notebooks/03_double_pulse_transients.ipynb) | Clamped-inductive DPT: golden vs Tier-2 vs Tier-3 waveforms, timing and energy; writes `figures/<vendor>/tier2/` and `tier3/` |
| [`04_buck_converter_benchmark`](notebooks/04_buck_converter_benchmark.ipynb) | Executed 13-cell buck bench: measured EPC2216 golden-vs-Tier-2 Vout / I_L / I_C (1x3) and fixed `figures/converter/` plots; N/A guards retained |
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
      (IRFS 2.16 vs 7.07 %, TI 1.54 vs 8.49 %, CPM2 0.61 vs 3.43 %;
      EPC 2.37 vs 2.04 % is the exception); the emitted netlist is
      1.70 / 2.15 / 7.22 / 0.97 % (IRFS / TI / EPC / CPM2), with 32 / 32 /
      32 / 24 fitted cells. All four DPTs completed on the common golden
      grid with device current `I(Lload)-I(Dclamp)`.
      **No significant speedup** — matched-condition solver speed
      0.5827–0.0946×, slower than golden on all four devices; the
      2026-11-20 speed audit closed this line of work. The Qg-emission
      experiment was measured, closed and reverted to the four-line emitter.
      The V2 physical-element path (RG/RD/RS, gate charge, body diode), the
      measured 2-D Qoss surface with optional temperature slice, and compact
      Vds-axis emission are implemented **opt-ins**; the shipped records keep
      `compact=false`, `source=default` and
      `physical_elements={include_gate_charge:false, include_body_diode:false,
      include_parasitics:false}`. Multi-trajectory fitting and CV are
      implemented via `config['tier3']['training_conditions']`, but the
      records report `generalization.status="not_measured"` and
      `cv_status="not_measured"` because no extra training conditions were
      configured.
      The `simplis` variant (Level-2 PWL charge tables fitted on voltage
      axes) holds Tier-2 static parity. The ANN residual channel candidate
      was rejected on all four vendors, so the final channel selection is
      `tier2` everywhere. The variant reaches 5.14545× solver / 2.92373×
      wall on TI in the 20-pulse adaptive bench; IRFS (1.07054×), EPC
      (1.14729×) and CPM2 (1.10037×, Tj port driven) sit at the measured
      golden ceiling. Its Id values changed because the old
      inductor-current-only metric hid clamp current — a metric correction,
      not a new model improvement or regression; TI Vds remains higher than
      `v1` on the like-for-like metric.
- [x] **Tier-4 — ideal-switch + loss-LUT fast tier** *(hardened dual-tier addition)*
      DPT completion **4/4**; guarded inductive-bench speed **3/4**
      (`method=LTspice default/trap (both arms)`; solver/wall:
      IRFS 1.2212× / 1.0994×, TI 6.4730× / 3.2734×,
      EPC 1.2440× / 1.1019× at the golden 100 ns cap; CPM2 1.3618× / 1.1586×
      with the Tj port driven. The historical CPM2 N/A was a 3-pin harness bug,
      not a vendor defect. Waveform gate retired by design; energy rows are
      DPT-source self-consistency; the recorded efficiency stays N/A without
      converter output power, while the measured buck bench reports 99.87 %
      for completed Tier-4 arms. The LUT supports a real
      measured grid, an optional temperature axis, Ron/Vt parameters and a
      caller-configured thermal network; the current records are a 25 °C
      slice with `thermal.status="not_configured"`.
- [x] **Tier-3 `urom` — `--tier 3 --variant urom`** *(final equal-sandbox
      result, regenerated records)*: canonical top-level C0/tier2 basis,
      ASCII `<vendor>_PIECSPICE_UROM.lib` with hash-matched
      `unified_validation.json`. **Overall `reject` 4/4**: independent G4
      timing/energy rejects every vendor; G3 fidelity accepts TI / EPC /
      CPM2 and rejects IRFS on device-current Id 9.0127 % > 5 %; G5 speed
      accepts TI (5.3021× / 3.0086×) and CPM2 (1.1023× / 1.0583×) and
      rejects IRFS (1.0073× / 0.9860×) and EPC (0.9863× / 1.0169×).
      Static rel-RMSE returns to the Tier-2 bars; the keyed `--basis ann`
      records and libs are separate, non-default and reject on all four;
      tuning is `not_run`. G4's former Tier-2 self-comparison is retained
      only as `G4_tier2_regression`, explicitly diagnostic and never scored.
      Per-vendor scope with `global_claim=false`: no global or portable
      fidelity/speed claim is made. The official-source fixed-topology
      tuning probe has been measured and closed without an accepted
      candidate; its tuning-on artifacts are temporary diagnostics, not
      final models. See
      [`phase_v_improvement_closure.md`](docs/plans/phase_v_improvement_closure.md).
- [x] **M4** — Switching Dynamics & Double Pulse Testbench
      *(metrics, resistive-load subset and inductive DPT)*
- [x] **M5** — Switched-Mode Converter Benchmark & Speedup Evaluation
      *(buck bench, `config/converter.yaml` + `simulation/testbenches/buck_converter.py`,
      and executed notebook 04 are complete for measured arms. EPC2216
      golden vs Tier-2: Vout 5.3077 / 5.3053 V, efficiency 97.55 / 96.91 %,
      I_L 1.0761 / 1.0815 A, I_C 0.0146 / 0.0204 A; Tier-4 measures 99.87 %
      on the same bench when it completes. Tier-3 `v1` timed out at the
      180 s guard, so no every-tier claim is made. No portable speedup claim:
      the recorded Tier-4 DPT efficiency remains N/A without converter
      output power, and matched-condition/adaptive numbers stay in Measured
      Results.)*
- [x] **M6** — Automated HTML Reporting & CLI Polish
      *(the report renders only metrics it is given)*

Backend: **LTspice 26 + spicelib**, out-of-process; no PySpice/ngspice
dependency remains. Known limitations and vendor-model quirks are tracked in
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
