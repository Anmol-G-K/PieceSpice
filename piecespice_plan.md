# PieceSpice: High-Fidelity Piecewise SPICE Modeling Framework

## 1. Executive Summary & Problem Motivation

### 1.1 The Power Electronics Simulation Dilemma
Simulating power electronic circuits (switched-mode power supplies, motor drives, inverters, and wide-bandgap GaN/SiC converters) requires balancing **numerical accuracy (fidelity)** and **simulation execution speed**. Current industry workflows are polarized between two extremes:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SIMULATION SPECTRUM                              │
├───────────────────────────────┬─────────────────────────────┬───────────────┤
│ Full Physics SPICE            │ PieceSpice (Proposed)       │ PLECS / Ideal │
│ (Vendor Subcircuits)          │ (Multi-Fidelity PWL/ABM)    │ (System-Level)│
├───────────────────────────────┼─────────────────────────────┼───────────────┤
│ • Stiff non-linear ODEs       │ • Continuous PWL (CPWL)     │ • Ideal 0/∞   │
│ • Picosecond time steps       │ • Charge-conserving Q(V)    │   switches    │
│ • Convergence errors          │ • Smoothed knee transitions │ • Instantaneous│
│ • Extremely slow (hours)      │ • 10× - 100× speedup        │   transitions │
│ • High fidelity (ringing/loss)│ • <5% error vs full SPICE   │ • 2D/3D LUT   │
│                               │ • Standard SPICE netlist    │   loss lookup │
│                               │   compatible                │ • No ringing/ │
│                               │                             │   parasitics  │
└───────────────────────────────┴─────────────────────────────┴───────────────┘
```

1. **Full SPICE Models (Vendor `.lib` / Subcircuits)**:
   - **Fidelity**: High. Accurately capture sub-threshold conduction, transconductance, internal gate resistance, temperature coefficients, and non-linear voltage-dependent parasitic capacitances ($C_{iss}, C_{oss}, C_{rss}$).
   - **Pain Points**:
     - *Stiff Differential Equations*: Steep exponential transfer curves and rapid non-linear capacitance collapse (varying over 3 orders of magnitude within a few volts) force SPICE adaptive time-step algorithms (Trapezoidal/Gear) down to picoseconds ($10^{-12}\,\text{s}$).
     - *Convergence Divergence*: Gradient discontinuities cause Newton-Raphson iteration oscillation (`Timestep too small`, `Singular matrix`, `Gmin stepping failed`).
     - *Simulation Bottleneck*: Simulating 10,000 switching cycles for closed-loop stability or thermal equilibrium is computationally prohibitive.

2. **PLECS (Piecewise Linear Electrical Circuit Simulation)**:
   - **Mechanism**: Replaces semiconductor switches with ideal binary resistors ($R_{\text{on}} \approx 0, R_{\text{off}} \approx \infty$) and solves a variable-topology linear state-space system in 2 integration steps per switching event.
   - **Loss Calculation**: Computes switching energy ($E_{\text{on}}, E_{\text{off}}$) and conduction drop ($V_{\text{on}}$) post-hoc via multi-dimensional Look-Up Tables (LUTs) indexed by $(V_{\text{blocking}}, I_{\text{conducting}}, T_j)$.
   - **Pain Points**: Omits gate drive impedance interactions, $di/dt$ and $dv/dt$ inductive spikes, Miller plateau dynamics, reverse recovery dynamics, and parasitic ringing. Cannot natively co-simulate arbitrary third-party SPICE ICs without dedicated proprietary environments.

3. **PieceSpice Solution**:
   - **Objective**: Synthesize fast, robust, piecewise-linear and behavioral SPICE macromodels directly from vendor `.lib` files or measured curves.
   - **Platform**: Python-native orchestration (`pyspice` + `ngspice`) using `uv` for reproducible environment and dependency management.
   - **Outcome**: Deliver 10× to 100× transient simulation acceleration in standard SPICE without convergence failures while preserving switching timing, dynamic losses, and resonant parasitic behaviors within <5% RMSE of vendor SPICE models.

---

## 2. Theoretical & Mathematical Foundations

### 2.1 Preserving Fidelity While Eliminating SPICE Convergence Failures

#### 1. Charge-Conserving Non-linear Capacitance ($Q(V)$ vs $C(V)$)
Conventional SPICE behavioral capacitance modeling using voltage-controlled current sources:
$$i_C(t) = C(v) \cdot \frac{dv(t)}{dt}$$
causes severe numerical instability and charge-drift error across large voltage excursions because discrete numerical differentiation amplifies high-frequency noise.

PieceSpice formulates parasitic capacitances ($C_{gs}, C_{ds}, C_{gd}$) via state charge integration:
$$Q_x(V) = \int_0^V C_x(v)\, dv \implies i_C(t) = \frac{d Q_x(v)}{dt}$$
By storing and interpolating $Q(V)$ piecewise or via smooth analytical charge functions, charge conservation is guaranteed at every solver time step, eliminating artificial energy creation/loss.

#### 2. $C^1$-Continuous Piecewise Linear (CPWL) Smoothing
Pure piecewise-linear curves introduce non-differentiable "corners" where the second derivative $\frac{d^2 I}{dV^2} \to \infty$. This triggers Newton-Raphson non-convergence at the boundary.
PieceSpice applies localized smoothing kernels at segment boundaries:
- **Softplus smoothing**: $f_{\epsilon}(x) = \epsilon \ln\left(1 + e^{x/\epsilon}\right)$
- **Hyperbolic tangent smoothing**: $g_{\epsilon}(x) = \frac{1}{2}\left(1 + \tanh(x/\epsilon)\right)$
This maintains linear speed in conduction regions while preserving continuous Jacobians for the SPICE solver.

#### 3. Optimal Segment Allocation: PWLF & RDP Algorithms
To balance model compactness and accuracy:
- **Ramer-Douglas-Peucker (RDP)**: Iteratively reduces point density on static $I-V$ and $C-V$ curves subject to maximum perpendicular distance tolerance $\epsilon$.
- **Piecewise Linear Fit (`pwlf`)**: Applies continuous piecewise regression with dynamic programming / differential evolution to determine global optimal breakpoint locations for a fixed number of segments $K$.

---

## 3. Multi-Fidelity Model Hierarchy

PieceSpice provides three configurable tiers of approximation:

| Tier | Name | Target Use Case | Speedup | Fidelity Scope |
|---|---|---|---|---|
| **Tier 1** | **Behavioral Fast (PLECS-in-SPICE)** | Converter system sizing, loop stability, thermal steady-state | **50× - 200×** | Ideal switch + $R_{\text{ds(on)}}(T) +$ linear $C_{\text{oss,eq}} +$ LUT switching loss injection |
| **Tier 2** | **Continuous Piecewise Linear (CPWL)** | Switching transient optimization, gate driver sizing | **10× - 50×** | Multi-segment $I_d(V_{ds}, V_{gs})$ surfaces, piecewise $Q_{\text{oss}}(V), Q_{\text{rss}}(V)$, PWL body diode |
| **Tier 3** | **Harmonized Macromodel** | EMI pre-compliance, parasitic ringing, high-fidelity loss | **3× - 15×** | Full continuous $C^1$ behavioral sources, charge-based non-linear caps, package parasitics ($L_s, L_d, L_g, R_g$), Foster/Cauer thermal network |

---

## 4. Architecture & Pipeline Breakdown

```
                    ┌──────────────────────────────┐
                    │    Vendor SPICE Library      │
                    │ (.lib / .subckt / .model)    │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Phase 1: Ingestion, Dialect Normalization & Parameter Extraction │
 │ • PSpice/LTspice to NgSpice syntax transpiler (handle .PARAM)    │
 │ • Unified Device Schema (Pydantic models)                        │
 └─────────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Phase 2: Piecewise Generation & Approximation Engine             │
 │ • DC I-V Sweeps (Id-Vds, Id-Vgs)                                 │
 │ • Capacitance Sweeps (Ciss, Coss, Crss vs Vds)                   │
 │ • Charge Integration: Q(V) = ∫ C(v) dv                          │
 │ • Optimal Breakpoints (PWLF / RDP / Adaptive Error Bounding)     │
 │ • C¹ Smooth Continuous Piecewise Assembly                        │
 └─────────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Phase 3: NgSpice Dual Simulation Harness                         │
 │ • Golden Reference Run (Vendor SPICE model)                      │
 │ • Candidate Run (PieceSpice Generated Model)                     │
 │ • Testbenches: DC Curve Sweep, Double Pulse Test (DPT), Buck     │
 └─────────────────────────────────┬────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Phase 4: Metrics, Verification & Automated Reporting             │
 │ • Static Error: L_inf, RMSE on Id(Vds, Vgs) and Qoss(Vds)        │
 │ • Dynamic Error: tr, tf, td(on), td(off), dv/dt, di/dt           │
 │ • Energy Error: Eon, Eoff, Eoss % difference                     │
 │ • Speed Benchmark: Execution time ratio & solver iterations      │
 │ • HTML / Markdown / Jupyter Notebook reports                     │
 └──────────────────────────────────────────────────────────────────┘
```

### 4.1 Phase 1: SPICE Parsing & Dialect Normalization (`piecespice/parser/`)
- **Vendor Syntax Harmonization**:
  - Automatically convert PSpice `.PARAM` definitions inside `.SUBCKT` into global scopes or pre-evaluate expressions `{2*pWidth}` before loading into NgSpice.
  - Parse unencrypted `.subckt` hierarchies, `.model` statements (LEVEL 1, 3, BSIM, VDMOS), diode models, and dependent sources ($E, G, F, H$).
- **Data Models (`piecespice/models/schema.py`)**:
  - Strongly typed schemas with `pydantic` for device parameters ($V_{\text{th}}, R_{\text{ds(on)}}, Q_g, Q_{\text{oss}}, R_g, L_{\text{package}}$).

### 4.2 Phase 2: Piecewise Approximation Engine (`piecespice/approximation/`)
- **`pwl.py`**: Standard 1D/2D Piecewise Linear interpolator.
- **`pwlf_fit.py`**: Optimal continuous piecewise linear fitting with customizable breakpoint counts ($K=3, 5, 8$).
- **`rdp_compress.py`**: Ramer-Douglas-Peucker reduction for dynamic curve simplification with guaranteed error tolerance $\epsilon$.
- **`charge_model.py`**: Converts non-linear $C(V)$ data into charge lookup tables $Q(V)$ and produces standard SPICE subcircuits using charge-conserving behavioral current sources (`B_source` / `G_source`).
- **`smoother.py`**: $C^1$ transition kernels (softplus/sigmoid) across piecewise linear regions.

### 4.3 Phase 3: NgSpice Simulation Harness (`piecespice/simulation/`)
- **`engine.py`**: Robust wrapper over `PySpice.Spice.NgSpice.Shared.NgSpiceShared` with automated initialization, `.spiceinit` management, and error handling.
- **Testbenches (`piecespice/simulation/testbenches/`)**:
  1. `static_sweep.py`: Multi-curve DC characteristics ($I_d-V_{ds}$ at various $V_{gs}$, transfer curves $I_d-V_{gs}$, $R_{\text{ds(on)}}$ vs $I_d$).
  2. `capacitance_sweep.py`: AC small-signal and large-signal charge extraction ($C_{\text{iss}}, C_{\text{oss}}, C_{\text{rss}}$ vs $V_{ds}$).
  3. `double_pulse_test.py` (DPT): Industry-standard inductive switching test bench for measuring $t_{\text{rise}}, t_{\text{fall}}, t_{\text{delay}}, E_{\text{on}}, E_{\text{off}}, V_{\text{overshoot}}, di/dt, dv/dt$.
  4. `buck_converter.py`: Continuous switched-mode converter operating at 100 kHz - 1 MHz to measure steady-state waveforms, efficiency, and wall-clock execution time.

### 4.4 Phase 4: Comparison & Metrics Engine (`piecespice/metrics/` & `comparison/`)
- **Static Metrics**:
  - Max Absolute Error ($L_\infty$), Root Mean Square Error ($L_2$), Relative Error ($\%$) over operating voltage/current ranges.
- **Switching & Dynamic Metrics**:
  - Turn-on delay $t_{d(on)}$, rise time $t_r$, turn-off delay $t_{d(off)}$, fall time $t_f$.
  - Peak drain current overshoot $I_{\text{peak}}$, peak drain-source voltage spike $V_{\text{peak}}$.
  - Switching Energy:
    $$E_{\text{on}} = \int_{t_1}^{t_2} v_{ds}(t) \cdot i_d(t)\, dt, \quad E_{\text{off}} = \int_{t_3}^{t_4} v_{ds}(t) \cdot i_d(t)\, dt$$
- **Solver Performance Metrics**:
  - Wall-clock simulation duration, total transient time steps, Newton-Raphson iteration count, solver rejection rate.

### 4.5 Phase 5: CLI, Validation & Reporting (`piecespice/cli/`, `reporting/`)
- **Typer CLI**:
  - `piecespice parse <model.lib>`: Inspect and extract parameters.
  - `piecespice approximate <model.lib> --tier 2 --segments 5`: Generate piecewise model netlist.
  - `piecespice test <model.lib> --bench dpt`: Run comparison between SPICE and PieceSpice.
  - `piecespice report <model.lib> --out report.html`: Generate complete visual validation report.
- **Interactive Jupyter Notebooks (`notebooks/`)**:
  - Step-by-step walkthroughs for DC extraction, DPT switching comparison, buck converter speedup demonstration, and parameter sensitivity studies.

---

## 5. Dependency Management (Strict `uv`)

All packages are strictly managed via `uv` in `pyproject.toml`:

### Core Dependencies (`[project.dependencies]`)
- `pyspice>=1.5`: Python interface to NgSpice shared library.
- `numpy>=2.0`, `scipy>=1.15`: Numerical computation, curve integration, and optimization.
- `pwlf>=2.6.0`: Continuous piecewise linear fitting.
- `rdp>=0.8`: Ramer-Douglas-Peucker polyline simplification.
- `pydantic>=2.10`: Parameter data structures and schema validation.
- `typer>=0.15`: Command-line interface.
- `rich>=13.9`: Formatted terminal output, tables, and progress tracking.
- `pyyaml>=6.0`: Configuration and model metadata serialization.
- `pandas>=2.2`, `polars>=1.0`: Tabular waveform and metric processing.
- `matplotlib>=3.10`, `seaborn>=0.13`: Publication-quality waveform plots.
- `tabulate>=0.10`, `jinja2>=3.1`: Markdown/HTML report generation.
- `shapely>=2.0`: Geometric curve analysis.

### Dev Dependencies (`[dependency-groups.dev]`)
- `pytest>=8.3`, `pytest-cov>=6.0`, `pytest-benchmark>=5.2`: Unit tests, test coverage, and execution benchmarks.
- `hypothesis>=6.120`: Property-based testing for curve approximations.
- `ruff>=0.9`: Fast linting and formatting.
- `jupyter>=1.1`, `notebook>=7.3`, `nbconvert>=7.16`, `ipykernel>=6.29`, `ipywidgets>=8.1`: Interactive analysis and automated notebook verification.

---

## 6. Implementation Roadmap & Milestones

| Sprint | Duration | Phase / Deliverables | Success Verification Checkpoint |
|---|---|---|---|
| **Sprint 1** | Week 1-2 | **Parser & Normalization Engine**<br>• Vendor `.lib` parser with `.PARAM` subcircuit flattening<br>• Pydantic device schema<br>• PySpice NgSpice shared initialization | Parse all models in `SpiceModels/` (`CSD16412Q5A`, `csd18510kcs`, `irfs7730pbf`) without NgSpice parser errors. |
| **Sprint 2** | Week 3-4 | **Approximation Engine (Static & Capacitance)**<br>• DC $I_d(V_{ds}, V_{gs})$ PWLF / RDP extraction<br>• Charge-conserving $Q(V)$ extraction from $C_{iss}, C_{oss}, C_{rss}$<br>• Generation of Tier 1 & Tier 2 SPICE subcircuits | Static $I_d-V_{ds}$ RMSE < 3%; $Q_{\text{oss}}(V)$ charge conservation error < 1% across 0-100V. |
| **Sprint 3** | Week 5-6 | **Simulation Harness & DPT Testbench**<br>• Automated Double Pulse Testbench (DPT) in PySpice<br>• Automated Buck Converter testbench<br>• Golden SPICE vs PieceSpice dual simulation runner | Automated DPT run produces overlaid transient waveforms ($v_{ds}, i_d, v_{gs}$) with zero convergence failures. |
| **Sprint 4** | Week 7-8 | **Comparison Engine & Metrics Suite**<br>• Transient timing extraction ($t_r, t_f, t_d$)<br>• Switching energy integration ($E_{\text{on}}, E_{\text{off}}$)<br>• Wall-clock speedup and iteration benchmarking | Switching loss error < 10%; simulation speedup > 20× on 100-cycle buck converter. |
| **Sprint 5** | Week 9-10 | **CLI, Validation & Reporting**<br>• Typer CLI commands (`approximate`, `simulate`, `compare`)<br>• 7 Interactive Jupyter Notebooks<br>• Pytest regression suite with GitHub Actions | `pytest` suite passes 100%; CLI runs end-to-end model generation and HTML report output. |

---

## 7. Verified Technical References & Literature

1. **PLECS Foundational Principle**:
   - Allmeling, J. H., & Hammer, W. P. (1999). *PLECS – Piecewise-Linear Electrical Circuit Simulation for Simulink*. Proceedings of the IEEE 1999 International Conference on Power Electronics and Drive Systems (PEDS '99), Vol. 1, pp. 355–360.
   - Plexim GmbH. *PLECS User Manual & Thermal Modeling Guide*. [https://www.plexim.com/download/documentation](https://www.plexim.com/download/documentation)

2. **Power Semiconductor Device Modeling & Compact Models**:
   - Mantooth, H. A., Peng, K., Santi, E., & Hudgins, J. L. (2015). *Modeling of Wide Bandgap Power Semiconductor Devices—Part I & Part II*. IEEE Transactions on Electron Devices, Vol. 62, No. 2, pp. 423–442. [DOI: 10.1109/TED.2014.2366367](https://ieeexplore.ieee.org/document/7014282/)
   - Hefner, A. R. (1991). *An Investigation of the Drive Circuit Requirements for the Power Insulated Gate Bipolar Transistor (IGBT)*. IEEE Transactions on Power Electronics, Vol. 6, No. 2, pp. 208–219. [NIST Publication Archive](https://www.nist.gov/publications/investigation-drive-circuit-requirements-power-insulated-gate-bipolar-transistor-igbt)
   - Drofenik, U., & Kolar, J. W. (2005). *A General Scheme for Calculating Switching- and Conduction-Losses of Power Semiconductors in Numerical Circuit Simulations of Power Electronic Systems*. Proceedings of the International Power Electronics Conference (IPEC Niigata 2005). [ETH PES](https://www.pes.ee.ethz.ch/)

3. **SPICE Simulation & Numerical Methods**:
   - Pejovic, P., & Maksimovic, D. (1994). *PETS: A Program for Estimation of Transients in Switched-Mode Power Supplies*. IEEE Transactions on Power Electronics, Vol. 9, No. 6, pp. 573–580.
   - SIMPLIS Technologies (2023). *Piecewise Linear (PWL) Modeling Principles vs. SPICE Non-linear Iterative Solvers*. [SIMPLIS Documentation](https://www.simplistechnologies.com/documentation/simplis/)

4. **Industry Application Notes on Power Device SPICE Modeling**:
   - Infineon Technologies (2020). *Simulation Models for Power MOSFETs (Application Note AN 2014-02)*. [Infineon MOSFET Simulation](https://www.infineon.com/)
   - Texas Instruments. *Understanding and Using SPICE Models for Power MOSFETs & PSpice for TI Guide*. [Texas Instruments Technical Documents](https://www.ti.com/)
   - Efficient Power Conversion (EPC) (2020). *GaN FET Device Models and Transient Simulation Accuracy*. [EPC Device Models](https://epc-co.com/epc/design-support/device-models)