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
   - **Platform**: Python-native orchestration (`pyspice` + `ngspice`) using `uv` for reproducible environment and dependency management. Root-level modular code organization.
   - **Outcome**: Deliver 10× to 100× transient simulation acceleration in standard SPICE without convergence failures while preserving switching timing, dynamic losses, and resonant parasitic behaviors within <5% RMSE of vendor SPICE models.

---

## 2. Specific Mathematical Formulations & Algorithmic Methods

PieceSpice employs six core algorithmic methods to convert complex non-linear vendor SPICE models into ultra-fast, robust piecewise models:

### 2.1 Algorithm 1: Continuous Piecewise Linear Regression (PWLF)
For static characteristic curves (e.g., $I_d-V_{ds}$ at fixed $V_{gs}$, or $Q_{\text{oss}}-V_{ds}$), PieceSpice finds global optimal breakpoint placements that minimize squared error while guaranteeing $C^0$ continuity across all segment boundaries.

Given $N$ sampled data points $(x_i, y_i)$ and a chosen number of line segments $K$:
$$\hat{y}(x) = \beta_0 + \sum_{k=1}^K \beta_k \max(0, x - \xi_k)$$
where $\xi = (\xi_1, \xi_2, \dots, \xi_{K-1})$ are the internal breakpoint coordinates.

**Algorithmic Workflow**:
1. **Global Search**: Differential evolution or dynamic programming explores the continuous space of breakpoint locations $\xi$.
2. **Local Least-Squares**: For a fixed breakpoint vector $\xi$, solve the linear regression system:
   $$\mathbf{X}_\xi \boldsymbol{\beta} = \mathbf{y} \implies \boldsymbol{\beta} = (\mathbf{X}_\xi^T \mathbf{X}_\xi)^{-1} \mathbf{X}_\xi^T \mathbf{y}$$
3. **Error Metric**: Compute $L_2$ RMSE and $L_\infty$ maximum absolute error:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}(x_i))^2}, \quad L_\infty = \max_i |y_i - \hat{y}(x_i)|$$
4. **Adaptive Selection**: Increment segment count $K$ until $\text{RMSE} \le \text{tolerance}$ (typically $<1\%$).

---

### 2.2 Algorithm 2: Native 2D Ramer-Douglas-Peucker (RDP) Simplification
To prune high-density simulation data without losing critical non-linear inflection points (e.g., threshold knee, saturation transition):

**Mathematical Formulation**:
For a sequence of points $P_1, P_2, \dots, P_M$ in $\mathbb{R}^2$:
1. Form line segment between start $P_1 = (x_1, y_1)$ and end $P_M = (x_M, y_M)$.
2. For each intermediate point $P_i = (x_i, y_i)$, compute perpendicular distance:
   $$d(P_i, \overline{P_1 P_M}) = \frac{|(y_M - y_1)x_i - (x_M - x_1)y_i + x_M y_1 - y_M x_1|}{\sqrt{(x_M - x_1)^2 + (y_M - y_1)^2}}$$
3. Find point $P_{\text{max}}$ with distance $d_{\text{max}} = \max_i d(P_i, \overline{P_1 P_M})$.
4. If $d_{\text{max}} > \epsilon$: recursively split into sub-curves $P_1 \dots P_{\text{max}}$ and $P_{\text{max}} \dots P_M$.
5. If $d_{\text{max}} \le \epsilon$: discard intermediate points and retain line segment $\overline{P_1 P_M}$.

---

### 2.3 Algorithm 3: Charge-Conserving Capacitance Modeling ($Q(V)$) & Stored Energy
Standard SPICE $i = C(v) \cdot \frac{dv}{dt}$ current sources create numerical drift and solver non-convergence due to derivative discretization noise. PieceSpice converts all non-linear capacitance curves into stored charge representations.

**Mathematical Formulation**:
1. **Charge Integration**:
   $$Q_{\text{oss}}(V) = \int_0^V C_{\text{oss}}(v)\, dv$$
   Numerically evaluated using cumulative Simpson/Trapezoidal quadrature over $V \in [0, V_{\text{max}}]$.
2. **Capacitive Energy Integration**:
   $$E_{\text{oss}}(V) = \int_0^V v \cdot C_{\text{oss}}(v)\, dv = V \cdot Q_{\text{oss}}(V) - \int_0^V Q_{\text{oss}}(v)\, dv$$
3. **Linear Equivalent Capacitances**:
   $$C_{\text{oss,eq(charge)}} = \frac{Q_{\text{oss}}(V_{\text{bus}})}{V_{\text{bus}}}, \quad C_{\text{oss,eq(energy)}} = \frac{2 E_{\text{oss}}(V_{\text{bus}})}{V_{\text{bus}}^2}$$
4. **SPICE Implementation**: Synthesized as a charge-conserving behavioral source $B_Q$ or multi-breakpoint PWL charge table evaluated directly in NgSpice.

---

### 2.4 Algorithm 4: $C^1$-Continuous Knee Smoothing (Continuous Jacobians)
To prevent Newton-Raphson iteration oscillation at piecewise boundaries where second derivatives jump ($\frac{d^2I}{dV^2} \to \infty$), PieceSpice injects localized $C^1$ smoothing kernels:

**Mathematical Formulation**:
- **Softplus Kernel**:
  $$f_{\epsilon}(x) = \epsilon \ln\left(1 + \exp\left(\frac{x}{\epsilon}\right)\right), \quad \lim_{\epsilon \to 0} f_\epsilon(x) = \max(0, x)$$
  First derivative (Jacobian):
  $$\frac{df_\epsilon(x)}{dx} = \frac{1}{1 + \exp(-x/\epsilon)} = \sigma\left(\frac{x}{\epsilon}\right)$$
- **Hyperbolic Tangent Knee Transition**:
  $$I_d(V_{ds}, V_{gs}) = g_m(V_{gs}) \cdot V_{ds,\text{sat}} \cdot \tanh\left(\frac{V_{ds}}{V_{ds,\text{sat}}}\right)$$
This guarantees that all Jacobian matrix entries $\frac{\partial I}{\partial V}$ remain smooth, finite, and strictly monotonic throughout the entire solution space.

---

### 2.5 Algorithm 5: Automated Double Pulse Test (DPT) Feature Extraction
The DPT algorithm extracts dynamic switching parameters and energy losses from inductive transient waveforms:

```
           Pulse 1 (Inductor Charge)       Pulse 2 (Test Pulse)
  V_gate:  ┌───────────────────────┐       ┌──────────────────
           │                       │       │
  ─────────┘                       └───────┘
                                   |<--Turn-Off-->|  |<--Turn-On-->|
```

**Algorithmic Extraction Steps**:
1. **Threshold Identification**:
   - $V_{ds,10\%} = 0.1 \cdot V_{\text{bus}}, \quad V_{ds,90\%} = 0.9 \cdot V_{\text{bus}}$
   - $I_{d,10\%} = 0.1 \cdot I_{\text{load}}, \quad I_{d,90\%} = 0.9 \cdot I_{\text{load}}$
2. **Timing Figures**:
   - Rise time $t_r$: Time interval between $I_d(t) = 10\%$ and $90\%$ (turn-on).
   - Fall time $t_f$: Time interval between $I_d(t) = 90\%$ and $10\%$ (turn-off).
   - Turn-on delay $t_{d(on)}$: Time from $V_{gs}(t) = 10\%$ to $V_{ds}(t) = 90\%$.
   - Turn-off delay $t_{d(off)}$: Time from $V_{gs}(t) = 90\%$ to $V_{ds}(t) = 10\%$.
3. **Switching Energy Losses**:
   $$E_{\text{on}} = \int_{t_{\text{on,start}}}^{t_{\text{on,end}}} v_{ds}(t) \cdot i_d(t)\, dt, \quad E_{\text{off}} = \int_{t_{\text{off,start}}}^{t_{\text{off,end}}} v_{ds}(t) \cdot i_d(t)\, dt$$
4. **Slew Rate Extrema**:
   $$\left(\frac{dv}{dt}\right)_{\text{max}} = \max_t \left|\frac{dv_{ds}}{dt}\right|, \quad \left(\frac{di}{dt}\right)_{\text{max}} = \max_t \left|\frac{di_d}{dt}\right|$$

---

### 2.6 Algorithm 6: PSpice/LTspice-to-NgSpice Dialect Transpiler
Vendor SPICE models frequently use dialect extensions that fail in native NgSpice. The transpiler algorithm performs the following transformations:

1. **Parameter Scope Flattening**: Parses `.PARAM` statements inside `.SUBCKT`, evaluates numerical and arithmetic expressions (e.g. `.PARAM perim={2*pWidth}` $\to$ `.PARAM perim=1.2356`), and hoists definitions to global scope.
2. **Expression Expansion**: Replaces bracketed parameter references in device instances (e.g. `M1 d g s s NMOS W={pWidth}`) with literal evaluated values.
3. **Table & Poly Syntax Normalization**: Converts PSpice `TABLE {expr} = (x1, y1) (x2, y2)` syntax to NgSpice `B_source` lookup constructs.

---

## 3. Standard Datasheet Curves Definition

PieceSpice automates extraction, piecewise fitting, and golden SPICE verification for the following standard datasheet curves:

| Curve # | Name | Formulation / Axes | Engineering Significance |
|---|---|---|---|
| **Curve 1** | **Output Characteristics** | $I_d$ vs $V_{ds}$ for $V_{gs} \in [3.0\text{V}, 4.5\text{V}, 6.0\text{V}, 10.0\text{V}, 15.0\text{V}]$ | Captures ohmic region slope ($R_{\text{ds(on)}}$), knee transition, and saturation current limit. |
| **Curve 2** | **Transfer Characteristics** | $I_d$ vs $V_{gs}$ at fixed $V_{ds} \ge 10\text{V}$ across $T_j \in [25^\circ\text{C}, 125^\circ\text{C}, 175^\circ\text{C}]$ | Extracts threshold voltage $V_{\text{th}}$, subthreshold slope, and transconductance $g_{\text{fs}}$. |
| **Curve 3** | **On-State Resistance ($R_{\text{ds(on)}}$)** | $R_{\text{ds(on)}} = V_{ds} / I_d$ vs $I_d$ and vs $V_{gs}$ | Defines primary conduction loss across current range and gate overdrive voltages. |
| **Curve 4** | **Capacitances vs $V_{ds}$** | $C_{\text{iss}}, C_{\text{oss}}, C_{\text{rss}}$ vs $V_{ds}$ (Log Scale: $0.1\text{V} - V_{\text{breakdown}}$) | Non-linear junction capacitances controlling switching speed, ringing, and charge loss. |
| **Curve 5** | **Gate Charge Curve** | $V_{gs}$ vs $Q_g = \int I_g\, dt$ at constant $I_g, I_d, V_{\text{bus}}$ | Identifies $Q_{\text{gs}}$, Miller plateau charge $Q_{\text{gd}}$, and total gate charge $Q_g$. |
| **Curve 6** | **Body Diode Forward Characteristic** | $I_{\text{sd}}$ vs $V_{\text{sd}}$ (3rd Quadrant) across temperatures | Evaluates dead-time conduction voltage drop and body diode losses. |
| **Curve 7** | **Double Pulse Test (DPT) Transients** | $v_{ds}(t), i_d(t), v_{gs}(t), P_{\text{inst}}(t)$ during turn-on/turn-off | Yields $t_r, t_f, t_{d(on)}, t_{d(off)}, E_{\text{on}}, E_{\text{off}}, E_{\text{oss}}, dv/dt, di/dt$. |

---

## 4. Architectural & Implementation Constraints

To ensure codebase cleanliness, repeatability, and presentation quality:

### 4.1 Strict Package Management (`uv`)
- All dependencies strictly added and locked via `uv` (`pyproject.toml` + `uv.lock`).
- All scripts, tests, and notebooks executed strictly through `uv run`.

### 4.2 Root-Level Code vs Visualization Separation
- **Core Algorithms (Root `.py` directories)**: Pure Python modules containing classes, functions, numerical routines, and testbenches directly in project root. Zero visualization code in `.py` modules.
- **Visual Exploration & Analysis (`notebooks/`)**: Interactive Jupyter notebooks that import root modules, execute testbenches, and render plots/tables.
- **Automated Verification (`tests/`)**: Standard `pytest` suite ensuring 100% executable integrity.

### 4.3 Mandatory Plotting Constraints
- **EAST-Aligned Outside Legends**: All plot legends MUST be located strictly on the **EAST side (outside the plot canvas to the right)** via `loc='center left', bbox_to_anchor=(1.02, 0.5)` to ensure unobstructed waveform viewing.
- **Legibility**: Figure DPI $\ge 150$ for display and $300$ for file export, font size $\ge 11\,\text{pt}$, dashed grid lines (`alpha=0.4`), and explicit engineering units (V, A, ns, $\mu\text{J}$, pF, $\text{m}\Omega$) on all axes.

### 4.4 Self-Contained HTML Reporting
- Automatic generation of interactive, standalone HTML reports using `jinja2` and `plotly` without external server dependencies.

---

## 5. Complete Root-Level File Structure

```
PieceSpice/ (Project Root)
├── pyproject.toml              # Build config, dependencies, pytest configuration
├── uv.lock                     # Strict uv lockfile
├── README.md                   # Project overview & quickstart
├── piecespice_plan.md          # Comprehensive architecture & execution plan
├── SpiceModels/                # Vendor SPICE libraries (.lib, .spi)
│   ├── CSD16412Q5A.lib
│   ├── csd18510kcs.lib
│   ├── irfs7730pbf.spi
│   └── ...
├── common/                     # SHARED TYPES & PLOTTING UTILITIES
│   ├── __init__.py
│   ├── types.py                # Pydantic data models (CurveData, WaveformData, PWLModelSpec)
│   └── plotting.py             # Standardized plotting helper with EAST legend constraint
├── parser/                     # SPICE PARSING & DIALECT TRANSPILER
│   ├── __init__.py
│   ├── preprocessor.py         # PSpice/LTspice .PARAM & expression flattener for NgSpice
│   ├── netlist.py              # Subcircuit tree & pin inspection
│   └── extractor.py            # Extracts device parameters
├── models/                     # DEVICE ABSTRACTIONS & NETLIST BUILDERS
│   ├── __init__.py
│   ├── base.py                 # SemiconductorDevice base class
│   ├── mosfet.py               # MOSFETDevice model definition
│   └── netlist_builder.py      # Emits Tier 1, 2, 3 SPICE subcircuits
├── approximation/              # CORE APPROXIMATION & FITTING ALGORITHMS
│   ├── __init__.py
│   ├── pwlf_fit.py             # Optimal Continuous PWL fitting with fixed/adaptive segments
│   ├── rdp_compress.py         # 2D RDP curve reduction (Numpy 2.0+ compatible)
│   ├── charge_model.py         # C(V) -> Q(V) numerical integration & charge source synthesis
│   └── smoother.py             # C¹ smoothing functions (softplus, tanh)
├── simulation/                 # SIMULATION ENGINE & TESTBENCHES
│   ├── __init__.py
│   ├── engine.py               # NgSpice / PySpice runner with error isolation
│   └── testbenches/
│       ├── __init__.py
│       ├── static_sweep.py     # Id-Vds, Id-Vgs, Rds(on), Body Diode sweeps
│       ├── capacitance.py      # Ciss, Coss, Crss small-signal & Qoss sweeps
│       ├── gate_charge.py      # Constant-current gate charge Q_g sweep
│       ├── double_pulse.py     # Double Pulse Test (DPT) switching transients & losses
│       └── buck_converter.py   # Switched-mode buck converter transient runner
├── metrics/                    # METRICS & COMPARISON ROUTINES
│   ├── __init__.py
│   ├── static.py               # L_inf, L_2 RMSE, relative error, R² for DC curves
│   ├── switching.py            # tr, tf, td(on), td(off), dv/dt, di/dt extraction
│   └── loss.py                 # Eon, Eoff, Eoss numerical integration & percentage error
├── reporting/                  # AUTOMATED HTML/MD REPORTING
│   ├── __init__.py
│   └── html_report.py          # Jinja2 + Plotly standalone self-contained HTML report generator
├── cli/                        # TYPER CLI INTERFACE
│   ├── __init__.py
│   └── main.py                 # Typer CLI entrypoint
├── notebooks/                  # JUPYTER NOTEBOOKS (Visualizations & Analysis)
│   ├── 01_datasheet_curves_extraction.ipynb  # Extract standard datasheet curves from SPICE
│   ├── 02_piecewise_approximation.ipynb     # PWLF, RDP, and Charge Q(V) generation
│   ├── 03_double_pulse_transients.ipynb     # DPT switching dynamics & loss comparison
│   ├── 04_buck_converter_benchmark.ipynb    # Multi-cycle converter speedup & ripple
│   └── 05_comprehensive_validation_report.ipynb # Full end-to-end comparison report
└── tests/                      # AUTOMATED TEST SUITE
    ├── test_parser.py          # Netlist parser & preprocessor tests
    ├── test_approximation.py   # PWLF, RDP, and charge integration tests
    ├── test_metrics.py         # Curve error & switching energy tests
    └── test_reporting.py       # HTML report generation tests
```

---

## 6. In-Depth Milestone & Task Breakdown (TODO List)

### Milestone 1: Environment, Parser & SPICE Dialect Normalization
- [x] Configure strict `uv` environment with `pyspice`, `ngspice`, `pwlf`, `pydantic`, `typer`, `plotly`.
- [x] Implement `parser.preprocessor` to flatten PSpice `.PARAM` scopes and evaluate curly-brace expressions `{...}`.
- [x] Implement `parser.netlist` to discover subcircuit names, ports, and models in `.lib` and `.spi` files.
- [x] Implement `common.plotting` enforcing the EAST-aligned legend rule.
- [x] Validate parsing against `irfs7730pbf.spi` and `csd18510kcs.lib`.

### Milestone 2: Automated Datasheet Curves Simulation
- [x] Implement `simulation.testbenches.static_sweep`:
  - [x] **Curve 1**: Output Characteristics ($I_d$ vs $V_{ds}$ at $V_{gs} = 3.0\text{V}, 4.5\text{V}, 6.0\text{V}, 10.0\text{V}$).
  - [x] **Curve 2**: Transfer Characteristics ($I_d$ vs $V_{gs}$ at $V_{ds} = 10\text{V}$).
- [x] Implement `simulation.testbenches.capacitance`:
  - [x] **Curve 4**: $C_{\text{iss}}, C_{\text{oss}}, C_{\text{rss}}$ vs $V_{ds}$ log sweep ($0.1\text{V} - 100\text{V}$).
- [x] Implement `simulation.testbenches.gate_charge`:
  - [x] **Curve 5**: Gate Charge $V_{gs}$ vs $Q_g$ under constant current drive.
- [x] Build Notebook `notebooks/01_datasheet_curves_extraction.ipynb` rendering all curves with EAST legends.

### Milestone 3: Piecewise Approximation & Charge Integration Engines
- [ ] Implement `approximation.pwlf_fit` for continuous piecewise linear regression with $K$ segments.
- [ ] Implement `approximation.rdp_compress` with native Numpy 2.0 2D point pruning.
- [ ] Implement `approximation.charge_model` calculating $Q_{\text{oss}}(V) = \int C_{\text{oss}}(v)\, dv$ and stored capacitive energy $E_{\text{oss}}$.
- [ ] Implement `models.netlist_builder` synthesizing executable Tier 2 CPWL subcircuits with charge sources.
- [ ] Build Notebook `notebooks/02_piecewise_approximation.ipynb` evaluating fit residuals and segment counts.

### Milestone 4: Switching Dynamics & Double Pulse Testbench (DPT)
- [ ] Implement `simulation.testbenches.double_pulse` for inductive clamped load switching.
- [ ] Implement `metrics.switching` to extract $t_r, t_f, t_{d(on)}, t_{d(off)}, dv/dt, di/dt$.
- [ ] Implement `metrics.loss` to compute $E_{\text{on}}$ and $E_{\text{off}}$ via $\int v_{ds}(t) \cdot i_d(t)\, dt$.
- [ ] Build Notebook `notebooks/03_double_pulse_transients.ipynb` overlaying Golden SPICE vs PieceSpice Tier 2 switching waveforms.

### Milestone 5: Switched-Mode Converter Benchmark & Speedup Evaluation
- [ ] Implement `simulation.testbenches.buck_converter` for closed-loop / open-loop multi-cycle operation.
- [ ] Benchmark execution wall-clock time, iteration counts, and solver time step histories.
- [ ] Build Notebook `notebooks/04_buck_converter_benchmark.ipynb` proving $>20\times$ speedup and zero convergence errors.

### Milestone 6: Automated HTML Reporting & CLI Polish
- [x] Implement `reporting.html_report` emitting interactive, publication-quality Plotly + Jinja2 HTML dashboards.
- [x] Implement Typer CLI (`parse`, `sweep`, `report`).
- [x] Build Notebook `notebooks/05_comprehensive_validation_report.ipynb` generating `validation_report.html`.
- [x] Verify complete `pytest` test suite passing 100%.

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