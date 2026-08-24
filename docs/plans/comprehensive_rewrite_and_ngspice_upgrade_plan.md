# M1, M2 & Notebook 01 Implementation & ngspice Upgrade Plan

**Scope:** Milestones M1 (Parser & Dialect Normalization), M2 (Datasheet Curves Extraction Pipeline), and `notebooks/01_datasheet_curves_extraction.ipynb`.  
**Target Simulator:** ngspice-39 (64-bit Shared Library)  
**Status:** SPECIFICATION & ACTION PLAN  

---

## 1. Core ngspice Engine Binary Upgrade Inventory

The active ngspice runtime used by PySpice resides in:
`<project_root>/.venv/Lib/site-packages/PySpice/Spice/NgSpice/Spice64_dll/`

| Target Path (inside `Spice64_dll/`) | Action | Source File (from `download/ngspice-39_dll_64.7z` or `ngspice-39_64.7z`) | Technical Reason |
|---|---|---|---|
| `dll-vs/ngspice.dll` | **REPLACE** | `Spice64_dll/dll-vs/ngspice.dll` (v39.0.0, ~7.8 MB) | Core engine DLL. Restores native `ddt()` support for dynamic $C_{gd}$ sources; updates BSIM and Newton-Raphson solvers. |
| `dll-vs/ngspice.dll.bak34` | **BACKUP** | Existing ngspice-34 DLL | Instant rollback safety copy. |
| `dll-mingw/libngspice-0.dll` | **REPLACE** | `Spice64_dll/dll-mingw/libngspice-0.dll` | MinGW shared library export for CFFI runtime binding. |
| `dll-mingw/libomp140.x86_64.dll` | **ADD/UPDATE** | `Spice64_dll/dll-vs/libomp140.x86_64.dll` | MSVC OpenMP runtime for multi-threaded matrix operations. |
| `dll-mingw/libgomp-1.dll` | **ADD/UPDATE** | `Spice64_dll/dll-mingw/libgomp-1.dll` | GNU OpenMP runtime dependency. |
| `dll-mingw/libwinpthread-1.dll` | **ADD/UPDATE** | `Spice64_dll/dll-mingw/libwinpthread-1.dll` | POSIX threading support library. |
| `share/ngspice/scripts/*` | **UPDATE** | `share/ngspice/scripts/*` | Updated `spinit` and standard SPICE mathematical constants. |
| `share/ngspice/codemodels/*` | **UPDATE** | `lib/spice/*.cm` (`spice2poly.cm`, `analog.cm`, etc.) | XSPICE codemodel libraries compiled for ngspice-39 ABI. |
| `.spiceinit` | **MAINTAIN** | Project root `.spiceinit` | Set `set ngbehavior=ltpsa` (ensures PSpice `LEVEL=7` maps to `BSIM3v3.2` while supporting LTspice syntax). |

---

## 2. File-by-File Rewrite & Refactor Matrix (M1, M2 & Notebook 01)

| File | Status / Action | Why It Needs Refactoring / Updating | How to Implement |
|---|---|---|---|
| [`parser/pow_harden.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/parser/pow_harden.py) | **REWRITE** | ngspice $\ge 36$ strictly evaluates analytical Jacobians ($\frac{d}{dx} x^{0.5} = 0.5 x^{-0.5}$). When base evaluates to $0$ during $G_{\min}$ stepping, it halts with fatal `Error: 0, -0.5 out of range for pow` on Wolfspeed CPM2 $G_1/G_{11}$. | 1. Parse B/G/E expressions via Python `ast`.<br>2. Replace $X^{0.5}$ / $\text{pow}(X, 0.5)$ with `sqrt(max(X, 1e-12))` (*ngspice `sqrt` has internal derivative clamping*).<br>3. Replace $X^k$ with `pwr(max(X, 1e-12), k)`.<br>4. Transform $\log(1 + \exp(x))$ to numerically stable softplus `(max(x, 0) + log(1 + exp(-abs(x))))`. |
| [`parser/preprocessor.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/parser/preprocessor.py) | **UPDATE** | 1. ngspice-34 required commenting out `ddt(` sources; ngspice-39 supports native `ddt()`.<br>2. Must wire `pow_harden` AST rewriter into Pass 3. | 1. Remove `ddt(` comment-out stripping regex.<br>2. Hook `parser/pow_harden.py` as mandatory AST transform for all behavioral sources.<br>3. Retain case-insensitive multi-pass `.PARAM` expansion. |
| [`parser/spice_literal.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/parser/spice_literal.py) | **RETAIN / VERIFY** | Converts SPICE suffixes (`1702.69p`, `2825.325u`, `10Meg`) to Python scientific floats before evaluation. | Ensure full coverage for all unit prefixes in vendor `.lib` and `.spi` files without dropping symbolic parameters. |
| [`parser/netlist.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/parser/netlist.py) | **RETAIN / VERIFY** | Resolves subcircuit trees, handles PSpice `PARAMS:` headers without corrupting pin lists, and provides semantic pin ordering (VM-07). | Ensure `resolve_pin_connections()` dynamically matches declared `.subckt` tokens against YAML `pin_order_semantic`. |
| [`common/roles.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/common/roles.py) | **RETAIN / VERIFY** | Single source of truth for terminal roles (`drain`, `gate`, `source`, `tj`). | Provides canonical role lookups for all M2 testbenches. |
| [`common/diagnostics.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/common/diagnostics.py) | **UPDATE** | Deduplicates and classifies ngspice engine console warnings (e.g. Wolfspeed $V_J$ diode clamps and $G_{\min}$ stepping notices). | Add ngspice-39 specific message patterns (`pow domain`, `version table notice`, `convergence ladder progress`). |
| [`simulation/engine.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/simulation/engine.py) | **UPDATE** | Core CFFI driver for ngspice shared library. Must handle ngspice-39 initialization, stepped-OP execution, and console suppression. | 1. Ensure `run_dc_sweep_builder` executes isolated per-bias fresh circuits.<br>2. Wrap circuit load and analysis inside `ConsoleQuiet` OS fd redirection.<br>3. Handle solver option ladders (`reltol`, `gmin`, `method=gear`). |
| [`simulation/testbenches/static_sweep.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/simulation/testbenches/static_sweep.py) | **UPDATE** | Sweeps static datasheet curves (Curve 1: Output, Curve 2: Transfer, Curve 3: $R_{ds(on)}$, Curve 6: Body Diode, Curve 7: $V_{th}$). | 1. Use stepped-OP engine builders to eliminate multi-point DC convergence traps.<br>2. Ensure body diode testbench subtracts internal $10\text{ M}\Omega$ bleed resistor current.<br>3. Enforce strict polarity check (`_check_sign`). |
| [`simulation/testbenches/capacitance.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/simulation/testbenches/capacitance.py) | **UPDATE** | Curve 4: $C_{iss}, C_{oss}, C_{rss}$ small-signal AC sweeps across $V_{ds}$. | 1. Ensure 4th port $T_j$ voltage source is driven (`V(tjsrc) = temperature`).<br>2. With ngspice-39 `ddt()` enabled, verify $C_{rss}$ dynamic charge source extraction.<br>3. Apply NaN masking and breakdown region trimming. |
| [`simulation/testbenches/gate_charge.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/simulation/testbenches/gate_charge.py) | **UPDATE** | Curve 5: Gate charge $V_{gs}(Q_g)$ extraction. | 1. Execute `reconstruct_gate_charge` quasi-static 3-segment model (bypassing MODPEX transient mirror collapse VM-02).<br>2. Apply datasheet $Q_{gd}$ calibration scaling (`qgd_cal_nC`). |
| [`simulation/testbenches/xinst.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/simulation/testbenches/xinst.py) | **RETAIN / VERIFY** | Shared helper attaching subcircuit instances with arbitrary port roles and thermal node driving. | Ensure zero net omissions across 3-pin and 4-pin device topologies. |
| [`pipeline/curves.py`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/pipeline/curves.py) | **UPDATE** | Unified config-driven runner (`run_all_curves(vendor)`). | Orchestrates all 7 curves for a vendor from `config/inputs/<vendor>.yaml` and emits standardized `config/outputs/<vendor>/` YAML files. |
| [`notebooks/01_datasheet_curves_extraction.ipynb`](file:///c:/Users/GNKRISHNAN/Documents/GitHub/PieceSpice/notebooks/01_datasheet_curves_extraction.ipynb) | **UPDATE / VERIFY** | End-to-end interactive visual verification notebook for all 7 datasheet curves across all 4 vendor models. | 1. Thin wrapper calling `run_all_curves(VENDOR)`.<br>2. Verify clean execution (0 errors, 7 figures with EAST-aligned outside legends) on Infineon IRFS7730PBF, TI CSD16412Q5A, EPC EPC2216, and Wolfspeed CPM2-1200 under ngspice-39. |
