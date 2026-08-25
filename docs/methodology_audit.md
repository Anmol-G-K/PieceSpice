# PieceSpice Methodology Audit

**Scope:** every `.py` file in the repository (excluding `.venv/`, `tests/` read for context), read top-to-bottom against `piecespice_plan.md` §2 (the six core algorithms) and the milestone descriptions.

**Purpose:** document what is actually implemented in code — not what the plan says. No fixes proposed.

---

## 1. Implemented vs Planned

| Plan item | Status | Where it actually lives |
|---|---|---|
| §2.1 PWL regression, adaptive K | **Implemented** | `approximation/pwlf_fit.py` (`fit_piecewise_linear`, `fit_adaptive`) |
| §2.2 RDP simplification | **Implemented** | `approximation/rdp_compress.py` (`compress_rdp`, `rdp_epsilon_for_max_error`) |
| §2.3 Charge-conserving C(V) | **Implemented** | `approximation/charge_model.py` (`integrate_charge`, `pwl_charge_segments`, `synthesize_charge_source`) |
| §2.4 C¹ knee smoothing | **Partially implemented** | `approximation/smoother.py` kernels exist; only the tanh knee is used in the emitted model |
| §2.5 DPT extraction | **Not implemented as planned** | M4-subset resistive-load bench instead: `simulation/testbenches/simple_switching.py` + `metrics/switching.py`, `metrics/loss.py` |
| §2.6 Dialect transpiler | **Implemented (subset)** | `parser/preprocessor.py`, `parser/spice_literal.py`, `parser/pow_harden.py` |
| Curves 1–7 extraction | **Implemented** | `simulation/testbenches/static_sweep.py`, `capacitance.py`, `gate_charge.py`; orchestrated by `pipeline/curves.py` |
| Tier-2 synthesis | **Implemented** | `models/netlist_builder.py` + `pipeline/fitting.py` |
| Curve 7 = DPT per plan §3 table | **Deviates** | Plan lists DPT as Curve 7; implementation extracts threshold voltage as Curve 7 and treats switching as a separate M4-subset stage |
| Converter-level simulation (M5) | **Not implemented** | Nothing in codebase |

---

## 2. How Each Algorithm Is Actually Implemented

### 2.1 PWL Regression — `approximation/pwlf_fit.py`

- **Core fit**: delegates to the third-party `pwlf` library (`PiecewiseLinFit.fit(K)`), which does differential-evolution global search over breakpoint placement plus local least squares. K=1 degenerates to plain `np.polyfit`.
- **Sanitation**: NaN/inf dropped, stable sort by x, duplicate-x collapsed by averaging y (`_sanitize_xy`). Fewer samples than K+1 → forced single-segment fallback.
- **Continuity guarantee** (`_result_from_breaks`): after pwlf returns breakpoints, segments are re-fit independently with `np.polyfit` per segment. Three repair passes then run:
  - *Pass A*: a segment holding exactly one sample gets a slope that connects its predecessor's endpoint through that sample (preserves C⁰ and keeps the feature).
  - *Pass B*: remaining starved segments are merged into a neighbor by deleting the shared breakpoint (iterative, ≤64 iterations).
  - *Exact chaining*: final intercepts are re-derived so each segment starts exactly where the previous ends (`prev_at_break - slope*brk`). Slopes stay least-squares; continuity is exact.
- **Adaptive loop** (`fit_adaptive`): increments K from `min_segments` to `max_segments` (default 12), stops when RMSE ≤ tolerance (absolute or relative to peak-to-peak span, default 1%). Early exit on stagnation: current RMSE > 0.999 × previous RMSE. Tracks best-so-far separately.
- **Result container**: `PWLFitResult` dataclass with breakpoints/slopes/intercepts/RMSE/L∞/stop_reason and an `evaluate()` using `searchsorted`.

### 2.2 RDP — `approximation/rdp_compress.py`

- Textbook Ramer-Douglas-Peucker but **iterative with an explicit stack** (no recursion limit on 10⁵-point arrays).
- Perpendicular distance computed analytically via the cross-product form (`_pldist_2d`), vectorized over all intermediate points at once.
- Optional **relative epsilon**: ε scaled by bounding-box diagonal before running.
- Post-guarantees: endpoints kept, x strictly ascending (defensive dedupe), final true sample force-retained.
- `rdp_epsilon_for_max_error`: binary search (40 iterations) over ε so the interpolated compressed curve stays within an absolute max-error budget of the original samples.

### 2.3 Charge modeling — `approximation/charge_model.py`

- `integrate_charge`: Q(V)=∫C dV and E(V)=∫v·C dV via `scipy.cumulative_trapezoid`. Negative capacitance raises (treated as upstream corruption). Equivalent caps at V_ref: C_eq(charge)=Q/Vref, C_eq(energy)=2E/Vref².
- `pwl_charge_segments`: **greedy farthest-point selection**, not RDP: start with endpoints, repeatedly insert the sample whose linear-interpolation error on Q(V) is largest, until within budget (default 0.5% of Q_max) or 64 points.
- `synthesize_charge_source`: emits an ngspice capacitor with `Q={...}` expression built from chained soft-abs ramps with **delta slopes** (s_k − s_{k−1}); comment documents that absolute slopes diverged 21× in an audit.

### 2.4 Smoother — `approximation/smoother.py`

- `softplus` (log-sum-exp safe form), `sigmoid`, `tanh_saturation` (Id = gm·Vds_sat·tanh(Vds/Vds_sat)), `pwl_smooth` (smoothstep blend within ±ε of interior breaks), `fit_tanh_knee` (Id_sat from plateau; Vds_sat where curve reaches tanh(1)≈0.7616·Id_sat; gm = Id_sat/Vds_sat).
- **Usage reality**: only `tanh_saturation` / `fit_tanh_knee` feed the emitted model. `pwl_smooth`, `softplus`, `sigmoid` exist but the emitted netlist uses raw soft-abs ramps (C⁰), not these kernels.

### 2.5 Parser / preprocessor — `parser/`

- `netlist.py`: regex-based `.SUBCKT/.ENDS` parser with a definition stack for nested subckts; splits PSpice `PARAMS:` header suffix. `resolve_pin_connections` maps semantic roles to positional ports, validating tokens against the header.
- `preprocessor.py`: joins continuations, strips comments, collects `.PARAM` pairs (multiple per line), evaluates them to a fixed point (handles forward references) with a whitelisted-function `eval`, substitutes `{param}` braces, hoists params above subckts. Two special rules:
  - B-source lines containing `ddt(` are commented out (ngspice-34 lacks ddt in B-expressions); documented as exactly-correct for DC extraction.
  - Optional fractional-power hardening wraps non-integer-power bases in `abs()` (`pow_harden.py`) for ngspice ≥36; off by default.
- `spice_literal.py`: SPICE numeric suffixes (p/n/u/m/k/Meg/G/T) translated to scientific notation for the Python evaluator.
- `pow_harden.py`: regex-based operand-boundary walk; skips integer exponents, already-abs-wrapped bases, non-negative literals. Conservative, never rewrites unparseable text.

### 2.6 Extraction testbenches — `simulation/testbenches/`

- `static_sweep.py` (DC): output family (Vgs-stepped Vds sweeps via ngspice DC sweep), transfer (fixed-Vds Vgs sweep), Rds(on) (ohmic region, min-current mask, mΩ output), body diode (gate-tied-to-source reverse conduction with a 10 MΩ bleed anchor whose parasitic current is analytically subtracted; coarse-step retry on divergence), Vth (drain-gate tied, JEDEC constant-current interpolation). All benches normalize the vendor lib first (cached copy under `reports/normalized/`) and assert sign conventions hard-fail rather than warn.
- `capacitance.py` (AC small-signal at f_meas, default 1 MHz): Sim A injects 1 A AC into gate → Ciss=1/(ω|Vg|). Sim B injects into drain through a 1 H bias inductor → Coss=1/(ω|Vd|), Crss=(|Vg|/|Vd|)·Ciss. Non-physical probe results masked NaN, trailing invalid tail trimmed. AC runs retried up to 3× on spurious shared-lib failures.
- `gate_charge.py`: **quasi-static reconstruction, not transient measurement**. The plan's constant-current Ig transient is not implemented — module docstring documents VM-02 (MODPEX sense-and-mirror network pumps amp-level junk into the gate; Vgs collapses to −37 V). Instead builds the standard three-segment datasheet model: Qgs = ∫Ciss(Vbus)dVgs to plateau, Qgd = ∫Crss(Vds)dVds across the drain swing, tail = ∫Ciss(Vds_on)dVgs. Optional datasheet-Qgd calibration rescales the Miller segment (VM-03 workaround). Plateau-clipping folds Miller charge to Vgs_max when the plateau exceeds the window.
- `simple_switching.py` (M4-subset): Vbus—R_load—DUT with R sized for 1 A at Vbus; 50 kHz, 50% duty, 20 ns edges, fixed 2 ns output step, 3 periods. Returns `SwitchingRun`; `steady_state_slice` keeps the last period. Installs the notebook-safe console suppressor at import.

### 2.7 Fitting pipeline & synthesis — `pipeline/fitting.py`, `models/netlist_builder.py`

- `fit_pwl_model(vendor)`:
  - Curve 1 → shared Vds breakpoint grid (linear linspace over sweep range, NOT PWLF-selected breakpoints) with one Id row per Vgs, resampled onto a common length.
  - Curve 3 → scalar rds_on = median of positive mΩ column (fallback 5 mΩ if missing).
  - Curve 4 → re-runs the capacitance bench (~25 pts) and integrates Coss → Qoss(V) table; linear equivalents cq/ce recorded.
  - Curve 7 → vth scalar. Curve 6 → body-diode sample arrays.
  - Metadata defaults: r_gate=1.5 Ω, r_source=r_drain=1e-4 Ω, **c_iss_lin = 3×c_eq_charge, c_rss_lin = 0.15×c_eq_energy** (heuristic ratios, not extracted).
  - Emits `<device>_PIECESPICE_T2.lib` ASCII.
- `netlist_builder.build_tier2_subcircuit`: RG/RD/RS parasitics; `G_CH di ss VALUE={(PWL(Vgs) ramps)*tanh(V(di,ss)/vds_sat)}` where vds_sat is derived from rds_on and the top-curve plateau current (clamped to [5%, 25%] of the Vds span) — **a single shared knee for all Vgs curves**; origin anchor prepends (Vgs=0, Id=0) when absent; C_QOSS as delta-slope charge expression; C_ISS/C_RSS as plain linear caps; body diode as `.MODEL D(IS,N,RS)` with IS/N estimated from two points of the extracted curve (N clamped [0.5, 4], vt fixed 26 mV).
- Validation helpers `compare_tier2_vs_golden(_full)`: stepped OP comparison at each Vgs/Vds grid point through PySpice.

### 2.8 Metrics — `metrics/`

- `static.compute_curve_error`: RMSE, L∞, rel-RMSE normalized by golden peak-to-peak span, R².
- `compare.compare_curve`: interpolates candidate onto golden grid; adds MAPE and pass flag at rel-RMSE ≤5%. `ComparisonReport` aggregates with worst-curve summary.
- `switching.extract_switching_times`: IEC-style crossings; gate plateau from the **95th percentile** of Vgs (duty-cycle agnostic fix); turn-on uses first crossings, turn-off last crossings; dv/dt, di/dt via gradient. Missing crossings yield 0.0.
- `loss.compute_switching_energy`: trapezoidal ∫v·i dt over a caller-supplied window, returned in µJ.
- `fitness.compare_switching_fitness`: tolerances ±20% timing / ±15% energy / ±5% waveform; accepts pre-computed runs to avoid double simulation; waveform compared after time-grid interpolation.

### 2.9 Infrastructure — `common/`, `config/yaml_io.py`, `cli/main.py`

- `diagnostics.py`: console-line classification registry (regex→severity/explanation), SHA-256 dedup over masked messages (floats/instances/device names replaced), repetition counts. `ConsoleQuiet` dup2-devnull fd redirect around analysis. `install_notebook_quiet_console()` replaces PySpice's `send_char` callback with a suppress-everything version that still feeds the collector; wired at `common/__init__.py` import so binding precedes any NgSpiceShared instance.
- `yaml_io.py`: `config/inputs/<vendor>.yaml` source of truth; outputs written under `config/outputs/<vendor>/` with status/conditions/extracted/notes schema; numpy→builtin conversion.
- `engine.py`: PySpice wrapper adding circuit title + solver options to exceptions; RELTOL/ABSTOL/VNTOL/GMIN/ITL options surface (ignored gracefully if rejected); drains ngspice console buffer into diagnostics after analyses.
- `plotting.py`: rcPreset enforcing EAST legends, 300 dpi, serif/STIX, boxed axes, engineering tick formatters.
- `cli/main.py`: typer commands `parse`, `sweep` (single DC curve + HTML report), `curves` (full pipeline).

### 2.10 Notebooks — `notebooks/simple_plotting.py`

The only notebook-folder script (per user direction). Simulates both models live via the shared library, produces 3-cycle overlay, dynamic edge zooms, and a metrics JSON under `reports/switching/`. Edge windows derived per-waveform from percentile levels and settle-walking (no fixed fractions of period).

---

## 3. Deviations From the Plan's Mathematical Formulation

| # | Plan formulation | Actual implementation | Nature |
|---|---|---|---|
| D1 | §2.1: "global optimal breakpoint placements … Differential evolution or dynamic programming" | pwlf library (differential evolution) for K≥2 — matches; but the *final emitted* channel breakpoints are NOT PWLF output at all: `pipeline/fitting.py` uses a plain linspace Vds grid of the raw Curve-1 samples. PWLF is applied inside notebooks/demos, not to the production channel fit. | Structural deviation |
| D2 | §2.1: adaptive K until RMSE ≤ tol "typically <1%" | Production fitting uses `_FIT_REL_TOL = 0.02` (2%) and `_MAX_SEGMENTS = 6`; moreover those constants are defined but the production path never invokes `fit_adaptive` for the channel (see D1) | Looser than planned / partially unused |
| D3 | §2.3: "cumulative Simpson/Trapezoidal quadrature" | Trapezoid only (`cumulative_trapezoid`) | Simplification |
| D4 | §2.3: E(V) identity via ∫Q | Direct trapezoid of v·C(v) — mathematically equivalent, different evaluation route | Equivalent-form deviation |
| D5 | §2.3: "synthesized as charge-conserving behavioral source B_Q or multi-breakpoint PWL charge table" | Emitted as `C… Q={delta-slope soft-abs ramp chain}` — the nonlinear-capacitor Q= form, not a B-source | Implementation choice, documented in-code |
| D6 | §2.4: "injects localized C¹ smoothing kernels" at piecewise boundaries | The emitted channel/charge ramps are soft-abs (C⁰ only, slope jumps at anchors). Only the Vds saturation is smooth (tanh). `pwl_smooth` exists unused. | Gap between claim and emission |
| D7 | §2.4: gm(Vgs) function multiplying the tanh | Single tanh knee shared by ALL Vgs curves; gm enters only through each Vgs row's PWL values. Documented fidelity limit (worst full-sweep errors at low/mid Vds). | Simplification with known cost |
| D8 | §2.5 DPT: inductive load, double pulse | Resistive load, single continuous 50 kHz pulse train; td/tr/tf/Eon/Eoff extracted, Eoss/dv/dt/di/dt computed but no inductor, no two-pulse structure | Scope reduction (user-directed subset) |
| D9 | §2.6: "Table & Poly Syntax Normalization: converts PSpice TABLE … to NgSpice B_source" | **No TABLE syntax converter exists.** Preprocessor handles .PARAM flattening, brace substitution, ddt-commenting, power hardening only. Vendor models relying on TABLE would fail. | Not implemented |
| D10 | §2.6 transpiler scope | Additionally: ddt()-containing sources are *commented out*, not converted — correct for DC but silently changes transient behavior if such models were simulated | Deliberate documented limitation |
| D11 | §3 Curve 2 across Tj ∈ {25,125,175}°C | Single temperature only (YAML `temperature`, default 25 °C); no temperature family anywhere | Not implemented |
| D12 | §3 Curve 7 = DPT | Curve 7 in code = threshold voltage; switching moved to separate stage | Reordering |
| D13 | §3 Curve 5 = measured Qg transient | Quasi-static analytic reconstruction (Option A) with optional datasheet Qgd rescale | Workaround for VM-02, documented |
| D14 | Body diode "across temperatures", GaN third-quadrant note | Single temperature; same topology used for GaN (documented in docstring) | Partial |
| D15 | Parasitics RG/RD/RS from vendor data | Fixed heuristic defaults (1.5 Ω / 0.1 mΩ / 0.1 mΩ), overridable via metadata; never extracted from the vendor netlist | Heuristic placeholder |
| D16 | Linear equivalent caps "at the operating point" | c_iss_lin = 3×C_eq_charge(Coss), c_rss_lin = 0.15×C_eq_energy(Coss) — ratios of the OUTPUT cap, not extracted from Ciss/Crss curves | Heuristic placeholder |
| D17 | vds_sat derivation | Clamped heuristic from rds_on × plateau current bounded to [5%,25%] of Vds span — no optimization/least squares | Heuristic |
| D18 | Body diode IS/N extraction | Two-point estimate with N clamped [0.5,4] and vt frozen at 26 mV; RS reused from rds_on | Approximate |
| D19 | §4.2 "Zero visualization code in .py modules" | `gate_charge.plot_gate_charge`, `common/plotting.py`, `metrics/compare.py` plotting functions live in .py modules | Violated (pragmatically) |
| D20 | Plan file tree shows `parser/extractor.py` | Does not exist; parameter extraction happens implicitly in testbenches | Stale plan reference |

---

## 4. Vendor Models Exercised & What Was Extracted

| Vendor / device | Library | Model type (as encountered) | Extracted successfully | Known issues hit |
|---|---|---|---|---|
| Infineon IRFS7730PBF | `irfs7730pbf.spi` | MODPEX sense-and-mirror behavioral | All 7 curves; switching golden+Tier-2 (waveform rel-RMSE 4.92%) | VM-02: raw Qg transient impossible → quasi-static reconstruction |
| TI CSD16412Q5A | `CSD16412Q5A.lib` | BSIM3v3 LEVEL=7 style with heavy .PARAM usage | All 7 curves **only after normalization** (raw fails: undefined parameter ptrc1); switching both models after Tier-2 regeneration (RMSE 17.6%) | Raw lib unusable without preprocessor; earlier stale Tier-2 lib caused always-on behavior (resolved by regeneration, not code change) |
| EPC EPC2216 | `EPC2216_V0_PSPICE.net` | PSPICE behavioral GaN | All 7 curves; switching both models @ Vbus=11 V (RMSE 7.08%; on-state ~22 mV vs ~51 mV) | Subckt port order is gate-first — yaml pin_order corrected accordingly; vendor model hangs at Vbus ≥12 V in ngspice-46 batch mode while ≤11 V completes (unresolved vendor-side issue) |
| Wolfspeed CPM2-1200 | cpm2 net | SiC behavioral with 4th thermal port Tj | All 7 curves with Tj driven by V-source in °C; switching golden truncated to pre-turn-off window (Tier-2 full 3 cycles, shared-window RMSE 1.68%) | Vendor model aborts at its first gate falling edge in BOTH backends (numerical blow-up to −7.3 V); Crss flat / Qgd depressed at high Vgs under ngspice-34 (earlier observation); requires Tj tie or the X-instance omits the port |

Extraction mechanics common to all four: dialect normalization → positional pin resolution from yaml semantic map → per-bench circuits through `SpiceEngine` → YAML outputs under `config/outputs/<vendor>/`.

Switching-stage status per vendor (M4-subset, 50 kHz / 50% / 1 A):

| Vendor | Golden tran | Tier-2 tran | Fitness evaluated |
|---|---|---|---|
| IRFS7730PBF | ✅ (ngspice-46 capture) | ✅ shared-lib | Yes |
| CSD16412Q5A | ✅ (normalized lib) | ✅ (regenerated lib) | Yes |
| EPC2216 | ✅ @ 11 V | ✅ | Yes |
| CPM2-1200 | ⚠️ truncated at 14 µs | ✅ (via ngspice-46; 3-pin instance) | Shared window only |

---

## 5. Methodology Gaps (plan describes X, code does Y or nothing)

1. **Channel breakpoints are not optimized.** The plan's centerpiece algorithm (global optimal PWL breakpoint placement) is implemented and tested but the production Tier-2 channel uses uniformly-spaced sample grids of the raw extraction. The fitted-PWL machinery currently influences notebooks more than the shipped model.

2. **No per-Vgs saturation knees.** One vds_sat serves the whole output family; the plan's gm(Vgs)·Vds_sat·tanh formulation is realized only in degenerate form. This is the primary cause of the documented full-sweep scorecard failures (worst 43% at CPM2 high-Vgs).

3. **Gate capacitances are linear guesses.** Despite extracting full Ciss(Vds)/Crss(Vds) curves, the emitted model uses `3×C_eq_oss` and `0.15×C_eq_oss` heuristics for input/reverse capacitance. The charge-modeling machinery to emit Qiss/Qrss tables exists (`PWLModelSpec` carries the fields; builder ignores them).

4. **No C¹ smoothing on emitted PWL ramps.** `smoother.pwl_smooth` and softplus kernels were built for this purpose but the netlist emitter uses C⁰ soft-abs chains. Slope discontinuities remain at every transfer anchor.

5. **DPT (plan Algorithm 5 / Curve 7) absent.** No inductive clamp, no double-pulse structure, no Eoss/DPT-specific extraction. The resistive-load subset computes E_on/E_off over ±200-sample windows around gate mid-crossings — a fixed-width bracket, not the plan's edge-delimited integral bounds.

6. **PSpice TABLE/poly constructs unsupported.** The transpiler section promises conversion; none exists. Any vendor model using them cannot be normalized (none of the four exercised vendors required it, which is why the gap is invisible in practice).

7. **Temperature is a scalar.** Every extraction and validation runs at one temperature; the plan's temperature-family curves and thermal-port-aware modeling are unrealized (Tj port is merely driven by a fixed voltage equal to °C).

8. **Rds(on) reported as a median scalar.** Curve 3 extracts the full Rds(on)-vs-Id curve, but fitting collapses it to `median(positive values)` — the curve itself is stored but never modeled.

9. **v_plateau/vth enter the spec but not the model.** `PWLModelSpec.v_th` and `v_plateau` are carried and reported; the emitted subcircuit derives its behavior purely from the PWL anchor chain, so these scalars are informational only (threshold emerges from the anchor positions, not from the extracted Vth).

10. **Validation coverage is asymmetric.** Stepped-OP static validation (`compare_tier2_vs_golden_full`) exists in-library, but the switching fitness path duplicates metric logic in `notebooks/simple_plotting.py` (its own edge-window and metrics functions) instead of consuming `metrics/fitness.py` — two implementations of the same extraction math now coexist, with slightly differing plateau inference (95th percentile vs median-of-last-quarter in the fitness module).

11. **Sign-convention safety net is assert-based.** `_check_sign` hard-fails on polarity mismatches — good — but it validates only the upper half of a sweep and only asserts positivity; a model that flips sign mid-sweep below 50% wrongness passes.

12. **Golden-capture asymmetry in practice.** Because of backend quirks, the four vendors' golden waveforms were captured through different routes (shared-lib vs ngspice-46 subprocess, different Vbus for EPC, truncated span for CPM2). Cross-vendor fitness numbers are therefore not strictly like-for-like.

---

## Appendix A — Module Inventory (what each file contains)

| File | Lines | Contents |
|---|---|---|
| `approximation/pwlf_fit.py` | 354 | pwlf-backed PWL fit, repair passes, adaptive loop |
| `approximation/rdp_compress.py` | 158 | iterative RDP, relative ε, ε binary search |
| `approximation/charge_model.py` | 306 | Q/E integration, greedy PWL charge tables, Q=-source emission |
| `approximation/smoother.py` | 187 | softplus/sigmoid/tanh/pwl_smooth kernels, fit_tanh_knee |
| `parser/netlist.py` | 160 | subckt tree parse, pin role resolution |
| `parser/preprocessor.py` | 192 | continuation join, param fixed-point eval, substitution, ddt rule |
| `parser/spice_literal.py` | 58 | SPICE suffix literal translation |
| `parser/pow_harden.py` | 107 | fractional-power abs() wrapping |
| `pipeline/curves.py` | 173 | 7-curve orchestration |
| `pipeline/fitting.py` | 258 | spec assembly, Tier-2 emission, stepped-OP validators |
| `models/netlist_builder.py` | 210 | Tier-2 subcircuit text synthesis |
| `models/base.py`, `models/mosfet.py` | 36 | pydantic device shells (largely vestigial — pipeline uses PWLModelSpec directly) |
| `simulation/engine.py` | 211 | PySpice wrapper, options surface, diagnostics drain |
| `simulation/testbenches/static_sweep.py` | 462 | DC benches + sign assertions |
| `simulation/testbenches/capacitance.py` | 204 | AC small-signal Ciss/Coss/Crss |
| `simulation/testbenches/gate_charge.py` | 245 | quasi-static Qg reconstruction + plot helper |
| `simulation/testbenches/simple_switching.py` | 134 | M4-subset resistive switching |
| `simulation/testbenches/xinst.py` | 41 | positional X-instance attach w/ Tj handling |
| `metrics/static.py` | 35 | RMSE/L∞/rel-RMSE/R² |
| `metrics/compare.py` | 305 | comparison containers + plot suite |
| `metrics/switching.py` | 129 | IEC crossing extraction |
| `metrics/loss.py` | 29 | ∫v·i dt |
| `metrics/fitness.py` | 206 | golden-vs-Tier-2 scoring |
| `common/types.py` | 166 | CurveData/WaveformData/PWLModelSpec contracts |
| `common/diagnostics.py` | 516 | classification registry, dedup, ConsoleQuiet, notebook quiet |
| `common/roles.py` | 24 | role→node map |
| `common/plotting.py` | 184 | house plot style |
| `config/yaml_io.py` | 137 | input/output YAML I/O |
| `cli/main.py` | 84 | parse/sweep/curves commands |
| `reporting/html_report.py` | 123 | Jinja2/Plotly HTML report |
| `notebooks/simple_plotting.py` | ~290 | standalone switching comparison runner |

## Appendix B — Test Coverage Map

Tests exist for: parser (`test_parser`, `test_preprocessor`, `test_netlist`), approximation (`test_pwlf_fit`, `test_rdp_compress`, `test_approximation`, `test_charge_model`, `test_smoother`), metrics (`test_metrics`), gate charge (`test_gate_charge`), reporting (`test_reporting`), config (`test_yaml_io`), diagnostics (`test_diagnostics`), netlist builder (`test_netlist_builder`), TI end-to-end (`test_vendor_ti`). Suite total: 97 passing at audit time. No tests cover `simple_switching.py`, `metrics/fitness.py`, or `notebooks/simple_plotting.py`.
