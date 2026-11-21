# Error Analysis and Improvement Research: Tier-3 (`v1`, `simplis`), UROM, ANN/Hybrid, and Tier-4

**Date:** 2026-11-21
**Status:** Research only. No code, configuration, model, or record was changed.
**Evidence classes:** `[P]` project-measured record or source code; `[L]` peer-reviewed paper or official documentation; `[I]` inference drawn from the two.

---

## Abstract

PieceSpice has a working extraction and emission pipeline, but the measured records show that its remaining error is concentrated less in static curve fitting than in four places: (i) generalization beyond the single training trajectory and the single 25 °C operating point; (ii) missing physical elements (gate charge, drain charge, body diode, parasitics, temperature); (iii) validation metrics that are not independent of the model or the bar they score against; and (iv) an emitter/table architecture that does not reproduce the speed mechanism claimed by the model-order-reduction literature it cites.

Tier-3 `v1` achieves 0.61–4.24 % static fit error but 2.85–12.69 % Vds and 1.47–11.00 % Id DPT error, and is slower than golden on all four devices (0.0953–0.5661×). The IRFS held-out Id error is 33.27 %, which is the clearest single signal that the model generalizes poorly. Tier-3 `simplis` holds Tier-2 static parity but has 3.03–8.70 % Vds and 1.77–8.52 % Id DPT error; its only ≥5× solver result (TI 5.527×) is golden-stiffness-driven and not portable. The canonical UROM is the closed Tier-2 emitter under a new name; its G1 and G4 gates are regression/self-consistency checks, its G3 uses inductor current rather than device current, and its G5 “speed” gate passes a 0.6457× (slower) EPC result. The ANN basis is rejected 4/4; its charge network is fitted but not emitted, and its channel has no monotonicity, passivity, temperature, or held-out-condition validation. Tier-4 is an ideal switch with a comment-only loss table whose finite grid has one to three nodes; its energy “error” is 0 % by construction and its efficiency is correctly N/A.

The report ranks 15 findings by severity, maps each to primary literature or official documentation, and proposes a three-tier improvement program: P0 metric integrity, P1 physical fidelity and generalization, P2 performance and reproducibility.

---

## 1. Research question, method, and evidence discipline

### 1.1 Question

> Given the current measured records and emitted models, what are the actual error sources in Tier-3 `v1`, Tier-3 `simplis`, UROM, the ANN/hybrid bases, and Tier-4; which of them are model-form, data, numerical, or validation errors; and which improvements are supported by peer-reviewed journals/conferences and official simulator/vendor documentation?

### 1.2 Scope

In scope: the four named model tracks, their emitters, their training/fitting routines, the validation records under `reports/comparison/<vendor>/`, and the documented claims in `README.md`, `AGENTS.md`, `docs/plans/`, and the deep-research syntheses.

Out of scope: new simulations, new parameter extraction, code changes, and any claim that a literature result transfers to PieceSpice without re-measurement.

### 1.3 Evidence discipline

Every quantitative statement about PieceSpice is read from a named record or source function. Every statement about method behavior is attributed to the cited paper or official document. Statements that combine the two are marked `[I]`. No number in this report is an estimate substituted for a missing measurement; where a record is N/A, the report says N/A.

The literature base is restricted to:

- peer-reviewed journals and conferences (IEEE Transactions, SIAM, Elsevier, AIP, PMLR);
- official simulator documentation (Plexim PLECS 5.0, SIMPLIS, LTspice 26 Help);
- official vendor documentation (Wolfspeed);
- official metrology guidance (JCGM/BIPM).

No blog, forum answer, Wikipedia page, or unattributed web summary is used as evidence.

### 1.4 Sources that could not be read in full

Some publisher PDFs were blocked by bot protection or returned HTTP 403/522. In those cases the bibliographic metadata and abstract were verified through Crossref and Semantic Scholar, and the claim is limited to what the abstract states. This is recorded again in §11.

---

## 2. Executive findings, severity-ranked

| ID | Finding | Evidence | Severity |
|---|---|---|---|
| F1 | UROM G5 “speed” passes a model slower than golden (EPC 0.6457× solver / 0.8673× wall) | `reports/comparison/epc2216/unified_validation.json#speed_adaptive_20`; `_evaluate_gates` G5 | **Critical (semantic)** |
| F2 | UROM G3 scores `I(Lload)` while Tier-3 `v1`/`simplis` score `I(Lload)−I(Dclamp)`; Id numbers are not comparable across tiers | `pipeline/unified_rom.py::_run_dpt_arm`; `pipeline/tier3.py::_device_current` | **Critical (metric)** |
| F3 | UROM canonical C0 G1 and G4 are regression/self-consistency checks against Tier-2 bars, not independent fidelity gates | `TIMING_BAR_COLUMN["tier2"]="tier2_metrics"`; `STATIC_BARS_PCT`; `_timing_bars` | **Critical (validation)** |
| F4 | ANN charge networks are fitted but never emitted; the emitted Q tables are resampled measured charge | `approximation/ann_charge.py::_sample_table(x, q)`; `fit_charge_qv`; `models/ann_netlist_builder.py::_charge_lines` | **High** |
| F5 | Tier-3 `v1` trains on one two-pulse trajectory at one operating point; TPWL theory explicitly limits validity to the covered trajectory region and recommends multiple training inputs | `capture_trajectory`; Rewienski & White 2003 §4.2/§5 | **High** |
| F6 | Tier-3 `v1` omits gate charge, drain charge, body diode, and parasitics; edge timing is correspondingly wrong (IRFS tr 0.836 ns vs 25.84 ns) | 4-line emitter; `tier3_validation.json#dpt_comparison` | **High** |
| F7 | Tier-3 cell count is selected by an analytic `predict()` static error, not by the emitted static error; EPC fit 2.37 % becomes emitted 7.22 % | `approximation/tpwl_deim.py::build_model`; `models/tpwl_netlist_builder.py::emitted_static_rmse` | **High** |
| F8 | `simplis` builds all C–V tables from a Vgs = 0 AC sweep; Cgd is then emitted on `V(di,gi)` at high Vgs, where the table was not measured | `pipeline/tier3_simplis.py::_build_spec` / `_decompose_cv`; SIMPLIS Level-2 docs | **High** |
| F9 | Tier-4 LUT is not a LUT: 1–3 finite nodes, zero complete cells, no interpolation exercised; energy “error” is 0 % by construction | `tier4_validation.json#lut.interpolation`; `_energy_comparison` | **High** |
| F10 | No model track has a temperature axis; every record is 25 °C | All configs; `dpt_metrics`; `unified_validation` | **High** |
| F11 | G2 gates finite gm/gds but not their sign; negative differential conductance can pass. G6 gates only Qoss monotonicity, not Qiss/Qrss | `_derivative_checks`; `_charge_checks`; `_evaluate_gates` | **Medium-High** |
| F12 | Benchmarking has 3 repeats, no confidence interval, no environment metadata, and no statistical effect-size test | `runtime.repeats=3`; `_run_interleaved_speed` | **Medium** |
| F13 | Validation bars are circular: endpoint 4.13 % is the measured worst endpoint; IRFS/C0 passes a 10.0 % bar at 9.9993 % | `ENDPOINT_BAR_PCT`; `DPT_BARS_PCT`; records | **Medium** |
| F14 | Names overstate model forms: “TPWL” is a terminal table fit; “SIMPLIS” uses a Tier-2 channel; “UROM” canonical is Tier-2 | Emitters and records | **Medium** |
| F15 | Tier-3 speed architecture is mismatched to the cited TPWL mechanism: a dense nested `table()` is evaluated per Newton iteration, whereas TPWL’s speed comes from a reduced state-space ODE | `models/tpwl_netlist_builder.py` docstring; Rewienski & White 2003 | **Medium-High** |

### 2.1 The central diagnosis

The static curve fit is not the main bottleneck. The records show a model that reproduces the measured static family reasonably well, then loses accuracy where the physics it omitted lives: gate charge, drain charge, reverse conduction, package parasitics, temperature, and off-trajectory operating points. The validation layer then makes those losses hard to see because several gates score the model against a bar derived from the same model family, or against a different current definition.

The literature is consistent on the direction of repair:

- TPWL validity is tied to the region covered by its linearization points; outside that region the reduced model may be no better than a linear model, and multiple training inputs are the stated extension [Rewienski & White 2003; Rewienski & White 2006].
- DEIM has an approximation error bound, but the bound is on the nonlinear term approximation, not on an arbitrary terminal fit that never forms the full state equations [Chaturantabut & Sorensen 2010; Wirtz et al. 2014].
- Charge-based modeling is required for consistent large-signal capacitance behavior [Ward & Dutton 1978; Yang, Epler & Chatterjee 1983].
- ANN compact models become credible when they are physics-constrained, trained over the full operating region, and validated against measured switching, not just static I–V [Yang et al. 2022; Tung et al. 2024; Gong et al. 2026; Cho et al. 2026].
- Loss LUTs are useful only when they are filled over the relevant voltage/current/temperature grid and coupled to a thermal model; PLECS explicitly interpolates inside the grid, extrapolates outside it, and warns that thermal loss is not the electrical power consumed by the device [Plexim PLECS 5.0 Thermal Concepts; Drofenik & Kolar 2005].

---

## 3. Tier-3 `v1` (TPWL / POD-DEIM terminal model)

### 3.1 What is actually implemented

`pipeline/tier3.py::run_vendor` performs five steps:

1. `capture_trajectory(vendor)` runs the golden clamped-inductive DPT and saves internal node voltages (`.save V(xdut:*)`), terminal voltages `[Vgs, Vds]`, and external currents `[I(Lload)−I(Dclamp), −I(Vgs)]`.
2. `approximation/tpwl_deim.py::build_model` scales the first-pulse operating points, computes a POD basis of the trajectory, seeds DEIM anchors, completes them by farthest-point covering, and fits per-cell affine channel and charge coefficients.
3. Cell count is chosen from `s_grid = (8,16,24,32)` by static Curve-1 RMSE.
4. `models/tpwl_netlist_builder.py::emit_subcircuit` emits a flat nested `table()` channel plus an optional two-exponential `C_QOSS`; the emitted subcircuit is four lines: `.subckt`, `C_QOSS`, `G_CH`, `.ends`.
5. Validation runs the emitted DPT, a forced-2 ns runtime bench, and records the results.

The model is **not** a reduced state-space TPWL model in the sense of the 2003 paper. It is a terminal behavioral map: a piecewise-affine channel and charge model fitted to trajectory + static samples, then sampled onto a table. This distinction matters for every error claim below.

### 3.2 Measured error decomposition

Source: `reports/comparison/<vendor>/tier3_validation.json` (regenerated 2026-11-21).

| Vendor | Static fit / emitted | Held-out Id RMSE | DPT Vds / Id | Forced-2 ns solver | ROM points / iterations |
|---|---:|---:|---:|---:|---:|
| IRFS7730PBF | 4.2400 / 4.1543 % | 33.2716 % | 12.6934 / 11.0027 % | 0.5661× | 7,376 / 20,004 |
| CSD16412Q5A | 1.5419 / 2.1531 % | 0.4460 % | 2.9248 / 4.8962 % | 0.0953× | 64,665 / 380,226 |
| EPC2216 | 2.3667 / 7.2226 % | 0.4297 % | 2.8461 / 1.4694 % | 0.3907× | 9,440 / 33,785 |
| CPM2-1200 | 0.6058 / 0.9749 % | 0.2956 % | 4.3785 / 5.1932 % | 0.1659× | 32,250 / 205,154 |

Edge metrics (golden / Tier-3, from the same records):

| Vendor | tr (ns) | tf (ns) | e_on (µJ) | e_off (µJ) |
|---|---:|---:|---:|---:|
| IRFS | 25.844 / 0.836 | 26.539 / 1.410 | 45.432 / 3.154 | 77.720 / 2.414 |
| TI | 1.096 / 0.106 | 1.662 / 0.624 | 0.380 / 0.336 | 0.953 / 0.151 |
| EPC | 0.846 / 1.485 | 1.227 / 2.255 | 0.398 / 0.616 | 0.238 / 0.372 |
| CPM2 | 4.703 / 0.850 | 4.139 / 2.429 | 2.791 / 1.811 | 1.652 / 0.871 |

Two observations are immediate. The small full-waveform RMSE does not imply edge fidelity: a model can be close in RMS while being 30× too fast on `tr`. And the IRFS held-out Id error (33.27 %) is an order of magnitude above its emitted static error (4.15 %), which is the signature of a model that has fitted the static family but not the trajectory dynamics.

### 3.3 Root causes

#### R1. Single-trajectory training and no coverage metric

The literature is explicit: TPWL linearizations cover only the region of state space visited by the training input. If a test trajectory leaves that region, the piecewise-linear model “will most likely not provide a significantly better approximation … than a simple linear reduced model”; the paper recommends multiple training inputs to extend validity to different amplitudes [L: Rewienski & White 2003, §4.2 and §5]. The 2006 LAA paper adds error estimates under its assumptions [L: Rewienski & White 2006].

PieceSpice trains on one two-pulse DPT at one `Vbus`, one `Iload`, one `Vgs_on`, one temperature, and one gate edge [P: `capture_trajectory`; `config/inputs/<vendor>.yaml#dpt`]. The static Curve-1 family is added to the cell fit, but the charge slopes and gate charge are DPT-only [P: `_fit_cells`]. There is no coverage test, no distance-to-anchor guard at emission, and no out-of-trajectory validation. The emitter simply `table()`-interpolates and clamps [P: `models/tpwl_netlist_builder.py`].

**Consequence:** the recorded DPT error is an in-sample or near-trajectory error, not a generalization error. A different bus voltage, load current, gate resistance, or temperature is unvalidated. The IRFS held-out Id result is the visible symptom.

#### R2. DEIM contributes only two seeds; the rest is farthest-point covering

`_anchors_for` computes the SVD of `ws` (shape 2×K), takes `vt = V^T` (shape K×2), sets `k = min(s, vt.shape[1])`, and calls `deim_indices(vt[:, :k])`. For s = 8…32, `k = 2`. The remaining s−2 anchors come from `_farthest_completion` [P: `approximation/tpwl_deim.py`]. The EPC record confirms `deim_seed_count = 2` at `cells = 32` [P: `reports/comparison/epc2216/tier3_validation.json#algorithm`].

This is not a coding error: the operating plane is two-dimensional, so the POD basis has rank at most two. But the label “POD-DEIM anchors” overstates the mechanism. The method is **DEIM-seeded farthest-point covering**. The DEIM error bound applies to the two seed selections, not to the 30 farthest-point anchors.

**Consequence:** there is no a-posteriori error estimate for the cell partition, and no guarantee that the 32 cells cover the operating plane uniformly. The literature provides a-posteriori DEIM error estimators [L: Wirtz, Sorensen & Haasdonk 2014] that PieceSpice does not use.

#### R3. Cell count selected by the analytic model, not the emitted model

`build_model` chooses the best candidate by `cand.predict(...)` static RMSE, i.e. the analytic weighted-affine model [P: `build_model`]. The emitted model is a different object: a nested `table()` with clamping, a Curve-2 gate factor, a low-Vds saturation factor, and a per-vendor policy. The emitted static error is computed later by `emitted_static_rmse`, but it is not used for selection [P: `models/tpwl_netlist_builder.py`].

The EPC row shows the gap: 2.37 % fit error becomes 7.22 % emitted error [P: `tier3_validation.json#function_fit`]. This is a direct model-selection error, not a fitting error.

#### R4. Channel and charge are fitted sequentially from different data, with no joint objective

`_fit_cells` fits the channel from static rows only (to avoid VM-02 DPT corruption) and the drain/gate charge from DPT rows only [P: `_fit_cells`]. The charge fit subtracts the channel prediction from the DPT current, then fits the residual with `dws` (smoothed derivatives). There is no joint least-squares over channel + charge, no cross-term regularization, and no guarantee that the resulting total current is consistent with the measured terminal current.

The literature on charge-based modeling treats terminal current and charge as one consistent system: `i = i_ch + dQ/dt`, with charge conservation enforced across the whole operating surface [L: Ward & Dutton 1978; Yang, Epler & Chatterjee 1983]. Sequential fitting can satisfy each sub-problem while violating the coupled physics.

#### R5. Omitted gate charge, drain charge, body diode, and parasitics

The emitted `v1` subcircuit is exactly four lines: `.subckt`, one `C_QOSS`, one `G_CH`, `.ends` [P: `models/tpwl_netlist_builder.py`; `tier3_validation.json`]. There is:

- no `C_QD` (fitted drain-charge slopes were negative at the blocking point, ~−2.5 nF on TI, and caused an exponential blow-up; the builder omits it) [P: module docstring];
- no `C_QG` (the Qg emission experiment timed out in both dense and compressed forms and was reverted) [P: module docstring; `docs/plans/tier3_method_candidates.md` §18.1];
- no body diode;
- no `RG`/`RD`/`RS`;
- no temperature port.

The literature says gate charge and drain charge are not optional for switching fidelity: they determine the Miller plateau, the gate current, the drain-voltage transition, and the energy [L: Ward & Dutton 1978; Yang et al. 2022; DeBoi et al. 2024]. The measured edge errors are consistent with this omission: IRFS `tr` 0.836 ns vs 25.844 ns, CPM2 `tr` 0.850 ns vs 4.703 ns, IRFS `e_on` 3.154 µJ vs 45.432 µJ [P: records].

#### R6. No temperature, gate-resistance, or operating-condition dependence

Every table is at 25 °C. The DPT config fixes `vgs_on`, `vbus`, `i_load_target`, and `edge_time`. The Wolfspeed dynamic-tuning study varies `RG`, `Tj`, `Vbus`, and `Iload` over 10–20 DPT conditions because a static-characteristic model does not generalize across them [L: DeBoi, Nelson & Curbow 2024]. The paper reports, after tuning four gate-side scalars over those conditions, `dv/dt_off` error 48.9 % → 18.5 % and `E_off` error 26.4 % → 12.9 %, at unchanged model complexity. That is evidence that the missing dimensions are first-order, not second-order.

PieceSpice has a fixed-topology tuning API for exactly four gate-side scalars, but the shipped record is `tuning.mode="off"` / `verdict="not_run"` [P: `unified_validation.json#tuning`]. The Phase V probe used one recorded DPT condition and a metric-subset held-out arm, which is weaker than the paper’s 10–20-condition procedure [P: `docs/plans/phase_v_improvement_closure.md`].

#### R7. Emission is not part of the fit objective

The table is generated from `blended_channel` on a per-vendor grid: 17×21 for IRFS, 19×25 for EPC, 21×31 for CPM2, 25×41 for TI [P: `_POLICY_BY_VENDOR`]. The emitter then applies:

- a Curve-2 gate factor;
- a low-Vds factor `Vds/(Vds+Vsat)`, with `Vsat = 0.2 V` (0.5 V for EPC);
- `max(0, ...)` or a softplus clamp;
- a high-Vds saturation hold for IRFS.

None of these transformations is in the fitting objective. The fit minimizes error of the affine blend; the emitted model minimizes a different functional. The EPC 2.37 → 7.22 % gap is the measurable cost.

#### R8. Table evaluation is the wrong speed mechanism

The TPWL paper’s speed comes from projecting the nonlinear state equations onto a small basis and simulating a small ODE; the online cost is `O(s q^2)` for s linearizations and q reduced states, not a dense table lookup at every Newton iteration [L: Rewienski & White 2003, §4 and Table 1]. PieceSpice instead emits a nested `table()` expression: an outer Vgs table whose entries are inner Vds tables. Each Newton iteration evaluates many table lookups with linear interpolation [P: `emit_subcircuit`]. The builder docstring records that an earlier B-source/weighted-blend form failed, that a hard Voronoi `if()` cascade timed out, and that dense meshes were measured slower; the flat `table()` form is the only one that completed [P: module docstring].

The measured result is consistent: Tier-3 `v1` is slower than golden on every device, including 0.0953× on TI where the ROM takes ~10.5× longer [P: `tier3_validation.json#runtime`]. This is not a solver-tuning problem; it is an architecture mismatch between the emitted object and the method whose speed is claimed.

#### R9. Qoss is one-dimensional and Vgs-independent

`C_QOSS` is integrated from the AC Coss at `cap_ref_Vds` and emitted either as a dense table or as a monotone two-exponential fit (fit RMS 0.09–0.19 % according to the builder docstring) [P: `_qoss_expr`]. It depends only on Vds, not Vgs. The literature and the Wolfspeed study both note that device capacitances depend on both Vds and Vgs and on the measurement stimulus; a single-bias C–V curve is a simplification [L: DeBoi et al. 2024; Yang et al. 2022].

The two-exponential form is a good monotone approximation, but it is still a one-dimensional charge. During the Miller plateau, the effective Cgd is a function of Vdg, which depends on Vgs. A Vgs-independent Qoss cannot capture that.

#### R10. The model is validated on the same 2-pulse DPT used to train it

The validation DPT is the same bench and the same `config/inputs/<vendor>.yaml#dpt` used by `capture_trajectory` [P: `pipeline/tier3.py`; `dpt_trajectory.py`]. There is no held-out condition, no different `Vbus`/`Iload`/`RG`, no temperature sweep, and no device-to-device hold-out. The “held-out” language in the record refers to the second pulse of the same trajectory, not to an independent operating condition [P: `build_model` uses `split` at the second Vgs rising edge].

This is the single largest methodological gap in the Tier-3 claim: the DPT error is a near-training-set error.

### 3.4 Literature-backed improvement program

#### I1. Multi-trajectory training with a coverage metric

Train on DPTs at a designed set of conditions: at least two `Vbus` values, two `Iload` values, two `RG` values (where the vendor model exposes gate resistance), and two temperatures. This follows the TPWL recommendation for multiple training inputs and the Wolfspeed tuning design [L: Rewienski & White 2003 §5; DeBoi et al. 2024].

Add a coverage metric at emission: for each table grid point, compute the distance to the nearest anchor in the scaled operating plane and the local linearization residual. Reject or flag queries outside a validated radius. This turns the implicit “table clamps” behavior into an explicit validity domain.

**Expected effect:** the largest reduction in off-trajectory error; directly addresses the IRFS 33 % held-out result. **Effort:** high (multiple golden DPT captures). **Risk:** the vendor model’s VM-02/VM-10 defects may limit some conditions; those must be recorded as N/A, not imputed.

#### I2. Select cells and anchors by the emitted model, not the analytic blend

Replace the `cand.predict` static RMSE selection with `emitted_static_rmse` plus a held-out DPT error. The emitted model is the object that ships; selecting on a different object is a known source of the EPC 2.37 → 7.22 % gap [P: `build_model` vs `emitted_static_rmse`].

**Expected effect:** closes the emission gap; likely 1–3 pp on EPC and IRFS. **Effort:** low. **Risk:** low; it only changes the selection criterion.

#### I3. Use a-posteriori error estimation / adaptive sampling

The DEIM literature provides an a-posteriori error estimator that is decomposed offline/online and obtained by a one-dimensional auxiliary ODE [L: Wirtz, Sorensen & Haasdonk 2014]. PieceSpice currently has no error estimator; it has a fixed `s_grid` and a farthest-point completion.

A practical version: hold out a portion of the trajectory, compute the reduced-model residual on it, and add anchors where the residual is largest. This is standard adaptive sampling and is consistent with the DEIM error-bound literature [L: Chaturantabut & Sorensen 2010; Wirtz et al. 2014].

**Expected effect:** better anchor placement at the same cell count, or fewer cells for the same error. **Effort:** medium. **Risk:** the terminal model is not a state-space reduced model; the estimator must be adapted to the terminal residual, not copied literally.

#### I4. Fit channel and charge jointly with charge conservation

Formulate one residual over the DPT samples:

$$
r(t) = i_\text{meas}(t) - \left[i_\text{ch}(V_{gs},V_{ds}) + \frac{d}{dt}Q_d(V_{gs},V_{ds})\right],
$$

plus the gate-charge residual

$$
r_g(t) = i_{g,\text{meas}}(t) - \frac{d}{dt}Q_g(V_{gs},V_{ds}),
$$

with monotonicity constraints on `Q_d` and `Q_g`. This follows the charge-oriented model of Ward & Dutton and the charge-conservation analysis of Yang, Epler & Chatterjee [L: Ward & Dutton 1978; Yang et al. 1983]. It also avoids the sequential-fit inconsistency in R4.

**Expected effect:** better consistency between the channel and the charge; likely reduces the Id DPT error and the edge-energy error. **Effort:** high (nonlinear joint optimization). **Risk:** identifiability; need regularization and held-out validation.

#### I5. Reintroduce C_QD and C_QG with passivity/monotonicity constraints

The reason `C_QD` was omitted is physical: fitted drain-charge slopes were negative at the blocking point and caused an exponential blow-up [P: module docstring]. That is a constraint violation, not a reason to omit the element. A passivity-preserving fit (PRIMA-style projection or constrained least squares) can enforce non-negative capacitance and monotone charge [L: Odabasioglu, Celik & Pileggi 1998]. The Qg experiment was closed because the emitted forms timed out; a lower-order, monotone Qg table with a stable integration form is the literature-consistent alternative [L: Ward & Dutton 1978; Yang et al. 1983].

**Expected effect:** the Miller plateau and gate current become representable; edge timing and e_on/e_off should improve materially. **Effort:** high. **Risk:** reintroduces stiffness; must be gated by the same completion guard.

#### I6. Add body diode and parasitics

Reverse conduction and package parasitics are part of the device terminal behavior [L: Mantooth et al. 2015; DeBoi et al. 2024]. The Tier-2 emitter already has a body diode and `RG/RD/RS`; Tier-3 v1 omits them. A `v1` variant that reuses the Tier-2 diode/parasitic block but keeps the TPWL channel/Qoss is the lowest-risk way to test the effect.

**Expected effect:** better reverse conduction and gate-loop response; likely reduces edge timing error. **Effort:** medium. **Risk:** additional nodes can slow the solver; measure both fidelity and completion.

#### I7. Make the charge tables two-dimensional and temperature-aware

Replace `Qoss(Vds)` with `Qoss(Vds,Vgs)` (or at least `Qoss(Vds)` plus a Vgs-dependent gate charge), and add a temperature axis interpolated from measured or datasheet data. The Wolfspeed model uses `CGD(VGD)`, `CDS(VDS)`, `CGS(VGS)`, with temperature dependence via lookup tables and a thermal network [L: DeBoi et al. 2024]. The ANN literature uses separate IV and QV networks with all terminal charges [L: Tung et al. 2024].

**Expected effect:** better large-signal charge and switching energy. **Effort:** high (data collection). **Risk:** table size and evaluation cost; use vector fitting or a compact rational form [L: Gustavsen & Semlyen 1999] to keep the expression small.

#### I8. Emission-aware grid optimization

Optimize the table grid for the emitted model, not just the fit: denser near the knee and the low-Vds region, coarser in the saturated plateau, with continuity constraints at cell boundaries. The current per-vendor grids are fixed constants [P: `_POLICY_BY_VENDOR`]. The EPC gap and the IRFS edge errors are the targets.

**Expected effect:** closes part of the fit-to-emission gap without changing the model form. **Effort:** medium. **Risk:** table size vs evaluation cost; measure both.

#### I9. Reproduce the TPWL speed mechanism or stop claiming it

If speed is a goal, the emitted object must be a reduced state-space model with a small number of states and piecewise-linear matrices, not a dense terminal table. The paper’s own results show `O(s q^2)` online cost and a small reduced model [L: Rewienski & White 2003, Table 1]. Implementing that in a standard SPICE subcircuit is non-trivial: it requires behavioral state equations (`ddt()`), piecewise gains, and a stable initialization. A cheaper alternative is to accept that the current terminal table is a fidelity model and drop the speed claim for `v1`; the record already says the forced row is a fidelity control, not a speed claim.

**Expected effect:** either a real speed path or an honest scope. **Effort:** very high (new emitter architecture). **Risk:** convergence and portability; the current `table()` form is the only measured-complete form.

#### I10. Add statistical validation and a true hold-out

Use blocked cross-validation by operating condition: train on some `Vbus`/`Iload`/`T` combinations, test on held-out combinations. Report mean and worst-case error with confidence intervals. The computer-experiment literature provides the design and uncertainty framework [L: Sacks et al. 1989; Kennedy & O'Hagan 2001; JCGM 100:2008].

**Expected effect:** converts the current near-training error into a defensible generalization estimate. **Effort:** medium-high (more simulations). **Risk:** vendor defects may make some conditions infeasible; record them as N/A.

### 3.5 Acceptance criteria for a repaired `v1`

A repaired `v1` should not be accepted on the current single-condition DPT alone. Minimum:

1. **Coverage:** at least two `Vbus`, two `Iload`, and two temperatures, with the training/test split declared before fitting.
2. **Fidelity:** held-out DPT Vds and Id errors reported separately, with worst-case and median; edge `tr/tf/e_on/e_off` reported; no N/A silently converted to pass.
3. **Emission:** selection by emitted-model error, not analytic `predict`.
4. **Physical guards:** Qd/Qg monotone and zero-anchored; `gm ≥ 0`, `gds ≥ 0` (or a documented exception); body diode present when reverse conduction is exercised.
5. **Speed:** if claimed, measured against golden on the same method, with a confidence interval and a declared effect-size threshold; otherwise no speed claim.

---

## 4. Tier-3 `simplis` (SIMPLIS-style Level-2 charge tables)

### 4.1 What is actually implemented

`pipeline/tier3_simplis.py::run_vendor` reuses the closed Tier-2 channel, body diode, and parasitics, and splices in three charge tables:

- `CGS = Ciss − Crss`, emitted on `V(gi,ss)`;
- `CDS = Coss − Crss`, emitted on `V(di,ss)`;
- `CDG = Crss`, emitted on `V(di,gi)`;

plus the documented 10 MΩ `RGS`. The tables are fitted to a 0.5 % charge budget with a greedy farthest-point rule and forced on-state breakpoint; `CGS` is emitted as a linear chord unless a measured nonlinearity test says PWL, and a PWL CGS on the Vds axis is rejected [P: `models/simplis_netlist_builder.py`; `pipeline/tier3_simplis.py`].

### 4.2 Measured results

| Vendor | Static | DPT Vds / Id | Charge mode | Channel | Adaptive solver / wall |
|---|---:|---:|---|---|---:|
| IRFS | 7.0683 % | 8.7044 / 8.5167 % | `tier2` | `tier2` | 1.10032× / 1.00922× |
| TI | 8.4935 % | 5.1405 / 4.2160 % | `tables` | `tier2` | 5.52744× / 3.49078× |
| EPC | 2.0394 % | 3.0315 / 1.7744 % | `tables` | `tier2` | 1.07080× / 1.03305× |
| CPM2 | 3.4287 % | 3.8506 / 4.6150 % | `tables` | `tier2` | N/A (VM-10) |

Source: `reports/comparison/<vendor>/tier3_validation_simplis.json`. Static parity with Tier-2 is within ±0.0002 pp; the charge tables do not change the static DC result because capacitors are open at DC.

### 4.3 Root causes

#### S1. All C–V tables come from a Vgs = 0 small-signal sweep

The capacitance bench measures Ciss/Coss/Crss vs Vds at 1 MHz with the gate at 0 V (or a 100 MΩ bleed). The decomposition `CGS=Ciss−Crss`, `CDS=Coss−Crss`, `CDG=Crss` is therefore valid at Vgs = 0 [P: `simulation/testbenches/capacitance.py`; `_decompose_cv`]. During switching, Vgs is at the drive level and Vdg is not Vds; `CDG` is emitted on `V(di,gi)`, but the table was measured at `V(di,gi) = Vds` because Vgs = 0 [P: emitter comment]. This is a physical axis mismatch, not a fitting error.

SIMPLIS’s own documentation says the extraction test conditions should be changed to reflect the expected voltage, current, and temperature of the device in the circuit, and that Level 2 typically uses a linear CGS with four-segment CDG/CDS [L: SIMPLIS MOSFET Models]. PieceSpice uses one Vgs = 0 sweep and a variable segment count, so it captures the Level-2 *form* but not the Level-2 *bias dependence*.

#### S2. CGS is forced linear on the Vds axis

`cgs_linearity_decision` compares the chord deviation of `Qiss(Vds)` to the 0.5 % budget. If the chord is within tolerance, a linear capacitor is emitted; if not, a PWL CGS is rejected unless a genuine Vgs-axis table is supplied [P: `models/simplis_netlist_builder.py`]. On the current data, CGS resolves linear on 4/4 [P: `tier3_validation_simplis.json#algorithm.cgs_decision`]. That is consistent with SIMPLIS’s “CGS is typically linear” statement, but it also means the model has no gate-voltage dependence in Cgs, which is a known simplification [L: DeBoi et al. 2024].

#### S3. The charge fit criterion is charge error, not capacitance error

The greedy fit minimizes the max absolute error of `Q(V)` relative to `Q(Vmax)` (0.5 % budget). A monotone Q table can meet that budget while the derivative `C=dQ/dV` has visible steps or local errors, especially at low Vds where Q is small. The record reports `max_abs_charge_error_c`, not a C–V error [P: `_fit_record`; `function_fit`]. SIMPLIS documentation explicitly notes that PWL capacitance shows steps while the charge integral matches [L: SIMPLIS PWL Capacitors]. That is acceptable for charge-conserving simulation, but it means the 0.5 % number is not a capacitance-accuracy claim.

#### S4. Linear extrapolation above the last breakpoint

The emitted `Q=` chain uses delta slopes and no `table()` end clamp; above the last breakpoint it extrapolates with the last segment slope [P: emitter comment]. The fit forces no breakpoint above `Vbus`, so a DPT overshoot above `Vbus` uses extrapolated charge. The literature on PWL capacitors emphasizes charge continuity, not extrapolation; a clamp or a final saturation segment is the safer construction [L: SIMPLIS PWL Capacitors].

#### S5. No temperature, no gate-resistance dependence, no body-diode change

The variant inherits Tier-2’s 25 °C channel/diode/parasitics and adds charge tables at 25 °C. It does not address the temperature or gate-drive dimensions that the Wolfspeed study shows to be first-order [L: DeBoi et al. 2024].

#### S6. The speed result is golden-stiffness-driven

TI’s 5.527× solver result comes from a golden with 131,558 iterations versus the ROM’s 12,073 [P: `tier3_validation_simplis.json#speed_adaptive_20`]. Every ROM, including one with no simplis table content, lands in the same ~0.30–0.33 s band on TI; the golden is the stiff side [P: `docs/plans/tier3_method_candidates.md` §16.5]. IRFS and EPC are ~1.07–1.10×, and CPM2 is N/A. The result is a condition-specific row, not a portable SIMPLIS speed class.

### 4.4 Improvement program

#### S-I1. Multi-bias C–V extraction

Measure Ciss/Coss/Crss at several Vgs biases (e.g. 0, threshold, drive) and, where possible, with a pulsed bias that keeps the device in the desired region. Use the resulting 2D tables for `CDG(Vgs,Vdg)` and `CDS(Vgs,Vds)`. This follows the SIMPLIS extraction guidance and the Wolfspeed capacitance-dependence evidence [L: SIMPLIS MOSFET Models; DeBoi et al. 2024].

**Expected effect:** removes the largest physical axis error in S1; likely improves Vds edge and Id overshoot. **Effort:** high (new extraction). **Risk:** device self-heating and bias-hold; document the conditions.

#### S-I2. Separate CGS treatment

Keep CGS linear only when a genuine Vgs-axis measurement supports it; otherwise use a 2D CGS table. The current fail-closed `cgs_axis` rule is correct; the missing piece is the Vgs-axis data [P: emitter `cgs_axis`; L: SIMPLIS Level 2].

#### S-I3. Gate on C–V error as well as Q error

Report both `max |Q−Q_fit| / Q_max` and `max |C−C_fit| / C_max` (or a capacitance-weighted error), and use both in the selection. A 0.5 % charge budget alone can hide a large local capacitance error [P: `_fit_anchored_charge`; L: SIMPLIS PWL Capacitors].

#### S-I4. Replace linear extrapolation with a physical end segment

Add a final saturation segment or clamp at `Vbus`/`Vmax`; do not let the last slope extrapolate indefinitely. This is a small emitter change with a clear physical rationale [L: SIMPLIS PWL Capacitors; P: emitter comment].

#### S-I5. Add temperature and gate-drive dimensions

At minimum, run the charge extraction and DPT at two temperatures and two gate drives; fit the temperature dependence with a linear or quadratic table, as in the Drofenik/Kolar loss scheme and the Wolfspeed model [L: Drofenik & Kolar 2005; DeBoi et al. 2024].

#### S-I6. Improve the channel only with evidence

The channel is currently `tier2` on 4/4. The hybrid `hinge_c` channel was rejected on the strict all-metric DPT rule; the ANN residual was also rejected [P: `tier3_validation_simplis.json#channel_candidates`]. Do not reopen the channel without a new, independently validated candidate. The charge side is the higher-value target.

#### S-I7. Keep the speed claim scoped

The only ≥5× result is TI and is golden-driven. A portable speed claim requires a non-stiff golden and a measured effect size with uncertainty. If the goal is converter-level speed, use Tier-4 (ideal switch) and accept its retired waveform fidelity; do not attribute SIMPLIS simulator speed to an LTspice `.lib` [L: SIMPLIS “What is SIMPLIS”; Plexim PLECS Spice docs].

---

## 5. UROM (unified one-ROM)

### 5.1 What it is

`pipeline/unified_rom.py::run_unified` builds one emitted library, runs an equal-sandbox DPT and a 20-pulse adaptive speed bench, computes G0–G6, and writes a record. The canonical default is `emit="tier2"`, i.e. the closed Tier-2 emitter renamed to `<vendor>_PIECSPICE_UROM.lib`; the keyed `ann` and `hybrid` bases are non-default candidates [P: `run_unified`; `_build_emitted`].

The canonical record is accepted 4/4 under the final bars:

| Vendor | Verdict | Static | G3 Vds / Id vs bar | Solver / wall |
|---|---|---:|---|---:|
| IRFS | accept | 7.0683 % | 9.9993 / 0.5751 vs 10.0 / 0.60 | 1.05625× / 1.02859× |
| TI | accept | 8.4935 % | 1.9838 / 0.0448 vs 5.583 / 0.048 | 5.39601× / 3.46991× |
| EPC | accept | 2.0394 % | 0.7042 / 0.0464 vs 4.894 / 0.049 | 0.645733× / 0.867339× |
| CPM2 | accept | 3.4287 % | 2.8933 / 0.2152 vs 3.0 / 0.261 | 1.10127× / 1.09625× |

Source: `reports/comparison/<vendor>/unified_validation.json`.

### 5.2 Gate-by-gate audit

| Gate | What it actually measures | What it misses | Assessment |
|---|---|---|---|
| G0 equal sandbox | Deck-text equality after stripping the DUT X-line and Tj source; explicit `.options` equality; `.temp` equality; tolerance regex; 3 repeats; clock spread | LTspice version/OS/CPU/thread count; default options (only explicit `.options` compared); statistical power; warm-up; CPU load | Good foundation; incomplete environment record |
| G1 static | Mean of per-curve peak-normalised rel RMSE; endpoint | Worst curve; per-curve max; application-derived target; the bar is the Tier-2 measured value | Regression gate, not absolute fidelity |
| G2 derivative | Finite gm/gds; Qoss monotone; Vds=0 and cutoff anchors; ANN probe | gm/gds sign (fractions are diagnostics only); Qiss/Qrss monotonicity; passivity; total charge conservation | **Gate gap** |
| G3 DPT waveform | `I(Lload)` on a common 7000-point grid; per-vendor bars | Device current; time alignment; edge-specific metrics; statistical uncertainty; bars partly set at measured values | **Metric mismatch + knife-edge** |
| G4 timing/energy | Six metrics within ±5 % of a bar column | For C0 the bar is `tier2_metrics`, i.e. the same model family; near-zero relative bands; no uncertainty | **Self-consistency for C0** |
| G5 speed | Counts present, completion, clock spread | No minimum speedup; EPC 0.6457× passes | **Not a speed gate** |
| G6 charge | Numeric mirror of emitted Q ramp vs source table; Qoss monotone/zero; 0.5 % budget | Qiss/Qrss monotonicity; physical charge accuracy; C–V error; temperature | Emission-fidelity gate, not physical-charge gate |

#### F1 in detail: G5 passes a slower model

`_evaluate_gates` computes `g5_pass = has_counts and clock_ok and golden.completion and urom.completion`. There is no comparison of `solver_speedup` or `wall_speedup` to 1, let alone to a minimum effect size [P: `_evaluate_gates`]. EPC’s canonical record therefore passes G5 at 0.645733× solver / 0.867339× wall [P: `unified_validation.json#speed_adaptive_20`]. A reader who sees “G5 pass” and “accept” can reasonably infer a speed result that the data contradicts. This is the most serious semantic defect in the UROM record.

**Correction:** split the verdict into `fidelity_verdict` and `speed_verdict`. G5 should pass only if the measured ratio exceeds a declared threshold with a confidence interval; otherwise it should be `measured_slower` / `not_claimed`, not `pass`.

#### F2 in detail: G3 current definition

`_run_dpt_arm` constructs `DoublePulseRun.i_d` from `np.abs(result.get("I(Lload)"))` [P: `pipeline/unified_rom.py`]. Tier-3 `v1` and `simplis` use `_device_current`, which returns `I(Lload)−I(Dclamp)` when the clamp trace exists [P: `pipeline/tier3.py::_device_current`]. The UROM G3 Id numbers (e.g. TI 0.0448 %) are therefore on the inductor-current metric; the Tier-3 Id numbers (e.g. TI 4.8962 %) are on the device-current metric. They are not comparable, and the UROM number is the more forgiving one because it hides clamp current during overshoot. The project’s own Tier-3 record documents the metric correction [P: `docs/plans/tier3_method_candidates.md` §18.3].

**Correction:** use `I(Lload)−I(Dclamp)` in `_run_dpt_arm` for all bases and re-baseline the G3 bars and the G4 timing/energy table. Until then, no cross-tier Id comparison is valid.

#### F3 in detail: G4 is self-consistency for C0

`TIMING_BAR_COLUMN = {"ann": "golden_metrics", "hybrid": "golden_metrics", "tier2": "tier2_metrics"}` [P: `pipeline/unified_rom.py`]. `_timing_bars_for_column` reads `reports/comparison/<vendor>/dpt_metrics.json`. For `emit="tier2"`, the emitted model is the same Tier-2 model that produced `tier2_metrics`; the 5 % band is therefore a reproducibility check, not an independent fidelity test. The record labels the bar source honestly, but the gate name “timing/energy” and the overall verdict “accept” can be over-read.

**Correction:** for every basis, score G4 against the golden DPT (or an independent measured reference) and label the Tier-2 comparison as a regression check. If the goal is to accept the Tier-2 model, that is a legitimate separate claim; it should not be presented as a new-ROM validation.

#### F11 in detail: G2 and G6 gaps

`_derivative_checks` computes `gm_nonnegative_fraction` and `gds_nonnegative_fraction` but only reports them; the pass predicate is finite gm/gds, Qoss monotone, Vds=0 anchor, cutoff anchor [P: `_derivative_checks`]. A model with negative differential conductance can pass G2. The monotone-neural-network literature treats monotonicity as a physical constraint, not a diagnostic [L: Daniels & Velikova 2010; Gong et al. 2026].

`_charge_checks` checks per-table ramp error but `G6` uses `charge["monotone_q"]` and `q_zero_anchored`, which are computed from `spec.q_oss_charge_c` only [P: `_charge_checks`; `_evaluate_gates`]. Qiss and Qrss monotonicity are not gated. The ANN emitter’s `delta_ramp_chain` does not reject negative slopes; the simplis emitter does (`_pairs`), but the ANN path does not [P: `models/ann_netlist_builder.py`; `models/simplis_netlist_builder.py::_pairs`].

**Correction:** gate `min(dQ_oss/dV) ≥ 0`, `min(dQ_iss/dV) ≥ 0`, `min(dQ_rss/dV) ≥ 0`, and `gm ≥ 0`, `gds ≥ 0` (or a documented physical exception). Add a charge-conservation check across the three terminal charges.

#### F13 in detail: circular bars

`ENDPOINT_BAR_PCT = 4.13` is described as “measured worst endpoint” [P: constant provenance]. A bar set at the measured worst value cannot distinguish a good model from the model that defined the bar. The IRFS G3 bar 10.0 % is passed at 9.9993 %; the margin is 0.0007 pp, which is smaller than the repeat-to-repeat variation of a waveform RMSE. The CPM2 bar 3.0 % is passed at 2.8933 %. These are knife-edge acceptances.

**Correction:** derive bars from application requirements or measurement uncertainty, not from the model’s own measured values. Report the margin and its uncertainty; do not treat a 0.0007 pp margin as evidence of robustness.

### 5.3 What the UROM acceptance does and does not show

It shows:

- the closed Tier-2 emitter can be reproduced under a renamed subcircuit with hash-identical identity;
- the equal-sandbox harness can run both arms on the same deck/options/temperature and complete the 20-pulse bench;
- the canonical C0 record is internally consistent with the Tier-2 bars.

It does not show:

- that a new reduced-order model form has been validated;
- that the ANN or hybrid bases work (both reject 4/4);
- that the model is faster than golden in a portable sense (EPC is slower; TI is golden-driven);
- that the Id metric is comparable with Tier-3;
- that the model generalizes beyond the one DPT condition.

### 5.4 Improvement program

1. **Unify the current definition** (P0). Use `I(Lload)−I(Dclamp)` in `_run_dpt_arm`; re-baseline G3 bars and G4 rows. Re-run all four records.
2. **Split fidelity and speed verdicts** (P0). Add `fidelity_verdict` and `speed_verdict`; G5 passes only on a declared speed effect with CI; otherwise `not_claimed`.
3. **Make G4 independent** (P0). Score every basis against the golden DPT; keep the Tier-2 comparison as a labeled regression check.
4. **Close G2/G6 gaps** (P0). Gate gm/gds sign and all three charge monotonicities; add a charge-conservation check.
5. **Derive bars from requirements/uncertainty** (P1). Replace the measured-value bars with application targets or uncertainty-derived bands; report margin and CI.
6. **Add statistical validation** (P1). More repeats, bootstrap CI, environment metadata (LTspice version, OS, CPU, threads), and a paired test across repeats. The GUM framework and computer-experiment literature provide the vocabulary [L: JCGM 100:2008; Sacks et al. 1989].
7. **Add input hashes/manifest** (P2). Hash the vendor lib, YAML configs, extracted curves, and emitted lib; include the hashes in the record. This closes the stale-artifact class.
8. **Define UROM as a harness or require a non-C0 pass** (P2). If UROM is a model family, require at least one non-C0 basis to pass before claiming a new model. If it is a harness, rename the claim accordingly.

---

## 6. ANN and hybrid bases

### 6.1 Channel model

The ANN channel is a one-hidden-layer network with at most 10 neurons and ISRU/tanh/sigmoid activation [P: `approximation/ann_channel.py`]. The emitted form is

$$
I_\text{raw} = V_{ds}\exp(y_1),
\qquad
y_1 = b_2 + \sum_i w_{2i}\phi(w_{1i0}V_{gs}+w_{1i1}V_{ds}+b_{1i}),
$$

with a tanh Vth cutoff and a low-Vds blend between `y_1(Vds)` and `y_1(Vsat)` [P: `models/ann_netlist_builder.py::emit_ann_channel`]. The fit uses analytic first/second derivatives via a Jet2 autodiff class, with loss weights on Id, gm, gds, gm′, gds′ [P: `ann_channel.py`].

The source paper uses a hybrid ANN + behavioral cutoff and log-scale C–V networks, validated against DPT measurements, and reports 30 % less computation than the Angelov model [L: Yang et al. 2022]. The Tung paper uses separate IV and QV networks with all terminal currents and charges, studies activation functions, and reports ~40× speed versus a conventional compact model in large circuits [L: Tung et al. 2024]. The Gong paper adds monotonicity/symmetry constraints and reports a 3.2× mean-relative-error reduction and up to 70 % transient-time reduction versus ANN models [L: Gong et al. 2026]. The 2026 review emphasizes dataset selection, model parameters, and performance metrics [L: Cho et al. 2026].

PieceSpice’s ANN channel captures the cutoff and zero-Vds guards, but not the physics constraints that these papers identify as the difference between a data fit and a compact model.

### 6.2 Charge model: the fitted network is not emitted

`fit_charge_qv` fits `log(Q+q0)` with analytic `dQ/dV` and `d²Q/dV²` residuals and stores the coefficients in `AnnChargeFit.coefficients` [P: `approximation/ann_charge.py::_fit_one`]. But the emitted tables are built by

```python
tables[name] = _sample_table(x, q)
```

where `x, q` are the **measured** charge data, not the fitted network [P: `fit_charge_qv`]. `_sample_table` interpolates the measured `q`, applies `maximum.accumulate`, and sets `q[0]=0`. The emitter then calls `delta_ramp_chain` on `charge_fit.tables`, i.e. on the resampled measured data [P: `models/ann_netlist_builder.py::_charge_lines`].

**Consequence:** the ANN charge network is dead weight. The ANN basis is effectively an ANN channel plus the measured Q tables. The reported charge-fit diagnostics (optimizer cost, q0 offsets) describe a network that never reaches the netlist. This is a concrete correctness defect, not a modeling choice.

**Correction:** either emit the fitted `Q(v)` (after a monotone projection) and validate it, or remove the charge network and document that the ANN basis uses measured Q tables. The former is the research direction; the latter is the honest short-term fix.

### 6.3 Residual/hybrid basis

`fit_channel_residual` fits `I_res = I_golden − I_C0` as

$$
I_\text{res} = V_{ds}\left(b_2+\sum_i w_{2i}\phi(w_{1i0}V_{gs}+w_{1i1}V_{ds}+b_{1i})\right)\cdot\frac{1}{2}\left[1+\tanh(\alpha(V_{gs}-V_{th}))\right],
$$

with signed weights [P: `approximation/ann_residual.py`]. The `Vds` factor pins zero at Vds=0; the tanh cutoff suppresses sub-threshold correction; the residual can be signed. This is the right structural idea: keep the physical C0 base and learn only the residual, as in the source-backed hybrid framework [L: Yang et al. 2022; Gong et al. 2026].

But the fit uses all supplied samples; there is no blocked hold-out or independent-condition test [P: `fit_channel_residual`]. The later selection gate rejects the hybrid 4/4 because it does not strictly improve every DPT metric [P: `tier3_validation_simplis.json#channel_candidates`]. That is the correct outcome given the evidence, but it does not tell us whether the residual form is capable of improvement; it tells us the current fit/validation did not demonstrate it.

### 6.4 Error sources

| ID | Error | Evidence | Consequence |
|---|---|---|---|
| A1 | Charge net fitted but not emitted | `fit_charge_qv` / `_sample_table` | The ANN charge claim is not realized |
| A2 | No monotonicity/passivity constraint on the channel | `fit_channel_iv` loss is data/derivative, not constrained | Negative gm/gds possible; G2 does not gate sign |
| A3 | `exp(y)` unbounded | `emit_ann_channel` | Overflow / convergence risk outside training range |
| A4 | No temperature input | All ANN records 25 °C | Fails the first-order temperature dependence |
| A5 | No reverse conduction/body diode in the ANN channel | `G_CH` only; body diode from Tier-2 | Reverse conduction still Tier-2 |
| A6 | No blocked hold-out | `fit_channel_iv` uses all samples | No generalization estimate |
| A7 | Training data from one DPT + Curve-1 | `ann_fit_for_vendor` | Same coverage problem as Tier-3 |
| A8 | `out_of_range_check` is a range check, not an accuracy estimate | `AnnChannelFit.out_of_range_check` | May be over-read as validation |
| A9 | G2 ANN probe can pass while gm/gds are negative | `_derivative_checks` | Physical inconsistency not gated |
| A10 | Fixed cutoff `alpha=10` | `_CUTOFF_ALPHA` | Subthreshold shape not fitted |

### 6.5 Improvement program

1. **Emit or remove the charge network** (P0). If emitted, enforce `Q(0)=0`, monotone `Q`, non-negative `C`, and validate `C(V)` against the measured sweep. If not, delete the dead fit and relabel the ANN basis.
2. **Physics-constrained architecture** (P1). Use monotone/partially monotone networks or constrained weights [L: Daniels & Velikova 2010; Runje & Shankaranarayana 2023], plus a physics-informed loss on `∂I/∂Vds ≥ 0`, `∂I/∂Vgs ≥ 0`, and charge conservation [L: Raissi et al. 2019; Gong et al. 2026].
3. **Bound the output transform** (P1). Replace `exp(y)` with a bounded positive transform (softplus/sigmoid scaled) or clamp `y` to a physical range. The ANN literature uses log-scale targets and behavioral guards specifically to avoid this [L: Yang et al. 2022].
4. **Blocked cross-validation by operating condition** (P1). Hold out entire `Vbus`/`Iload`/`T` conditions; report held-out error. Use the computer-experiment design/uncertainty framework [L: Sacks et al. 1989; Kennedy & O'Hagan 2001].
5. **Add temperature and gate-drive inputs** (P1). A network with `(Vgs,Vds,T)` inputs and separate IV/QV heads is the source-backed direction [L: Tung et al. 2024; Gong et al. 2026; DeBoi et al. 2024].
6. **Uncertainty quantification** (P2). Ensemble or Bayesian last layer; report prediction intervals and out-of-distribution flags. The metrology framework provides the reporting vocabulary [L: JCGM 100:2008].
7. **Deployment checks** (P2). Verify LTspice primitive support (`exp`, `tanh`, `sqrt`), expression length, and convergence at the training-range corners; Tung’s direct-multiplication Verilog-A approach is the efficient deployment reference [L: Tung et al. 2024].

---

## 7. Tier-4 (ideal switch + loss LUT)

### 7.1 What is implemented

`models/tier4_ideal_switch_builder.py` emits:

```spice
.SUBCKT <vendor>_PIECESPICE_T4 d g s
S_SW d s g s SW_T4
.MODEL SW_T4 SW(Ron=<Curve-3>, Roff=<measured Vgs=0 | 1e12>, Vt=<Curve-7 Vth>, Vh=0)
C_QOSS d s <measured linear Coss at cap_ref_Vds>
* loss tables as comments
.ENDS
```

The loss LUT (`e_on`, `e_off`, `r_on`, `v_on`) is metadata only; no active loss element or thermal network [P: emitter; `pipeline/tier4.py`].

### 7.2 Measured results

| Vendor | Solver / wall | Golden → T4 iterations | Energy status |
|---|---:|---:|---|
| IRFS | 1.1896× / 1.0952× | 12,718 → 6,165 | self-consistency, 0 % |
| TI | 6.3540× / 3.3328× | 131,558 → 6,201 | self-consistency, 0 % |
| EPC | 1.2474× / 1.0608× | 16,514 → 6,296 | self-consistency, 0 % |
| CPM2 | N/A | N/A | self-consistency, 0 % |

DPT completion is 4/4; the waveform gate is `retired_by_design`; efficiency is `output_power_required`; the LUT has one `constant_node` (IRFS, TI), three `grid_interpolation` nodes (EPC), one `single_node` (CPM2), and zero complete 2×2 cells [P: `tier4_validation.json`].

### 7.3 Error sources

#### T1. The LUT is a single-point table, not a lookup table

The grid is derived from the config, which specifies one `Vbus` and one `Iload` per vendor. The resulting table has 1–3 finite nodes and no complete cell; off-node queries return NaN [P: `approximation/loss_lut.py::energy_at`; `tier4_validation.json#lut.interpolation`]. PLECS’s thermal documentation describes a genuine 2D/3D lookup table with linear interpolation inside the grid and extrapolation outside [L: Plexim PLECS 5.0 Thermal Concepts]. PieceSpice has neither the grid nor the interpolation in practice.

**Consequence:** the “LUT” cannot answer a different bus voltage or load current; it is a measured operating-point record. A converter design that sweeps `Vbus`/`Iload` cannot use it.

#### T2. The energy gate is self-consistency

`_energy_comparison` fills the LUT from `dpt_metrics.json#golden_metrics` and then compares the LUT value to the same `golden_metrics`. The relative error is 0 % by construction; the record explicitly labels it `source_self_consistency` / `independent_energy_gate: false` [P: `pipeline/tier4.py::_energy_comparison`]. This is honest, but it is not a validation of the loss model. The project’s own README says the energy rows are source self-consistency; the research finding is that this is the only energy evidence, so no independent claim is possible.

#### T3. No temperature axis or thermal network

The LUT is at `conditions.temperature` (25 °C) and has no `Tj` axis; there is no Cauer/Foster network and no junction-temperature feedback [P: records; emitter]. Drofenik & Kolar show that conduction and switching losses are temperature-dependent, fit them with temperature-dependent polynomials, and inject them into a thermal RC network to compute `Tj(t)` [L: Drofenik & Kolar 2005]. PLECS’s thermal description uses `E_on(v_block,i_on,T)`, `E_off(...)`, and `v_on(i,T)` with a thermal network [L: Plexim PLECS 5.0 Thermal Concepts]. Tier-4 has none of these dimensions.

#### T4. Efficiency is N/A because there is no output-power definition

`efficiency_at` returns `output_power_required` when `p_out_w` is None; `Vbus·Iload` is bus input power, not converter output power [P: `loss_lut.py::efficiency_at`]. This is the correct behavior per PLECS’s warning that thermal losses are not the electrical power consumed by the device and that efficiency needs an explicit output-power definition [L: Plexim PLECS 5.0 Thermal Concepts]. The improvement is not to fabricate an efficiency; it is to build a converter bench with a defined `Pout` (M5).

#### T5. Ron is extracted at `Vgs_fixed`, not at the DPT gate drive

`_r_on_from_curve` uses `conditions.Vgs_fixed` as the Curve-3 gate drive; CPM2 extracts at 20 V but switches at 18 V [P: `_curve_gate_drive_v` docstring; `config/inputs/cpm2-1200.yaml`]. The emitted Ron is therefore not the Ron at the switching gate drive. The record discloses this, but the conduction loss is biased.

#### T6. Vt is the JEDEC threshold, not the switching threshold

`Vt` comes from Curve-7 `vth_v`, a low-current JEDEC threshold [P: emitter]. An ideal switch’s threshold should represent the gate voltage at which the device transitions under the actual load; the JEDEC Vth can be significantly lower than the Miller plateau / effective turn-on. SIMPLIS’s Level-2 model uses `VT0` plus hysteresis and a forward-gain segment; the extraction conditions are meant to reflect the application [L: SIMPLIS MOSFET Models]. Using Curve-7 Vth without calibration is a timing/duty-cycle error.

#### T7. Linear Coss is a solver stabilizer, not a fidelity model

The emitter uses one measured linear Coss at `cap_ref_Vds`; it slows dV/dt and adds L–C overshoot, and the waveform gate is retired [P: emitter docstring; `tier4_validation.json#solver_stabilization`]. This is disclosed and acceptable for a solver stabilizer, but it means the emitted model has no voltage-dependent output capacitance. A converter-level model that needs dV/dt/ringing must use a nonlinear charge model or a behavioral stage.

#### T8. No body diode, Miller plateau, or reverse recovery

The ideal switch has no body diode, no gate charge, and no reverse recovery [P: emitter]. PLECS’s own behavioral models include reverse recovery and limited di/dt for exactly these effects [L: Plexim “Power Semiconductors”]. Tier-4 explicitly retires them; that is a scope decision, not a fidelity result. For converter efficiency and dead-time loss, they matter.

#### T9. Speed is golden-driven

TI’s 6.354× is the same pattern as Tier-3: the golden takes 131,558 iterations; the ideal switch takes 6,201 [P: `tier4_validation.json#speedup`]. IRFS 1.1896×, EPC 1.2474×, CPM2 N/A. No portable speed class follows. The project says this; the research finding is that the only large number is a property of the TI golden, not of the ideal-switch tier.

### 7.4 Improvement program

1. **Build a real loss grid** (P1). Measure `e_on/e_off` at a `Vbus × Iload` grid (at least 3×3) and, where possible, two temperatures. Use linear interpolation inside the grid and a documented extrapolation policy; PLECS’s thermal docs are the reference [L: Plexim PLECS 5.0 Thermal Concepts].
2. **Add a temperature axis and thermal network** (P1). Fit `E_on(v,i,T)`, `E_off(v,i,T)`, `v_on(i,T)` and a Cauer/Foster network; compute `Tj(t)` from the loss. Drofenik & Kolar provide the scheme; PLECS provides the data-model convention [L: Drofenik & Kolar 2005; Plexim PLECS 5.0 Thermal Concepts].
3. **Add an active loss/thermal element or a converter post-processing path** (P1). The current comment-only LUT cannot affect the electrical solution. Either add a loss source/thermal network for converter-tier simulation, or keep the electrical model ideal and compute losses offline with a clearly labeled post-processor. Do not call the comment table an electrical loss model.
4. **Calibrate Ron and Vt at the actual gate drive** (P1). Extract Ron at `dpt.vgs_on` (or interpolate `Ron(Vgs)`), and calibrate Vt to the DPT gate plateau/turn-on rather than the JEDEC low-current Vth. Disclose the chosen definition.
5. **Add body diode and gate charge** (P1/P2). A converter-tier model needs reverse conduction and gate charge for dead-time and gate-drive loss. If the project keeps the ideal switch, state that these are retired and do not use the model for those observables.
6. **Build the converter bench (M5)** (P2). Define `Pout`, measure efficiency, and compare the LUT-based estimate to the measured converter efficiency. This is the only way to turn the energy self-consistency into an independent validation.
7. **Keep the speed claim scoped** (P0). The TI result is golden-driven; report per-vendor and do not generalize. If a portable speed claim is desired, benchmark on a non-stiff golden and report a confidence interval.

---

## 8. Cross-cutting error sources

### 8.1 Temperature is missing everywhere

Every record is at 25 °C. The literature is unambiguous that SiC/GaN device behavior, capacitances, threshold, and losses vary with temperature, and the Wolfspeed tuning procedure varies `Tj` explicitly [L: DeBoi et al. 2024; Mantooth et al. 2015; Drofenik & Kolar 2005]. A 25 °C-only model is a data slice, not a device model. The emitter metadata says this for Tier-4; the same disclosure should apply to Tier-3/UROM.

### 8.2 Gate resistance and gate charge are missing

Tier-3 `v1` has no `RG`, no `C_QG`, and no `C_QD`; Tier-4 has no gate charge at all. The Wolfspeed study identifies the gate-side capacitances and internal gate resistance as the flexible tuning parameters because they control transient behavior without changing conduction [L: DeBoi et al. 2024]. The measured edge errors (IRFS tr 0.836 vs 25.844 ns) are the expected consequence.

### 8.3 Validation statistics are thin

The speed benches use 3 repeats and report medians. There is no confidence interval, no paired test, no environment metadata, and no warm-up control [P: `_run_interleaved_speed`; `runtime.repeats=3`]. For a ratio near 1 (IRFS 1.056×, CPM2 1.101×), the repeat-to-repeat spread can be comparable to the effect. The GUM framework distinguishes Type A (statistical) and Type B (systematic) uncertainty and requires a combined uncertainty statement [L: JCGM 100:2008]. The computer-experiment literature adds design and prediction-uncertainty methods [L: Sacks et al. 1989; Kennedy & O'Hagan 2001].

### 8.4 Bars and metrics are not independent

The static bar is the measured Tier-2 value; the endpoint bar is the measured worst endpoint; the DPT bars are measured Tier-2/golden values with two documented relaxations. The G3 Id metric differs between UROM and Tier-3. The G4 bar for C0 is the Tier-2 table. These are regression/consistency checks, not independent validation. The project is honest about some of this in prose, but the gate names and the verdict “accept” do not encode the distinction.

### 8.5 Data provenance and stale artifacts

The records list input curve filenames but not hashes. `endpoint_comparison.json` is stale (2026-11-18) and contradicts the current full-sweep/UROM static records; `dpt_metrics.json` is a legacy input artifact not written by current production code, and its golden energies use the old `I(Lload)` metric [P: file timestamps; `rg` for writers]. UROM G4 still reads `dpt_metrics.json`. This is a reproducibility risk: a reader can cite a stale number that looks authoritative.

**Correction:** hash every input and emitted artifact, include the hashes in the record, and regenerate or delete stale comparison JSONs. Add a manifest that names the exact source of every bar.

### 8.6 Model-form theory is not matched to the emitted object

The TPWL and DEIM papers provide error bounds for reduced state equations and nonlinear-term interpolation [L: Rewienski & White 2003; Chaturantabut & Sorensen 2010; Wirtz et al. 2014]. PieceSpice’s `v1` is a terminal table fit, not a reduced state equation, so those bounds do not apply. The ANN papers provide compact-model constraints and deployment patterns, but PieceSpice’s ANN charge net is not emitted and its channel is not monotone. The Tier-4 design follows PLECS’s ideal-switch concept but not its thermal/loss data model. The names and the theory should be reconciled, or the claims should be narrowed.

---

## 9. Prioritized roadmap

### P0 — Metric integrity (do first; no new data required)

| Action | Why | Acceptance |
|---|---|---|
| P0.1 Unify current definition in UROM G3 | UROM uses `I(Lload)`; Tier-3 uses `I(Lload)−I(Dclamp)` | Same `current_definition` in all records; re-baselined G3/G4 |
| P0.2 Split fidelity and speed verdicts; fix G5 | EPC 0.6457× passes a “speed” gate | `speed_verdict` is `not_claimed` unless ratio > threshold with CI |
| P0.3 Make G4 independent for C0 | C0 G4 compares to Tier-2 bars | G4 scored against golden for every basis; Tier-2 comparison labeled regression |
| P0.4 Gate gm/gds sign and Qiss/Qrss monotonicity | G2/G6 gaps | Negative gm/gds or non-monotone charge fails |
| P0.5 Fix ANN charge emission or remove the net | Fitted network is dead | Emitted Q matches the fitted/validated table; or ANN record says measured Q |
| P0.6 Regenerate/delete stale comparison JSONs; hash inputs | Stale `endpoint_comparison`/`dpt_metrics` | Every cited number has a hash-matched source |

### P1 — Physical fidelity and generalization

| Action | Why | Acceptance |
|---|---|---|
| P1.1 Multi-condition DPT training + blocked CV (Tier-3) | Single-trajectory TPWL coverage limit | Held-out conditions reported; no in-sample-only claim |
| P1.2 Emitted-model-in-the-loop cell selection | EPC 2.37→7.22 % gap | Selection uses `emitted_static_rmse` + held-out DPT |
| P1.3 Add C_QD/C_QG/body diode/parasitics | Edge timing and energy errors | Edge metrics improve; all runs complete |
| P1.4 2D charge tables + temperature axis | C(V) depends on Vgs and T | Q(Vgs,Vds,T) table; charge monotone; T validated |
| P1.5 Multi-bias C–V extraction for simplis | Cgd axis mismatch at Vgs=0 | CDG measured/emitted on the correct bias; C–V error reported |
| P1.6 ANN monotone/physics constraints + held-out | No monotonicity/generalization | Constrained net; held-out error; G2 sign gate |
| P1.7 Tier-4 loss grid + T axis + thermal/converter bench | LUT is one point; energy self-consistency | ≥3×3 grid; T axis; efficiency with defined Pout |

### P2 — Performance and reproducibility

| Action | Why | Acceptance |
|---|---|---|
| P2.1 Reduce table evaluation cost / revisit state-space emitter | `v1` slower than golden | Measured per-iteration cost; speed claim only if effect > CI |
| P2.2 Statistical benchmark protocol | 3 repeats, no CI | ≥5 repeats or bootstrap CI; environment metadata |
| P2.3 Application-derived bars | Circular measured bars | Bars from requirements/uncertainty; margin reported |
| P2.4 Artifact manifest/signing | Reproducibility | Lib + record + inputs hashed and linked |
| P2.5 Clarify model-form names | TPWL/SIMPLIS/UROM overclaim | Names match the emitted object; scope statements updated |

---

## 10. Source ledger

### Peer-reviewed journals and conferences

1. M. Rewieński and J. White, “A trajectory piecewise-linear approach to model order reduction and fast simulation of nonlinear circuits and micromachined devices,” *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, vol. 22, no. 2, pp. 155–170, Feb. 2003. DOI: 10.1109/TCAD.2002.806601.
2. M. Rewieński and J. White, “Model order reduction for nonlinear dynamical systems based on trajectory piecewise-linear approximations,” *Linear Algebra and its Applications*, vol. 415, no. 2–3, pp. 426–454, 2006. DOI: 10.1016/j.laa.2003.11.034.
3. S. Chaturantabut and D. C. Sorensen, “Nonlinear model reduction via discrete empirical interpolation,” *SIAM Journal on Scientific Computing*, vol. 32, no. 5, pp. 2737–2764, 2010. DOI: 10.1137/090766498.
4. D. Wirtz, D. C. Sorensen, and B. Haasdonk, “A posteriori error estimation for DEIM reduced nonlinear dynamical systems,” *SIAM Journal on Scientific Computing*, vol. 36, no. 2, pp. A311–A338, 2014. DOI: 10.1137/120899042.
5. P. Yang, W. Ming, J. Liang, I. Ludtke, S. Berry, and K. Floros, “Hybrid data-driven modeling methodology for fast and accurate transient simulation of SiC MOSFETs,” *IEEE Transactions on Power Electronics*, vol. 37, no. 1, pp. 440–451, Jan. 2022. DOI: 10.1109/TPEL.2021.3101713.
6. C.-T. Tung, S. Salahuddin, and C. Hu, “A SPICE-compatible neural network compact model for efficient IC simulations,” in *2024 International Conference on Simulation of Semiconductor Processes and Devices (SISPAD)*, 2024, pp. 1–4. DOI: 10.1109/SISPAD62626.2024.10733248.
7. F. Gong et al., “A physics-constrained neural network method for compact modeling of semiconductor devices,” *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, vol. 45, no. 9, pp. 4352–4365, 2026. DOI: 10.1109/TCAD.2025.3650386.
8. M. Cho, D. Jang, O.-J. Lee, C.-H. Kim, and S. Jung, “Artificial neural networks in semiconductor device compact modeling: A review,” *Applied Physics Reviews*, vol. 13, no. 2, 021327, 2026. DOI: 10.1063/5.0333553.
9. M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” *Journal of Computational Physics*, vol. 378, pp. 686–707, 2019. DOI: 10.1016/j.jcp.2018.10.045.
10. H. Daniels and M. Velikova, “Monotone and partially monotone neural networks,” *IEEE Transactions on Neural Networks*, vol. 21, no. 6, pp. 906–917, 2010. DOI: 10.1109/TNN.2010.2044803.
11. D. Runje and S. M. Shankaranarayana, “Constrained monotonic neural networks,” in *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202, pp. 29338–29353, 2023. URL: https://proceedings.mlr.press/v202/runje23a.
12. A. Odabasioglu, M. Celik, and L. T. Pileggi, “PRIMA: Passive reduced-order interconnect macromodeling algorithm,” *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, vol. 17, no. 8, pp. 645–654, 1998. DOI: 10.1109/43.712097.
13. B. Gustavsen and A. Semlyen, “Rational approximation of frequency domain responses by vector fitting,” *IEEE Transactions on Power Delivery*, vol. 14, no. 3, pp. 1052–1061, 1999. DOI: 10.1109/61.772353.
14. D. E. Ward and R. W. Dutton, “A charge-oriented model for MOS transistor capacitances,” *IEEE Journal of Solid-State Circuits*, vol. 13, no. 5, pp. 703–708, 1978. DOI: 10.1109/JSSC.1978.1051123.
15. P. Yang, B. D. Epler, and P. K. Chatterjee, “An investigation of the charge conservation problem for MOSFET circuit simulation,” *IEEE Journal of Solid-State Circuits*, vol. 18, no. 1, pp. 128–138, 1983. DOI: 10.1109/JSSC.1983.1051909.
16. E. Santi, K. Peng, H. A. Mantooth, and J. L. Hudgins, “Modeling of wide-bandgap power semiconductor devices—Part II,” *IEEE Transactions on Electron Devices*, vol. 62, no. 2, pp. 434–442, 2015. DOI: 10.1109/TED.2014.2373373.
17. J. Sacks, W. J. Welch, T. J. Mitchell, and H. P. Wynn, “Design and analysis of computer experiments,” *Statistical Science*, vol. 4, no. 4, pp. 409–423, 1989. DOI: 10.1214/ss/1177012413.
18. M. C. Kennedy and A. O’Hagan, “Bayesian calibration of computer models,” *Journal of the Royal Statistical Society: Series B*, vol. 63, no. 3, pp. 425–464, 2001. DOI: 10.1111/1467-9868.00294.

### Official simulator, vendor, and metrology documentation

19. Plexim GmbH, *PLECS 5.0 Documentation — Thermal Concepts*. URL: https://docs.plexim.com/plecs/5.0/thermal-modeling/thermal-concepts/ (accessed 2026-11-21).
20. Plexim GmbH, *PLECS 5.0 Documentation — How PLECS Spice Works*. URL: https://docs.plexim.com/plecs/5.0/spice/core_concepts/ (accessed 2026-11-21).
21. Plexim GmbH, *Power Semiconductors* (ideal switch models, two steps per switching event, behavioral reverse recovery). URL: https://www.plexim.com/products/plecs/electrical/semiconductors (accessed 2026-11-21).
22. SIMPLIS Technologies, *MOSFET Models* (Level 2 structure, linear transconductance, CGS linear, CDG/CDS four segments, extraction test conditions). URL: https://www.simplistechnologies.com/documentation/simplis/sp_semi/topics/mosfet_models.htm (accessed 2026-11-21).
23. SIMPLIS Technologies, *PWL Capacitors* (charge-axis PWL, charge continuity, capacitance steps). URL: https://simplistechnologies.com/documentation/simplis/sp_00_master/topics/pwl_capacitors.htm (accessed 2026-11-21).
24. Analog Devices, *LTspice 26 Help*: “B. Arbitrary Behavioral Voltage or Current Sources”; “Capacitors”; “.OPTIONS” (installed official documentation, accessed 2026-11-21).
25. B. DeBoi, B. Nelson, and A. Curbow, “Accuracy evaluation and proposed dynamic tuning procedure of a compact SiC SPICE model,” Wolfspeed, PCIM Europe 2024. URL: https://forum.wolfspeed.com/uploads/VQL8VYDXXYC8/ws-spice-dynamic-tuning.pdf (accessed 2026-11-21).
26. U. Drofenik and J. W. Kolar, “A general scheme for calculating switching- and conduction-losses of power semiconductors in numerical circuit simulations of power electronic systems,” *International Power Electronics Conference (IPEC)*, Niigata, 2005. URL: https://www.ams-publications.ee.ethz.ch/uploads/tx_ethpublications/drofenik_id356_IPEC05.pdf (accessed 2026-11-21).
27. JCGM, *Evaluation of measurement data — Guide to the expression of uncertainty in measurement*, JCGM 100:2008. DOI: 10.59161/jcgm100-2008e.

### Project records used for all PieceSpice numbers

- `reports/comparison/<vendor>/tier3_validation.json`
- `reports/comparison/<vendor>/tier3_validation_simplis.json`
- `reports/comparison/<vendor>/unified_validation.json`
- `reports/comparison/<vendor>/unified_ann_trap-default.json`
- `reports/comparison/<vendor>/tier4_validation.json`
- `reports/comparison/<vendor>/full_sweep_comparison.json`
- `reports/comparison/<vendor>/dpt_metrics.json` (legacy input artifact; see §8.5)
- `config/inputs/<vendor>.yaml`
- source functions cited inline by module/function name.

---

## 11. Limitations of this research

1. **No new measurements.** All PieceSpice numbers are quoted from the 2026-11-21 records. The report does not re-run any simulation, so it cannot detect a record that is internally consistent but physically wrong.
2. **Some papers were read only through verified metadata/abstracts.** Publisher bot protection blocked several full texts. Claims are limited to the abstract in those cases; the source ledger marks the bibliographic record.
3. **The literature results are authors’ results under their conditions.** The 30 % (Yang), 40× (Tung), 3.2×/70 % (Gong), and 48.9→18.5 % / 26.4→12.9 % (Wolfspeed) numbers are not PieceSpice measurements and are not promises of the same gain here.
4. **The report does not select a single winning architecture.** It identifies error sources and evidence-backed directions. The P0 items are correctness/metric fixes; the P1 items require new data and a declared acceptance protocol.
5. **The report does not claim that the vendor golden is physically correct.** The project’s own AES-STD100 caveat applies: the vendor models exceed datasheet current ratings, and matching the golden is not matching the datasheet. A repaired model should be validated against both the golden and the datasheet where they disagree, with the disagreement documented.
6. **No claim of exhaustive literature coverage.** The source set is sufficient to support the findings, not a systematic review of every MOR, ANN, or thermal-model paper.
