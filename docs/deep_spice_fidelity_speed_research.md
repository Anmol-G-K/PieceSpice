# Deep SPICE Fidelity-vs-Speed Research — Decision-Grade Synthesis and PieceSpice Decision

**Date:** 2026-11-20 (IST)
**Type:** Research documentation only. No code, config, JSON, test, simulation, lint/format/full-suite
run, or `SpiceModels/**` edit was made in this wave. No external number below was re-measured here;
every number is the cited source's own measurement under the stated conditions, and every
PieceSpice number is a recorded value from the cited internal artifact. Engineering interpretation
is tagged **[INFERENCE]**. This document **reconciles rather than replaces**
`docs/fidelity_speed_research.md` (the prior five-track synthesis); where the two disagree, the
disposition is stated explicitly in §5.

**Seven tracks consolidated here:**

| Track | Report | Subject |
|---|---|---|
| A | `local://deep_compact_theory.md` | Compact-model / charge-conservation theory |
| B | `local://deep_circuit_mor.md` | Circuit-level MOR / terminal-equivalent reduction |
| C | `local://deep_solver_fairness.md` | Solver-level equal-comparison methodology |
| D | `local://deep_vendor_models.md` | Vendor / journal behavioral & table model techniques |
| E | `local://deep_hybrid_architectures.md` | Hybrid / multirate / event-driven / co-simulation architectures |
| F | `local://equal_solver_experiments.md` | **Controlled equal-solver experiments (Gear on both arms, 32/32 runs)** |
| P | `docs/fidelity_speed_research.md` | Prior five-track synthesis (reconciled here) |

---

## 1. Executive answer

**The binding rule changes the answer.** A ROM-vs-golden speed or fidelity number is admissible
only if golden and ROM use the **same solver, integration method, tolerances, options and bench**.
If Gear is used for the ROM it must be used for the golden, with the same options and tolerances;
otherwise the comparison is invalid. This rule is now enforced by a completed experiment
(Track F, 32/32 runs, `.options method=gear` injected into every netlist on both arms): the
published Tier-3 `simplis` "≈5×" TI result is **not method-invariant** — it becomes **0.856× on
TI and 0.020× on CPM2** under Gear-both, because the PWL charge tables make the *ROM* the stiff
side under Gear (Track F §2.3–2.4, §1). Tier-2 (4.407× TI, 0.975–1.074× elsewhere) and Tier-4
(6.053× TI, 1.198–1.422× elsewhere) keep their TI advantage under Gear because that advantage is
**golden-side stiffness**, not a portable property of the emitted model (Track F §§2.3, 3).

**What this means for the goal.** The 10–100× headline is not literal; a *significant,
fidelity-preserving* speedup is the goal. Under the equal-solver rule the honest picture is:

- **No single ASCII device model in the surveyed literature or in our harness has a portable,
  method-invariant, waveform-fidelity-preserving ≥5× speedup.** This remains *evidence of
  absence in these seven reports*, not a proof of impossibility (prior synthesis §6, F6).
- The only ≥4× results that **survive the equal-solver rule** are Tier-2 (4.407× solver) and
  Tier-4 (6.053× solver) on **TI**, and both are explained by the TI golden taking ~1.2–1.7×10⁵
  adaptive iterations while the ROMs take ~6×10³–1.4×10⁴ (Track F §2.3; plan §16.2 item 2). On
  the other three devices the same models give 0.975–1.422×. Tier-4 is significant on TI but
  **not portable**, and its DPT waveform observables are retired by design (near-zero `tr`/`tf`,
  up to 6,906 V/ns on IRFS), so it is legitimate only for converter-tier energy/efficiency
  observables (Track F §2.2, §3; README §"Tier-4").
- **Significant speed that is real but lives at a different layer** — SIMPLIS 10–50×, PLECS "two
  steps per event", event-driven EMT 11.6×, solver kernels — can only be claimed as a separately
  reported backend with its own observable contract, never as an LTspice `.lib` property (Tracks
  B, D, E; plan §16.2 item 3).

**Decision:** (1) make the equal-solver protocol mandatory infrastructure before any speed row is
accepted; (2) treat fidelity-preserving *model-level* engineering (C1 charge ramps, breakpoint
placement, charge-primitive discipline, output-based DPT tuning) as the only legitimate path that
survives a simulator export, with a realistic portable ceiling of order 1× (≤1.07× measured in
project; plan §12); (3) keep TI-class stiffness and the Tier-4 ideal-switch tier as explicitly
scoped, non-portable or dual-path results; (4) pursue converter-tier speed only behind a distinct
backend and energy/efficiency gate. Each path is quantified and gated in §6.

---

## 2. The equal-solver rule (central)

### 2.1 Formal statement

A comparison of golden against a ROM is **valid** only when, for both arms:

1. One simulator, one binary version, one solver core, one registered option set. A changed
   solver setting creates a **new experiment**; its timing is never compared to the other arm's
   old setting (Track C §4.1.1; Track E "G0"; Track A "G0").
2. Deck identity except the DUT: byte-identical `.options`, `.tran` card, stimulus edges,
   breakpoints, `.measure`, temperature and `.param`; only the DUT subcircuit body differs
   (Track C §4.1.2).
3. Effective-settings proof from the written netlist/log; otherwise the row is `UNVERIFIED` and
   excluded from speed claims (Track C §4.1.3).
4. Completion guard: both arms reach the same `t_last` (relative difference ≤1e-9), no
   "time step too small"/convergence abort, iterations >0, trace points >0. Aborts are failures,
   never fast results (Track C §4.1.4; Track F §0).
5. Two clocks: solver elapsed and wall clock reported separately; a wall ratio is not a solver
   speedup (Track C §4.1.5; Track F §0 caveat 1).
6. ≥3 interleaved repeats per arm (golden, ROM, ROM, golden, …), median plus min/max, same machine
   load and thread count (Track C §4.1.6).
7. Waveform observables (`vgs/vds/id`, dv/dt, Miller, ringing, timing) reported **separately**
   from average/energy/efficiency observables, with separate pass/fail (Track C §1.6; Track F
   §2.2; AGENTS.md plot/metric invariants).

**Corollary (method-invariance).** Matching method on both arms is necessary but not sufficient
for a *portable* claim: the method itself must be declared, and a claim that survives under only
one integrator is a numerical artifact of that integrator/model pair, not a model-level speed
class (Track F §1). The operational gate is therefore
`min(speedup_default, speedup_gear) ≥ 1.5` on identical benches with guards on both arms (Track F
H1).

### 2.2 Why tolerance / integrator knobs are out of scope as speedups

- LTspice 26 Help, *Integration Methods*: Gear "requires more time steps and is therefore
  generally slower than trap integration" and "imposes some artificial numerical dampening ...
  which can make unstable circuits look stable"; trap ringing amplitude "increases with
  increasing time step" (`.../LTspiceHelp/integrationmethods.htm`, body, copyright 1998–2026;
  Track C L1). Help *Improve Transient Convergence*: `method=gear` = "Larger truncation error,
  tends to dampen oscillations, usually slower"; "with the exception of `itl4`, increasing values
  compromises accuracy" (`.../convergence.htm`; Track C L2). Help *SPICE*: `trtol` "may give a
  speed of 2×" but `trtol` "is remembered between program invocations" — a persistent global
  setting, not a portable netlist gain (`.../spice.htm`; Track C L3).
- Cross-simulator defaults differ (LTspice `trtol=2`, ngspice 7, PSpice 7; method and order are
  version- and preset-dependent), so "default" is not a stable baseline (Track C L4–L5, L8,
  L11–L16).
- Derived integrator behavior explains why fidelity is method-dependent: for a lightly damped
  1 MHz / Q=10 mode at `h/T=0.1`, trapezoidal gives ~3 % frequency error vs ~10 % for Gear-2;
  at `h/τ≈1` the two methods disagree by ~36 % in effective decay time of a stiff real pole
  (Track C §1.2, tables derived from Nichols et al., *IEE Proc. CDS* 141(4) 242–250 1994,
  DOI 10.1049/ip-cds:19941246, §§3–4, and SUNDIALS/CVODE §3.2.1–3.2.4,
  `https://sundials.readthedocs.io/en/latest/cvode/Mathematics_link.html`). **[INFERENCE]** These
  are direct timing observables, so a method change moves the waveform, not just the step count.

### 2.3 Consequence for prior project comparisons

The plan §12 screen used "golden DEFAULT as the sole reference; fairness runs on golden under
every option"; its only ≥1.5× paths were Gear variants that "all fail the fidelity band"
(Vds +1.17 to +1.98 pp), with fidelity-passing paths ≤1.07× (plan §12; Track C L23). Under the
new rule that screen is **invalid for both fidelity attribution and speedup**: the arms did not
share a solver setting (Track C §6). Track F re-ran Gear on both arms and closed this question
(§3 below). The authoritative TI 5.4338461538461535× solver / 3.152941252612414× wall
default-solver result (`reports/comparison/csd16412q5a/tier3_validation_simplis.json` →
`speed_adaptive_20`) used byte-identical benches and the same solver on both arms, so it remains
a valid equal-solver result *at its stated method* (Track C §6) — but it must be labelled
`method=trap/default`, not compared numerically to a Gear row (§5.1).

---

## 3. The decisive equal-solver experiment (Track F)

### 3.1 What was done

Gear was selected with the official card `.options method=gear`, injected into every generated
netlist before `.end`; `TranDirective` has no method/tolerance field and
`SpiceEngine.run_transient` offers no options argument (`simulation/engine.py:36-42,74-80,225-235`;
`simulation/netlist_text.py:109-115`; Track F §0). Equal-solver contract verified per row: identical
method card, other options left at LTspice defaults on both arms (no `reltol`/`abstol`/`vntol`/
`trtol`/`gmin` override anywhere), identical `.tran`/`.temp`/bench/stress, model under test the
only difference. `gear_card_present=true` in all 32 rows. No tolerance was relaxed (Track F §0).

**Completion:** 32/32 runs completed. Adaptive: all 16 rows `t_last=1.76e-4 s`, `points_ok=true`;
DPT: all 16 rows reached 14 µs, positive iterations, trace points ≥50 % of golden (worst 0.68× on
TI). No timeout, abort or missing raw file; no N/A rows (Track F §0).

**Gear actually took effect (A/B control, same harness):** Tier-3 `simplis` iteration counts change
14× (TI 12,073 → 172,303) and 92× (CPM2 10,592 → 975,443) when Gear is enabled; Tier-4 changes
little (near-linear switch plus one fixed `C_QOSS`) (Track F §0, A/B table).

### 3.2 Fidelity under Gear-both — 2-pulse DPT (forced 2 ns, 14 µs)

Relative RMSE = RMSE / golden peak-to-peak (project convention,
`pipeline/tier3.py:92-108`). Golden spans: IRFS 38.98 V / 138.53 A; TI 13.5 V / 13.4 A; EPC
11.8 V / 13.2 A; CPM2 20.8 V / 26.8 A (Track F §2.1).

| Device | ROM | Vds rel-RMSE [%] | Id rel-RMSE [%] | Compare pts |
|---|---|---:|---:|---:|
| IRFS7730PBF | Tier-2 | 12.771 | 0.5699 | 7,194 |
| IRFS7730PBF | Tier-3 simplis | 8.439 | 0.5298 | 7,123 |
| IRFS7730PBF | Tier-4 | 16.093 | 0.5910 | 7,135 |
| CSD16412Q5A (TI) | Tier-2 | 5.467 | 0.0442 | 7,221 |
| CSD16412Q5A (TI) | Tier-3 simplis | 7.273 | 0.0466 | 7,296 |
| CSD16412Q5A (TI) | Tier-4 | 10.560 | 0.0971 | 7,188 |
| EPC2216 | Tier-2 | 5.075 | 0.0461 | 7,241 |
| EPC2216 | Tier-3 simplis | 6.840 | 0.0493 | 7,607 |
| EPC2216 | Tier-4 | 7.263 | 0.1423 | 7,096 |
| CPM2-1200 | Tier-2 | 4.122 | 0.2176 | 7,145 |
| CPM2-1200 | Tier-3 simplis | **29.210** | 0.2585 | 14,609 |
| CPM2-1200 | Tier-4 | 9.582 | 0.1691 | 7,115 |

Vds relative error spans 4.12–29.21 % (CPM2 Tier-3 simplis is a hard fail); Id spans
0.044–0.591 % (Track F §2.1, §4-H4). **Fidelity is method-dependent:** e.g. IRFS Tier-2 Vds is
12.77 % under Gear vs 8.53 % in the project default-solver record; neither is "the" model error
(Track F §1). Timing/energy under the same runs confirm that Tier-4's near-zero `tr`/`tf` and
large `dv/dt` coexist with small Id RMSE (0.097–0.591 %) and close Tier-4 `E_off` on **three of
four** devices — TI 17.973→17.914 µJ (−0.33 %), EPC 15.540→15.705 µJ (+1.06 %), CPM2
56.478→55.692 µJ (−1.39 %) — but **not** IRFS (596.003→533.185 µJ, **−10.54 %**); the earlier
"within ~1 % on all four" wording was wrong (Track F §2.2). Energy agreement (3/4) does not
validate the switching waveform: the waveform-retirement / energy-tier argument survives on Id
RMSE and `E_off` for 3/4, not "all four" (Track F §2.2, §3).

### 3.3 Speed under Gear-both — 20-pulse adaptive (176 µs, 3 repeats)

`speedup = golden / ROM`; greater than 1 means ROM faster. All rows guarded, completed (Track F §2.3).

| Device | Model | Solver s (median) | Wall s (median) | Iterations | Points | Solver speedup | Wall speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| IRFS | golden | 0.408 | 1.02046 | 19,158 | 5,138 | — | — |
| IRFS | Tier-2 | 0.380 | 0.98985 | 14,505 | 3,733 | 1.074× | 1.031× |
| IRFS | Tier-3 simplis | 0.356 | 0.93776 | 13,007 | 3,501 | 1.146× | 1.088× |
| IRFS | Tier-4 | 0.287 | 0.82723 | 5,890 | 2,087 | 1.422× | 1.234× |
| TI | golden | 1.701 | 2.91832 | 122,754 | 40,809 | — | — |
| TI | Tier-2 | 0.386 | 0.94745 | 13,877 | 3,695 | **4.407×** | 3.080× |
| TI | Tier-3 simplis | 1.986 | 3.28065 | 172,303 | 45,396 | **0.856×** | 0.890× |
| TI | Tier-4 | 0.281 | 0.84783 | 5,918 | 2,126 | **6.053×** | 3.442× |
| EPC | golden | 0.357 | 0.96342 | 13,666 | 3,631 | — | — |
| EPC | Tier-2 | 0.366 | 0.93245 | 13,601 | 3,371 | 0.975× | 1.033× |
| EPC | Tier-3 simplis | 0.390 | 0.97988 | 19,565 | 4,480 | 0.915× | 0.983× |
| EPC | Tier-4 | 0.298 | 0.86104 | 5,390 | 2,029 | 1.198× | 1.119× |
| CPM2 | golden | 0.388 | 0.97704 | 15,025 | 3,789 | — | — |
| CPM2 | Tier-2 | 0.366 | 0.96202 | 12,768 | 3,527 | 1.060× | 1.016× |
| CPM2 | Tier-3 simplis | 19.436 | 25.08825 | 975,443 | 271,393 | **0.020×** | 0.039× |
| CPM2 | Tier-4 | 0.286 | 0.82304 | 5,472 | 2,021 | 1.357× | 1.187× |

### 3.4 Method dependence — default-solver rows vs Gear-both

| Device | Tier-2 default → Gear | Tier-3 simplis default → Gear | Tier-4 default → Gear |
|---|---|---|---|
| IRFS | 1.074× → 1.074× | 1.0433333333333334× → 1.146× | 1.178× → 1.422× |
| TI | ~1.17× → 4.407× | **5.4338461538461535× → 0.856×** | 6.594× → 6.053× |
| EPC | 1.011× → 0.975× | 1.1365079365079365× → 0.915× | 1.212× → 1.198× |
| CPM2 | 0.945× → 1.060× | N/A → **0.020×** | N/A → 1.357× |

Tier-3 `simplis` default-solver rows are the `speed_adaptive_20` values from the per-vendor
`reports/comparison/<vendor>/tier3_validation_simplis.json` (source of truth — §5.1); Tier-2 and
Tier-4 default rows are unchanged from the earlier README/benchmark tables, and Gear rows from
Track F §2.3 (Track F §2.4). Only Tier-3 `simplis` flips sign; its former ~5× TI result is a numerical
artifact of the integrator/model pair (Track F §1, §3). **[INFERENCE]** The Gear-induced collapse
of Tier-3 is a *model-level solver interaction* (PWL breakpoints force the ROM's LTE controller to
take tiny steps under Gear), not an accuracy gain.

### 3.5 CPM2 condition-specific completion

CPM2 golden DPT completed under Gear although it aborts under the default solver
(`config/inputs/cpm2-1200.yaml:88-90`, defect VM-10). This is condition-specific and must **not**
be read as VM-10 being resolved; the CPM2 golden adaptive default-solver failure
(`no raw output`, `rc=1`, plan §16.4) is a separate bench condition (Track F §0 caveat 3, §2.4).

---

## 4. Cross-track mechanism synthesis — model-level vs simulator-level

The seven tracks converge on a two-layer taxonomy. **Model-level** speed is carried by the emitted
netlist at a fixed kernel and survives export; **simulator-level** speed is a binary/kernel
property and cannot be expressed in ASCII SPICE (Tracks A "taxonomy", B §0, E §0). Waveform
observables are always gated separately from energy/efficiency.

### 4.1 Model-level — the only legitimate ASCII-ROM speed layer

| Lever | Evidence (source + locator + conditions) | Layer / observable | Disposition |
|---|---|---|---|
| Charge as state variable (`Q=`), not `C(V)` primitive | Ward & Dutton, *IEEE JSSC* 13(5) 1978 703–708, DOI 10.1109/JSSC.1978.1051123, Eqs. 3–7/11, Figs. 5–7; Yang, Epler & Chatterjee, *IEEE JSSC* 18(1) 1983 128–138, DOI 10.1109/JSSC.1983.1051909, §§I–II, V–VIII, Table I p.135 (18–85 % SPICE2 CPU reduction, tolerances unstated); Smedes, TU/e thesis 1991 §3.4.2 p.33 (reciprocity vs conservation mutually exclusive) (Track A S1/S2/S4) | M / convergence + waveform | **ADOPT** the primitive; do **not** import the timestep ratios as DPT speedups (equal-solver status unverified) |
| C¹-smooth charge ramps / breakpoint placement | LTspice Help *C. Capacitor*: `Q=` symbolically differentiated, no-internal-node parasitics give "matrix smaller, faster to solve"; Help *B. Arbitrary behavioral sources*: `uplim`/`dnlim` "continuous 1st derivative transition width" (Track A S7a/S7b); Smedes Eq. 3.96 `hyp(X,ε)=½(X+√(X²+ε²))`; project pre-fix 0.1 V CDS breakpoint crossed 64× with golden dt to 0.018 ns, fixed by zero-anchored uncapped fits (plan §16.2 item 5, §16.3) | M / convergence + waveform | **ADOPT (already landed)**; the one genuinely measured model-level lever in-project. Our soft-abs ramps are C⁰, not C¹ (Track A §1.3) — any C¹ change is a fidelity-neutral hardening until proven otherwise |
| Consistent 3-terminal charge set with nonreciprocal transcapacitance | Fjeldly et al., Wiley 1998 Ch.1 §§1.5.3.1–1.5.3.2 Eqs. 1.78–1.88 pp.38–41; BSIM4.3.0 Manual §7.2.4 Eq. 7.2.16 (partition 0/100, 50/50, 40/60); Roy & Enz, NSTI 2007 p.609 (WD fails with field-dependent mobility) (Track A S3/S5/S8) | M / waveform | **PARTIAL** — validate partition on our velocity-saturated devices; no new internal nodes |
| Output-based DPT tuning of fixed-complexity scalars | Wolfspeed, DeBoi/Nelson/Curbow, "Accuracy Evaluation and Proposed Dynamic Tuning Procedure of a Compact SiC SPICE Model", §3.1 Eqs. 2–4, §3.2 Fig. 6–8 (`https://forum.wolfspeed.com/uploads/VQL8VYDXXYC8/ws-spice-dynamic-tuning.pdf`): dv/dt_off 48.9→18.5 %, Eoff 26.4→12.9 % at fixed model complexity (Track D D2) | M / waveform (fidelity only) | **ADOPT as offline calibration**; no speed claim; obey the C_DS/Qoss reverse-recovery warning |
| Smooth logistic / tanh charge forms and derivative-aware fitting | EPC2216 vendor netlist lines 45–68 (logistic caps), CPM2 bare-die lines 109–119 (tanh/limit charge-current), onsemi TND6260/D §Capacitances p.6 (Track D D6/D7/D3); Tung, Kao & Hu, *IEEE TED* 69(11) 6476–6479 2022, DOI 10.1109/TED.2022.3208514, §II-A/B Eqs. 1–4 (Track D D10) | M / waveform + Jacobian smoothness | **PARTIAL / offline** — authoring and fitting upgrades; the 6.5×/40× numbers are Python/digital model-evaluation, not SPICE transients |
| Guarded ANN + physical cutoff + voltage-dependent caps | Yang et al., *IEEE TPEL* 37(1) 440–451 2022, DOI 10.1109/TPEL.2021.3101713: ~30 % less runtime at fixed SIMetrix solver (abstract-verified); the "DPT ≤11 % of hardware, 1.5–3× waveform RMS, 2.82 % vs 4.26 % spectrum, 20-run/40 µs" detail is **CONDITIONAL** (IEEE copy read by prior Track A only; institution PDF 403 in Tracks D/E) (Track D D8; prior synthesis §3 A5/E2) | M / waveform + speed | **PARTIAL** — fidelity path; ~30 % is modest and against a heavy Angelov baseline; single temperature |
| ROHM L3 high-speed behavioral `.lib` | ROHM white paper 67WP017E Rev.001 Apr 2025, §"Evaluation of models" Table 1 (`https://fscdn.rohm.com/.../new_spice_model_for_faster_simulation_of_power_semiconductors_wp-e.pdf`): 179 s → 79 s = 2.27× **[INFERENCE: 179/79]** in LTspice 24.1.3, Core i7-8700, qualitative DPT "similar", algorithm undisclosed (Track D D1) | M / waveform + speed | **ADOPT as benchmark only** — proof that in-class model-level >2× exists; not a portable recipe; record golden revision |
| Per-iteration expression cost | plan §16.2 item 4: 42 % fewer `abs(` tokens changed solver time 0.317→0.303 s (−4.4 %) while per-iteration cost rose 25.9→34.3 µs; expression cost <5 % of solver time | M / speed | **REJECT as speed lever** — segment/token removal cannot create a new speed class |
| Self-heating / thermal node | BSIMSOI v4.4 Manual Ch.5 §5.2 pp.35–36 (T node with `shMod` ON and Rth≠0); Zwiedinger, Univ. Florida PhD 1997 §4 Table 4-1 pp.58–59 (AC +71/+78 %, transient +116.7 % time, 5 nodes ≈ double BJT matrix); Infineon AN 2020-17 §3.2 (Track A S6/S15/S10) | M / energy + convergence | **DUAL-PATH / REJECT as default** — separate, explicitly labelled thermal variant only; CPM2 `Tj` port is volts=°C (AGENTS.md) |
| Fidelity-deleting damping (`Rp=1 mΩ`, `level=1`) | Wolfspeed PRD-07913 Rev.3 Aug 2026 §2.1 pp.12–13: "improves simulation speed by several orders of magnitude" but "will not correctly predict voltage overshoot" (Track A S14) | M / waveform loss | **REJECT as waveform-ROM speed**; a distinct model name with sacrificed observables named in its header if ever built |

### 4.2 Simulator-level — real speed, wrong layer for the ASCII deliverable

| Architecture | Evidence (source + locator + conditions) | Conditions / caveats | Disposition |
|---|---|---|---|
| SIMPLIS PWL topology solve | SIMPLIS official docs *What is SIMPLIS?* / *SIMPLIS — Piecewise Linear, All the Time* / *Accuracy of SIMPLIS PWL Models* (`simplistechnologies.com/documentation/simplis/...`): "typically 10 to 50 times faster"; one linear system solved per PWL topology (Track E SIM-1/2/3) | Vendor claim with no bench table; PWL device artifact is separate from SPICE `.lib`; Q(V)-over-C(V) fidelity argument transfers (Track B SIM-2) | **DUAL-PATH** — optional backend; never an LTspice ratio |
| PLECS event switching / sub-cycle average | Plexim official docs *How PLECS Works* (event-detection loop, interpolation), *Ideal Switch Models* "only two steps are needed for each switching event" (qualitative, no factor), *SPICE* coexistence (Track E PLE-1/2/4/5) | Ideal config retires finite edges, Miller droop, reverse recovery; energy/efficiency gate only; MNA vs state-space not bit-identical | **DUAL-PATH** — converter tier; separate observables |
| Event-driven / multirate / DSED | Yadav & Nutaro, arXiv:2608.22226v1 (23 Aug 2026) §III Table I: 62.50 M fixed steps → 10.62 M event advances, 11.6× (15.3× at 20 µs sync), NRMS <0.2 % vs own RK4 on 1806-state grid EMT (Track E EVT-1); DSED ECCE 2019 tutorial abstract up to 10×/100× (abstract-only); MR-FPGA *Machines* 10(11) 1073 2022 Tables 2–3 (Track E MR-1) | Different kernel/formulation/reference; DSED Table III numeric cells garbled → no ratio quoted; not device DPT | **REJECT as ROM evidence**; converter/grid architecture only |
| Solver kernels (KLU, APS, HPP, multithreading) | Track C L3/L11/L19, prior synthesis C3: ngspice KLU/OpenMP 65.4→32.4 s, Spectre APS 1.5–2×, HSPICE HPP up to 7×, PrimeSim Figure 2 1×/3×/10× | Real speed, lives in binary; ratios lack circuit/guard or are vendor assertions | **REJECT for portable ROM**; dual backend only |
| Circuit MOR (PRIMA/TurboMOR, TPWL/PWP, pH reduction, thermal ROM) | Track B B1–B8: PRIMA 17,343 s vs 1.23 s (different simulator, 859 R/17,892 K); TurboMOR 9.32×/84.76× offline (MATLAB, state size 163,697); PWP 791→102 s = 7.7× (MATLAB op-amp); thermal 48,595→12 states, ANSYS >500 min vs ~5 s (FE-vs-ROM) (Track B §2) | SOLVER-MISMATCH / SOLVER-SETTINGS-UNSTATED; large-N targets; none emits an LTspice three-terminal `.lib` | **REJECT as device-ROM speed**; passivity and training-coverage rules ADOPT as guards (Track B §3) |

**[INFERENCE]** The reason model-level work cannot reach the simulator-level class here is scale:
removing work from a ~10-element netlist whose DPT cost is adaptive-step stiffness cannot produce
the savings seen when reducing a 163,697-state RC grid or a 48,595-node thermal model (Track B
§"Cross-cutting"; plan §16.2 item 4).

---

## 5. Reconciliation with the prior synthesis (`docs/fidelity_speed_research.md`)

The prior synthesis is retained as the record of the first five-track pass. This section states
every material change introduced by Tracks A–F. No prior file is edited or deleted.

| Prior-synthesis item | Disposition here | Reason / new evidence |
|---|---|---|
| §5 gate 1: "golden at LTspice 26 defaults as the sole reference"; no `method` change in either arm | **SUPERSEDED** by the equal-solver rule (§2) | The golden must use whatever method/tolerances the ROM uses. A Gear ROM requires a Gear golden (Tracks C §6, F §0). The prior wording permits exactly the invalid comparison the new rule forbids |
| §3 / §6 "no fidelity-preserving ≥5× standard-SPICE model within this survey" | **REFINED, not overturned** | Under Gear-both, Tier-4 TI reaches 6.053× and Tier-2 TI 4.407× (Track F §2.3) — method-consistent and significant on TI, but golden-stiffness-driven and non-portable (0.975–1.422× elsewhere). Tier-3 5.4338461538461535× collapses to 0.856× under Gear. The no-*portable*-≥5× conclusion stands |
| §2 baseline default-solver final values — authoritative TI 5.4338461538461535×/3.152941252612414× wall (etc.) from the per-vendor JSONs | **RETAINED with method label and source-of-truth artifact** | Valid equal-solver result at `method=default/trap`; must not be compared numerically to a Gear row (Track F §0 caveat 2, §2.4). Older 5.299×/3.069× and 5.079×/3.070× sets are discarded; see §5.1 |
| Prior "Gear variants all fail fidelity band (Vds +1.17…+1.98 pp)" (plan §12 quoted) | **RE-LABELLED invalid comparison** | Golden-default vs ROM-Gear; re-run Gear-both in Track F gives Vds 4.12–29.21 % for the ROM set (§3.2) — a different, now valid measurement, not a repair of the old row |
| CPM2 golden adaptive N/A (`no raw output`, rc=1) (§2, §5) | **RETAINED for default solver; SUPPLEMENTED** | Under Gear-both CPM2 golden DPT and adaptive both complete (Track F §0, §2.3); condition-specific, VM-10 not resolved (§3.5) |
| §4 path #2 ANN ~30 %, #3 ROHM 2.27× | **RETAINED as precedents, flagged method-status-unverified for transfer** | D1 states no tolerance/step change; Yang fixed solver; neither supplies iteration counts or equal-solver proof for our harness (Track D §2.4) |
| §4 path #1 dual-tier architecture (highest strategic priority) | **UNCHANGED** | Tracks D/E reaffirm: waveform-resolving ASCII ROM default + converter tier with energy/efficiency gate and DPT gate explicitly retired (Track D H6; Track E §3) |
| §5 gate 4 hardware ≤11 % DPT band for the ANN path | **RETAINED, CONDITIONAL** | CONDITIONAL provenance unchanged (Track D D8); do not transfer a hardware figure to our golden-vs-ROM gate |
| §7 audit corrections F1–F6 (22 claims, 18 PASS, 4 CONDITIONAL, 0 FAIL) | **PRESERVED** | This document inherits the CONDITIONAL tags and adds no new source claims from unread documents |

### 5.1 Baseline artifact resolution

Four different "final default-solver adaptive speedup" sets exist; this report names one
authoritative baseline (the 20-pulse `speed_adaptive_20` Tier-3 `simplis` solver speedup, wall
speedup in parentheses) and discards the other three:

| Vendor | Authoritative (source of truth) | Discarded |
|---|---|---|
| IRFS | 1.0433333333333334× (1.0471858009679291×) | 0.961× (summary JSON), 1.074× (benchmark, Tier-2) |
| TI | 5.4338461538461535× (3.152941252612414×) | 5.299×/3.069× (docs), 5.079×/3.069–3.070× (README + summary JSON), 4.858× (benchmark, Tier-2) |
| EPC | 1.1365079365079365× (1.089009401676902×) | 1.080× (summary JSON), error/stall (benchmark, Tier-2) |
| CPM2 | N/A (golden adaptive aborts) | 1.124× (summary JSON), 0.977× (benchmark, Tier-2) |

Source of truth: `reports/comparison/<vendor>/tier3_validation_simplis.json`, the final live
per-vendor record, which carries per-variant guards (`anti_fake_guard`, `guard_run`, `completed`,
`selected_candidate`, `scope`). The older sets are superseded artifacts, not method effects:
5.299×/3.069× appears only in `docs/*.md` (plan §16.4 and the prior synthesis); 5.079×/3.069–3.070×
comes from `README.md` and `reports/comparison/simplis_char_speed_summary.json` (JSON wall value
3.069577557324401×); 4.858× is `speedup_benchmark.json#vendors.csd16412q5a.long_adaptive`, a
**Tier-2** ROM, a different tier. The TI spread 5.079–5.434× is a repo provenance issue, not a
method effect. Every default-solver number in this report now uses the per-vendor JSON values;
the Gear-both Track F rows (TI Tier-3 0.856×, TI Tier-4 6.053×, etc.) are a different,
method-labelled measurement and are unchanged. No external paper/vendor number was altered.

---

## 6. Ranked decision paths and quantified improvement plans

All numbers below are **(measured)** recorded values or **(source)** external figures under the
stated conditions; proposed thresholds are **(project decision)**. Every hypothesis is unmeasured
in this wave and gated as shown. G0 = the equal-solver protocol of §2 on every row.

### Path 0 — Equal-solver protocol adoption (mandatory infrastructure; no speed number)

- **Change:** register one option set per experiment (method, order, `trtol`, `reltol`, Newton
  tolerances, `plotwinsize=0`, step ceiling, temperature, solver core), apply it to both arms,
  dump effective settings from the written netlist/log, interleave ≥3 repeats, and enforce the
  completion guard (Track C §4). Add the `.options method=gear` injection as a **temp-harness**
  reproducibility step; do not add a method/tolerance field to `TranDirective` for this research
  task (Track F §3).
- **Acceptance:** deck diff = DUT body only; effective settings provably equal; guard pass on both
  arms. **[INFERENCE]** This is validity infrastructure, not a speed lever, but it is the
  prerequisite for every row below.

### Path 1 — Fidelity-preserving model-level engineering (only exportable speed layer)

- **Hypotheses (unmeasured):** H1 C¹ charge ramps via `hyp`/`uplim`/`dnlim` reduce
  breakpoint-driven step rejections (Track A H1); H2 consistent 3-terminal charge set replaces
  heuristic gate caps (Track A H2); H3 output-based DPT tuning of emitted scalars reduces
  waveform-metric error at unchanged complexity (Track D D2/H1); H4 derivative-aware fitting and
  guarded ANN shapes improve the Jacobian (Track D D10/D8).
- **Quantified anchor:** fidelity-passing in-form changes measured ≤1.07× absolute (plan §12);
  public model-level precedents ~30 % (Yang) and 2.27× (ROHM) are method-status-unverified for
  transfer. **[INFERENCE]** Expect iteration-count movement, not a new speed class (Track A H7).
- **Gates:** G0; charge budget ≤0.5 % of `Q(Vmax)` on all 12 tables; DPT Vds/Id and IEC timing
  no worse than the recorded bars; no reintroduction of the 0.1 V CDS breakpoint churn (64
  crossings pre-fix); iterations **and** solver time reported; a speed claim permitted only if
  all fidelity gates pass and `min(speedup_default, speedup_gear) ≥ 1.5` (Track F H1). Otherwise
  report as fidelity/robustness hardening with **no** speed claim.

### Path 2 — Golden-side stiffness as the only ≥4× host (scoped, not portable)

- **Measured:** TI golden 122,754 iterations under Gear / 131,558 at defaults; the other three
  goldens 13,666–19,158 under Gear. Tier-2 4.407× and Tier-4 6.053× on TI are consistent with
  this; the same models give 0.975–1.422× elsewhere (Track F §2.3, H3; plan §16.2 item 2).
- **Decision:** reserve any ≥4× claim explicitly to a vendor whose golden exceeds 10⁵ adaptive
  iterations, state the method, and never generalize it to a portable ROM class.

### Path 3 — Tier-4 ideal-switch converter tier (measured, dual-path)

- **Measured:** Tier-4 6.053× solver / 3.442× wall on TI under Gear, 1.198–1.422× elsewhere
  (Track F §2.3); DPT waveform observables retired by design near-zero `tr`/`tf`, up to 6,906 V/ns
  IRFS; Tier-4 `E_off` is close to golden on TI (17.973→17.914 µJ, −0.33 %), EPC (15.540→15.705
  µJ, +1.06 %) and CPM2 (56.478→55.692 µJ, −1.39 %) but not IRFS (596.003→533.185 µJ, −10.54 %)
  — 3/4, not "all four" (Track F §2.2). Default-solver CPM2 gold adaptive N/A becomes 1.357×
  under Gear because the golden completes (§3.5).
- **Gate:** energy/efficiency gate only on a converter bench; DPT waveform gate explicitly
  retired; report Tier-4 and waveform tiers as separate rows, never a combined speedup. Its TI
  6× is *significant but not portable* — this is the direct answer to the 10–100× target.

### Path 4 — Converter / simulator-level backends (dual-backend only)

- **Source figures (not transferable):** SIMPLIS 10–50× (vendor, no bench table), PLECS qualitative
  "two steps per event", event-exact EMT 11.6× (grid EMT, own RK4 reference), DSED abstract-only
  10×/100× (Track E §2). Prior synthesis path #1 keeps one emitted ASCII `.lib` as the shared data
  contract and reports Path A (LTspice waveform) and Path B (PLECS/SIMPLIS energy) separately.
- **Gate:** separate backend, separate observable contract, no external factor attributed to the
  LTspice emission; PLECS Spice netlist import is an interface fact, not a bit-identical guarantee
  (Track E PLE-4/PLE-5).

### Path 5 — Rejections (closed)

- Tolerance/integrator knobs as speed (LTspice `trtol` "2×" is an unguarded, persistent global
  trade; Gear is slower and damping — Track C L1–L3, §2.2; plan §12).
- Nonlinear MOR as a SPICE deliverable (large-N targets, cluster switching = the T3-v1 failure
  mode — Track B §3; prior synthesis path #6).
- ngspice `external` + compiled-C Taylor model (different solver settings, not LTspice-exportable
  — Track D D9/H5; prior synthesis path #4).
- Per-iteration expression/token reduction as a speed lever (<5 % of solver time — plan §16.2).

### 6.1 Method-invariance summary

| ROM | min(default, Gear) solver speedup across 4 devices | Passes H1 (≥1.5 globally)? |
|---|---|---|
| Tier-2 | 0.975× (EPC) | No — only TI >1.5× |
| Tier-3 simplis | 0.020× (CPM2) | No — collapses under Gear |
| Tier-4 | 1.198× (EPC) | No as global claim; TI 6.053× is scoped |

Source: Track F §2.4. **[INFERENCE]** No current ROM is a portable, method-invariant speed class.

---

## 7. Mandatory decision gates for any implementation

1. **Equal solver (G0).** Same simulator/version/core/options/method/tolerances/bench on both
   arms; effective settings proven from the deck; method declared in every row (§2).
2. **Completion guard.** DPT: `t_last ≥ 0.999×14 µs`, ≥50 % of golden points, positive iterations;
   adaptive: 3/3 repeats, `t_last ≥ 0.999×176 µs`, points ≥50 % of median, iterations >0, 90 s
   timeout; EPC golden keeps the 100 ns cap for the default-solver row (plan §16.5; Track F §0).
3. **Method-invariance.** Require `min(speedup_default, speedup_gear) ≥ 1.5` with guards on both
   arms before any global speed claim (Track F H1); otherwise scope the claim to the passing
   method/device.
4. **Static no-regression.** Bar IRFS 7.0683 %, TI 8.4935 %, EPC 2.0394 %, CPM2 3.4287 %, worst
   endpoint ≤4.13 %; proposed gate **(project decision)** |Δ| ≤ +0.05 pp (plan §16.4; prior
   synthesis §5 gate 3).
5. **DPT no-regression under each method independently.** Default-solver bars IRFS
   8.529/0.532 %, TI 5.583/0.048 %, EPC 4.894/0.049 %, CPM2 2.675/0.261 %; Gear-both bars from
   §3.2. No increase in any Vds/Id cell and no worse tr/tf/e_on/e_off on **each** solver setting
   (Track F H4; prior synthesis §5 gate 4).
6. **Charge-stiffness gate for any PWL charge ROM.** Require
   `iterations_ROM(Gear) ≤ 2×iterations_golden(Gear)` and
   `points_ROM(Gear) ≤ 2×points_golden(Gear)`. Measured: TI simplis 1.40× (passes iteration gate,
   still slower); CPM2 simplis 64.9× iterations / 71.6× points (hard fail) (Track F H2).
7. **No fabricated speedup.** Every ratio from the guarded bench with iterations/points/`t_last`
   for both arms; inherited paper/vendor ratios never copied (prior synthesis §5 gate 5).
8. **Deliverable form.** ASCII SPICE subcircuit (`.write_text(..., encoding="ascii")`) under
   LTspice 26 batch unless a dual-path is explicitly approved; `SpiceModels/**` read-only; pin
   order from `.subckt` header (AGENTS.md).
9. **Answer the real observable.** A candidate that wins on energy but deletes switching
   observables has not passed the waveform gate; report the two separately (Track F §2.2; prior
   synthesis §5 gate 7).

---

## 8. Failures, access issues and unresolved conflicts retained

- **Inaccessible / blocked primary sources** (no number taken from them): ST UM1575 and
  Infineon AN 2014-02 power-MOSFET notes (Track A X1/X3); GaN Systems GN006 (A X2); original VF
  1999 and Gustavsen/Deschrijver passivity-enforcement full texts (Track B B9/B10); TPWL journal
  version (B6); Yang TPEL 2022 institution PDF 403 in Tracks D/E (CONDITIONAL figures); Tung
  SISPAD 2024 TU-Wien fetch aborted and eScholarship WAF (D11); Wang *Energies* MDPI 403 (D12);
  JENG-2020 Cloudflare 403 (Track E MR-2); Gear 1971 ACM 403 (Track C L18); Nagel & Pederson 1973
  image-only; AD "SPICE Differentiation" timed out (Track C ledger).
- **Abstract-only or garbled:** DSED ECCE 2019 tutorial 10×/100×/hundreds (abstract only); DSED
  2017 Table III numeric cells garbled — **no ratio quoted** (Track E DSED-2/3); PrimeSim Figure 2
  1×/3×/10× without circuit/guard (prior synthesis C3).
- **Solver-confounded source results:** Kubulus et al. 5–60 % (different max step, reltol, gmin,
  trtol, rshunt between arms — Track D D9); plan §12 Gear variants vs golden-default (Track C L23, §6).
- **Source conflicts not resolved by guesswork:** second-order Gear A-stability (IPST-05 §I says
  not A-stable; Nichols et al. §3 and SUNDIALS §3.2.4 say orders 1–2 are A-stable) — pin method and
  record step ratios rather than relying on the label "Gear" (Track C §1.7). LTspice `plotwinsize`/
  `trtol` defaults differ between the local help, public mirrors and versions — pin explicitly
  (Track C §1.7, L5/L7).
- **VM-10:** CPM2 turn-off abort is vendor-side; Gear completion is condition-specific, not a
  fix (§3.5).

---

## 9. Source and track ledger

Full per-source URLs, DOIs, section/figure/table locators and access status are in the ledger of
each track report; this section registers the consolidation. Primary sources are peer-reviewed
papers, authoritative/vendor manuals and official tool Help unless noted.

- **Track A** (`local://deep_compact_theory.md` §5 ledger): S1 Ward & Dutton 1978 (DOI
  10.1109/JSSC.1978.1051123); S2 Yang/Epler/Chatterjee 1983 (DOI 10.1109/JSSC.1983.1051909);
  S3 Fjeldly et al. Wiley 1998 (publisher excerpt); S4 Smedes TU/e 1991; S5 BSIM4.3.0 Manual
  (`https://cmosedu.com/cmos1/BSIM4_manual.pdf`); S6 BSIMSOIv4.4 Manual; S7a/S7b LTspice Help
  (`https://ltwiki.org/files/LTspiceHelp.chm/html/B-device.htm`,
  `https://ltwiki.org/LTspiceHelp/LTspiceHelp/C_Capacitor.htm`); S8 Roy & Enz 2007; S9 Yang et al.
  engrxiv 2067 (preprint); S10 Infineon AN 2020-17; S11 EPC AN005 2019; S12 Venugopalan EECS-2013-166;
  S13 Hefner IEEE TPEL 6(2) 1991; S14 Wolfspeed PRD-07913 Rev.3; S15 Zwiedinger 1997.
- **Track B** (`local://deep_circuit_mor.md` §5 ledger): B1 PRIMA (IEEE TCAD 17(8) 1998);
  B2 TurboMOR (arXiv:1507.00219v1); B3 Gueye et al. IEEE TPD 38(5) 2023 (DOI
  10.1109/TPWRD.2023.3272927); B4 pyMOR pH tutorial; B5 Gugercin et al. *Automatica* 48(9) 2012;
  B6 Rewienski & White TPWL ICCAD 2001; B7 Dong & Roychowdhury PWP IEEE TCAD 27(2) 2008;
  B8 Dong/Griffo/Wang IEEE TPEL 35(8) 2020; B9–B12 passivity/VF mechanism or context only.
- **Track C** (`local://deep_solver_fairness.md` §7 ledger): L1–L6 AD LTspice 26 shipped Help
  (`integrationmethods.htm`, `convergence.htm`, `spice.htm`, `options--setsimulatoroptions.htm`,
  `tran--doanonylineartransientanaly.htm`, `LTspiceChangeLog.txt`); L7–L10 ngspice 47 manual
  (`https://ngspice.sourceforge.io/docs/ngspice-47-manual.pdf` §§1.4, 11.1.4, 15.6.2.2);
  L11–L13 PSpice Ref/User Guides; L14–L16 Spectre 19.1 Ref; L17 Nichols et al. 1994; L19
  SUNDIALS/CVODE; L20 Meliopoulos IPST'05; L21/L22 Pels et al. multirate; L23 plan §12.
- **Track D** (`local://deep_vendor_models.md` §5 ledger): D1 ROHM 67WP017E; D2 Wolfspeed dynamic
  tuning; D3 onsemi TND6260/D; D4 Infineon/IR AN-1194; D5–D7 repo vendor model files
  (`SpiceModels/CSD16412Q5A.lib`, `EPC2216_V0_PSPICE.net`, `CPM2-1200-0025B-Bare-Die.lib`,
  read-only); D8 Yang et al. TPEL 2022 (CONDITIONAL); D9 Kubulus et al. TPEL 2025 (DOI
  10.1109/TPEL.2024.3510037); D10 Tung et al. TED 2022; D11/D12 CONDITIONAL.
- **Track E** (`local://deep_hybrid_architectures.md` §5 ledger): SIMPLIS official docs (S-WIS,
  S-APWL, S-ACC, S-REST), Plexim official docs (P-HOW, P-SEM, P-CODEGEN, P-SPICE, P-DEMO),
  DSED (DSED-2017, DSED-TUT), E-ORNL arXiv:2608.22226v1, MR-FPGA *Machines* 10(11) 1073 2022,
  HIL-TYP/HIL-GDS/HIL-MW.
- **Track F** (`local://equal_solver_experiments.md` §5 ledger): AD LTspice 26 Help
  (`integrationmethods.htm:43-48`, `convergence.htm:135-150`,
  `options--setsimulatoroptions.htm:207-209`); repo locators `simulation/engine.py:36-42,74-80,
  225-235,305-331`, `simulation/netlist_text.py:85-115`, `simulation/testbenches/double_pulse.py`,
  `simulation/testbenches/speedup_bench.py`, `pipeline/tier3.py:92-108`, `README.md:283-311,367-390,
  407-426`, `docs/plans/tier3_method_candidates.md:403-427`. Throwaway harness/JSONs were deleted;
  `git status --short` clean at hand-off.
- **Internal baselines:** `docs/plans/tier3_method_candidates.md` §§12–16; `README.md`; `AGENTS.md`;
  `reports/comparison/speedup_benchmark.json`; prior synthesis `docs/fidelity_speed_research.md`.

**End state.** Under the binding equal-solver rule, the decisive measured fact is method
dependence: Tier-3 `simplis` is not a portable speed class (5.4338461538461535× → 0.856× on TI,
N/A → 0.020× on CPM2 under Gear-both; §5.1), while Tier-2 and Tier-4 keep significant TI-only gains
(4.407× and 6.053× solver) rooted in golden-side stiffness. No current ROM passes the global
method-invariance gate. The defensible program is equal-solver infrastructure first,
fidelity-preserving model-level engineering with no promised speed class second, and explicitly
scoped dual-tier/converter backends for anything above the ~1× portable ceiling. This wave
produced no code, measurement or `SpiceModels/**` change; all numbers are traced to the seven
reports and internal artifacts named above.
