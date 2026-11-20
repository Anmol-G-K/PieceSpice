# Device Waveform Fidelity vs Simulation Speed — Research Synthesis and PieceSpice Decision

**Date:** 2026-11-20 (IST)
**Type:** Read-only synthesis of five completed research tracks against the measured internal baseline in
`docs/plans/tier3_method_candidates.md` §§10-16. No code, config, JSON, test, lint, formatter, or
simulation was run in this wave; `SpiceModels/**` was untouched. No new PieceSpice measurement is
claimed. Every external number below is the cited source authors'/vendors' measurement under the
stated conditions; engineering interpretation is tagged **[INFERENCE]**. Where a report quotes a
pre-fix figure and plan §16 quotes the post-fix figure, both are given with their condition
(§2.3 and §7, discrepancy notes).

Track reports consolidated here: `local://research_behavioral_rom.md` (A),
`local://research_nonlinear_mor.md` (B), `local://research_spice_solver.md` (C),
`local://research_simulator_architectures.md` (D), `local://research_power_electronics.md` (E).

---

## 1. Executive answer

**What can deliver both waveform fidelity and speed.** Only two things, and neither is a single
magic netlist:

1. **Model-level convergence engineering inside a fixed simulator.** A device model can be
   rewritten so the same nonlinear solve converges in fewer Newton/time-point steps while the
   terminal observables remain unchanged. The cleanest primary-source proof is ROHM's L3
   behavioral `.lib`: on a PFC bench in LTspice 24.1.3, **179 s -> 79 s (44 % of original, 2.27x
   [INFERENCE: arithmetic])** with "similar" DPT turn-off loss waveforms to L1; no tolerance or
   step-size change is stated and the model ships as an ordinary SPICE library
   (E3, ROHM white paper 67WP017E, §"Evaluation of models" Table 1). A peer-reviewed example is
   Yang et al., IEEE TPEL 37(1) 2022: a 10-neuron ANN + physical guards + voltage-dependent
   capacitors in SIMetrix gives **~30 % less runtime** at fixed simulator/bench, which is printed
   in the abstract and verified via Semantic Scholar (A5/E2, DOI 10.1109/TPEL.2021.3101713).
   The same paper's table/figure-level figures — DPT timing/energy within 11 % of hardware,
   waveform RMS 1.5-3x better than the commercial Angelov model, VDS spectrum 2.82 % vs 4.26 %,
   and a 20-run / 40 us / Table IV bench — are **CONDITIONAL**: cited from the IEEE copy read by
   Track A (`ieeexplore.ieee.org/document/9506845`) and not corroborated by Track E, whose
   institution PDF was Cloudflare-403-blocked (only the abstract was verified there). Both
   headline results are *model-level*, *fidelity-preserving*, and *standard-SPICE-shaped* — but
   both are modest (~1.4x-2.27x), device/bench-specific, and require reproduction in our harness.
2. **A dual-tier architecture** that keeps the waveform-resolving ASCII SPICE ROM as the default
   deliverable and adds a converter-level ideal-switch/loss-LUT model for system-level speed.
   The fast tier deliberately **does not** preserve dv/dt, Miller plateau, ringing, or device
   transients, so its gate must be energy/efficiency, not DPT Vds/Id (P-DPE: Plexim's own Spice
   demo states the ideal-switch PLECS configuration misses turn-on/off transients, Miller droop,
   and diode reverse recovery that the SPICE configuration captures [P-DEMO, *SPICE
   Configuration*, *Conclusion*]). Plexim documents a first-class coexistence path for our ASCII
   netlist (Netlist component, configurable subsystem, and syntax support for `tanh`, `table()`,
   `limit`, `pwr/pwrs`, `ddt()` and `Q={...}` [P-SPICE, P-SYN]).

**What cannot deliver both.** No single standard-SPICE device model in the surveyed primary
literature has evidence for a portable, fidelity-preserving **>=5x** speedup. Large gains are of
three kinds, all disqualified as a PieceSpice "ROM is fast" claim:

- **Simulator-architecture speed.** SIMPLIS 10-50x comes from a topological linear solve per PWL
  topology ([S-USER](https://www.simplistechnologies.com/documentation/simetrix/user_manual/topics/introduction_whatissimplis.htm),
  *What is SIMPLIS?*; [S-APWL](https://www.simplistechnologies.com/documentation/simplis/what_is_simplis/topics/simplis_is_all_pwl_all_the_time.htm),
  *Piecewise Linear, All the Time*); PLECS "only two steps per switching event" from ideal
  switches ([P-SEM](https://www.plexim.com/products/plecs/electrical/semiconductors), *Fast*);
  ngspice KLU/OpenMP, PSpice `SOLVER=1`, Spectre APS 1.5-2x, HSPICE HPP up to 7x, PrimeSim
  Figure 2 1x/3x/10x from the simulator kernel (Track C official docs). These change *which
  simulator runs*, not the emitted netlist.
- **Fidelity-deleting simplification.** Ideal-switch + loss-map models (PLECS converter tier,
  Dini et al. IEEE Access 2023 "50 times longer" in LTspice [F1]) buy speed by deleting the very
  switching observables our DPT gates measure.
- **Tolerance / numerical tricks.** Official LTspice help itself warns that increasing `trtol`
  may give "a speed of 2x ... [without] adversely affecting simulation accuracy" but that
  "with the exception of `itl4`, increasing values compromises accuracy"
  (LTspice `spice.htm`, *SPICE*; `convergence.htm`, *Improve Transient Convergence*). Our own
  audit closed these: no fidelity-passing path exceeded **1.07x**, Gear variants failed the Vds
  band (+1.17 to +1.98 pp), and loosened tolerances aborted/timed out or ran 1.9x slower
  (plan §12). They are excluded by the project's anti-fabrication rule.

**Model-level vs simulator-level speed, stated plainly.** Model-level speed is carried by the
subcircuit and survives a simulator export; simulator-level speed lives in the solver binary and
does not. Internal evidence makes the distinction decisive: on TI, a 42 % reduction in emitted
`abs(` tokens (19 -> 11) changed solver time only 0.317 -> 0.303 s (**-4.4 %**) while iterations
fell 28 % and per-iteration cost *rose* 25.9 -> 34.3 us; per-iteration expression cost is <5 % of
solver time (plan §16.2 item 4). Therefore rewriting our emitted expressions cannot buy a new
speed class under LTspice Newton, and the ~5.3x we do measure is explained below as a property of
the golden, not of our ROM.

---

## 2. Internal baseline — recorded values only (plan §16)

All figures in this section are the project's already-measured final (post-fix) values from
`docs/plans/tier3_method_candidates.md` §16.4, cross-checked against
`reports/comparison/<vendor>/tier3_validation_simplis.json`. No re-measurement was made for this
synthesis. Static is the channel-only 41-point DC sweep (capacitors open at DC); DPT is the
completed 2 ns forced / 14 us clamped-inductive run; adaptive is the 20-pulse / 176 us / 3-repeat
bench with medians and completion guards. All DPT rows have t_last = 1.4e-5 s.

| Vendor | Tier-2 static full-sweep rel-RMSE | Final `simplis` DPT Vds / Id | Guarded adaptive solver / wall | Completion guard |
|---|---:|---:|---|---|
| IRFS7730PBF | 7.0683 % (AGENTS: 7.07 %) | 8.529 / 0.532 % | 1.074x / 1.022x (12,718 -> 8,440 iters; 3,432 -> 2,483 pts) | 3/3, t_last 176 us both |
| TI CSD16412Q5A | 8.4935 % (AGENTS: 8.49 %) | 5.583 / 0.048 % | **5.299x / 3.069x** (131,558 -> 12,073 iters; 44,050 -> 2,949 pts) | 3/3, t_last 176 us both |
| EPC2216 | 2.0394 % (AGENTS: 2.04 %) | 4.894 / 0.049 % | 1.124x / 1.095x (16,514 -> 12,123 iters; 4,047 -> 2,934 pts) | golden 100 ns cap, 3/3, t_last 176 us |
| CPM2-1200 | 3.4287 % (AGENTS: 3.43 %) | 2.675 / 0.261 % | golden adaptive **N/A** (`LTSpiceError ... no raw output ... rc=1`); ROM completed 0.310 s / 10,592 iters / 2,586 pts | ROM guard PASS; no ratio reported |

Worst static endpoint <=4.13 % (AGENTS.md "Traps"); the Tier-2 channel form is closed after
exhausted in-form ablations and must not be re-opened without new evidence (AGENTS.md).

**TI 5x is golden-side stiffness, not a portable ROM speed class** (plan §16.2 items 2 and 4):
the TI golden takes ~0.175 ns steps everywhere — 1.738 s / 131,558 iterations / 44,050 trace
points. All six TI ablations, including `tier2_charge` with *zero* simplis table content, land in
0.300-0.332 s / 8,831-12,768 iterations -> **5.19-5.74x**; the plain Tier-2 reference is
0.331 s / 10,195 iterations = 5.21x. The CPM2 control golden (0.347 s) and best ROM (~1.20x)
never reach 5x. The fast goldens on IRFS, EPC and CPM2 already ride the measured
171.875 ns = 176 us / 1024 adaptive ceiling; no measured PWL variant exceeds ~1.23x there
(plan §16.5).

**No tolerance changes.** `trtol`, `reltol`, integration method (`method=gear`), and related
knobs are rejected as speed arguments (plan §12; Track C official-doc classification). The one
genuine model-level lever already landed is breakpoint placement: pre-fix TI crossed a 0.1 V CDS
breakpoint 64 times inside its conduction band (30.09 % of on-state samples in 0.09-0.11 V,
golden dt down to 0.018 ns), fixed by zero-anchored, uncapped charge fits meeting the declared
0.5 % of Q(Vmax) budget on all 12 tables (worst achieved error ratio 0.9796) and a DPT-derived
breakpoint clamped to Vbus (plan §16.2 item 5, §16.3). C1-smooth charge ramps with delta slopes
are an authoring rule, not a new speed claim (Track C §8-B).

**2.3 Source discrepancy to carry forward.** The four-track reports and the earlier dev-log
entries quote *pre-fix* figures: adaptive TI 5.079x, IRFS 0.961x, EPC 1.080x, CPM2 1.124x; DPT
Vds TI 6.063 %, IRFS 11.260 %, EPC 5.250 %, CPM2 2.786 % (plan §15.4, §16.6). The final JSONs
give TI 5.299x / IRFS 1.074x / EPC 1.124x / CPM2 N/A and DPT Vds 5.583 / 8.529 / 4.894 /
2.675 % after the zero-anchor fits, DPT-derived breakpoint, RGS and temperature cleanup. Both
sets are correct under their stated conditions; §2 uses the final (post-fix) set only, as
instructed.

---

## 3. Consolidated evidence table (all five tracks)

"Speed" and "fidelity" columns report only what the source printed under its own bench; blank or
qualitative means the source printed no portable number. **Evidence quality** values:
*peer-reviewed* (full text read), *vendor* (official vendor document), *official docs* (tool
help/manual), *preprint* (arXiv/author manuscript), *abstract-only* (publisher index or
indexed abstract only). PieceSpice verdicts: ADOPT / PARTIAL / REJECT / DUAL-PATH.

| # / track | Method — primary source, exact locator | Fidelity observable and result | Speed mechanism and result | Simulator / backend | Standard-SPICE compatible? | Evidence quality | Verdict and reason |
|---|---|---|---|---|---|---|---|
| A5 / E2 | Hybrid ANN SiC MOSFET — Yang et al., *IEEE TPEL* 37(1) 440-451 2022, DOI **10.1109/TPEL.2021.3101713**; IEEE copy `ieeexplore.ieee.org/document/9506845` as cited by Track A (§III, §V-A/B, Tables I-IV); author PDF Cloudflare 403, abstract indexed (Track E) | DPT delay/rise/fall/Eon/Eoff IEC 60747-8 all within **11 %** of hardware **[CONDITIONAL]**; waveform RMS **1.5-3x better than Angelov** **[CONDITIONAL]**; VDS spectrum 2.82 % vs 4.26 % (Table II, Fig 16) **[CONDITIONAL]** | 10-neuron sigmoid ANN + physical cutoff/low-VDS guards + small voltage-dependent caps replaces >80-parameter Angelov; **~30 % less runtime** (abstract-verified), 20-run average / 40 us bench / Table IV **[CONDITIONAL]**, fixed solver | SIMetrix 8.00g, 40 us DPT, default solver, i7-8650U | Yes — VCCS + three `C(V)` elements, same shape as our emitter | peer-reviewed; **30 % PASS** (abstract via Semantic Scholar), table/figure numbers **CONDITIONAL** (IEEE copy per Track A only; Track E 403, abstract-only) | **ADOPT (model-level reason)** — product-shape match; 30 % is abstract-verified but against a heavy baseline; table/figure numbers CONDITIONAL; must be re-measured in our harness |
| E3 | ROHM L3 high-speed behavioral `.lib` — white paper 67WP017E Rev.001, Apr 2025, §"Evaluation of models" Table 1 | DPT turn-off power-loss waveforms "similar" to L1; qualitative only, no numeric error | Model-internal convergence/calculation engineering, algorithm undisclosed; PFC **179 s -> 79 s = 44 % of original (2.27x)** | LTspice 24.1.3, Core i7-8700 | Yes — ordinary `partnumber_L3.lib` | vendor | **ADOPT as benchmark/precedent** — proof that 2.27x model-level is possible in our simulator class; not a portable recipe |
| A1 | NN IV+QV compact model — Tung, Salahuddin, Hu, SISPAD 2024, DOI 10.1109/SISPAD62626.2024.10733248, `in4.iue.tuwien.ac.at/pdfs/sispad2024/SISPAD_2024_10.pdf` (abstract, §II, Table I, Figs 2-9) | ID/IG, QG/QS/QD plus gm, gds, gm', gds', dQ/dV, d²Q/dV² in loss (Eqs 1-5) | Direct multiplication instead of Verilog-A loops/arrays; ISRU `x/sqrt(1+x²)`; **~40x** vs BSIM-CMG | Cadence-class simulator, **digital** ring oscillators (25K-100K stages) | Plausibly (scalar B-expression) [INFERENCE] | peer-reviewed | **PARTIAL** — adopt IV/QV split, `VDS·exp(y)` transform, Q0 offset, derivative loss; reject 40x as a DPT speed claim |
| A2 | NN IV/CV with derivative-aware loss — Tung, Kao, Hu, *IEEE TED* 69(11) 6476-6479 2022, DOI 10.1109/TED.2022.3208514, `escholarship.org/content/qt7bn8c5wk/...pdf` (§II-A/B, §IV) | ID to 3rd derivatives, QG/Cgg/Cgd with 2nd derivative (Eqs 1-4) | Fewer evaluation lines; 1M evals 13.51 s vs BSIM 87.28 s (~6.5x) | Python inference, 12 nm GAAFET TCAD | C-code in commercial SPICE "requires more research" (§IV) | peer-reviewed | **PARTIAL** — reuse output transforms and loss; speed is Python model-eval, not SPICE transient |
| A4 | KAN / Fourier-KAN — Novkin & Amrouch, arXiv:2503.15209v1, `arxiv.org/html/2503.15209v1` (Table II, §IV-B/C/D/F, Figs 4, 8) | Static ID 0.09 % train / 0.4 % test; QD/QS 0.03 % — yet faulty derivatives break a 25-stage RO | **Negative result**: inference 3-4.8 ms (KAN) vs 0.04-0.05 ms (MLP), 60-120x slower; symbolic form "unusable" (§IV-D) | HSPICE Level-72 golden, GPU timing | No usable emitter | preprint | **REJECT as emitter; high-value counter-evidence** — better static fit degraded dynamic simulation; supports our timing/energy gates |
| A6 | Charge-based SiC physics model — Salcines et al., arXiv:2104.14733v1, `arxiv.org/html/2104.14733v1` (abstract, Eqs 1-6) | IV 5 mV-800 V, few mA-30 A; no runtime | None claimed | B1505A + pulsed/DPT extraction | No speed result | preprint | **REJECT as ROM; PARTIAL as extraction** — needs unavailable doping/trap geometry data |
| A7 / E4 | Compensated NN waveform model — Wang et al., *Energies* 17, 6061 2024, DOI 10.3390/en17236061 (§5.1, Figs 10-14) | vgs/vds rrms 0.9-9.0 %, Eoff error 4.7 % at unseen condition | **None measured**; 0.01 ns fixed-step state model + auto-regressive NN | Python, no SPICE | No | peer-reviewed | **PARTIAL** — knowledge+correction idea, dual backend only; not a speed or accuracy benchmark for us |
| A8 | Separated ANNs — Chankla et al., *Sci. Rep.* 15, 18941 2025, DOI 10.1038/s41598-025-03005-8 | R²>99.96 % over 10^-12-10^-1 mA/mm via OFF/transition/ON nets | Not a speed method | TCAD, static ID-VG only | No | peer-reviewed | **PARTIAL technique** — range partitioning for near-zero curves |
| A9 | X-parameters / PHD — Verspecht & Root, *IEEE Microw. Mag.* 7(3) 44-57 2006, `janverspecht.com/pdf/phd_ieeemicrowavemagazine.pdf`; TU Wien & Keysight official pages (403 caveat) | Linearized harmonic response at an LSOP; < -20 dB 2nd-harmonic mismatch (TU Wien Fig 4) | Harmonic-superposition coefficients in harmonic-balance tools | NVNA + ADS/SystemVue | No SPICE transient emitter | peer-reviewed + vendor (403 on some pages) | **REJECT** — wrong observable (periodic harmonic) and wrong simulator class |
| A10 | Vector-fit -> SPICE RLC — Gustavsen & Semlyen DOI 10.1109/61.772353; Antonini DOI 10.1109/TEMC.2003.815528; scikit-rf docs "Equivalent Electrical Circuits" | Rational H(s) fit to S/Y/Z; compact netlist | Element-count reduction; no speed number | Frequency-domain fitter -> SPICE | Yes (RLC/VCCS) | peer-reviewed + official docs | **PARTIAL, passive only** — right emitter, wrong (linear) device scope |
| B1 | Original TPWL — Rewienski & White, *IEEE TCAD* 22(2) 2003, `rle.mit.edu/cpg/publications/pub159.pdf` | Training-input trajectory only; 7 V-trained model fails a 9 V step | Small reduced linear systems; N=100->q=10; generation 80.7 s vs 2756.5 s | MATLAB-like state model | No | peer-reviewed | **REJECT** — same mechanism already failed at emission (plan §§9-11) |
| B2 | Adaptive-LP TPWL — Chathukulam et al., ECMS 2018, `scs-europe.net/dlib/2018/.../0293_simo_ecms2018_0823.pdf` (Table 1) | Output error 4.2 % vs 6.74 % uniform | **None** — online time 0.1 s in all rows | Synthetic circuits | No | conference paper | **REJECT / PARTIAL offline knot heuristic** — no runtime win |
| B3 | DEIM/EIM — Chaturantabut & Sorensen, *SIAM J. Sci. Comput.* 32(5) 2010, DOI 10.1137/090766498 | RC ladder n=1000, r=m=10: eps_DEIM 8.60e-3 (QDEIM §2.2.2) | Removes O(N) lifting bottleneck | Large-state ODE | No | peer-reviewed | **REJECT as SPICE** — our emitted model has no O(N) lift |
| B4 | QDEIM — Drmac & Gugercin, *SIAM J. Sci. Comput.* 38(2) 2016, DOI 10.1137/15M1019271 (Theorem 2, §2.2.2, §3) | eps_QDEIM 6.07e-3 (r=m=10); 220 of 10,000 rows match full accuracy | Cheaper offline pivoted-QR selection, not online speed | Offline linear algebra | No netlist effect | peer-reviewed | **PARTIAL** — drop-in offline anchor selection only |
| B5 | GNAT — Carlberg et al., *J. Comput. Phys.* 242 2013, `arxiv.org/html/1207.1349v1` (Table 3) | Drag RD 0.68 % consistent sample collection | 17M DOF->283 modes, 512->4 cores; CR 438, **WT 3.42x** | Intrusive CFD | No | peer-reviewed | **REJECT** — large-N/core-count CFD result; sample-mesh idea maps to parasitic removal already measured with fidelity cost |
| B6 | Local reduced basis — Pagani et al., EPFL TR 25.2017 (§5, Table 1) | Field NRMSE 9.6e-3 at 35.5x (case II) | k-means 12.8x / 35.5x vs global 5.6x / 3.4x | Intrusive PDE | No | technical report | **REJECT as method** — cluster switching == the weight blend that broke our emission |
| B7 | Operator inference — Peherstorfer & Willcox, *CMAME* 306 2016, DOI 10.1016/j.cma.2016.03.025 | L2 convergence Theorem 1; no wall-clock number | n-dim ODE vs N-dim PDE | Black-box trajectories, low-order polynomial | No | peer-reviewed | **PARTIAL offline / REJECT deliverable** — needs internal states; exp/tanh/Q(V) not polynomial |
| B8 | Autoencoder ROM — Gruber et al., *CMAME* 393 2022, `arxiv.org/html/2110.03442v3` (Table 5) | Snapshot/prediction error | Online 0.002186 s (FCNN) vs 0.02675 s (POD) at N=256; offline 3-4 orders larger | PyTorch PDE surrogate | No | preprint | **REJECT / PARTIAL fast tier** — surrogate cost, not LTspice DPT time |
| B9 | LDNets — Regazzoni et al., *Nat. Commun.* 15:1834 2024, DOI 10.1038/s41467-024-45323-x | NRMSE 1.88e-5; abstract 5x smaller error, >10x fewer params | Tiny latent ODE; no wall-clock printed | PDE benchmark | No | peer-reviewed | **REJECT / PARTIAL converter research** — no circuit or SPICE experiment |
| C1 | LTspice options — AD-shipped LTspice 26 Help (`.../integrationmethods.htm`, `convergence.htm`, `options--setsimulatoroptions.htm`, `spice.htm`), exact topic titles in Track C §9 | `trtol` "may give speed 2x" but "increasing values compromises accuracy"; Gear "generally slower ... numerical dampening" | Fewer steps via relaxed truncation error / different integrator; **no guarded positive ratio** | LTspice 26 | n/a (knobs, not netlist) | official docs | **REJECT as speed argument** — tolerance/numerical trade; our audit closed them (plan §12) |
| C2 | LTspice C1 authoring / breakpoint placement — B-source Help `uplim`/`dnlim` "continuous 1st derivative transition width"; `tripdv`/`tripdt` fidelity guard | Remove derivative discontinuities; protect edge timing | Fewer Newton/step rejections; our measured pre-fix 64-crossing defect fixed | LTspice | Yes — authoring rule for emitted netlist | official docs + internal measurement | **ADOPT (already landed)** — the one genuine model-level speed lever measured in-project |
| C3 | Solver kernels — ngspice §1.1.2/12.10 (KLU, OpenMP 65.4->32.4 s, 627 inverters); PSpice `SOLVER=1`; Spectre APS 1.5-2x, `errpreset` up to 7.5x; HSPICE HPP up to 7x; PrimeSim Fig 2 1x/3x/10x | Same equations; vendor-asserted accuracy, some presets measurement-specific | Parallel/JIT/partitioned kernel; ratios lack circuit/guard or live in the binary | Various commercial kernels | No | official docs / vendor | **REJECT for portable ROM** — real speed, wrong layer; dual backend only |
| D1 | SIMPLIS — 10-50x, `simplistechnologies.com/.../introduction_whatissimplis.htm` (*What is SIMPLIS?*); `.../simplis_is_all_pwl_all_the_time.htm` | FDD8882 loss deltas <0.5 % total / ~1 % conduction / <2 % turn-on-off vs SPICE (S-APPB Exercises 1-3, one device) | Linear solve per PWL topology; **10-50x vendor claim**, no table in source | SIMetrix/SIMPLIS | Conversion at placement only — second artifact | vendor | **DUAL-PATH only** — never claim its speed in LTspice; loss data supports waveform/loss cross-validation |
| D2 | PLECS ideal switch + loss LUT — `docs.plexim.com/plecs/5.0/how-plecs-works/`; `plexim.com/products/plecs/electrical/semiconductors` (*Fast*); `docs.plexim.com/plecs/5.0/thermal-modeling/thermal-concepts` | v(i,T), E_on/E_off(v,i,T) tables with interpolation/extrapolation; **dv/dt, Miller, ringing, reverse recovery lost** (P-DEMO Figs 6-7) | "Only two steps per switching event"; **qualitative only, no factor** | PLECS | Coexists: Netlist component + configurable subsystem; our vocabulary supported (P-SPICE, P-SYN) | vendor/official docs | **DUAL-PATH — ADOPT for converter tier**, retire DPT waveform gate explicitly |
| D3 | DSED — IEEE TPEL 34(12) 2019 DOI 10.1109/TPEL.2019.2907731; ECCE 2019 tutorial `ieee-ecce.org/2019/conference/tutorials-4/` | "same accuracy" claim; no bench table in source | Up to 10x vs PE software, 100x vs Simulink (20 switches); hundreds of folds (600) | DSED toolchain | No | peer-reviewed + tutorial abstract (abstract-only numbers) | **REJECT for ROM** — converter/system tier only |
| D4 | ORNL event-exact EMT — Yadav & Nutaro, arXiv:2608.22226v1 (23 Aug 2026), `arxiv.org/html/2608.22226v1` (Table I) | NRMS i_L/v_dc 0.15/0.12 % vs own RK4, 1806 states | 62.5 M fixed steps -> 10.6 M events; **11.6x**, 15.3x at 20 us sync | Native C + Python | No | preprint | **REJECT as ROM evidence** — grid EMT, no device DPT; converter-tier architecture evidence only |
| D5 | Typhoon HIL average vs switching — `typhoon-hil.com/.../switching_and_average_models_grid-connected_bat_inv.html` (Table 1) | Filtered Vab closely matches; switching ripple lost | Fixed 1 us step; matrix memory 75.66->4.74 %, converter resources 3/3->0/3 | HIL hardware | No | vendor | **PARTIAL converter tier** — resource trade, not a desktop speedup |
| D6 | MathWorks fidelity ladder / real-time bounds — `mathworks.com/help/simscape-electrical/real-time-simulation.html`, `.../choose-power-electronic-converter-block.html` | Averaged models exclude switching events; PWL includes on-state I-V | No factor; C >=20 us, HDL >=1 us sample time | Simscape/Speedgoat | No | official docs | **PARTIAL taxonomy / alternative converter backend** |
| E1 | Kubulus et al., *IEEE TPEL* 40(4) 5201-5211 2025, DOI 10.1109/TPEL.2024.3510037 (§III, Tables I-II, IV, VI-VII; speed set in **§VI Conclusion**, abstract only prints "up to 59 %") | CPM2 fall 32.5 ns vs 33 ns measurement; turn-on metrics within a few %; large voltage-overshoot mismatch in both models. **Hedge:** these are the paper's model-vs-*hardware* errors, not our CPM2 golden-vs-ROM DPT gate (our golden is a vendor SPICE model, not hardware) — no direct fidelity comparison claimed **[INFERENCE]** | Taylor-linearized channel/charge + external C callback; **avg 7 % / 36.8 %, best 24.6 % / 59 %** (§VI Conclusion) | ngspice `external` sources + compiled C | No (external-process path) | peer-reviewed (author manuscript) | **PARTIAL idea only — REJECT under our rules**: Tables I/III relax max step/reltol/gmin/rshunt between arms for both CPM2 and SCT4018K (incl. SCT4018K `trtol` 2->4, `gmin` 1e-12->1e-14); no-tolerance rule violated |
| E5 | Wolfspeed dynamic tuning — DeBoi/Nelson/Curbow, `forum.wolfspeed.com/uploads/VQL8VYDXXYC8/ws-spice-dynamic-tuning.pdf` (Fig 6) | dv/dt_off 48.9->18.5 %, Eoff 26.4->12.9 %; structure unchanged | **None** — complexity held constant | MATLAB + LTspice | Yes (vendor compact `.lib`) | vendor | **PARTIAL methodology** — tune cap scalars against DPT metrics; fidelity-only, no speed |
| E6 | Ceccarelli et al., APEC 2017, DOI 10.1109/APEC.2017.7930813 (institution abstract) | Wide-condition switching/loss vs experiment | "few seconds" absolute; **no ratio** | PSpice + Simulink + ANSYS | Partially (two backends) | abstract-only | **PARTIAL / low weight** — architecture precedent, no importable speed |
| F1 | Dini et al., *IEEE Access* 11 79120-79143 2023, DOI 10.1109/ACCESS.2023.3298526 (§IV, §VI) | DPT RMSE 1.24 A / 2.39 V / 5.81 V; inverter efficiency mean err 0.78 % | LUT + ideal switch, quantized transients; 10 s inverter 40 min vs "50 times longer" LTspice (>33 h) | Simulink | No | peer-reviewed | **REJECT as ROM evidence** — simulator swap + waveform detail sacrificed; converter-tier note only |
| F2 | Yadav & Nutaro, arXiv:2608.22226 (same as D4) | See D4 | See D4 (11.6x) | Native C/Python | No | preprint | **REJECT as ROM evidence** — do not quote as SPICE-ROM ratio |

**Cross-cutting result from the table.** No row in tracks B, C, D, or F emits an LTspice-compatible
ASCII three-terminal `.lib` *and* reports a guarded, fidelity-preserving speedup. The only rows
that come close are A5/E2 (~30 %, SIMetrix) and E3 (2.27x, LTspice, vendor), which is why they
anchor §4 paths #2 and #3. Track A's 40x, track C's solver ratios, and track D/E's 10-50x /
11.6x / 50x are all measured at a different layer.

---

## 4. Ranked candidate paths for PieceSpice

### #1 — Dual-tier architecture (DUAL-PATH; highest strategic priority)

**Keep** the current ASCII SPICE ROM (Tier-2 channel + charge tables) as the mandatory,
waveform-resolving device deliverable for LTspice 26 batch. **Add** a converter-level
ideal-switch + conduction/switching-loss LUT model for system/thermal studies. This is the only
route that honestly supplies both observables: waveform fidelity in Path A and converter speed in
Path B, reported separately.

- **Evidence:** Plexim's own docs and demo: ideal switches "toggle instantaneously", "only two
  steps are needed for each switching event" [P-SEM, *Fast*]; conduction `v_on(i,T)` and switching
  `E_on/E_off(v,i,T)` tables with interpolation/extrapolation and a negative-loss diagnostic
  [P-TC, *Conduction Losses*, *Switching Losses*, *Loss Calculation*]; configurable subsystems let
  one schematic toggle PLECS vs SPICE configurations and overlay results
  [P-SPICE, *Best Practices -> Use Configurable Subsystems*;
  `plexim.com/products/plecs/plecs_spice`, *Keep all levels of detail in one model*].
- **Explicit fidelity sacrifice:** Plexim states the PLECS ideal-switch configuration turns on
  instantaneously and misses turn-on/off transients, Miller-effect gate droop and diode reverse
  recovery that require the SPICE configuration [P-DEMO, *PLECS Configuration*, *SPICE
  Configuration*, *Conclusion*]. Therefore the fast tier's gate is **energy/efficiency vs golden
  on a converter bench**, never DPT Vds/Id or tr/tf; the DPT waveform gate is retired for that
  tier, not silently failed.
- **PLECS-Spice netlist compatibility facts:** our emitted element vocabulary is documented in
  PLECS Spice syntax — `tanh`, `table(x, ...)`, `limit(x,y,z)`, `pwr`/`pwrs`, `ddt()` [P-SYN,
  *Mathematical Functions*, *SPICE-Specific Functions*, *Circuit State Access*] and charge-based
  capacitors `Cname pos neg Q={function body}` [P-SYN capacitor page, *Charge-based Capacitor*].
  This covers Tier-2's `G_CH = PWL(Vgs)*tanh(Vds/s)` and the `Q={...}` charge chains; **direct
  import is plausible from syntax but was not attempted** [INFERENCE]; no bit-identical behavior
  is claimed. Caveat: "Electrical connections between PLECS and SPICE components within the same
  circuit are not allowed and will cause the simulation to fail" [P-SPICE tutorial PDF, note at
  end of §2] — SPICE networks attach through control/gate-drive signals and ground; MNA solves the
  SPICE part and state-space the PLECS part [P-SPICE, *SPICE Highlighting*].
- **Cost / gating:** separate licensed tool, a converter bench not yet built (plan §14), and an
  explicit energy gate. No numeric PLECS-vs-SPICE speed factor exists in the official pages
  consulted; none is imported. SIMPLIS may be added as an optional second backend (10-50x is
  SIMPLIS-only [D1]); SIMPLIS's own loss deltas (<0.5-2 %, one device) support cross-validation.

### #2 — Model-level behavioral ANN + physics guards + small voltage-dependent caps (ADOPT, fidelity path)

Yang et al. (A5/E2) is the closest product-shape match: one VCCS `IDS(VDS,VGS)` + three voltage-
dependent capacitors `CGS, CGD, CDS(VDS)`; a 1-hidden-layer **10-neuron** sigmoid ANN with a
physical cutoff (`I = f_ANN·½(1+tanh(α(VGS−Vth)))`, α=10 below threshold) and a low-VDS linear
guard (`I = f_ANN(Vsat,VGS)·VDS/Vsat`, Vsat=2 V) so the network never owns the near-zero region
(§III-B Eqs 3-6); small ANNs for the capacitances after a log transform of Ciss/Coss/Crss
(§III-C, Eq 7). Fidelity (all **CONDITIONAL** — IEEE copy per Track A only; Track E institution
PDF 403-blocked, abstract-only verified): DPT timing/energy <=11 % of hardware (Table I), waveform
RMS 1.5-3x better than Angelov (Table II), VDS spectrum 2.82 % vs 4.26 % (Fig 16). Speed: **~30 %
less runtime** at fixed SIMetrix default solver (abstract-verified); the 40 us bench / 20 repeats /
Table IV framing is **CONDITIONAL** as above — a genuine *model-level* gain.

**Requirements:** measured low-V IV plus DPT-extracted high-V IV to 700 V (slowed turn-on with
100 Ω gate resistor, §III-A), log-scale CV from 0.01-1000 V, 25 °C only, 1200 V module. Written in
Verilog-A for SIMetrix; re-emitting the sigmoid MLP as LTspice scalar B-source expressions is
mechanical **[INFERENCE]** and must be timed in our harness. **Limits:** single temperature;
neuron count by trial-and-error; 1200 V module vs our 30-100 V lateral devices; 30 % is against a
deliberately heavy Angelov baseline, so it is a *fidelity improvement with modest speed*, not a
5x claim. Adopt the guards and the derivative-aware loss lesson from A1/A2; the KAN negative
result (A4) confirms static RMSE alone is an unsafe selector.

### #3 — ROHM L3 standard LTspice `.lib` (ADOPT as benchmark, not a recipe)

E3 is the cleanest form-factor match: an ordinary `.lib` in LTspice 24.1.3, PFC bench
179 s -> 79 s (44 % of original; 2.27x [INFERENCE: arithmetic]), DPT turn-off loss waveforms
"similar" to L1 (qualitative only), no stated tolerance/step change. Treat it as **proof that
model-level >2x is achievable in the same simulator class**, and as a warning that vendor golden
internals change between level labels (record the golden revision). It is a vendor white paper
with undisclosed algorithm, one bench, no numeric waveform error, no iteration counts — **not a
portable recipe** for our emitter.

### #4 — Taylor-linearized channel/charge with external callback (PARTIAL idea only; REJECT as implemented)

Kubulus et al. (E1) replace nonlinear channel/charge equations with first-order multivariate
Taylor expansions; coefficients are exposed as ngspice `external` sources updated by a compiled C
callback (§III, Eq 1). Reported acceleration: avg **7 % / 36.8 %**, best **24.6 % / 59 %**
(**§VI Conclusion**, not the abstract — the abstract prints only "up to 59 %"); 5-20 % over most
of the CPM2 range, up to 60 % for SCT4018K (§IV Figs 15-16); CPM2-1200-0160B is one of our
devices. Fidelity vs experiment is detailed (Tables VI-VII). **Hedge:** those are the paper's
model-vs-*hardware* errors, not directly comparable to our CPM2 golden-vs-ROM DPT gate (our golden
is a vendor SPICE model, not hardware); no fidelity result is transferred **[INFERENCE]**.
**Why rejected as a path:** the two arms are *not* at identical solver settings — CPM2 original
0.7 ns / reltol 0.001 / gmin 1e-12 / rshunt 1 MEG vs approximate 4 ns / reltol 0.004 / gmin 1e-18 /
no rshunt (§IV Tables I-II), and SCT4018K 5 -> 6 ns / reltol 0.01 in both / `gmin` 1e-12 -> 1e-14 /
`trtol` 2 -> 4 (§IV Table III) — so part of the gain is a tolerance/step change; the ngspice
external-process path cannot run under `simulation/engine.py` LTspice batch. Only the general idea
(reduce expensive B-source expression cost, re-validate our exact switching metrics) transfers,
and that is close to our already-closed Tier-2 PWL form (AGENTS.md "Traps").

### #5 — SIMPLIS / event-driven / PLECS simulator-level speed (DUAL BACKEND only)

SIMPLIS' 10-50x is explicitly attributed to its PWL topological linear solver [S-USER, S-WIS,
S-APWL]; event-driven EMT reports a guarded 11.6x at <0.2 % NRMS on 1806-state grid models
[E-ORNL Table I, preprint], and the ECCE tutorial claims up to 10x/100x/hundreds of folds
[E-DSED-TUT]; PLECS gives "two steps per switching event" qualitatively. All are genuine but live
in a different simulator and/or delete device-level observables. Use only as a clearly labeled
second backend behind the same extracted data contract: Path A stays LTspice/ASCII, Path B (PLECS
or optional SIMPLIS) reports its own observables. **No external speed factor may be attributed to
our LTspice emission** (plan §16.5; Track D §8).

### #6 — Nonlinear MOR: GNAT / local RB / QDEIM / operator inference / autoencoders (REJECT; QDEIM PARTIAL)

Every method in track B targets state dimensions N ~ 10^3-10^7 (GNAT 17M DOF -> 283 modes, WT
3.42x; local RB 31,765-node PDE FOM, 35.5x; DEIM removes an O(N) lift; autoencoder online
0.002186 s at N=256). Our emitted device is a ~10-element subcircuit whose DPT cost is adaptive-
step stiffness, so removing an O(N) lift cannot create a new speed class **[INFERENCE]**, consistent
with our measured <=1.07x fidelity-preserving ceiling (plan §12). None of the nine source families
emits a standard-SPICE ASCII subcircuit. Cluster/weight switching is the exact algorithmic shape
that produced non-symmetric Jacobians and timestep collapse in our T3-v1 emission (plan §10).
**Partial:** QDEIM's pivoted-QR anchor selection is a pure offline linear-algebra upgrade to
`approximation/tpwl_deim.py`; it cannot change emitted adaptive-step behavior and claims no
speedup. Reaffirm the training-data rule: keep the Curve-1 static family in training (plan §11;
TPWL's 7 V->9 V failure).

### #7 — Solver knobs (classify and mostly REJECT)

Official LTspice surface (AD-shipped LTspice 26 Help, exact topics in Track C §9):

- **Tolerance / numerical — REJECT as speed:** `trtol` ("may give speed 2x" but accuracy warning),
  `reltol`/`abstol`/`vntol`/`chgtol`/`gmin`/`gshunt`/`cshunt` (Newton stop criteria/artificial
  shunts), `method=gear` (generally slower, damps oscillations), ngspice `newtrunc`/
  `nostepsizelimit`, PSpice `.AUTOCONVERGE`, Spectre `errpreset` (up to 7.5x on one ring-oscillator
  frequency testbench, error 0.06-5.1 % — direct vendor evidence that small RMSE != edge fidelity),
  tolerance windows. Our audit closed all of these (plan §12).
- **Simulator-kernel — REJECT for portable netlist:** LTspice JIT matrix compiler and multithreading,
  ngspice KLU/OpenMP, PSpice `SOLVER=1`, Spectre APS `++aps` (1.5-2x), HSPICE HPP (up to 7x),
  PrimeSim GPU (Figure 2 1x/3x/10x, no circuit/guard). Real speed, wrong layer.
- **Output/workload — REJECT as speed evidence:** LTspice plot-compression options, ngspice
  `INTERP` ("simulation itself is not influenced"), `AUTOSTOP` (removes full-waveform evidence).
- **Fidelity guards — ADOPT as guards only:** `tripdv`/`tripdt`, strict step caps, `itl4` as a
  robustness diagnostic ("no accuracy penalty, but may slow down simulations" — not a speedup).
- **Model-level robustness — ADOPT (already done):** C1-smooth ramps via `uplim`/`dnlim` class of
  expressions and breakpoint placement out of the conduction band (§2). No new speed claim.

---

## 5. Decision gates for any implementation

Numbers marked **(measured)** are recorded values from plan §16 / the cited sources; numbers
marked **(project decision)** are proposed thresholds selected here, not sourced results.

1. **Same simulator and options.** Golden at LTspice 26 defaults as the sole reference; the only
   difference between arms is the model-level change under test. No `trtol`/`reltol`/`method`
   change in either arm; netlist-identical guard runs (measured rule, plan §16.5; Track C §8-I).
2. **Completion guard (measured, plan §16.5).** Every timing row reports iterations, trace points
   and t_last. DPT: >=50 % of golden points and t_last >= 0.999 x 14 us. Adaptive: 3/3 repeats,
   median solver/wall/iterations/points > 0, netlist-identical guard t_last >= 0.999 x 176 us,
   points >= 0.5 x median, iterations > 0, 90 s timeout; EPC golden keeps the documented 100 ns
   cap. Aborts/timeouts are failures, never speedups (CPM2 golden adaptive is N/A today).
3. **Static no-regression.** Current recorded bars (measured): IRFS 7.0683 %, TI 8.4935 %, EPC
   2.0394 %, CPM2 3.4287 %, worst endpoint <=4.13 %. Proposed gate **(project decision): |Δ| <=
   +0.05 pp** against these values (the recorded post-fix parity band is |Δ| <= 0.0002 pp, plan
   §15.1). The closed Tier-2 channel form is not re-opened without new evidence (AGENTS.md).
4. **DPT no-regression.** Current recorded DPT Vds/Id (measured): IRFS 8.529/0.532 %, TI
   5.583/0.048 %, EPC 4.894/0.049 %, CPM2 2.675/0.261 %, all t_last = 1.4e-5 s. Proposed gate
   **(project decision): no increase beyond measurement variance in any Vds or Id cell, and no
   tr/tf/e_on/e_off |relative error| worse than the current recorded best prior** (plan §15.3,
   §15.5). Do not repeat the simplis-v1 mistake of reporting small full-waveform RMSE while
   CPM2 tf was +73.11 % and TI tf +47.15 % (plan §15.3). A hardware-validated fidelity reference
   for the ANN path is Yang's <=11 % DPT timing/energy band (A5 Table I) — **CONDITIONAL** (IEEE
   copy per Track A only; Track E hit 403 and verified only the abstract, which prints the ~30 %
   runtime but not the 11 % table figure). Adopting that bound as an explicit acceptance target is
   a **(project decision)** derived from a source fact, not a PieceSpice measurement.
5. **No fabricated speedup.** Any ratio must be produced by our guarded `speedup_bench` under
   gates 1-2, with iterations/points/t_last reported for both arms; inherited paper/vendor ratios
   are never copied. The measured context is: ~1.23x is the ceiling where the golden already
   rides the adaptive step ceiling (plan §16.5); ~5.3x exists only on TI and is golden-side
   stiffness (plan §16.2); published model-level ratios are ~30 % (A5/E2) and 2.27x (E3).
6. **Deliverable form.** Standard ASCII SPICE subcircuit (`.write_text(..., encoding="ascii")`)
   running under LTspice 26 batch, unless a dual-path is explicitly approved; then Path A and Path
   B must report different observables with separate fidelity gates (Track D §6 anti-fabrication
   rules 1-4). `SpiceModels/**` read-only; pin order from `.subckt` header (AGENTS.md).
7. **Answer the research question, not a proxy.** A candidate that improves static RMSE but
   degrades timing/energy is rejected (A4 KAN; our T3-v1 1.8-2.9 M iterations / TI timeout,
   plan §§12, 16.5). A candidate that is fast only in another simulator is classified as a dual
   backend, not a ROM speedup.

---

## 6. Final recommendation

**Within this survey there is no single standard-SPICE device model with primary-source evidence
for both a portable >=5x and waveform fidelity; this is evidence of absence in these five reports,
not a proof of impossibility.** The two genuine model-level, switching-related speed results are
Yang's ~30 % (SIMetrix, ANN+guards; abstract-verified; the "DPT-validated" 1.5-3x-better-waveform
detail is **CONDITIONAL** — IEEE copy per Track A only, Track E 403/abstract-only) and
ROHM's 179 s -> 79 s (LTspice, L3 `.lib`, qualitative DPT equivalence, vendor white paper).
Everything reporting >=10x changes the simulator or deletes the switching observables. Our own
guarded baseline says the same thing: no emitted PWL variant exceeds ~1.23x where the golden is
already fast, and TI's ~5.3x solver / 3.069x wall is a property of the stiff golden that even a
zero-content ROM reproduces (plan §16.2, §16.5).

**Pursue two things in parallel:**

1. **Dual-tier architecture (#1)** for speed: keep the ASCII SPICE ROM as the waveform-fidelity
   deliverable and add a PLECS ideal-switch/loss-LUT converter tier (optionally SIMPLIS) whose
   gate is energy/efficiency, with the DPT waveform gate explicitly retired for that tier.
2. **Model-level behavioral improvements (#2)** for fidelity: the 10-neuron ANN + physical
   cutoff/low-VDS guards + log-CV-trained voltage-dependent capacitors, augmented with
   DPT-extracted high-VDS/high-ID points and derivative-aware loss; keep all existing static, DPT,
   timing/energy, and completion gates as hard constraints. Reject the Kubulus backend under our
   no-tolerance/no-external-process rules (#4) and all nonlinear-MOR methods as SPICE deliverables
   (#6).

**If the user insists on one model and one simulator,** the realistic documented target is a
**modest ~30 %-2.3x model-level speedup** with unchanged waveform fidelity, and it requires
reproduction in our guarded harness — starting from #2's structure and #3's convergence
engineering. Claiming more requires a different simulator architecture and a different fidelity
contract, which is a product decision, not a research finding.

---

## 7. Source ledger (primary sources, exact locators, access caveats)

Internal (measured, not external): `docs/plans/tier3_method_candidates.md` §§9-16;
`AGENTS.md`; `reports/comparison/<vendor>/tier3_validation_simplis.json`;
`local://simplis_pipeline_audit.md`, `local://simplis_emitter_audit.md`,
`local://simplis_rootcause.md`, `local://simplis_official_recheck.md`,
`local://negative_vdg_feasibility.md`, `local://hybrid_channel_feasibility.md`.

**Track A (behavioral ROM):** full report `local://research_behavioral_rom.md`.
1. Tung, Salahuddin, Hu, SISPAD 2024, DOI 10.1109/SISPAD62626.2024.10733248 —
   <https://in4.iue.tuwien.ac.at/pdfs/sispad2024/SISPAD_2024_10.pdf> (abstract, §II, Table I,
   Figs 1-9); open copy <https://escholarship.org/uc/item/7fh7q151>.
2. Tung, Kao, Hu, *IEEE TED* 69(11) 6476-6479 2022, DOI 10.1109/TED.2022.3208514 —
   <https://escholarship.org/content/qt7bn8c5wk/qt7bn8c5wk_noSplash_fbbdb9b6454a8b0010859d186f70f37e.pdf>
   (§II-A/B, §IV).
3. Yang et al., *IEEE TED* pp. 4181-4188, 2021 — <https://ieeexplore.ieee.org/document/9477046/>
   (**abstract + §I only**; publisher exposes no DOI string in the accessible record).
4. Novkin & Amrouch, arXiv:2503.15209v1 (**preprint**) — <https://arxiv.org/html/2503.15209v1>
   (Table II, §IV-B/C/D/F, §V, Figs 4, 8).
5. Yang, Ming, Liang et al., *IEEE TPEL* 37(1) 440-451 2022 —
   <https://ieeexplore.ieee.org/document/9506845/> (abstract, §I, §III, §IV, §V-A/B, Tables I-IV);
   DOI 10.1109/TPEL.2021.3101713 verified via Crossref/Semantic Scholar in Track E; author
   institution PDF **403-blocked** there, full text read in Track A.
6. Salcines et al., arXiv:2104.14733v1 (**preprint**) — <https://arxiv.org/html/2104.14733v1>
   (abstract, §§I-V, Eqs 1-6, Figs 1-6).
7. Wang et al., *Energies* 17, 6061 2024, DOI 10.3390/en17236061 —
   <https://mdpi-res.com/d_attachment/energies/energies-17-06061/article_deploy/energies-17-06061.pdf>
   (§§2-6, Figs 10-14, Table 1).
8. Chankla et al., *Sci. Rep.* 15, 18941 2025, DOI 10.1038/s41598-025-03005-8 —
   <https://www.nature.com/articles/s41598-025-03005-8>.
9. Verspecht & Root, *IEEE Microw. Mag.* 7(3) 44-57 2006 —
   <https://www.janverspecht.com/pdf/phd_ieeemicrowavemagazine.pdf>; TU Wien
   <https://www.tuwien.at/en/etit/e354/microwave/nonlinear-device-characterization-large-signal-analysis>
   (§"X-parameters / Polyharmonic Distortion Modelling"); Keysight
   <https://www.keysight.com/us/en/lib/resources/miscellaneous/x-parameters-open-documentation-trademark-usage-and-partnerships.html>
   and <https://www.keysight.com/us/en/product/S94514B/nonlinear-x-parameters.html> (**403 / binary
   abort** on some pages; claims limited to returned official snippets).
10. Gustavsen & Semlyen DOI 10.1109/61.772353; Antonini DOI 10.1109/TEMC.2003.815528;
    Grivet-Talocia & Gustavsen DOI 10.1002/9781119140931; locators in scikit-rf docs
    <https://scikit-rf-official.readthedocs.io/en/stable/tutorials/VectorFitting.html>
    ("Equivalent Electrical Circuits", "Direct implementation of the State-Space Equations").
11. Tung, Chen, Salahuddin, Hu, *IEEE TED* pp. 6487-6493 2025, DOI 10.1109/TED.2025.3627591 —
    <https://ieeexplore.ieee.org/document/11261537/> (**abstract only**; no speed figure).

**Track B (nonlinear MOR):** full report `local://research_nonlinear_mor.md`.
12. Rewienski & White, *IEEE TCAD* 22(2) 2003 — <https://www.rle.mit.edu/cpg/publications/pub159.pdf>.
13. Rewienski & White, *LAA* 415(2-3) 2006 —
    <https://www.sciencedirect.com/science/article/pii/S0024379504001727>.
14. Rewienski & White, DAC 2003 pp. 490-495 —
    <https://store.computer.org/csdl/proceedings-article/dac/2003/01219055/12OmNBfqG5j>.
15. Chathukulam et al., ECMS 2018 —
    <https://www.scs-europe.net/dlib/2018/ecms2018acceptedpapers/0293_simo_ecms2018_0823.pdf?download=true>
    (Table 1).
16. Chaturantabut & Sorensen, *SIAM J. Sci. Comput.* 32(5) 2010, DOI 10.1137/090766498 —
    <https://epubs.siam.org/doi/10.1137/090766498>.
17. Drmac & Gugercin, *SIAM J. Sci. Comput.* 38(2) 2016, DOI 10.1137/15M1019271 —
    <https://arxiv.org/html/1505.00370v2> (Theorem 2, §2.2.2, §3).
18. Carlberg et al., *J. Comput. Phys.* 242 2013 — <https://arxiv.org/html/1207.1349v1>
    (§3, §5, §6.2.5 Table 3).
19. Pagani, Manzoni, Quarteroni, EPFL TR 25.2017 —
    <https://www.epfl.ch/labs/mathicse/wp-content/uploads/2018/10/Report-25.2017_SP_AM_AQ.pdf>
    (journal version CMAME 2018 not used for the numbers).
20. Peherstorfer & Willcox, *CMAME* 306 2016, DOI 10.1016/j.cma.2016.03.025 —
    <https://kiwi.oden.utexas.edu/papers/Non-intrusive-model-reduction-Peherstorfer-Willcox.pdf>.
21. Gruber et al., *CMAME* 393 2022, article 114764 (**preprint**) —
    <https://arxiv.org/html/2110.03442v3> (Table 5).
22. Regazzoni et al., *Nat. Commun.* 15:1834 2024, DOI 10.1038/s41467-024-45323-x —
    <https://www.nature.com/articles/s41467-024-45323-x>.

**Track C (solver):** full report `local://research_spice_solver.md`.
23. LTspice 26 Help (AD-shipped, installed with the project tool), version-stamped
    "Copyright (c) 1998-2026 by Analog Devices Inc.": `file:///C:/Users/GNKRISHNAN/AppData/Local/Programs/ADI/LTspice/LTspiceHelp/integrationmethods.htm`
    (*Integration Methods*), `.../convergence.htm` (*How To Improve Convergence*),
    `.../options--setsimulatoroptions.htm` (*.OPTIONS - Set Simulator Options*),
    `.../spice.htm` (*SPICE*), `.../barbitrarybehavioralvoltageorcur.htm` (*B. Arbitrary Behavioral
    Voltage or Current Sources*: `table`, `uplim`, `dnlim`, `tripdv`/`tripdt`),
    `.../nodeset--supplyhintsforinitialdc.htm`, `.../transientanalysisoptions.htm`,
    `.../savestate.htm`, `.../LTspiceChangeLog.txt` (entries 24.0.0, 24.1.0, 26.0.0, 26.0.2).
24. ROHM, *How to Use LTspice Models: Tips for Improving Convergence*, 65AN082E —
    <https://fscdn.rohm.com/en/products/databook/applinote/common/how_to_use_ltspice_models_tips_for_improving_convergence_an-e.pdf>.
25. Ngspice User's Manual 47+ — <https://ngspice.sourceforge.io/docs/ngspice-html-manual/manual.xhtml>
    (§§1.1, 8, 9, 11.1-11.2, 12.9-12.10 Table 12.1, 12.12, 24.5).
26. Cadence, *PSpice A/D Reference Guide* v16.5 — <https://home.agh.edu.pl/~godek/pspcref.pdf>
    (pp. 35, 49, 66, 71-76, 95-96, 112-114, 126-127); *PSpice User Guide*
    <https://resources.pcb.cadence.com/pspiceuserguide>; Cadence Community tolerance blog
    <https://community.cadence.com/cadence_blogs_8/b/pcb/posts/pspice-tolerances>.
27. Cadence Community, Spectre Tech Tips: *Accuracy 101*
    <https://community.cadence.com/cadence_blogs_8/b/cic/posts/spectre-tech-tips-accuracy-101>;
    *Optimizing Spectre APS Performance*
    <https://community.cadence.com/cadence_blogs_8/b/cic/posts/spectre-optimizing-spectre-aps-performance>;
    *How to Improve the Spectre X Simulation Performance*
    <https://community.cadence.com/cadence_blogs_8/b/cic/posts/spectre-tech-tips-how-to-improve-the-spectre-x-simulation-performance>.
    The linked Spectre User Guide <https://support.cadence.com/apex/techpubDocViewerPage?path=spectreuser/spectreuser21.1/spectreuserTOC.html>
    is **login-gated** and was not read anonymously.
28. Synopsys, *PrimeSim SPICE* datasheet
    <https://www.synopsys.com/content/dam/synopsys/implementation%26signoff/datasheets/primesim-spice-ds.pdf>
    (Figure 2: 1X/3X/10X, no circuit/guard printed); *HSPICE Precision Parallel* white paper
    <https://www.synopsys.com/content/dam/synopsys/implementation%26signoff/white-papers/HSPICE-HPP-wp.pdf>
    ("up to 7X", Figures 3); *PrimeSim XA* datasheet
    <https://www.synopsys.com/content/dam/synopsys/implementation%26signoff/datasheets/primesim-xa-ds.pdf>.

**Track D (simulator architectures):** full report `local://research_simulator_architectures.md`.
29. SIMPLIS official docs: <https://www.simplistechnologies.com/documentation/simetrix/user_manual/topics/introduction_whatissimplis.htm>
    (*What is SIMPLIS?*), <https://www.simplistechnologies.com/documentation/simplis/what_is_simplis/topics/what_is_simplis.htm>,
    <https://www.simplistechnologies.com/documentation/simplis/what_is_simplis/topics/simplis_is_all_pwl_all_the_time.htm>,
    <https://simplis.com/documentation/simplis/what_is_simplis/topics/accuracy_of_pwl_models.htm>,
    <https://www.simplistechnologies.com/documentation/simplis/ast_01/topics/1_0_5_pop_and_ac_analysis.htm>,
    <https://www.simplistechnologies.com/documentation/simplis/ast_applications/topics/b_multi_level_mosfet_driver_applications.htm>
    (*Application B*, Exercises 1-3; 10-50x is a **vendor claim without a bench table** in the source).
30. PLECS official docs: <https://docs.plexim.com/plecs/5.0/how-plecs-works/> (*Physical Model
    Equations*, *Event Detection Loop*, *Interpolation of Non-Sampled Switching Events*),
    <https://www.plexim.com/products/plecs/electrical/semiconductors> (*Ideal Switch Models*,
    *Fast*), <https://docs.plexim.com/plecs/5.0/thermal-modeling/thermal-concepts>,
    <https://docs.plexim.com/plecs-demos/plecs-spice/demo-plecs-spice/> (Figs 5-7, *Conclusion*),
    <https://docs.plexim.com/plecs/latest/spice/using-spice/> (*Netlist Component*, *Best Practices*),
    <https://docs.plexim.com/plecs/latest/spice/netlist_syntax/> (*Mathematical Functions*,
    *SPICE-Specific Functions*), and the capacitor page
    <https://docs.plexim.com/plecs/latest/spice/netlist_syntax/syntax/elements/capacitor/>
    (*Charge-based Capacitor*), <https://docs.plexim.com/plecs/5.0/codegeneration/>,
    <https://docs.plexim.com/plecs/latest/components-by-category> (*Sub-cycle Averaging*),
    <https://www.plexim.com/products/plecs/plecs_spice>.
31. DSED: <https://www.ieee-ecce.org/2019/conference/tutorials-4/> (*Tutorials-AM4*, **tutorial
    abstract only** for the 10x/100x figures), DOI 10.1109/TPEL.2019.2907731 and
    DOI 10.23919/TEMS.2017.8086106 (**CrossRef metadata, no abstract exposed**).
32. ORNL event-exact EMT, Yadav & Nutaro, arXiv:2608.22226v1 (**preprint**) —
    <https://arxiv.org/html/2608.22226v1> (§II-A-C, §III, Table I, Fig 4).
33. Typhoon HIL, <https://www.typhoon-hil.com/documentation/typhoon-hil-application-notes/References/switching_and_average_models_grid-connected_bat_inv.html>
    (Table 1, Figs 6-8), <https://typhoon-hil.com/documentation/typhoon-hil-software-manual/concepts/switch_level_gds_oversampling.html>.
34. MathWorks, <https://www.mathworks.com/help/simscape-electrical/real-time-simulation.html> and
    <https://www.mathworks.com/help/simscape-electrical/ug/choose-power-electronic-converter-block.html>.

**Track E (power-electronics evidence):** full report `local://research_power_electronics.md`.
35. Kubulus et al., *IEEE TPEL* 40(4) 5201-5211 2025, DOI 10.1109/TPEL.2024.3510037 —
    <https://vbn.aau.dk/ws/files/755272008/Approximate_SPICE_modelling_method_for_SiC_MOSFETs.pdf>
    (mirror <https://core.ac.uk/download/646127962.pdf>; code
    <https://github.com/AAU-CoDE/Approximate-SPICE-modelling>).
36. Yang et al., *IEEE TPEL* 37(1) 440-451 2022, DOI 10.1109/TPEL.2021.3101713 —
    <https://doi.org/10.1109/TPEL.2021.3101713>; abstract via Semantic Scholar
    <https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/TPEL.2021.3101713?fields=title,abstract,externalIds,openAccessPdf>,
    Crossref <https://api.crossref.org/works/10.1109/tpel.2021.3101713>, OpenAlex
    <https://api.openalex.org/works/W3191010194> (**institution PDF 403-blocked in this track; abstract-only numbers here**).
37. ROHM white paper 67WP017E Rev.001, Apr 2025 —
    <https://fscdn.rohm.com/en/products/databook/white_paper/discrete/sic/mosfet/new_spice_model_for_faster_simulation_of_power_semiconductors_wp-e.pdf>
    (§"Evaluation of models", Table 1).
38. Wang et al., *Energies* 17(23) 6061 2024, DOI 10.3390/en17236061 —
    <https://mdpi-res.com/d_attachment/energies/energies-17-06061/article_deploy/energies-17-06061.pdf>.
39. DeBoi, Nelson, Curbow (Wolfspeed) —
    <https://forum.wolfspeed.com/uploads/VQL8VYDXXYC8/ws-spice-dynamic-tuning.pdf> (Fig 6).
40. Ceccarelli et al., APEC 2017, DOI 10.1109/APEC.2017.7930813 —
    <https://research.tue.nl/p/en/research-outputs/a-fast-electro-thermal-co-simulation-modeling-approach-for-sic-po>
    (**abstract only**; no ratio).
41. Dini et al., *IEEE Access* 11 79120-79143 2023, DOI 10.1109/ACCESS.2023.3298526 —
    <https://cris.vub.be/ws/portalfiles/portal/105567456/97944009.pdf> (§III Figs 26-27, §IV, §VI).
42. Yang, Zhao, Gu, Ng, *Microelectronics Journal* 165, 106836 2025, DOI 10.1016/j.mejo.2025.106836 —
    <https://www.sciencedirect.com/science/article/abs/pii/S1879239125002851>
    (**captcha-blocked; no abstract or full text retrievable; NOT used as evidence**).
43. Nelson et al., *IEEE OJPEL* 1 499-512 2020, DOI 10.1109/OJPEL.2020.3036034, and *IEEE OJPEL*
    2 106-123 2021, DOI 10.1109/OJPEL.2021.3056075 — titles/DOIs verified from Kubulus refs
    [25]-[26]; full text **403-blocked**, so **no quantitative claim** is taken from them.

**Access-caveat summary:** abstract-only (3, 11, 31, 40), preprint (4, 6, 21, 32), 403/blocked or
login-gated (9, 27, 36, 42, 43), vendor claim without full bench table (29, 23 Figure 2, 28),
mechanism undisclosed (37). No number in this document was inferred from an unread source; every
[INFERENCE] is an explicit arithmetic or design interpretation of a read source.

---

## 8. Evidence-audit corrections

Applied 2026-11-20 from the independent audit `local://research_evidence_audit.md`. Audit verdict,
quoted verbatim from its §5 final claim table: **22 claims, 18 PASS, 4 CONDITIONAL, 0 FAIL**.
Exact changes to this document:

- **F1 (Yang TPEL 2022):** kept the ~30 % runtime headline as abstract-printed and verified via
  Semantic Scholar (PASS). Downgraded the table/figure-level numbers — 11 % DPT timing/energy,
  1.5-3x waveform RMS vs Angelov, 2.82 % vs 4.26 % spectrum, and the 20-run / 40 us / Table IV
  bench framing — to **CONDITIONAL** in §1, §3 row A5/E2, §4 path #2, §5 gate 4 and §6, noting the
  IEEE copy cited by Track A (`ieeexplore.ieee.org/document/9506845`) is not corroborated by Track
  E (institution PDF Cloudflare-403-blocked; only the abstract verified there). No URL invented.
- **F2 (Kubulus):** corrected the 7 % / 36.8 % / 24.6 % / 59 % locator from "(abstract)" to
  **§VI Conclusion** (abstract prints only "up to 59 %") in §3 row E1 and §4 path #4. Added the
  omitted SCT4018K solver-setting changes (`trtol` 2 -> 4, `gmin` 1e-12 -> 1e-14, 5 -> 6 ns,
  reltol 0.01 in both arms; Table III) to the tolerance caveat alongside the CPM2 changes.
  Rejection under the project's no-tolerance/no-external-process rules is unchanged.
- **F3 (scope):** no unscoped "only model-level speedup" claim was present — §1, §3 cross-cutting
  and §6 already acknowledge ROHM's 2.27x from Track E alongside Yang's ~30 %; the §6 no->=5x
  statement is now additionally scoped "within this survey".
- **F4 (LDNet):** §3 row B9 already reads ">10x fewer params" and "5x smaller error" (parameter
  count and normalized error, not runtime); confirmed, no speed factor imported.
- **F5 (Kubulus fidelity hedge):** §3 row E1 and §4 path #4 now state explicitly that the paper's
  fidelity numbers are model-vs-*hardware*, not our CPM2 golden-vs-ROM DPT gate (our golden is a
  vendor SPICE model), so no direct fidelity comparison is claimed.
- **F6 (final recommendation):** §6 now scopes the no-fidelity-preserving >=5x standard-SPICE
  statement to "within this survey ... evidence of absence ... not a proof of impossibility".
  Path B's explicit energy/efficiency gate with the DPT waveform gate retired (§1, §4 path #1,
  §6 point 1) was confirmed and left intact.

No code, config, JSON, test, simulation or `SpiceModels/**` change was made in this correction pass.
