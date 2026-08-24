# PieceSpice Development Log

Timestamped record of all development work. Newest entries at the top.
All timestamps are IST (UTC+05:30).

---

## 2026-08-24 19:55 IST — M3.5/M3.6 complete (Tier-2 synthesis + fitting pipeline) with deep-audit fixes: Vgs-key truncation bug, C_QOSS slope accumulation

### Delivered

**M3.5 — models/netlist_builder.py rewritten**: emits a real, ngspice-parseable
Tier-2 CPWL MOSFET subcircuit from a PWLModelSpec:
- Parasitics RG/RD/RS from metadata defaults (device-agnostic).
- `G_CH di ss VALUE={...}` behavioral channel source: chained soft-abs ramp
  PWL transfer (Curves 1+2 endpoint currents per Vgs) × tanh saturation knee;
  vds_sat derived from Rds(on) × top-curve plateau current so the ohmic slope
  matches Curve 3. Ramps encode DELTA slopes.
- `C_QOSS di ss Q={...}` charge-conserving output capacitance from the
  Curve-4-derived Qoss table (plan §2.3); linear-C fallback when the table is
  unavailable. C_ISS/C_RSS linear equivalents from metadata.
- Body diode: IS/N/RS derived from Curve-6 samples (two-point exponential
  fit, N clamped to [0.5, 4]); orientation anode=source, cathode=drain.
- ASCII-only emission; NetlistBuilder class retained as thin wrapper.

**M3.6 — pipeline/fitting.py::fit_pwl_model(vendor)**:
consumes config/outputs/<vendor>/*.yaml (Curves 1/2/3/4/6/7), reconstructs
shared Vds breakpoints + per-Vgs Id rows, Rds(on), Qoss charge table
(re-runs the capacitance bench for full-resolution data), body-diode points,
Vth; assembles PWLModelSpec -> build_tier2_subcircuit -> writes
`reports/models/<device>_PIECESPACE_T2.lib` (name corrected to _T2).

**common/types.py — PWLModelSpec extended**: q_iss/q_rss tables,
body_diode_vsd/isd points, v_plateau, structural validators (row/count
matching, ascending breakpoints); CurveData/WaveformData/SimulationResult
accidentally dropped during the rewrite were restored immediately.

### Deep audit of M3.5/M3.6/M3.7 — two critical bugs found & fixed

**Bug A (critical): C_QOSS emission used absolute segment slopes.**
Same accumulation class as the earlier charge_model finding: soft-abs ramps
never turn off, so absolute slopes sum and Q diverges linearly. Proven:
emitted Q at table breakpoints off by up to 1.9e-7 C. Fix: delta slopes
(slope_k - slope_{k-1}). Post-fix worst error vs table: 1.3e-23 (exact).
Found by evaluating the emitted expression at table breakpoints - the
expression text was syntactically valid but semantically divergent.

**Bug B (critical, data): Vgs key truncation corrupted the transfer axis.**
curves.py wrote family keys with int(v): TI's 4.5 V curve stored as
'curve_id_vgs_4_v'. The fitting stage therefore labeled that curve's
breakpoint Vgs=4.0 instead of 4.5 - shifting the Tier-2 transfer axis. The
earlier acceptance numbers (worst 0.53%) were internally consistent but
built on the mislabeled key. Fixes: canonical `%g` formatter (`vgs_key()`)
in pipeline/curves.py; YAML validator accepts `%g` family keys via regex;
TI outputs regenerated (key now 'curve_id_vgs_4.5_v').

Also fixed: fitting.py's vgs-key parser handles float keys; incomplete test
removed before commit.

### Final acceptance (post-fix, trustworthy)

Simulated generated Tier-2 lib vs vendor golden at Vds=10 V, all TI curves:

| Vgs | Golden Id | Tier-2 Id | Error |
|---|---|---|---|
| 4.5 V | 76.5 A | 76.5 A | 0.07% |
| 5.0 V | 98.9 A | 98.4 A | 0.51% |
| 6.0 V | 144.7 A | 144.0 A | 0.52% |
| 8.0 V | 236.4 A | 235.1 A | 0.53% |
| 10.0 V | 325.5 A | 323.8 A | 0.51% |

WORST 0.53% - inside the <5% target, now on a provably correct Vgs axis.

Verified additionally: ohmic region sane (Id=2.99 A @ Vgs=10/Vds=50 mV vs
expected ~3 A); real lib parses and simulates cleanly through PySpice;
SPICE line lengths within conservative limits.

### Tests

New tests/test_netlist_builder.py: spec structural validation, subckt
structure (SUBCKT/ENDS/G_CH/tanh/C_QOSS/D_BODY), ascii-only emission,
end-to-end simulation matching spec endpoints within 2%. Suite: **97/97**.

### Remaining

M3.8 only: rebuild notebook 02 on real vendor data (fit residuals, segment
counts, golden-vs-Tier-2 overlay). scripts/ directory retained empty per
user rule (never delete directories without confirmation).

---


## 2026-08-24 17:28 IST — Milestone 3 started: PWL fitting (M3.1), RDP compression (M3.2), deep-audit bug fixes

### Context

With M1/M2 closed and the code-quality audit remediated, Milestone 3
(Piecewise Approximation & Charge Integration Engines) began. Core project
idea re-confirmed from `piecespice_plan.md` §1: PLECS-class simulation speed
with SPICE-class resolution — CPWL models, charge-conserving Q(V)
capacitances, C1-smooth knees, standard SPICE netlist output, <5% RMSE vs
vendor golden.

The end artifact of M3 is a **Tier-2 MOSFET subcircuit**, not curve fits;
fitting/compression utilities are the parameter-extraction stage that model
synthesis consumes (tasks M3.3–M3.8 remain).

### M3.1 — approximation/pwlf_fit.py rewritten (plan §2.1)

- `PWLFitResult` dataclass: breakpoints, per-segment slopes/intercepts,
  predicted_y, RMSE + L-inf max-abs error, K used, converged flag,
  stop_reason, notes; `evaluate(x)` for arbitrary-point evaluation.
- `fit_piecewise_linear(x, y, K)`: input sanitization (NaN/inf mask, sort,
  duplicate-x averaging); K=1 degenerates to plain least squares; pwlf
  failure falls back to K-1 instead of killing the pipeline.
- `fit_adaptive(...)`: plan §2.1 step 4 — increments K until RMSE meets an
  absolute or relative tolerance (default 1% of y-span), with max_segments
  cap and stagnation early-exit.
- Legacy tuple API retained (`fit_piecewise_linear_tuple`).

**Real-data validation**: TI CSD16412Q5A transfer characteristic
(conducting region) — adaptive fit found K=2, breakpoint at Vgs=2.63 V
(the threshold knee), second-segment slope 51.8 S (= gfs), relative RMSE
0.594% of span. Well inside the plan's <5% fidelity target.

Tests: `tests/test_pwlf_fit.py` (12 tests incl. exact-K recovery within
0.2 V of true knees, adaptive minimal-K selection, stagnation reporting,
NaN/unsorted sanitization, legacy API).

### M3.2 — approximation/rdp_compress.py hardened + extended (plan §2.2)

- Iterative explicit-stack RDP core replaces recursion (old code would hit
  Python's recursion limit on dense sweeps; verified at 120k samples).
- NaN/inf dropped, unsorted input sorted, output x strictly ascending,
  endpoints always retained.
- NEW `relative=True` epsilon mode: tolerance as fraction of bounding-box
  diagonal -> one setting transfers across curves of different units
  (verified: uniform unit rescale gives identical compression).
- NEW `rdp_epsilon_for_max_error()`: binary-searches the largest epsilon
  whose RECONSTRUCTED curve stays within an absolute fidelity budget.
  "Tell me the tolerance for <10 mA deviation" instead of guessing.

**Real-data validation**: TI body-diode curve 13 -> 7 points (54%) at
0.6 mA max deviation; budget search found epsilon=1.3e-2 for a 10 mA
budget with actual deviation 6x under it.

Tests: `tests/test_rdp_compress.py` (11 tests incl. large-array core,
relative-scale invariance, knee retention, budget-search accuracy).

### Deep audit of M3.1/M3.2 — two real bugs found and fixed

**Bug #1 (critical, fixed)**: `fit_adaptive` early-exit compared
`result.rmse > 0.999 * best.rmse`, but after an improvement result IS best,
so the check was trivially true and the loop exited right after its first
improvement past min_segments+1. Proven impact: sine probe returned K=3 /
RMSE 1.68 when K=10 / RMSE 0.137 exists — a 12x worse fit silently
returned. Fix tracks prev_rmse separately and exits only on STAGNATION.
Regression test added.

**Bug #2 (major for SPICE use, fixed)**: `_result_from_breaks` filled a
sample-starved segment (<2 samples) with a parallel line through the lone
sample -> C0-DISCONTINUOUS model (proven 45-unit jump at a breakpoint). In
an emitted SPICE PWL source this means Newton oscillation and wrong
operating points. Fix: trailing isolated samples become continuous
connectors (slope derived from predecessor endpoint value through the
point); interior starved segments merge into a neighbor by dropping the
shared breakpoint; final pass chains all intercepts so every segment starts
exactly where the previous ended (exact C0 to float precision - worst jump
over 50 randomized trials: 1.3e-14). Three regression tests added.

**Followup fixes from audit observations**:
- RDP epsilon guard: negative/NaN epsilon now raises ValueError (previously
  silently disabled pruning).
- `PWLFitResult.stop_reason` field ('tolerance_met' | 'stagnated' |
  'cap_reached') disambiguates what converged=False used to conflate -
  documented that converged=False does NOT mean invalid; RMSE always
  quantifies actual quality.
- Dead `repaired` flag removed; line-formatting artifact cleaned.

### Housekeeping

- scripts/ directory cleaned of all one-off probe/fix/verify scripts
  (kept empty per user instruction - directory itself retained).
- Plan TODO list updated to reflect true milestone state (M2 includes
  curves 3/6/7; M3 marked skeletons-only; M5 notebook 04 flagged as
  hardcoded placeholder numbers).
- Suite: 71/71 passing (was 46 before M3).

### Next in M3

M3.3 charge_model (Q(V)/E(V) integration, equivalent capacitances, PWL
charge-table synthesis) -> M3.4 smoother.py (softplus/tanh C1 kernels) ->
M3.5 netlist_builder Tier-2 subcircuit synthesis -> M3.6 pipeline fitting
stage -> M3.7 golden-vs-Tier2 validation -> M3.8 notebook 02 rebuild.

---


## 2026-08-24 13:21 IST — ngspice rollback to 34 + full code-quality audit remediation (41 findings)

### Rollback to ngspice-34 (user decision)

After the pow-wall RCA concluded that CPM2's G1 cannot run on ngspice >= 36
(internal differentiation creates fatal negative-exponent pows regardless of
abs() hardening), the user chose to return to the known-good state:

- All ngspice-39-era working-tree changes stashed via
  `git stash push -m "ngspice39-work-WIP-pow-hardening"` (recoverable, not
  deleted). Git used ONLY for this stash per user instruction.
- DLL restored: `dll-vs/ngspice.dll.bak34` -> `ngspice.dll`
  (`.bak39` retained on disk).
- Verification: pytest 46/46; notebook 01 (cpm2-1200) 18 cells / 0 errors /
  7 figures; all four vendors re-run clean end-to-end.

The development-log entry of 02:40 IST (phases A-E, NEXT STEPS options)
remains as the permanent record of the migration attempt.

---

### Code-quality audit (strict, all .py files)

Report: `docs/code_quality_report.md` — 7 Critical, 14 Major, 20 Minor
findings. All remediated in batches with pytest green after each batch and a
full 4-vendor regression at the end.

#### Critical fixes

1. **C1 — single pipeline**: new `pipeline/curves.py::run_all_curves(vendor)`
   replaces the IR-hardcoded script AND the copy-pasted notebook logic.
   Config-driven throughout: per-vendor `Cap_ref_Vds`, `Isd_ref`,
   `Id_th_target`, pin map, temperature.
2. **C2 — rds_on sign convention**: `np.abs()` removed, negation +
   `_check_sign` like every other bench.
3. **C4 companion** — canonical output keys extended in `config/yaml_io.py`
   (`rds_on_mohm_at_Id_load`, `curve_rds_mohm`, `curve_vsd`, `cap_ref_vds`,
   `Isd_ref`, `body_diode_vsd_at_Isd_ref_v`, `Id_th_target`) so per-vendor
   conventions are recorded, comparable across vendors.
4. **C5 — switching metrics real**: `metrics/switching.py` now extracts
   td(on)/tr/td(off)/tf via IEC 60747-8 level crossings on vgs/v_ds
   (interpolated); fabricated constants removed.
5. **C6 — CLI fake metrics removed**; report omits summary cards until M3
   golden data exists. New CLI command: `piecespice curves <vendor>`.
6. **C7 — pow_harden wired**: preprocessor Pass 3 applies abs()-hardening to
   behavioral-source bodies behind opt-in flag
   `normalize_pspice_netlist(..., harden_powers=True)`.

#### Major fixes

- **M1**: `common/roles.py` = single ROLE_NODES registry; static_sweep,
  capacitance, gate_charge all delegate (was four drifting copies).
- **M2**: Tj handling unified; gate_charge gained Tj drive (previously
  floated the net on Wolfspeed models).
- **M3**: dead `run_gate_charge` transient (VM-02-broken, zero call sites)
  deleted with pointer note.
- **M4**: dead probe block + misleading narration removed from engine
  fallback ladder.
- **M5/M6**: normalized copies moved to `reports/normalized/`
  (SpiceModels read-only); per-run cache (`_COPY_CACHE`) replaces O(N)
  mkstemp-per-bias-point churn. Verified live: SpiceModels/normalized empty
  after runs; cap values unchanged (2825.8 pF @ 7.5 V).
- **M8**: `_check_sign` upgraded warn-only -> hard AssertionError when >50%
  of upper-sweep samples have wrong polarity.
- **M9/M10**: stable curve ids (`output_characteristic`, `rds_on_vgs_<v>V`,
  ...); rds axis inversion declared via `metadata.axis_convention="current_x"`;
  body-diode condition moved to metadata.
- **M11**: notebook 01 rebuilt as thin pipeline wrapper with one markdown
  section + plot cell per curve (17 cells, 0 errors, 7 figures).
- **M12**: module-level logging (`_LOGGER`) in engine/preprocessor/
  capacitance/static_sweep/netlist_builder; print() removed from library
  paths.
- **M13/M14**: spice_literal suffix lookup simplified to exact dict access;
  `_KV_RE` split into `PARAM_KV_RE` (preprocessor) vs `HEADER_PARAM_RE`
  (netlist).

#### Minor fixes (m1-m20)

Scratch scripts deleted; unused `lookup()`/alias/dead helpers pruned;
wildcard PySpice imports made explicit; local imports hoisted; PEP 604
typing standardized; conftest owns path bootstrap; CLI docstring points to
`curves`; docstrings corrected (engine fallback behavior, Tier-2-only
builder); shared `trim_trailing_invalid()` sanitizer in common.types;
PWLModelSpec canonical charge names (`q_oss_vs_v`, `q_oss_charge_c`) with
legacy fold; m20 vendor script deleted (pipeline supersedes).

---

### ngspice diagnostics deduplication (user request)

Problem: Wolfspeed body-diode model emits
`Warning: bodydiode_...: junction potential VJ too large, limited to 2.0`
hundreds of times per sweep, flooding console and notebook output.

New module `common/diagnostics.py`:

1. **Registry**: every known ngspice message pattern classified by severity
   (error/warning/info), stable name, and human explanation - VJ clamps,
   param clamps, floating nodes, singular matrices, convergence ladder
   progress, timestep-too-small, aborted runs, pow domain errors, parse
   errors, PySpice version-table notice.
2. **SHA-256 dedup**: messages normalized first (instance names -> `<dev>`,
   floats -> `<f>`, ints -> `<n>`) then hashed; repeats collapse into one
   aggregate record with count + up to 3 raw samples.
3. **Suppression layers**:
   - `NgspiceLogDedupFilter` attached DIRECTLY to the emitting PySpice
     loggers (logging filters do not inherit from parent loggers - attaching
     to 'PySpice' alone did not work);
   - `ConsoleQuiet`: OS fd-level (dup2 to devnull) redirect around ngspice
     calls, because the C library printf bypasses Python entirely. The quiet
     window must cover circuit LOAD (include/parse), not just the analysis
     call - that was the second miss.
4. **Engine integration**: dc/transient/ac calls drain the shared-lib console
   buffer into the collector after every run and wrap load+analysis in
   ConsoleQuiet. Callers get one summary block per sweep:
   `[WARNING] diode_vj_clamp x255` + explanation + single sample.

Verified live: raw VJ warnings on console 255 -> 1 (first occurrence only);
fresh-kernel notebook shows exactly one line. NOTE: an already-running
kernel keeps stale modules - Restart Kernel & Run All Cells required to pick
up the change.

Tests: new `tests/test_diagnostics.py` (5 tests: dedup counts, severity
classification, numeric-variation hashing, singleton reset, normalization).
Suite now 51/51.

### Final state

- Simulator: ngspice-34 (bundled), `.bak39`/`.bak34` retained for any future
  revisit.
- Regression: pytest 51/51; all 4 vendors x 7 curves PASS through
  `run_all_curves`; notebook clean.
- Uncommitted per standing rule; git touched only for the stash noted above.

---

---

## 2026-08-24 02:40 IST — CPM2-1200 (Wolfspeed SiC) onboarding & ngspice-39 upgrade: full investigation log

### Scope

Onboard `SpiceModels/CPM2-1200-0025B-Bare-Die.lib` (Wolfspeed 1200 V SiC
MOSFET bare die, ports `d g s Tj`), create config, run notebook 01, validate
against datasheet. This expanded into the deepest compatibility
investigation of the project so far.

---

### PHASE A — Model analysis

**Model anatomy** (read line by line):
- Top subckt `.subckt CPM2-1200-0025B d g s Tj` — **4th port Tj = junction
  temperature node (volts = °C)**. First vendor model in the project with a
  thermal port.
- Internal subckts: `gmos_CPM2-1200-0025B d1 g1 s1 Tj` (main switch),
  `cgdmos_CPM2-1200-0025B d2 g` (gate-drain charge emulator).
- Parasitic network: Ls/R_Ls (1n/10Ω), Rg (1.1Ω), Lg/R_Lg (1n/10Ω),
  Ld/R_Ld (1n/10Ω) series elements; R_A 1T drain bleed.
- `G1` switch (gmos): large `if(V(d1,s1)<0, branch_neg, branch_pos)`
  expression using `log(1+exp(...))`, fractional powers `**p5` (p5=2.2)
  and `**p7` (p7=0.5), temperature-dependent bias sources e2/e3 producing
  NET1/NET2 from V(Tj).
- `E100`: gate clamp `limit(V(g1,s1), -10, +25)`.
- `G11` (cgdmos): charge-based capacitor
  `k1*(...)**-k2 * ddt(v(g,d1))` — **uses ddt()**.
- Body diode: three parallel diodes `bodydiode_...` with bv=1600.
- Gheat: power-dissipation feedback source into Tj node
  (`abs(V(d,s)*I(Vdrain_s)) + abs(V(g1,g2)*I(Vgate_s))`).

**Initial syntax findings**: spice-suffixed .PARAM literals (`1702.69p`,
`2825.325p`) that our Python evaluator could not parse; ddt() usage;
4th port; no obvious dialect blockers.

---

### PHASE B — Defects found & fixed (chronological)

#### B1. Spice-literal .PARAM drop (preprocessor bug)
`.param k1=1702.69p` etc. were being **silently discarded** because our
Python `_evaluate` couldn't parse the suffix notation.
**Fix:** new module `parser/spice_literal.py`
(`parse_spice_literal`, `translate_spice_literals`) translating SPICE
suffixes to scientific notation before eval; unresolvable expressions are
now left in place for ngspice's own parser instead of dropped.

#### B2. ddt() unsupported in ngspice-34 B-sources
Minimal probe: `b1 x 0 I={1e-9*ddt(v(d))}` → "no such function 'ddt'".
Confirmed missing on the bundled shared-lib build (ngspice-34) across all
compatibility modes (none/psa/ltpsa/spice3).
**Fix (workaround):** preprocessor commented out any B/G/E-source whose
expression contained `ddt(`. Mathematically exact for DC/AC extraction
(capacitors are open circuits); transients deferred to M4.
**Status after v39 upgrade:** ddt() works; this workaround was REMOVED
(see Phase E).

#### B3. VM-07 pin-order defect (EPC lineage, generalised)
Testbench X-instances hardwired IR/TI pin order. Fixed earlier via
`resolve_pin_connections()` reading `.subckt` headers against YAML
`pin_order_semantic`. For CPM2 this correctly resolved
`['drain', 'gate', 'source', 'tj']`.

#### B4. Capacitance bench net omission (critical)
`capacitance.py` built connections from a role list that **filtered out
'tj'**, so `xtj=None` was passed as an actual node AND the SOURCE port was
dropped entirely (only `[xg, xd]` connected). Result: floating thermal
node → V(Tj)=undefined → all AC probes degenerate (Coss stuck at
1.59e8 F = the max(vd,1e-15) floor).
**Fix:** full positional connection list built from ALL roles:
```
conn = {r: node_by_role.get(r, f"tj_{r}") for r in roles}
xd = conn["drain"]; xg = conn["gate"]
xtj = next((conn[r] for r in roles if r.lower() == "tj"), None)
has_tj = xtj is not None
xargs = [conn[r] for r in roles if r.lower() != "tj"] + ([xtj] if has_tj else [])
...
circuit.X("M1", subcircuit_name, *xargs)
if has_tj:
    circuit.V(f"tjsrc_{attempt}", xtj, circuit.gnd,
              float(self.engine.temperature) @ u_V)
```
**No net omitted.** Dead debug block (`_plot_vectors` remnant) removed from
`_ac_probe` at the same time.

#### B5. Engine stepped-OP infrastructure
`simulation/engine.py` rewritten: `run_dc_sweep_builder(builder, ...)` runs
per-point fresh-circuit OPs (the pattern proven by the capacitance bench);
legacy native-.dc path preserved via `use_native_dc=True`; `create_circuit`
restored (a test depended on it). All 46 tests passing at this point.

---

### PHASE C — ngspice-39 upgrade (user-supplied archives)

User placed in `download/`:
- `ngspice-39_64.7z` (full package: exe only, **no dll**)
- `ngspice-39_dll_64.7z` (dll package: `dll-vs/ngspice.dll` + mingw deps)

**Swap procedure (non-destructive):**
1. Backup bundled ngspice-34: `dll-vs/ngspice.dll` → `ngspice.dll.bak34`
   (also pre-existing `ngspice-34.dll` copy retained).
2. Extracted `Spice64_dll/dll-vs/ngspice.dll` (+ libomp140, mingw runtime
   dlls) from `ngspice-39_dll_64.7z` into `download/_extract/`.
3. Copied new dll over `dll-vs/ngspice.dll`; mingw deps into `dll-mingw/`.
4. Rollback path: single-file rename/copy restore.

**Result:** DLL loads through PySpice cffi. PySpice prints
"Unsupported Ngspice version 39" (its version table stops at 34) — cosmetic
warning only. RC smoke test (V=R+C divider, OP) → v(out)=5.0 V exactly.
`ddt()` now WORKS in B-sources (transient completes; derivative-of-constant
returns 0 correctly).

**Note:** during debugging a ngspice-47 dll was also briefly installed
(user supplied `ngspice-47_dll_64.7z`); it required extra deps
(samplerate.dll, sndfile.dll from `ngspice-47_64.7z`). It parsed even more
strictly (rejected ternary `?:` continuation lines outright with "syntax
error"). Rolled back to 39.

---

### PHASE D — The pow(0, -0.5) wall (RCA)

With ngspice-39 active and pins correct, every DC sweep of the gmos G1
source aborts: `Error: 0, -0.5 out of range for pow in b.xm1.xgmos.bg1`,
through ALL convergence strategies (dynamic gmin, true gmin, source
stepping, transient-op).

**Investigated and ruled out:**
- Not the model's parameter values (Python ground-truth evaluation of the
  same equation gives physically correct monotonic Id(Vgs)).
- Not pin wiring inside the benches (bench uses resolve_pin_connections;
  verified positional mapping (d,g,s,Tj)→('d','g','0','tj_M1')).
- Not stale/cached libraries (neutering G1 to a constant makes the error
  vanish instantly; restoring it brings it back).
- Not Gheat or G11 alone (gmos-only subckt reproduces the failure).
- Not abs()-placement: hardened expression confirmed present verbatim in
  the loaded netlist text.

**Root cause analysis:**
1. ngspice evaluates BOTH branches of `if()` (no short-circuit). During
   gmin/source stepping, node voltages traverse extremes where
   `log(1+exp(x))` underflows to exactly 0 (x < ~-745 ⇒ exp underflow),
   and/or `abs(v)**0.5` hits base 0.
2. Any subsequent pow with that zero base and ANY exponent is fatal on 39
   ("0, -0.5 out of range"; the -0.5 in the message corresponds to the
   **differentiation** term ngspice forms internally for the behavioral
   source: d/dx x^k involves k·x^(k−1) → negative fractional exponents at
   x=0 are fatal). On ngspice-34 the same condition silently returned
   garbage values instead of erroring — which is precisely VM-08.
3. Therefore: **abs()/max() floors applied to the branch bodies cannot
   fully protect the run**, because the fatal pow comes from ngspice's own
   internal differentiation of the fractional powers, not from the literal
   expression text we control.

**Mitigations attempted (all verified in netlist text before each run):**
- abs() around every fractional-power base (6 sites in G1+G11) → error
  persists.
- max(base, 1n) floors around log terms → error persists.
- Ternary `?:` rewrite of if() → rejected by v47 parser (syntax), and on 39
  still fails (same internal mechanism).
- Single-line collapse of continuations → parse OK, same runtime error.
- Constant-G1 control test → converges instantly, proving everything else
  in the circuit is sound.

---

### PHASE E — Current state of the tree

**Code changes present (all tested where testable):**
- `parser/pow_harden.py` (new): conservative fractional-power base
  hardening; unit-tested (11 cases incl. real G1/G11 bodies, idempotency).
- `parser/preprocessor.py`: imports pow_harden; `_harden_bsource_expression`
  applies it to B/G/E `value {...}` bodies only; ddt comment-out workaround
  removed (v39 supports ddt); spice-literal translation integrated.
- `parser/spice_literal.py` (new): SPICE suffix handling.
- `simulation/engine.py`: `use_native_dc` flag (default False → stepped-OP
  via `run_dc_sweep_builder`), legacy native .dc retained; transient and AC
  unchanged.
- `simulation/testbenches/static_sweep.py`: output-char, transfer, rds_on,
  threshold rewired to stepped-OP builders; body-diode partially rewired
  (per-point loop in place; needs final verification pass).
- `simulation/testbenches/capacitance.py`: full-role connections + Tj drive
  + per-point fresh normalized copies + NaN masking of non-physical points +
  trailing-region trimming.
- `simulation/testbenches/xinst.py` (new): shared `attach_x_instance` with
  Tj handling.
- `config/inputs/cpm2-1200.yaml` (new): Vgs 0–20 (user limit), Cap_ref_Vds
  7.5 V, Id_th_target 5 mA, temperature key driving Tj.
- `config/inputs/{irfs7730pbf,cpd16412q5a,epc2216}.yaml`: pin maps.
- `vendor_model_defects.md`: VM-07 (pin order), VM-08 (ngspice-34 evaluator)
  documented.
- Notebook 01: config-driven, pin-resolution aware, currently set to
  VENDOR="cpm2-1200".

**Validation snapshot (from working runs, pre-pow-wall):**
- Vth @ 5 mA: 2.166 V (datasheet typ ~2.4–3.0 V ✓)
- Rds(on) @ 14 V: 35.4 mΩ (spec 25–34 mΩ @ 20 V ✓ class)
- Ciss @ 7.5 V: 2826 pF (datasheet 2788 pF @ 1000 V ✓ excellent)
- Coss trend: 2526→444 pF falling (datasheet 220 pF @ 1000 V ✓ shape)
- Output characteristics at Vgs ≤ 12 V match datasheet Figure 1 closely
  (user visual confirmation: "high fidelity, match extremely").
- Known-bad on 34: Vgs ≥ 15 curves inverted (VM-08), Crss flat ~1 pF
  (ddt disabled), Qgd/Qg depressed.

**Open items (blocked on decision, see Next Steps):**
- pow-wall mitigation choice for ngspice ≥ 36.
- body-diode bench final verification pass.
- notebook re-run once unblocked.
- TI raw-lib load failure on 39 needs investigation when revisited
  (raw CSD16412Q5A.lib fails ngSpice_Circ; must go through normalizer —
  bench already does this, manual probes did not).

---

### NEXT STEPS (proposed, pending user direction)

**Option 1 — Preprocessor G1 rewrite (recommended):** replace the entire
`if()`-based G1 with a branch-free algebraically-equivalent form that avoids
fractional powers of potentially-zero bases entirely, e.g. clamp the gate
overdrive term: `L1 = log(1+exp(vov/p9))` computed as
`max(vov,0)/p9 + log(1+exp(-|vov|/p9))` (numerically stable softplus), and
replace `X**2.2` with `pwr(max(X,1e-12), 2.2)`. All transforms stay in the
normalizer; vendor original untouched.

**Option 2 — Accept partial support:** document CPM2 as
"ngspice-34-compatible (with VM-08 caveats above Vgs≈13 V); ngspice-39
unsupported", roll back the dll, continue M3 with existing extraction.

**Option 3 — Hybrid dual-engine:** keep both dlls; select per-vendor via
yaml key (`engine: ngspice34|ngspice39`); IR/TI/EPC on either, CPM2 pinned
to 34 until Option 1 lands.


---

## 2026-08-24 00:32 IST — Wolfspeed CPM2-1200 onboarding: Tj port wiring + capacitance net omission fixed

**Model**: `SpiceModels/CPM2-1200-0025B-Bare-Die.lib` (Wolfspeed 1200 V SiC
bare die; ports d g s Tj — 4th port is the junction-temperature node).

**Root causes found & fixed (in order of discovery):**

1. **Preprocessor dropped SPICE-suffixed .PARAM literals** (`1702.69p`):
   our Python evaluator couldn't parse suffix notation and silently discarded
   the line. Fix: new `parser/spice_literal.py` translates spice suffixes
   (p/n/u/Meg etc.) before eval; unresolvable lines are now left in place for
   ngspice's own parser.

2. **ngspice-34 lacks `ddt()` in B-source expressions** ("no such function
   ddt"). The model's Cgd charge source uses it. Preprocessor now comments out
   B/G sources containing `ddt(` — exactly-correct for DC/AC extraction
   (capacitors are open); transients (M4) will need ngspice >= 36.

3. **Tj port omission in capacitance bench** (the big one): `conn` was built
   from `core_roles` which filtered out 'tj', so `xtj=None` and the X-instance
   received a literal None node — shorting the thermal node and leaving the
   SOURCE port floating. With V(Tj)=0 instead of 25 C, NET1/NET2 collapse and
   every AC probe degenerated (vd→0 ⇒ Coss = 1.59e8 F constant).
   Fix: full positional connection list from ALL roles (source included),
   xtj driven by V(engine.temperature) from the yaml. **No net omitted.**

4. **Dead debug block removed** from `_ac_probe` (leftover `_plot_vectors`
   probe that did nothing).

5. Config: `config/inputs/cpm2-1200.yaml` created (Vgs 0–20 V per user limit,
   Cap_ref_Vds=7.5 V datasheet point, Id_th_target=5 mA SiC convention).

**Notebook result (18 cells, 0 errors, 7 figures):**
- Vth @ 5 mA: **2.166 V** (datasheet typ ~2.4 V ✓)
- Rds(on) @ 14 V: **35.4 mΩ** (spec class ~25 mΩ ✓ at lower drive)
- Ciss @ 7.5 V: **2826 pF** (stable across sweep)
- Coss @ 7.5 V: **1087 pF**, falls 2526→444 pF to 50 V (classic SiC shape ✓)
- Qg @ 18 V: **51.7 nC**; Qgs 17.6 nC; plateau 6.22 V
- Reverse Vsd @ 10 A: 1000 mV

**Known residual defects (documented, not code bugs):**
- VM-08: ngspice-34 mis-evaluates the G1 switch equation at high gate drive —
  Rds(on) rises above Vgs≈15 V (107 mΩ @ 18 V is inverted vs physics).
  Ground-truth Python evaluation confirms the equation itself is fine.
- Crss ≈ 1 pF flat: consequence of disabling the ddt-based Cgd source;
  only the fixed C_CGD remains until ngspice ≥ 36 or M3 charge-model replaces it.

**Tests**: 46 passed.

---

## 2026-08-23 21:01 IST — EPC2216 saturation-current RCA: model vs datasheet mismatch

### Observation

With pin order corrected (VM-07), EPC2216 extracts sanely in most studies but the
output characteristics conduct far more than the datasheet: transfer Id ≈ 26–27 A
@ Vgs=5–6 V / Vds=10 V vs a datasheet Figure-1 maximum around 3.4 A (user reading).

### In-depth verification performed

| Quantity | Model | Datasheet | Verdict |
|---|---|---|---|
| Rds(on) ohmic @ Vgs=5 V | 22.7 mΩ | 20–26 mΩ (@1.5 A) | ✅ match |
| Vth @ 11 mA (GaN convention) | 1.52 V | 0.7–2.5 V band | ✅ in band |
| Ciss @ Vds=12 V | 93 pF | ~98 pF typ | ✅ −5 % |
| Saturation Id @ Vgs=5/Vds=3 | ~26 A | ~3.4 A max per Fig. 1 | ❌ ~8× high |

Both vendor file variants (PSPICE .net and LTspice .lib) behave identically → not a
dialect issue. The gswitch equation itself:
`Id = A1·log(1+exp(vov/k3))·Vds/(1+(x0_0+x0_1·Vgs)·Vds)`
has no physical saturation mechanism — no velocity saturation or channel-pinchoff
term. The apparent ~26 A ceiling is purely the denominator's Vds-compression, which
grows with Vgs: the model even predicts Id *decreasing* from 27.6 A to 26.2 A as Vgs
rises 4→6 V at Vds=10 — non-physical.

### Root cause (condensed)

The fitted parameters describe only the ohmic region and the specific curve family
EPC extracted at; outside that window the behavioral expression extrapolates without
a current-limiting term. Two possible contributors, in order of likelihood:
1. **Parameter/datasheet scope mismatch**: `aWg=187 mm` gate width in the file may
   correspond to a larger die than the datasheet being compared against, or EPC's
   published .net expects per-die scale factors (`si/so/sr/aWg`) to be set by the
   customer for the specific variant.
2. **Isothermal extrapolation**: real GaN parts fold over hard with self-heating;
   SPICE never heats.

(A remote OCR ambiguity in Figure-1 axes cannot be fully excluded; worth confirming
whether ID really tops at ~3.4 A.)

### What matches, what doesn't — extraction guidance

- ✅ Trust: Rds(on), Vth, capacitances, gate charge (all within/near spec bands)
- ❌ Distrust: absolute saturation-region magnitude (VM-04 class, now also applies
  to EPC); reference operating points via transfer-curve interpolation instead of
  assuming Vgs→Id mapping.
- 🎯 M3 Tier-2 fitting should target **datasheet** curves; this is precisely why
  PieceSpice fits its own macromodels rather than trusting vendor files.

No code changes needed — documented analysis only.

---

---

## 2026-08-23 20:24 IST — EPC2216 (eGaN) onboarding: VM-07 pin-order defect found & fixed

### New vendor

`SpiceModels/EPC2216_V0_PSPICE.net` — EPC2216 eGaN FET, fully behavioral PSpice
model (gswitch B-source + capacitance emulators), ports `(gatein drainin sourcein)`.
New input config `config/inputs/epc2216.yaml`: Vds ≤ 15 V, Vgs ≤ 6 V (abs max),
GaN-appropriate conditions including Vth @ 11 mA (not the Si 250 uA convention),
Id_load 25 A, no Qgd calibration.

### Root cause of initial failure: VM-07 pin-order swap

The model "simulated but produced zero/garbage currents." Systematic elimination
(dialect syntax OK, Temp variable OK, all subckt nodes connected, minimal replica
of the switch source worked standalone) isolated the fault to testbench
instantiation: `.subckt EPC2216 gatein drainin sourcein` is GATE-FIRST while our
benches hardwired the IR/TI drain-first order. We had been driving the gate with
Vds and the drain with Vgs.

### Fixes shipped

1. **Config-driven pin map** (`config/inputs/*.yaml`): new `pins.pin_order_semantic`
   key maps roles to actual .subckt header tokens.
2. **Semantic validation** (`parser.netlist.resolve_pin_connections`): parses the
   real `.subckt` header from the model file, validates every declared token
   against it (raises on unknown tokens), returns connection roles in positional
   header order. Dynamic - works for any vendor without code changes.
3. **All testbenches pin-aware**: every method accepts `pin_roles=`; topologies
   that tie pins (threshold Vgd-tie, reverse-conduction gate-source tie) derive
   their tie pattern from the roles positionally.
4. **Normalized copies relocated**: `normalized_lib_copy()` now writes to
   `SpiceModels/normalized/<stem>_norm<suffix>` instead of temp files beside the
   originals (vendor files never touched); `PIECESPICE_KEEP_NORM=1` retains them.
5. **Capacitance bench robustness**: fresh normalized copy per bias point (ngspice
   shared lib intermittently fails after many loads of one include path) plus
   AC-probe retry (3 attempts).

### Verification

- EPC2216 notebook 01 end-to-end: 18 cells, 0 errors, 7 figures. Sane values:
  transfer Id ≈ 26 A @ 6 V/10 V, Rds(on)-region ~68 mΩ @ 25 A, Ciss ≈ 93 pF,
  Vth @ 11 mA = 1.52 V vs datasheet 1.4–2 V ✅.
- Regression: TI and IR notebook runs both 0 errors (numeric port maps resolve
  unchanged). pytest: 46 passed.
- Documented as **VM-07** in `vendor_model_defects.md`.

---

---

## 2026-08-23 16:57 IST — Temperature support analysis (TI vs IR) & deferred decisions

### Question
Can passing temperature as an input parameter make the massive Ids curves match the
datasheet (Id @ Vgs=10 V, Vds=10 V ≈ 325 A on TI vs IDC = 52 A)?

### Empirical finding: both models accept .TEMP, with very different fidelity

Measured Id in the Rds(on) region (Vgs=10 V, Vds=0.1 V) at 25 °C vs 125 °C:

- TI CSD16412Q5A: 11.79 → 8.93 A (**−24 %**, Rds +32 %) — matches the datasheet
  normalized-Rds-vs-temperature curve. Mechanism: full BSIM3v3.2 thermal set
  (`KT1/KT2/KT1L` Vth tempco, `UTE` mobility tempco) plus `RTEMP` resistors
  (`TC1=3.1e-3`, `TC2=7.2e-6`) and body-diode `TRS1`.
- IR IRFS7730PBF: 39.77 → 36.87 A (**−7 %**) — comes almost entirely from RS/RD
  resistor drift; the LEVEL=1 core's VTO/KP/LAMBDA are single-temperature constants,
  so channel response is essentially cosmetic. Real device Rds rises ~50–60 % over
  the same range.

Delivery mechanism is simulator-level `.TEMP` (vendor models take no temperature
`.PARAM`); our `SpiceEngine(temperature=...)` already provides it, and the audit fix
made the capacitance bench respect it.

### Answer to the core question: No — temperature cannot fix the saturation magnitude

To read 52 A at Vds = 10 V, effective Rds would need to be ~192 mΩ, a ~21× rise.
Silicon MOSFET Rds rises only ~0.7–1 %/°C; even at Tj,max (~175 °C) that is a
~1.5–2× increase → Id still ~160–200 A at Vds = 10 V. The datasheet IDC is a
thermal-equilibrium rating (RθJC × PD limited), not a point on the I-V plane:
at 325 A × 10 V = 3.3 kW into a 63 W package the real part is destroyed long before
any electrical steady state that SPICE could solve for. No `.TEMP` value reconciles
the two.

### Options identified (for later)

1. **Accept & document** (current state): trust ohmic region + datasheet-characterized
   biases; treat saturation magnitude as unphysical (VM-04/VM-06).
2. **Soft current clamp in Tier 2 synthesis**: PieceSpice's own PWL macromodel can
   include a power/thermal-limit term vendor models lack — making our models more
   datasheet-faithful than the sources (VM-04 implication #3).
3. **Electro-thermal co-simulation** (M5+ stretch): RC thermal network driving a
   scaling factor for genuine current fold-over.

**Decision: deferred** until Tier 2 synthesis design (M3). Near-term action item if
wanted: wire the config `temperature:` key through to testbench engine construction
(plumbing exists; config key currently informational only).

---

## 2026-08-23 00:55 IST — ngspice compatibility mode investigation & project .spiceinit

### Question
Does PySpice/ngspice work with different BSIM levels, and which `ngbehavior`
compatibility mode maximizes vendor-model support (PSpice, LTspice, Infineon)?

### BSIM level matrix (empirically probed on this machine's ngspice via PySpice)

| Model | Invocation | Works |
|---|---|---|
| MOS1/2/3 | LEVEL=1/2/3 | yes |
| **BSIM3v3.2 (PSpice-style, TI models)** | LEVEL=7 VERSION=3.2 | yes |
| BSIM4 v4.0–4.8, 4.10–4.11 | LEVEL=14 VERSION=x | yes |
| BSIM-bulk / SOI | LEVEL=58 / 54 / 55 / 10 | yes |
| EKV, HiSIM2/HV | — | not compiled in this binary |

Conclusion: PySpice handles all major BSIM families; PSpice vendor numbering
(LEVEL=7) resolves to the same BSIM3v3.2 core.

### ngbehavior mode comparison (decisive experiment)

Loading a minimal LEVEL=7 VERSION=3.2 circuit under each mode:

- `psa`, `ltpsa` → OK
- `all`, `ps`, `ltspice`, `spice3`, `hsimspice` → FAIL with "Device type MOS7
  not available in this binary" — in those modes LEVEL=7 maps to a legacy
  MOS7 device that isn't compiled into the Windows binary, instead of the
  PSpice mapping to BSIM3v3.2.

**`all` is therefore NOT maximum compatibility** for this project: it breaks
every PSpice-dialect vendor model (TI NexFET line). The winning mode is
`ltpsa`: full PSpice semantics (LEVEL=7 → BSIM3v3.2) plus LTspice syntax
tolerance — i.e., both major vendor dialects accepted simultaneously.

### Changes

- `.spiceinit` (project-local): `set ngbehavior=ltpsa`. Note there is also a
  global `~/.spiceinit` (was `ltpsa`) — the project file takes precedence when
  ngspice runs from the repo CWD, which is how PySpice's shared library and
  all testbenches/notebooks execute. The home-dir file should ideally be
  removed or aligned by the user; project file now guarantees correct behavior
  regardless.
- No code changes required; parser/preprocessor already normalizes dialect
  quirks before ngspice sees them.

### Vendor verification under ltpsa (both vendors pass)

- Infineon IRFS7730PBF: Rds(on) @10 V = 2.51 mΩ, transfer/Rds values identical
  to baseline (Id@10/10 = 1668 A — VM-04 kA-domain artifact, model-inherent).
- TI CSD16412Q5A: Rds(on) @10 V = 8.48 mΩ (spec 9–11), @4.5 V = 12.15 mΩ
  (spec 13–16); Id@Vgs=10/Vds=10 = 325 A (see analysis below).
- `uv run pytest`: 46 passed.

### Analysis note: why Id >> datasheet IDC at high Vds (both vendors)

The ~300 A (TI) / ~1.7 kA (IR) currents at Vgs=10 V, Vds=10 V are NOT circuit
errors: node-by-node audit of the TI subckt found no floating/misconnected
nodes, and ohmic-region extraction matches the datasheet spec bands. Causes,
in order of dominance:
1. Datasheet "continuous drain current" is a THERMAL rating (RθJC-limited),
   not an electrical limit - an unheated SPICE die legitimately conducts far
   more at 10 V.
2. PSpice-fitted BSIM parameters run under ngspice show weaker saturation /
   PSCBE channel-length-modulation behavior at high Vds (dialect drift).
3. No electro-thermal feedback in either macromodel (first-order RTEMP only).

---

## 2026-08-23 00:20 IST — M1/M2 audit fixes: robustness, contracts, error isolation

### Scope

Audit of all M1 (parser, preprocessor, common) and M2 (static_sweep, capacitance,
gate_charge, engine) files found 31 issues; all fixed in this pass. No M3+ files touched.

### P0 — correctness blockers
- **Vth silent clamp** (`static_sweep.run_threshold_voltage`): `np.interp` silently
  returned Vth≈0 when the swept current never reached the 250 µA target. Now raises a
  descriptive `ValueError` naming the sweep ceiling and target.
- **Temp-lib race**: the fixed-name `_temp_norm_<lib>` file (3 duplicated copies across
  testbenches) let concurrent runs clobber each other while ngspice held it open.
  Consolidated into `normalized_lib_copy()` / `discard_normalized_lib()` in
  `static_sweep.py` using PID-tagged `tempfile.mkstemp`; capacitance and gate_charge
  now import the shared implementation. Old private name kept as alias.
- **PWLModelSpec ↔ netlist_builder contract** (`common/types.py`,
  `models/netlist_builder.py`): codified `PWL_META_KEYS` (mirroring `yaml_io.KEYS`) as
  the producer/consumer key contract for Tier-2 metadata. The builder merges metadata
  over documented defaults, warns on unknown keys, notes missing ones — never crashes.

### P1 — logic & data-integrity fixes
- **Preprocessor** (`parser/preprocessor.py`): joins `+` continuation lines before any
  processing; `.PARAM` lines now accept multiple assignments (`.PARAM a=1 b={2*a}`)
  plus the bare `.PARAM name expr` form; param table is case-insensitive (last
  definition wins, original spelling preserved for output); unresolvable params emit an
  explicit warning instead of failing silently downstream; degenerate/self-referential
  resolutions return None rather than passing expression text through.
- **Netlist parser** (`parser/netlist.py`): nested `.SUBCKT` definitions no longer
  orphan outer frames (open-definition stack); named `.ENDS <name>` matches its own
  definition; unterminated inner frames are closed at their parent's `.ENDS` so
  raw_lines never leak between siblings; subckt names may contain any non-whitespace
  characters.
- **Gate charge** (`simulation/testbenches/gate_charge.py`): `_interp_curve` handles
  non-monotonic x-axes (flip descending; re-sort + warn if unordered);
  plateau-above-Vgs_max case folds the Miller segment at the final gate voltage so the
  returned Qg(Vgs) stays voltage-monotonic (flagged via new `plateau_clipped`
  breakpoint); stacked duplicate docstring on `run_gate_charge` merged (the discarded
  one carried the rail-clamp rationale, now in the surviving docstring with a VM-02
  warning); `plot_gate_charge` takes a `device_name` argument instead of hardcoding
  "IRFS7730PBF" in every title.
- **CurveData validation** (`common/types.py`): pydantic model validator rejects
  mismatched x_data/y_data lengths at construction instead of exploding later in
  `to_numpy()`. Family fields populated consistently: body-diode now carries
  `family_param="Vgs", family_value=0.0`, threshold carries `Id_target`;
  capacitance curves record measurement frequency in `metadata.note`.

### P2 — robustness & diagnostics
- **Engine** (`simulation/engine.py`): solver-option surface (reltol/abstol/vntol/gmin/
  itl1/itl4, applied via .OPTIONS with graceful fallback to defaults when a backend
  rejects them); DC/transient/AC calls wrapped with circuit title + options in effect;
  new `run_ac_sweep` used by the capacitance bench. Honest docstring about the PySpice
  hard-crash limitation. Temperature note added: vendor subcircuits are not always
  temperature-parameterized — the option is set but may have no effect (safe fallback).
- **Body diode**: bare `except Exception` replaced by targeted `RuntimeError` retry with
  a logged diagnostic; known 10 MΩ bleed-anchor parasitic current (Vsd/10MΩ) now
  subtracted analytically from exported Isd; family fields set.
- **Capacitance**: uses the shared engine (its temperature setting was previously
  ignored due to a hardcoded 25 °C simulator); constructor accepts `f_meas` with range
  sanity check — out-of-band frequencies warn and continue, invalid values fall back to
  the 1 MHz default (vendor-model limitation policy: warn, don't crash); top-of-sweep
  breakdown guard warns when capacitances explode >100x.
- **Sign-convention check** (`_check_sign`): warns when extracted current traces trend
  negative, catching vendor models whose internal source orientation flips polarity.
- **Mutable default arg** `vgs_list=[...]` → `None` sentinel.

### Not changed (deliberate)
- `SimulationResult` left as-is (unused scaffolding; wiring it into testbench returns
  would change public APIs that M4/M5 will consume).
- ngspice stdout capture beyond exception wrapping (full capture requires PySpice
  internals; deferred).

### Verification
`uv run pytest`: **46 passed**. Notebook 01 re-executed end-to-end on BOTH vendors:
18 cells / 0 errors / 7 figures each; all extracted values identical to pre-fix baseline
(TI: Rds(on) 8.47/12.24 mΩ, Vth 1.882 V, Qg 2.53 nC; IR: plateau 4.90 V, Qgd 79 nC,
Qg 226.65 nC, Vth 4.346 V). The new engine diagnostics are visible in the TI run
(body-diode coarse-step retry now logs its reason).

---

## 2026-08-22 23:30 IST — Notebook 01 converted to config-driven input

### Change

`notebooks/01_datasheet_curves_extraction.ipynb` no longer hardcodes a `DEVICES`
registry. Every simulation condition now comes from `config/inputs/<vendor>.yaml`
via `config.yaml_io.load_input()` — the same source of truth used by
`scripts/generate_outputs_irfs7730pbf.py`. Switching devices is a one-line change
(`VENDOR = "csd16412q5a"`); outputs/plots are unchanged.

### New file

- `config/inputs/csd16412q5a.yaml` — first TI vendor input config. Captures the exact
  SLPS207A datasheet conditions the notebook previously hardcoded (Vgs family 4.5–10 V,
  transfer @ Vds=15 V/301 pts, dual Rds(on) gate drives via new `Vgs_alt: 4.5` key,
  caps to 25 V/40 pts @ 1 MHz, Qg window Id=10 A/Vbus=12.5 V/Vgs→4.5 V,
  `Qgd_typ_nC:` empty = no VM-03 calibration).

### Supporting fix

- `config/yaml_io.py`: `CONFIG_ROOT` was relative (`Path("config")`) and only resolved
  when the process CWD happened to be the repo root — notebook kernels run from
  `notebooks/`, which broke config discovery. Now anchored on the package's parent
  directory so it works from any working directory.

### Verification

Notebook re-executed end-to-end with nbconvert: **18 cells, 0 errors, 7 figures**.
All extracted figures match the previous hardcoded-condition run exactly
(Rds(on) 8.47 / 12.24 mΩ, Vth 1.882 V, Ciss/Coss/Crss 412/348/33 pF,
body-diode 837.5 mV, Qg 2.53 nC). `uv run pytest`: **46 passed**.

Cross-checked with the IR vendor too: `VENDOR = "irfs7730pbf"` runs clean
(18 cells, 0 errors, 7 figures) and reproduces the known IR results
(plateau 4.90 V, Qgd 79 nC calibrated, Qg 226.65 nC, Vth 4.346 V VM-01 artifact).
This required adding the missing `Vgs_alt: 4.5` key to `config/inputs/irfs7730pbf.yaml`
(the notebook's dual Rds(on) study needs a second gate drive for both vendors).

---

## 2026-08-22 21:49 IST — YAML-based config & outputs (config-driven pipeline)

### Structure created

```
config/
  inputs/
    irfs7730pbf.yaml          # single source of truth for all sim conditions
  outputs/
    irfs7730pbf/
      curve1_output_characteristics.yaml   (pass)
      curve2_transfer_characteristic.yaml  (pass)
      curve3_rds_on.yaml                   (pass)
      curve4_capacitances.yaml             (pass)
      curve5_gate_charge.yaml              (pass)
      curve6_body_diode.yaml               (pass)
      curve7_threshold_voltage.yaml        (partial - VM-01)
      gate_charge_transient.yaml           (blocked - VM-02)
```

### New utility module

**`config/yaml_io.py`** — the single interface between testbenches and the YAML tree:
- `load_input(vendor_name)` — loads `config/inputs/<vendor>.yaml`; raises with a
  helpful message listing available configs when missing.
- `save_output(vendor_name, testbench_name, status, conditions_used, extracted, notes)`
  — writes `config/outputs/<vendor>/<testbench>.yaml` with the standardized schema
  (`device / testbench / status / conditions_used / extracted / notes`). Recursively
  converts numpy scalars/arrays to plain Python before dumping.
- `load_output(vendor, tb)` — read-back helper for verification.
- `KEYS` dict — canonical extracted-key names (vth_v, rds_on_10v_mohm, gfs_s,
  ciss_f/coss_f/crss_f, v_plateau_v, qgs_nc, qgd_nc, q_tail_nc, qg_total_nc,
  curve_* arrays) so downstream milestones can consume outputs uniformly.

### Input config populated with real values

`irfs7730pbf.yaml` mirrors the actual values used by the testbenches and notebook 01:
Vgs family [3, 4.5, 6, 8, 10], Vds sweep max 10 V/101 pts, transfer @ 10 V/15 V/151 pts,
Rds(on) ohmic sweep 2.5 V/51 pts with Id>1 A mask, JEDEC Vth @ 250 µA/6 V/3001 pts,
body diode 1 V/51 pts, capacitances 0.1–60 V/25 pts @ 1 MHz, Qg reconstruction
Id_load=100 A/Vbus=38 V/Vgs→10 V/Qgd cal 79 nC, plus library path, subcircuit name and
pin map.

### Output generation

`scripts/generate_outputs_irfs7730pbf.py` — runs every passing testbench using only
values from the input config and writes all eight output files via `yaml_io`
(zero inline YAML logic). Statuses reflect reality: six curves pass, Curve 7 is partial
(VM-01 effective-VTO artifact), the raw gate-charge transient is blocked (VM-02).
Example verified content: curve5 records plateau 5.067 V, Qgs 67.26 nC, Qgd 79.0 nC
(calibrated), total 226.17 nC with full sampled curves.

### Tests

`tests/test_yaml_io.py` (7 tests): input load + numeric typing (caught and fixed a YAML
1.1 quirk where `1.0e6` parsed as string — now written as `1.0e+06`), missing-vendor
error, output roundtrip, numpy serialization, invalid-status rejection, canonical-key
enforcement across all real output files, and coverage of all expected per-curve outputs.

Suite: **46 passed**. No existing testbench logic touched — the generator script is
the integration point until a CLI command supersedes it.

---

## 2026-08-22 21:07 IST — TI CSD16412Q5A refined to exact datasheet test conditions

### Trigger

User supplied the CSD16412Q5A datasheet tables (SLPS207A). The previous run used the
IRFS7730PBF-style conditions (Id=100 A, Vbus=38 V) — far beyond this 25 V / ±10 A-class
device's operating envelope. All studies re-run under the *exact* datasheet test
conditions (Id=10 A, Vbus=Vds=12.5 V, gfs @ Vds=15 V, Qg window to Vgs=4.5 V).

### Notebook changes

- Per-device datasheet conditions in the `DEVICES` registry: `i_load`, `v_bus`,
  `vgs_max`, `vds_cap`, `qgd_typ`.
- Transfer curve now swept at Vds=15 V for TI (the gfs test condition) vs 10 V for IR.
- Rds(on) extracted for both Vgs=10 V and Vgs=4.5 V (both are datasheet conditions).
- Gate charge reconstructed to Vgs_max=4.5 V with Vbus=12.5 V, Id=10 A.

### Validation vs datasheet (all under correct test conditions)

| Quantity | Extracted | Datasheet | Verdict |
|---|---|---|---|
| Rds(on) @ Vgs=10 V, Id=10 A | **8.47 mΩ** | 9–11 mΩ | slightly below min — model optimistic |
| Rds(on) @ Vgs=4.5 V, Id=10 A | **12.24 mΩ** | 13–16 mΩ | slightly below min |
| VGS(th) @ 250 µA | **1.882 V** | 1.7–2.3 V (typ 2.0) | ✅ in spec |
| Body diode Vsd @ Isd=10 A | **837.5 mV** | 850–1000 mV | just under min, ~1.5% off |
| gfs @ Vds=15 V, Id=10 A | 41.0 S | typ 33 S | +24% |
| Ciss @ 12.5 V | **412 pF** | 410–530 pF | ✅ at lower bound |
| Coss @ 12.5 V | **347.5 pF** | 350–450 pF | −0.7% below bound |
| Crss @ 12.5 V | **33.3 pF** | 32–42 pF | ✅ |
| Plateau voltage | 2.81 V | — | consistent with Vth≈1.9 + margin |
| Qgs | **1.16 nC** | typ 1.4 nC | −17% |
| Qgd | **0.58 nC** | typ 0.7 nC | −17% |
| Total Qg @ 4.5 V | **2.53 nC** | 2.9–3.8 nC | −13%, consistent bias |

Overall: every quantity within ~25% of datasheet and most inside or grazing the spec
band — dramatically better agreement than the IR MODPEX model (whose Vth missed the
datasheet band entirely). The TI model is a physically faithful macromodel; residual
biases (Rds(on)/Qg slightly low, gfs high) suggest the vendor fitted toward typical
silicon at 27 °C TNOM.

The earlier "unphysically low Qg = 5.3 nC" concern is resolved: that run integrated to
Vgs=10 V at a 100 A plateau — outside this device's world. Under real conditions Qg =
2.53 nC vs datasheet 2.9–3.8 nC is credible.

Artifacts: `reports/nb01_ti_refined/*.png` (7 figures). Notebook re-executed 9/9 cells,
zero errors. `uv run pytest -q`: 39 passed.

---

## 2026-08-22 20:43 IST — Second vendor onboarding: TI CSD16412Q5A (M1 + M2 characterization)

### Goal

Run the full M1/M2 device-characterization pipeline on a second vendor model,
`SpiceModels/CSD16412Q5A.lib` (TI NexFET, PSpice dialect), without creating new
modules — extend the existing machinery to support both vendors.

### ngspice compatibility (no global edits)

The TI file is PSpice dialect and crashes ngspice as-is:
`Undefined number [ptrc1]` — the `.MODEL RTEMP RES (TC1={ptrc1} ...)` references
subckt params that the raw file's single-pass parse cannot resolve, killing the shared
library. Running the file through our existing `parser.preprocessor.normalize_pspice_netlist`
resolves all params (`pWidth`, `perim`, `ptrc1/2`) to literals, after which the model
reaches an operating point cleanly. Note: `LEVEL=7 Version=3.2` is PSpice-speak for
BSIM3v3.2; ngspice maps LEVEL 7 to BSIM3 and simulates it correctly — no source-level
model translation needed. All normalization flows through `_normalized_lib_file()` in
the testbenches; no global ngspice config was touched.

### Changes

- `notebooks/01_datasheet_curves_extraction.ipynb`: setup cell now has a `DEVICES`
  registry (IRFS7730PBF / CSD16412Q5A) with per-device config (`lib`, `subckt`,
  `i_load`, `v_bus`, `qgd_typ`); all study cells parameterized off `DEVICE`. The TI
  entry sets `qgd_typ=None` (VM-03 calibration is IR-specific — TI uses real terminal
  capacitances, not a MODPEX mirror network). Capacitance sweep ceiling made
  vendor-aware: TI's body diode (DBD, BV=27) avalanches near 27 V — beyond that the AC
  probe collapses and Coss explodes to ~40 uF (physical breakdown, not a bug), so the
  TI sweep caps at 25 V vs 60 V for IR.
- `simulation/testbenches/static_sweep.py::run_body_diode`: ngspice DC sweeps are
  step-size-sensitive near the diode high-conduction knee (TI passes 1.0 V / 39 pts
  but fails 1.0 V / 41 pts). Added automatic coarse-step retry (25-point grid) +
  optional explicit `sweep_step` argument.
- `tests/test_vendor_ti.py` (new, 5 tests): subckt discovery, clean 3-pin parse,
  param resolution in normalized output, OP under ngspice, plausible Rds(on).
- Notebook summary cell fixed to report capacitances at a vendor-appropriate bias.

### Full pipeline run (notebook 01, DEVICE = "CSD16412Q5A")

9/9 cells executed, 7 figures at 300 dpi embedded, zero errors. Extracted figures:

| Quantity | Extracted | Datasheet (SLPS207A) | Note |
|---|---|---|---|
| Rds(on) @ Vgs=10 V, Id=50 A | 8.60 mOhm | ~4.3 mOhm typ (Id=25 A rated) | measured at 2x rated current |
| Vth @ Id=250 uA | 1.882 V | 1.35–2.35 V window | ✅ inside spec (real BSIM3 subthreshold) |
| Ciss @ Vds=25 V | 405 pF | 780 pF @ 10 V | sensible trend |
| Coss @ Vds=25 V | 262 pF | 220 pF @ 10 V | ✅ |
| Crss @ Vds=25 V | 26 pF | 80 pF @ 10 V | decreasing trend consistent |
| Body diode Vsd @ Isd=50 A | 997 mV | — | beyond rating; extrapolation |
| Plateau voltage | 5.02 V | — | from transfer curve @ 100 A |

Gate-charge quasi-static reconstruction: total Qg = 5.3 nC is **unphysically low** —
the TI model's terminal CGS0/CGD0 (1–6 pF) plus BSIM3 oxide cap give a tiny gate
charge because this "TRANSIENT" model was calibrated for switching waveforms rather
than datasheet Qg curves (datasheet Qg ~ 18 nC typ). Recorded as a known limitation:
the reconstruction is only as good as the model's capacitance data; for the TI part,
Curve 5 targets should come from the datasheet, not this macromodel.

Artifacts: `reports/nb01_ti_final/*.png` (7 figures).

### Verification

`uv run pytest -q`: **39 passed** (34 previous + 5 new TI tests). No regressions.

---

## 2026-08-22 16:10 IST — M1/M2 implementation audit, test suite expansion (8 → 34), VM-03 fix shipped

### Audit scope

Milestone 1 (env, parser, dialect normalization, plotting) and Milestone 2 (datasheet
curve testbenches 1–6), including pytest verification and a fixability pass over
`vendor_model_defects.md`.

### Implementation errors found and fixed

1. **`parser/preprocessor.py` — forward-referenced params never resolved.** A param
   defined via an earlier expression referencing a LATER param (`.PARAM a {b*2}` before
   `.PARAM b 3`) stayed symbolic forever; single-pass substitution silently left `{a}`
   unresolved in device lines. Fixed with an iterate-to-fixed-point parameter table.
   Also: the docstring claimed `$` comments were stripped but they weren't — now they
   are; `eval` hardened with a safe function whitelist (min/max/sqrt/exp/log/...,
   numpy-backed when available) instead of bare eval on arbitrary expressions;
   unresolvable params now stay cleanly symbolic instead of crashing or fabricating
   values.
2. **`parser/netlist.py` — PSpice header params corrupted pin list.**
   `.SUBCKT DEMO 1 2 3 PARAMS: scale=1.5 temp=25` parsed "PARAMS:", "scale=1.5",
   "temp=25" as three extra pin nodes, and the `parameters` dict stayed empty despite
   being a model field. Added `_split_header_params` to split the header tail into real
   pins and key=value params.
3. **`simulation/testbenches/static_sweep.py::run_rds_on` — hardcoded conduction mask.**
   `id_arr > 1.0` silently discarded all data for any device conducting under 1 A at the
   swept bias. Now an explicit `min_current: float = 1.0` parameter (default preserves
   old behavior for the IRFS7730PBF study).

### Vendor-model defect fixability assessment (`vendor_model_defects.md`)

- **VM-01, VM-02, VM-04, VM-05:** NOT fixable in place — all live inside the vendor
  `.spi` file, which must not be mutated (results would no longer be "golden").
  Mitigations already documented/shipped.
- **VM-03 (charge-inaccurate Cgd): FIXED on our side.** The Qgd under-prediction came
  from our reconstruction relying on the vendor model's integrated small-signal Crss.
  `reconstruct_gate_charge` gained an optional `qgd_cal_nC` datasheet-calibration
  argument that rescales the Miller segment to the datasheet target (reported as
  `qgd_cal_scale` in breakpoints). Verified against IRFS7730PBF @ Id=100 A, Vbus=38 V:
  Qgd 36.0 nC → **79.0 nC** (datasheet typ), total Qg 183.8 → **226.8 nC** vs datasheet
  271 nC; Qgs/tail segments provably untouched by the rescale.

### Test suite expansion (8 → 34 tests, all passing)

New files:
- `tests/test_preprocessor.py` (7): forward-ref resolution, `$`/`;` comment stripping,
  safe function evaluation, symbolic passthrough of unresolvable params, param hoisting.
- `tests/test_netlist.py` (5): vendor-file subckt discovery incl. thermal subckt, pin
  count check, PSpice PARAMS-header handling, multi-subckt/comment handling.
- `tests/test_gate_charge.py` (7): three-segment structure, DataFrame format +
  monotonicity, ascending/descending capacitance-axis equivalence, out-of-range Iload
  error, plateau clamping to v_gs_max, Qgd datasheet calibration exactness.
- `tests/test_parser.py` rewritten (+8 net): original two tests plus forward-ref, `$`
  comments, safe functions, PSpice header params, plain-subckt regression, and five
  reconstruct_gate_charge regression tests.

Three initial failures were test-expectation bugs (interpolation landing exactly on a
knot point; tolerance too tight; wrong assumption about symbolic passthrough) — fixed
in the tests, not papered over in the code.

### Verification

`uv run pytest -q`: **34 passed**. No regressions.

---

## 2026-08-22 16:10 IST — M1/M2 audit: 3 implementation bugs fixed, VM-03 fixable defect fixed, test suite 8 → 34

### Audit scope

Full review of Milestone 1 (preprocessor, netlist parser, plotting) and Milestone 2
(static_sweep, capacitance, gate_charge testbenches), pytest verification, missing-test
gap fill, and a fixability pass over `vendor_model_defects.md`.

### Implementation errors found & fixed

**M1 — `parser/preprocessor.py`** (rewritten)
1. *Forward-referenced params never resolved*: single-pass substitution left `.PARAM a {b*2}`
   unresolved when `b` was defined later. Fixed with iterated evaluation to a fixed point.
2. *`$` comments not stripped* despite the docstring promising it. Now stripped alongside `;`.
3. *Unsafe bare `eval`*: replaced with a safe function whitelist
   (min/max/sqrt/exp/log/log10/pow/sin/cos/tan via numpy, math fallback) and empty builtins.
4. Unresolvable params now stay symbolic instead of crashing.

**M1 — `parser/netlist.py`**
5. *PSpice header params misread as pins*: `.SUBCKT DEMO 1 2 3 PARAMS: scale=1.5` produced
   nodes `['1','2','3','PARAMS:','scale=1.5','temp=25']` and an empty parameters dict.
   Added `_split_header_params`: pin list is now clean, params land in `SubcircuitInfo.parameters`.

**M2 — `simulation/testbenches/static_sweep.py`**
6. *Hardcoded conduction mask* in `run_rds_on` (`id_arr > 1.0`) silently dropped all data
   for low-current devices. Exposed as parameter `min_current=1.0`.

### Vendor-model defect fixability assessment (`vendor_model_defects.md`)

- **VM-01, VM-02, VM-04, VM-05**: NOT fixable without mutating the vendor `.spi`
  (explicitly forbidden — results would no longer be golden). Mitigations already shipped;
  statuses unchanged in the catalog.
- **VM-03 (charge-inaccurate Cgd)**: FIXABLE on our side. Added optional
  `qgd_cal_nC` to `reconstruct_gate_charge` — rescales the Miller segment to a datasheet
  Qgd target so total charge is physically meaningful; the applied factor is reported as
  `qgd_cal_scale` in breakpoints. Verified on IRFS7730PBF @ Id=100 A, Vbus=38 V:
  raw Qg = 183.8 nC (−32% vs datasheet 271); calibrated Qg = **226.8 nC**
  (Qgs 67.2 / Qgd 79.0 exact / tail 80.6), remaining −16% gap attributed to Qgs/tail
  overdrive inflation (VM-01) which calibration cannot address.

### Tests: 8 → 34 (all passing)

New/expanded files:
- `tests/test_preprocessor.py` — 7 tests: forward refs, `$`/`;` comments, safe functions,
  unresolvable-symbolic passthrough, param hoisting before .SUBCKT, original param case.
- `tests/test_netlist.py` (new file, then merged into expanded parser tests) and
  `tests/test_parser.py` — vendor-file parsing (both subckts present, 3-pin check),
  PSpice PARAMS-header handling, plain-subckt regression.
- `tests/test_gate_charge.py` (new) — 7 tests: three-segment structure & charge sum rule,
  DataFrame format + monotonicity, ascending/descending Crss-axis equivalence,
  Iload-out-of-range ValueError, plateau clamped to v_gs_max, Qgd calibration exactness.

Three initial failures were bad test expectations (interpolation landing exactly on a
knot point; symbolic-passthrough semantics), corrected in the tests — not code bugs.

Final: `uv run pytest -q` → **34 passed**.

---

## 2026-08-22 15:21 IST — Root cause analysis: Id = 1.5–1.7 kA vs datasheet ID = 246 A (new defect VM-04)

### Trigger

User flagged Curve 1 output characteristics showing Id ≈ 1.5–1.7 kA at Vgs = 10 V against
the IRFS7730PBF Absolute Maximum Ratings (ID = 246 A @ TC=25 °C silicon-limited, 195 A
package-limited, IDM 984 A pulsed).

### Analysis performed

- Re-ran Curve 1 and Curve 2 sweeps; measured Id @ Vgs=10 V, Vds=10 V = **1668.1 A**.
- Hand-solved the LEVEL=1 core equations with the model's own parameters
  (KP=465.97, VTO=4.34579, LAMBDA=0.0605061, RS=2.03 mΩ, RD=0.1 mΩ):
  analytic Id_sat @ 10 V/10 V = **1727 A** — matches the measurement to ~4%.
- Verified the fitter had no free parameter left: predicted Rds(on) =
  1/(β·Vov) + RS + RD = 2.509 mΩ vs extracted 2.52 mΩ ✅; gfs @ 100 A ≈ 215 S vs
  datasheet 249 S (−14%, expected for LEVEL=1).
- Thermal cross-check of the datasheet rating: ID = 246 A is "Silicon Limited" —
  P = 246² × 2.52 mΩ ≈ 152 W < PD max 375 W. At 1668 A, P ≈ 7 kW = 18.7× rated PD.

### Verdict

Not an extraction bug. The vendor model *does* conduct kA-level current by design:
LEVEL=1 has no maximum-current parameter, no mobility degradation, no velocity
saturation, no thermal feedback — so Id grows as Vov² without limit once KP and VTO are
pinned by Rds(on) and the transfer shape. The datasheet's 246 A is a thermal/package
limit that a purely electrical macromodel cannot encode. The model's realistic operating
point is Id = 246 A at Vgs ≈ 5.66 V (interpolated from Curve 2).

Recorded as defect **VM-04** in the new `vendor_model_defects.md` catalog, alongside
VM-01 (no subthreshold / fake-high Vth), VM-02 (sense-and-mirror transient pathology),
VM-03 (charge-inaccurate Cgd), and VM-05 (corrupted t=0 operating point).

### Operating rules adopted for this device model

1. Never drive beyond ~250–350 A continuous in testbenches.
2. Derive operating points by transfer-curve interpolation at the target load current,
   never from "Vgs = 10 V".
3. Tier 2 synthesis should consider a soft current clamp near datasheet ID.

---

## 2026-08-22 15:40 IST — Plotting overhaul + notebook 01 updated with gate-charge study

### Motivation

User review: existing plots were "not professional enough — bad layout and placement."
All notebook figures flow through `common/plotting.py`, so the fix was made centrally.
300 dpi resolution retained per project constraint (plan §4.3); everything else overhauled.

### Changes

**`common/plotting.py` (rewritten)**
- Replaced the fragile two-pass `tight_layout` legend-width hack with
  **constrained layout** (`layout="constrained"`) — matplotlib now reserves exactly the
  space the EAST legend needs; no clipped legends, no dead whitespace, deterministic.
- Added engineering-unit tick formatting: `EngFormatter` derived from each curve's
  unit field (A -> kA on high-current plots, F -> pF/nF, V, mOhm...), so tick labels
  read "250 A", "13 nF" instead of raw floats/scientific notation.
- On log axes: minor formatter set to NullFormatter + LogLocator(subs='auto') to kill
  duplicate minor labels.
- Typography scale retuned (12/13/14 pt), title padding, thinner spines (0.9),
  major/minor tick sizes, grid drawn below data (`axes.axisbelow`), minor grid off,
  legend styling (square frame, edgecolor 0.75).
- Default figure size 10.5 x 5.8 (16:9-ish, better for notebooks/reports than the old
  12 x 6.75 that shrank the axes when a long legend was attached).
- `plot_waveforms` updated to the same machinery.

**`simulation/testbenches/gate_charge.py`**
- `plot_gate_charge` migrated to constrained layout + EngFormatter("nC") x-axis;
  removed `fig.tight_layout()`.

**`notebooks/01_datasheet_curves_extraction.ipynb` (rebuilt and re-executed)**
- Added **Study 5 - Gate Charge (quasi-static reconstruction)** using
  `reconstruct_gate_charge` at datasheet conditions (Id = 100 A, Vbus = 38 V), with
  printed breakpoint comparison vs datasheet typ values (Qgs 55 / Qgd 79 / Qg 271 nC)
  and the annotated three-segment plot. The intro table's "gate charge excluded" note
  replaced by an explanation of the quasi-static approach.
- All cells re-executed via nbconvert (inline backend): 7 figures embedded at 300 dpi,
  execution counts 1-9, no errors.
- Fixed LIB path to be repo-root-relative-resolved instead of hardcoded absolute.

### Verification

- Visual QA of exported figures (output-characteristics family and gate-charge plot):
  legend fully outside right with zero overlap, no clipped text, engineering-unit ticks
  (kA / nC), annotations readable, Miller plateau line visible, no layout flaws.
- Gate-charge numbers in-notebook: Vgp 5.07 V, Qgs 67.3 nC, Qgd 36.0 nC, Qg 183.2 nC.
- Full pytest suite: **8 passed** — no regressions.

---

## 2026-08-22 14:47 IST — Curve 5 (Gate Charge) delivered via quasi-static reconstruction (Option A); datasheet cross-validation analysis

### Context

Curve 5 (Vgs vs Qg under constant-current gate drive) was blocked on the IRFS7730PBF
vendor subcircuit (`SpiceModels/irfs7730pbf.spi` → `irfb_s_sl7730pbf`) by a MODPEX
sense-and-mirror network pathology: under dynamic drain transitions the EV16/CAP/FI1/FI2
network enters high conduction and pumps amp-level junk current into the internal gate
node, driving Vgs to -37 V. Confirmed structural (Gear/reltol/timestep changes have no
effect). Full root-cause analysis in `gate_charge_error_analysis.md`.

Decision: do NOT attempt to fix the transient testbench; implement Option A —
quasi-static Qg reconstruction from already-validated Curve 2 (transfer) and
Curve 4 (Ciss/Coss/Crss vs Vds) data.

### Changes made

**File: `simulation/testbenches/gate_charge.py`**

1. Module docstring rewritten to document the vendor-model blockage and the Option A
   approach, cross-referencing `gate_charge_error_analysis.md`.
2. Added imports: `pandas`, `common.types.CurveData`.
3. Added `_interp_curve(curve, x)` — interpolates a `CurveData` onto arbitrary x,
   handling both ascending and descending x-axes (np.interp requires ascending;
   CapacitanceTestbench returns log-spaced Vds which is ascending, but defensive
   handling added after an initial bug where descending axes produced negative Qgd).
4. Added `_crss_integral(crss_curve, v_hi, v_lo, n_points=200)` — integrates Crss(Vds)
   over the drain swing from v_hi down to v_lo on a descending axis so dQ is positive
   as the drain collapses. Returns (total charge C, descending Vds array, cumulative
   charge array).
5. Added **`reconstruct_gate_charge(transfer_curve, ciss_curve, crss_curve, v_bus=40.0,
   i_load=50.0, v_gs_max=10.0, v_ds_on=0.1, n_points=200)`** — the core three-segment
   quasi-static model:
   - Segment 1 (Qgs): plateau voltage Vgp found by interpolating the transfer curve at
     Id = Iload; Qgs = Ciss(Vds = Vbus) x Vgp.
   - Segment 2 (Qgd / Miller plateau): Qgd = integral of Crss(Vds) dVds over the drain
     swing Vbus -> Vds,on, expressed on a cumulative axis while Vgs stays flat at Vgp.
   - Segment 3 (tail): Ciss(Vds ~ Vds,on) integrated from Vgp -> Vgs_max.
   - Returns `(pd.DataFrame([Qg_nC, Vgs_V]), breakpoints_dict)` where breakpoints dict
     has keys `v_plateau_V`, `q_gs_nC`, `q_gd_nC`, `q_tail_nC`, `q_total_nC`.
6. Added **`plot_gate_charge(df, breakpoints, save_path=None)`** — plots Vgs vs Qg with:
   - dashed horizontal line at the Miller plateau voltage,
   - vertical markers + arrow annotations at Qgs, Qgs+Qgd, and total Qg @ 10 V,
   - EAST-outside legend (`loc='center left', bbox_to_anchor=(1.02, 0.5)`) per the
     project's mandatory plotting constraint (plan §4.3).
7. The blocked transient routine `run_gate_charge(...)` is retained unchanged in the
   module for reference only — no code path calls it.

No changes to any other module. No mutation of the vendor `.spi` file. No new transient
simulation attempted.

### Verification results (real device: irfs7730pbf.spi)

Pipeline run: StaticSweepTestbench.run_transfer_characteristics +
CapacitanceTestbench.run -> reconstruct_gate_charge.

Initial run (v_bus=40 V, i_load=50 A):
- Plateau voltage Vgp = 4.70 V
- Qgs = 62.30 nC | Qgd = 36.99 nC | tail = 85.89 nC | **Qg total = 185.18 nC**
- Within expected 150-200 nC range. Monotonicity check passed.
- Plot visually confirmed (vision check): pre-plateau rise, flat Miller plateau at
  4.7 V, tail rise to 10 V, annotations present, legend EAST outside.

Full pytest suite: **8 passed** — no regressions.

Artifacts produced:
- `reports/gate_charge_curve5.csv` — [Qg_nC, Vgs_V] table
- `reports/gate_charge_reconstructed.png` — annotated three-segment plot

Bug fixed during verification: first run produced Qgd = -36.99 nC because Crss curve
x-axis ordering interacted badly with np.interp over a descending sweep axis; resolved
by `_crss_integral` keeping the physically meaningful descending-Vds integration order.

---

## 2026-08-22 14:47 IST — Datasheet cross-validation: why extracted values deviate from the IRFS7730PBF datasheet

Datasheet dynamic/static tables (Tj = 25 °C) were compared against model-extracted
values, re-run under exact datasheet conditions (Id = 100 A, Vds = 38 V).

### Agreement (extraction machinery validated)

| Quantity | Model-extracted | Datasheet typ | Error |
|---|---|---|---|
| Ciss @ Vds=25 V | 13 381 pF | 13 660 pF | −2% |
| Coss @ Vds=25 V | 1 108 pF | 1 120 pF | −1% |
| Crss @ Vds=25 V | 685 pF | 690 pF | −0.7% |
| Rds(on) @ Vgs=10 V | 2.52 mΩ | 2.2 (max 2.6) mΩ | within spec |

Conclusion: Curve 2/4 testbenches measure the vendor model faithfully.

### Root cause of deviations (traced to specific lines of irfs7730pbf.spi)

Line 21-23 of the vendor file:

```spice
M1 9 7 8 8 MM L=100u W=100u
.MODEL MM NMOS LEVEL=1 IS=1e-32
+VTO=4.34579 LAMBDA=0.0605061 KP=465.97
```

1. **Vth deviation (extracted 4.30 V vs datasheet window 2.1–3.7 V)** — structural:
   - `IS=1e-32` eliminates subthreshold conduction entirely. A LEVEL=1 SPICE MOSFET
     conducts zero current until VGS > VTO, whereas a real device reaches the JEDEC
     Vth definition current (250 uA) gradually below threshold.
   - Consequently the MODPEX fitter could not fit VTO to the datasheet Vth; it fitted
     an *effective strong-inversion onset* (VTO = 4.35 V) tuned jointly with KP and
     LAMBDA to match the high-current transfer characteristic. The datasheet's Vth
     quantity simply does not exist inside this macromodel.
   - Same trade-off visible in gfs: model ~350 S near high Id vs datasheet 249 S — a
     single square-law LEVEL=1 core cannot match Vth, gfs, and Rds(on) simultaneously.

2. **Qgs too high (67 nC @ 100 A conditions vs datasheet 55 nC)** — consequence of the
   inflated effective threshold: the reconstructed plateau sits at ~5.07 V instead of
   the real device's lower plateau, so segment 1 absorbs excess charge
   (Qgs ~ Ciss x Vplateau).

3. **Qgd too low (~36 nC vs datasheet 79 nC) and Qg total low (184 vs 271 nC, −32%)** —
   the MODPEX sense-and-mirror network reproduces small-signal Crss accurately at fixed
   bias (hence the perfect Crss @ 25 V match), but its large-signal charge delivery
   across the full 38 V -> 0 drain swing under-integrates versus the real device's Cgd
   bulge during collapse. Extracted Crss does rise from 685 pF @ 25 V toward ~3.4 nF at
   low Vds, but not steeply enough in integrated charge.

### Datasheet-condition reconstruction summary (Id=100 A, Vbus=38 V)

| Segment | Reconstructed | Datasheet typ |
|---|---|---|
| Plateau voltage | 5.07 V | — |
| Qgs | 67.25 nC | 55 nC |
| Qgd | 35.95 nC | 79 nC |
| Tail | 80.57 nC | — |
| Qg total @ 10 V | 183.78 nC | 271 nC |

### Implication for PieceSpice

Every observed deviation is a documented defect of the golden vendor model itself, not
of the extraction pipeline. This supports the framework premise: vendor "golden" models
carry systematic errors (behavioral DC core without subthreshold; AC-calibrated but not
charge-accurate capacitance emulation). Validation targets must be chosen deliberately
(datasheet values vs vendor-model values), and this should inform Tier 2 model targets.

TODO carried forward: consider appending this cross-validation as §8 of
`gate_charge_error_analysis.md`.

---

## 2026-08-22 (morning) — Project indexing report (pre-work baseline)

Prior to implementation, a full repository index was performed:

- Walked full directory tree; read every source module.
- Cross-referenced `piecespice_plan.md` against disk state.
- Verified `uv run pytest -q`: 8 passed.
- Key findings recorded:
  - Milestones 1 and static portion of 2 solidly done (Curves 1, 2, 3, 4, 6 working).
  - Milestones 3–5 skeletal: netlist_builder ignores fitted breakpoints; no
    double_pulse.py or buck_converter.py; metrics/switching.py returns hardcoded
    timing placeholders; notebooks 02–05 are unexecuted stubs.
  - Missing vs plan: parser/extractor.py, approximation/smoother.py,
    simulation/testbenches/double_pulse.py, simulation/testbenches/buck_converter.py.
  - Known blocker documented in gate_charge_error_analysis.md (Curve 5 transient).
  - Plan checkboxes stale in both directions; README.md empty; unused deps in
    pyproject.toml (polars, shapely, rdp, seaborn, pandas, hypothesis,
    pytest-benchmark).
