# Migration TODO — PySpice/NgSpice → spicelib + LTSpice

**Source plan:** [`docs/spicelib_evaluation.md`](spicelib_evaluation.md) (§2.2 file-by-file changes, §2.3 milestones M1–M6).
**Status:** planning only. No code written.
**Decision locked (user):** complete migration to spicelib + LTspice; no PySpice/NgSpice path is retained in the repo going forward. The entire PySpice-era codebase is safely archived outside this repository, so deletion here is safe — the archive, not feature flags, is the fallback.

Each task lists **what/where**, **depends on**, and **gotchas / decisions to make first**. Tasks are ordered so dependencies flow downward.

---

## Phase 0 — Prerequisites & Decisions (before any code)

### T0.1 — ✅ DONE — Verify LTspice availability and version
**Result:** LTspice 26.0.2 at `C:\Users\GNKRISHNAN\AppData\Local\Programs\ADI\LTspice\LTspice.exe`; `-version` flag works; headless `-b -ascii` batch run of an RC `.tran` netlist produced valid .raw (3,020 pts ASCII) + .log + .op.raw; 16 solver threads; default integrator modified-trap. Gotchas found: MSYS-style paths are rejected ("No schematic nor netlist specified") — native Windows paths required; an .op.raw sidecar file appears next to the transient raw and parsers must ignore it.
- **Where:** host machine only; no repo changes.
- **Depends on:** nothing.
- **Gotchas:**
  - LTspice is a system dependency, not pip-installable. Confirm `LTspice.exe` path (typically `C:\Program Files\ADI\LTspice\LTspice.exe` or `%LOCALAPPDATA%\Programs\ADI\LTspice\`). Decide whether the path goes in a config key, env var (`LTSPICE_EXE`), or auto-discovery helper.
  - Record the exact LTspice version — behavioral syntax support differs between LTspice XVII and LTspice 24.
  - Linux/Wine is out of scope unless explicitly decided otherwise.
- **Decision needed:** pin a minimum supported LTspice version in the README before any testbench work starts.

### T0.2 — ✅ DONE (partial) — Add `spicelib`, remove `pyspice`
**Status:** `spicelib>=1.6.3` added by user and importable; `pyspice` **still listed** in pyproject.toml — final removal deferred to T6.3 per the archive-safe cleanup, or pulled forward whenever the code no longer imports it.
- **Where:** `pyproject.toml` + `uv.lock`.
- **Depends on:** T0.1 (spicelib's SimRunner must be able to find the executable to smoke-test).
- **Gotchas:**
  - **DECIDED: hard cutover.** Remove `pyspice` in this task — do NOT keep a transitional dual-install. The simulation layer will be broken between this task and M2 completion; that is accepted (PySpice-era work is archived externally, and the migration proceeds straight through M2 before any extraction is needed again). Sequence T0.2 → T1.x → T2.x without pausing.
  - `.spiceinit` deprecation: check repo for the file and any docs referencing it; also check `common/__init__.py` — it installs the PySpice `send_char` suppressor at import time, which will fail once pyspice is gone even if no simulation runs.

### T0.3 — Define the new engine interface contract
- **Where:** design note inside `simulation/engine.py` docstring (no code yet).
- **Depends on:** nothing.
- **Gotchas:**
  - The current `SpiceEngine` API is PySpice-shaped: callers build `Circuit` objects and pass them in. Every testbench constructs circuits differently. Decide the replacement contract up front: raw netlist string in → parsed waveform dict out is the cleanest given spicelib's model.
  - **Decision needed:** does anything still need *complex* AC results (Curve 4 uses complex node voltages)? spicelib's RawRead returns trace magnitudes/phases from LTspice ASCII raw files — verify complex reconstruction works before promising Curve-4 parity.
- **Why first:** every task in Phases 1–3 implements against this contract; getting it wrong forces rework across all testbenches.

---

## Milestone 1 — Environment, Parser & Dialect Normalization

### T1.1 — ✅ DONE — Rewrite diagnostics around LTspice `.log` parsing
**Delivered:** `common/diagnostics_ltspice.py` (new module; legacy ngspice section in `diagnostics.py` marked LEGACY, deleted at T6.3). `LTSPICE_DIAGNOSTICS` registry re-keyed to real LTspice 26.0.2 phrasing — validated against an actual failing run (floating-node circuit): `singular_matrix`, `timestep_too_small`, `simulation_aborted`, `parse_error`, `netlist_line_error`, **`floating_node`** (added after real-log check), `param_clamp`, `power_warning` + info banners. `LTSpiceLogDiagnostics` collector keeps the ingest/report/reset contract and adds `parse_log_file()`/`parse_log_text()` for out-of-process capture. Masking fix: letter+digit tokens (`NODEx17`) collapse via explicit `<tok>` pattern (the inherited `\b\d+\b` rule was blind to digits inside words). `common/__init__.py` suppressor installs wrapped try/except — package import survives pyspice uninstall (was a time bomb). `report_and_reset_ltspice` renamed to `report_and_reset` (alias kept until T6.3). Tests: 6 new, 11/11 across both diagnostics modules.

Key discovery: LTspice gmin-stepped through a topology ngspice calls singular — no singular-matrix message appeared; the practical failure signal is the floating-node warning. Engine wiring of `parse_log_file()` deferred to T2.1 by design.

<details><summary>Original task description</summary>
- **Where:** `common/diagnostics.py`.
- **Depends on:** T0.2 (spicelib importable), T0.1 (a real `.log` file to parse).
- **What changes:**
  - Remove `ConsoleQuiet` (fd dup2 redirect) and `install_notebook_quiet_console()` (PySpice `send_char` hook). Both exist solely because ngspice writes C-level stdout in-process. Out-of-process LTspice makes both obsolete — console text lands in a log file instead.
  - Keep the classification registry concept (`NGSPICE_DIAGNOSTICS`) but re-key patterns to LTspice log phrasing ("WARNING: ...", timestep messages, singular-matrix errors).
  - Keep SHA-256 dedup/masking logic unchanged (it operates on text lines regardless of source).
- **Gotchas:**
  - `common/__init__.py` imports/install calls the notebook suppressor — that wiring must be removed in this same task or every module import breaks after pyspice removal.
  - Existing tests (`tests/test_diagnostics.py`, 97-test suite) assert ngspice message patterns; they need rewriting to LTspice equivalents in this task, not deferred to M6.
- **Decision needed:** which LTspice warnings map to existing severity classes? Start with a minimal registry (singular matrix, timestep too small, unknown subckt) rather than porting all rules.

</details>

### T1.2 — Simplify `parser/preprocessor.py`
**Status:** NOT STARTED. Decomposition:
1. *Native-handling verification first (gate for deletion):* run all 4 vendor libs through plain LTspice `.op`/`.tran` with zero preprocessing — confirm native handling of TI ptrc1 param chain, EPC dialect, CPM2 behavioral network, IRFS MODPEX constructs. Any failure → new defect entry, not transpiler resurrection.
2. Delete: fixed-point param evaluator, `spice_literal.py`, `pow_harden.py`, ddt()-commenting rule.
3. Keep: continuation joining, comment stripping; `parser/netlist.py` untouched.
4. Decide: fate of `normalized_lib_copy()` cache layer in `static_sweep.py`.
5. Rewrite: `test_preprocessor.py` / `test_vendor_ti.py`.
- **Where:** `parser/preprocessor.py`, possibly `parser/pow_harden.py`.
- **Depends on:** T0.2.
- **What changes:**
  - LTspice natively evaluates `.PARAM` hierarchies, `{}` expressions, fractional powers, and PSpice TABLE/POLY. The fixed-point param evaluator, `spice_literal` translation, `pow_harden` abs()-wrapping, and the `ddt()` commenting rule can all be retired.
  - Retain: continuation joining, comment stripping, subcircuit extraction support used by `parser/netlist.py`.
- **Gotchas:**
  - **DECIDED: full deletion** of the param evaluator, spice_literal translation, pow_harden, and the ddt() commenting rule — the PySpice-era implementation is archived externally, so nothing needs flagging. Before deleting, run the four vendor libs through plain LTspice once (T1.3 smoke test extended) to confirm native handling of the quirks the preprocessor used to mask (e.g., TI ptrc1 forward-referenced param chain); if any lib fails natively, record it as a new defect entry rather than resurrecting transpiler code.
  - `normalized_lib_copy()` in `static_sweep.py` writes normalized copies under `reports/normalized/`; if normalization becomes a no-op, decide whether the copy step disappears entirely or remains as a cache layer.
  - Tests `test_preprocessor.py` / `test_vendor_ti.py` encode current normalization behavior; expect significant rewrite here.

### T1.3 — 🔶 MOSTLY DONE — Smoke-test LTspice CLI invocation
**Status:** proven de-facto during T0.1/T1.1 (direct `LTspice.exe -b -ascii` runs: clean RC `.tran` → valid ASCII .raw/.log; failing floating-node netlist → log captured & classified). Remaining sliver: one invocation through spicelib's SimRunner itself (we drove the exe directly so far) — folds into T1.2's vendor-native test rather than standing alone.

<details><summary>Original task description</summary>
- **Where:** throwaway script or first cut of `simulation/engine.py`.
- **Depends on:** T0.1–T0.3, T1.1.
- **What changes:** run one trivial netlist (RC divider `.tran`) through `spicelib.SimRunner`, confirm: netlist write → subprocess launch → exit code capture → `.raw` parse → `.log` parse.
- **Gotchas:**
  - First run may trigger LTspice's license/EULA dialog on a fresh install — accept manually once before automation.
  - Confirm `-ascii` flag produces parseable raw output; spicelib reads binary too, but ASCII eases debugging.
  - Measure wall-clock overhead of subprocess+file I/O per run — this informs whether per-bias-point loops (25-point cap sweeps, stepped-OP validation over ~41×5 points) stay viable or need batching into single multi-step netlists.

</details>
---

## Milestone 2 — Datasheet Curve Extraction Parity

### T2.1 — Rewrite `simulation/engine.py` around SimRunner + RawRead
- **Where:** `simulation/engine.py`.
- **Depends on:** T0.3 (contract), T1.3 (smoke test).
- **What changes:**
  - New API surface per T0.3 decision: `run(netlist_text, analysis_directives) -> {trace_name: ndarray}` plus timeout handling and non-zero-exit raising.
  - Port `_collect_diagnostics` to read the `.log` produced by each run and feed T1.1's classifier.
  - Temperature propagation: PySpice set `.temp` via simulator kwargs; LTspice needs explicit `.temp`/global temp card or per-device temp parameters — decide mechanism here.
- **Gotchas:**
  - Solver options surface (RELTOL/ABSTOL/VNTOL/GMIN/ITL1/ITL4) has no 1:1 LTspice mapping (`.options` names differ: `reltol`, `abstol` exist; ITL counters don't). Map what exists, drop the rest deliberately.
  - The engine currently swallows option-rejection warnings — preserve that tolerance in the new wrapper.
- **Decision needed:** synchronous-only execution or expose spicelib's parallel runners? Single-threaded is sufficient for the pipeline today; don't buy complexity now.

### T2.2 — Introduce a netlist-text builder utility
- **Where:** new small module (e.g., `simulation/netlist_text.py`) or methods on the engine.
- **Depends on:** T2.1.
- **What changes:** replace PySpice `Circuit` object-tree construction with formatted SPICE text generation shared by all benches: element lines, X-instance line assembly (subsuming `xinst.py`), `.include` placement, analysis directive blocks, node naming consistency (`d`, `g`, `0`, `tj_<ref>`).
- **Gotchas:**
  - `common/roles.py` ROLE_NODES maps source→node "0" (ground); keep that convention exactly.
  - Ground node name: LTspice accepts `0`; PySpice's `circuit.gnd` mapped to 0 internally — make sure generated text never emits literal `gnd` (ngspice treats it specially; LTspice doesn't).

### T2.3 — Migrate `static_sweep.py` (Curves 1, 2, 3, 6, 7)
- **Where:** `simulation/testbenches/static_sweep.py`.
- **Depends on:** T2.1, T2.2.
- **What changes:** each bench emits its DC-sweep netlist, runs once, extracts Id from the .raw current trace. Sign conventions flip: PySpice negated source currents (`-analysis.Vds`); LTspice raw traces carry the sign of the element as declared — re-derive the sign bookkeeping and keep `_check_sign` assertions intact.
- **Gotchas:**
  - Body-diode bench relies on a coarse-step retry when DC sweeps diverge at the knee; LTspice convergence behavior differs — implement retry only if the failure reproduces, don't port blindly.
  - Vth bench ties drain-gate via positional xargs including the tj port append — reproduce the exact connection order from `resolve_pin_connections`.
  - **Parity gate:** run all 4 vendors and diff extracted values against existing YAML outputs in `config/outputs/` before declaring done. Small numeric differences expected (different solver tolerances); decide an acceptance band now (suggestion: rel diff <1% on curve scalars, <0.5% span on curve arrays).
- **Decision needed:** sweep implementation — LTspice `.dc` vs stepping a voltage source with `.step`. `.dc` matches semantics directly.

### T2.4 — Migrate `capacitance.py` (Curve 4)
- **Where:** `simulation/testbenches/capacitance.py`.
- **Depends on:** T2.1, T2.2.
- **What changes:** two AC probe circuits (gate-driven, drain-driven through 1 H bias inductor) emitted as `.ac lin 1 f f` netlists; extract |V| (and phase if needed) from raw traces; compute Ciss/Coss/Crss identically.
- **Gotchas:**
  - Complex-value extraction from LTspice ASCII raw must be validated early (magnitude+phase → complex) — this is the highest technical-risk item in M2.
  - The 1 A AC drive is expressed as `Ixxx n+ n- DC 0 AC 1` — confirm LTspice accepts the identical form.
  - NaN masking of non-physical points and trailing-envelope trimming are pure post-processing — port unchanged.
  - HOST-02 disappears (stateless processes): remove `_ac_retries=3` machinery only after confirming no other transient failure mode needs retries.
- **Parity gate:** same as T2.3 — diff against stored Curve-4 YAML outputs for all vendors.

### T2.5 — Confirm `gate_charge.py` needs no algorithmic change
- **Where:** `simulation/testbenches/gate_charge.py`.
- **Depends on:** T2.3, T2.4 (consumes their outputs).
- **What changes:** none expected — quasi-static reconstruction consumes Curve 2 + Curve 4 data objects. Verify it runs unmodified on migrated outputs; adjust only if data-container shapes changed.
- **Gotchas:** VM-02 means there is NO transient fallback; do not attempt a constant-current Ig transient in LTspice as part of migration (evaluation doc explicitly keeps reconstruction essential).

### T2.6 — Full curve-parity regression across 4 vendors
- **Where:** comparison script/reporting only.
- **Depends on:** T2.3–T2.5.
- **What changes:** regenerate `config/outputs/<vendor>/*.yaml` via the new backend; produce a side-by-side diff table vs committed outputs for IRFS7730PBF, CSD16412Q5A, EPC2216, CPM2-1200.
- **Gotchas:**
  - EPC2216 hang at Vbus ≥12 V should disappear (EPC-01) — if it doesn't, that's evidence LTspice didn't fix time-step control as claimed; document either way.
  - CPM2 turn-off abort (VM-08) should resolve in LTspice; re-run the switching golden capture afterward (Phase 3) to confirm the full 60 µs span now simulates.
- **This is the M2 exit criterion** — parity against the archived PySpice-era YAML outputs is the acceptance bar. (`pyspice` removal already happened at T0.2 under the hard-cutover decision.)

---

## Milestone 3 — Tier-2 Synthesis Validation

### T3.1 — Validate Tier-2 netlist syntax under LTspice
- **Where:** `models/netlist_builder.py` (validation only; edit only if syntax fails).
- **Depends on:** T2.6 (vendor models runnable in LTspice for golden comparison).
- **What changes:** take each `<device>_PIECESPICE_T2.lib` and simulate in LTspice:
  - `G_CH di ss VALUE={...}` — evaluation doc says LTspice supports it; verify the soft-abs ramp chains (`(x)+abs(x))/2`) evaluate identically. If not, switch emission to `B_CH` form (~30-line change flagged in the plan).
  - `C_QOSS di ss Q={...}` nonlinear capacitor — **highest-risk item**: the evaluation doc asserts "LTspice supports identical charge-source expressions" but gives no evidence. Verify specifically that `Q=` with a brace expression is legal LTspice syntax; if unsupported, a redesign (B-source integrator or PWL capacitor) is required — escalate before proceeding.
  - `.MODEL D(...)` body diode — trivially portable.
- **Gotchas:**
  - ngspice-specific syntax crept into the emitter historically (ASCII-only rule, `Q=` keyword choice was made FOR ngspice-34). Audit the emitted text line by line rather than trusting "identical support."
  - Numeric formatting `%.6e` — fine everywhere.

### T3.2 — Move stepped-OP validation to the new engine
- **Where:** `pipeline/fitting.py` (`compare_tier2_vs_golden`, `compare_tier2_vs_golden_full`).
- **Depends on:** T2.1, T3.1.
- **What changes:** replace per-point PySpice OP circuits with generated OP netlists through SimRunner.
- **Gotchas:**
  - Current code spawns one circuit per grid point (~41 × #Vgs ≈ 205 subprocess launches per vendor). With disk-I/O-per-run overhead this may become slow — measure; if unacceptable, batch into a `.step`-based sweep or reduce grid density. **Decision needed on acceptable wall-clock.**
  - These validators are called by notebooks; signature stability matters for notebook compatibility (user requirement: don't break previous notebooks).

---

## Milestone 4 — Switching Dynamics & DPT

### T4.1 — Migrate `simple_switching.py`
- **Where:** `simulation/testbenches/simple_switching.py`.
- **Depends on:** T2.1, T2.2, T3.1 (Tier-2 must run under LTspice first).
- **What changes:** emit the Vbus–Rload–DUT pulse circuit as text; `.tran 0 <end> 0 <step>`; parse `V(drain)`, `V(gate)`, load current from raw; return the same `SwitchingRun` dataclass.
- **Gotchas:**
  - Keep `F_SW_HZ/DUTY/N_PERIODS_STEADY` constants and `steady_state_slice` untouched — downstream metrics depend on them.
  - The notebook-safe console suppressor import disappears with pyspice (T1.1) — ensure module still imports cleanly.
  - Fixed 2 ns output step: express as LTspice maximum-timestep parameter; verify actual returned time grid respects it (raw decimation settings can silently thin data).

### T4.2 — Re-run golden transients for all 4 vendors
- **Where:** data capture only (`data/<vendor>_golden.csv` regeneration).
- **Depends on:** T4.1.
- **What changes:** recapture goldens previously blocked/truncated:
  - CPM2-1200: expect full 60 µs span now (VM-08 resolved claim) — if the turn-off abort persists, LTspice did NOT fix it and the golden stays truncated; update `docs/methodology_audit.md` + defect register accordingly.
  - EPC2216: attempt YAML Vbus=12 V again (EPC-01 claim); fall back to 11 V if it still hangs.
  - TI: VM-09 claims BSIM3 `.tran` works natively.
- **Gotchas:** solver differences mean golden waveforms will differ slightly from ngspice captures — fitness baselines shift; re-baseline rather than mixing backends within one comparison.

### T4.3 — Build `double_pulse.py` (new DPT bench)
- **Where:** new file `simulation/testbenches/double_pulse.py`.
- **Depends on:** T4.1.
- **What changes:** clamped-inductive-load two-pulse circuit; extraction of tr/tf/td/Eon/Eoff/Eoss per plan §2.5 using existing `metrics/switching.py` + `metrics/loss.py`.
- **Gotchas:**
  - This is NEW scope (the methodology audit records DPT as never implemented) — not a migration. Size it honestly: freewheeling diode selection, L value, gate drive levels are design decisions needing user input (per project rules: get device/bench specs from yaml, not hardcoded).
  - Requires a yaml schema addition for DPT conditions (L, duty timing, dead-time) — extend `config/inputs/*.yaml` deliberately.

---

## Milestone 5 — Converter Benchmark (future scope)

### T5.1 — `buck_converter.py` bench
- **Depends on:** T4.x complete; separate planning session recommended before starting.
- **Gotchas:** entirely new territory (PWM source modeling, output filter, load); not started during migration proper.

### T5.2 — Benchmark harness + notebook 04 population
- **Depends on:** T5.1.
- **Gotchas:** wall-clock comparisons are only fair same-backend; compare Tier-2-in-LTspice vs vendor-in-LTspice, never mixed.

---

## Milestone 6 — CLI, Tests, Cleanup

### T6.1 — Update `cli/main.py`
- **Where:** `cli/main.py`.
- **Depends on:** M2 complete.
- **What changes:** `parse` command unaffected (parser retained). `sweep`/`curves` commands already delegate to pipeline — verify they run without pyspice imports anywhere on the path.
- **Gotchas:** `reporting/html_report.py` is engine-independent — no change expected despite being listed in the eval doc.

### T6.2 — Rewrite affected tests
- **Where:** `tests/` (test_diagnostics, test_preprocessor, test_vendor_ti, test_gate_charge fixtures, conftest).
- **Depends on:** corresponding phase tasks.
- **Gotchas:**
  - Decide mock-vs-live-LTspice strategy: live-LTspice tests are slow (~seconds each) and need the exe present in CI/dev; mocks are fast but won't catch syntax regressions. **Recommendation:** unit tests mocked, one marked slow end-to-end test per milestone gated behind an env flag.
  - `conftest.py` may seed ngspice-specific fixtures.

### T6.3 — Final purge & documentation
- **Where:** `pyproject.toml`, README badges (ngspice badge), `docs/*`, memory of `.spiceinit`.
- **Depends on:** everything above.
- **What changes:** drop `pyspice` (if not already at T2.6), delete dead preprocessor paths (T1.2 decision executed), update README quick-start (LTspice prerequisite), dev-log entry per house rule.

---

## Dependency Graph (condensed)

```
T0.1 ─┬─▶ T0.2 ─┬─▶ T1.1 ─▶ T1.3 ─▶ T2.1 ─┬─▶ T2.2 ─┬─▶ T2.3 ─┬─▶ T2.6 ─▶ [drop pyspice]
T0.3 ─┘         │                          │         ├─▶ T2.4 ─┤
                └─▶ T1.2                    │         └─▶ T2.5 ──┘
                                            └──────────────▶ T3.1 ─▶ T3.2
                                                                 T4.1 ─┬─▶ T4.2
                                                                       └─▶ T4.3
                                                       (M5/T6 tail work)
```

## Top Risks (decide before writing code)

1. **`C… Q={expr}` support in LTspice** (T3.1) — the whole Tier-2 charge story hinges on it; verify in the first week with a 5-line netlist before committing to the migration.
2. **Complex AC extraction from raw files** (T2.4) — Curve 4 parity depends on it.
3. **Subprocess-launch overhead vs per-point loops** (T2.3, T3.2) — may force netlist batching redesign.
4. **Hard-cutover window** (T0.2, decided) — the repo cannot run any simulation between removing pyspice and finishing M2; keep that gap short by sequencing T1 → T2 without side quests.
