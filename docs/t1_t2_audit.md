# T1/T2 Migration Audit — Traceability & Logical Flaws

**Date:** 2026-08-26
**Scope:** migration tasks T0.1–T1.2 (executed) and T2.x (not started). Every `.py` file scanned for backend bindings; callers, tests, and docs cross-checked.
**Method:** automated grep-matrix (PySpice/spicelib/LTspice/ngspice mentions per file) + manual caller-graph and test-alignment review.

---

## 1. Backend Binding Matrix (current state)

| File | PySpice | spicelib | Status in migration |
|---|:---:|:---:|---|
| `simulation/engine.py` | 5 refs | **0** | ❌ NOT converted — still the PySpice/ngspice engine; every bench runs through it |
| `simulation/engine_ltspice_proposal.py` | 2 (doc-only) | 4 (doc-only) | ⚠️ Proposal only — not implemented, not wired |
| `simulation/testbenches/static_sweep.py` | 2 | 0 | ❌ Not converted (blocked on engine) |
| `simulation/testbenches/capacitance.py` | 2 | 0 | ❌ Not converted |
| `simulation/testbenches/gate_charge.py` | 1 | 0 | 🔶 Imports PySpice units but algorithm is pure-Python; quasi-static path unaffected by backend |
| `simulation/testbenches/simple_switching.py` | 0 | 0 | ✅ Backend-neutral already (uses engine + diagnostics install) |
| `simulation/testbenches/xinst.py` | 2 | 0 | ❌ Not converted (subsumed by netlist-text builder at T2.2) |
| `parser/preprocessor.py` | 0 | 0 | ✅ Converted (thin pass-through) |
| `parser/netlist.py` | 0 | 0 | ✅ Backend-neutral, untouched as planned |
| `parser/spice_literal.py` / `pow_harden.py` | — | — | ✅ Deleted |
| `common/diagnostics.py` | 15 | 0 | 🔶 LEGACY section retained intentionally; deleted at T6.3 |
| `common/diagnostics_ltspice.py` | 3* | 1* | ✅ New layer (*refs are docstrings/comments only) |
| `common/__init__.py` | 1 | 0 | ✅ Import-safe (all suppressor installs wrapped) |
| `pipeline/curves.py` | 0 | 0 | ✅ Backend-neutral orchestration |
| `pipeline/fitting.py` | 2 | 0 | ❌ Validators use PySpice Circuit directly (`compare_tier2_vs_golden*`) — listed for M3 but never touched in T1 |
| `models/netlist_builder.py` | 0 | 0 | ✅ Emits text; backend-neutral (syntax validation is T3.1) |
| `cli/main.py` | 0 | 0 | ✅ Delegates to pipeline; clean |
| `notebooks/simple_plotting.py` | 0 | 0 | ✅ Clean of imports; two stale "ngspice-46" strings in labels/metadata remain |

**Headline finding:** *zero production code currently calls spicelib.* The dependency is installed and the proposal exists, but no module imports it. All simulation still flows through PySpice→ngspice-34 shared library. This is consistent with the plan (engine rewrite is T2.1), but it means **T1 delivered infrastructure around the new backend, not the new backend itself.**

---

## 2. Traceability Issues

### TI-1 — `simple_switching.py` calls a function that will be deleted
`install_notebook_quiet_console()` is invoked at module import (line 27). It lives in the LEGACY ngspice section of `diagnostics.py`, scheduled for deletion at T6.3. When that deletion happens without this import being removed first, **the switching testbench fails at import**. Traceability gap between T6.3 cleanup and this call site.

### TI-2 — `metrics/fitness.py` duplicates metric logic with different constants
Uses `np.median(v_gs[len//4:])` for plateau inference while `metrics/switching.py` was fixed to the 95th percentile (duty-cycle bug fix). Two sources of truth for the same math — one of them re-introduces a bug already fixed once. Flagged in methodology_audit §5.10; still unresolved after T1.

### TI-3 — `notebooks/simple_plotting.py` carries stale backend labels
Line ~51 comment says "Capture the verified inline ngspice-46 CSV"; metrics JSON `"backend"` field says `"golden=inline ngspice-46 CSV; tier2=shared-lib"`. After migration both claims become false — the metadata would lie about provenance. Cosmetic now, misleading later.

### TI-4 — `tests/conftest.py` hardcodes engine identity
Sets `config._metadata["Engine"] = "NgSpice + PySpice"`. Test infrastructure asserts the old backend name; nothing updates it post-migration.

### TI-5 — Deleted modules survive in `docs/methodology_audit.md` as live
The audit (correctly, being a snapshot) describes spice_literal/pow_harden as implemented. No pointer was added noting they were removed at T1.2. Future readers following the audit's file inventory will find dead references.

### TI-6 — README badge still advertises ngspice
README line 10: `[![ngspice](...badge-ngspice-34--shared-orange)]`. Post-cutover this is wrong, and there's no LTspice prerequisite mention anywhere in Quick Start. T6.3 covers it, but the README is user-facing *now*.

### TI-7 — `pyproject.toml` lists both backends
`pyspice>=1.5` and `spicelib>=1.6.3` coexist per your instruction ("both libraries there"). Consequence: the codebase can silently keep passing tests via ngspice while appearing migrated. There is no guard preventing new code from importing PySpice. Suggest a lint rule or CI grep gate at T2 completion ("no `from PySpice` outside LEGACY-marked files").

### TI-8 — `pipeline/fitting.py` validators absent from T1/M1 task list
`compare_tier2_vs_golden(_full)` build PySpice Circuits inline. They appear in eval-doc M3 (§2.3 Milestone 3 task 2), but they import PySpice *today*, meaning any code path touching them breaks on pyspice removal. They should have been enumerated in the T0.2 blast-radius check — traceability miss in my own earlier analysis (I caught capacitance/static_sweep/cli but missed fitting.py's direct imports).

### TI-9 — `gate_charge.py` imports PySpice units it doesn't need post-migration
`from PySpice.Unit import u_A, u_s, u_V` at line 19 — used only by the deleted transient bench remnants. Pure-Python reconstruction needs none of them. Small, but it's a PySpice string in a file the plan marked "Low (<30 lines)".

### TI-10 — Migration todo's T2 parity criterion references YAML outputs that will change
T2.6 exit criterion: diff against committed `config/outputs/*.yaml`. But those outputs were produced by ngspice-34 with preprocessor normalization; LTspice values will legitimately differ (different solver, tolerances). The acceptance band (<1% scalars) was suggested in the todo but never ratified — if applied naively, some curves may fail parity for reasons that are *expected physics/solver drift*, not bugs.

---

## 3. Logical Flaws & Risks

### LF-1 — Hard cutover left the repo in a state where its primary pipeline cannot run
Accepted decision (T0.2), but concretely: `run_all_curves()` → CapacitanceTestbench → ngspice fails on raw TI lib ("Undefined number [pwidth]"). Four tests skipped. Until T2.1–T2.3 land, **no vendor extraction is executable end-to-end**. Any urgent data need during this window has no workaround inside the repo (only the external archive).

### LF-2 — Engine proposal vs engine reality drift risk
`engine_ltspice_proposal.py` documents decisions (SimResult shape, directive objects, timeout policy). Nothing enforces the implementation to match; when T2.1 starts from scratch, the proposal can silently diverge. Mitigation: make the proposal THE file — implement in place, delete the proposal header.

### LF-3 — Diagnostics dual-collector ambiguity
Two singletons now exist: `get_diagnostics()` (ngspice) and `get_ltspice_diagnostics()`. The engine proposal says results carry classified warnings, but doesn't say WHICH collector feeds the global report, nor whether legacy `report_and_reset()` (called anywhere?) still aggregates during transition. Risk: warnings land in one collector while the CLI/report reads the other. Decide at T2.1: single collector interface with two parsers feeding it.

### LF-4 — `normalized_lib_copy` is now semantically dead weight
It still copies libs to `reports/normalized/` and cleans them up, but normalization is near-no-op. Harmless, yet it adds disk churn per run and confuses readers ("why normalize?"). The todo deferred the decision to M2 — fine — but note the copy also masks include-path issues: LTspice resolves `.include` relative to the netlist location, and copying changes effective paths. Verify include resolution when benches migrate.

### LF-5 — CPM2 native verification exercised `.op` only
Step 1 verified parsing + operating point, NOT transient. VM-08's turn-off abort was observed under `.tran`. LTspice may parse CPM2 fine and still abort at the falling edge — unknown until T4.2. The todo says golden recapture happens then; just don't treat "native .op OK" as "VM-08 resolved."

### LF-6 — EPC Vbus≥12 hang untested under LTspice
Same class as LF-5: EPC-01 claim ("LTspice handles GaN behavioral switching") has zero evidence in this repo yet. First EPC transient in M2 should explicitly test 12 V before anyone trusts the claim.

### LF-7 — Fitness tolerance baselines will shift silently at M2
Golden waveforms recaptured on LTspice differ numerically from ngspice captures (LF/TI-10 family). `metrics/fitness.py` compares golden-vs-tier2 captured on the same backend — safe — but `data/*_golden.csv` artifacts mix backends across vendors (EPC@11V ngspice-46, CPM2 truncated ngspice-46, others ngspice-34). Cross-vendor comparisons of stored goldens are invalid until uniformly recaptured. Not tracked as a task anywhere.

### LF-8 — `simple_plotting.py` Tier-2 sim still routes through the OLD engine
It calls `run_simple_switching` → `SpiceEngine` (PySpice). Works today; breaks the moment pyspice is pulled. It's the user-facing deliverable for plots, yet it appears nowhere in the M2 conversion list (eval doc §2.2 table omits notebooks entirely). Add explicit task or accept it breaks until T4.1.

---

## 4. Conversion Scorecard (honest)

| Layer | Converted? |
|---|---|
| Dependency manifest | 🔶 partial (both present, by choice) |
| Parser/dialect layer | ✅ complete (T1.2) |
| Diagnostics layer (new) | ✅ complete (T1.1) — unwired |
| Execution engine | ❌ untouched (T2.1) |
| DC benches | ❌ untouched (T2.2/2.3) |
| AC bench | ❌ untouched (T2.4) |
| Gate charge | ✅ effectively (backend-neutral algorithm) |
| Switching bench | ✅ neutral today / ⚠️ TI-1 import hazard |
| Netlist builder | ✅ neutral (validation pending T3.1) |
| Fitting validators | ❌ untouched (M3 scope) |
| Notebooks/plots | ✅ neutral imports / ⚠️ stale labels (TI-3), old-engine dependency (LF-8) |
| Tests | 🔶 rewritten where affected; skips document gaps |

## 5. Recommended immediate actions (no code written)

1. Fix TI-1 now (one-line guard) — cheapest way to defuse the T6.3 trap.
2. Add TI-8 (fitting validators) to the T2.1 blast-radius checklist.
3. Ratify the T2.6 parity acceptance band before running parity (else the exit criterion is unfalsifiable).
4. Record LF-7 as an explicit M4 task: uniform golden recapture on LTspice for all four vendors before any cross-vendor fitness claim.
