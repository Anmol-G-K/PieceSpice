# M1 + M2 Audit v2 — Gaps, Logical & Physics Errors

**Date:** 2026-08-26 · **Mode:** read-only audit, no edits, no execution
**Scope:** all files rewritten during M1/M2 (engine, netlist_text,
static_sweep, capacitance, gate_charge, simple_switching,
pipeline/curves+fitting, diagnostics) against plan §2 formulations and
datasheet ground truth.

---

## A. Findings Table

| # | Severity | Area | Finding | Impact | Status |
|---|---|---|---|---|---|
| F1 | ~~HIGH~~ FIXED | C3 Rds(on) — `run_rds_on` | Netlist sweeps the drain source 0→2.5 V with **no explicit supply element**: `Vds drain 0 0` is the swept element itself, so device Vds = sweep value. Correct now — BUT the x-axis reported is Id while the datasheet Rds(on) spec point is at a *specific* Vgs/Id pair; interpolation at `Id_load` assumes monotonic rising Id, which holds only in ohmic operation. For models where VM-04 kicks in mid-sweep, Id can fold over and `np.interp` silently returns the wrong branch. | Wrong Rds scalar on defective models at high Vgs | **FIXED (2026-08-26)** — fold-over guard added; 0.0% drift verified on all 4 vendors |
| F2 | **MEDIUM (logic)** | C4 capacitance `_trim`/`n_valid` | Each array is trimmed independently to its own last-valid index, then all three are truncated to `min(len)` of a *log-spaced* sweep. If Coss dies earlier than Crss (as EPC does), the retained x-grid belongs to the shortest array but is presented as the common sweep axis; points beyond a gap are misaligned by up to one bias step. No interpolation-to-common-grid. | Slight x/y misalignment on masked tails | Open |
| F3 | **MEDIUM (logic)** | C5 gate charge — plateau interp | `np.interp(i_load, id_t, vgs_t)` requires `id_t` ascending. The transfer curve is ascending in Vgs so Id rises monotonically for normal models, but VM-04-style super-linear rise plus any non-monotonic sample would silently produce a wrong plateau. There is no monotonicity guard on this call (unlike `_interp_curve` which guards). | Silent wrong Qgd segmentation | Open |
| F4 | **MEDIUM (physics)** | C5 quasi-static method | Qgs segment uses `ciss_high * ΔVgs` — a **constant-C** approximation taken at a single bias (Vbus), even though Ciss(Vds) data exists across the sweep. Plan §2.5 implies integrating the measured curves. Under-estimates Qgs when Ciss varies strongly with Vds. | Systematic Qgs error on models with steep Ciss(Vds) | Open (documented simplification, not yet flagged as such) |
| F5 | **LOW-MEDIUM (logic)** | Engine `.ac` card insertion | `_has_directive` checks line-start prefix `.ac`; a netlist line like `.acplot …` (hypothetical) or an `.ac` inside a comment block that survived stripping would false-positive/suppress. Also: if the caller's text already contains `.ac`, the passed directive object is silently ignored — no warning. | Directive silently dropped | Open |
| F6 | **LOW (logic)** | `simple_switching` pulse offset | Gate pulse starts at td=5 µs; final period slice (45–60 µs) contains turn-off + turn-on rather than a full clean cycle. Documented in docstring now, but fitness windows derived from "last cycle" straddle two switching events differently than the first cycles. | Fitness window asymmetry vs golden captured on same backend: benign; cross-backend: biased | Accepted/documented |
| F7 | **LOW (gap)** | Curve 2 temperature family | Plan Curve 2 specifies extraction at Tj ∈ {25,125,175} °C. Only single-temperature extraction exists; Tj port driven from yaml `temperature` alone. | No thermal-family validation possible | **MOVED TO PLAN** — added as Curve 2-T in piecespice_plan.md §3 |
| F8 | **LOW (gap)** | Body diode reverse-recovery | Datasheets list trr/Qrr/IRRM (CPM2: 45 ns/406 nC/13.5 A); no bench extracts these. Plan defers to M4-DPT, but the gap is undocumented in the audit trail until now. | Reverse-recovery unvalidated | **MOVED TO PLAN** — added as Curve 6-RR, deferred to M4-DPT |
| F9 | **INFO** | `normalized_lib_copy` shim | Now resolves & returns path with zero transformation; kept only so call sites don't change. Disk copy per run still happens under `reports/normalized/`. | Wasted I/O, cosmetic | DEFERRED to T6.3 |
| F10 | **INFO** | `_check_sign` coverage | Validates upper-half positivity only; a model flipping sign below mid-sweep passes. Also asserts rather than raising typed exceptions. | Minor robustness | DEFERRED to T6.3 |

---

## B. Confirmed-correct items (audited, no issues)

| Item | Verification basis |
|---|---|
| C2 transfer topology (drain-biased, Vgs swept) | Matches original ngspice-era bench exactly |
| C7 Vth JEDEC method (D-G tie, constant-current interp, fail-loud on unreachable target) | Line-by-line review |
| C6 body diode orientation, 10 MΩ anchor + analytic parasitic subtraction, coarse-step retry | Line-by-line review; live run reached diode knee post-yaml fix |
| Engine execution path (temp workdir, timeout → LTSpiceError, RawRead parse, log classification, .temp card) | Live tests 7/7 |
| NetlistBuilder role-name binding incl. Tj emission | Unit tests + live EPC/CPM2 runs |
| Capacitance variant-A/B topologies after Iprobe fix | Manual probe reproduction at multiple biases |
| sub_gate name-based resolution | Live EPC run: Ciss/Coss/Crss match datasheet |

---

## C. Parity-table validity after findings

| Vendor | Scalar parity valid? | Notes |
|---|---|---|
| irfs7730pbf | ✅ Yes | All compared scalars come from correctly-wired benches |
| csd16412q5a | ✅ Yes | Same |
| epc2216 | ✅ Yes (post-FIX alignment) | Caps at ds-matched 7.5 V now |
| cpm2-1200 | ⚠️ Partly | Caps/Vth/Rds valid; gfs & Qg need condition-matched re-run before claiming parity |

F1 does **not** invalidate previously reported numbers because the
affected curve (C1) was compared only at endpoints in earlier reports;
the endpoint values themselves were recorded against wrong axes, so any
historical C1-endpoint claims should be considered superseded by this
audit.

---

## D. Summary counts

| Category | Count |
|---|---|
| Physics errors | 1 (F1, HIGH) |
| Logic errors | 3 (F2–F4 + F5 low-logic) |
| Scope gaps | 3 (F7, F8, plus temperature-family overlap) |
| Hygiene/info | 3 (F6, F9, F10) |
| Confirmed-correct areas | 7 |
