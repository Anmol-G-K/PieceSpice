# Implementation Closeout — Acceptance Policy, Device-Current Bars, and Final Records

**Date:** 2026-11-21 (IST)
**Status:** Closeout. Documentation only; no code, configuration, model or
record was changed by this report.
**Evidence:** Every number is quoted from `reports/comparison/<vendor>/*.json`
or from the executed test/CI evidence in §4. No number is estimated,
imputed, or carried over from a superseded README.

---

## 1. What the implementation changed

**Acceptance became data.** Numeric gates now live in
`config/acceptance.yaml` (schema `piecespice.acceptance/1.0`), with each value
labelled `recorded_baseline` or `project_decision`. The UROM record copies the
policy path and SHA-256 and fails closed on a malformed file instead of
silently falling back. The shipped gates are: static 7.0683 / 8.4935 / 2.0394
/ 3.4287 % (IRFS / TI / EPC / CPM2), endpoint 4.13 %, timing band 5 % relative
plus 1 ps absolute, charge budget 0.5 %, and speed at least 1.05× solver with
a 95 % confidence interval from 2000 resamples.

**The DPT current identity was corrected.** Both golden and ROM now use
`I(Lload)-I(Dclamp)` (`current_definition` in every record). The old
inductor-current-only metric hid clamp current; its Id bars (0.60 / 0.048 /
0.049 / 0.261 %) are retired. G3 now uses a declared **Id 5 %** application
requirement, while Vds keeps the recorded per-vendor baselines (10.0 / 5.583 /
4.894 / 3.0 %). This is a metric correction, not a relaxed model result.

**G4 was made independent.** Timing/energy is scored against each vendor's
golden DPT crossing
(`timing_bar_source=tier3_validation.json#dpt_comparison.<metric>.golden`).
The former Tier-2 self-comparison survives only as `G4_tier2_regression`,
labelled `diagnostic only; never scored`. That diagnostic passes 4/4 while the
independent gate fails 4/4 — the single most consequential outcome of this
closeout.

**G5 requires an effect, not just a ratio.** A speed verdict now requires a
measured solver effect plus its confidence interval on the scoped 20-pulse
bench; near-unity and slower rows are rejects.

**Records are hash-manifested.** Each final record carries
`artifact_manifest` (input files plus emitted library) and
`artifact_manifest_verified.ok=true`; `emitted_lib_sha256` matches the
on-disk `.lib`. Static "worst endpoint" is measured at the configured
`Vds_sweep_max`.

**Basis and emitter work landed behind opt-ins.** ANN emits charge from its
fitted net (`c1` monotone, zero-anchored; G6 passes) with a soft gm/gds
monotone penalty (`hard_guarantee=false`) and hold-out fields that report
`not_measured` when no hold-out is reserved. Tier-3 `v1` keeps its four-line
default emitter; V2 physical elements (RG/RD/RS, gate charge, body diode),
the measured 2-D Qoss surface (optional temperature slice) and compact
Vds-axis emission are opt-in and off in the shipped records. Multi-trajectory
and CV are implemented through `config['tier3']['training_conditions']`.
`simplis` records the Cgd axis provenance, runs a measured C-V diagnostic,
and appends a physical end segment to every charge table. Tier-4's LUT gained
a real measured grid, an optional temperature axis, Ron/Vt parameters and a
caller-configured thermal network.

---

## 2. Final per-vendor results

### 2.1 Tier-3 `v1` (default TPWL)

| Device | Cells | Fit rel-RMSE | Emitted rel-RMSE | DPT Vds / Id | Forced solver |
|---|---:|---:|---:|---:|---:|
| IRFS7730PBF | 32 | 2.16 % | 1.6987 % | 12.6978 / 10.9582 % | 0.5827× |
| CSD16412Q5A (TI) | 32 | 1.54 % | 2.1531 % | 2.9248 / 4.8962 % | 0.0946× |
| EPC2216 | 32 | 2.37 % | 7.2226 % | 2.8461 / 1.4694 % | 0.4202× |
| CPM2-1200 | 24 | 0.61 % | 0.9749 % | 4.3785 / 5.1932 % | 0.1788× |

All four forced-bench rows are slower than golden. The default emission is
`compact=false`, `source=default`; no physical element is inferred. The V2,
2-D-charge and compact paths exist but are not exercised in these records, so
they carry no final fidelity or speed claim.

### 2.2 Tier-3 `simplis`

| Device | Static rel-RMSE | DPT Vds / Id | 20-pulse solver / wall | Channel / charge |
|---|---:|---:|---:|---|
| IRFS7730PBF | 7.0683 % | 8.7044 / 8.5167 % | 1.0705× / 1.0449× | tier2 / tier2 |
| CSD16412Q5A (TI) | 8.4935 % | 5.1162 / 4.2121 % | 5.1455× / 2.9237× | tier2 / tables |
| EPC2216 | 2.0394 % | 2.9490 / 1.7637 % | 1.1473× / 1.0686× | tier2 / tables |
| CPM2-1200 | 3.4287 % | 3.8525 / 4.5581 % | N/A / N/A | tier2 / tables |

TI's 5.1455× is driven by its stiff golden; CPM2's golden adaptive arm aborts
(`rc=1`), so its ratio is N/A. The selected charge mode is `tier2` only for
IRFS; TI / EPC / CPM2 retain `tables`. The ANN residual candidate is rejected
4/4. The Cgd axis is `vds_at_vgs0` because only one Vgs=0 sweep exists; a
bias-aware `Vdg=Vds−Vgs` sweep is preferred when available, and no bias axis
was synthesized. The C-V diagnostic (10 % budget, separate from the 0.5 %
charge budget) measures Cdg errors 15.825 / 13.224 / 6.806 / 50.692 % and Cgs
≤1.249 %, so the diagnostic is disclosed rather than hidden. All emitted
tables use `clamp_charge_above_last=True`: above the last breakpoint Q stays
constant instead of extrapolating the last slope.

### 2.3 Tier-3 `urom` canonical C0 — overall `reject` 4/4

| Device | Fidelity | G4 timing/energy | G5 speed | Overall | Static | Solver / wall | G3 Vds / Id vs bar |
|---|---|---|---|---|---|---|---|
| IRFS7730PBF | reject (Id 9.0127 %) | reject | reject | **reject** | 7.0683 % | 1.0073× / 0.9860× | 9.999 / 9.013 vs 10.0 / 5.0 |
| CSD16412Q5A (TI) | accept | reject | accept | **reject** | 8.4935 % | 5.3021× / 3.0086× | 1.984 / 1.259 vs 5.583 / 5.0 |
| EPC2216 | accept | reject | reject | **reject** | 2.0394 % | 0.9863× / 1.0169× | 0.704 / 0.662 vs 4.894 / 5.0 |
| CPM2-1200 | accept | reject | accept | **reject** | 3.4287 % | 1.1023× / 1.0583× | 2.893 / 3.209 vs 3.0 / 5.0 |

Fidelity accepts three of four; the independent G4 gate rejects all four;
speed accepts two of four. G0, G1, G2 and G6 pass on every C0 row. There is no
global or portable fidelity/speed claim (`global_claim=false`), and TI's
5.3021× is golden-stiffness driven. The keyed ANN records and libraries are
separate, non-default artifacts and reject 4/4; tuning is `not_run`.

### 2.4 Tier-4 (ideal switch + loss LUT)

| Device | Solver / wall | LUT status | Thermal | Efficiency |
|---|---:|---|---|---|
| IRFS7730PBF | 1.2212× / 1.0994× | constant_node (1×1) | not_configured | output_power_required |
| CSD16412Q5A (TI) | 6.4730× / 3.2734× | constant_node (1×1) | not_configured | output_power_required |
| EPC2216 | 1.2440× / 1.1019× | grid_interpolation (3 finite nodes) | not_configured | output_power_required |
| CPM2-1200 | N/A / N/A | single_node (2×2 grid, one finite node) | not_configured | output_power_required |

DPT completion is 4/4; speed is measured 3/4, with CPM2 contract-excluded
under VM-10. The DPT waveform gate is `retired_by_design`. Energy rows are
DPT-source self-consistency (`source_shared=true`,
`independent_energy_gate=false`), not an independent converter validation.
TI's 6.4730× is the stiff-golden case and is not portable.

---

## 3. What the verdicts mean

The C0 row is a **scoped diagnostic**, not a shipped acceptance. It shows
that the closed Tier-2 channel/charge emitter still reproduces static curves
at its recorded bars and that its device-current waveform error is acceptable
on TI / EPC / CPM2 under the declared 5 % Id requirement. It does **not**
show switching-edge, timing or energy fidelity: the independent G4 gate
rejects every vendor. The prior "accepted on all four" wording came from
scoring G4 against Tier-2 itself; that comparison is now explicitly
diagnostic-only, so the same records regrade to `reject` 4/4 without any
model change.

This distinction is the closeout's main result. Static and full-waveform RMSE
can look adequate while edge timing and energy remain wrong by large factors;
the independent golden gate is what makes that visible.

---

## 4. Executed gates

| Gate | Result |
|---|---|
| Full live suite, `uv run pytest` (356.55 s) | **719 passed, 4 skipped, 1 warning** |
| LTspice-free CI, `PIECESPICE_LIVE=0 uv run pytest -q --deselect tests/test_netlist_builder.py::test_tier2_simulates_and_matches` | **713 passed, 9 skipped, 1 deselected, 1 warning** |
| Live engine module | **19 passed** |
| GitHub CI run 44906010257 (commit 4535c38) | Both jobs succeeded |
| Record manifests | `artifact_manifest_verified.ok=true` on the final UROM records |

The LTspice-free count is labelled as such and is not a full-live count. The
full live run is the 719-passed baseline; the extra nine CI skips are the live
LTspice class selected off by `PIECESPICE_LIVE=0`, plus the one deselected
Tier-2 live match test.

---

## 5. Remaining limits (explicit)

1. **Timing and energy fidelity are not met.** G4 rejects all four canonical
   C0 rows against the independent golden. The record shows the scale: IRFS
   Tier-3 `tr` 1.184 ns vs golden 25.84 ns and E_on 7.938 µJ vs 45.43 µJ
   (Tier-2 12.93 µJ); TI E_off 0.151 µJ vs golden 0.953 µJ. Static and
   full-waveform agreement must not be read as switching-edge, `tr`/`tf`,
   `Eon`/`Eoff` or efficiency fidelity.
2. **CV is not measured.** Tier-3 `generalization.status="not_measured"` and
   `cv_status="not_measured"` because no extra `training_conditions` were
   configured. Multi-trajectory fitting is implemented but unexercised; no
   generalization claim is made.
3. **Thermal is not configured.** Tier-4 records report
   `thermal.status="not_configured"`; junction temperature is N/A rather than
   a fabricated ambient. A thermal network requires an explicit
   `config.thermal` block.
4. **Efficiency is N/A.** Tier-4 reports `output_power_required` on all four;
   `Vbus×Iload` is bus input power, not converter output power. No converter
   bench exists.
5. **Tier-4 waveform fidelity is not claimed.** The gate is retired by design;
   energy rows are source self-consistency, not independent validation.
6. **CPM2 speed is N/A.** The golden adaptive run aborts under the VM-10
   contract exclusion; no ratio is derived, including from the legacy
   completed-golden artifact.
7. **`simplis` has a legacy Cgd axis and an unmet C-V diagnostic on three
   vendors.** No bias-aware Cgd sweep exists in the current data; Cdg C-V error
   exceeds 10 % on IRFS / TI / CPM2 (EPC passes at 6.806 %).
8. **No global claim.** UROM speed is one declared method on one bench; TI's
   gain is golden-stiffness driven, EPC is slower on solver, and IRFS/CPM2 are
   near unity.

---

## 6. Evidence index

- Canonical C0 and ANN: `reports/comparison/<vendor>/unified_validation.json`,
  `unified_ann_trap-default.json`
- Tier-3 `v1`: `reports/comparison/<vendor>/tier3_validation.json`
- Tier-3 `simplis`: `reports/comparison/<vendor>/tier3_validation_simplis.json`
- Tier-4: `reports/comparison/<vendor>/tier4_validation.json`
- Tier-2 endpoint/speed context: `reports/comparison/<vendor>/endpoint_comparison.json`,
  `reports/comparison/speedup_benchmark.json`
- Policy: `config/acceptance.yaml`
