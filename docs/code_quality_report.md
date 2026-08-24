# PieceSpice Code Quality Report

**Scope:** all 45 `.py` files (~3,560 LOC) — parser, simulation, testbenches,
engine, approximation, metrics, models, config, CLI, reporting, scripts, tests.
**Audit only; no fixes applied.**

---

## Critical (blocks correctness or generalisation)

### C1. Vendor-specific script is the only output-generation path
- **File:** `scripts/generate_outputs_irfs7730pbf.py` (whole file)
- **Issue:** The canonical 7-curve pipeline exists *only* as a script hardcoded
  to `load_input("irfs7730pbf")`. For every other vendor (TI, EPC, CPM2) the
  same logic lives copy-pasted inside notebook 01 cells instead. Any bug fixed
  in one copy silently diverges in the others.
- **Fix:** Extract to `pipeline/run_curves(vendor: str)` module and have both
  the CLI and notebooks call it.

### C2. Testbench curve sign conventions differ per method
- **Files:** `simulation/testbenches/static_sweep.py`
  - L105 `id_arr = -np.array(analysis.Vds)` (output char)
  - L150 `id_arr = -np.array(analysis.Vds)` (transfer)
  - L198 `id_arr = np.abs(np.array(analysis.Vds))` (**abs, not negation**)
  - L264 `id_arr = -np.array(analysis.Vsweep)` (threshold)
  - L379 `isd_arr = -np.array(analysis.Vsd)` (body diode)
- **Issue:** rds_on uses `np.abs()` while all others use negation. If ngspice
  returns negative current for this topology, abs() masks a real sign error
  and produces positive-but-mirrored Rds curves. Copy-paste drift with
  correctness impact.
- **Fix:** Use one consistent convention (`-I(source)`) plus `_check_sign`,
  like the other four methods.

### C3. Hardcoded default Vgs sweep list assumes logic-level Si MOSFET
- **File:** `static_sweep.py`, L122 (`vgs_list = [3.0, 4.5, 6.0, 10.0]`)
- **Issue:** Default drives are IR-specific. A SiC part (Vth≈2.4–3 V, recommended
  drive 15–20 V) or GaN FET run without an explicit `vgs_list` gets curves at
  useless biases. Same class of problem: `vds_fixed=10.0`, `vgs_max=15.0`,
  `i_target=250e-6` (IR's 250 µA vs SiC's mA-class convention),
  `v_bus=40.0`, `i_load=50.0` defaults throughout.
- **Fix:** Make these required parameters or pull from `config.yaml_io.KEYS`;
  never bake datasheet conventions into testbench signatures.

### C4. Threshold i_target default contradicts project's own configs
- **Files:** `static_sweep.py` L262 (`i_target: float = 250e-6`) vs
  `config/inputs/cpm2-1200.yaml` (`Id_th_target: 5.0e-3`),
  `config/inputs/epc2216.yaml` (11 mA convention noted in comments).
- **Issue:** Three different threshold current conventions across vendors
  (250 µA / 1 mA / 5 mA / 11 mA). The bench signature default silently picks
  the IR one; extracted Vth values are not comparable between vendors unless
  every caller remembers to override.
- **Fix:** Move `Id_th_target` to a mandatory per-vendor YAML key (already done
  for cpm2/epc); remove the magic default.

### C5. `metrics/switching.py` returns fabricated constants
- **File:** `metrics/switching.py`, L24-27
- **Issue:** `rise_time_ns=15.2, fall_time_ns=8.4, delay_on_ns=12.1,
  delay_off_ns=24.5` are **hardcoded placeholder numbers returned as if
  measured**, while dv/dt and di/dt are genuinely computed. Downstream
  consumers cannot tell real metrics from stubs.
- **Fix:** Either implement the 10%–90% / 50% level-crossing extraction on
  v_ds/v_gs or raise NotImplementedError; never return invented data.

### C6. CLI `sweep` command injects fake report metrics
- **File:** `cli/main.py`, ~L60
  (`metrics={"static_rmse_pct": 0.8, "loss_error_pct": 3.2, "speedup_factor": 65.0}`)
- **Issue:** The generated HTML report always claims RMSE 0.8%, loss error
  3.2%, 65× speedup regardless of actual results. Misleading output presented
  as validation evidence.
- **Fix:** Compute via `metrics.static.compute_curve_error` against golden
  data, or omit the summary cards until M3 provides them.

### C7. pow_harden exists but is dead code — preprocessor never calls it
- **Files:** `parser/pow_harden.py` (whole module) vs `parser/preprocessor.py`
  (no import/call site — verified by grep).
- **Issue:** During the ngspice-39 investigation a fractional-power hardening
  pass was written and unit-tested, but the final rolled-back state has no
  wiring into Pass 3 of `normalize_pspice_netlist`. Anyone assuming it is
  active gets unhardened expressions on ngspice ≥ 36.
- **Fix:** Either integrate behind a flag (`harden_powers=True`) or delete the
  module until needed.

---

## Major (causes maintainability or portability issues)

### M1. Duplicated role→node mapping tables (three copies, one drifts)
- **Files:**
  - `static_sweep.py` L30 `ROLE_NODES = {"drain": "d", ...}`
  - `xinst.py` L12 `_ROLE_NODES = {...}` (identical)
  - `capacitance.py` L~110 `node_by_role = {...}` (recreated inline)
  - `gate_charge.py` L~270 inline dict again
- **Issue:** Four definitions of the same semantic map. A new role (e.g. kelvin
  source pin) must be added in four places; missing one silently misconnects.
- **Fix:** Single `ROLES` registry in `common/` imported everywhere;
  `xinst.x_connections` should be the only implementation.

### M2. Tj-port handling duplicated with three different idioms
- **Files:** `static_sweep.py` (inline `has_tj`/`core_roles` logic ×3 methods:
  transfer/rds use `attach_x_instance`, threshold/body-diode hand-roll
  `xargs + ["tj_M1"]`), `capacitance.py` (own `xtj` scan), `gate_charge.py`
  (**no Tj support at all**).
- **Issue:** gate_charge transient will silently float the Tj net on Wolfspeed
  models — exactly the bug class fixed in capacitance.py. Also three different
  tj-node naming schemes (`tj_M1`, `xtj`, `tj_{ref}`).
- **Fix:** Route every bench through `xinst.attach_x_instance`; delete the
  hand-rolled variants.

### M3. `run_gate_charge` transient is unreachable legacy code
- **File:** `gate_charge.py`, L~245-end
- **Issue:** Documented as structurally broken (VM-02), superseded by
  `reconstruct_gate_charge`, and called by nobody (verified: zero call sites).
  Still carries full transient netlist code, PWL source setup, and its own
  engine instantiation.
- **Fix:** Delete or move to an `attic/` module with a pointer note.

### M4. Dead placeholder code left in `engine._op_points_sweep`
- **File:** `simulation/engine.py`, L~118-124
- **Issue:** Contains a constructed-then-abandoned `probe = Circuit(...)`;
  `probe.raw_spice = ""`; and comment narration ("We cannot easily clone...")
  describing an approach that was never implemented. The method then does
  coarse-step/edge-shrink retries that duplicate `run_dc_sweep` fallback
  logic.
- **Fix:** Remove the probe block and misleading comments; keep only the
  retry ladder.

### M5. `normalized_lib_copy` writes into the vendor model tree
- **File:** `static_sweep.py` L32-43 — `out_dir = lib_path.parent / "normalized"`
- **Issue:** Runtime artifacts land next to vendor sources inside the repo tree
  (`SpiceModels/normalized/`). With `PIECESPICE_KEEP_NORM=1` they accumulate
  forever; `.gitignore` coverage is implicit and fragile.
- **Fix:** Direct artifacts to `reports/normalized/` or a tempdir; treat
  `SpiceModels/` as read-only input.

### M6. Per-point fresh normalized copy is O(N) file I/O with race-prone names
- **File:** `capacitance.py` (`temp_lib = normalized_lib_copy(lib_path)` inside
  the bias loop; comment says fresh path avoids shared-lib state pollution)
- **Issue:** 25 points → 25 mkstemp files + 25 normalize passes of the whole
  vendor lib per run; relies on PID+random suffix uniqueness. The underlying
  ngspice include-cache issue should be solved once (unique dir per run), not
  multiplied per point.
- **Fix:** Normalize once per run into a unique temp dir; reuse the single
  copy across points.

### M7. Magic numbers throughout benches
- **Examples (non-exhaustive):**
  - `capacitance.py`: `1e-15` probe floor, `1e-3` non-physical cap limit,
    `1e5..1e8 Hz` band check, `BIAS_INDUCTOR_H=1.0`, `GATE_BLEED_OHM=100e6`
    (last two are constants — good — but `F_MEAS_DEFAULT` band limits are
    inline literals)
  - `static_sweep.py`: `bleed_ohm = 10e6`, `coarse_step = vsd_max / 24`,
    shrink factors `(0.995, 0.99)`, retry factors `(4, 8)`
  - `engine.py`: identical retry factors duplicated
  - `netlist_builder.py`: `_META_DEFAULTS` r_gate 1.5Ω etc. (documented, but
    silent-fallback behaviour means wrong-device models build "successfully")
- **Fix:** Hoist to a `constants.py` or dataclass with names and units.

### M8. Sign-convention guard `_check_sign` is warn-only and heuristic
- **File:** `static_sweep.py` L~409-430
- **Issue:** Only checks the last fifth of samples trending negative; a model
  producing mirrored-but-positive curves passes silently. Given C2 (inconsistent
  negation), a flipped curve can ship as valid data.
- **Fix:** Assert expected polarity per-curve type (datasheet-defined) and fail
  loudly; make the check uniform across methods (currently absent entirely in
  rds_on).

### M9. Inconsistent return types across bench family
- **Issue:** `run_output_characteristics -> List[CurveData]`,
  `run_transfer_characteristics -> CurveData`,
  `run_rds_on -> CurveData` (with x/y axes *swapped* semantically: x=Current!),
  `run_threshold_voltage -> tuple[CurveData, float]`,
  `run_body_diode -> CurveData`, `CapacitanceTestbench.run -> list[CurveData]`.
- **Fix:** Standardise on returning `list[CurveData]` + a small typed summary
  object; document axis semantics in `CurveData.name`.

### M10. `rds_on` swaps axes without declaring it
- **File:** `static_sweep.py` L245-251 — `x_name="Drain Current (Id)"`,
  `y_name="On-Resistance"` (V/I derived, plotted against I).
- **Issue:** Every other curve is voltage-on-x. Plotting helpers, HTML report
  and interpolation callers assume monotonic-voltage x-axis; `np.interp(rx, ry,
  50)` in the outputs script works only by luck of ordering.
- **Fix:** Return Id-vs-Vds curve plus a separate scalar `rds_at(Id)` helper,
  or declare the inversion in metadata.

### M11. Notebook 01 duplicates the entire pipeline logic
- **File:** `notebooks/01_datasheet_curves_extraction.ipynb`
- **Issue:** Same seven testbench invocations as the vendor script (C1), with
  its own plotting code, its own save_output key choices, and vendor switching
  by editing a string. Three places to update per change.
- **Fix:** Thin notebook that imports the shared pipeline function (C1 fix).

### M12. Engine fallback prints to stdout instead of logging
- **Files:** `engine.py` (`[engine] .dc failed...`), `preprocessor.py`
  (`[preprocessor] WARNING:`), `capacitance.py`, `static_sweep.py`, `netlist_builder.py`
- **Issue:** Library code printing directly; callers can't suppress, filter, or
  route to the HTML report. Mixed print/logging styles.
- **Fix:** Module-level `logging.getLogger(__name__)`; keep prints only at CLI
  boundary.

### M13. `parse_spice_literal` suffix handling has redundant/dead branches
- **File:** `parser/spice_literal.py` L~40-48
- **Issue:** Loop re-checks `suf.lower() == key` after regex already matched
  case-insensitively; special-case `if suf == 'MEG' or (...)` duplicates the
  first check. Also 'mil' and 'M' (vs 'm') ambiguity unhandled — SPICE treats
  `M`/`m` both as milli and `MEG` as mega; table keys lowercase so `'T'=1e12`
  vs `'t'=1e12` fine, but a value like `10MIL` would parse `m`=milli then
  leave `IL` unmatched (returns None silently).
- **Fix:** Simplify to dict lookup `suffix_map[suf.lower()]`; add unit tests
  for MIL/MEG edge cases.

### M14. `_KV_RE` in two modules with different semantics, same name
- **Files:** `parser/preprocessor.py` L27 (`=` optional) vs
  `parser/netlist.py` L10 (`=` required)
- **Issue:** Same identifier, subtly different grammars. Refactoring one looks
  like refactoring both.
- **Fix:** Rename to reflect purpose (`PARAM_KV_RE`, `HEADER_PARAM_RE`) or
  share one module.

---

## Minor (style, clarity, dead code)

### m1. Leftover scratch/debug scripts in `scripts/`
- **Files:** `scripts/_check_nb01.py`, `scripts/_fix_order.py`,
  `scripts/_pow_draft.py` (dead first draft of pow_harden)
- **Issue:** Underscore-prefixed throwaways from debugging sessions; `_pow_draft`
  contains an older broken version of the shipped algorithm.
- **Fix:** Delete or move out of repo.

### m2. Unused `lookup()` closure in preprocessor
- **File:** `parser/preprocessor.py` L114-116
- **Issue:** Defined, documented, never called (params dict used directly).
- **Fix:** Remove.

### m3. `_is_int_literal` / `_ABS_WRAPPED_RE` partially dead in pow_harden
- **File:** `parser/pow_harden.py`
- **Issue:** After the final rewrite some earlier helpers/regexes
  (`_is_int_literal` referenced only in commented flows; `_ABS_WRAPPED_RE`
  unused by the shipped `harden_fractional_powers`) remain.
- **Fix:** Prune to what the active implementation uses; keep unit tests.

### m4. Backwards-compat alias nobody uses
- **File:** `static_sweep.py` L58 `_normalized_lib_file = normalized_lib_copy`
- **Issue:** Comment claims "older call sites" — grep shows none.
- **Fix:** Delete alias.

### m5. `from PySpice.Unit import *` wildcard imports
- **Files:** `static_sweep.py` L7, `capacitance.py` L8, `tests/*`
- **Issue:** Pollutes namespace (`u_V`, `u_A`, … appear "magic"); explicit
  imports used correctly elsewhere (`xinst.py`, `gate_charge.py` mixed style —
  local aliased imports inside function bodies, e.g. `u_V as _u_V`).
- **Fix:** Standardise on explicit `from PySpice.Unit import u_V, u_A, u_Ohm, u_Hz, u_F, u_s`.

### m6. Inline function-local imports scattered
- **File:** `gate_charge.py` (`from PySpice.Unit import u_V as _u_V...` inside
  `run_gate_charge`), `plot_gate_charge` (matplotlib inside), `capacitance.run`
  (`from pathlib import Path as _Path`).
- **Issue:** Imports hidden mid-function obscure dependencies; inconsistent
  with top-level import style.
- **Fix:** Move to module top (matplotlib lazy-import acceptable if startup
  cost matters — pick one policy).

### m7. Mixed typing styles: `Optional[List[str]]` vs `list[str] | None`
- **Files:** `static_sweep.py` (both styles in the same class),
  `capacitance.py` (PEP 604), `gate_charge.py` (PEP 604).
- **Fix:** Pick one (project requires Python ≥3.10 given `float | None`
  usage — standardise on PEP 604).

### m8. `os`/`tempfile` imports at top used only by two helpers
- **File:** `static_sweep.py` L3-4 — fine, but `Path` also imported and
  re-imported locally in capacitance.
- **Fix:** Consistency pass.

### m9. Curve name strings not standardized
- **Issue:** `"Output Characteristic"` vs `"Body Diode Forward Characteristic"`
  vs f-string composed names (`f"On-State Resistance (Vgs={vgs}V)"`).
  Downstream keying by name (reporting/report builders) is fragile.
- **Fix:** Enum/constant per curve kind; human label separate from stable id.

### m10. `family_param="Vgs"` on body-diode curve with `family_value=0.0`
- **File:** `static_sweep.py` body-diode return
- **Issue:** Gate-off is a condition, not a family parameter; inconsistent with
  threshold using `"Id_target"`.
- **Fix:** Use `metadata={"condition": "vgs=0"}` or a dedicated field.

### m11. `tests/test_netlist.py` depends on vendor file paths in repo root
- **Files:** `tests/test_netlist.py` (`SpiceModels/irfs7730pbf.spi`),
  `tests/test_vendor_ti.py` (`SpiceModels/CSD16412Q5A.lib`)
- **Issue:** Tests break if vendor files move/rename; also depend on
  cp1252-encoded vendor content loading with `errors='ignore'`.
- **Fix:** Small fixture vendor libs under `tests/data/` for parser tests;
  keep one integration smoke test against real files.

### m12. `conftest.py` present but near-empty utility usage
- **File:** `tests/conftest.py`
- **Issue:** Path bootstrapping done ad hoc (`sys.path.insert` in scripts);
  conftest could own it centrally.
- **Fix:** Move repo-root/path fixtures into conftest; drop per-script
  `sys.path.insert(0, ".")` (seen in `generate_outputs_irfs7730pbf.py`).

### m13. `cli/main.py` sweep command ignores most config surface
- **File:** `cli/main.py` `sweep(...)` — only `--vds-max`; no subckt pin map,
  no config file option, no output dir; report metrics fake (see C6).
- **Fix:** Accept `--vendor` and delegate to shared pipeline (C1); drop fake
  metrics.

### m14. Docstring/comment drift after refactors
- **Examples:** `engine.run_dc_sweep` docstring still narrates the removed
  stepped-OP design ("assembles an equivalent result object… Warns when the
  fallback is used" — the warning print remains but description references a
  removed mechanism); `static_sweep` header docstring unchanged since before
  Tj support; `capacitance` module docstring describes Sim A/B correctly but
  not NaN masking/trailing-trim behaviour now implemented.
- **Fix:** One docstring accuracy pass over touched modules.

### m15. Inconsistent curve-data trimming policy
- **Files:** `capacitance.py` trims trailing NaN region; static sweeps return
  whatever ngspice produced (can contain failed-tail zeros on partial
  convergence).
- **Fix:** Shared post-processing util (`sanitize_curve`) applied uniformly.

### m16. `models/netlist_builder.py` Tier-1/Tier-3 advertised but absent
- **File:** class docstring: "Emits SPICE subcircuit code for Tier 1, 2, or 3"
  — only `build_tier2_subcircuit` exists.
- **Fix:** Trim docstring to reality or stub the tiers explicitly.

### m17. `PWLModelSpec.charge_qoss_v/q` naming inconsistent with rest of spec
- **File:** `common/types.py` — `breakpoints_vds/breakpoints_id/vgs_values`
  vs `charge_qoss_v/charge_qoss_q` (units embedded mid-name, `qoss` casing).
- **Fix:** `charge_v`, `charge_q` or `q_oss_vs_v`, matching KEYS style.

### m18. `reporting/html_report.py` embeds large HTML template as module string
- **Issue:** Uneditable without touching Python; Jinja2 is already a dependency
  but template is raw f-string-style with `{{ }}` placeholders served without
  jinja2.Template actually being used consistently (imports jinja2 at top).
- **Fix:** Move to `templates/report.html.j2`, load via jinja2 Environment.

### m19. Duplicate `import numpy as np` placement inconsistencies & local re-imports
- **Files:** several (engine historically, capacitance trim helper uses np from
  module scope fine; gate_charge mixes pandas/numpy top-level with function-
  level PySpice imports).
- **Fix:** Style sweep.

### m20. `scripts/generate_outputs_irfs7730pbf.py` filename encodes vendor
- **Issue:** Will spawn `generate_outputs_csd16412q5a.py`,
  `generate_outputs_epc2216.py`… (the copy-paste factory of C1).
- **Fix:** Superseded by C1; delete after migration.

---

## Summary counts

| Severity | Count |
|---|---|
| Critical | 7 |
| Major | 14 |
| Minor | 20 |

**Top 3 remediation priorities:**
1. **C1/M11/M20** — unify the pipeline into one vendor-parameterised entry
   point (eliminates the largest duplication cluster).
2. **C2/C4/C5/C6** — correctness of reported results (sign conventions,
   threshold convention, fabricated metrics).
3. **M1/M2** — single role/Tj registry to stop multi-copy drift before more
   vendors arrive.
