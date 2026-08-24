# Tier-2 subcircuit synthesis & fitting pipeline (M3.5/M3.6, PieceSpice Aug 2026)

Verified patterns from building `models/netlist_builder.py` (Tier-2 CPWL
MOSFET synthesis) and `pipeline/fitting.py` (`fit_pwl_model(vendor)` —
outputs YAML → PWLModelSpec → .lib). Device-agnostic by design: every
parameter comes from fitted data or config; nothing tuned to one vendor part.

## Spec contract (PWLModelSpec) gotchas

- **Pydantic model rewrite trap**: when rewriting a types module, the old
  classes (CurveData/WaveformData/SimulationResult) get dropped and every
  downstream import dies at collection time with a cascade of ImportErrors.
  Always check what the rest of the codebase imports from the module before
  overwriting, and restore containers in the same change.
- Pydantic "class not fully defined" error = missing typing import
  (`Optional`) referenced inside the class body.
- Validate structural consistency in a `model_validator(mode="after")`:
  breakpoints_id row count == len(vgs_values), every row length ==
  len(breakpoints_vds), breakpoints strictly ascending.

## Emitted-netlist bugs found by acceptance testing

1. **Charge-form capacitors must be C elements, not B sources.** ngspice's
   `Q=` keyword is valid on `Cxxx n+ n- Q={expr}` only. Emitting
   `Bxxx n+ n- Q={...}` is an invalid element mix and fails to parse.
2. **Non-ASCII characters (§) in comment lines break parsing** on cp1252
   consoles. Keep emitted netlists ASCII-only; strip section symbols from
   generated comments.
3. **Chained-ramp PWL must anchor the first ramp at the FIRST DATA POINT**
   with the first segment slope, then delta slopes at subsequent breaks.
   Skipping the first anchor (starting deltas at break 2) makes the model
   return id_pts[0] for everything below the last breakpoint — 0 A instead
   of rated current.
4. **Delta-slope rule applies to ALL chained-ramp emissions** — channel
   transfer AND charge tables alike (see pwl-approximation-pitfalls.md).

## Acceptance-test pattern (Tier-2 vs golden)

- Instantiate via `c.X("M1", name, "d", "g", c.gnd)` — forgetting the X
  line while including the lib gives a silently-passing OP with all-zero
  currents (nothing is simulated).
- Compare against golden values interpolated from the SAME extracted-curve
  YAML keys the fitting consumed — but first verify those keys' Vgs labels
  are correct (see key-formatting bug below). Internally-consistent
  comparisons on mislabeled axes look like PASS while both sides are wrong.
- Assert `pytest.approx(want, rel=0.02)` per curve endpoint.

## Key-formatting collision bug (data corruption class)

Writing family keys as `f"curve_id_vgs_{int(v)}_v"` truncates 4.5 → 4: two
different drives collide or a curve gets stored under the wrong label, and
the fitting stage then builds the model on a shifted Vgs axis. Acceptance
tests comparing fit-vs-golden still PASS because both sides read the same
mislabeled data.

**Fix**: canonical formatter using `%g` (`curve_id_vgs_{v:g}_v` → 4 / 4.5
distinct), applied at write time in the pipeline; parsers strip prefix +
suffix; output validators accept float keys via regex
`^curve_id_vgs_[\d.]+_v$`. After changing key formats, REGENERATE affected
output files — stale files keep the old labels.

**Lesson**: acceptance tests must validate axis LABELS, not just values.
Check that stored metadata/keys match the config sweep list before trusting
a fit comparison.

## Line-length and expression size

Behavioral expressions grow O(breakpoints); 101 Vds breakpoints produced a
~3000-char C_QOSS line — within conservative parser limits but worth
watching. If it grows further, emit continuation lines (`+`).

## Workflow rules (user-mandated, this project)

- NEVER delete a directory without explicit user confirmation — even empty
  ones. Deleting unneeded FILES inside scripts/ is fine; the directory stays.
- Batch fixes (~3 at a time), pytest after each batch, short numbered summary
  after each batch.
- When asked "what's the status", report immediately from what's known — no
  new probe runs.
