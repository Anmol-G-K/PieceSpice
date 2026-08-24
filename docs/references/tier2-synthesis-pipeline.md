# Tier-2 Model Synthesis & Fitting Pipeline (M3 patterns)

Verified patterns from implementing `pipeline/fitting.py`,
`models/netlist_builder.py`, and the approximation modules. All validated on
real vendor data (TI CSD16412Q5A: worst endpoint error 0.53% vs <5% target).

## Channel source synthesis (tanh knee, plan §2.4)

`Id(vgs,vds) = Ipw(vgs) · tanh(Vds/vds_sat)` where:

- **Ipw(vgs)** = chained soft-abs ramps through every (vgs_k, Id_k) pair.
  Ramps MUST use delta slopes (`s_k − s_{k−1}`); absolute slopes accumulate
  exactly like the charge-source audit bug (proven 21× divergence).
- **Anchor placement**: the first ramp sits AT the first data point
  (vgs_0), not at vgs_1 — placing it at vgs_1 skips the first segment and
  returns id_0 for all vgs < vgs_1 (Id=0 at full drive).
- **vds_sat** derived from `Rds(on) × I_plateau(top curve)`, clamped to
  [5%, 25%] of the Vds sweep range so ohmic slope matches Curve 3.
- ngspice has NO `u()` step function in B-expressions — use the soft-abs
  form `(d + |d|)/2` everywhere.

## Charge-conserving capacitor emission

- Element must be a **C with `Q={expr}`**, never a B-source with Q= (invalid:
  B is a current source; the charge keyword belongs to nonlinear capacitors).
- Delta slopes as above; anchors at each previous table breakpoint.
- **Never round table charges to decimal places** — charges are ~1e-8..1e-6 C
  and `round(q, 9)` destroys up to 10% per entry. Format only at print time
  (%.6e keeps full precision).
- Non-ASCII characters (§, °, µ) in comment lines break ngspice parsing on
  cp1252 consoles — emit ASCII-only netlists.

## Verification recipe (catches semantic divergence)

Text-valid netlists can still be semantically wrong. Always:

1. Parse the emitted expression back and evaluate it in Python against the
   table at breakpoints AND the true integrated curve at segment midpoints
   (table error alone can look fine while emission diverges).
2. Instantiate via X-line and simulate — an include without an X-instance
   silently simulates an empty circuit returning zeros.
3. Compare against golden extracted values at fixed bias points.

## Vgs-key canonicalization (data-corruption class)

Family keys like `curve_id_vgs_{v}_v` written with `int(v)` TRUNCATE float
drives: TI's 4.5 V sweep stored as key `..._4_v`. Consequences: two drives
(4.0, 4.5) collide/overwrite; downstream fits label breakpoints at wrong
voltages while remaining internally consistent (golden-vs-model comparisons
using the same mislabeled keys PASS — the error is invisible until you check
the axis). Fix: `%g` formatter for writes, prefix-strip parser for reads,
and regenerate any outputs written with the truncated form.

## PWLModelSpec contract notes

Pydantic model_validator folding legacy field names works but adding new
fields requires `Optional` imported before class definition (pydantic raises
"class not fully defined" otherwise). Restoring accidentally-deleted core
containers (CurveData/WaveformData) from memory is easy to forget — grep for
their importers after rewriting shared type modules.

## Notebook sectioned rebuild pattern

For analysis notebooks: one markdown header + one code cell per curve
section, each code cell loading its saved output YAML independently (so
sections run standalone), plus a single setup cell doing sys.path/os.chdir
bootstrap. Execute with nbconvert fresh kernel; stale running kernels keep
pre-fix modules.
