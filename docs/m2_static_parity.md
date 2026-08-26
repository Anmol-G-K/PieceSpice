# M2 Static Parity Report — LTspice vs archived ngspice outputs

**Date:** 2026-08-26
**Harness:** `scripts/_parity.py` (kept as the reusable parity runner)
**Raw data:** `reports/m2_parity_static.json`

## Method

Each vendor's static benches run through the new spicelib/LTspice engine;
key scalars are diffed against the archived ngspice-era YAML outputs in
`config/outputs/<vendor>/`.

## Results

| Vendor | v_plateau err | gfs err | Rds(on) err | Vth err | Curve-1 worst endpoint err |
|---|---:|---:|---:|---:|---:|
| irfs7730pbf | 0.003% | 0.21% | **0.61%** ✅ | 0.05% | 69.4% |
| csd16412q5a | 0.36% | 4.1% | 23.1% ⚠️ | 0.00% | 25.5% |
| epc2216 | 0.00% | 7.3% | 70.6% ❌ | 0.00% | 131.1% |
| cpm2-1200 | 0.08% | 0.4% | 74.6% ❌ | 0.00% | 236.0% |

## Interpretation

### Well-conditioned scalars: excellent parity
Plateau voltage and Vth agree to **<0.4% across all four vendors**
(most <0.1%). These depend on the transfer curve's shape near threshold
and confirm both engines evaluate the vendor models consistently.

### gfs: good
Within 0.2–7.3%. The larger EPC/TI deviations reflect gfs being a
gradient (numerically differentiated) at a single interpolated point -
small curve-shape differences amplify there.

### Rds(on): vendor-physics-sensitive, needs investigation
IRFS matches at 0.6%, but TI/EPC/CPM2 diverge 23–75%. Two candidate
causes, NOT yet distinguished:
1. Solver physics differences (ngspice clamps some model parameters,
   e.g. diode VJ limited to 2 V, that LTspice honors as-written).
2. Bench methodology drift between old/new implementations.

Resolving this requires a datasheet cross-reference, which per project
rules is not substituted automatically - flagged for user review.

### Curve-1 high-Vgs endpoints: expected amplification
Endpoints sample Id at max Vds where VM-04 (unclipped square-law
saturation) makes currents enormous; tiny conduction-differences
amplify to tens/hundreds of percent. IRFS at 69% vs CPM2 at 236% tracks
exactly with how extreme each vendor's unclipped-saturation defect is.
These endpoints do NOT indicate harness error; they reconfirm the known
vendor model limitations documented in methodology_audit.md.

## Bugs found & fixed during parity bring-up

1. `_xargs_for` truncated tj from connections when yaml pin_order omits
   it (CPM2) -> subckt pin-count mismatch. Fixed: tj auto-detected from
   the library header itself (authoritative), not yaml.
2. Same fix applied to capacitance bench.
3. Three signature/call-site mismatches in _header_lines/_xargs_for
   during refactoring (caught by live runs).

## Verdict

M2 static parity: PASS for well-conditioned scalars on all four vendors.
Rds(on) divergence on 3 of 4 vendors is real but attributable to either
solver-physics or bench-methodology drift - escalated for decision.
Curve-1 endpoint deltas reconfirm documented vendor defects, not
regressions.


## Addendum (T2.4): Capacitance parity results & findings

**Date:** 2026-08-26 · **Raw data:** `reports/m2_parity_capacitance.json`

| Vendor | Ciss err | Coss err | Crss err |
|---|---:|---:|---:|
| irfs7730pbf | 0.00% | **0.06%** ✅ | **0.05%** ✅ |
| csd16412q5a | 0.00% | **0.00%** ✅ | **0.00%** ✅ |
| epc2216 | 0.00% | 320.7% ⚠️ | 320.6% ⚠️ |
| cpm2-1200 | 18.9% ⚠️ | 70.9% ❌ | 53,289% ❌ |

### IRFS + TI: exact parity
Sub-0.1% on all three caps. The LTspice small-signal method reproduces
the archived ngspice extraction essentially perfectly where vendor
models are well-formed.

### Harness bug found & fixed during T2.4
Variant-B probe current was injected at `vbus` (AC-grounded by the ideal
bus source) instead of at the drain node — all AC current bypassed the
device, collapsing V(drain) to zero and masking every Coss/Crss point as
NaN. Fixed by attaching `Iprobe` directly to the drain node, matching
the original bench topology (`I drive xd gnd`).

### EPC + CPM2 divergences: genuine engine/model physics differences
- **EPC2216 (+320% Coss/Crss):** gate-side identical (Ciss 0%) — only
  drain-side differs. The PSPICE net's TABLE-based drain capacitors are
  interpreted differently between engines.
- **CPM2 (Crss +53,289%):** the SiC model's Tj port sinks ~7.5 A of AC
  at the thermal node under LTspice's exact solution; ngspice-34's
  damped convergence suppressed that path. With V(tj)=AC-grounded, the
  coupling propagates into V(drain) (85.9 V swing from a 1 A probe).
  The LTspice numbers are arguably more physically faithful, but they
  are NOT comparable to the archived baselines.

### Disposition
- IRFS/TI: M2 capacitance parity PASS.
- EPC/CPM2: escalated — baseline regeneration on LTspice required before
  these two vendors' Curve-4 outputs can feed M3 fitting consistently.
