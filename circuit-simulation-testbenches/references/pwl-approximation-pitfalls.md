# PWL Approximation Algorithms — Pitfalls & Verified Patterns

Lessons from implementing `approximation/pwlf_fit.py` (adaptive PWL regression),
`approximation/rdp_compress.py`, `approximation/charge_model.py`, and
`approximation/smoother.py`. All patterns verified against real extracted
vendor curves (TI CSD16412Q5A, Wolfspeed CPM2-1200).

## Adaptive PWL fitting (pwlf wrapper)

### Bug pattern: self-comparison early-exit
```python
if result.rmse < best.rmse: best = result
...
if result.rmse > 0.999 * best.rmse: break   # WRONG
```
After an improvement, `result` IS `best`, so the check is trivially true and
the loop exits right after its first improvement. Proven impact: 12× worse fit
silently returned (K=3/RMSE=1.68 when K=10/RMSE=0.137 existed).

**Fix**: track `prev_rmse` separately; exit only on STAGNATION
(`result.rmse > 0.999 * prev_rmse`). Always add a regression test that forces
deep exploration (smooth sine signal keeps improving to K~6; a buggy exit
stops at K=3).

**Expose WHY the loop stopped**: a bare `converged: bool` conflates "fidelity
target met" with "segment budget exhausted" and reads like the fit failed.
Return `stop_reason: 'tolerance_met' | 'stagnated' | 'cap_reached'` instead,
and document that cap_reached ≠ invalid — RMSE always quantifies quality.

### Starved segments break C0 continuity
Segments with <2 samples must never be filled with "reuse neighbor slope +
anchor through the lone point" — that creates parallel lines with a jump at
the shared breakpoint (proven 45-unit discontinuity). A discontinuous model
emitted as a SPICE PWL source causes Newton oscillation.

**Fix sequence**:
1. Trailing isolated sample → continuous connector: slope derived from
   predecessor endpoint value through the sample.
2. Interior starved segment → merge into neighbor (drop shared breakpoint).
3. Final pass: chain intercepts so every segment starts exactly where the
   previous ended (exact C0 by construction).

Verify with line-to-line limits at breaks (`slope[i]*b + ic[i]` vs next), NOT
finite differences of evaluate() — a steep continuous line legitimately
changes by slope·dx across any epsilon.

### Input sanitization contract
NaN/inf mask → sort → average duplicate x values (np.add.at trick) before
pwlf; duplicate x otherwise destabilizes its solver.

## RDP compression

- **Iterative explicit-stack core**, not recursion: dense sweeps (100k+
  samples) blow Python's recursion limit.
- **Relative epsilon mode**: interpret epsilon as fraction of bounding-box
  diagonal → one tolerance transfers across unit scales. NOTE: only valid for
  UNIFORM rescaling of both axes; anisotropic scaling changes shape in (x,y)
  space and RDP is not affine-invariant.
- **Epsilon guard is required**: negative/NaN epsilon silently disables all
  pruning (every interior distance > eps). Raise ValueError.
- **Budget search**: binary-search epsilon so the RECONSTRUCTED curve
  (np.interp back onto original x) stays within max_error — lets users state
  fidelity budgets directly instead of guessing tolerances.

## Charge-conserving capacitance modeling

Why charge form: `i = C(v)·dv/dt` behavioral sources drift numerically
(discretized derivative); charge sources integrate exactly.

- Q(V) = ∫C dv, E(V) = ∫v·C(v) dv (cumulative_trapezoid).
- Two DIFFERENT equivalent linear capacitances (plan §2.3):
  `C_eq(charge) = Q(Vref)/Vref`, `C_eq(energy) = 2·E(Vref)/Vref²`.
  For collapsing MOSFET Coss they diverge (charge-eq > energy-eq); choice
  depends on whether peak voltage or ringing energy must be preserved.
- Reject negative capacitance samples — physically impossible, means upstream
  extraction corrupted.
- Reject v_ref <= 0 explicitly (C_eq(charge)=Q/Vref is 0/0-undefined there);
  returning 0 via a divide floor hides the misuse.

### CRITICAL: soft-abs ramp chains accumulate slopes
Emitting a monotonic PWL charge as chained softplus ramps
`sum s_k * softplus_ramp(x - v_k)` where `s_k` are ABSOLUTE segment slopes
diverges linearly — each ramp keeps contributing beyond its own interval
(softplus never turns off). Proven: emitted charge 21× too high at Vbus.

**Fix**: each term must use the DELTA slope between consecutive segments:
```
Q(x) = q0 + Σ (s_k − s_{k−1}) · ramp(x − v_{k−1})
```
Always verify emitted expressions against the table at breakpoints AND true
curve at segment midpoints — table error alone can look fine while emission
diverges.

PWL table construction itself (greedy farthest-point insertion on the charge
curve until within absolute-error budget) worked correctly first try.

**Never decimal-round emitted charge/slope coefficients.** Charges are
~1e-9..1e-6 C; `round(q, 9)` destroys up to ~10% per entry and the emitted
expression then deviates from its own table even with correct delta slopes.
Format at print time (%.6e keeps full precision); never round stored values.

### Element-syntax gotcha (ngspice)
Charge-form capacitors are **C elements with `Q={expr}`**, NOT B-sources —
`Bxxx n+ n- Q={...}` is invalid and fails circuit load. Also keep generated
netlists ASCII-only: non-ASCII chars in comment lines (e.g. '§') break
ngspice parsing on cp1252 consoles.

## Smoothing kernels (C1 knees)

- softplus via log-sum-exp: `max(z,0)*eps + eps*log1p(exp(-|z|))` — no overflow.
- tanh saturation knee: odd function gives third-quadrant conduction free.
- Smoothstep-rounded breakpoint tables: compare corner rounding quality via
  numeric slope NEAR the break vs hard-table slope, not value continuity
  (both lines agree AT the break by definition).
- Test tolerances must respect kernel math: softplus(0)=eps·ln 2 exactly;
  far tails underflow to identical values (test non-decreasing globally,
  strictly increasing only inside ±3ε knee window).

## Notebook demos of algorithms — no fabricated parameters

When a notebook demonstrates a kernel/model against real data, DERIVE the
demo parameters from the actual data or prior algorithm outputs. A demo that
invents gm/vds_sat values ("gm_est*3") diverges arbitrarily from the plotted
reference curve (proven 7× divergence) and destroys trust in the figure.
Correct pattern: fit parameters FROM the reference (e.g. Id_sat = plateau
value; Vds_sat = where the reference crosses tanh(1)=76% of Id_sat), then the
smoothed overlay coincides away from the knee by construction and deviation
is provably knee-localized.

Also plot BOTH the full original curve AND the compressed/fitted one in any
compression demo — drawing only retained points makes it impossible for the
reader to judge fidelity.
