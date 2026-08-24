# Piecewise fitting & curve-compression algorithms (PieceSpice M3)

Field notes from implementing/auditing the M3 approximation layer: adaptive
PWL regression, RDP compression, and how they map onto MOSFET macromodel
sub-blocks. Companion to `piecewise-extraction.md` (M2 extraction side).

## Tool-to-curve mapping (why different curves get different algorithms)

The final artifact is a MOSFET subcircuit, not a compressed curve — each
datasheet curve feeds a different sub-block, so each geometry gets its own
tool:

| Curve | Geometry | Algorithm | Feeds |
|---|---|---|---|
| Transfer (Id-Vgs) | piecewise regions with knee | adaptive PWL regression | channel source: Vth, gfs |
| Output (Id-Vds) | piecewise + saturation fold | PWL regression per Vgs family | channel source tanh-knee params |
| Body diode (Isd-Vsd) | smooth exponential | RDP pruning / param extraction (SPICE has native exponential D element) | diode IS/N |
| Capacitances (Ciss/Coss/Crss) | nonlinear C(V) | charge integration Q(V)=∫C dV (NOT curve compression) | charge-conserving sources |

RDP on an exponential doesn't "fit" anything — it just selects which samples
survive; SPICE's native diode model consumes the characteristic points.

## Adaptive PWL regression pitfalls (found by post-implementation audit)

1. **Stagnation vs self-comparison in incremental-K loops.** The natural
   early-exit `if result.rmse > 0.999 * best.rmse: break` is WRONG when
   `best` is updated from `result`: after every improvement result IS best,
   so the condition is trivially true and the loop exits right after the
   first improvement. Track `prev_rmse` separately; exit only when the
   current K fails to improve on the PREVIOUS K (stagnation). Impact seen:
   7x worse RMSE silently returned as "best".
2. **Sample-starved segments break C0 continuity.** When a breakpoint
   placement gives some segment <2 data points, do NOT fabricate a line
   through the lone sample with the neighbor's slope — that creates a jump
   discontinuity at the shared breakpoint (proven 45-unit jump). Repair by
   MERGING the starved segment into a neighbor (drop the shared breakpoint)
   and refitting; for trailing isolated samples prefer a continuous
   connector (slope derived from continuity-at-break + passing exactly
   through the point).
3. **Independent per-segment polyfits leave ~1e-8 jumps** at breakpoints
   because intercepts are fitted separately. Chain segments instead: keep
   slopes, derive each intercept so segment s starts exactly where s-1
   ended. Required before emitting PWL into SPICE — discontinuous sources
   cause Newton oscillation.
4. **Sanitize before fitting**: mask NaN/inf, sort by x, average duplicate-x
   samples (SPICE sweeps produce repeated x at clamps). pwlf's global search
   is unstable on unsorted/duplicated input.
5. **K=1 degenerate case**: plain polyfit, skip the library's search.

## Stop-semantics design lesson

A boolean `converged=False` is ambiguous — callers read it as "the fit is
bad" when it means "your fidelity target was too strict for max_segments."
Prefer an explicit `stop_reason` enum ("tolerance_met" | "stagnated" |
"cap_reached") plus always-populated rmse/max_abs_error, documented as the
actual quality measures.

## RDP specifics

- Use an explicit-stack iterative core; recursive RDP hits Python's recursion
  limit around 1000 points, far below real sweep sizes (10^5).
- Guard epsilon <= 0 / NaN with ValueError: negative tolerance silently
  disables all pruning.
- `relative=True` (epsilon = fraction of bounding-box diagonal) transfers
  between unit scales ONLY under uniform rescaling of both axes. RDP is not
  affine-invariant: scaling x alone changes the perpendicular-distance
  geometry and hence compression.
- Binary-search helper (`epsilon_for_max_error`) converts a fidelity budget
  ("max 10 mA deviation") directly into a tolerance — better than guessing.
- Guarantees to maintain: endpoints retained, x strictly ascending, NaN rows
  dropped pre-pass.

## Validation pattern

Fit known ground-truth piecewise signals (assert breakpoints within
tolerance of true knees, rmse ~ machine eps), plus a random-placement sweep
(50 trials) asserting worst breakpoint jump < 1e-9 — this is what caught
both bugs above after initial implementation "passed" basic tests.
