# M2 Bench Audit — Physics Errors Found by Line-by-Line Review (Aug 2026)

Post-migration audit of the rewritten DC benches caught two CRITICAL
physics errors that the automated parity check had MISSED. This file
records them as reusable lessons: the parity harness compared
self-consistent-but-wrong curves and reported clean passes on some of
them.

## P1 — Curve 1 output characteristics: axis inversion (CRITICAL)

**Buggy pattern:** placing the swept source between an intermediate bus
and the device drain, with `Vbus = Vgs + Vds_max`, then sweeping that
source 0→Vds_max.

**What actually happens:** sweeping the source *lowers* the real device
Vds from `(Vgs+Vmax)` down to `Vgs`. The bench records x = source setting
(0…10) while the device actually operates at Vds = 20…10 V. Verified
live: raw showed `V(drain)` going 20→10 V during the "0→10" sweep.

**Why parity missed it:** both old and new curves were smooth and
self-consistent; matching x-values compared different operating points.
The comparison was blind to the axis inversion. Only a line-by-line
review of raw node voltages exposed it (`V(drain)` decreasing during an
"increasing" sweep is the smoking gun).

**Correct pattern:** sweep a voltage source tied directly to the device
drain (drain → 0), exactly as the original PySpice bench did. No vbus
intermediate for a simple ground-referenced sweep. Reserve intermediate
buses for cases needing a floating or offset supply, and then compute
the x-axis from actual node voltages in the raw output, never from the
source's control value.

## P2 — Curve 3 Rds(on): same inverted traversal (HIGH)

Same vbus/swept-source pattern: device Vds traverses Vmax→0 while Id
increases. The scalar at Id_load can still match the old baseline by
luck when the load point lands in the ohmic tail (IRFS matched at 0.6%)
while other vendors diverge massively (23–75%) — the divergence was
real but its dominant cause was the traversal, mis-attributed to solver
physics in the first-pass analysis.

Lesson: when a scalar matches on one vendor but not others, before
blaming solver physics, check whether the sweep *direction* differs
between implementations — ohmic-region quantities are direction-
sensitive because Rds is only defined in a narrow Vds window.

## L1 — abs() disarms the polarity guard (MODERATE)

Applying `np.abs()` to extracted currents before `_check_sign` means the
sign guard can never fail: everything is positive by construction. If
you need polarity safety, validate the SIGNED trace and take abs() only
afterwards (or better: assert the known sign explicitly rather than
laundering through abs).

## L2 — no-op `.replace()` in retry paths (LOW)

`text.replace("same", "same")` in a coarse-retry branch is an unfinished
edit marker. Harmless at runtime but a review smell — the retry worked
only via directive mutation.

## L3–L6 minor items

- Fragile string-hack rendering an AC card (placeholder + replace).
- Gate-pulse offset makes the "steady-state slice" not a clean full cycle.
- Node-name resolution by dict lookup silently measures wrong nodes if a
  vendor uses nonstandard semantic roles.
- Directive-insertion guarded by substring match (".dc" in text) which
  comments can false-positive.

## Meta-lesson: what the audit method actually was

The catches came from **printing raw node voltages from the .raw file**
during a supposedly-known sweep and checking they moved in the expected
direction. Trace-level verification (does V(drain) rise when I sweep up?)
catches axis/direction bugs that curve-shape comparison never will.
For any new testbench, run this one-node-voltage sanity probe before
trusting extracted curves.
