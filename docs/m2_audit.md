# M2 Audit — Logical & Physics Errors in the LTspice Migration

**Date:** 2026-08-26
**Scope:** every file rewritten during M1/M2 (`simulation/engine.py`,
`netlist_text.py`, all testbenches, `pipeline/fitting.py`).
**Mode:** read-only audit. No fixes applied.

---

## CRITICAL — Physics Errors

### P1. Curve 1 output characteristics: wrong x-axis mapping (all vendors)
`run_output_characteristics` places the swept source `Vds` **between
vbus and drain**, with `Vbus = Vgs + Vds_max`. Sweeping the source
0→10 therefore drives the *actual device* Vds from `(Vgs+10) − 0 = 20 V`
down to `20 − 10 = 10 V`.

Verified live (CSD16412Q5A, Vgs=10): raw shows `V(drain)` going
**20 V → 10 V** while the bench records x as 0…10 and Id as
417 A → 305 A.

Consequences:
- The emitted "Id vs Vds" curve is actually Id over **[Vgs+Vmax, Vgs]**
  plotted against [0, Vmax] — an entirely wrong axis.
- Every downstream consumer of Curve 1 (endpoint currents, the fitting
  pipeline's breakpoint rows, Rds interpolation) ingests voltages that
  never existed at the device.
- The huge Curve-1 endpoint deltas in the parity report (25–236 %) are
  partly this bug, not only VM-04 physics: ngspice-era outputs came from
  a bench that swept the drain directly.
- The parity harness compared new-vs-old at matching x values, so this
  error did NOT show as a DIFF flag — both curves were "smooth"; the
  comparison was blind to the axis inversion.

**Required fix direction (not applied):** sweep a source tied directly
to drain (drain → 0), as the original PySpice bench did; drop the vbus
intermediate entirely.

### P2. Curve 3 Rds(on): same vbus/vbus_node pattern, inverted sweep
`run_rds_on` holds `Vdrive = Vds_max` at vbus and sweeps the `Vds`
element 0→2.5 V, so the device Vds runs 2.5 V → 0 V — decreasing while
x (Id) increases. The scalar at Id_load happened to match IRFS (0.6 %)
because the load point lands in the ohmic tail where both engines agree,
but TI/EPC/CPM2 divergences (23–75 %) are likely dominated by this
inverted traversal combined with VM-04, not by solver physics alone.
The curve-RMSE failures (56–238 %) in the parity report are consistent
with the same axis/direction problem.

**Same fix direction:** drive drain directly.

---

## MODERATE — Logic Issues

### L1. `_finish` uses `abs()` on currents — sign information destroyed
The old benches negated source currents with an explicit convention and
then validated polarity via `_check_sign`. The new `_finish` applies
`np.abs`, which silently converts a wrongly-oriented probe into plausible
data. `_check_sign` still runs but can now never fail on sign flips,
because everything is positive by construction. The polarity guard is
effectively disarmed.

### L2. C6 retry path contains a no-op `.replace()`
The coarse-step retry calls `make_text().replace("* body diode",
"* body diode")` — replacing a string with itself. Harmless at runtime
but evidence of an unfinished edit; the retry also re-runs with the SAME
netlist text, relying solely on `directive.step` mutation (which does
work). Should be cleaned when edits resume.

### L3. Engine `.ac` card rendering has a fragile string hack
`render_directive(ACDirective)` builds the card then applies
`.replace(" _ ", " ")` to patch a placeholder — works today but breaks
silently if formatting changes. Minor robustness issue.

### L4. `simple_switching.py` gate pulse timing differs from spec intent
Pulse starts at td=5 µs with period 20 µs — first cycle is offset; the
"3 cycles" window therefore contains turn-on at 5, 25, 45 µs and the
steady_state_slice takes 40–60 µs (turn-off + partial cycle). Matches
previous behavior so fitness numbers stay comparable, but worth noting
the last slice isn't a clean full cycle.

### L5. Capacitance variant-A gate node hardcodes `conn['gate']`
`_complex_voltage(res_a, sub_gate(pin_roles))` resolves the gate node by
name — correct for 3-pin devices, but if a vendor declared pins in a
nonstandard semantic order not covered by the conn dict, the probe would
silently measure the wrong node. Low risk given current yaml coverage.

### L6. `run_dc_sweep` appends directive only if absent — string match
`if ".dc" not in netlist_text.lower()` — a netlist containing ".dc" inside
a comment would suppress directive insertion. Edge case; unlikely but
real.

---

## CONFIRMED CORRECT (audited, no issues)

- **C2 transfer**: direct drain bias + Vgs sweep; matches old topology.
- **C7 Vth**: drain-gate tie via positional 'gd' nodes; JEDEC interp;
  ascending-current requirement satisfied.
- **C6 body diode**: src/drn orientation, 10 MΩ anchor + analytic
  parasitic subtraction, coarse-step retry logic — all correct (modulo
  L2 cosmetic).
- **Engine execution path**: temp workdir, timeout, RawRead parsing,
  log classification — verified against live runs.
- **NetlistBuilder**: role-by-name binding, Tj source emission,
  analysis-card insertion order.
- **Capacitance variant A/B topologies** (after the Iprobe fix): match
  the documented small-signal method exactly.

---

## Impact Assessment

| Finding | Severity | Blocks |
|---|---|---|
| P1 C1 axis inversion | **CRITICAL** | All Curve-1 consumers: fitting pipeline, endpoint metrics, Tier-2 channel breakpoints |
| P2 C3 inverted traversal | HIGH | Rds(on) curve + scalar; explains most parity DIFFs on TI/EPC/CPM2 |
| L1 abs() disarm | MODERATE | Polarity safety net |
| L2–L6 | LOW | Hygiene/robustness |

## Note on the parity report

The earlier claim "not a single DIFF traces to a harness bug" is hereby
**corrected**: P1/P2 are harness bugs that the parity method could not
see (both sides were self-consistent within their own wrong axes for
different reasons). The scalar passes (plateau/Vth/C4-gate-side) remain
valid — those benches are wired correctly.
