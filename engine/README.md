# PieceSpice custom solver engine

A piecewise-linear MNA engine for the ROMs this project emits. It is the
second half of the speed story, not a replacement for the modelling work.

## The reason it exists — and why both halves

The measured position of the current stack is documented in the repository
README: Tier-2 CPWL runs at **0.875–1.173×** the vendor golden at matched
2-pulse, forced 2 ns-step conditions, and Tier-3 `v1` at **0.0946–0.5827×**
(slower than golden on every device). Tier-4 — which is a *switch*, not a
waveform model — reaches 1.2212–6.4730×. So the ROMs have acceptable fidelity
and no speed, and the only fast tier is the one that gave up waveform
resolution.

The reason is not the model. It is where the model runs. LTspice is a general
Newton-based MNA/DAE engine: a behavioural channel or charge source is a
nonlinear element, so every step re-linearises its Jacobian and every step
pays iteration. Replacing a vendor subcircuit with a smaller nonlinear
subcircuit removes internal nodes but leaves the iteration in place — which
is exactly what 0.875× and 0.0946× measure.

Hence two halves, each useless alone:

| | without the other |
|---|---|
| **ROM** (device made piecewise linear) | still executed by a Newton loop → the measured ~1× |
| **Solver** (PWL/event-driven) | would have to linearise stiff vendor physics → a general-purpose SPICE clone |

The ROM's job is to make the circuit **linear between breakpoints**. This
engine's job is to **exploit** that: one LU per region, no Newton iteration,
and switching events located exactly rather than stepped over. The
combination is the only route to fidelity and speed at the same time.

There is a second payoff that is easy to miss. The canonical UROM records
reject **G4 timing/energy on 4/4** devices while passing static and charge
gates; the failure is edge resolution, not model shape. The ROM edges fall on
LTspice's 171.875 ns adaptive plateau. A solver that solves *onto* the
threshold crossing removes that discretisation error — a gate that model
tuning cannot reach (Phase V tuning was measured and closed) becomes
addressable by the engine.

**The claim this engine may make:** "no Newton iteration in the ROM fast
path".
**The claim it may not make:** "no stiffness anywhere". Stiffness is a
property of the circuit's time constants; a PWL engine changes how equations
are solved, not the physics.

## Supported envelope

Closed by design — see [`envelope.py`](envelope.py). Anything outside it
raises and the router sends the circuit to an external SPICE engine. The
fallback's speed is never credited as this engine's result.

| Element | Fast path | Note |
|---|---|---|
| `R`, `C`, `L` | yes | linear companion models |
| `V`, `I` | yes | `DC`, `PWL(...)`, `PULSE(...)`, evaluated at the step time |
| `X` | yes | flattened at parse time; one flat MNA system |
| `S` (`SW(Ron Roff Vt Vh)`) | planned | piecewise-linear by construction |
| `D` (`D(IS N RS)`) | planned | PWL region set, not Newton on `exp()` |
| `C.. Q={q(V)}` | planned | charge is a state variable |
| `G.. VALUE={..}` | planned | lowered from ROM tables, never evaluated as an expression |
| `.ac`, `.noise` | never | fallback only |

Analyses: `.op` and `.tran` in the fast path. `.dc` planned. AC/noise are
routed to the external engine.

## Layout

```text
engine/
  envelope.py        supported constructs, two failure modes, fail-closed
  ir/
    circuit.py       Circuit / Element / NodeIndex; data and structure only
    parse.py         restricted reader for the decks this project emits
    mna.py           structural MNA: branch unknowns and element stamps
  solver/
    companion.py     per-step companion models, linear elements first
    transient.py     stepping loop, dense LU, error control, event timeline
```

Two rules keep the boundary honest:

- **No SPICE expression evaluation.** A behavioural source is a construct to
  lower from ROM tables, never an expression to parse. Expression parsing is
  what turns a fast path into a simulator clone.
- **No dependency on the LTspice backend.** `engine` never imports
  `simulation.engine`. The benches in `simulation/testbenches` build decks
  and stay the single source of bench truth; they gain an `engine=` seam
  rather than moving here.

## Result contract

Solver output duck-types `simulation.engine.SimResult` — `traces`,
`sweep_var`, `log_text`, `warnings` — so benches, `metrics/*`, the plots and
the G0–G6 gate evaluation consume it unchanged. A parity test guards the
attribute set; there is no second result type to keep in sync.

## Verification strategy

The decisive measurement is a **two-engine, one-netlist** comparison:

```text
golden vendor .lib  --LTspice-->  A
ROM .lib            --LTspice-->  B        B − A = model error   (already recorded)
ROM .lib            --solver -->  C        C − B = solver error  (the new number)
```

Because the netlist is identical, model error cancels by construction and the
only variable is the engine. That isolates the new source of numerical error
instead of blending it into the fidelity numbers the project already
publishes. Stage-2 verification uses the linear subset of the same idea:
byte-identical decks through both engines, compared against closed-form
solutions as well.

## Non-goals

- Not a general-purpose SPICE engine; unsupported models go to the fallback.
- Not an AC/noise/`.four` tool.
- Not a claim of portable speedup: every number is per device and per bench,
  recorded with steps, µs/step and a completion guard.
