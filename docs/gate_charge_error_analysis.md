# Gate-Charge Testbench Error Analysis (Curve 5)

**Status:** Curves 1, 2, 3, 4, 6 — working. **Curve 5 (Gate Charge) — blocked by vendor-model pathology.**
**Device under test:** `SpiceModels/irfs7730pbf.spi` → subcircuit `irfb_s_sl7730pbf` (pins: 1=Drain, 2=Gate, 3=Source)
**Failing module:** `simulation/testbenches/gate_charge.py`
**Date:** 2026-08-21

---

## 1. What the study is

Datasheet Curve 5: gate-source voltage `Vgs` vs total gate charge `Qg = ∫ Ig dt`, measured under
constant-current gate drive. Datasheet conditions for IRFS7730PBF:

| Condition | Value |
|---|---|
| Gate drive | Constant current `Ig = 5 mA` |
| Bus voltage | `Vbus = 40 V` |
| Load current | `Id = 50 A` held during measurement |
| End of sweep | `Vgs = 10 V` |

Expected waveform: `Vgs(t)` rises monotonically (slope `Ig/Ciss`), flattens at the Miller plateau
while the drain collapses from 40 V to near 0 V, then rises again to 10 V.
Expected total charge: roughly 150–200 nC.

---

## 2. Symptom

Under constant-current gate drive the simulation produces physically impossible waveforms:

```
Q =   27.9 nC -> Vgs =  4.72 V   Vds = 26.28 V     <- device starts turning on
Q =  118.2 nC -> Vgs = -1.31 V   Vds = 40.69 V     <- device turned OFF again
Q =  238.7 nC -> Vgs =-10.20 V   Vds = 40.69 V
Q =  600.0 nC -> Vgs =-37.00 V   Vds = 40.69 V     <- final: gate at -37 V
```

- `Vgs` rises only to ~4.7–4.9 V, then **collides downward and keeps falling far below zero**
  even though the 5 mA source pushes current in continuously.
- The device enhances briefly (`Vds` falls), de-enhances (`Vds` returns to the rail clamp),
  and the gate ends up tens of volts negative (-14 V to -37 V depending on variant).
- Charge extraction is impossible: `Qg @ 10 V` never exists because `Vgs = 10 V` is never reached.

Additional corruption observed in one variant: the transient output at `t = 0` already showed
`Vg = 4.855 V` (should be 0 V) — i.e. the initial DC operating point itself landed on a wrong
solution of this model (consistent with the heavy gmin/source-stepping struggles this subcircuit
exhibits even in plain DC analyses).

---

## 3. Which circuit exactly

Two topologies were tried; both fail identically once the drain transitions:

### Topology A (ideal drain source)

```
X M1 d g 0            ; device under test
V bus d 0 40V         ; ideal 40 V on drain
I drive g 0 PWL       ; 0 A at t=0, 5 mA after 1 ns
.tran ...
```
Failure mode: with an ideal voltage source on the drain there is no load-current limit, so when
the channel enhances the macromodel conducts thousands of amps (see Curve 1: ~1668 A at
`Vgs=10 V, Vds=10 V`). The internal capacitance-sense network of the model is driven far outside
its calibrated range; `Vgs` swings +38.6 V → -31.6 V.

### Topology B (current load + rail clamp, datasheet-faithful)

```
X M1 d g 0
V bus vbus 0 40V
D clamp d vbus DCLAMP ; holds drain at ~rail before turn-on
I load 0 d 50A        ; constant 50 A into the drain
I drive g 0 PWL       ; 5 mA gate drive
.tran ...
```
Failure mode: drain now behaves correctly (clamped at 40.69 V pre-turn-on, collapses during
turn-on), but the gate still pumps negative as above.

> Note: the first clamp attempt had the diode reversed (`D clamp vbus d`), which let the drain run
> to the body-diode breakdown of the model (75 V). Correct orientation is `D clamp d vbus`.

---

## 4. Why it happens — root cause inside the vendor model

The vendor file is a MODPEX-generated behavioral macromodel ("Symmetry POWER MOS Model").
Its non-linear capacitances are NOT real capacitor elements on the terminals. They are synthesized
by an auxiliary sense-and-mirror network:

```spice
* relevant excerpt of irfb_s_sl7730pbf
M1    9 7 8 8 MM        ; MOSFET: drain=9, gate=7, source=8
RG    2 7 2.62576       ; external gate pin -> internal gate node 7
...
EV16 10 0 9 7 1        ; VCVS: v(10) = v(9) - v(7)      <- drain-gate difference
CAP  11 10 1.05e-8     ; 10.5 nF sense capacitor
RL    5 10 1           ; 1 ohm sense resistors
RCAP  6 10 1
D2    4 5 MD1          ; steep diodes, N=0.4 / N=0.5
D3    0 5 MD2
D4    0 6 MD3
VFI2  4 0 0            ; 0 V sense sources (measure network currents)
VFI1 11 6 0
FI2   7 9 VFI2 -1      ; CURRENT MIRRORS: inject the sensed currents
FI1   7 9 VFI1 -1      ;  into/out of the internal GATE node 7
```

The chain works as follows: `EV16` reproduces the drain-to-gate voltage across a sense capacitor;
the resulting network currents are sensed by `VFI1/VFI2`; controlled sources `FI1/FI2` mirror those
currents back **into the MOSFET's internal gate node (node 7)**. This is how the model emulates
`Cgd`/Miller current under dynamic drain transitions.

**The pathology:** when the device fully enhances and the drain collapses toward 0 V,
`EV16` forces `v(10) ≈ -Vgs ≈ -10 V`. This forward-biases the steep auxiliary diodes
(`N = 0.4` gives a very sharp exponential that conducts hard below 0.3 V) and puts the whole sense
network into a high-conduction state. The mirrored currents injected into node 7 are then
**amp-level**, not nano/microamp level.

Consequences per drive style:

1. **Current-source gate drive (the datasheet method):** the mirror overwhelms the 5 mA source.
   Net gate current goes strongly negative, `Vgs` collapses below threshold, the device turns off,
   the drain recharges to the rail, and the cycle repeats. Result: gate pumping to -14…-37 V.
2. **Voltage-ramp gate drive (stiff driver through 1 ohm):** the terminal trajectories come out
   clean (`vg` 0→10 V monotonic, `vd` collapses to 0.13 V at 50 A load — proper switching), but the
   driver branch has to absorb the junk current. Measured driver current integrates to
   approximately **-347 uC** instead of the expected ~+150 nC, so `Qg = ∫ Ig dt` remains unusable:
   the garbage current flows through the same gate pin the measurement uses.

Curves 1–4 and 6 are immune because they are pure DC operating points or small-signal AC analyses
around a fixed bias — they never traverse the enhancement transition dynamically, so the
sense-and-mirror loop never enters its pathological state.

---

## 5. Experiment log

| # | Variant | Outcome |
|---|---|---|
| 1 | Ideal 40 V source on drain, 5 mA CC gate drive, trapezoidal | `Vgs` +38.6 V → -31.6 V. Unusable. |
| 2 | `.options method=gear`, reltol=0.01 | Identical garbage — **not** an integrator/timestep issue. |
| 3 | Trapezoidal, 5 ns step | Identical garbage — rules out output-sampling aliasing. |
| 4 | 50 A current load + clamp diode reversed (`vbus→d`) | Drain runs to model breakdown (75 V); gate pumped to ±46 V. Worse. |
| 5 | Clamp corrected (`d→vbus`) | Drain correct (40.69 V ↔ collapse), gate still pumps to -37 V. |
| 6 | Instrumented internal nodes (`xm1.7`, `xm1.9`, `xm1.10`, `xm1.11`) | Confirms t=0 OP corruption (`Vg=4.855 V` at t=0) and sustained junk currents. |
| 7 | Voltage-ramp drive (PULSE ramp 30 us through 1 ohm) | Terminal waveforms perfect; driver branch absorbs amp-level mirror junk; `∫Ig dt = -347 uC` (garbage). |

Key deduction from rows 2–3: the failure is structural (model topology), not numerical.
Key deduction from row 7: the pathology lives entirely inside the auxiliary network's contribution
to the gate-pin current; the *terminal voltages themselves* behave correctly under stiff drive.

---

## 6. Candidate resolutions

| Option | Idea | Tradeoff |
|---|---|---|
| A. Quasi-static reconstruction (recommended) | Build `Qg(Vgs)` analytically from already-validated pieces: `Qgs = ∫ Ciss dVgs` (high-Vds Ciss up to plateau voltage from transfer curve), `Qgd = ∫ Crss dVds` over the drain swing (Miller segment), tail charge with low-Vds Ciss. Uses only Curve 2 + Curve 4 machinery. | Not a raw transient; assumes weak Vgs-dependence of the capacitances (standard industry hand-calculation assumption). |
| B. Different vendor model | Try `CSD19503KCS.lib` / `csd18510kcs.lib` (native TI models, likely no MODPEX sense-mirror network). | Solves nothing for IRFS7730PBF specifically; useful cross-check. |
| C. Patch the subcircuit | Replace the EV16/CAP/FI block with explicit time-varying charge elements or remove FI mirrors for the Qg study only. | Mutates vendor model; results no longer "golden". Last resort. |

Option A requires no new simulation physics, reuses `CapacitanceTestbench` +
`StaticSweepTestbench.run_transfer_characteristics`, and produces the datasheet-standard three-segment
charge curve with annotated Qgs / Qgd / Qg breakpoints.

---

## 7. Verified working extractions (same device, for reference)

| Quantity | Extracted | Sanity vs datasheet |
|---|---|---|
| Rds(on) @ Vgs=10 V, Id=50 A | 2.52 mOhm | typ 2.0 / max 2.6 mOhm — plausible |
| Vth @ Id=250 uA, Vds=10 V | 4.30 V | matches model `VTO=4.34` (datasheet range 2–4 V) |
| Ciss @ Vds=40 V | 13.26 nF | consistent with `CGSO·W` of the model |
| Coss @ Vds=40 V | 0.89 nF | plausible magnitude/shape |
| Crss @ Vds=40 V | 0.56 nF | plausible magnitude/shape |
| Body diode Vsd @ Isd=50 A | 816 mV | typical for 75 V-class MOSFET body diode |
