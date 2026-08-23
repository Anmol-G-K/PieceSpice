# Vendor Model Defect Catalog

**Purpose:** Track every structural defect found in vendor SPICE models during PieceSpice
extraction work. Each defect is root-caused to specific lines of the vendor file, its
impact on extraction studies is recorded, and its implication for PieceSpice's own
Tier 2/3 model synthesis is noted.

**Scope:** Currently covers two vendor model families:

| Vendor | File | Subcircuit | Core | Dialect |
|---|---|---|---|---|
| Infineon/IR | `SpiceModels/irfs7730pbf.spi` | `irfb_s_sl7730pbf` | MODPEX "Symmetry POWER MOS Model" v1.0 (2014-11-11), SPICE3 LEVEL=1 | PSpice-compatible |
| TI NexFET | `SpiceModels/CSD16412Q5A.lib` | `CSD16412Q5A` | TI Analog e-Lab Design Center (2018-02-10), BSIM3v3.2 | PSpice (`LEVEL=7 Version=3.2`) |

**Device reference (IR):** IRFS7730PBF — 75 V, 2.2 mΩ typ, ID 246 A @ 25 °C (silicon limited),
IDM 984 A pulsed, VGS ±20 V, PD 375 W @ TC=25 °C.

**Device reference (TI):** CSD16412Q5A — 25 V, 9–11 mΩ @ 10 V, ID 52 A @ 25 °C (thermal),
Avalanche-rated, VGS ±20 V, PD 63 W @ TC=25 °C. Datasheet SLPS207A.

---

## Defect Index

| ID | Defect | Vendor(s) | Severity | Studies affected | Status |
|---|---|---|---|---|---|
| VM-01 | No subthreshold conduction (fake-high Vth) | IR | High | Curve 2, 5, Vth extraction | Documented (unfixable without vendor mutation) |
| VM-02 | Capacitance sense-and-mirror pathology under dynamic drain transitions | IR | Critical | Curve 5 (transient) | Blocked → workaround shipped |
| VM-03 | Charge-inaccurate Cgd (AC-calibrated only) | IR | High | Curve 5 (reconstructed Qgd) | **Fixed (2026-08-22)** — datasheet-Qgd calibration in `reconstruct_gate_charge` |
| VM-04 | Unclipped saturation current (kA-domain Id) | IR + TI (see VM-06 for TI) | Medium | Curve 1 interpretation, all transients | Documented (unfixable without vendor mutation) |
| VM-05 | Corrupted initial DC operating point | IR | Low | Transient studies | Documented (unfixable without vendor mutation) |
| VM-06 | Saturation current far above datasheet IDC on physically-based models (thermal-limit confusion + dialect drift) | TI | Medium | Curve 1 interpretation, transient sanity checks | Documented (2026-08-23); partially simulator-dialect related |
| VM-07 | Subckt port order differs from Si convention (gate-first); testbench hardwired D,G,S | EPC | Critical (silent wrong results) | All studies on affected vendors | **Fixed (2026-08-23)** — config-driven pin map + semantic validation |

---

## VM-01 — No subthreshold conduction (fake-high Vth)

**Severity:** High
**Vendor file lines:** 21–23

```spice
M1 9 7 8 8 MM L=100u W=100u
.MODEL MM NMOS LEVEL=1 IS=1e-32
+VTO=4.34579 LAMBDA=0.0605061 KP=465.97
```

**Mechanism:** `IS=1e-32` eliminates subthreshold conduction entirely. A LEVEL=1 SPICE
MOSFET conducts zero current until VGS > VTO, then switches on hard. A real device
reaches the JEDEC Vth definition current (250 µA) gradually below threshold.

**Consequence:** MODPEX could not fit VTO to the datasheet Vth; it fitted an *effective
strong-inversion onset* (VTO = 4.346 V) tuned jointly with KP and LAMBDA to match the
high-current transfer characteristic. The datasheet quantity VGS(th) = 2.1–3.7 V does
not exist inside this macromodel.

**Observed impact:**

| Quantity | Extracted | Datasheet |
|---|---|---|
| Vth @ Id=250 µA, Vds=Vgs | 4.30 V | 2.1–3.7 V window |
| gfs @ Id=100 A | ~215–350 S | 249 S |

**PieceSpice implication:** Tier 2 synthesis should not inherit the fitted VTO as a
physical threshold. When synthesizing our own macromodel, either add a softplus-style
subthreshold knee (plan §2.4) or document that Vth is an effective parameter.

---

## VM-02 — Capacitance sense-and-mirror pathology (dynamic drain transitions)

**Severity:** Critical — blocks the datasheet-method gate-charge transient entirely
**Vendor file lines:** 33–56 (EV16 / CAP / RL / RCAP / D2–D4 / VFI1, VFI2 / FI1, FI2)

**Mechanism:** The model's non-linear capacitances are not real terminal capacitors.
They are synthesized by an auxiliary sense-and-mirror network: `EV16` reproduces the
drain-gate voltage across a sense capacitor (`CAP`, 10.5 nF); the network currents are
sensed by `VFI1`/`VFI2` and mirrored by controlled sources `FI1`/`FI2` into the MOSFET's
internal gate node (node 7). When the device fully enhances and the drain collapses
toward 0 V, `EV16` forces v(10) ≈ −Vgs, forward-biasing the steep auxiliary diodes
(N = 0.4, hard conduction below ~0.3 V). The sense network enters high conduction and
the mirrored currents become **amp-level**, overwhelming any realistic gate drive.

**Observed impact (constant-current gate drive, 5 mA):** Vgs rises to ~4.7–4.9 V, then
collapses monotonically to −14…−37 V while the device de-enhances; Qg = ∫Ig dt is
unrecoverable. Confirmed structural, not numerical (Gear integrator, reltol, and
timestep changes have no effect). Full experiment log: `gate_charge_error_analysis.md`.

**Workaround shipped:** Quasi-static three-segment reconstruction (Option A) in
`simulation/testbenches/gate_charge.py::reconstruct_gate_charge` — builds Qg(Vgs) from
validated Curve 2 (transfer) + Curve 4 (Ciss/Crss) data. No transient required.

**PieceSpice implication:** Any transient study that swings this model's drain while
driving the gate is suspect. Prefer quasi-static reconstruction, or use a different
vendor model (e.g. native TI models CSD19503KCS / csd18510kcs) for golden transients.

---

## VM-03 — Charge-inaccurate Cgd (small-signal calibrated only)

**Severity:** High
**Related to:** VM-02 network

**Mechanism:** The sense network reproduces Crss accurately *at fixed bias* (small-signal
AC — hence extracted Crss @ 25 V = 685 pF vs datasheet 690 pF, −0.7%), but its
large-signal charge delivery across the full drain swing integrates to less than the
real device's Cgd bulge during collapse.

**Observed impact** (quasi-static Qg reconstruction @ Id=100 A, Vbus=38 V):

| Segment | Reconstructed | Datasheet typ |
|---|---|---|
| Qgs | 67.3 nC | 55 nC |
| Qgd | 36.0 nC | 79 nC |
| Qg total @ 10 V | 183.2 nC | 271 nC (−32%) |

Extracted Crss does rise from 685 pF @ 25 V toward ~3.4 nF at low Vds, but not steeply
enough in integrated charge.

**PieceSpice implication:** Curve 4 small-signal capacitance data alone under-predicts
switching charge for this model family. When validating Tier 2 charge sources, target
the datasheet Qgd/Qg values, not the vendor model's integrated Crss.

**Fix shipped (2026-08-22):** `reconstruct_gate_charge` accepts an optional
`qgd_cal_nC` argument — the datasheet Qgd target — which rescales the Miller segment so
the reconstructed charge is physically meaningful. The applied scale factor is reported
in the breakpoints dict as `qgd_cal_scale`. Verified on IRFS7730PBF @ Id=100 A,
Vbus=38 V: Qgd 36.0 nC → 79.0 nC (datasheet typ), total Qg 183.8 → 226.8 nC (datasheet
271 nC); Qgs and tail segments are provably untouched by the rescale (covered by
`tests/test_gate_charge.py::test_qgd_datasheet_calibration`).

---

## VM-04 — Unclipped saturation current (kA-domain Id)

**Severity:** Medium (misleading, but physically consistent with the model's math)
**Vendor file lines:** 21–23 (same core as VM-01)

**Mechanism:** The LEVEL=1 core with KP=465.97, VTO=4.34579, LAMBDA=0.0605061 and
RS=2.03 mΩ / RD=0.1 mΩ degeneration has a saturation current of ~1.7 kA at Vgs=10 V.
LEVEL=1 has no parameter for a maximum drain current, and the fitter had no remaining
degree of freedom: KP and VTO were fixed by Rds(on) (predicted 1/(β·Vov)+RS+RD =
2.509 mΩ vs extracted 2.52 mΩ) and the transfer shape. Verified by hand-solving the
LEVEL=1 equations with degeneration:

| Vgs | Analytic Id_sat (with RS/RD) | Measured in sweep |
|---|---|---|
| 6 V | 343 A | ~275 A |
| 8 V | 996 A | ~975 A |
| 10 V | 1727 A | 1668 A ✅ |

**Why the datasheet differs:** ID = 246 A @ 25 °C is a **thermal/package limit**
("Silicon Limited": 246² × 2.52 mΩ ≈ 152 W < PD 375 W), not an electrical saturation
limit. At 1668 A dissipation would be ≈ 7 kW — 18.7× rated PD. Real silicon additionally
exhibits mobility degradation and velocity saturation at high overdrive, which bend the
Id–Vgs curve over hard; a square-law LEVEL=1 has neither, so current grows as Vov²
without limit.

**Observed impact:** Curve 1 (output characteristics) shows Id up to 1.67 kA at
Vgs=10 V, Vds=10 V. gfs keeps rising to ~338 S @ 1 kA where the real device would
collapse toward 0. The model's "realistic" operating point is Id = 246 A at Vgs ≈ 5.66 V.

**PieceSpice implication:**
1. Never drive this model beyond ~250–350 A continuous in testbenches.
2. Reference operating points via transfer-curve interpolation (e.g. plateau voltage at
   the target load current), never assume Vgs = 10 V maps to a sane current.
3. Tier 2 synthesis should consider a soft current clamp near the datasheet ID (or a
   thermal-feedback term) — a fidelity-vs-robustness trade-off PieceSpice exists to manage.

---

## VM-05 — Corrupted initial DC operating point (transients)

**Severity:** Low (transient studies only; all DC/AC studies unaffected)
**Evidence:** `gate_charge_error_analysis.md` §2, experiment row 6

**Mechanism:** In one transient variant the output at t = 0 already showed Vg = 4.855 V
(should be 0 V) — the initial DC solve landed on a wrong solution, consistent with the
heavy gmin/source-stepping struggles this subcircuit exhibits even in plain DC analyses.

**PieceSpice implication:** For any transient study on this model, assert the t = 0
operating point against the expected bias before trusting the waveform.

---

## VM-06 — TI BSIM3v3.2: saturation current far above datasheet IDC

**Severity:** Medium (misleading at first glance; model is correct inside its envelope)
**Vendor file:** `CSD16412Q5A.lib` lines 56–60 (M1/M2, W={pWidth}=0.6178 m, L=0.5 µm)
**Status:** Documented (2026-08-23)

**Observation:** At Vgs = 10 V, Vds = 10 V the TI model conducts **325 A**, while the
datasheet lists ID = 52 A @ 25 °C. The same pattern appears on IR (VM-04), so the
question was whether this is a wiring/configuration error on our side.

**Circuit-construction audit (all clean):**
- Node-by-node trace of the TI subckt (nodes 1–14): LDD→RDP→RDD→M1 drain;
  source M1→RSB→RSS→RSP→LSS; gate LGG→RGG; body diode DBD 8↔7; Miller caps
  CGD0/CGS0; clamp diodes DDG/DGD. No floating nodes, no misconnected pins.
- Testbench X-instance maps d,g,gnd → pins 1,2,3 exactly per the file header
  (`1=D 2=G 3=S`).
- Ohmic region validates the model: Rds(on) @ Vgs=10 V = **8.48 mΩ** (spec 9–11),
  @ Vgs=4.5 V = **12.15 mΩ** (spec 13–16). The model is calibrated correctly
  where the datasheet actually characterizes it.

**Root causes (in order of dominance):**

1. **IDC is a thermal rating, not an electrical limit.** "Continuous drain current
   @ 25 °C" is set by die/package heating (RθJC × PD). Nothing in the model's
   electrical physics prevents more current; an isothermal die at Vgs=10 V
   legitimately conducts hundreds of amps. Dissipation at the observed point:
   325 A × 10 V ≈ **3.3 kW** vs 63 W rated — a real die would fail long before,
   but SPICE never heats up.
2. **Simulator dialect drift.** The card is PSpice BSIM3v3.2 (`LEVEL=7
   Version=3.2`); ngspice maps LEVEL=7 → its BSIM3 core. Parameter semantics
   differ subtly (e.g. PSpice treats W as total drawn periphery with its own
   scaling conventions). With W/L ≈ 1.24×10⁶, small differences in
   velocity-saturation / PSCBE channel-length-modulation handling produce very
   large current differences at high Vds — the effective Rds at Vds=10 V reads
   30.7 mΩ, i.e. the curve keeps climbing where the real die compresses.
3. **No electro-thermal feedback.** Only a first-order RTEMP resistor tempco;
   no die-temperature rise, no mobility droop at power.

**Measured comparison points (ngspice via PySpice, ngbehavior=ltpsa):**

| Vendor | Vgs | Vds | Id | Implied P | Datasheet IDC |
|---|---|---|---|---|---|
| TI | 10 V | 10 V | 325 A | ~3.3 kW | 52 A (thermal) |
| IR | 10 V | 10 V | 1668 A | ~16.7 kW | 246 A (thermal) |

**PieceSpice implication:**
1. Same guidance as VM-04 — never assume Vgs = 10 V maps to a sane current on
   either vendor family; reference operating points via transfer-curve interpolation.
2. The TI model is trustworthy in the ohmic region and at datasheet-characterized
   bias points; distrust high-Vds saturation magnitude.
3. A ngspice-vs-PSpice cross-check (running the same card in both simulators) would
   quantify the dialect-drift component if it ever matters for validation.

---

## VM-07 — Non-standard subckt port order (EPC gate-first)

**Severity:** Critical when unfixed — every extraction silently runs with gate and
drain swapped
**Vendor file:** `SpiceModels/EPC2216_V0_PSPICE.net` line 6:
`.subckt EPC2216 gatein drainin sourcein`
**Status:** Fixed (2026-08-23)

**Mechanism:** Si MOSFET vendor libs (IR MODPEX, TI NexFET) declare ports in
numerical order `1=D 2=G 3=S`, matching the legacy drain-first instantiation our
testbenches hardwired (`X(M1 name d g gnd)`). EPC's GaN models instead use named
ports in **gate-first order**. With the hardwired order, Vds was applied across the
gate (abs-max ±6 V!) and Vgs across the power loop — producing near-zero or garbage
currents that initially looked like an ngspice compatibility defect.

**Diagnosis path (for future reference):**
1. Model parses and reaches an OP under ngspice — ruled out dialect issues.
2. B-source syntax (`if/log/exp`) verified working standalone — ruled out syntax.
3. Node-by-node trace of the subckt found no floating/misconnected nodes.
4. Minimal replica of the switch source produced correct current (~26 A at 5/5 V)
   while the full model through the testbench produced ~0 — isolating the fault to
   how the testbench instantiates the subckt, i.e. the port mapping.

**Fix shipped (2026-08-23):**
- `config/inputs/<vendor>.yaml` now declares a pin map:
  ```yaml
  pins:
    pin_order_semantic:
      drain: drainin      # role -> .subckt header token
      gate: gatein
      source: sourcein
  ```
- `parser.netlist.resolve_pin_connections()` parses the actual `.subckt` header from
  the model file, validates every declared token against it (raises on unknown
  tokens), and returns connection roles in positional header order.
- All testbench methods accept `pin_roles=` and build X-instance connections
  positionally; topologies that tie pins (threshold Vgd-tie, reverse-conduction
  gate-source tie) derive their tie pattern from the roles.
- Notebook 01 resolves pins once in its setup cell.
- Normalized copies are now written to `SpiceModels/normalized/` (originals never
  touched); set `PIECESPICE_KEEP_NORM=1` to retain them for inspection.

**Verification:** EPC2216 all curves extract sanely: Rds(on)-region point ~68 mΩ
@5 V/25 A, Ciss ≈ 93–97 pF @12 V, Vth @ 11 mA = 1.52 V vs datasheet 1.4–2 V band ✅,
transfer Id ≈ 26 A @ 6 V/10 V. TI and IR re-run clean (their numeric port maps
resolve unchanged).

---

## Summary: what this means for PieceSpice

The catalog now spans two model generations with a shared failure signature:
a **macromodel fitted to terminal DC/small-signal curves, with no physics-based
clipping** (no subthreshold, no mobility degradation, no thermal feedback,
capacitances emulated rather than physical). VM-01–05 are behavioral-fitting
artifacts of the MODPEX LEVEL=1 flow; VM-06 shows even a physically-based BSIM3v3.2
card exhibits the same "current >> IDC" symptom once die heating is ignored and the
card runs outside its native simulator dialect.
The vendor "golden" model is accurate only inside its calibrated envelope:

- ✅ Trust: DC curves at moderate bias, Rds(on), small-signal capacitances at fixed Vds
- ⚠️ Distrust: Vth as a physical value (IR), anything involving large-signal gate charge
  (IR), currents above ~350 A (IR) / ~50 A continuous-equivalent (TI), all dynamic drain
  transitions on the IR sense-and-mirror network
- 🎯 Tier 2 targets should be the **datasheet** values (Qgs/Qgd/Qg, gfs), not the vendor
  model's, wherever the two disagree

**Simulator configuration note (2026-08-23):** all extraction runs use ngspice via
PySpice with the project-local `.spiceinit` set to `ngbehavior=ltpsa`. This mode is the
only one that both (a) maps PSpice `LEVEL=7` to BSIM3v3.2 and (b) tolerates LTspice
syntax — `all` breaks TI models entirely ("Device type MOS7 not available"). Full
investigation in `development-log.md` (2026-08-23 00:55 entry).

Cross-references: `gate_charge_error_analysis.md` (VM-02 deep dive),
`development-log.md` (chronological record), `piecespice_plan.md` §2 (algorithms these
findings motivate).
