# Vendor Model Defect Catalog

**Purpose:** Track every structural defect found in vendor SPICE models during PieceSpice
extraction work. Each defect is root-caused to specific lines of the vendor file, its
impact on extraction studies is recorded, and its implication for PieceSpice's own
Tier 2/3 model synthesis is noted.

**Scope:** Currently covers `SpiceModels/irfs7730pbf.spi` (subcircuit `irfb_s_sl7730pbf`,
MODPEX "Symmetry POWER MOS Model" v1.0, generated 2014-11-11, SPICE3 format).

**Device reference:** IRFS7730PBF — 75 V, 2.2 mΩ typ, ID 246 A @ 25 °C (silicon limited),
IDM 984 A pulsed, VGS ±20 V, PD 375 W @ TC=25 °C.

---

## Defect Index

| ID | Defect | Severity | Studies affected | Status |
|---|---|---|---|---|
| VM-01 | No subthreshold conduction (fake-high Vth) | High | Curve 2, 5, Vth extraction | Documented |
| VM-02 | Capacitance sense-and-mirror pathology under dynamic drain transitions | Critical | Curve 5 (transient) | Blocked → workaround shipped |
| VM-03 | Charge-inaccurate Cgd (AC-calibrated only) | High | Curve 5 (reconstructed Qgd) | Documented |
| VM-04 | Unclipped saturation current (kA-domain Id) | Medium | Curve 1 interpretation, all transients | Documented |
| VM-05 | Corrupted initial DC operating point | Low | Transient studies | Documented |

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

## Summary: what this means for PieceSpice

All five defects share one root cause: a **behavioral LEVEL=1 macromodel fitted to
terminal DC/small-signal curves, with no physics-based clipping** (no subthreshold, no
mobility degradation, no thermal feedback, capacitances emulated rather than physical).
The vendor "golden" model is accurate only inside its calibrated envelope:

- ✅ Trust: DC curves at moderate bias, Rds(on), small-signal capacitances at fixed Vds
- ⚠️ Distrust: Vth as a physical value, anything involving large-signal gate charge,
  currents above ~350 A, all dynamic drain transitions
- 🎯 Tier 2 targets should be the **datasheet** values (Qgs/Qgd/Qg, gfs), not the vendor
  model's, wherever the two disagree

Cross-references: `gate_charge_error_analysis.md` (VM-02 deep dive),
`development-log.md` (chronological record), `piecespice_plan.md` §2 (algorithms these
findings motivate).
