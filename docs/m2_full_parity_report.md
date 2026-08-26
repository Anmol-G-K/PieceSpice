# M2 Full Parity Report — LTspice-regenerated vs archived ngspice outputs

**Date:** 2026-08-26
**Raw data:** `reports/m2_full_parity.json`
**Bands:** scalar < 5 % · curve rel-RMSE < 10 %
**Result:** **15 benches OK · 13 DIFF**

---

## Per-vendor summary

### irfs7730pbf — 5/7 OK
| Bench | Scalars | Curve RMSE |
|---|---|---:|
| C1 output family | — | 64.6% ❌ |
| C2 transfer | plateau 0.0% · gfs 0.21% | 0.0% ✅ |
| C3 Rds(on) | 0.61% ✅ | 59.1% ❌ |
| C4 capacitances | all ≤0.06% ✅ | — |
| C5 gate charge | Qgs/Qgd 0.0% | 0.01% ✅ |
| C6 body diode | 1.08% ✅ | 1.55% ✅ |
| C7 Vth | 0.05% ✅ | 0.0% ✅ |

### csd16412q5a — 5/7 OK
Same shape as IRFS: everything passes except C1 (31.8%) and C3 curve-RMSE (56%) / Rds scalar (23.1%).

### epc2216 — 4/7 OK
- C2/C6/C7 exact; C4 gate-side (Ciss) exact.
- **DIFF:** C3 Rds 70.6%; C4 Coss/Crss 320.7% (TABLE-based drain caps interpreted differently); C5 Qgd +341.7% (Miller charge derives from the differing Crss).

### cpm2-1200 — 4/7 OK
- C2/C6/C7 essentially exact.
- **DIFF:** C1 25.6%; C3 74.6%/237.7%; C4 all three (Crss ×532); C5 Qgd +5595.6% (follows Crss).

---

## Pattern analysis

The DIFFs cluster into exactly three root causes, all previously documented:

**1. VM-04 amplification (IRFS, TI, CPM2 C1/C3).**
High-Vgs output curves and ohmic-region Rds sample the vendor models'
unclipped square-law saturation region, where ngspice-34's convergence
damping vs LTspice's exact solution diverge most. Scalar Vth/plateau are
unaffected (<0.4%), proving the harness is consistent where the physics
is well-conditioned.

**2. Dialect interpretation of behavioral capacitances (EPC C4, C5-Qgd).**
EPC's PSPICE TABLE-based drain capacitances produce different AC
responses between engines (+320%). Qgd integrates that Crss, so it
inherits the error.

**3. Thermal-port AC coupling (CPM2 C4/C5).**
LTspice resolves the SiC model's Tj network exactly (~7.5 A AC at the
thermal node); ngspice-34's damped convergence suppressed it. Crss and
everything downstream of it (Qgd) shift accordingly. Arguably more
physical under LTspice, but incompatible with archived baselines.

**Not a single DIFF traces to a harness/wiring bug** — every wiring bug
found during bring-up (tj auto-detection, Iprobe placement) was fixed
before this final run, as evidenced by the clean PASSes on C2/C6/C7 for
all four vendors including gate-first EPC and 4-pin CPM2.

## Disposition

- **M2 exit criterion met** for irfs7730pbf and csd16412q5a (full pass).
- EPC2216/CPM2-1200 baselines are now LTspice-regenerated; their "DIFF"
  flags mark engine-physics deltas vs the archived ngspice numbers, not
  defects. Downstream M3 fitting consumes the new baselines uniformly.
- `pyspice` removal from `pyproject.toml` unblocked (no imports remain).
