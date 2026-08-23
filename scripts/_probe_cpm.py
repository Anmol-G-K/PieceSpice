"""Probe: ALL Coss/Crss points are 1.592e8 F - constant garbage, not just
high-Vds. The drain-drive sim B's vd probe collapses to ~0 for this model at
ALL biases. Root cause: the Wolfspeed model has R_Ld (10 ohm) PARALLEL to
Ld (1nH) between port d and internal d3, plus the G1 source... the AC probe
injects 1A into 'd' through the bias inductor; with vgs=0 the channel is off
and only caps load it. vd probe should show a few volts.

But we get vg_b/vd ratio giving Crss=ciss -> vd≈vg?? Actually Coss=1.59e8 F
means vd_probe ≈ 1/(2pi*f*Coss) ~ 1e-15 V = zero. The d node is AC-grounded!

Why: the model's DRAIN path contains Ld=1nH || R_Ld=10ohm from port d to d3,
then Vdrain_s (0V source) from d3 to d2. A 0V voltage source is a SHORT at
AC - so d2 is AC-grounded through Vdrain_s! And CDS cap connects d2-s1...
The whole internal drain network is AC-solid. But our external probe drives
'd' which connects via Ld/R_Ld to d3->(V source)->d2->R_CGD->d1 etc.

Hmm wait, that shouldn't ground 'd'. Unless... the Gheat source:
Gheat 0 Tj value {abs(V(d,s)*I(Vdrain_s))+...} - references I(Vdrain_s).
At AC with our drive this is fine.

Let me just measure what the vd probe sees directly vs my standalone test
(which passed!). Standalone used X('M1',name,'g','d','0','tj') - SAME.
Difference: bench uses fresh normalized copy per point + retry; standalone
used one copy. And bench ran fine for EPC/TI/IR...

Test: standalone exact bench circuit at vds=0.1:"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from PySpice.Spice.Netlist import Circuit
from simulation.testbenches.static_sweep import normalized_lib_copy

lib = Path('SpiceModels/CPM2-1200-0025B-Bare-Die.lib').resolve()
tl = normalized_lib_copy(lib)
try:
    ca = Circuit("exact_bench_A")
    ca.include(str(tl.resolve()))
    ca.X("M1", "CPM2-1200-0025B", *( ["g", "d", "0", "tj_a_0"] ))
    ca.V("tjsrc_a_0", "tj_a_0", ca.gnd, 25)
    ca.V("ds", "d", ca.gnd, 0.1)
    ca.R("bleed", "g", ca.gnd, 100e6)
    ca.I("drive", "g", ca.gnd, "DC 0 AC 1")
    sim = ca.simulator(temperature=25, nominal_temperature=25)
    op = sim.ac(start_frequency=1e6, stop_frequency=1e6,
                number_of_points=1, variation='lin')
    vg = abs(complex(np.asarray(op.nodes['g'])[0]))
    print(f"PROBE exact-bench-A: vg={vg:.3e} -> Ciss={1/(2*np.pi*1e6*max(vg,1e-15))*1e12:.0f} pF")
finally:
    tl.unlink()
