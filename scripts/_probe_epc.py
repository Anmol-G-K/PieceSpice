"""Probe 22: bisect the EPC file. Test A: params + rd/rs/rg + bswitch ONLY
(no conv resistors, no diodes, no caps)."""
import sys
sys.path.insert(0, '.')
from pathlib import Path

norm_lines = """\
.PARAM si=0.1122
.PARAM so=0.5517
.PARAM sr=187.0
.PARAM aWg=187
.PARAM A1=12.40932
.PARAM k2=1.983
.PARAM k3=0.09
.PARAM rpara=0.01752
.PARAM rpara_s_factor=0.3671
.PARAM aITc=0.0035
.PARAM arTc=-0.0062
.PARAM k2Tc=0.00068
.PARAM x0_0=5.112
.PARAM x0_0_TC=-0.001
.PARAM x0_1=0.5175
.PARAM x0_1_TC=0.0
.subckt EPC2216 gatein drainin sourcein
rd drainin drain {((1-rpara_s_factor)*rpara*(1-arTc*(Temp-25)))}
rs sourcein source {(rpara_s_factor*rpara*(1-arTc*(Temp-25)))}
rg gatein gate {(rg_value)}
bswitch drain source I=if(v(drain,source)>0, (A1*(1-aITc*(Temp-25))*log(1.0+exp((v(gate,source)-(k2*(1-k2Tc*(Temp-25))))/k3))* v(drain,source)/(1 + (x0_0*(1-x0_0_TC*(Temp-25))+x0_1*(1-x0_1_TC*(Temp-25))*v(gate,source))*v(drain,source))), (-A1*(1-aITc*(Temp-25))*log(1.0+exp((v(gate,drain)-(k2*(1-k2Tc*(Temp-25))))/k3))* v(source,drain)/(1 + (x0_0*(1-x0_0_TC*(Temp-25))+x0_1*(1-x0_1_TC*(Temp-25))*v(gate,drain))*v(source,drain)) ))
.ends
"""

from PySpice.Spice.Netlist import Circuit
p = Path('SpiceModels/_bisect_a.net')
p.write_text(norm_lines)
try:
    c = Circuit("bisect_a")
    c.include(str(p.resolve()))
    c.X("M1", "EPC2216", "d", "g", c.gnd)
    c.V("ds", "d", c.gnd, 5@u_V if False else 5)
    c.V("gs", "g", c.gnd, 5)
    op = c.simulator(temperature=25, nominal_temperature=25).operating_point()
    i = -float(op.Vds[0])
    print(f"VAR bisect-A (switch only): Id={i:.4f} A")
except Exception as e:
    print("VAR bisect-A FAIL:", str(e)[:120])
finally:
    p.unlink(missing_ok=True)

import PySpice.Unit
