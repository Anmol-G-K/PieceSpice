"""Fix ordering bug: in both threshold and body-diode, the V(tjsrc) is created
BEFORE the X that references tj_M1 - fine in SPICE (order doesn't matter), but
the double-X (X without tj, then X with tj) creates a duplicate instance.
Cleaner: build xargs including tj BEFORE the X call."""
p = 'simulation/testbenches/static_sweep.py'
src = open(p).read()

# Threshold fix: move V before X properly - actually order irrelevant in SPICE,
# but the duplicate X IS a problem. Restructure:
old1 = '''            circuit.X("M1", subcircuit_name, *xargs)
            if has_tj:
                circuit.V("tjsrc_M1", "tj_M1", circuit.gnd,
                          float(self.engine.temperature)@u_V)
                # Tj port needs to be part of X; rebuild with the extra port.
                circuit.X("M1", subcircuit_name, *xargs, "tj_M1")
            circuit.V("sweep", "gd", circuit.gnd, 0@u_V)'''
new1 = '''            if has_tj:
                circuit.V("tjsrc_M1", "tj_M1", circuit.gnd,
                          float(self.engine.temperature)@u_V)
                xargs.append("tj_M1")
            circuit.X("M1", subcircuit_name, *xargs)
            circuit.V("sweep", "gd", circuit.gnd, 0@u_V)'''
assert old1 in src, "threshold pattern"
src = src.replace(old1, new1)

old2 = '''            circuit.X("M1", subcircuit_name, *xargs)
            if has_tj:
                circuit.V("tjsrc_M1", "tj_M1", circuit.gnd,
                          float(self.engine.temperature)@u_V)
                xargs = xargs + ["tj_M1"]
                circuit.X("M1", subcircuit_name, *xargs)
            circuit.V("sd", src_node, xdrain, 0@u_V)'''
new2 = '''            if has_tj:
                circuit.V("tjsrc_M1", "tj_M1", circuit.gnd,
                          float(self.engine.temperature)@u_V)
                xargs.append("tj_M1")
            circuit.X("M1", subcircuit_name, *xargs)
            circuit.V("sd", src_node, xdrain, 0@u_V)'''
assert old2 in src, "bodydiode pattern"
src = src.replace(old2, new2)

open(p,'w').write(src)
print("ordering fixed")
