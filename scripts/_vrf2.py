"""Re-verify after negation fix."""
import sys
sys.path.insert(0, '.')
from simulation.engine import SpiceEngine
from simulation.testbenches.static_sweep import StaticSweepTestbench

engine = SpiceEngine()
tb = StaticSweepTestbench(engine)
tr = tb.run_transfer_characteristics(
    "SpiceModels/CSD16412Q5A.lib", "CSD16412Q5A",
    vds_fixed=10.0, vgs_max=6.0, vgs_points=21,
    pin_roles=["drain", "gate", "source"])
print("transfer ok:", len(tr.x_data), "pts, max Id =",
      f"{max(tr.y_data):.3f} A, min Id = {min(tr.y_data):.4f} A")
assert min(tr.y_data) >= 0
print("L1 verified: sign-validated, positive conduction data")

oc = tb.run_output_characteristics(
    "SpiceModels/CSD16412Q5A.lib", "CSD16412Q5A", vgs_list=[10.0],
    vds_max=10.0, vds_points=6, pin_roles=["drain", "gate", "source"])
vals = oc[0].y_data
print("C1:", [f"{v:.1f}" for v in vals])
assert all(v >= 0 for v in vals)
print("C1 verified positive")
