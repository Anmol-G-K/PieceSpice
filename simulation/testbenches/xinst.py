"""Shared X-instance construction with thermal-node (Tj) support.

Some vendor models (Wolfspeed SiC) expose a 4th port 'Tj' - a junction-
temperature node where voltage in volts equals temperature in C. The
testbenches drive it from a fixed V source at the configured temperature.
"""
from typing import List

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_V

_ROLE_NODES = {"drain": "d", "gate": "g", "source": "0"}


def x_connections(pin_roles: List[str]) -> List[str]:
    """Map resolved pin roles to external node names, positionally."""
    return [_ROLE_NODES.get(r, r) for r in pin_roles]


def attach_x_instance(
    circuit: Circuit,
    ref: str,
    subcircuit_name: str,
    pin_roles: List[str],
    temperature: float = 25.0,
) -> None:
    """Instantiate the DUT subcircuit with correct pin order and Tj drive.

    Builds the X-instance connections positionally from *pin_roles*. A role
    named 'tj' marks the model as having a thermal port: the port is driven
    by a dedicated V source at *temperature* volts (= degC) and is excluded
    from the role->node mapping (it must appear exactly once).
    """
    tj_roles = [r for r in pin_roles if r.lower() == "tj"]
    core_roles = [r for r in pin_roles if r.lower() != "tj"]
    conn = x_connections(core_roles)
    if len(tj_roles) == 1:
        conn.append(f"tj_{ref}")
        circuit.V(f"tjsrc_{ref}", f"tj_{ref}", circuit.gnd,
                  float(temperature)@u_V)
    circuit.X(ref, subcircuit_name, *conn)
