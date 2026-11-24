"""Canonical role -> external-node mapping for all testbenches.

Single source of truth: the benches and simulation.netlist_text import this
instead of redefining their own dict. Adding a new role (e.g. a kelvin-source
pin) is a one-line change in one place.
"""

from typing import Dict, List

# Role name -> external circuit node used by the benches.
# NOTE: these are the literal node names emitted into every generated
# netlist; the historical 'd'/'g' shorthand is not used anywhere.
ROLE_NODES: Dict[str, str] = {
    "drain": "drain",
    "gate": "gate",
    "source": "0",
}


def x_connections(pin_roles: List[str]) -> List[str]:
    """Map resolved pin roles to external node names, positionally.

    Unknown roles (e.g. 'tj') pass through unchanged so callers can drive
    them explicitly.
    """
    return [ROLE_NODES.get(r, r) for r in pin_roles]
