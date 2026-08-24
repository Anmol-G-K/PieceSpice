"""Canonical role -> external-node mapping for all testbenches.

Single source of truth (Major fix M1 / m9): every bench imports from here
instead of redefining its own dict. Adding a new role (e.g. a kelvin-source
pin) is a one-line change in one place.
"""

from typing import Dict, List

# Role name -> external circuit node used by the static benches.
ROLE_NODES: Dict[str, str] = {
    "drain": "d",
    "gate": "g",
    "source": "0",
}


def x_connections(pin_roles: List[str]) -> List[str]:
    """Map resolved pin roles to external node names, positionally.

    Unknown roles (e.g. 'tj') pass through unchanged so callers can drive
    them explicitly.
    """
    return [ROLE_NODES.get(r, r) for r in pin_roles]
