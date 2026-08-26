"""Shared netlist-text construction for the LTspice backend.

Replaces the retired PySpice Circuit object tree (and simulation/
testbenches/xinst.py). Testbenches assemble complete SPICE netlists as
text; the engine writes and executes them.

Conventions:
- Ground node is '0'.
- Role -> node mapping comes from common.roles (single source of truth).
- Every generated circuit ends with .end.
"""
from __future__ import annotations

from pathlib import Path

from common.roles import ROLE_NODES


def _conn_for_roles(pin_roles: list[str]) -> list[str]:
    """Positional external nodes for core roles.

    Roles are matched BY NAME (drain/gate/source), not position -
    EPC's gate-first header must still bind its gate to the gate
    node. Unknown roles pass through unchanged.
    """
    canonical = {"drain": "drain", "gate": "gate", "source": "0"}
    return [canonical.get(r, r) for r in pin_roles]



class NetlistBuilder:
    """Accumulates element lines, then renders a complete netlist."""

    def __init__(self, title: str):
        self.title = title
        self._lines: list[str] = [f"* {title}"]
        self._subckt_includes: list[str] = []

    # -- structure -----------------------------------------------------------

    def include(self, lib_path: str | Path) -> "NetlistBuilder":
        self._subckt_includes.append(f".include {Path(lib_path).resolve()}")
        return self

    def raw_line(self, line: str) -> "NetlistBuilder":
        """Append a pre-formatted SPICE line."""
        self._lines.append(line)
        return self

    # -- elements ------------------------------------------------------------

    def v_dc(self, name: str, n_pos: str, n_neg: str, value: float) \
            -> "NetlistBuilder":
        self._lines.append(f"V{name} {n_pos} {n_neg} DC {value:g}")
        return self

    def v_pulse(self, name: str, n_pos: str, n_neg: str, *,
                v1: float, v2: float, td: float, tr: float, tf: float,
                pw: float, period: float) -> "NetlistBuilder":
        self._lines.append(
            f"V{name} {n_pos} {n_neg} PULSE({v1:g} {v2:g} {td:g} "
            f"{tr:g} {tf:g} {pw:g} {period:g})")
        return self

    def r(self, name: str, n_a: str, n_b: str, ohms: float) \
            -> "NetlistBuilder":
        self._lines.append(f"R{name} {n_a} {n_b} {ohms:g}")
        return self

    def c(self, name: str, n_a: str, n_b: str, farads: float) \
            -> "NetlistBuilder":
        self._lines.append(f"C{name} {n_a} {n_b} {farads:.6e}")
        return self

    def l(self, name: str, n_a: str, n_b: str, henries: float) \
            -> "NetlistBuilder":
        self._lines.append(f"L{name} {n_a} {n_b} {henries:g}")
        return self

    def i_dc(self, name: str, n_pos: str, n_neg: str, amps: float) \
            -> "NetlistBuilder":
        self._lines.append(f"I{name} {n_pos} {n_neg} DC {amps:g}")
        return self

    def x_instance(self, ref: str, subcircuit_name: str,
                   pin_roles: list[str], *, temperature: float | None = None,
                   extra_nodes: dict | None = None) -> "NetlistBuilder":
        """Append an X-instance in positional role order.

        Roles map to nodes via common.roles; unknown roles pass through
        unchanged. A role named 'tj' (thermal port) gets a dedicated node
        driven by a V source at `temperature` volts (= degC), unless the
        caller supplies its own node via extra_nodes['tj'].
        """
        tj_roles = [r for r in pin_roles if r.lower() == "tj"]
        core_roles = [r for r in pin_roles if r.lower() != "tj"]
        conn = _conn_for_roles(core_roles)
        if len(tj_roles) == 1:
            node = (extra_nodes or {}).get("tj", f"tj_{ref}")
            conn.append(node)
            self.v_dc(f"tjsrc_{ref}", node, "0",
                      float(temperature if temperature is not None else 25.0))
        self._lines.append(
            f"X{ref} {' '.join(conn)} {subcircuit_name}")
        return self

    # -- render ----------------------------------------------------------------

    def render(self, analysis_card: str | None = None) -> str:
        parts = list(self._subckt_includes)
        parts.extend(self._lines)
        if analysis_card:
            parts.append(analysis_card)
        parts.append(".end")
        return "\n".join(parts) + "\n"
