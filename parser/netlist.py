"""SPICE netlist and subcircuit header inspection."""

import re
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel

_HEADER_PARAMS_RE = re.compile(
    r'\bPARAMS\s*:\s*(.+)$', re.IGNORECASE
)
_KV_RE = re.compile(
    r'([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^\s=]+)'
)
_SUBCKT_RE = re.compile(r'^\.SUBCKT\s+(\S+)\s+(.+)', re.IGNORECASE)
_ENDS_RE = re.compile(r'^\.ENDS(?:\s+(\S+))?\s*$', re.IGNORECASE)


class SubcircuitInfo(BaseModel):
    name: str
    nodes: List[str]
    raw_lines: List[str]
    parameters: Dict[str, str] = {}


def resolve_pin_connections(
    lib_path: Path,
    subcircuit_name: str,
    pin_order_semantic: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Resolve X-instance connection roles for a subcircuit, in header order.

    Vendor subckts use different port orders (IR/TI: D,G,S; EPC: G,D,S), and a
    swapped gate/drain silently destroys extraction accuracy or the model.

    Args:
        lib_path: Vendor model file.
        subcircuit_name: Subcircuit to resolve.
        pin_order_semantic: Map of role -> header port token, e.g.
            ``{"drain": "drainin", "gate": "gatein", "source": "sourcein"}``.
            Tokens are validated against the actual ``.subckt`` header
            (semantic check). With ``None``, falls back to the legacy Si
            convention (drain, gate, source) with a warning.

    Returns:
        Role names in the subckt's positional port order, e.g.
        ``["gate", "drain", "source"]`` for EPC2216. The caller maps each role
        to its external circuit node (d-node/g-node/gnd) positionally.
    """
    subs = NetlistParser.parse_subcircuits(lib_path)
    info = next((s for s in subs if s.name.lower() == subcircuit_name.lower()), None)
    if info is None:
        raise ValueError(
            f"Subcircuit {subcircuit_name!r} not found in {lib_path.name}; "
            f"available: {[s.name for s in subs]}"
        )
    ports = [p.lower() for p in info.nodes]

    if not pin_order_semantic:
        print(f"[pins] WARNING: no pin map declared for {subcircuit_name!r}; "
              f"assuming legacy (drain, gate, source) port order.")
        return ports[:3] if len(ports) >= 3 else ["drain", "gate", "source"]

    unknown = {str(t).lower() for t in pin_order_semantic.values()} - set(ports)
    if unknown:
        raise ValueError(
            f"pin_order_semantic tokens {sorted(unknown)} not in "
            f"{subcircuit_name!r} header ports {info.nodes}"
        )
    role_to_port = {r: str(t).lower() for r, t in pin_order_semantic.items()}
    conn_by_port = {role_to_port[r]: r for r in role_to_port}

    roles: List[str] = []
    for port in ports:
        if port in conn_by_port:
            roles.append(conn_by_port[port])
        else:
            print(f"[pins] NOTE: port {port!r} of {subcircuit_name!r} has no "
                  f"declared role; treating as an extra port named {port!r}.")
            roles.append(port)
    return roles



def _split_header_params(rest: str) -> tuple[List[str], Dict[str, str]]:
    """Split a .SUBCKT header tail into pin nodes and PARAMS: key=value pairs.

    PSpice syntax: ``.SUBCKT NAME n1 n2 ... PARAMS: a=1 b={2*a}``.
    Without this, the param tokens are misread as extra pin nodes.
    """
    m = _HEADER_PARAMS_RE.search(rest)
    if not m:
        return rest.split(), {}
    nodes = rest[: m.start()].split()
    params = {k: v for k, v in _KV_RE.findall(m.group(1))}
    return nodes, params


class NetlistParser:
    """Parses SPICE netlists and extracts subcircuit definitions.

    Handles nested subcircuit definitions by tracking the open-definition
    stack: each .SUBCKT opens a frame and only its own matching .ENDS closes
    it (a named ``.ENDS <name>`` is matched to its definition; an unnamed
    .ENDS closes the innermost open frame). Vendor libraries that define
    helper subckts inside subckts therefore parse correctly instead of
    orphaning the outer definitions.
    """

    @staticmethod
    def parse_subcircuits(file_path: Path) -> List[SubcircuitInfo]:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        lines = content.splitlines()

        subcircuits: List[SubcircuitInfo] = []
        # Stack of (info, name_lower). Innermost frame is last.
        stack: List[tuple[SubcircuitInfo, str]] = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('*'):
                continue

            subckt_match = _SUBCKT_RE.match(stripped)
            if subckt_match:
                name = subckt_match.group(1)
                nodes, params = _split_header_params(subckt_match.group(2))
                info = SubcircuitInfo(
                    name=name, nodes=nodes, raw_lines=[line], parameters=params
                )
                stack.append((info, name.lower()))
                continue

            ends_match = _ENDS_RE.match(stripped)
            if ends_match and stack:
                ends_name = ends_match.group(1)
                if ends_name is None:
                    # Unnamed .ENDS closes the innermost open frame.
                    info, _ = stack.pop()
                else:
                    # Named .ENDS closes the matching frame; anything opened
                    # inside it (and never closed) is closed here too so no
                    # raw_lines leak between sibling definitions.
                    target = ends_name.lower()
                    while stack and stack[-1][1] != target:
                        orphaned, _ = stack.pop()
                        orphaned.raw_lines.append(line)
                        subcircuits.append(orphaned)
                    if stack:
                        info, _ = stack.pop()
                    else:
                        # .ENDS names an unknown subckt; ignore line.
                        continue
                info.raw_lines.append(line)
                subcircuits.append(info)
                continue

            if stack:
                stack[-1][0].raw_lines.append(line)

        return subcircuits
