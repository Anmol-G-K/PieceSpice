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


class SubcircuitInfo(BaseModel):
    name: str
    nodes: List[str]
    raw_lines: List[str]
    parameters: Dict[str, str] = {}


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
    """Parses SPICE netlists and extracts subcircuit definitions."""

    @staticmethod
    def parse_subcircuits(file_path: Path) -> List[SubcircuitInfo]:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        lines = content.splitlines()

        subcircuits: List[SubcircuitInfo] = []
        current_subckt: Optional[SubcircuitInfo] = None

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('*'):
                continue

            subckt_match = re.match(
                r'^\.SUBCKT\s+([A-Za-z0-9_]+)\s+(.+)', stripped, re.IGNORECASE
            )
            if subckt_match:
                name = subckt_match.group(1)
                nodes, params = _split_header_params(subckt_match.group(2))
                current_subckt = SubcircuitInfo(
                    name=name, nodes=nodes, raw_lines=[line], parameters=params
                )
                continue

            ends_match = re.match(r'^\.ENDS', stripped, re.IGNORECASE)
            if ends_match and current_subckt:
                current_subckt.raw_lines.append(line)
                subcircuits.append(current_subckt)
                current_subckt = None
                continue

            if current_subckt:
                current_subckt.raw_lines.append(line)

        return subcircuits

