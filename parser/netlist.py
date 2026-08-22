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
