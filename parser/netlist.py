"""SPICE netlist and subcircuit header inspection."""

import re
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel


class SubcircuitInfo(BaseModel):
    name: str
    nodes: List[str]
    raw_lines: List[str]
    parameters: Dict[str, str] = {}


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

            subckt_match = re.match(r'^\.SUBCKT\s+([A-Za-z0-9_]+)\s+(.+)', stripped, re.IGNORECASE)
            if subckt_match:
                name = subckt_match.group(1)
                nodes = subckt_match.group(2).split()
                current_subckt = SubcircuitInfo(name=name, nodes=nodes, raw_lines=[line])
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
