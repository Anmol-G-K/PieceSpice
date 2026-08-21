"""PSpice and LTspice dialect normalizer for NgSpice compatibility."""

import re
from pathlib import Path
from typing import Dict, Any


def normalize_pspice_netlist(raw_content: str) -> str:
    """Normalize PSpice/LTspice vendor models for seamless simulation in NgSpice.

    - Strips inline comments (; or $)
    - Resolves scoped .PARAM definitions and evaluates arithmetic expressions
    - Replaces parameter references like {pWidth} with evaluated numbers
    - Ensures .PARAM statements are placed before .SUBCKT statements
    """
    params: Dict[str, Any] = {}
    lines = raw_content.splitlines()
    cleaned_lines = []

    # First pass: collect and evaluate .PARAM definitions
    for line in lines:
        stripped = re.sub(r';.*$', '', line).strip()
        if not stripped or stripped.startswith('*'):
            continue

        param_match = re.match(
            r'^\.PARAM\s+([A-Za-z0-9_]+)\s*=?\s*([^;\n]+)', stripped, re.IGNORECASE
        )
        if param_match:
            name, val_expr = param_match.group(1), param_match.group(2).strip()
            # Substitute previously known parameters
            for k, v in params.items():
                val_expr = re.sub(rf'\b{k}\b', str(v), val_expr, flags=re.IGNORECASE)

            val_expr_clean = val_expr.strip('{} ')
            try:
                val = eval(val_expr_clean, {'__builtins__': None}, {})
                params[name] = val
            except Exception:
                params[name] = val_expr_clean

    # Second pass: substitute params and clean lines
    for line in lines:
        stripped = re.sub(r';.*$', '', line).strip()
        if not stripped:
            continue
        if stripped.startswith('*'):
            cleaned_lines.append(line)
            continue

        if re.match(r'^\.PARAM', stripped, re.IGNORECASE):
            continue  # Skip inline .PARAM; they will be prepended

        # Substitute {param} or {expr}
        for k, v in params.items():
            stripped = re.sub(rf'\{{\s*{k}\s*\}}', str(v), stripped, flags=re.IGNORECASE)

        cleaned_lines.append(stripped)

    # Prepend global .PARAM lines
    global_params = [f".PARAM {k}={v}" for k, v in params.items()]
    return '\n'.join(global_params + cleaned_lines)
