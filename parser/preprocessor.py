"""PSpice and LTspice dialect normalizer for NgSpice compatibility."""

import re
from typing import Dict, Any

# Safe functions available inside .PARAM expression evaluation.
_SAFE_FUNCS: Dict[str, Any] = {}
try:  # keep working even if numpy is unavailable in some embedding
    import numpy as _np
    _SAFE_FUNCS.update(
        min=_np.minimum, max=_np.maximum, abs=_np.abs,
        sqrt=_np.sqrt, exp=_np.exp, log=_np.log, log10=_np.log10,
        pow=_np.power, sin=_np.sin, cos=_np.cos, tan=_np.tan,
    )
except ImportError:  # pragma: no cover
    import math as _math
    _SAFE_FUNCS.update(
        min=min, max=max, abs=abs, sqrt=_math.sqrt, exp=_math.exp,
        log=_math.log, log10=_math.log10, pow=_math.pow,
        sin=_math.sin, cos=_math.cos, tan=_math.tan,
    )

_PARAM_RE = re.compile(
    r'^\.PARAM\s+([A-Za-z0-9_]+)\s*=?\s*([^;\n$]+)', re.IGNORECASE
)


def _evaluate(expr: str, params: Dict[str, Any]) -> Any:
    """Evaluate a .PARAM expression string against known params (safe subset)."""
    text = expr.strip().strip('{} ')
    for k, v in params.items():
        text = re.sub(rf'\b{k}\b', repr(v), text, flags=re.IGNORECASE)
    try:
        return eval(text, {'__builtins__': None}, dict(_SAFE_FUNCS))
    except Exception:
        return text  # leave unresolved; caller decides


def normalize_pspice_netlist(raw_content: str) -> str:
    """Normalize PSpice/LTspice vendor models for seamless simulation in NgSpice.

    - Strips inline comments (``;`` and ``$``)
    - Resolves scoped .PARAM definitions, including forward references between
      .PARAM statements (iterated until stable), and evaluates arithmetic
      expressions with a safe function whitelist (min/max/sqrt/exp/log/...)
    - Replaces parameter references like ``{pWidth}`` with evaluated numbers
    - Ensures .PARAM statements are placed before .SUBCKT statements

    Vendor files commonly define params in any order and across continuation
    lines; single-pass substitution silently leaves ``{a}`` unresolved when
    ``a`` is defined in terms of a later ``b``. This normalizer iterates the
    parameter table to a fixed point before substituting device lines.
    """
    params: Dict[str, Any] = {}
    pending: Dict[str, str] = {}  # name -> raw expression (possibly forward-ref)
    lines = raw_content.splitlines()
    cleaned_lines = []

    # Pass 1: collect .PARAM definitions
    for line in lines:
        stripped = re.sub(r';.*$', '', line)
        stripped = re.sub(r'\$.*$', '', stripped).strip()
        if not stripped or stripped.startswith('*'):
            continue

        m = _PARAM_RE.match(stripped)
        if m:
            pending[m.group(1)] = m.group(2).strip()

    # Pass 2: iterate param evaluation until fixed point (handles forward refs)
    for _ in range(len(pending) + 1):
        progressed = False
        for name, expr in list(pending.items()):
            val = _evaluate(expr, params)
            evaluated = not isinstance(val, str) or val != expr.strip().strip('{} ')
            if evaluated and name not in params:
                params[name] = val
                del pending[name]
                progressed = True
        if not progressed:
            break
    # Anything left in pending stays symbolic (e.g. references undefined names)

    # Pass 3: substitute params and clean lines
    for line in lines:
        stripped = re.sub(r';.*$', '', line)
        stripped = re.sub(r'\$.*$', '', stripped).strip()
        if not stripped:
            continue
        if stripped.startswith('*'):
            cleaned_lines.append(line)
            continue

        if re.match(r'^\.PARAM', stripped, re.IGNORECASE):
            continue  # Skip inline .PARAM; they will be prepended

        for k, v in params.items():
            stripped = re.sub(rf'\{{\s*{k}\s*\}}', str(v), stripped, flags=re.IGNORECASE)

        cleaned_lines.append(stripped)

    global_params = [f".PARAM {k}={v}" for k, v in params.items()]
    return '\n'.join(global_params + cleaned_lines)
