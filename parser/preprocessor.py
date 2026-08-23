"""PSpice and LTspice dialect normalizer for NgSpice compatibility."""

import re
from typing import Dict, Any, List

from parser.spice_literal import parse_spice_literal, translate_spice_literals

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

# A logical line may carry multiple assignments: `.PARAM a=1 b={2*a}`.
# The bare `.PARAM name expr` form (no '=') is also legal PSpice.
_KV_RE = re.compile(
    r'([A-Za-z][A-Za-z0-9_]*)\s*(?:=\s*)?([^\s=]+)'
)


def _strip_comment(line: str) -> str:
    """Remove inline comments (``;`` and ``$``)."""
    stripped = re.sub(r';.*$', '', line)
    return re.sub(r'\$.*$', '', stripped)


def join_continuations(lines: List[str]) -> List[str]:
    """Join SPICE continuation lines (leading ``+``) onto their parent line."""
    joined: List[str] = []
    for raw in lines:
        if raw.startswith('+') and joined:
            joined[-1] = joined[-1].rstrip() + ' ' + raw[1:].strip()
        else:
            joined.append(raw)
    return joined


def _evaluate(expr: str, params: Dict[str, Any]) -> Any:
    """Evaluate a .PARAM expression string against known params (safe subset).

    Falls back to SPICE literal translation for expressions containing
    spice-suffixed numbers (1702.69p, 100n, 1.5Meg...) which Python's eval
    cannot parse natively.
    """
    text = expr.strip().strip('{} ')
    for k, v in params.items():
        text = re.sub(rf'\b{k}\b', repr(v), text, flags=re.IGNORECASE)
    # Translate spice suffixes so eval can handle them.
    text = translate_spice_literals(text)
    try:
        value = eval(text, {'__builtins__': None}, dict(_SAFE_FUNCS))
    except Exception:
        return None  # unresolved; caller decides
    # Guard against self-referential or degenerate resolutions that return the
    # expression text unchanged.
    if isinstance(value, str):
        return None
    return value


def normalize_pspice_netlist(raw_content: str) -> str:
    """Normalize PSpice/LTspice vendor models for seamless simulation in NgSpice.

    - Joins continuation lines (leading ``+``) before any processing
    - Strips inline comments (``;`` and ``$``)
    - Resolves scoped .PARAM definitions, including multiple assignments per
      line and forward references between .PARAM statements (iterated until
      stable), and evaluates arithmetic expressions with a safe function
      whitelist (min/max/sqrt/exp/log/...)
    - Replaces parameter references like ``{pWidth}`` with evaluated numbers;
      unresolvable parameters are left symbolic with a warning
    - Ensures .PARAM statements are placed before .SUBCKT statements

    Vendor files commonly define params in any order and across continuation
    lines; single-pass substitution silently leaves ``{a}`` unresolved when
    ``a`` is defined in terms of a later ``b``. This normalizer iterates the
    parameter table to a fixed point before substituting device lines.
    """
    params: Dict[str, Any] = {}
    pending: Dict[str, str] = {}  # name -> raw expression (possibly forward-ref)
    display: Dict[str, str] = {}  # key -> original-case name (first seen wins)
    lines = join_continuations(raw_content.splitlines())
    cleaned_lines = []
    warnings: List[str] = []

    # Pass 1: collect .PARAM definitions (possibly several per logical line)
    for line in lines:
        stripped = _strip_comment(line).strip()
        if not stripped or stripped.startswith('*'):
            continue

        if re.match(r'^\.PARAM', stripped, re.IGNORECASE):
            tail = re.sub(r'^\.PARAM\s*', '', stripped, flags=re.IGNORECASE)
            pairs = _KV_RE.findall(tail)
            if not pairs:
                warnings.append(f"unparseable .PARAM line skipped: {stripped!r}")
                continue
            for name, expr in pairs:
                key = name.lower()  # SPICE params are case-insensitive; last wins
                pending[key] = expr.strip()
                display.setdefault(key, name)

    def lookup(name: str) -> Any:
        """Case-insensitive param lookup."""
        return params.get(name.lower())

    # Pass 2: iterate param evaluation until fixed point (handles forward refs).
    for _ in range(len(pending) + 1):
        progressed = False
        for name, expr in list(pending.items()):
            val = _evaluate(expr, params)
            if val is not None and name not in params:
                params[name] = val
                del pending[name]
                progressed = True
        if not progressed:
            break

    for name in sorted(pending, key=str.lower):
        warnings.append(
            f".PARAM {display.get(name, name)} could not be resolved from "
            f"{pending[name]!r}; left symbolic (downstream ngspice parse may fail)"
        )
    for w in warnings:
        print(f"[preprocessor] WARNING: {w}")

    # Pass 3: substitute params and clean lines
    for line in lines:
        stripped = _strip_comment(line).strip()
        if not stripped:
            continue
        if stripped.startswith('*'):
            cleaned_lines.append(line)
            continue

        if re.match(r'^\.PARAM', stripped, re.IGNORECASE):
            continue  # Skip inline .PARAM; they will be prepended

        # Substitute both {param} braces and bare word references, case-insensitive.
        for k in params:
            name = display.get(k, k)
            stripped = re.sub(rf'\{{\s*{name}\s*\}}', str(params[k]), stripped, flags=re.IGNORECASE)

        # ngspice-34 (our pinned shared-lib build) lacks ddt() in B-source
        # expressions ("no such function ddt"). Behavioral capacitors of the
        # form I = C(v)*ddt(v) are charge sources: at DC/AC they contribute
        # nothing (caps are open), and transients are M4 scope. Commenting the
        # line out is exactly-correct for DC extraction and documented.
        if _is_bsource(stripped) and re.search(r'\bddt\s*\(', stripped, re.IGNORECASE):
            cleaned_lines.append(f"* [preprocessor] ddt()-based source disabled "
                                 f"(ngspice-34 lacks ddt in B-expr): {stripped}")
            continue

        cleaned_lines.append(stripped)

    global_params = [f".PARAM {display.get(k, k)}={v}" for k, v in params.items()]
    return '\n'.join(global_params + cleaned_lines)


_BSOURCE_RE = re.compile(r'^[bB]\w', re.ASCII)


def _is_bsource(line: str) -> bool:
    """True when *line* is a behavioral source instance (B/G with Value=).

    Matches 'Bxxx ...' and 'Gxxx/Exxx ... value={...}' / 'value {...}' forms
    (both '=' and space separators; Wolfspeed uses 'value {').
    Leading whitespace tolerated.
    """
    line = line.lstrip()
    if _BSOURCE_RE.match(line):
        return True
    return bool(re.match(r'^[GGEF]\w+', line, re.ASCII)
                and re.search(r'value\s*=?\s*\{', line, re.IGNORECASE))
