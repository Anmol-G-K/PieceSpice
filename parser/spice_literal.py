"""SPICE literal handling for the preprocessor.

ngspice natively understands spice-suffixed numeric literals (1702.69p, 1.5Meg,
100n, ...). Our Python evaluator cannot. Strategy: try to evaluate; if the
expression is a pure literal that Python can't parse, translate the suffix to
scientific notation ourselves so hoisted .PARAM lines stay numerically correct.
"""

import re

# SPICE unit suffixes (case-sensitive per convention) -> multiplier.
_SPICE_SUFFIXES = {
    't': 1e12, 'g': 1e9, 'meg': 1e6, 'k': 1e3,
    'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15,
}
_SUFFIX_RE = re.compile(
    r'^([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(' + '|'.join(_SPICE_SUFFIXES) + r')$',
    re.IGNORECASE,
)


def parse_spice_literal(text: str):
    """Parse a SPICE numeric literal with optional unit suffix.

    Returns a float, or None if *text* is not a plain/suffixed number
    (e.g. an expression like '{2*pWidth}' or a bare identifier).
    """
    t = text.strip().strip('{}').strip()
    try:
        return float(t)
    except ValueError:
        pass
    m = _SUFFIX_RE.match(t)
    if m:
        # Regex matched case-insensitively against the table keys; a single
        # lowercased lookup is exact and total here.
        return float(m.group(1)) * _SPICE_SUFFIXES[m.group(2).lower()]
    return None


def translate_spice_literals(expr: str) -> str:
    """Replace spice-suffixed literals inside an expression string with
    scientific-notation equivalents so Python/eval can parse them.

    Only touches standalone numbers with suffixes; leaves identifiers alone.
    """
    def _sub(m):
        val = parse_spice_literal(m.group(0))
        return repr(val) if val is not None else m.group(0)

    # Number optionally followed by suffix, bounded by non-identifier chars.
    pattern = re.compile(
        r'(?<![\w.])([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?(?:\s*)?'
        r'(?:t|g|meg|k|m|u|n|p|f))(?![\w])',
        re.IGNORECASE,
    )
    out = pattern.sub(_sub, expr)
    return out
