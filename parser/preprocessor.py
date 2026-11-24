"""T1.2 steps 2+3: rewrite preprocessor as a thin pass-through.

LTspice handles .PARAM/{}/TABLE/POLY/fractional powers natively (verified
4/4 vendor libs, step 1). The preprocessor keeps only the mechanical text
hygiene that any engine benefits from: continuation joining + comment
stripping. API name normalize_pspice_netlist is RETAINED so callers
(static_sweep, capacitance, gate_charge, cli) keep importing unchanged -
their behavior becomes a near-no-op normalization, which is correct.
"""
from __future__ import annotations

import logging
import re
from typing import List

_LOGGER = logging.getLogger(__name__)


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


def normalize_pspice_netlist(raw_content: str, *,
                             harden_powers: bool = False) -> str:
    """Mechanical netlist hygiene for LTspice.

    Since T1.2 of the spicelib migration, LTspice natively evaluates
    .PARAM hierarchies, ``{}`` expressions, PSpice TABLE/POLY syntax and
    fractional powers (verified natively against all four supported vendor
    libraries). This function therefore only performs engine-agnostic text
    hygiene:

    - joins continuation lines (leading ``+``)
    - strips inline comments (``;`` and ``$``)

    The signature (including the ignored *harden_powers* kwarg) is retained
    so existing callers import and run unchanged; it has no effect.
    """
    if harden_powers:
        _LOGGER.info("harden_powers=True ignored: LTspice handles "
                     "fractional powers natively.")
    lines = join_continuations(raw_content.splitlines())
    cleaned = [_strip_comment(line).rstrip() for line in lines]
    return "\n".join(cleaned)
