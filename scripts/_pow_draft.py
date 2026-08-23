"""Fractional-power hardening for behavioral B-source expressions.

ngspice >= 36 raises a fatal domain error when a fractional power receives a
negative base ("out of range for pow"). ngspice evaluates BOTH branches of
if(), so expressions that are mathematically safe in the taken branch still
crash in the untaken branch.

Safe expansion strategy (conservative, targeted - NOT blanket abs()):

Only rewrite power operators whose exponent is fractional; wrap the base in
abs():   BASE ** EXP   ->   (abs(BASE)) ** EXP

Safety rules:
- Integer exponents untouched (pow(neg, int) is well-defined).
- Bases already wrapped in abs(...) untouched.
- Non-negative numeric literal bases untouched.
- Applied ONLY inside behavioral-source expression bodies, never .PARAM.
- Unparseable expressions pass through unchanged - never corrupt.
"""

import re


def _find_last_top_level_power(expr: str):
    """Find last top-level '**' or '^' outside parens/braces.
    Returns (base_str, op_str, expo_str) or None."""
    depth = 0
    for i in range(len(expr) - 1, 0, -1):
        ch = expr[i]
        if ch in ')':
            depth += 1
        elif ch == '(':
            depth -= 1
            # base may be a parenthesized group ending here
        elif depth == 0:
            if ch == '^':
                return expr[:i], '^', expr[i + 1:]
            if ch == '*' and expr[i - 1] == '*':
                return expr[:i - 1], '**', expr[i + 1:]
    return None


def _is_int_literal(s: str) -> bool:
    s = s.strip()
    return bool(re.fullmatch(r'[+-]?\d+', s))


def _is_num_literal_nonneg(s: str) -> bool:
    s = s.strip().rstrip(',')
    try:
        return float(s) >= 0
    except ValueError:
        return False


def _hard_once(expr: str):
    """One hardening step on the last top-level power. Returns (new, changed)."""
    found = _find_last_top_level_power(expr)
    if not found:
        return expr, False
    base, op, expo = found
    b = base.rstrip()

    if _is_int_literal(expo):
        return expr, False
    if _is_num_literal_nonneg(b):
        return expr, False
    if re.search(r'abs\s*\($', b, re.IGNORECASE):
        # base ends with 'abs(' i.e. already inside an abs call
        return expr, False

    # find where the base operand starts: walk backwards over identifier/
    # number/closing paren groups from the operator
    j = len(b)
    depth = 0
    while j > 0:
        ch = b[j - 1]
        if ch in ')]}':
            depth += 1
        elif ch in '([{':
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and (ch.isalnum() or ch in '_.$'):
            pass
        elif depth == 0 and ch in '+-' and j > 1 and not (b[j - 2].isalnum()):
            pass  # unary sign part of operand
        else:
            break
        j -= 1
    operand = b[j:].strip()

    if re.fullmatch(r'abs\s*\(.*\)', operand, re.IGNORECASE | re.DOTALL):
        return expr, False
    try:
        if float(operand) >= 0:
            return expr, False
    except ValueError:
        pass

    new = f"{b[:j]}(abs({operand})){op}{expo}"
    return new, True


def harden_fractional_powers(expr: str) -> str:
    """Harden all fractional powers in *expr*. Idempotent."""
    changed = True
    guard = 0
    while changed and guard < 50:
        changed = False
        expr, changed = _hard_once(expr)
        guard += 1
    return expr
