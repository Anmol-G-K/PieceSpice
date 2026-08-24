"""Fractional-power hardening for behavioral B-source expressions.

ngspice >= 36 raises a fatal domain error when a fractional power receives
a negative base ("out of range for pow"). ngspice evaluates BOTH branches
of if(), so expressions that are mathematically safe in the taken branch
still crash in the untaken branch.

Safe expansion strategy (conservative, targeted - NOT blanket abs()):

Rewrite power operators whose exponent is not a provably-integer literal,
wrapping the base:   BASE ** EXP  ->  (abs(BASE)) ** EXP

Safety rules:
- Integer literal exponents untouched.
- Bases already abs()-wrapped (directly or inside outer parens) untouched.
- Non-negative numeric literal bases untouched.
- Only called on behavioral-source expression bodies, never .PARAM.
- Unparseable expressions pass through unchanged - never corrupt.
"""
import re
_ABS_WRAPPED_RE = re.compile('^\\(*\\s*[Aa][Bb][Ss]\\s*\\([^()]*\\)\\s*\\)*$')

def _find_base_start(expr: str, op_start: int) -> int:
    """Return index where the operand ending just before op_start begins."""
    j = op_start
    while j > 0 and expr[j - 1].isspace():
        j -= 1
    if j == 0:
        return 0
    ch = expr[j - 1]
    if ch == ')':
        depth = 0
        while j > 0:
            c = expr[j - 1]
            if c == ')':
                depth += 1
            elif c == '(':
                depth -= 1
                if depth == 0:
                    j -= 1
                    break
            j -= 1
        k = j
        while k > 0 and (expr[k - 1].isalnum() or expr[k - 1] == '_'):
            k -= 1
        return k
    if ch.isalnum() or ch in '_.':
        while j > 0 and (expr[j - 1].isalnum() or expr[j - 1] in '_.'):
            j -= 1
        return j
    return j

def _is_abs_wrapped(base: str) -> bool:
    """True if base is 'abs(...)' possibly wrapped in redundant parens."""
    s = base.strip()
    while True:
        if s.startswith('(') and s.endswith(')'):
            depth = 0
            wraps = True
            for idx, ch in enumerate(s):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0 and idx != len(s) - 1:
                        wraps = False
                        break
            if wraps and depth == 0:
                inner = s[1:-1].strip()
                if re.fullmatch('[Aa][Bb][Ss]\\s*\\(.*\\)', inner, re.DOTALL):
                    return True
                s = inner
                continue
        break
    return bool(re.fullmatch('[Aa][Bb][Ss]\\s*\\(.*\\)', s.strip(), re.DOTALL))

def harden_fractional_powers(expr: str) -> str:
    """Wrap fractional-power bases in abs(). Idempotent; conservative."""
    guard = 0
    while guard < 100:
        edited = False
        for m in re.finditer('\\*\\*', expr):
            i = m.start()
            rest = expr[i + 2:].lstrip()
            em = re.match('[A-Za-z][A-Za-z0-9_]*|\\d+\\.?\\d*|\\.\\d+', rest)
            if not em:
                continue
            expo = em.group(0)
            if re.fullmatch('\\d+', expo):
                continue
            bstart = _find_base_start(expr, i)
            base = expr[bstart:i].strip()
            if not base:
                continue
            if _is_abs_wrapped(base):
                continue
            try:
                if float(base) >= 0:
                    continue
            except ValueError:
                pass
            expr = expr[:bstart] + f'(abs({base}))' + expr[i:]
            edited = True
            break
        if not edited:
            return expr
        guard += 1
    return expr