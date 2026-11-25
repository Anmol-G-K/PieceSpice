"""Reproducible stage-6 run: solver vs LTspice on the same double pulse.

    uv run python scripts/run_solver_validation.py epc2216
    uv run python scripts/run_solver_validation.py            # every vendor

Writes ``reports/comparison/<vendor>/solver_validation.json`` and prints the
digest.  Requires LTspice and an already-fitted Tier-2 ROM per vendor.
"""

from __future__ import annotations

import json
import sys

from engine.harness import summarise, validate_vendor
from pipeline.registry import VENDORS


def main(argv: list[str]) -> int:
    vendors = argv[1:] or list(VENDORS)
    failures = 0
    for vendor in vendors:
        try:
            record = validate_vendor(vendor)
            print(summarise(record))
            print(f"  written: reports/comparison/{vendor}/solver_validation.json")
        except Exception as exc:                     # noqa: BLE001 - reported
            failures += 1
            print(f"{vendor}: FAILED - {type(exc).__name__}: {exc}")
        print(json.dumps({})[:0], end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
