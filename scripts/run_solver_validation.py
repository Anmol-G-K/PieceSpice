"""Reproducible stage-6 run: solver vs LTspice on the same double pulse.

    uv run python scripts/run_solver_validation.py epc2216
    uv run python scripts/run_solver_validation.py            # every vendor

Writes ``reports/comparison/<vendor>/solver_validation.json`` and prints the
digest.  Requires LTspice and an already-fitted Tier-2 ROM per vendor.
"""

from __future__ import annotations

import argparse
import sys

from engine.harness import summarise, validate_vendor
from pipeline.registry import VENDORS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("vendors", nargs="*", default=None)
    parser.add_argument("--basis", default="tier2", choices=("tier2", "ann"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    vendors = args.vendors or list(VENDORS)
    failures = 0
    for vendor in vendors:
        try:
            record = validate_vendor(vendor, basis=args.basis,
                                     repeats=args.repeats)
            print(summarise(record))
            suffix = "" if args.basis == "tier2" else f"_{args.basis}"
            print(f"  written: reports/comparison/{vendor}/solver_validation{suffix}.json")
        except Exception as exc:                     # noqa: BLE001 - reported
            failures += 1
            print(f"{vendor}: FAILED - {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
