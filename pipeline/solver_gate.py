"""Stage 7: the G7 gate — the solver against the *same* ROM in LTspice.

G0-G6 score a model against the vendor golden, so their numbers carry the model
term and the engine term together.  G7 scores the engine alone: the reference is
the emitted library executed by LTspice, so model error cancels by construction
and what is left is the solver's own contribution.

Like every other gate in this project it **reads the recorded JSON and computes
nothing**.  A gate that recomputed its input could disagree with the record it is
supposed to be scoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.solver_policy import load_solver_bars

ROOT = Path(__file__).resolve().parent.parent

#: Basis to record path.  The solver ships one ROM basis at a time; the record
#: name follows it so a G7 verdict can never be quoted without its basis.
BASIS_RECORDS = {
    "tier2": "solver_validation.json",
    "ann": "solver_validation_ann.json",
    "t4ann": "solver_validation_t4ann.json",
}

DEFAULT_BASIS = "t4ann"


def record_path(vendor: str, basis: str = DEFAULT_BASIS) -> Path:
    if basis not in BASIS_RECORDS:
        raise ValueError(f"unknown solver basis {basis!r} "
                         f"(choose from {sorted(BASIS_RECORDS)})")
    return ROOT / "reports" / "comparison" / vendor / BASIS_RECORDS[basis]


def g7_solver_equivalence(vendor: str, *, basis: str = DEFAULT_BASIS,
                          policy_path: str | Path | None = None
                          ) -> dict[str, Any]:
    """Return the G7 gate for *vendor*, scored from the recorded JSON."""
    bars, sources = load_solver_bars(policy_path)
    path = record_path(vendor, basis)
    if not path.is_file():
        return {"pass": False, "status": "not_measured",
                "detail": {"reason": f"{path} is missing",
                           "basis": basis}}

    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("basis") != basis:
        return {"pass": False, "status": "mismatch",
                "detail": {"reason": "record basis does not match the request",
                           "record_basis": record.get("basis"),
                           "basis": basis}}

    comparison = record.get("comparison", {})
    # The ceiling arm is the like-for-like one: the deck's max step is a bound
    # in SPICE, not a fixed step, and the reference runs the same way.
    engine = comparison.get("solver_error_C_minus_B_capped") or {}
    speed = record.get("speed_capped") or {}
    vds = float((engine.get("vds") or {}).get("rel_rmse_pct", float("nan")))
    idd = float((engine.get("id") or {}).get("rel_rmse_pct", float("nan")))
    speedup = speed.get("solver_elapsed_speedup")

    bar = bars["equivalence_rel_pct_max"]
    speed_bar = bars["min_solver_speedup"]
    finite = vds == vds and idd == idd
    fidelity_ok = bool(finite and max(vds, idd) <= bar)
    speed_ok = speedup is not None and float(speedup) >= speed_bar
    return {
        "pass": bool(fidelity_ok and speed_ok),
        "status": "measured" if finite else "not_measured",
        "detail": {
            "basis": basis,
            "record": str(path.relative_to(ROOT)),
            "engine_vds_pct": vds,
            "engine_id_pct": idd,
            "equivalence_bar_pct": bar,
            "equivalence_ok": fidelity_ok,
            "solver_speedup": speedup,
            "speed_bar": speed_bar,
            "speed_ok": speed_ok,
            "sources": sources,
            "rule": ("engine error is measured against the same ROM executed by "
                     "LTspice, so the model term cancels"),
        },
    }


def summarise(basis: str = DEFAULT_BASIS,
              vendors: tuple[str, ...] | None = None) -> dict[str, dict]:
    """G7 for every configured vendor, as a digest."""
    from pipeline.registry import VENDORS

    return {vendor: g7_solver_equivalence(vendor, basis=basis)
            for vendor in (vendors or VENDORS)}
