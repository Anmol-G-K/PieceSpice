"""The solver-facing ROM contract.

``engine.rom.spec`` turns a fitted :class:`common.types.PWLModelSpec` into the
tables and planes this engine steps; ``engine.rom.artifact`` is the versioned,
hash-checked on-disk form.  The emitted SPICE subcircuit remains the product
artifact -- this is the same fit expressed in the form a piecewise-linear
solver can actually exploit.
"""

from __future__ import annotations

from engine.rom.artifact import dump, load, manifest, payload_sha256
from engine.rom.spec import SCHEMA, SolverRom, circuit_with_rom


def solver_rom(vendor: str, basis: str = "tier2",
               vds_max: float | None = None) -> SolverRom:
    """Build the solver ROM from a named basis.

    The solver consumes tables, so which fitted basis supplies them is a
    choice.  Making that choice explicit is the point: an implicit Tier-2
    default is what made early measurements Tier-2 without saying so.
    """
    if basis == "tier2":
        from pipeline.fitting import build_pwl_spec

        return SolverRom.from_spec(build_pwl_spec(vendor))
    if basis == "ann":
        return SolverRom.from_ann_fit(vendor, vds_max=vds_max)
    if basis == "t4ann":
        return SolverRom.from_t4_ann(vendor)
    raise ValueError(f"unknown solver ROM basis {basis!r} "
                     "(tier2 | ann | t4ann)")

__all__ = [
    "SCHEMA",
    "SolverRom",
    "circuit_with_rom",
    "dump",
    "load",
    "manifest",
    "payload_sha256",
    "solver_rom",
]
