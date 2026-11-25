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

__all__ = [
    "SCHEMA",
    "SolverRom",
    "circuit_with_rom",
    "dump",
    "load",
    "manifest",
    "payload_sha256",
]
