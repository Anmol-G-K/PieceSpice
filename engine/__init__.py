"""PieceSpice custom solver: a piecewise-linear MNA engine for ROM models.

Why this exists
---------------
The emitted ROMs are already small and, on three of four devices, already
fidelity-acceptable.  They are still slow.  Measured at matched 2-pulse,
forced 2 ns-step conditions the Tier-2 CPWL model runs at 0.875-1.173x the
vendor golden and Tier-3 ``v1`` runs at 0.0946-0.5827x -- slower than golden
on every device.  The ROM removed the device's physics but the solver kept
the work: LTspice is a general Newton-based MNA/DAE engine, so every
behavioural channel and charge source has its Jacobian re-linearised on every
step.  The iteration count barely moved.

That is the reason this package exists, and it is a reason about *one half
only*.  Neither half delivers speed alone:

* A solver without a ROM would have to linearise the vendor model's stiff
  physics -- that is a general-purpose SPICE clone, which the feasibility
  review rejects as a multi-year project.
* A ROM without this solver stays inside a Newton loop, which is why the
  measured speedup is ~1x.

The ROM's job is to make the device *piecewise linear*: between breakpoints
the circuit is a linear system, so a step is one LU solve and no Newton
iteration.  This engine's job is to *exploit* that: solve per region, locate
switching events exactly instead of stepping over them, and never iterate on
a Jacobian.  The two are complementary by construction, not by hope.

A second consequence is worth stating because it changes what is being
measured: exact event location removes a discretisation error that no amount
of model tuning can remove.  The current UROM records reject G4
timing/energy on all four devices while passing static and charge gates --
that is an edge-resolution failure, not a model-shape failure.  A solver that
lands exactly on the threshold crossing attacks the gate directly.

See ``docs/research/custom-engine-feasibility.md`` for the evidence review and
``engine/README.md`` for the supported envelope and layout.

Layout
------
``engine.ir``      circuit representation, restricted netlist reader, MNA
                   assembly.  No simulation.
``engine.solver``  companion models, time integration, event handling.
``engine.envelope``  the closed set of elements and analyses the fast path
                   serves, and the two failure modes for everything else.

The result container duck-types ``simulation.engine.SimResult`` so benches,
metrics, plots and the existing G0-G6 gates consume solver output unchanged.
The dependency arrow points one way: this package never imports the LTspice
backend.
"""

from __future__ import annotations

from engine.envelope import (
    EnvelopeError,
    NotYetImplemented,
    UnsupportedConstruct,
    missing,
)

__all__ = [
    "EnvelopeError",
    "NotYetImplemented",
    "UnsupportedConstruct",
    "missing",
]

#: Version of the solver-facing ROM artifact contract.  Bumped when the
#: artifact schema changes shape; recorded alongside every measurement.
ROM_ARTIFACT_SCHEMA = "piecespice.solver_rom/1.0"
