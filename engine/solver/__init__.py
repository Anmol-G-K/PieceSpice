"""Time integration and device evaluation for the PieceSpice custom solver.

The decomposition is the one the feasibility review asks for -- event
scheduler, time integrator, linear solver and device evaluator are separable
-- but they are kept in one process.  There is no serialization boundary in
the step loop: a process hop costs more than the step it would protect.

``transient.py``
    Stepping loop: companion assembly, dense LU, trapezoidal with
    backward-Euler restart, local error control, and the event timeline.
``companion.py``
    Per-element companion models for one time step.  Linear elements only
    write into the assembled matrices; piecewise-linear devices report the
    region they are in so the scheduler can decide whether a step crossed a
    breakpoint.

Dense LU is a decision, not an omission.  The DPT has ~6 MNA unknowns and a
Tier-4 DUT has 3 nodes; at that size a dense solve beats any sparse
factorisation, and KLU's symbolic setup is pure overhead.  Measured on this
machine: 1.71 us per 7x7 solve under numba against 19.10 us under numpy.
Sparse/IDA+KLU support is a later stage, gated on profiling that shows
factorisation is actually the bottleneck.
"""

from __future__ import annotations

__all__: list[str] = []
