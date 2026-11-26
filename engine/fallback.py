"""Stage 8: capability routing with a fail-closed external fallback.

The custom solver serves a closed envelope.  A circuit outside it is **not**
approximated and **not** silently degraded -- it is routed to an external SPICE
engine, or the run stops.  Three rules make that honest:

* **The decision is made before the first step.**  Routing after a partial run
  would leave a trace that looks complete and is not.
* **No fallback speed is ever credited.**  The external engine runs in its own
  process; its time is reported as the fallback's, never as the solver's.
* **No fallback means no answer.**  If the caller cannot supply one, the call
  raises with the reason instead of returning something approximate.

The process boundary matters for licensing too: ngspice and Xyce are GPL, and
this project is MIT, so the external engine is executed as a subprocess rather
than linked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from engine.envelope import EnvelopeError, NotYetImplemented, UnsupportedConstruct
from engine.ir.circuit import Circuit
from engine.solver.companions import LinearSystem
from engine.solver.result import SolverResult
from engine.solver.transient import solve_transient

#: Callable that solves a circuit with an external engine.  Its signature is
#: deliberately the circuit, not the deck: the caller owns the deck plumbing and
#: the process boundary.
Fallback = Callable[[Circuit], SolverResult]


class FallbackUnavailable(RuntimeError):
    """The circuit is outside the envelope and no fallback was supplied."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            f"circuit is outside the custom solver envelope ({reason}) and no "
            "external SPICE fallback was supplied; refusing to approximate")


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Where a circuit goes, and why."""

    engine: str                    # "solver" | "spice"
    reason: str = ""
    unsupported: str = ""

    @property
    def is_fallback(self) -> bool:
        return self.engine == "spice"


def plan_route(circuit: Circuit) -> RouteDecision:
    """Decide the engine for *circuit* before any stepping happens.

    Compiling the system is the capability check: it refuses a construct that is
    outside the envelope (`UnsupportedConstruct`) or declared but unimplemented
    (`NotYetImplemented`).
    """
    try:
        LinearSystem.compile(circuit)
    except UnsupportedConstruct as exc:
        return RouteDecision("spice", f"unsupported construct: {exc.tag}",
                             exc.tag)
    except NotYetImplemented as exc:
        return RouteDecision("spice", f"not implemented yet: {exc.tag}", exc.tag)
    except EnvelopeError as exc:                       # pragma: no cover - base
        return RouteDecision("spice", str(exc), "")
    return RouteDecision("solver")


def solve(circuit: Circuit, *, fallback: Fallback | None = None,
          **solver_kwargs: Any) -> SolverResult:
    """Solve with the custom engine, routing outside the envelope to *fallback*.

    Any ``solver_kwargs`` (``engine``, ``adaptive``, ``max_step``, ``end_time``)
    reach :func:`engine.solver.transient.solve_transient` unchanged, so routing
    never changes the solver's own semantics.
    """
    decision = plan_route(circuit)
    if not decision.is_fallback:
        return solve_transient(circuit, **solver_kwargs)
    if fallback is None:
        raise FallbackUnavailable(decision.reason)
    result = fallback(circuit)
    # The provenance travels with the result: a reader must be able to see that
    # this answer did not come from the custom solver.
    result.warnings.append(
        f"routed to external SPICE: {decision.reason}; "
        "fallback time is not solver time")
    return result
