"""Result container for the custom solver.

The attribute set intentionally mirrors ``simulation.engine.SimResult``
(``traces``, ``sweep_var``, ``log_text``, ``warnings``) so that benches,
``metrics/*``, the plots and the G0-G6 gate evaluation consume solver output
without a second code path.  A parity test guards the contract; see
``tests/test_engine_solver.py``.

Two fields are ours and are not part of the SPICE result: ``steps`` and
``solver_time_s``.  They are recorded because a speedup claim without a step
count cannot be checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class Event:
    """A localised region discontinuity on the time line.

    ``time`` is the instant the device's region changed, bisected on the step
    fraction rather than snapped to a step boundary -- that localisation is
    what makes edge timing measurable instead of quantised to the step.
    """

    time: float
    device: str
    from_region: int
    to_region: int


@dataclass
class SolverResult:
    """One completed solver run."""

    traces: dict[str, np.ndarray]
    sweep_var: str = "time"
    log_text: str = ""
    warnings: list[str] = field(default_factory=list)
    steps: int = 0
    solver_time_s: float = 0.0
    method: str = "trap"
    step_time_s: float = 0.0
    events: tuple[Event, ...] = ()

    def __post_init__(self) -> None:
        lengths = {len(v) for v in self.traces.values()}
        if len(lengths) > 1:
            raise ValueError(
                f"traces have inconsistent lengths: {sorted(lengths)}")
        if self.sweep_var not in self.traces:
            raise ValueError(f"missing sweep variable {self.sweep_var!r}")

    def get(self, name: str) -> np.ndarray:
        try:
            return self.traces[name]
        except KeyError:
            raise KeyError(
                f"trace {name!r} not found; available: {sorted(self.traces)}")

    @property
    def n_points(self) -> int:
        return len(self.traces[self.sweep_var])

    def voltage(self, node: str) -> np.ndarray:
        return self.get(f"V({node})")

    def current(self, element: str) -> np.ndarray:
        return self.get(f"I({element})")
