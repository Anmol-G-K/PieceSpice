"""Solver-side benches: the same double-pulse topology, built as a circuit.

``simulation/testbenches/double_pulse.py`` builds the DPT as netlist *text* for
LTspice.  This module builds the same topology as an
:class:`engine.ir.circuit.Circuit`, so the two engines can be fed decks that
differ only in which model is instantiated -- the condition for model error to
cancel in a comparison.

The bench is deliberately explicit about the two things that decide whether a
comparison is fair: the device current is ``I(Lload) - I(Dclamp)`` on both
arms, and the clamp diode is a piecewise-linear element here rather than
LTspice's exponential one.  That second difference is a *solvability* cost of
the PWL form, and it is why the harness measures it instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.ir.circuit import (
    AnalysisDirective,
    Circuit,
    DeviceModel,
    Element,
    ElementKind,
    NodeIndex,
    Waveform,
)
from engine.rom.spec import SolverRom


@dataclass(frozen=True, slots=True)
class DptConditions:
    """Operating point of the clamped-inductive double pulse."""

    vbus: float = 11.0
    vgs_on: float = 5.0
    i_load_target: float = 10.0
    l_load: float = 5.5e-6
    t1_on: float = 1.0e-6
    t1_width: float = 5.0e-6
    t2_on: float = 10.0e-6
    t2_width: float = 2.0e-6
    edge_time: float = 20e-9
    t_end: float = 14e-6
    max_step: float = 2e-9
    temperature_c: float = 25.0
    clamp_isat: float = 1e-12
    clamp_n: float = 1.0
    clamp_rs: float = 1e-3

    @classmethod
    def from_config(cls, config: dict) -> DptConditions:
        dpt = config.get("dpt", config)
        return cls(
            vbus=float(dpt["vbus"]),
            vgs_on=float(dpt["vgs_on"]),
            i_load_target=float(dpt.get("i_load_target", 10.0)),
            l_load=float(dpt["l_load"]),
            t1_on=float(dpt["t1_on"]),
            t1_width=float(dpt["t1_width"]),
            t2_on=float(dpt["t2_on"]),
            t2_width=float(dpt["t2_width"]),
            edge_time=float(dpt["edge_time"]),
            t_end=float(dpt["t_end"]),
            max_step=float(dpt["max_step"]),
            temperature_c=float(config.get("conditions", {})
                                .get("temperature", 25.0)),
        )

    def gate_waveform(self) -> Waveform:
        """Two pulses, same breakpoints as ``double_pulse._gate_pwl``."""
        edge = self.edge_time
        # The waveform must span the whole run even when a caller shortens
        # t_end for a quick check; otherwise the breakpoints are not monotone.
        stop = max(self.t_end, self.t2_on + self.t2_width + 2 * edge)
        points = [
            (0.0, 0.0),
            (self.t1_on - edge, 0.0),
            (self.t1_on, self.vgs_on),
            (self.t1_on + self.t1_width - edge, self.vgs_on),
            (self.t1_on + self.t1_width, 0.0),
            (self.t2_on - edge, 0.0),
            (self.t2_on, self.vgs_on),
            (self.t2_on + self.t2_width - edge, self.vgs_on),
            (self.t2_on + self.t2_width, 0.0),
            (stop, 0.0),
        ]
        return Waveform.pwl(points)


def instantiate(rom: SolverRom, ref: str,
                connections: dict[str, str]) -> tuple[Element, ...]:
    """Rename a ROM's elements for one instance in a bench.

    Pin nodes are mapped through *connections* (``d``/``g``/``s``); internal
    nodes are prefixed with the instance reference, exactly as the deck reader
    does for an ``X`` instance, so a trace named ``V(dut.gi)`` means the same
    thing on both arms.
    """
    prefix = f"{ref}."
    mapping = {"d": connections["d"], "g": connections["g"],
               "s": connections["s"],
               "di": f"{prefix}di", "gi": f"{prefix}gi", "ss": f"{prefix}ss"}
    return tuple(
        Element(name=f"{prefix}{e.name}", kind=e.kind,
                nodes=tuple(mapping[n] for n in e.nodes),
                value=e.value, waveform=e.waveform, model=e.model,
                expression=e.expression, data=e.data)
        for e in rom.to_circuit().elements
    )


def dpt(rom: SolverRom, conditions: DptConditions | None = None, *,
        ref: str = "dut") -> Circuit:
    """The clamped-inductive double pulse with *rom* as the device under test."""
    cond = conditions or DptConditions()
    device = instantiate(rom, ref, {"d": "drain", "g": "gate", "s": "0"})
    elements = (
        Element(name="Vbus", kind=ElementKind.VSOURCE, nodes=("vbus", "0"),
                waveform=Waveform.constant(cond.vbus)),
        Element(name="Lload", kind=ElementKind.INDUCTOR,
                nodes=("vbus", "drain"), value=cond.l_load),
        Element(name="Dclamp", kind=ElementKind.DIODE,
                nodes=("drain", "vbus"), model="DCLAMP"),
        Element(name="Vgs", kind=ElementKind.VSOURCE, nodes=("gate", "0"),
                waveform=cond.gate_waveform()),
    ) + device
    nodes: list[str] = []
    for element in elements:
        nodes.extend(element.nodes)
    directive = AnalysisDirective(kind="tran", raw=".tran (bench)",
                                  end_time=cond.t_end, start_time=0.0,
                                  max_step=cond.max_step)
    return Circuit(
        title=f"DPT {rom.device}",
        elements=elements,
        models={"DCLAMP": DeviceModel(
            name="DCLAMP", family="D",
            params={"IS": str(cond.clamp_isat), "N": str(cond.clamp_n),
                    "RS": str(cond.clamp_rs)})},
        node_index=NodeIndex.build(nodes),
        directives=(directive,),
        temperature_c=cond.temperature_c,
    )
