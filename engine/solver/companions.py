"""Companion models: turning elements into matrix entries for one step.

The math here is textbook SPICE, but three conventions are pinned because a
sign error in any of them produces waveforms that look plausible and are wrong:

* **KCL rows** are "currents leaving the node via elements equals the current
  injected into it", so a conductance *g* between ``a`` and ``b`` stamps
  ``+g`` on the diagonal and ``-g`` off it, and an injected current adds to the
  right-hand side.
* **Branch currents** for voltage sources and inductors run from the element's
  first node to its second.  That is also LTspice's convention for
  ``I(L...)``, which is why the bench's inductor current compares directly.
* **Ground** (``None`` in a stamp) has no row, so a grounded terminal simply
  contributes nothing.

Piecewise-linear devices are not stamped here: they live in
:mod:`engine.solver.device` and are asked for the linear form of the region the
solver has selected.  That separation is what lets the region set be resolved
by a fixed-point over a finite set instead of by Newton iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.envelope import (
    BEHAVIOURAL_SOURCE_TAG,
    CHARGE_CAP_TAG,
    PWL_CAP_TAG,
    PWL_CHANNEL_TAG,
    check,
)
from engine.ir import mna
from engine.ir.circuit import Circuit, ElementKind
from engine.solver.channel import ChannelDevice, ChannelTable
from engine.solver.device import (
    ChargeCapDevice,
    ChargeTable,
    DiodeDevice,
    DiodeModel,
    PwlDevice,
    PwlTable,
    SwitchDevice,
)

#: Which envelope tag each representable element kind maps to.
_ENVELOPE_TAG: dict[ElementKind, str] = {
    ElementKind.RESISTOR: "R",
    ElementKind.CAPACITOR: "C",
    ElementKind.INDUCTOR: "L",
    ElementKind.VSOURCE: "V",
    ElementKind.ISOURCE: "I",
    ElementKind.DIODE: "D",
    ElementKind.SWITCH: "S",
    ElementKind.CHARGE_CAPACITOR: CHARGE_CAP_TAG,
    ElementKind.PWL_CAP: PWL_CAP_TAG,
    ElementKind.PWL_CHANNEL: PWL_CHANNEL_TAG,
    ElementKind.BEHAVIOURAL_SOURCE: BEHAVIOURAL_SOURCE_TAG,
}

#: Element kinds implemented as a piecewise-linear device rather than a
#: constant conductance.
_DEVICE_KINDS = frozenset(
    {ElementKind.DIODE, ElementKind.SWITCH, ElementKind.PWL_CAP,
     ElementKind.PWL_CHANNEL})

#: Integration methods this kernel supports.
METHODS: tuple[str, ...] = ("be", "trap")

#: Region fixed-point limit.  A piecewise-linear device set that has not
#: settled after this many rounds is chattering, not converging.
MAX_REGION_ITERATIONS = 12


def _check_supported(circuit: Circuit) -> None:
    """Fail before stepping, not halfway through a trace.

    A circuit that raises on step 400 would otherwise leave a partial waveform
    that could be mistaken for a completed run.
    """
    for element in circuit.elements:
        tag = _ENVELOPE_TAG[element.kind]
        check(tag)
        if element.kind is ElementKind.RESISTOR:
            assert element.value is not None
            if element.value <= 0.0:
                raise ValueError(
                    f"{element.name}: a zero or negative resistance has no "
                    "conductance form; represent a short as a voltage source")
        if element.kind is ElementKind.CAPACITOR:
            assert element.value is not None
            if element.value < 0.0:
                raise ValueError(f"{element.name}: negative capacitance")
        if element.kind is ElementKind.INDUCTOR:
            assert element.value is not None
            if element.value < 0.0:
                raise ValueError(f"{element.name}: negative inductance")
        if element.kind is ElementKind.PWL_CAP and not isinstance(
                element.data, ChargeTable):
            raise TypeError(
                f"{element.name}: a PWL_CAP must carry a ChargeTable payload")
        if element.kind is ElementKind.PWL_CHANNEL and not isinstance(
                element.data, ChannelTable):
            raise TypeError(
                f"{element.name}: a PWL_CHANNEL must carry a ChannelTable "
                "payload")


@dataclass(frozen=True, slots=True)
class LinearSystem:
    """A circuit compiled for repeated linear solves.

    Compiling once is what makes the structure verifiable: the unknown count,
    node ordering, element positions and device table regions are fixed here
    and never re-derived.
    """

    circuit: Circuit
    structure: mna.MnaStructure
    capacitors: tuple[mna.Stamp, ...] = ()
    inductors: tuple[mna.Stamp, ...] = ()
    devices: tuple[PwlDevice, ...] = ()

    @classmethod
    def compile(cls, circuit: Circuit) -> LinearSystem:
        _check_supported(circuit)
        structure = mna.build(circuit)
        stamps = {s.name: s for s in structure.stamps}
        temperature = (circuit.temperature_c
                       if circuit.temperature_c is not None else 27.0)
        return cls(
            circuit=circuit,
            structure=structure,
            capacitors=tuple(stamps[e.name]
                             for e in circuit.of_kind(ElementKind.CAPACITOR)),
            inductors=tuple(stamps[e.name]
                            for e in circuit.of_kind(ElementKind.INDUCTOR)),
            devices=tuple(_device_for(stamps[e.name], circuit, temperature)
                          for e in circuit.elements
                          if e.kind in _DEVICE_KINDS),
        )

    @property
    def size(self) -> int:
        return self.structure.size

    def node(self, name: str) -> int | None:
        return self.structure.node_index.index(name)

    def device(self, name: str) -> PwlDevice:
        for device in self.devices:
            if device.name == name:
                return device
        raise KeyError(f"no device named {name!r}")

    @property
    def discontinuous(self) -> tuple[PwlDevice, ...]:
        return tuple(d for d in self.devices if d.discontinuous)


def _device_for(stamp: mna.Stamp, circuit: Circuit,
                temperature_c: float) -> PwlDevice:
    element = stamp.element
    # A declared `.model` wins over a sampled table: the emitter's Shockley
    # parameters are the model, and the table is only the fallback when a deck
    # carries a tabulated diode instead.
    if element.kind is ElementKind.DIODE and element.model is None             and isinstance(element.data, PwlTable):
        return DiodeDevice(element.name, stamp.nodes, element.data)
    if element.kind is ElementKind.DIODE:
        model = circuit.models[element.model or ""]
        params = {k.upper(): model.get(k) for k in model.params}
        diode = DiodeModel.build(
            isat=float(params.get("IS") or 1e-14),
            n=float(params.get("N") or 1.0),
            rs=float(params.get("RS") or 0.0),
            temperature_c=temperature_c,
        )
        return DiodeDevice.from_model(element.name, stamp.nodes, diode)
    if element.kind is ElementKind.SWITCH:
        model = circuit.models[element.model or ""]
        params = {k.upper(): model.get(k) for k in model.params}
        return SwitchDevice.from_model(
            element.name, stamp.nodes,
            {k.lower(): v for k, v in params.items() if v is not None})
    if element.kind is ElementKind.PWL_CAP:
        assert isinstance(element.data, ChargeTable)
        return ChargeCapDevice(element.name, stamp.nodes, element.data)
    if element.kind is ElementKind.PWL_CHANNEL:
        assert isinstance(element.data, ChannelTable)
        return ChannelDevice(element.name, stamp.nodes, element.data)
    raise AssertionError(f"no device for {element.kind!r}")   # pragma: no cover


@dataclass(slots=True)
class DynamicState:
    """Everything the companion models need from the previous step."""

    vc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ic: np.ndarray = field(default_factory=lambda: np.zeros(0))
    il: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vl: np.ndarray = field(default_factory=lambda: np.zeros(0))
    regions: dict[str, int] = field(default_factory=dict)

    @classmethod
    def zeros(cls, system: LinearSystem) -> DynamicState:
        return cls(
            vc=np.zeros(len(system.capacitors)),
            ic=np.zeros(len(system.capacitors)),
            il=np.zeros(len(system.inductors)),
            vl=np.zeros(len(system.inductors)),
            regions={},
        )


def _stamp_conductance(G: np.ndarray, a: int | None, b: int | None,
                       g: float) -> None:
    if a is not None:
        G[a, a] += g
        if b is not None:
            G[a, b] -= g
    if b is not None:
        G[b, b] += g
        if a is not None:
            G[b, a] -= g


def _inject(b_vec: np.ndarray, a: int | None, bb: int | None,
            current: float) -> None:
    """Inject *current* into transition ``a`` -> ``b`` (leaving a, entering b)."""
    if a is not None:
        b_vec[a] -= current
    if bb is not None:
        b_vec[bb] += current


def assemble(system: LinearSystem, state: DynamicState, *,
             t: float, h: float, method: str,
             G: np.ndarray, rhs: np.ndarray,
             regions: dict[str, int] | None = None) -> None:
    """Fill ``G`` and ``rhs`` for one time step.  Both buffers are zeroed first.

    ``h`` is the step length in seconds and is unused for purely resistive
    elements, so an operating-point solve can call with any positive value.

    Device regions are taken from ``state.regions`` (absent = region 0).  The
    caller is responsible for having settled them; this function never asks a
    device to change region mid-assembly.
    """
    if method not in METHODS:
        raise ValueError(f"unknown integration method {method!r}")
    regions = state.regions if regions is None else regions
    G.fill(0.0)
    rhs.fill(0.0)

    cap_index = 0
    ind_index = 0
    for stamp in system.structure.stamps:
        element = stamp.element
        kind = element.kind
        a = stamp.nodes[0]
        bb = stamp.nodes[1]

        if kind is ElementKind.RESISTOR:
            assert element.value is not None
            _stamp_conductance(G, a, bb, 1.0 / element.value)

        elif kind is ElementKind.CAPACITOR:
            assert element.value is not None
            c = element.value
            v_prev = float(state.vc[cap_index])
            if method == "trap":
                g_eq = 2.0 * c / h
                i_eq = g_eq * v_prev + float(state.ic[cap_index])
            else:
                g_eq = c / h
                i_eq = g_eq * v_prev
            _stamp_conductance(G, a, bb, g_eq)
            # i_C(leaving a) = g_eq * (v_a - v_b) - i_eq: the equivalent source
            # injects i_eq at a and extracts it at b.
            if a is not None:
                rhs[a] += i_eq
            if bb is not None:
                rhs[bb] -= i_eq
            cap_index += 1

        elif kind is ElementKind.INDUCTOR:
            assert element.value is not None
            k = stamp.branch
            assert k is not None
            r_eq = (2.0 * element.value / h if method == "trap"
                    else element.value / h)
            i_prev = float(state.il[ind_index])
            if method == "trap":
                v_eq = -r_eq * i_prev - float(state.vl[ind_index])
            else:
                v_eq = -r_eq * i_prev
            _stamp_branch(G, rhs, a, bb, k)
            G[k, k] -= r_eq
            rhs[k] = v_eq
            ind_index += 1

        elif kind is ElementKind.VSOURCE:
            k = stamp.branch
            assert k is not None
            assert element.waveform is not None
            _stamp_branch(G, rhs, a, bb, k)
            rhs[k] = element.waveform.value(t)

        elif kind is ElementKind.ISOURCE:
            assert element.waveform is not None
            _inject(rhs, a, bb, element.waveform.value(t))

        elif kind in _DEVICE_KINDS:
            device = system.device(element.name)
            device.stamp(regions.get(element.name, 0), G=G, rhs=rhs, h=h)

        else:                                   # pragma: no cover - guarded above
            raise AssertionError(
                f"unsupported element kind reached the kernel: {kind!r}")


def _stamp_branch(G: np.ndarray, rhs: np.ndarray, a: int | None,
                  b: int | None, k: int) -> None:
    """Couple branch current *k* into the KCL rows and write its equation row."""
    if a is not None:
        G[a, k] += 1.0
        G[k, a] += 1.0
    if b is not None:
        G[b, k] -= 1.0
        G[k, b] -= 1.0


def assemble_op(system: LinearSystem, state: DynamicState, *, t: float,
                G: np.ndarray, rhs: np.ndarray,
                regions: dict[str, int] | None = None) -> None:
    """Operating point: capacitors open, inductors short-circuited.

    This is the same initial condition LTspice uses with ``uic`` off, which is
    the default in every deck this project writes.  Getting it wrong would
    offset the whole trace and look like solver error.

    Charge-defined capacitors are open at DC, exactly like linear ones: a
    capacitor that conducts at DC would bias the device it is modelling.
    """
    regions = state.regions if regions is None else regions
    G.fill(0.0)
    rhs.fill(0.0)
    for stamp in system.structure.stamps:
        element = stamp.element
        kind = element.kind
        a = stamp.nodes[0]
        bb = stamp.nodes[1]

        if kind is ElementKind.RESISTOR:
            assert element.value is not None
            _stamp_conductance(G, a, bb, 1.0 / element.value)
        elif kind in (ElementKind.CAPACITOR, ElementKind.PWL_CAP):
            continue                            # open at DC
        elif kind is ElementKind.INDUCTOR:
            k = stamp.branch
            assert k is not None
            _stamp_branch(G, rhs, a, bb, k)
            rhs[k] = 0.0                        # v_a - v_b = 0
        elif kind is ElementKind.VSOURCE:
            k = stamp.branch
            assert k is not None
            assert element.waveform is not None
            _stamp_branch(G, rhs, a, bb, k)
            rhs[k] = element.waveform.value(t)
        elif kind is ElementKind.ISOURCE:
            assert element.waveform is not None
            _inject(rhs, a, bb, element.waveform.value(t))
        elif kind in _DEVICE_KINDS:
            device = system.device(element.name)
            device.stamp(regions.get(element.name, 0), G=G, rhs=rhs, h=1.0)
        else:                                   # pragma: no cover - guarded above
            raise AssertionError(
                f"unsupported element kind reached the kernel: {kind!r}")


def branch_voltage(stamp: mna.Stamp, x: np.ndarray) -> float:
    """Voltage across a two-terminal stamp, from a solution vector."""
    a, b = stamp.nodes[0], stamp.nodes[1]
    va = 0.0 if a is None else float(x[a])
    vb = 0.0 if b is None else float(x[b])
    return va - vb


def capacitor_current(system: LinearSystem, state: DynamicState, *,
                      h: float, method: str, x: np.ndarray) -> np.ndarray:
    """Recompute ``i_C`` from the solved voltages (needed for trap history)."""
    out = np.empty(len(system.capacitors))
    for index, stamp in enumerate(system.capacitors):
        assert stamp.element.value is not None
        c = stamp.element.value
        v_now = branch_voltage(stamp, x)
        if method == "trap":
            g_eq = 2.0 * c / h
            i_eq = g_eq * float(state.vc[index]) + float(state.ic[index])
        else:
            g_eq = c / h
            i_eq = g_eq * float(state.vc[index])
        out[index] = g_eq * v_now - i_eq
    return out
