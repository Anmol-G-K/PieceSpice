"""Circuit representation for the PieceSpice custom solver.

Three modules, one direction of dependency:

``circuit.py``
    The IR itself: ``Circuit``, ``Element``, ``ElementKind``, device
    declarations, source waveforms, and the node index.  Pure data plus
    structural queries -- no numerics.
``parse.py``
    A restricted SPICE reader that turns a deck this project emits (benches
    via ``simulation.netlist_text``, ROMs via ``models.*_netlist_builder``)
    into the IR.  It is not a general SPICE front end: anything outside
    :mod:`engine.envelope` raises instead of being approximated.
``mna.py``
    Structural modified-nodal analysis: node incidence, branch unknowns for
    voltage sources and inductors, and element stamps.  Assembly is separated
    from evaluation so the structure can be verified once and reused by every
    time step.

The IR deliberately carries *no* SPICE expression tree.  A behavioural
``G.. VALUE={..}`` is recognised as a construct and rejected, because the
solver consumes the ROM's piecewise-linear tables -- not the smooth form
emitted for LTspice.  Lowering the tables is the ROM contract's job.
"""

from __future__ import annotations

__all__: list[str] = []
