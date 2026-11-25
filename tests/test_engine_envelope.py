"""The solver envelope is a contract, not documentation.

These tests exist because the envelope is the one place where "we do not
model that" and "we have not modelled that yet" must stay distinguishable: the
first routes a circuit to the external SPICE engine, the second must stop the
caller rather than let an approximation through.
"""

from __future__ import annotations

import pytest

from engine import NotYetImplemented, UnsupportedConstruct, envelope


@pytest.mark.parametrize("tag", ["R", "C", "L", "V", "I", "X", "D", "S", "PWL_CAP"])
def test_available_elements_resolve(tag: str) -> None:
    support = envelope.check(tag)
    assert support.tag == tag
    assert support.available


@pytest.mark.parametrize("tag", ["C:Q", "G:VALUE"])
def test_planned_elements_report_not_yet_implemented(tag: str) -> None:
    with pytest.raises(NotYetImplemented):
        envelope.check(tag)


@pytest.mark.parametrize("tag", ["K", "E", "B", "T", "J", "not-an-element"])
def test_unknown_elements_are_unsupported(tag: str) -> None:
    with pytest.raises(UnsupportedConstruct):
        envelope.check(tag)


def test_planned_and_unsupported_are_distinct_exception_types() -> None:
    # The router acts differently on the two: one falls back, one aborts.
    assert not issubclass(NotYetImplemented, UnsupportedConstruct)
    assert issubclass(NotYetImplemented, envelope.EnvelopeError)
    assert issubclass(UnsupportedConstruct, envelope.EnvelopeError)


@pytest.mark.parametrize("name", ["op", "tran", ".tran", "TRAN", ".op"])
def test_transient_and_op_analyses_are_accepted(name: str) -> None:
    assert envelope.require(name).available


def test_dc_is_planned_but_not_yet_available() -> None:
    with pytest.raises(NotYetImplemented):
        envelope.require("dc")


@pytest.mark.parametrize("name", ["ac", "noise", "four", "tf"])
def test_ac_class_analyses_are_outside_the_fast_path(name: str) -> None:
    with pytest.raises(UnsupportedConstruct):
        envelope.require(name)


def test_missing_is_the_planned_set_only() -> None:
    missing = set(envelope.missing())
    assert missing == {"C:Q", "G:VALUE"}
    # .dc is planned as an analysis but is not an element tag.
    assert "dc" not in missing


def test_every_support_record_matches_its_key() -> None:
    for key, support in {**envelope.SUPPORTED, **envelope.ANALYSES}.items():
        assert support.tag == key
        assert support.description
        if support.available:
            assert not support.planned, f"{key} is available and planned"


def test_error_messages_name_the_construct() -> None:
    with pytest.raises(UnsupportedConstruct) as unsupported:
        envelope.check("K")
    assert "K" in str(unsupported.value)

    with pytest.raises(NotYetImplemented) as planned:
        envelope.check("G:VALUE")
    assert "G:VALUE" in str(planned.value)
