"""Stage-6 measurement harness: three arms, one bench, one record."""

from __future__ import annotations

from engine.harness.comparison import (
    SCHEMA,
    Arm,
    summarise,
    validate_vendor,
)

__all__ = ["SCHEMA", "Arm", "summarise", "validate_vendor"]
