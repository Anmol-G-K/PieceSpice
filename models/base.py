"""Base class for semiconductor devices."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class SemiconductorDevice(BaseModel, ABC):
    """Abstract base class for semiconductor device models."""
    name: str
    subcircuit_name: str
    pin_mapping: Dict[str, int] = Field(default_factory=dict)
    parameters: Dict[str, float] = Field(default_factory=dict)
    source_file: Optional[str] = None
