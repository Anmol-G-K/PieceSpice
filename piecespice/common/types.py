"""Pydantic schemas and dataclasses for PieceSpice."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
import numpy as np


class CurveData(BaseModel):
    """1D or 2D Characteristic Curve container."""
    name: str
    x_name: str
    x_unit: str
    y_name: str
    y_unit: str
    x_data: List[float]
    y_data: List[float]
    family_param: Optional[str] = None
    family_value: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.x_data), np.array(self.y_data)


class WaveformData(BaseModel):
    """Time-domain transient waveform."""
    name: str
    time: List[float]
    values: List[float]
    unit: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.time), np.array(self.values)


class PWLModelSpec(BaseModel):
    """Piecewise linear segment specification."""
    device_name: str
    tier: int = 2
    breakpoints_vds: List[float]
    breakpoints_id: List[List[float]]
    vgs_values: List[float]
    charge_qoss_v: List[float]
    charge_qoss_q: List[float]
    rds_on: float
    v_th: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SimulationResult(BaseModel):
    """Container for simulation run outputs."""
    testbench: str
    success: bool
    wall_clock_seconds: float
    time_steps_count: int = 0
    iterations_count: int = 0
    waveforms: Dict[str, WaveformData] = Field(default_factory=dict)
    curves: Dict[str, CurveData] = Field(default_factory=dict)
    error_message: Optional[str] = None
