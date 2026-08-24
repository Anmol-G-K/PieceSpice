"""Pydantic schemas and dataclasses for PieceSpice."""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field, model_validator
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

    @model_validator(mode="after")
    def _check_lengths(self) -> "CurveData":
        if len(self.x_data) != len(self.y_data):
            raise ValueError(
                f"{self.name!r}: x_data ({len(self.x_data)}) and "
                f"y_data ({len(self.y_data)}) lengths differ"
            )
        return self

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
    """Piecewise linear segment specification.

    The ``metadata`` dict carries the optional per-device parameters consumed by
    ``models.netlist_builder`` when emitting a Tier 2 subcircuit. Producers (the
    future M3 fitting pipeline) must populate keys from ``PWL_META_KEYS``;
    the builder validates against it and falls back to documented defaults for
    missing optional keys rather than failing.
    """
    device_name: str
    tier: int = 2
    breakpoints_vds: List[float]
    breakpoints_id: List[List[float]]
    vgs_values: List[float]
    # Canonical names (M17); 'charge_qoss_v/q' remain accepted aliases.
    q_oss_vs_v: List[float] = Field(default_factory=list)
    q_oss_charge_c: List[float] = Field(default_factory=list)
    charge_qoss_v: List[float] = Field(default_factory=list)
    charge_qoss_q: List[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _merge_legacy_charge_names(self) -> "PWLModelSpec":
        """Fold legacy field names into the canonical ones."""
        if not self.q_oss_vs_v and self.charge_qoss_v:
            self.q_oss_vs_v = self.charge_qoss_v
        if not self.q_oss_charge_c and self.charge_qoss_q:
            self.q_oss_charge_c = self.charge_qoss_q
        return self
    rds_on: float
    v_th: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Canonical metadata key contract between the M3 fitting pipeline (producer)
# and models.netlist_builder (consumer). Mirrors config.yaml_io.KEYS.
PWL_META_KEYS: Dict[str, str] = {
    "r_gate": "Internal gate resistance [Ohm]",
    "r_source": "Source parasitic resistance [Ohm]",
    "r_drain": "Drain parasitic resistance [Ohm]",
    "c_iss_lin": "Linear equivalent input capacitance [F]",
    "c_oss_lin": "Linear equivalent output capacitance [F]",
    "c_rss_lin": "Linear equivalent reverse-transfer capacitance [F]",
}


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


def trim_trailing_invalid(x_arr, y_arr):
    """Trim trailing non-finite/non-positive samples from a curve.

    Shared post-processing (Minor m15): capacitance-style benches mask
    degenerate points as NaN; this trims the masked tail so interpolators
    see a clean envelope. Returns (x, y) numpy arrays.
    """
    import numpy as np
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    valid = np.where(np.isfinite(y) & (y > 0))[0]
    if len(valid) == 0:
        return x, y
    k = valid[-1] + 1
    return x[:k], y[:k]
