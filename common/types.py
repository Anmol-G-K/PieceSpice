"""PWLModelSpec: the producer/consumer contract for Tier-2 model synthesis.

Extended for M3.5: carries everything netlist_builder needs to synthesize a
complete MOSFET subcircuit - channel PWL families, charge tables, body-diode
points, and parasitic/metadata parameters. All fields device-agnostic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field, model_validator


class PWLModelSpec(BaseModel):
    """Piecewise-linear model specification (producer -> builder contract)."""

    device_name: str
    tier: int = 2

    # ---- channel: Id(Vds) family -----------------------------------------
    # breakpoints_vds: ascending Vds breakpoints shared by all Vgs curves.
    # breakpoints_id[k] = Id samples at Vgs vgs_values[k], evaluated on the
    # same breakpoint grid (len == len(breakpoints_vds)).
    breakpoints_vds: List[float]
    breakpoints_id: List[List[float]]
    vgs_values: List[float]

    # ---- charge-conserving capacitances ----------------------------------
    # q_oss_vs_v / q_oss_charge_c: monotonic Qoss(V) table [V / C].
    q_oss_vs_v: List[float] = Field(default_factory=list)
    q_oss_charge_c: List[float] = Field(default_factory=list)
    # Qiss and Qrss tables (same form) - added M3.5; optional.
    q_iss_vs_v: List[float] = Field(default_factory=list)
    q_iss_charge_c: List[float] = Field(default_factory=list)
    q_rss_vs_v: List[float] = Field(default_factory=list)
    q_rss_charge_c: List[float] = Field(default_factory=list)

    # legacy aliases folded into canonical fields
    charge_qoss_v: List[float] = Field(default_factory=list)
    charge_qoss_q: List[float] = Field(default_factory=list)

    # ---- static scalars ----------------------------------------------------
    rds_on: float                    # on-resistance [Ohm] at rated drive
    v_th: float                      # threshold voltage [V]
    # body diode (Curve 6): Isd vs Vsd sample points
    body_diode_vsd: List[float] = Field(default_factory=list)
    body_diode_isd: List[float] = Field(default_factory=list)
    # gate-charge plateau info (Curve 5), used for reporting/Qgd checks
    v_plateau: Optional[float] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "PWLModelSpec":
        """Fold legacy aliases + validate structural consistency."""
        if not self.q_oss_vs_v and self.charge_qoss_v:
            self.q_oss_vs_v = self.charge_qoss_v
        if not self.q_oss_charge_c and self.charge_qoss_q:
            self.q_oss_charge_c = self.charge_qoss_q

        if len(self.breakpoints_id) != len(self.vgs_values):
            raise ValueError(
                f"breakpoints_id rows ({len(self.breakpoints_id)}) must "
                f"match vgs_values ({len(self.vgs_values)})")
        w = len(self.breakpoints_vds)
        for i, row in enumerate(self.breakpoints_id):
            if len(row) != w:
                raise ValueError(
                    f"breakpoints_id[{i}] has {len(row)} points, expected {w}")
        if any(b <= a for a, b in zip(self.breakpoints_vds,
                                      self.breakpoints_vds[1:])):
            raise ValueError("breakpoints_vds must be strictly ascending")
        return self

    def channel_slope_at(self, vgs: float) -> float:
        """Interpolated saturation-region slope (gfs estimate) at a Vgs."""
        idx = np.clip(np.searchsorted(self.vgs_values, vgs),
                      0, len(self.vgs_values) - 1)
        row = np.asarray(self.breakpoints_id[idx], dtype=float)
        vd = np.asarray(self.breakpoints_vds, dtype=float)
        # slope over the last 30% of Vds range = saturated region estimate
        lo = max(1, int(len(vd) * 0.7))
        span = vd[-1] - vd[lo]
        return float((row[-1] - row[lo]) / span) if span > 0 else 0.0


# Canonical metadata key contract between the fitting pipeline (producer)
# and models.netlist_builder (consumer). Mirrors config.yaml_io.KEYS.
PWL_META_KEYS: Dict[str, str] = {
    "r_gate": "Internal gate resistance [Ohm]",
    "r_source": "Source parasitic resistance [Ohm]",
    "r_drain": "Drain parasitic resistance [Ohm]",
    "c_iss_lin": "Linear equivalent input capacitance [F]",
    "c_oss_lin": "Linear equivalent output capacitance [F]",
    "c_rss_lin": "Linear equivalent reverse-transfer capacitance [F]",
}


def trim_trailing_invalid(x_arr, y_arr):
    """Trim trailing non-finite/non-positive samples from a curve.

    Shared post-processing (Minor m15): capacitance-style benches mask
    degenerate points as NaN; this trims the masked tail so interpolators
    see a clean envelope. Returns (x, y) numpy arrays.
    """
    x = np.asarray(x_arr, dtype=float)
    y = np.asarray(y_arr, dtype=float)
    valid = np.where(np.isfinite(y) & (y > 0))[0]
    if len(valid) == 0:
        return x, y
    k = valid[-1] + 1
    return x[:k], y[:k]


# ---------------------------------------------------------------------------
# Core curve/waveform containers (restored - used across the entire pipeline)
# ---------------------------------------------------------------------------

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
                f"y_data ({len(self.y_data)}) lengths differ")
        return self

    def to_numpy(self) -> tuple:
        return np.array(self.x_data), np.array(self.y_data)


class WaveformData(BaseModel):
    """Time-domain transient waveform."""
    name: str
    time: List[float]
    values: List[float]
    unit: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_numpy(self) -> tuple:
        return np.array(self.time), np.array(self.values)


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
