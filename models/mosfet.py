"""Power MOSFET device data model."""

from typing import Optional
from models.base import SemiconductorDevice


class MOSFETDevice(SemiconductorDevice):
    """MOSFET device with extracted datasheet parameters."""
    drain_pin: str = "1"
    gate_pin: str = "2"
    source_pin: str = "3"

    v_th: float = 3.0
    rds_on: float = 0.010
    c_iss: Optional[float] = None
    c_oss: Optional[float] = None
    c_rss: Optional[float] = None
    q_g: Optional[float] = None
    q_oss: Optional[float] = None
    r_gate_internal: float = 1.5
    l_source_package: float = 1e-9
    l_drain_package: float = 1e-9
