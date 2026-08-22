"""Generates executable SPICE subcircuits from PieceSpice piecewise specifications."""

from common.types import PWLModelSpec, PWL_META_KEYS

# Documented fallbacks for optional metadata keys the producer has not yet
# populated. Missing keys never crash the builder; unknown keys are reported.
_META_DEFAULTS = {
    "r_gate": 1.5,
    "r_source": 1e-4,
    "r_drain": 1e-4,
    "c_iss_lin": 1500e-12,
    "c_oss_lin": 500e-12,
    "c_rss_lin": 50e-12,
}


def _resolve_metadata(spec: PWLModelSpec) -> dict:
    """Merge spec metadata over documented defaults; warn on unknown keys."""
    meta = dict(_META_DEFAULTS)
    for k, v in spec.metadata.items():
        if k in _META_DEFAULTS:
            meta[k] = v
        else:
            print(
                f"[netlist_builder] WARNING: unknown metadata key {k!r} for "
                f"{spec.device_name} (known keys: {sorted(_META_DEFAULTS)}); ignored."
            )
    missing = [k for k in _META_DEFAULTS if k not in spec.metadata]
    if missing:
        print(
            f"[netlist_builder] NOTE: {spec.device_name} metadata missing "
            f"{missing}; using defaults {({k: _META_DEFAULTS[k] for k in missing})}."
        )
    return meta


class NetlistBuilder:
    """Emits SPICE subcircuit code for Tier 1, 2, or 3 models."""

    @staticmethod
    def build_tier2_subcircuit(spec: PWLModelSpec) -> str:
        """Build Tier 2 Continuous Piecewise Linear (CPWL) MOSFET subcircuit."""
        meta = _resolve_metadata(spec)
        subckt_name = f"{spec.device_name}_PIECESPICE_T2"
        lines = [
            f"* PieceSpice Tier 2 CPWL Model for {spec.device_name}",
            f".SUBCKT {subckt_name} 1 2 3",
            f"* Pin definition: 1=Drain, 2=Gate, 3=Source",
            f"RG 2 20 {meta['r_gate']}",
            f"RS 30 3 {meta['r_source']}",
            f"RD 1 10 {meta['r_drain']}",
            "",
            "* Voltage-controlled channel conduction source",
            f"B_CH 10 30 I = max(0, (V(20,30) - {spec.v_th}) / max({spec.rds_on}, 1e-4) * tanh(max(0, V(10,30)) / 0.5))",
            "",
            "* Output capacitance charge source (Qoss)",
            f"C_OSS 10 30 {meta['c_oss_lin']}",
            f"C_ISS 20 30 {meta['c_iss_lin']}",
            f"C_RSS 20 10 {meta['c_rss_lin']}",
            "",
            "* Body diode",
            f"D_BODY 30 10 D_BD",
            f".MODEL D_BD D IS=1e-10 RS=0.002 N=1.2",
            f".ENDS {subckt_name}",
        ]
        return '\n'.join(lines)
