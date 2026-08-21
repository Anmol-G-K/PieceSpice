"""Generates executable SPICE subcircuits from PieceSpice piecewise specifications."""

from piecespice.common.types import PWLModelSpec


class NetlistBuilder:
    """Emits SPICE subcircuit code for Tier 1, 2, or 3 models."""

    @staticmethod
    def build_tier2_subcircuit(spec: PWLModelSpec) -> str:
        """Build Tier 2 Continuous Piecewise Linear (CPWL) MOSFET subcircuit."""
        subckt_name = f"{spec.device_name}_PIECESPICE_T2"
        lines = [
            f"* PieceSpice Tier 2 CPWL Model for {spec.device_name}",
            f".SUBCKT {subckt_name} 1 2 3",
            f"* Pin definition: 1=Drain, 2=Gate, 3=Source",
            f"RG 2 20 {spec.metadata.get('r_gate', 1.5)}",
            f"RS 30 3 {spec.metadata.get('r_source', 1e-4)}",
            f"RD 1 10 {spec.metadata.get('r_drain', 1e-4)}",
            "",
            "* Voltage-controlled channel conduction source",
            f"B_CH 10 30 I = max(0, (V(20,30) - {spec.v_th}) / max({spec.rds_on}, 1e-4) * tanh(max(0, V(10,30)) / 0.5))",
            "",
            "* Output capacitance charge source (Qoss)",
            f"C_OSS 10 30 {spec.metadata.get('c_oss_lin', 500e-12)}",
            f"C_ISS 20 30 {spec.metadata.get('c_iss_lin', 1500e-12)}",
            f"C_RSS 20 10 {spec.metadata.get('c_rss_lin', 50e-12)}",
            "",
            "* Body diode",
            f"D_BODY 30 10 D_BD",
            f".MODEL D_BD D IS=1e-10 RS=0.002 N=1.2",
            f".ENDS {subckt_name}",
        ]
        return '\n'.join(lines)
