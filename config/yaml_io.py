"""YAML I/O utility for PieceSpice simulation configs.

Single interface between testbenches and the YAML config tree:

    config/inputs/<vendor>.yaml          - simulation conditions (source of truth)
    config/outputs/<vendor>/<tb>.yaml    - per-testbench results after each run

Testbenches call load_input()/save_output(); no YAML logic inline anywhere else.
Output files use consistent `extracted` key names so later milestones (verification,
macromodel fitting) can consume them uniformly.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Anchor on the project root (parent of the package dir) so config resolution
# works from any working directory (notebook kernels, scripts, tests).
CONFIG_ROOT = Path(__file__).resolve().parent.parent / "config"
INPUT_DIR = CONFIG_ROOT / "inputs"
OUTPUT_DIR = CONFIG_ROOT / "outputs"


def load_input(vendor_name: str) -> Dict[str, Any]:
    """Load config/inputs/<vendor_name>.yaml.

    Raises FileNotFoundError with a helpful message if missing.
    """
    path = INPUT_DIR / f"{vendor_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No input config for vendor '{vendor_name}' at {path}. "
            f"Available: {[p.stem for p in INPUT_DIR.glob('*.yaml')]}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def _to_builtin(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python for yaml.safe_dump."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def save_output(
    vendor_name: str,
    testbench_name: str,
    status: str,
    conditions_used: Dict[str, Any],
    extracted: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> Path:
    """Write config/outputs/<vendor_name>/<testbench_name>.yaml.

    Args:
        vendor_name: input-config stem (e.g. 'irfs7730pbf').
        testbench_name: output file stem (e.g. 'curve1_output_characteristics').
        status: 'pass' | 'partial' | 'blocked'.
        conditions_used: subset of the input conditions this run consumed.
        extracted: quantities produced by this testbench (None where unavailable).
        notes: free-form context (convergence quirks, vendor defects, etc.).

    Returns the written path.
    """
    if status not in ("pass", "partial", "blocked"):
        raise ValueError(f"status must be pass|partial|blocked, got {status!r}")
    path = OUTPUT_DIR / vendor_name / f"{testbench_name}.yaml"
    _write_yaml(path, _to_builtin({
        "device": vendor_name,
        "testbench": testbench_name,
        "status": status,
        "conditions_used": conditions_used,
        "extracted": extracted if extracted is not None else {},
        "notes": notes,
    }))
    return path


def load_output(vendor_name: str, testbench_name: str) -> Dict[str, Any]:
    """Read back a previously written output file (verification convenience)."""
    path = OUTPUT_DIR / vendor_name / f"{testbench_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No output for {vendor_name}/{testbench_name} at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Canonical extracted-quantity key names (shared across all output files).
# Testbenches must populate these keys so downstream milestones can rely on them.

KEYS = {
    # static
    "vth_v": "Gate threshold voltage @ Id=250uA [V]",
    "rds_on_10v_mohm": "On-resistance @ Vgs=10V, Id=10A [mOhm]",
    "rds_on_45v_mohm": "On-resistance @ Vgs=4.5V, Id=10A [mOhm]",
    "gfs_s": "Transconductance @ Vds=15V, Id=10A [S]",
    "body_diode_vsd_10a_v": "Body-diode forward voltage @ Isd=10A [V]",
    # capacitances
    "ciss_f": "Input capacitance [F]",
    "coss_f": "Output capacitance [F]",
    "crss_f": "Reverse-transfer capacitance [F]",
    # gate charge
    "v_plateau_v": "Miller plateau voltage [V]",
    "qgs_nc": "Gate-source charge [nC]",
    "qgd_nc": "Gate-drain charge [nC]",
    "q_tail_nc": "Post-plateau tail charge [nC]",
    "qg_total_nc": "Total gate charge at Vgs_max [nC]",
    # curves (full sampled data, kept as lists for portability)
    "curve_vgs": "Swept gate-source voltage [V]",
    "curve_vds": "Swept drain-source voltage [V]",
    "curve_id": "Drain current [A]",
    "curve_isd": "Source (body-diode) current [A]",
    "curve_qg_nc": "Gate charge [nC]",
}
