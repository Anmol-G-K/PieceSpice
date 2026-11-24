"""PWLModelSpec: the producer/consumer contract for Tier-2 model synthesis.

Extended for M3.5: carries everything netlist_builder needs to synthesize a
complete MOSFET subcircuit - channel PWL families, charge tables, body-diode
points, and parasitic/metadata parameters. All fields device-agnostic.

Unified-ROM extension: ``metadata`` additionally carries the additive keys in
``UNIFIED_META_KEYS`` (spec 4.6) - provenance, C1 epsilon, tuning theta, ANN
fit record and speed-row completion counts. Absent keys are never injected.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

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
        if self.metadata:
            _normalize_unified_metadata(self.metadata, self.device_name)

        return self

    def channel_slope_at(self, vgs: float) -> float:
        """Interpolated saturation-region slope (gfs estimate) at a Vgs."""
        if (len(self.breakpoints_vds) < 2 or not self.vgs_values
                or not self.breakpoints_id):
            return 0.0
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


# ---------------------------------------------------------------------------
# Unified-ROM metadata extension (spec 4.6) - additive only.
#
# ``PWL_META_KEYS`` above remains the Tier-2 scalar contract: same names,
# same semantics. Nothing below injects keys into ``metadata``; a spec built
# by today's fitting code produces exactly the same dict as before. Callers
# opting into the unified ROM declare these keys explicitly, and the model
# validator only checks/normalises what is present.
# ---------------------------------------------------------------------------

#: Fixed-topology tuning scalars (spec 4.6): A_GD/A_GS/A_RGI are
#: dimensionless multipliers; O_GD is an additive charge offset [F].
UNIFIED_TUNING_KEYS: Tuple[str, ...] = ("A_GD", "O_GD", "A_GS", "A_RGI")

#: Identity of the WOL calibration transform (A_* = 1, O_GD = 0 F). This is
#: the mathematical no-op of Eqs. 2-4, NOT a fitted value and NOT any of the
#: Wolfspeed Table 3 final numbers. It is the only neutral starting point a
#: producer may write into metadata before a calibration has run.
UNIFIED_TUNING_IDENTITY: Dict[str, float] = {
    "A_GD": 1.0, "O_GD": 0.0, "A_GS": 1.0, "A_RGI": 1.0,
}

#: Provenance for every number introduced by the fixed-topology layer
#: (spec 4.3 / 5). ``hint_*`` entries are source values used only as scale
#: hints for brackets; they are never emitter defaults. ``project_decision``
#: entries carry their rationale inline so no constant is untraceable.
UNIFIED_TUNING_PROVENANCE: Dict[str, str] = {
    "transform": "WOL Eqs.2-4: Cgd_eff=A_GD*(Cgd+O_GD), "
                 "Cgs_eff=A_GS*Cgs, Rgi_eff=A_RGI*Rgi; O_GD >= 0 here and "
                 "applied as O_GD*V on Q_RSS(V) (zero at V=0).",
    "source": "DeBoi/Nelson/Curbow (Wolfspeed), Accuracy Evaluation and "
              "Proposed Dynamic Tuning Procedure of a Compact SiC SPICE "
              "Model, https://forum.wolfspeed.com/uploads/VQL8VYDXXYC8/"
              "ws-spice-dynamic-tuning.pdf, §2.1 Eq.1, §3.1 Eqs.2-4/Table 2, "
              "§3.2 Table 3; vendor bench, hardware reference.",
    "hint_A_GD": "WOL §3.2 Table 3 final A_GD=1.47 (scale hint only)",
    "hint_O_GD": "WOL §3.2 Table 3 final O_GD=5.8 pF (scale hint only)",
    "hint_A_GS": "WOL §3.2 Table 3 final A_GS=0.68 (scale hint only)",
    "hint_A_RGI": "WOL §3.2 Table 3 final A_RGI=0.84 (scale hint only)",
    "identity": "Mathematical identity of the transform, not a WOL value.",
    "multiplier_bounds": "project_decision: one-octave [0.5, 2.0] bracket "
                         "about identity; brackets all four WOL hints, "
                         "keeps A_* strictly positive and identifiable.",
    "offset_bound": "source data: Q_RSS(Vcap_max) from curve4 c_eq_energy "
                    "and config/inputs/<vendor>.yaml "
                    "conditions.Vds_cap_max; project_decision: cap at one "
                    "full Q_RSS at the sweep top so monotonicity and the "
                    "charge budget are preserved.",
    "frozen": "project_decision: C_DS/Qoss, channel breakpoints, rds_on, "
              "v_th, body diode and element count are frozen; WOL §3.1 "
              "omits C_DS because it over-predicts reverse recovery.",
}

#: ANN IV/QV fit record fields (spec 4.6).
UNIFIED_ANN_FIT_KEYS: Dict[str, str] = {
    "activation": "Hidden-layer activation name ('isru' default candidate)",
    "neurons": "Hidden-layer neuron count (<= 10 in v1)",
    "loss_weights": "Named derivative-aware loss weights",
    "gm_gds_q_derivative_terms": "Derivative terms used in the IV loss",
    "train_range": "Bias range used for training [V]",
    "q0_offset": "Charge Q0 offset [C]; null when not used",
    "out_of_range_check": "'pass' or 'fail'",
}

#: Hard cap from spec 2.3 / 8 R7: one hidden layer, <= 10 neurons. The
#: 16-neuron configuration is the known E3 Fig. 12 spurious-current case.
UNIFIED_ANN_MAX_NEURONS: int = 10

#: Declared solver method labels (mirrors ``pipeline.unified_rom.METHODS``).
#: Only these labels may appear in unified metadata or a speed row.
UNIFIED_METHOD_LABELS: Tuple[str, ...] = ("trap/default", "gear-both")

#: Completion counts every printed speed row must carry (spec 4.6 / G0).
UNIFIED_SPEED_ROW_KEYS: Tuple[str, ...] = (
    "n_pulses", "max_step", "method_label", "t_last", "iterations", "points")
#: ``max_step`` may legitimately be null; the rest must carry real values.
UNIFIED_SPEED_ROW_REQUIRED: Tuple[str, ...] = (
    "n_pulses", "method_label", "t_last", "iterations", "points")

#: Recognised unified metadata keys. ``urom_*`` names are the spec 4.6
#: metadata spellings; the unprefixed names are the unified validation-record
#: spellings. Both are accepted; neither is injected by default.
UNIFIED_META_KEYS: Dict[str, str] = {
    "urom_channel_basis": "Channel basis: 'ann' or 'tier2'",
    "urom_c1_eps": "C1 smoothing width [V]; 0.0 = legacy C0 ramp",
    "urom_tuning_theta": "Tuning theta (UNIFIED_TUNING_KEYS)",
    "urom_method_label": "Solver method label: 'trap/default' or 'gear-both'",
    "urom_provenance": "Source/config/doc locator per constant",
    "unified_provenance": "Unified-record spelling of urom_provenance",
    "channel_basis": "Unified-record spelling of urom_channel_basis",
    "c1_eps": "Unified-record spelling of urom_c1_eps [V]",
    "method": "Unified-record solver method label",
    "method_label": "Speed-row solver method label",
    "theta": "Unified-record spelling of urom_tuning_theta",
    "theta_bounds": "Bounds for UNIFIED_TUNING_KEYS scalars",
    "ann_fit": "ANN IV/QV fit record (UNIFIED_ANN_FIT_KEYS)",
    "external_source_status": "External-source availability, e.g. "
                              "'conditional_until_url_and_section_pinned'",
    "c1": "C1 record: eps_by_table, eps_rule, c0_bit_identical_at_eps0, "
          "breakpoint_crossings",
    "tuning": "Tuning record: theta, bounds, weights, conditions, verdict",
    "runtime": "Runtime record: iterations, points, t_last, repeats, times",
    "data_provenance": "Curve/DPT provenance; carries external_source_status",
    "speed_adaptive_20": "Speed row (UNIFIED_SPEED_ROW_KEYS)",
    "speed_rows": "Named speed rows; each validated as a speed row",
}


def _normalize_unified_metadata(meta: Dict[str, Any], device: str) -> None:
    """Additively validate the unified keys present in ``meta``.

    Units/edge cases: ``c1_eps``/``urom_c1_eps`` are smoothing widths [V],
    finite and >= 0 (0 = the closed C0 emitter); theta values are scalars
    (O_GD in farads) or null; provenance/fit/runtime/c1/tuning/speed rows are
    mappings. A speed row must contain every ``UNIFIED_SPEED_ROW_KEYS`` key
    and real (non-null) completion counts for ``UNIFIED_SPEED_ROW_REQUIRED``:
    a row missing iterations/points/t_last/method label is a fabricated speed
    row and is rejected. Absent keys stay absent (no defaults injected), so
    existing Tier-2 callers keep byte-identical metadata behaviour. No NaN is
    accepted for any numeric field.
    """
    def fail(msg: str) -> None:
        raise ValueError(f"{device}: unified metadata: {msg}")

    for key in ("c1_eps", "urom_c1_eps"):
        if meta.get(key) is not None:
            try:
                value = float(meta[key])
            except (TypeError, ValueError):
                value = float("nan")
            if not math.isfinite(value) or value < 0.0:
                fail(f"{key} must be a finite number >= 0")
    if (meta.get("c1_eps") is not None and meta.get("urom_c1_eps") is not None
            and float(meta["c1_eps"]) != float(meta["urom_c1_eps"])):
        fail("c1_eps and urom_c1_eps disagree")

    for key in ("theta", "urom_tuning_theta", "theta_bounds", "c1", "tuning",
                "runtime", "data_provenance", "unified_provenance",
                "urom_provenance"):
        if meta.get(key) is not None and not isinstance(meta[key], dict):
            fail(f"{key} must be a mapping")

    # Fixed-topology theta (T3): named fields only, in UNIFIED_TUNING_KEYS
    # order, finite scalars or null. A positional tuple is rejected: the
    # four knobs are not interchangeable and a mis-slot is a silent defect.
    theta = meta.get("theta")
    if theta is None:
        theta = meta.get("urom_tuning_theta")
    if theta is not None:
        unknown = [k for k in theta if k not in UNIFIED_TUNING_KEYS]
        if unknown:
            fail(f"theta has unknown fields {unknown}; expected "
                 f"{list(UNIFIED_TUNING_KEYS)}")
        for key, value in theta.items():
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                fail(f"theta[{key!r}] must be a finite number or null")
            if not math.isfinite(number):
                fail(f"theta[{key!r}] must be a finite number or null")
            if key == "O_GD" and number < 0.0:
                fail("theta[O_GD] must be >= 0 (monotone Q_RSS offset)")

    bounds = meta.get("theta_bounds")
    if bounds is not None:
        for key, pair in bounds.items():
            if key not in UNIFIED_TUNING_KEYS:
                fail(f"theta_bounds has unknown field {key!r}")
            if (isinstance(pair, (str, bytes)) or not isinstance(pair, (list, tuple))
                    or len(pair) != 2):
                fail(f"theta_bounds[{key!r}] must be a (lo, hi) pair")
            lo, hi = pair
            try:
                lo, hi = float(lo), float(hi)
            except (TypeError, ValueError):
                fail(f"theta_bounds[{key!r}] must be numeric")
            if not (math.isfinite(lo) and math.isfinite(hi)
                    and lo < hi):
                fail(f"theta_bounds[{key!r}] must be finite with lo < hi")
            if key == "O_GD" and lo < 0.0:
                fail("theta_bounds[O_GD] lower bound must be >= 0")
        if theta is not None:
            for key, value in theta.items():
                if value is None or key not in bounds:
                    continue
                lo, hi = (float(v) for v in bounds[key])
                if not (lo - 1e-12 <= float(value) <= hi + 1e-12):
                    fail(f"theta[{key!r}] outside theta_bounds[{key!r}]")

    if meta.get("external_source_status") is not None:
        status = meta["external_source_status"]
        if not isinstance(status, str) or not status.strip():
            fail("external_source_status must be a non-empty string")

    for key in ("urom_method_label", "method", "method_label"):
        value = meta.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or value not in UNIFIED_METHOD_LABELS:
            fail(f"{key} must be one of {list(UNIFIED_METHOD_LABELS)}; "
                 f"got {value!r}")

    fit = meta.get("ann_fit")
    if fit is not None:
        if not isinstance(fit, dict):
            fail("ann_fit must be a mapping")
        if "neurons" in fit and (
                isinstance(fit["neurons"], bool)
                or not isinstance(fit["neurons"], int)
                or fit["neurons"] <= 0
                or fit["neurons"] > UNIFIED_ANN_MAX_NEURONS):
            fail(f"ann_fit.neurons must be a positive integer <= "
                 f"{UNIFIED_ANN_MAX_NEURONS}")
        if "activation" in fit and not isinstance(fit["activation"], str):
            fail("ann_fit.activation must be a string")
        if fit.get("q0_offset") is not None:
            try:
                q0 = float(fit["q0_offset"])
            except (TypeError, ValueError):
                q0 = float("nan")
            if not math.isfinite(q0):
                fail("ann_fit.q0_offset must be a finite number or null")
        if "out_of_range_check" in fit and fit["out_of_range_check"] not in (
                "pass", "fail"):
            fail("ann_fit.out_of_range_check must be 'pass' or 'fail'")

    rows = []
    if "speed_adaptive_20" in meta:
        rows.append(("speed_adaptive_20", meta["speed_adaptive_20"]))
    if "speed_rows" in meta:
        named = meta["speed_rows"]
        if not isinstance(named, dict):
            fail("speed_rows must be a mapping of row name -> speed row")
        rows.extend((f"speed_rows[{name!r}]", row)
                    for name, row in named.items())
    for label, row in rows:
        if row is None:
            continue
        if not isinstance(row, dict):
            fail(f"{label} must be a mapping")
        missing = [k for k in UNIFIED_SPEED_ROW_KEYS if k not in row]
        if missing:
            fail(f"{label} missing speed-row keys {missing}")
        empty = [k for k in UNIFIED_SPEED_ROW_REQUIRED if row[k] is None]
        if empty:
            fail(f"{label} has null completion counts {empty}")
        row_method = row.get("method_label")
        if row_method not in UNIFIED_METHOD_LABELS:
            fail(f"{label} has undeclared method_label {row_method!r}; "
                 f"expected {list(UNIFIED_METHOD_LABELS)}")


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
