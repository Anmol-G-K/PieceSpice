"""Tier registry for the PieceSpice CLI.

Single source of truth for:

* the selectable solver tiers (2 = CPWL, 3 = TPWL + POD-DEIM,
  4 = ideal switch + measured loss LUT);
* the Tier-3 variants registered per vendor (default ``v1``);
* the emitted library filename for a (vendor, tier, variant) selection;
* reading recorded golden / Tier-2 / Tier-3 DPT metrics for ``compare``;
* reading recorded Tier-4 energy/efficiency and speed rows for ``compare``.

Tier-2 has one model form and no variants.  Tier-3 keeps variants per vendor:
the default variant emits ``<vendor>_PIECESPICE_T3.lib`` and every other
variant emits ``<vendor>_PIECESPICE_T3_<variant>.lib``.  Tier-4 is variant-free
and emits ``<vendor>_PIECESPICE_T4.lib``.  Its gate is LUT energy versus the
recorded golden DPT crossing plus inductive-bench speed; the DPT waveform
gate is retired by design and no converter bench has been built (M5).
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_VARIANT = "v1"
TIERS: tuple[int, ...] = (2, 3, 4)
VENDORS: tuple[str, ...] = (
    "irfs7730pbf", "csd16412q5a", "epc2216", "cpm2-1200",
)

# Registered Tier-3 variants per vendor, default first.  "v1" is the current
# validated terminal TPWL/POD-DEIM form; "simplis" is the SIMPLIS-style
# Level-2 (V,Q) charge-table variant, added behind the default.
VARIANTS: dict[str, tuple[str, ...]] = {
    vendor: (DEFAULT_VARIANT, "simplis") for vendor in VENDORS
}

DPT_METRICS: tuple[str, ...] = (
    "td_on_ns", "tr_ns", "td_off_ns", "tf_ns",
    "dv_dt_max_v_per_ns", "di_dt_max_a_per_ns",
    "e_on_uj", "e_off_uj", "i_peak_a",
)

ROOT = Path(__file__).resolve().parent.parent

# Shared-source disclosure for the Tier-4 energy rows (audit D1).  The LUT is
# filled from the recorded golden DPT crossing, so a zero error at that
# crossing is by construction; no independent converter-energy gate exists.
T4_ENERGY_DISCLOSURE = (
    "LUT reproduces its own golden DPT crossing; not an independent "
    "converter validation")


def variants_for(vendor: str) -> tuple[str, ...]:
    """Known Tier-3 variants for *vendor*, default first."""
    return VARIANTS.get(vendor, (DEFAULT_VARIANT,))


def tier3_lib_name(vendor: str, variant: str = DEFAULT_VARIANT) -> str:
    """Emitted Tier-3 library filename for *variant*."""
    suffix = "" if variant == DEFAULT_VARIANT else f"_{variant}"
    return f"{vendor}_PIECESPICE_T3{suffix}.lib"


def tier4_lib_name(vendor: str) -> str:
    """Emitted Tier-4 library filename (variant-free by design)."""
    return f"{vendor}_PIECESPICE_T4.lib"


def lib_path(vendor: str, tier: int,
             variant: str = DEFAULT_VARIANT) -> Path:
    """Expected emitted library path for a tier/variant selection."""
    name = (f"{vendor}_PIECESPICE_T2.lib" if tier == 2
            else tier4_lib_name(vendor) if tier == 4
            else tier3_lib_name(vendor, variant))
    return ROOT / "reports" / "models" / name


def fit(vendor: str, tier: int = 2,
        variant: str = DEFAULT_VARIANT) -> Path:
    """Fit the selected tier and return the emitted library path.

    Tier-2 calls the closed ``pipeline.fitting.fit_pwl_model`` unchanged.
    Tier-3 calls ``pipeline.tier3.run_vendor`` (build + validation).
    Tier-4 calls ``pipeline.tier4.run_vendor`` (ideal switch + loss LUT,
    variant-free; DPT waveform gate retired by design).
    """
    if tier == 4:
        from pipeline.tier4 import run_vendor
        return Path(run_vendor(vendor)["emitted_lib"])
    if tier == 3:
        if variant == "simplis":
            from pipeline.tier3_simplis import run_vendor as run_simplis
            return Path(run_simplis(vendor)["emitted_lib"])
        from pipeline.tier3 import run_vendor
        return Path(run_vendor(vendor, variant=variant)["emitted_lib"])
    raise ValueError(f"unknown tier {tier}; expected one of {TIERS}")


def validate(vendor: str, tier: int = 2,
             variant: str = DEFAULT_VARIANT):
    """Validate the selected tier.

    Tier-2: endpoint-current comparison of its emitted lib against the
    extracted golden curves.  Tier-3: the recorded build + DPT run, with its
    anti-fake completion guard, returned as the measured record.
    Tier-4: the ideal-switch/LUT build + DPT-derived energy and inductive-bench
    speed record, with the same anti-fake completion guard; DPT waveform gate
    retired by design and no converter bench built.
    """
    if tier == 4:
        from pipeline.tier4 import run_vendor
        return run_vendor(vendor)
    if tier == 2:
        from pipeline.fitting import compare_tier2_vs_golden
        return compare_tier2_vs_golden(vendor)
    if tier == 3:
        if variant == "simplis":
            from pipeline.tier3_simplis import run_vendor as run_simplis
            return run_simplis(vendor)
        from pipeline.tier3 import run_vendor
        return run_vendor(vendor, variant=variant)


def compare_rows(vendor: str) -> list[tuple[str, float | None,
                                            float | None, float | None]]:
    """Recorded DPT metrics as ``(metric, golden, tier2, tier3)`` rows.

    Numbers come only from existing JSON reports.  A missing tier yields
    ``None`` so the caller can render "N/A" rather than fabricate a value.
    """
    base = ROOT / "reports" / "comparison" / vendor
    golden: dict = {}
    tier2: dict = {}
    dpt_path = base / "dpt_metrics.json"
    if dpt_path.exists():
        data = json.loads(dpt_path.read_text(encoding="utf-8"))
        golden = data.get("golden_metrics") or {}
        tier2 = data.get("tier2_metrics") or {}
    tier3: dict = {}
    t3_path = base / "tier3_validation.json"
    if t3_path.exists():
        data = json.loads(t3_path.read_text(encoding="utf-8"))
        for metric, row in (data.get("dpt_comparison") or {}).items():
            tier3[metric] = row.get("tier3")
    keys = [m for m in DPT_METRICS
            if m in golden or m in tier2 or m in tier3]
    return [(m, golden.get(m), tier2.get(m), tier3.get(m)) for m in keys]


def _t4_text(value) -> str:
    """Render a recorded value, or N/A when it was never measured."""
    if value is None or (isinstance(value, float) and value != value):
        return "N/A"
    return str(value)


def tier4_rows(vendor: str) -> list[tuple[str, float | None,
                                        float | None, str]]:
    """Tier-4 energy/efficiency and speed rows for ``compare``.

    Only the two loss metrics and the two guarded speed ratios are reported:
    Tier-4 deliberately retires the DPT waveform gate.  Every energy row
    carries the shared-source disclosure: the LUT was filled from the
    recorded golden DPT crossing, so a zero relative error there is by
    construction, not an independent converter-energy gate.  Numbers come
    only from ``tier4_validation.json``; an unrecorded golden yields ``None``
    so the CLI renders N/A instead of 0 or "fast".
    """
    path = ROOT / "reports" / "comparison" / vendor / "tier4_validation.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    energy = data.get("energy_comparison") or {}
    rows: list[tuple[str, float | None, float | None, str]] = []
    for metric in ("e_on_uj", "e_off_uj"):
        row = energy.get(metric) or {}
        note = str(row.get("disclosure") or T4_ENERGY_DISCLOSURE)
        if row.get("kind"):
            note = f"[{row['kind']}] {note}"
        rows.append((
            metric,
            row.get("golden"),
            row.get("tier4"),
            f"abs={_t4_text(row.get('abs_error'))} uJ; "
            f"rel={_t4_text(row.get('rel_error_pct'))}%; {note}",
        ))
    speed = data.get("speedup") or {}
    t_last = speed.get("t_last_s")
    # Only a measured trace endpoint may be called t_last.  The pipeline's
    # end_time_s is the configured .tran stop schedule and is labelled as such.
    end_label = "t_last" if t_last is not None else ".tran stop schedule"
    end_time = t_last if t_last is not None else speed.get("end_time_s")
    # Both time-based ratios are gated by the same measured arms; attach the
    # golden/Tier-4 iterations and trace points to the wall row as well (R3).
    run_counts = (
        f"iters {_t4_text(speed.get('golden_iterations'))}/"
        f"{_t4_text(speed.get('tier4_iterations'))}; "
        f"points {_t4_text(speed.get('golden_points'))}/"
        f"{_t4_text(speed.get('tier4_points'))}"
    )
    counts = (
        f"status={_t4_text(speed.get('status'))}; {run_counts}; "
        f"{end_label}={_t4_text(end_time)} s"
    )
    rows.append(("solver_speedup", None, speed.get("solver_speedup"), counts))
    rows.append(("wall_speedup", None, speed.get("wall_speedup"), run_counts))
    return rows
