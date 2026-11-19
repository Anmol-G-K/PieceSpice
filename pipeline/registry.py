"""Tier registry for the PieceSpice CLI.

Single source of truth for:

* the selectable solver tiers (2 = CPWL, 3 = TPWL + POD-DEIM);
* the Tier-3 variants registered per vendor (default ``v1``);
* the emitted library filename for a (vendor, tier, variant) selection;
* reading recorded golden / Tier-2 / Tier-3 DPT metrics for ``compare``.

Tier-2 has one model form and no variants.  Tier-3 keeps variants per vendor:
the default variant emits ``<vendor>_PIECESPICE_T3.lib`` and every other
variant emits ``<vendor>_PIECESPICE_T3_<variant>.lib``.
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_VARIANT = "v1"
TIERS: tuple[int, ...] = (2, 3)
VENDORS: tuple[str, ...] = (
    "irfs7730pbf", "csd16412q5a", "epc2216", "cpm2-1200",
)

# Registered Tier-3 variants per vendor, default first.  "v1" is the current
# validated terminal TPWL/POD-DEIM form; further validated variants are
# appended here (they emit the suffixed library name).
VARIANTS: dict[str, tuple[str, ...]] = {
    vendor: (DEFAULT_VARIANT,) for vendor in VENDORS
}

DPT_METRICS: tuple[str, ...] = (
    "td_on_ns", "tr_ns", "td_off_ns", "tf_ns",
    "dv_dt_max_v_per_ns", "di_dt_max_a_per_ns",
    "e_on_uj", "e_off_uj", "i_peak_a",
)

ROOT = Path(__file__).resolve().parent.parent


def variants_for(vendor: str) -> tuple[str, ...]:
    """Known Tier-3 variants for *vendor*, default first."""
    return VARIANTS.get(vendor, (DEFAULT_VARIANT,))


def tier3_lib_name(vendor: str, variant: str = DEFAULT_VARIANT) -> str:
    """Emitted Tier-3 library filename for *variant*."""
    suffix = "" if variant == DEFAULT_VARIANT else f"_{variant}"
    return f"{vendor}_PIECESPICE_T3{suffix}.lib"


def lib_path(vendor: str, tier: int,
             variant: str = DEFAULT_VARIANT) -> Path:
    """Expected emitted library path for a tier/variant selection."""
    name = (f"{vendor}_PIECESPICE_T2.lib" if tier == 2
            else tier3_lib_name(vendor, variant))
    return ROOT / "reports" / "models" / name


def fit(vendor: str, tier: int = 2,
        variant: str = DEFAULT_VARIANT) -> Path:
    """Fit the selected tier and return the emitted library path.

    Tier-2 calls the closed ``pipeline.fitting.fit_pwl_model`` unchanged.
    Tier-3 calls ``pipeline.tier3.run_vendor`` (build + validation).
    """
    if tier == 2:
        from pipeline.fitting import fit_pwl_model
        return fit_pwl_model(vendor)
    if tier == 3:
        from pipeline.tier3 import run_vendor
        return Path(run_vendor(vendor, variant=variant)["emitted_lib"])
    raise ValueError(f"unknown tier {tier}; expected one of {TIERS}")


def validate(vendor: str, tier: int = 2,
             variant: str = DEFAULT_VARIANT):
    """Validate the selected tier.

    Tier-2: endpoint-current comparison of its emitted lib against the
    extracted golden curves.  Tier-3: the recorded build + DPT run, with its
    anti-fake completion guard, returned as the measured record.
    """
    if tier == 2:
        from pipeline.fitting import compare_tier2_vs_golden
        return compare_tier2_vs_golden(vendor)
    if tier == 3:
        from pipeline.tier3 import run_vendor
        return run_vendor(vendor, variant=variant)
    raise ValueError(f"unknown tier {tier}; expected one of {TIERS}")


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
