#!/usr/bin/env python3
"""Independent verifier for the final architecture pass.

The verifier reads the committed comparison records only; it never runs a
simulator and never edits a model, pipeline, or config file.  It applies the
user's final gates directly to the recorded numbers:

* timing/energy: relative error <= 7.0 % (5.0 % target + 2.0 % tolerance);
  every row reports its margin and any row above 7.0 % fails.
* speed: solver speedup >= 1.50x.  The wall-clock ratio is reported next to
  it and is never substituted for the solver gate.
* G3: the Vds/Id bars already stored in the records are used as-is.  The
  verifier does not invent bars and does not score the diagnostic Tier-2
  regression columns.

All four artifacts are covered for every vendor: Tier-4, canonical UROM,
the ANN variant, and the Tier-3 ``simplis`` variant.  Missing records, rows,
bars, or speed fields fail closed with an explicit message.

Exit status is zero only when every applicable gate passes.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

VENDORS = ("cpm2-1200", "irfs7730pbf", "epc2216", "csd16412q5a")
TIERS = ("Tier-4", "UROM", "ANN", "simplis")
TIMING_ENERGY_ROWS = (
    "td_on_ns",
    "tr_ns",
    "td_off_ns",
    "tf_ns",
    "e_on_uj",
    "e_off_uj",
)
TIER4_ROWS = ("e_on_uj", "e_off_uj")

TARGET_PCT = 5.0
TOLERANCE_PCT = 2.0
GATE_PCT = TARGET_PCT + TOLERANCE_PCT
SPEED_GATE = 1.50
EPS = 1.0e-9

RECORD_FILES = {
    "Tier-4": "tier4_validation.json",
    "UROM": "unified_validation.json",
    "ANN": "unified_ann_trap-default.json",
    "simplis": "tier3_validation_simplis.json",
}


def _finite(value: object) -> float | None:
    """Return a finite float, or None for non-numeric/bool/NaN/inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nested(obj: object, *keys: str) -> dict:
    """Return a nested mapping, or an empty one when a level is malformed."""
    current: object = obj
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _relative_error_pct(value: float, bar: float) -> float:
    """Absolute relative error against the recorded bar, in percent."""
    if bar == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return abs(value - bar) / abs(bar) * 100.0


@dataclass
class RowResult:
    metric: str
    value: float | None
    bar: float | None
    error_pct: float | None
    margin_pp: float | None
    status: str
    reason: str = ""

    @property
    def passed(self) -> bool:
        return (self.error_pct is not None
                and self.error_pct <= GATE_PCT + EPS)


@dataclass
class G3Result:
    status: str
    vds: float | None
    id: float | None
    vds_bar: float | None
    id_bar: float | None
    bar_source: str
    reason: str = ""

    @property
    def retired(self) -> bool:
        return self.status == "retired"

    @property
    def gate_ok(self) -> bool:
        if self.retired:
            return True
        return (self.status == "measured"
                and self.vds is not None and self.id is not None
                and self.vds_bar is not None and self.id_bar is not None
                and self.vds <= self.vds_bar + EPS
                and self.id <= self.id_bar + EPS)


@dataclass
class SpeedResult:
    solver: float | None
    wall: float | None
    status: str
    reason: str = ""

    @property
    def passed(self) -> bool:
        return (self.status == "measured"
                and self.solver is not None
                and self.solver >= SPEED_GATE - EPS)


@dataclass
class TierResult:
    tier: str
    vendor: str
    rows: list[RowResult]
    speed: SpeedResult
    g3: G3Result
    error: str = ""

    @property
    def failures(self) -> list[str]:
        prefix = f"{self.tier}/{self.vendor}"
        if self.error:
            return [f"{prefix}: {self.error}"]
        out: list[str] = []
        for row in self.rows:
            if row.passed:
                continue
            if row.error_pct is None:
                out.append(f"{prefix} {row.metric}: not measured "
                           f"({row.reason or row.status})")
            else:
                out.append(
                    f"{prefix} {row.metric}: relative error "
                    f"{row.error_pct:.2f}% > {GATE_PCT:.2f}% "
                    f"(value {row.value:.6g}, bar {row.bar:.6g}, "
                    f"margin {row.margin_pp:+.2f} pp)")
        if not self.speed.passed:
            wall = (f"{self.speed.wall:.4f}x"
                    if self.speed.wall is not None else "n/a")
            if self.speed.solver is None:
                out.append(f"{prefix} speed: solver_speedup missing "
                           f"(wall {wall} reported, not substituted; "
                           f"{self.speed.reason or self.speed.status})")
            else:
                out.append(
                    f"{prefix} speed: solver speedup "
                    f"{self.speed.solver:.4f}x < {SPEED_GATE:.2f}x "
                    f"(wall {wall} reported separately; not substituted; "
                    f"{self.speed.reason or self.speed.status})")
        if not self.g3.gate_ok:
            out.extend(self._g3_failures(prefix))
        return out

    def _g3_failures(self, prefix: str) -> list[str]:
        g3 = self.g3
        if g3.status == "missing":
            return [f"{prefix} G3: {g3.reason or 'no recorded Vds/Id bars'}"]
        if g3.status != "measured":
            return [f"{prefix} G3: not measured "
                    f"({g3.reason or g3.status})"]
        out: list[str] = []
        if g3.vds is not None and g3.vds_bar is not None \
                and g3.vds > g3.vds_bar + EPS:
            out.append(f"{prefix} G3 Vds: {g3.vds:.3f}% > "
                       f"{g3.vds_bar:.3f}% ({g3.bar_source})")
        if g3.id is not None and g3.id_bar is not None \
                and g3.id > g3.id_bar + EPS:
            out.append(f"{prefix} G3 Id: {g3.id:.3f}% > "
                       f"{g3.id_bar:.3f}% ({g3.bar_source})")
        if not out:
            out.append(f"{prefix} G3: bars/metrics incomplete "
                       f"({g3.reason or g3.status})")
        return out

    @property
    def status(self) -> str:
        return "FAIL" if self.failures else "PASS"


def _row(metric: str, value: object, bar: object,
         status: object = "measured") -> RowResult:
    state = str(status) if status is not None else "measured"
    number = _finite(value)
    target = _finite(bar)
    if state != "measured":
        return RowResult(metric, number, target, None, None, state,
                         f"status={state}")
    if number is None or target is None:
        return RowResult(metric, number, target, None, None, "missing",
                         "value or bar absent")
    error = _relative_error_pct(number, target)
    return RowResult(metric, number, target, error, GATE_PCT - error,
                     "measured")


def _rows_from_mapping(mapping: object, metrics: tuple[str, ...],
                       source: str) -> list[RowResult]:
    if not isinstance(mapping, dict):
        return [RowResult(metric, None, None, None, None, "missing",
                          f"{source} absent")
                for metric in metrics]
    rows: list[RowResult] = []
    for metric in metrics:
        item = mapping.get(metric)
        if not isinstance(item, dict):
            rows.append(RowResult(metric, None, None, None, None, "missing",
                                  f"row absent from {source}"))
            continue
        rows.append(_row(metric, item.get("value"), item.get("bar"),
                         item.get("status", "measured")))
    return rows


def _rows_from_pairs(mapping: object, metrics: tuple[str, ...],
                     source: str) -> list[RowResult]:
    """Simplis records pair a golden bar with the Tier-3 measurement."""
    if not isinstance(mapping, dict):
        return [RowResult(metric, None, None, None, None, "missing",
                          f"{source} absent")
                for metric in metrics]
    rows: list[RowResult] = []
    for metric in metrics:
        item = mapping.get(metric)
        if not isinstance(item, dict):
            rows.append(RowResult(metric, None, None, None, None, "missing",
                                  f"row absent from {source}"))
            continue
        rows.append(_row(metric, item.get("tier3"), item.get("golden"),
                         item.get("status", "measured")))
    return rows


def _bars(obj: object) -> tuple[float, float] | None:
    if not isinstance(obj, dict):
        return None
    values = obj.get("bars_vds_id_pct")
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return None
    vds_bar = _finite(values[0])
    id_bar = _finite(values[1])
    if vds_bar is None or id_bar is None or vds_bar <= 0 or id_bar <= 0:
        return None
    return vds_bar, id_bar


def _g3_unified(record: dict) -> G3Result:
    detail = _nested(record, "gates", "G3_dpt_waveform", "detail")
    arm = detail.get("urom")
    if not isinstance(arm, dict):
        # The golden arm is a 0 % self-comparison control; never substitute
        # it for a missing candidate measurement.
        return G3Result("missing", None, None, None, None, "",
                        "G3 urom candidate arm absent")
    bars = _bars(arm)
    vds = _finite(arm.get("vds_rel_rmse_pct"))
    ids = _finite(arm.get("id_rel_rmse_pct"))
    if bars is None:
        return G3Result("missing", vds, ids, None, None, "",
                        "no recorded Vds/Id bar in the G3 urom arm")
    status = str(arm.get("status", "measured"))
    if status != "measured":
        return G3Result(status, vds, ids, bars[0], bars[1],
                        "gates.G3_dpt_waveform.detail.urom", f"status={status}")
    if vds is None or ids is None:
        return G3Result("missing", vds, ids, bars[0], bars[1],
                        "gates.G3_dpt_waveform.detail.urom",
                        "Vds/Id metric absent")
    return G3Result("measured", vds, ids, bars[0], bars[1],
                    "gates.G3_dpt_waveform.detail.urom.bars_vds_id_pct")


def _g3_simplis(record: dict, canonical: dict | None) -> G3Result:
    dpt_run = record.get("dpt_run")
    waveform = dpt_run.get("waveform") if isinstance(dpt_run, dict) else None
    if not isinstance(waveform, dict):
        waveform = record.get("dpt_waveform")
    if not isinstance(waveform, dict):
        return G3Result("missing", None, None, None, None, "",
                        "simplis dpt_run.waveform absent")
    vds = _finite(waveform.get("vds_rel_rmse_pct"))
    ids = _finite(waveform.get("id_rel_rmse_pct"))
    bars = _bars(waveform)
    source = "simplis dpt_run.waveform.bars_vds_id_pct"
    if bars is None and canonical is not None:
        bars = _bars(_nested(canonical, "gates", "G3_dpt_waveform",
                             "detail", "urom"))
        source = ("same-vendor unified_validation.json"
                  "#gates.G3_dpt_waveform.detail.urom.bars_vds_id_pct")
    if bars is None:
        return G3Result("missing", vds, ids, None, None, "",
                        "no recorded Vds/Id bar in simplis or UROM record")
    if vds is None or ids is None:
        return G3Result("missing", vds, ids, bars[0], bars[1], source,
                        "Vds/Id metric absent")
    return G3Result("measured", vds, ids, bars[0], bars[1], source)


def _speed_tier4(record: dict) -> SpeedResult:
    row = record.get("speedup")
    if not isinstance(row, dict):
        return SpeedResult(None, None, "missing", "speedup block absent")
    solver = _finite(row.get("solver_speedup"))
    wall = _finite(row.get("wall_speedup"))
    status = str(row.get("status", "missing"))
    if status != "measured":
        return SpeedResult(solver, wall, status, f"status={status}")
    if solver is None:
        return SpeedResult(None, wall, "missing", "solver_speedup absent")
    guards = _nested(record, "guards")
    if guards.get("speed_completion") is not True:
        return SpeedResult(solver, wall, "not_measured",
                           "guards.speed_completion is not true")
    return SpeedResult(solver, wall, "measured")


def _speed_unified(record: dict) -> SpeedResult:
    detail = _nested(record, "gates", "G5_speed", "detail")
    if not detail:
        return SpeedResult(None, None, "missing", "G5_speed detail absent")
    solver = _finite(detail.get("solver_speedup"))
    wall = _finite(detail.get("wall_speedup"))
    if solver is None:
        return SpeedResult(None, wall, "missing", "solver_speedup absent")
    if detail.get("scoped_row_valid") is False:
        return SpeedResult(solver, wall, "not_measured",
                           "scoped_row_valid is false")
    completion = detail.get("completion")
    if isinstance(completion, dict) and completion:
        if not all(completion.get(key) is True for key in ("golden", "urom")):
            return SpeedResult(solver, wall, "not_measured",
                               "run incomplete")
    return SpeedResult(solver, wall, "measured")


def _speed_simplis(record: dict) -> SpeedResult:
    row = record.get("speed_adaptive_20")
    if not isinstance(row, dict):
        row = record.get("speed")
    if not isinstance(row, dict):
        return SpeedResult(None, None, "missing",
                           "speed_adaptive_20 block absent")
    solver = _finite(row.get("solver_speedup"))
    wall = _finite(row.get("wall_speedup"))
    if solver is None:
        return SpeedResult(None, wall, "missing", "solver_speedup absent")
    for arm in ("golden", "tier3_simplis"):
        side = row.get(arm)
        if not isinstance(side, dict) or side.get("completed") is not True:
            return SpeedResult(solver, wall, "not_measured",
                               f"{arm} completion missing or false")
    return SpeedResult(solver, wall, "measured")


def evaluate_tier4(record: dict, vendor: str) -> TierResult:
    comparison = record.get("energy_comparison")
    rows: list[RowResult] = []
    for metric in TIER4_ROWS:
        if not isinstance(comparison, dict):
            rows.append(RowResult(metric, None, None, None, None, "missing",
                                  "energy_comparison absent"))
            continue
        item = comparison.get(metric)
        if not isinstance(item, dict):
            rows.append(RowResult(metric, None, None, None, None, "missing",
                                  "row absent from energy_comparison"))
            continue
        rows.append(_row(metric, item.get("tier4"), item.get("golden"),
                         item.get("status", "measured")))
    # Tier-4 has no timing rows and its Vds/Id waveform gate is retired.
    if record.get("waveform_gate") == "retired_by_design":
        g3 = G3Result("retired", None, None, None, None,
                      "tier4_validation.json#waveform_gate")
    else:
        g3 = G3Result("missing", None, None, None, None, "",
                      "Tier-4 record does not declare a retired waveform gate")
    return TierResult("Tier-4", vendor, rows, _speed_tier4(record), g3)


def evaluate_unified(record: dict, vendor: str, tier: str) -> TierResult:
    rows_map = _nested(record, "gates", "G4_timing_energy", "detail", "rows")
    rows = _rows_from_mapping(
        rows_map, TIMING_ENERGY_ROWS,
        "gates.G4_timing_energy.detail.rows")
    return TierResult(tier, vendor, rows, _speed_unified(record),
                      _g3_unified(record))


def evaluate_simplis(record: dict, vendor: str,
                     canonical: dict | None) -> TierResult:
    rows = _rows_from_pairs(record.get("dpt_comparison"),
                            TIMING_ENERGY_ROWS, "dpt_comparison")
    return TierResult("simplis", vendor, rows, _speed_simplis(record),
                      _g3_simplis(record, canonical))


def _read_json(path: Path) -> tuple[dict | None, str]:
    if not path.is_file():
        return None, f"missing record {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable record {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"malformed record {path}: top level is not an object"
    return data, ""


def evaluate_records(root: Path | str) -> list[TierResult]:
    base = Path(root) / "reports" / "comparison"
    canonical: dict[str, dict | None] = {}
    for vendor in VENDORS:
        record, _ = _read_json(base / vendor / "unified_validation.json")
        canonical[vendor] = record

    results: list[TierResult] = []
    for tier in TIERS:
        for vendor in VENDORS:
            path = base / vendor / RECORD_FILES[tier]
            record, error = _read_json(path)
            if record is None:
                results.append(TierResult(
                    tier, vendor, [],
                    SpeedResult(None, None, "missing", error),
                    G3Result("missing", None, None, None, None, "", error),
                    error=error))
                continue
            if tier == "Tier-4":
                results.append(evaluate_tier4(record, vendor))
            elif tier == "simplis":
                results.append(evaluate_simplis(record, vendor,
                                                canonical.get(vendor)))
            else:
                results.append(evaluate_unified(record, vendor, tier))
    return results


def _g3_cell(result: TierResult) -> str:
    g3 = result.g3
    if g3.retired:
        return "retired"
    if (g3.status != "measured" or g3.vds is None or g3.id is None
            or g3.vds_bar is None or g3.id_bar is None):
        return "MISSING"
    verdict = "PASS" if g3.gate_ok else "FAIL"
    return (f"{g3.vds:.3f}/{g3.id:.3f} vs "
            f"{g3.vds_bar:.3f}/{g3.id_bar:.3f} {verdict}")


def _g4_cell(result: TierResult) -> str:
    if not result.rows:
        return "missing"
    missing = [row for row in result.rows if row.error_pct is None]
    if missing:
        return f"{missing[0].metric} {missing[0].status}"
    worst = min(result.rows, key=lambda row: row.margin_pp)
    failed = sum(not row.passed for row in result.rows)
    suffix = f" [{failed} fail]" if failed else ""
    return (f"{worst.metric} {worst.error_pct:.1f}% "
            f"({worst.margin_pp:+.1f}pp){suffix}")


def _speed_cell(result: TierResult) -> str:
    solver = (f"{result.speed.solver:.4f}"
              if result.speed.solver is not None else "missing")
    wall = (f"{result.speed.wall:.4f}"
            if result.speed.wall is not None else "n/a")
    return f"{solver}/{wall}x"


def _margin_cell(result: TierResult, metric: str) -> str:
    row = next((item for item in result.rows if item.metric == metric), None)
    if row is None:
        return "-"
    if row.margin_pp is None:
        return "MISS"
    return f"{row.margin_pp:+.2f}"


def format_report(results: list[TierResult]) -> str:
    header = ("tier", "vendor", "G3 Vds/Id vs bar", "G4 worst row",
              "solver/wall", "verdict")
    body = []
    for result in results:
        body.append((
            result.tier,
            result.vendor,
            _g3_cell(result),
            _g4_cell(result),
            _speed_cell(result),
            result.status,
        ))
    widths = [len(item) for item in header]
    for cells in body:
        widths = [max(width, len(cell))
                  for width, cell in zip(widths, cells)]

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(width)
                         for cell, width in zip(cells, widths)).rstrip()

    lines = [
        "FINAL ARCHITECTURE VERIFICATION (recorded JSON only)",
        (f"gates: timing/energy |rel err| <= {GATE_PCT:.1f}% "
         f"({TARGET_PCT:.1f}% target + {TOLERANCE_PCT:.1f}% tolerance); "
         f"solver speedup >= {SPEED_GATE:.2f}x; "
         "G3 uses recorded Vds/Id bars"),
        "",
        line(header),
        line(tuple("-" * width for width in widths)),
    ]
    lines.extend(line(cells) for cells in body)
    lines.extend([
        "",
        ("PER-ROW TIMING/ENERGY MARGINS (pp below the 7.0% gate; "
         "negative = fail; '-' = not in this artifact)"),
    ])
    margin_header = ("tier/vendor",) + TIMING_ENERGY_ROWS
    margin_rows = []
    for result in results:
        margin_rows.append(
            (f"{result.tier}/{result.vendor}",)
            + tuple(_margin_cell(result, metric)
                    for metric in TIMING_ENERGY_ROWS))
    margin_widths = [len(item) for item in margin_header]
    for cells in margin_rows:
        margin_widths = [max(width, len(cell))
                         for width, cell in zip(margin_widths, cells)]

    def margin_line(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(width)
                         for cell, width in zip(cells, margin_widths)).rstrip()

    lines.append(margin_line(margin_header))
    lines.extend(margin_line(cells) for cells in margin_rows)
    lines.extend([
        "",
        ("provenance: Tier-4 energy rows are scored as recorded "
         "(source_self_consistency, independent_energy_gate=false); "
         "simplis G3 bars come from the same vendor's recorded UROM G3 arm "
         "when the simplis record has none; Tier-2 diagnostic columns are "
         "never scored."),
    ])
    failures = [failure for result in results for failure in result.failures]
    if failures:
        lines.extend(["", f"FAILURES ({len(failures)}):"])
        lines.extend(f"  - {failure}" for failure in failures)
    else:
        lines.extend(["", "ALL GATES PASS"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Verify the final architecture gates from recorded "
                     "JSON only."))
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (default: parent of scripts/)")
    args = parser.parse_args(argv)
    results = evaluate_records(args.root)
    print(format_report(results))
    return 0 if all(result.status == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
