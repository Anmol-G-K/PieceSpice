"""Typer command line interface for PieceSpice."""

from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from parser.netlist import NetlistParser
from parser.preprocessor import normalize_pspice_netlist
from pipeline import registry
from reporting.html_report import generate_html_report
from simulation.testbenches.static_sweep import StaticSweepTestbench

app = typer.Typer(help="PieceSpice: SPICE to Piecewise-Linear Modeling Pipeline")
console = Console()

# Tier-4 disclosure (audit D1 / PLECS M5): the LUT is filled from the recorded
# golden DPT crossing, so a zero error there is by construction.  There is no
# independent converter-energy gate and no converter bench has been built.
T4_ENERGY_CAVEAT = (
    "LUT reproduces its own golden DPT crossing; not an independent "
    "converter validation")
T4_GATE_NOTE = (
    "Tier-4 gate: LUT-vs-recorded-golden-DPT energy at the configured crossing "
    "and speed on the existing inductive bench; DPT waveform gate retired by "
    "design; no converter bench built")


@app.command()
def parse(
    lib_file: Path = typer.Argument(..., help="Path to vendor SPICE model (.lib/.spi)")
):
    """Parse and inspect subcircuits in a SPICE library."""
    if not lib_file.exists():
        console.print(f"[bold red]Error:[/bold red] File {lib_file} does not exist.")
        raise typer.Exit(1)

    subcircuits = NetlistParser.parse_subcircuits(lib_file)
    table = Table(title=f"Parsed Subcircuits in {lib_file.name}")
    table.add_column("Subcircuit Name", style="cyan")
    table.add_column("Pin Count", style="magenta")
    table.add_column("Pin Nodes", style="green")

    for s in subcircuits:
        table.add_row(s.name, str(len(s.nodes)), " ".join(s.nodes))

    console.print(table)


@app.command()
def sweep(
    lib_file: Path = typer.Argument(..., help="Path to vendor SPICE model (.lib/.spi)"),
    subckt: str = typer.Option(..., "--subckt", "-s", help="Subcircuit name to simulate"),
    vds_max: float = typer.Option(10.0, "--vds-max", help="Max Vds for sweep"),
    report_html: Path = typer.Option(Path("report.html"), "--report", "-r", help="HTML report output path")
):
    """Run DC output-characteristics sweep and generate an HTML report.

    For the full seven-curve datasheet extraction use the
    ``curves`` command (config-driven pipeline).
    """
    console.print(f"[bold green]Running DC output characteristics sweep on {subckt}...[/bold green]")
    bench = StaticSweepTestbench()
    curves = bench.run_output_characteristics(
        lib_path=lib_file,
        subcircuit_name=subckt,
        vds_max=vds_max
    )
    console.print(f"Generated {len(curves)} output curves.")
    # Golden-reference metrics come from pipeline.fitting.compare_tier2_vs_golden
    # / metrics.compare; until they are computed the report omits the summary
    # cards rather than showing unmeasured values.
    generate_html_report(
        device_name=subckt,
        output_path=report_html,
        curves=curves,
        metrics={}
    )
    console.print(f"[bold green]HTML report generated:[/bold green] {report_html.resolve()}")




@app.command()
def curves(
    vendor: str = typer.Argument(..., help="Vendor config stem in config/inputs/ (e.g. irfs7730pbf)")
):
    """Run all seven datasheet curve extractions for a vendor."""
    from pipeline.curves import run_all_curves
    console.print(f"[bold green]Extracting all curves for {vendor}...[/bold green]")
    outputs = run_all_curves(vendor)
    for name, path in outputs.items():
        console.print(f"  [cyan]{name}[/cyan] -> {path}")
    console.print(f"[bold green]{len(outputs)} output files written.[/bold green]")



class Tier(int, Enum):
    """Solver tier accepted by --tier."""

    CPWL = 2
    TPWL = 3
    IDEAL_LUT = 4


def _num(value, fmt: str = ".4g") -> str:
    """Render a measured number, or N/A when it was not recorded."""
    if value is None or (isinstance(value, float) and value != value):
        return "N/A"
    return format(value, fmt)


def _yn(value) -> str:
    """Render a tri-state gate flag without turning missing data into a pass."""
    return {True: "yes", False: "no", None: "N/A"}.get(value, "N/A")


def _tier3_summary(record: dict) -> None:
    """Print the measured Tier-3 DPT facts, including the anti-fake counts."""
    dpt = record["dpt_run"]
    state = "complete" if dpt["completed"] else "FAILED"
    console.print(
        f"[bold green]Tier-3 DPT {state}[/bold green] for {record['vendor']} "
        f"variant={record.get('variant', registry.DEFAULT_VARIANT)}: "
        f"trace_points={dpt['trace_points']}/{dpt['golden_trace_points']} "
        f"iterations={_num(dpt['iterations'], 'd')}")
    wf = dpt.get("waveform") or {}
    if wf:
        console.print(
            f"  Vds rel-RMSE={_num(wf.get('vds_rel_rmse_pct'))}% "
            f"Id rel-RMSE={_num(wf.get('id_rel_rmse_pct'))}% "
            f"compare_points={wf.get('compare_points')}")
    if dpt.get("error"):
        console.print(f"  [bold red]error:[/bold red] {dpt['error']}")
    console.print(f"  emitted lib: {record['emitted_lib']}")


def _tier4_summary(record: dict) -> None:
    """Print DPT-derived Tier-4 energy and inductive-bench speed honestly."""
    console.print(
        f"[bold green]Tier-4 ideal-switch + loss-LUT[/bold green] for "
        f"{record['vendor']}: DPT waveform gate retired by design")
    console.print(
        "  energy comparison is LUT-vs-recorded-golden-DPT at the configured "
        "crossing; speed comes from the existing inductive bench; no "
        "converter bench built")
    energy = record.get("energy_comparison") or {}
    for metric in ("e_on_uj", "e_off_uj"):
        row = energy.get(metric) or {}
        console.print(
            f"  {metric}: golden={_num(row.get('golden'))} uJ "
            f"tier4={_num(row.get('tier4'))} uJ "
            f"abs={_num(row.get('abs_error'))} uJ "
            f"rel={_num(row.get('rel_error_pct'))}%")
        console.print(
            f"    [yellow]{row.get('disclosure') or T4_ENERGY_CAVEAT}[/yellow]")
    top_disclosure = record.get("energy_gate_disclosure")
    if top_disclosure and str(top_disclosure) != T4_ENERGY_CAVEAT:
        console.print(f"  [yellow]{top_disclosure}[/yellow]")
    console.print(
        "  e_on/e_off relative error is 0 by construction at the recorded "
        "golden DPT crossing; no independent converter-energy gate")
    speed = record.get("speedup") or {}
    console.print(
        f"  speed status={speed.get('status') or 'N/A'}: "
        f"solver={_num(speed.get('solver_speedup'))}x "
        f"wall={_num(speed.get('wall_speedup'))}x "
        f"iters={_num(speed.get('golden_iterations'), 'd')}/"
        f"{_num(speed.get('tier4_iterations'), 'd')} "
        f"points={_num(speed.get('golden_points'), 'd')}/"
        f"{_num(speed.get('tier4_points'), 'd')} "
        f"t_last={_num(speed.get('t_last_s'))} s")
    if record.get("emitted_lib"):
        console.print(f"  emitted lib: {record['emitted_lib']}")


def _unified_summary(record: dict) -> None:
    """Print the unified G0-G6 record with method label and guard counts."""
    gates = record.get("gates") or {}
    console.print(
        f"[bold green]Unified one-ROM[/bold green] for "
        f"{record.get('vendor')}: variant={record.get('variant')} "
        f"basis={record.get('channel_basis')} method={record.get('method')}")
    console.print(
        f"  verdict={record.get('verdict') or 'N/A'} "
        f"all_pass={_yn(gates.get('all_pass'))}")
    for name in ("G0", "G1_static", "G2_derivative", "G3_dpt_waveform",
                 "G4_timing_energy", "G5_speed", "G6_charge"):
        gate = gates.get(name) or {}
        console.print(
            f"  {name}: pass={_yn(gate.get('pass'))} "
            f"status={gate.get('status') or 'N/A'}")
    speed = record.get("speed_adaptive_20") or {}
    console.print(
        "  speed row: "
        f"n_pulses={_num(speed.get('n_pulses'), 'd')} "
        f"max_step={_num(speed.get('max_step'))} "
        f"method={speed.get('method_label') or 'N/A'} "
        f"solver={_num(speed.get('solver_speedup'))}x "
        f"wall={_num(speed.get('wall_speedup'))}x "
        f"global_claim={_yn(speed.get('global_claim'))}")
    runtime = record.get("runtime") or {}
    for arm in ("golden", "urom"):
        console.print(
            f"  {arm}: iterations={_num((runtime.get('iterations') or {}).get(arm), 'd')} "
            f"points={_num((runtime.get('points') or {}).get(arm), 'd')} "
            f"t_last={_num((runtime.get('t_last') or {}).get(arm))} s "
            f"complete={_yn((runtime.get('completion') or {}).get(arm))}")
    console.print(f"  repeats={_num(runtime.get('repeats'), 'd')}")
    if record.get("emitted_lib"):
        console.print(f"  emitted lib: {record['emitted_lib']}")


@app.command()
def fit(
    vendor: str = typer.Argument(..., help="Vendor config stem in config/inputs/"),
    tier: Tier = typer.Option(Tier.CPWL, "--tier",
                              help="Solver tier: 2 (CPWL), 3 (TPWL/urom) or 4 (ideal LUT)"),
    variant: str = typer.Option(registry.DEFAULT_VARIANT, "--variant",
                                help="Tier-3 variant: v1 (default), simplis, urom (unified); ignored by Tier-4"),
) -> None:
    """Fit a Tier-2, Tier-3 or Tier-4 subcircuit for VENDOR."""
    if tier == 2:
        path = registry.fit(vendor, 2)
        console.print(f"[bold green]Tier-2 model fitted:[/bold green] {path}")
        return
    if tier == 4:
        _tier4_summary(registry.validate(vendor, 4))
        return
    if tier == 3 and variant == registry.UNIFIED_VARIANT:
        _unified_summary(registry.validate(vendor, 3, variant))
        return
    record = registry.validate(vendor, 3, variant)
    _tier3_summary(record)


@app.command()
def validate(
    vendor: str = typer.Argument(..., help="Vendor config stem in config/inputs/"),
    tier: Tier = typer.Option(Tier.CPWL, "--tier",
                              help="Solver tier: 2 (CPWL), 3 (TPWL/urom) or 4 (ideal LUT)"),
    variant: str = typer.Option(registry.DEFAULT_VARIANT, "--variant",
                                help="Tier-3 variant: v1 (default), simplis, urom (unified); ignored by Tier-4"),
) -> None:
    """Validate VENDOR against the golden model at the selected tier."""
    if tier == 4:
        _tier4_summary(registry.validate(vendor, 4))
        return
    if tier == 3:
        if variant == registry.UNIFIED_VARIANT:
            _unified_summary(registry.validate(vendor, 3, variant))
            return
        _tier3_summary(registry.validate(vendor, 3, variant))
        return
    lib = registry.lib_path(vendor, 2)
    if not lib.exists():
        console.print(
            f"[bold red]No Tier-2 lib for {vendor}[/bold red]: {lib}. "
            "Run `fit <vendor> --tier 2` first.")
        raise typer.Exit(1)
    rows, worst = registry.validate(vendor, 2)
    table = Table(title=f"Tier-2 endpoint validation — {vendor}")
    table.add_column("Vgs (V)")
    table.add_column("Golden (A)")
    table.add_column("Tier-2 (A)")
    table.add_column("Rel error (%)")
    for vgs, golden, t2, rel in rows:
        table.add_row(f"{vgs:g}", _num(golden), _num(t2), _num(rel))
    console.print(table)
    console.print(f"Worst endpoint relative error: {_num(worst)}%")


@app.command()
def compare(
    vendor: str = typer.Argument(..., help="Vendor config stem in config/inputs/"),
) -> None:
    """Compare recorded DPT, Tier-4 energy/speed and unified-ROM for VENDOR."""
    from pipeline.unified_rom import compare_unified
    unified = compare_unified(vendor)
    rows = registry.compare_rows(vendor)
    if not rows and not unified.get("present"):
        console.print(
            f"[bold red]No recorded metrics for {vendor}[/bold red]. "
            "Run the tier fits/validations first.")
        raise typer.Exit(1)
    table = Table(title=f"Golden vs Tier-2 vs Tier-3 — {vendor}")
    table.add_column("Metric")
    table.add_column("Golden")
    table.add_column("Tier-2")
    table.add_column("Tier-3")
    for metric, golden, t2, t3 in rows:
        table.add_row(metric, _num(golden), _num(t2), _num(t3))
    console.print(table)

    t4_rows = registry.tier4_rows(vendor)
    console.print(f"[bold]{T4_GATE_NOTE}[/bold]")
    console.print(
        f"[yellow]{T4_ENERGY_CAVEAT} (e_on/e_off relative error is 0 by "
        "construction at the recorded golden DPT crossing; no independent "
        "converter-energy gate)[/yellow]")
    if not t4_rows:
        console.print(
            f"[yellow]No recorded Tier-4 energy/speed for {vendor}[/yellow]. "
            "Run `fit <vendor> --tier 4` first.")
    else:
        table = Table(title=f"Tier-4 energy & speed (waveform gate retired) — {vendor}")
        table.add_column("Quantity")
        table.add_column("Golden")
        table.add_column("Tier-4")
        table.add_column("Detail")
        for metric, golden, t4, detail in t4_rows:
            table.add_row(metric, _num(golden), _num(t4), detail)
        console.print(table)

    if not unified.get("present"):
        console.print(
            "[yellow]No recorded unified validation for "
            f"{vendor}[/yellow]. Run `fit <vendor> --tier 3 --variant urom` first.")
        return
    console.print("[bold]Unified one-ROM (recorded only; never recomputed)[/bold]")
    _unified_summary(unified)

if __name__ == "__main__":
    app()
