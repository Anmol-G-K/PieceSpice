"""Typer command line interface for PieceSpice."""

from enum import Enum
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from parser.netlist import NetlistParser
from parser.preprocessor import normalize_pspice_netlist
from simulation.testbenches.static_sweep import StaticSweepTestbench
from reporting.html_report import generate_html_report
from pipeline import registry

app = typer.Typer(help="PieceSpice: SPICE to Piecewise-Linear Modeling Pipeline")
console = Console()


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


def _num(value, fmt: str = ".4g") -> str:
    """Render a measured number, or N/A when it was not recorded."""
    if value is None or (isinstance(value, float) and value != value):
        return "N/A"
    return format(value, fmt)


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


@app.command()
def fit(
    vendor: str = typer.Argument(..., help="Vendor config stem in config/inputs/"),
    tier: Tier = typer.Option(Tier.CPWL, "--tier",
                              help="Solver tier: 2 (CPWL) or 3 (TPWL)"),
    variant: str = typer.Option(registry.DEFAULT_VARIANT, "--variant",
                                help="Tier-3 variant (default: v1)"),
) -> None:
    """Fit a Tier-2 or Tier-3 subcircuit for VENDOR."""
    if int(tier) == 2:
        path = registry.fit(vendor, 2)
        console.print(f"[bold green]Tier-2 model fitted:[/bold green] {path}")
        return
    record = registry.validate(vendor, 3, variant)
    _tier3_summary(record)


@app.command()
def validate(
    vendor: str = typer.Argument(..., help="Vendor config stem in config/inputs/"),
    tier: Tier = typer.Option(Tier.CPWL, "--tier",
                              help="Solver tier: 2 (CPWL) or 3 (TPWL)"),
    variant: str = typer.Option(registry.DEFAULT_VARIANT, "--variant",
                                help="Tier-3 variant (default: v1)"),
) -> None:
    """Validate VENDOR against the golden model at the selected tier."""
    if int(tier) == 3:
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
    """Compare recorded golden vs Tier-2 vs Tier-3 DPT metrics."""
    rows = registry.compare_rows(vendor)
    if not rows:
        console.print(
            f"[bold red]No recorded DPT metrics for {vendor}[/bold red]. "
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

if __name__ == "__main__":
    app()
