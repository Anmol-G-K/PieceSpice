"""Interactive PieceSpice workbench.

A small, keyboard-driven front end over the existing registry/pipeline APIs.
It deliberately does not reimplement any model logic: every run goes through
the same fit/validate entry points as the Typer commands.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import questionary
from questionary import Style
from rich.console import Console
from rich.table import Table

from cli.branding import banner
from config.yaml_io import INPUT_DIR, load_input
from parser.netlist import NetlistParser

ROOT = Path(__file__).resolve().parent.parent
SPICE_DIR = ROOT / "SpiceModels"
FIGURES = ROOT / "figures"
MODEL_SUFFIXES = {".lib", ".spi", ".net", ".cir"}

STYLE = Style([
    ("qmark", "fg:#38bdf8 bold"),
    ("question", "fg:#e2e8f0 bold"),
    ("answer", "fg:#fbbf24 bold"),
    ("pointer", "fg:#fbbf24 bold"),
    ("highlighted", "fg:#0f172a bg:#38bdf8 bold"),
    ("selected", "fg:#38bdf8"),
    ("separator", "fg:#64748b"),
    ("instruction", "fg:#94a3b8"),
    ("text", "fg:#e2e8f0"),
])

TIERS: list[tuple[str, str]] = [
    ("Tier-2 CPWL  ·  static I-V", "tier2"),
    ("Tier-3 v1  ·  TPWL/POD-DEIM", "tier3"),
    ("Tier-3 simplis  ·  PWL charge", "tier3_simplis"),
    ("UROM C0  ·  canonical Tier-2 basis", "urom_tier2"),
    ("UROM ANN  ·  measured positive-conductance basis", "urom_ann"),
    ("Tier-4  ·  ideal switch + loss LUT", "tier4"),
    ("Tier-4 ANN  ·  C-inf ANN switch", "tier4_ann"),
]


def _ask(question):
    """Run a questionary prompt and abort cleanly on Ctrl-C/Esc."""
    answer = question.ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


def _configured_models() -> list[dict[str, Any]]:
    """Vendor configs currently wired into the pipeline."""
    models: list[dict[str, Any]] = []
    for path in sorted(INPUT_DIR.glob("*.yaml")):
        try:
            cfg = load_input(path.stem)
        except Exception:
            continue
        if not isinstance(cfg, dict) or not cfg.get("library"):
            continue
        models.append({
            "vendor": cfg.get("device") or path.stem,
            "stem": path.stem,
            "library": str(cfg["library"]),
            "subcircuit": cfg.get("subcircuit") or "N/A",
        })
    return models


def _model_files() -> list[Path]:
    """Vendor libraries available under the documented SpiceModels folder."""
    if not SPICE_DIR.exists():
        return []
    return sorted(
        p for p in SPICE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES
    )


def _configured_libraries() -> set[Path]:
    out: set[Path] = set()
    for item in _configured_models():
        out.add((ROOT / item["library"]).resolve())
    return out


def _model_choices() -> list[questionary.Choice]:
    choices = []
    for m in _configured_models():
        lib = ROOT / m["library"]
        choices.append(questionary.Choice(
            title=(f"{m['vendor']}  ·  {lib.name}  ·  {m['subcircuit']}"
                   + ("" if lib.exists() else "  [missing]")),
            value=m,
        ))
    return choices


def _print_catalog(console: Console) -> None:
    table = Table(title="Model catalog", header_style="bold #38bdf8")
    table.add_column("Vendor", style="#e2e8f0")
    table.add_column("Library", style="#94a3b8")
    table.add_column("Subcircuit", style="#fbbf24")
    table.add_column("Status")
    for m in _configured_models():
        lib = ROOT / m["library"]
        table.add_row(m["vendor"], str(lib.relative_to(ROOT))
                      if lib.is_relative_to(ROOT) else str(lib),
                      m["subcircuit"], "configured" if lib.exists()
                      else "[red]missing[/red]")
    console.print(table)

    files = _model_files()
    if files:
        table = Table(title="SpiceModels files", header_style="bold #38bdf8")
        table.add_column("File", style="#94a3b8")
        table.add_column("State")
        configured = _configured_libraries()
        for p in files:
            table.add_row(str(p.relative_to(ROOT)),
                          "configured" if p.resolve() in configured
                          else "[yellow]needs config[/yellow]")
        console.print(table)
    else:
        console.print("[yellow]No model files in SpiceModels/.[/yellow] "
                      "Drop a .lib/.spi/.net file there and add its "
                      "config/inputs/<vendor>.yaml.")


def inspect_model(console: Console) -> dict[str, Any] | None:
    """Parse a SpiceModels file and optionally return its config."""
    files = _model_files()
    if not files:
        console.print("[yellow]SpiceModels/ has no .lib/.spi/.net files.[/yellow]")
        return None
    configured: dict[Path, dict[str, Any]] = {}
    for m in _configured_models():
        configured[(ROOT / m["library"]).resolve()] = m
    choice = _ask(questionary.select(
        "Inspect a model library",
        choices=[questionary.Choice(str(p.relative_to(ROOT)), value=p)
                 for p in files],
        style=STYLE,
    ))
    try:
        subckts = NetlistParser.parse_subcircuits(choice)
    except Exception as exc:
        console.print(f"[red]Could not parse {choice}: {exc}[/red]")
        return None
    table = Table(title=f"{choice.name} subcircuits",
                  header_style="bold #38bdf8")
    table.add_column("Name", style="#fbbf24")
    table.add_column("Pins", style="#94a3b8")
    for s in subckts:
        table.add_row(s.name, " ".join(s.nodes))
    console.print(table)
    cfg = configured.get(choice.resolve())
    if cfg:
        console.print(f"[green]Configured vendor:[/green] {cfg['vendor']} "
                      f"({cfg['subcircuit']})")
    else:
        console.print(
            "[yellow]No config/inputs mapping for this file.[/yellow] "
            "Add `library`, `subcircuit`, `pins`, `conditions`, and `dpt` "
            "to a vendor YAML before running it. SpiceModels/ stays read-only."
        )
    return cfg


def _read_record(vendor: str, tier_key: str) -> dict[str, Any] | None:
    """Read the recorded result for a tier without rerunning it."""
    import json
    base = ROOT / "reports" / "comparison"
    paths = {
        "tier2": base / "no-single-record.json",
        "tier3": base / vendor / "tier3_validation.json",
        "tier3_simplis": base / vendor / "tier3_validation_simplis.json",
        "urom_tier2": base / vendor / "unified_validation.json",
        "urom_ann": base / vendor / "unified_ann_trap-default.json",
        "tier4": base / vendor / "tier4_validation.json",
        "tier4_ann": base / vendor / "tier4_validation_ann.json",
    }
    path = paths[tier_key]
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_tier(vendor: str, tier_key: str, *, build: bool,
              console: Console) -> dict[str, Any] | None:
    """Run or read the selected tier through the canonical pipeline entry."""
    if not build:
        rec = _read_record(vendor, tier_key)
        if rec is None:
            console.print("[yellow]No recorded result; enable "
                          "'Build / validate' to run it.[/yellow]")
        return rec
    from pipeline import registry
    try:
        with console.status("[bold #38bdf8]Running model…[/bold #38bdf8]",
                            spinner="dots"):
            if tier_key == "tier2":
                return {"path": registry.fit(vendor, 2)}
            if tier_key == "tier3":
                from pipeline.tier3 import run_vendor
                return run_vendor(vendor)
            if tier_key == "tier3_simplis":
                from pipeline.tier3_simplis import run_vendor
                return run_vendor(vendor)
            if tier_key == "urom_tier2":
                from pipeline.unified_rom import run_unified
                return run_unified(vendor, emit="tier2")
            if tier_key == "urom_ann":
                from pipeline.unified_rom import run_unified
                return run_unified(vendor, emit="ann")
            if tier_key == "tier4":
                from pipeline.tier4 import run_vendor
                return run_vendor(vendor)
            if tier_key == "tier4_ann":
                from pipeline.tier4 import run_vendor_ann
                return run_vendor_ann(vendor)
    except Exception as exc:
        console.print(f"[bold red]Run failed:[/bold red] {exc}")
        return None
    raise ValueError(tier_key)


def _summary(record: dict[str, Any] | None, console: Console) -> None:
    if not record:
        return
    if "gates" in record and "vendor" in record:
        from cli.main import _unified_summary
        _unified_summary(record)
        return
    if "dpt_run" in record and "vendor" in record:
        from cli.main import _tier3_summary
        _tier3_summary(record)
        return
    if "energy_comparison" in record or "speedup" in record:
        from cli.main import _tier4_summary
        _tier4_summary(record)
        return
    console.print(record)


def _plot_files(vendor: str, tier_key: str) -> list[Path]:
    roots = [FIGURES / vendor]
    roots.append(FIGURES / "converter")
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.png")):
            if root.name == "converter" and vendor not in p.name:
                continue
            if tier_key in p.name or "dpt" in p.name or "buck" in p.name:
                out.append(p)
    return out


def plot_gallery(vendor: str, tier_key: str, console: Console) -> None:
    """Let the user open any measured plot produced for the vendor."""
    files = _plot_files(vendor, tier_key)
    if not files:
        console.print("[yellow]No plots found for this vendor/tier. "
                      "Run notebook 03 (DPT) or 04 (converter) first.[/yellow]")
        return
    picked = _ask(questionary.checkbox(
        "Open measured plots",
        choices=[questionary.Choice(str(p.relative_to(ROOT)), value=p)
                 for p in files],
        style=STYLE,
    ))
    for p in picked:
        console.print(f"[cyan]{p}[/cyan]")
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(p)], check=False)
            else:
                subprocess.run(["xdg-open", str(p)], check=False)
        except Exception as exc:
            console.print(f"[yellow]Could not open automatically: {exc}[/yellow]")


def _doctor(console: Console) -> None:
    table = Table(title="Environment", header_style="bold #38bdf8")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("SpiceModels/", "present" if SPICE_DIR.exists()
                  else "[red]missing[/red]")
    table.add_row("configured vendors", str(len(_configured_models())))
    table.add_row("model files", str(len(_model_files())))
    table.add_row("figures/", "present" if FIGURES.exists()
                  else "[yellow]absent[/yellow]")
    try:
        from simulation.engine import SpiceEngine
        SpiceEngine()
        table.add_row("LTspice", "[green]resolved[/green]")
    except Exception as exc:
        table.add_row("LTspice", f"[red]{exc}[/red]")
    console.print(table)


def run_wizard() -> None:
    """Launch the interactive workbench."""
    console = Console()
    banner(console)
    if not sys.stdin.isatty():
        console.print(
            "[yellow]Interactive mode needs a terminal.[/yellow] "
            "Use `--help` for scriptable commands.")
        return
    try:
        while True:
            action = _ask(questionary.select(
                "What do you want to do?",
                choices=[
                    "Run a model",
                    "Model catalog",
                    "Plot gallery",
                    "Doctor",
                    "Quit",
                ],
                style=STYLE,
            ))
            if action == "Quit":
                break
            if action == "Model catalog":
                _print_catalog(console)
                inspect_model(console)
                continue
            if action == "Doctor":
                _doctor(console)
                continue
            if action == "Plot gallery":
                vendor = _ask(questionary.select(
                    "Vendor",
                    choices=[m["vendor"] for m in _configured_models()],
                    style=STYLE,
                ))
                plot_gallery(vendor, "dpt", console)
                continue
            models = _model_choices()
            if not models:
                console.print("[yellow]No configured models.[/yellow] "
                              "Add a config/inputs/<vendor>.yaml.")
                continue
            chosen = _ask(questionary.select("Model", choices=models,
                                             style=STYLE))
            tier = _ask(questionary.select(
                "Tier",
                choices=[questionary.Choice(label, value=key)
                         for label, key in TIERS],
                style=STYLE,
            ))
            actions = _ask(questionary.checkbox(
                "Actions",
                choices=[
                    questionary.Choice("Build / validate", value="build",
                                       checked=True),
                    questionary.Choice("Show speed / benchmark rows",
                                       value="speed", checked=True),
                    questionary.Choice("Open plots", value="plots"),
                ],
                style=STYLE,
            ))
            rec = _run_tier(chosen["stem"], tier, build="build" in actions,
                            console=console)
            _summary(rec, console)
            if "speed" in actions and rec:
                for key in ("speed_adaptive_20", "speedup", "runtime"):
                    if rec.get(key):
                        console.print(f"[bold]recorded {key}:[/bold] {rec[key]}")
            if "plots" in actions:
                plot_gallery(chosen["vendor"], tier, console)
    except KeyboardInterrupt:
        console.print("\n[#94a3b8]Goodbye.[/#94a3b8]")
    except (EOFError, OSError) as exc:
        console.print(
            "[yellow]Interactive input is unavailable in this terminal "
            f"({exc}). Use `--help` for scriptable commands.[/yellow]")
    except Exception as exc:
        if exc.__class__.__name__ == "NoConsoleScreenBufferError":
            console.print(
                "[yellow]Interactive input is unavailable in this terminal "
                "(no console screen buffer). Use `--help` for scriptable "
                "commands.[/yellow]")
        else:
            raise
