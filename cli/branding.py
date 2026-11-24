"""PieceSpice terminal branding.

The mark is a deliberately designed 26x18 bitmap rendered at two terminal
pixels per cell with ``▀``/``▄``.  It keeps the previous logo's essence:
chamfered die, ice-white P spine, cyan P loop and amber switching bolt.  A
fixed three-colour palette is used instead of converting every source pixel
to a terminal cell, which is what made the old render soft.

The landing composition uses a low-contrast circuit strip, a hero panel and a
three-step workbench strip; it is terminal-native and does not depend on an
image protocol.
"""

from importlib.metadata import PackageNotFoundError, version

from rich.align import Align
from rich.box import HEAVY
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

_CYAN = "#38bdf8"
_ICE = "#e2e8f0"
_LOOP = "#7dd3fc"
_GOLD = "#fbbf24"
_DIE = "#0b1220"
_DIM = "#64748b"

# 26x18 bitmap.  C=cyan die, D=die fill, W=ice P spine, B=cyan loop,
# G=gold bolt, .=transparent.  Two rows collapse into one terminal cell.
_MARK = [
    "..CCCCCCCCCCCCCCCCCCCCCC..",
    ".CCCCCCCCCCCCCCCCCCCCCCCC.",
    "CCDDDDDDDDDDDDDDDDDDDDDDCC",
    "CCDDDDDDDDDDDDDDDDDDDDDDCC",
    "CCDDDDWWBBBBBBDDDDDDDDDDCC",
    "CCDDDDWWBBBBBBDDDDDDDDDDCC",
    "CCDDDDWWDDDDBBDDDDDDDDDDCC",
    "CCDDDDWWDDDDBBDDDDDDDDDDCC",
    "CCDDDDWWDDDDBGGGGDDDDDDDCC",
    "CCDDDDWWBBBGGGGDDDDDDDDDCC",
    "CCDDDDWWDDDDGGGGDDDDDDDDCC",
    "CCDDDDWWDDDDDDGGGGDDDDDDCC",
    "CCDDDDWWDDDDGGGGDDDDDDDDCC",
    "CCDDDDWWDDDDDGGDDDDDDDDDCC",
    "CCDDDDDDDDDDDDDDDDDDDDDDCC",
    "CCDDDDDDDDDDDDDDDDDDDDDDCC",
    ".CCCCCCCCCCCCCCCCCCCCCCCC.",
    "..CCCCCCCCCCCCCCCCCCCCCC..",
]
_PALETTE = {"C": _CYAN, "D": _DIE, "W": _ICE, "B": _LOOP, "G": _GOLD}


def _mark_markup() -> str:
    """Render the two-row bitmap into crisp half-block markup."""
    lines: list[str] = []
    for y in range(0, len(_MARK), 2):
        line: list[str] = []
        for x in range(len(_MARK[0])):
            top = _MARK[y][x]
            bottom = _MARK[y + 1][x]
            if top == "." and bottom == ".":
                line.append(" ")
            elif top == bottom:
                line.append(f"[{_PALETTE[top]}]█[/]")
            elif bottom == ".":
                line.append(f"[{_PALETTE[top]}]▀[/]")
            elif top == ".":
                line.append(f"[{_PALETTE[bottom]}]▄[/]")
            else:
                line.append(
                    f"[{_PALETTE[top]} on {_PALETTE[bottom]}]▀[/]")
        lines.append("".join(line))
    return "\n".join(lines)


def _version() -> str:
    try:
        return version("piecespice")
    except PackageNotFoundError:
        return "dev"


def _circuit() -> Text:
    """A very faint circuit trace, not a competing visual element."""
    return Text.from_markup(
        "[#172a45]──◇────────●──────────◇──────────╋──────────●────────◇──[/]"
    )


def _wordmark() -> Text:
    return Text.assemble(
        ("PIECE", "bold #e2e8f0"),
        ("SPICE", "bold #38bdf8"),
        ("  /  ", "#475569"),
        ("BEHAVIORAL ROM STUDIO", "bold #fbbf24"),
    )


def _steps(console: Console) -> Table:
    table = Table.grid(expand=True, padding=(0, 1))
    width = max(18, (console.width - 8) // 3)
    for _ in range(3):
        table.add_column(width=width)
    cells = []
    for idx, title, detail in (
        ("01", "SELECT MODEL", "configured vendors"),
        ("02", "CHOOSE TIER", "C0 · ANN · T4"),
        ("03", "INSPECT", "plots · speed"),
    ):
        cells.append(Panel(
            Text.assemble(
                (idx + "  ", "bold #fbbf24"),
                (title + "\n", "bold #e2e8f0"),
                (detail, "#94a3b8"),
            ),
            border_style="#253b59",
            box=HEAVY,
            padding=(0, 1),
            width=width,
        ))
    table.add_row(*cells)
    return table


def banner(console: Console, *, vendor_count: int | None = None) -> None:
    """Render the branded landing screen."""
    vend = 0 if vendor_count is None else vendor_count
    art = Text.from_markup(_mark_markup())
    hero = Table.grid(expand=True, padding=(0, 2))
    hero.add_column(no_wrap=True)
    hero.add_column(ratio=1)
    hero.add_row(
        Align.center(art),
        Group(
            _wordmark(),
            Text("measurement-first · SPICE-syntax · one model at a time",
                 style="#94a3b8"),
            Text(""),
            Text.assemble(
                ("● ", "#4ade80"), (f"{vend} configured vendors", "#cbd5e1"),
                ("   │   ", "#334155"), ("LTspice backend", "#cbd5e1"),
                ("   │   ", "#334155"), (f"v{_version()}", _DIM),
            ),
        ),
    )
    status = Text.assemble(
        ("● ", "#4ade80"), ("READY", "bold #e2e8f0"),
        ("   ┃   ", "#334155"), ("INTERACTIVE WORKBENCH", "#94a3b8"),
        ("   ┃   ", "#334155"), ("ESC/Q QUIT", "#64748b"),
    )
    console.print(status)
    console.print(_circuit())
    console.print(Panel(
        hero,
        title="[bold #fbbf24]PIECESPICE[/bold #fbbf24]",
        subtitle="[#64748b]interactive model workbench[/#64748b]",
        border_style="#1d4ed8",
        box=HEAVY,
        padding=(0, 2),
    ))
    console.print(_steps(console))
    console.print(_circuit())
    console.print(
        "[#64748b]↑↓ move   enter select   esc/q quit   ctrl-c cancel[/#64748b]"
    )
    console.print(Rule(style="#152a45"))
