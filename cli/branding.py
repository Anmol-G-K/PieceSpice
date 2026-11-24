"""PieceSpice terminal branding.

The mark is a deliberately designed 20x20 bitmap rendered at two terminal
pixels per cell with ``▀``/``▄``.  A 20x20 bitmap is physically square in a
normal terminal cell (one cell is approximately twice as tall as it is wide):
20 columns x 10 rows.  That is the same proportion as the original die mark
and small enough for a short terminal; an earlier 26x18 version was 1.44:1
and read as a wide banner rather than a logo.

The mark uses a solid ice-white P, a separate gold zigzag bolt and a cyan die
rim.  The P loop is deliberately not cyan: it disappeared into the rim at
small sizes.  The bolt is drawn as a connected 2-3 px zigzag instead of the
old 1 px outline, which broke into disconnected dots.
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
_GOLD = "#fbbf24"
_DIE = "#0f1f35"
_DIM = "#64748b"

# 20x20 bitmap.  C=cyan die, D=die fill, W=ice P, G=gold bolt,
# .=transparent.  Two rows collapse into one terminal cell.
_MARK = [
    "..CCCCCCCCCCCCCCCC..",
    ".CCCCCCCCCCCCCCCCCC.",
    "CCDDDDDDDDDDDDDDDDCC",
    "CCWWWWWWWWDDDDGGGDCC",
    "CCWWWWWWWWDDDGGGDDCC",
    "CCWWWDDDWWDDGGGDDDCC",
    "CCWWWDDDWWDGGGDDDDCC",
    "CCWWWDDDWWDDGGGDDDCC",
    "CCWWWDDDWWDDDGGGDDCC",
    "CCWWWWWWWWDDDDGGGDCC",
    "CCWWWWWWWWDDDGGGDDCC",
    "CCWWWDDDDDDDGGGDDDCC",
    "CCWWWDDDDDDGGGDDDDCC",
    "CCWWWDDDDDDDGGGDDDCC",
    "CCWWWDDDDDDDDGGGDDCC",
    "CCWWWDDDDDDDDDGGDDCC",
    "CCWWWDDDDDDDDGGDDDCC",
    "CCDDDDDDDDDDDDDDDDCC",
    ".CCCCCCCCCCCCCCCCCC.",
    "..CCCCCCCCCCCCCCCC..",
]
_PALETTE = {"C": _CYAN, "D": _DIE, "W": _ICE, "G": _GOLD}


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
        ("BEHAVIORAL", "bold #e2e8f0"),
        (" ROM STUDIO", "bold #fbbf24"),
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
                ("● ", "#4ade80"), (f"{vend} vendors", "#cbd5e1"),
                ("   │   ", "#334155"), ("LTspice", "#cbd5e1"),
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
