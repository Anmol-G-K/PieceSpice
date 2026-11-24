"""PieceSpice terminal branding.

The mark is a compact Unicode-block rendering of the die/faceted-P artwork:
die rim in cyan, P spine in ice white, loop in cyan, switching bolt in gold.
It is intentionally terminal-native (no image protocol dependency).
"""

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

LOGO = r"""
[#38bdf8]        ▄▄▄▄▄▄▄▄▄▄▄▄▄▄        [/]
[#38bdf8]      █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█      [/]
[#38bdf8]      █  [#e2e8f0]████[/][#38bdf8]        █      [/]
[#38bdf8]      █  [#e2e8f0]██[/][#38bdf8]    [#fbbf24]▄▄▄[/][#38bdf8]  █      [/]
[#38bdf8]      █  [#e2e8f0]██████[/][#38bdf8]    █      [/]
[#38bdf8]      █  [#e2e8f0]██[/][#38bdf8]  [#fbbf24]▄▄███[/][#38bdf8]   █      [/]
[#38bdf8]      █  [#e2e8f0]██[/][#38bdf8]  [#fbbf24]██▀▀▀[/][#38bdf8]   █      [/]
[#38bdf8]      █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█      [/]
[#38bdf8]        ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀        [/]
"""


def banner(console: Console) -> None:
    """Render the branded welcome panel."""
    art = Text.from_markup(LOGO.strip("\n"))
    title = Text.from_markup(
        "[bold #e2e8f0]PieceSpice[/bold #e2e8f0]  "
        "[#38bdf8]behavioral ROM studio[/#38bdf8]\n"
        "[#94a3b8]SPICE in · measured curves · piecewise models out[/#94a3b8]"
    )
    console.print(Panel(
        Align.center(art + Text("\n") + title),
        border_style="#38bdf8",
        title="[bold #fbbf24]PIECESPICE[/bold #fbbf24]",
        subtitle="[#94a3b8]interactive model workbench[/#94a3b8]",
        padding=(0, 2),
    ))
