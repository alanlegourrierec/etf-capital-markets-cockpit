"""
Palette and Plotly styling, in a dark and a light version.

Colour rules used across the app.
  categorical  fixed order, the colour follows the entity (the fund is always
               blue, its reference always orange, the premium always green)
  sequential   a single hue, from pale to dark, for magnitudes
  diverging    blue and red around a neutral grey, for signs. Red always
               marks the stressed side, meaning a wide spread, a discount,
               a loss or a redemption
  status       green, amber and red are kept for states, never for series

The palettes were checked for hue gaps that colour-blind readers can tell
apart and for enough contrast on each of the two backgrounds.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

THEMES: Dict[str, Dict[str, object]] = {
    "dark": {
        "page": "#0d0d0d", "surface": "#1a1a19", "raised": "#232321",
        "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "cat": ["#3987e5", "#d95926", "#199e70", "#c98500",
                "#d55181", "#0ca30c", "#9085e9", "#e66767"],
        "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "pos": "#3987e5", "neg": "#e66767", "mid": "#383835",
        "pos_strong": "#86b6ef", "neg_strong": "#f2a3a3",
        "pos_weak": "#1c5cab", "neg_weak": "#a33030",
        "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
    },
    "light": {
        "page": "#f9f9f7", "surface": "#fcfcfb", "raised": "#ffffff",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "cat": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#6250d6", "#e34948"],
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "pos": "#2a78d6", "neg": "#e34948", "mid": "#f0efec",
        "pos_strong": "#184f95", "neg_strong": "#a82a2a",
        "pos_weak": "#9ec5f4", "neg_weak": "#f3b1b0",
        "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
    },
}

# Fixed roles of the categorical colours (the colour follows the entity)
FUND, REFERENCE, PREMIUM, FOURTH = 0, 1, 2, 3


def colors(theme: str) -> Dict[str, object]:
    return THEMES.get(theme, THEMES["dark"])


def cat(theme: str, slot: int) -> str:
    return colors(theme)["cat"][slot]            # type: ignore[index]


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def seq_scale(theme: str) -> List[List[object]]:
    """Sequential scale. Low values fade into the background, high values stand out."""
    steps = colors(theme)["seq"]
    n = len(steps) - 1                           # type: ignore[arg-type]
    return [[i / n, c] for i, c in enumerate(steps)]   # type: ignore[arg-type]


def div_scale(theme: str, red_high: bool = True) -> List[List[object]]:
    """
    Diverging scale, blue to neutral grey to red.
    red_high=True   positive values in red (spread z-score, stress)
    red_high=False  positive values in blue (gain, premium)
    """
    c = colors(theme)
    blue = [c["pos_strong"], c["pos"], c["pos_weak"]]
    red = [c["neg_weak"], c["neg"], c["neg_strong"]]
    lo, hi = (blue, red) if red_high else (red[::-1], blue[::-1])
    stops = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    return [list(t) for t in zip(stops, [lo[0], lo[1], lo[2], c["mid"], hi[0], hi[1], hi[2]])]


def ordinal(theme: str, n: int) -> List[str]:
    """n shades of one hue, from the quietest to the strongest."""
    steps = colors(theme)["seq"]
    usable = steps[2:]                           # shades too close to the background are left out
    if n >= len(usable):
        return list(usable)
    idx = [round(i * (len(usable) - 1) / max(n - 1, 1)) for i in range(n)]
    return [usable[i] for i in idx]


def style(fig: go.Figure, theme: str, height: Optional[int] = None, legend: bool = True,
          hovermode: str = "x unified", spikes: bool = True, margin: Optional[dict] = None) -> go.Figure:
    """Applies the common styling (background, type, quiet grid, legend, hover)."""
    c = colors(theme)
    fig.update_layout(
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
        font=dict(family=FONT, size=12, color=c["ink2"]),
        colorway=c["cat"],
        margin=margin or dict(l=54, r=18, t=40 if legend else 16, b=38),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="left", x=0,
                    font=dict(size=11, color=c["ink2"]), bgcolor="rgba(0,0,0,0)",
                    itemsizing="constant", traceorder="normal"),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor=c["raised"], bordercolor=c["axis"],
                        font=dict(family=FONT, size=12, color=c["ink"])),
        uirevision="keep",
        bargap=0.25, bargroupgap=0.12,
    )
    if height:
        fig.update_layout(height=height)
    tick = dict(size=11, color=c["muted"])
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=c["axis"], tickcolor=c["axis"],
                     ticks="outside", ticklen=3, tickfont=tick,
                     title_font=dict(size=11, color=c["muted"]), title_standoff=6)
    fig.update_yaxes(gridcolor=c["grid"], gridwidth=1, zeroline=False, showline=False,
                     tickfont=tick, title_font=dict(size=11, color=c["muted"]), title_standoff=6)
    if spikes and hovermode.startswith("x"):
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                         spikethickness=1, spikedash="solid", spikecolor=c["muted"])
    fig.update_annotations(font=dict(family=FONT, size=11, color=c["muted"]))
    return fig


def panel_label(fig: go.Figure, text: str, row: int, theme: str) -> None:
    """Small label at the top left of a panel (figures stacked in one column)."""
    idx = "" if row == 1 else str(row)
    fig.add_annotation(text=text, xref="paper", yref=f"y{idx} domain", x=0.0, y=1.0,
                       xanchor="left", yanchor="bottom", showarrow=False, yshift=2,
                       font=dict(family=FONT, size=11, color=colors(theme)["muted"]))


def sync_hover(fig: go.Figure, rows: int) -> go.Figure:
    """
    Synchronises hover across stacked panels. Every series is attached to the
    bottom horizontal axis, so one vertical cursor reads all panels at once.
    Lines and labels anchored on the axes left unused are attached to the
    width of the figure.
    """
    fig.update_traces(xaxis="x" if rows == 1 else f"x{rows}")
    for item in list(fig.layout.shapes) + list(fig.layout.annotations):
        if str(item.xref).endswith("domain"):
            item.xref = "paper"
    fig.update_layout(hoversubplots="axis", hovermode="x unified")
    return fig


def empty(theme: str, message: str, height: int = 260) -> go.Figure:
    """Empty figure carrying a message (data missing or not configured)."""
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    style(fig, theme, height=height, legend=False)
    fig.update_annotations(font=dict(family=FONT, size=13, color=colors(theme)["muted"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


GRAPH_CONFIG = {"displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def log_ticks(lo: float, hi: float) -> list:
    """1-2-5 ticks for a log axis, easier to read than the defaults."""
    import math
    if not (lo > 0 and hi > lo):
        return []
    out = []
    for exp in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** exp
            if lo * 0.9 <= v <= hi * 1.1:
                out.append(v)
    return out
