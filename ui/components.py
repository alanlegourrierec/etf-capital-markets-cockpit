"""
Reusable interface pieces (tiles, cards, tables).

All styling goes through CSS classes (assets/style.css) that rest on theme
variables. Switching between dark and light needs no recalculation in Python
for these elements.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from dash import dcc, html

from ui.theme import GRAPH_CONFIG

Formatter = Union[str, Callable[[object], str]]
STATUS_ICON = {"good": "●", "warning": "▲", "critical": "■", "neutral": "○"}


# ------------------------------------------------------------------ formats
def num(v, digits: int = 1, signed: bool = False, suffix: str = "") -> str:
    """Readable number. Returns a dash when the value is missing."""
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return "–"
    return f"{v:{'+' if signed else ''},.{digits}f}{suffix}"


def bp(v, digits: int = 0, signed: bool = False) -> str:
    return num(v, digits, signed, " bp")


def money(v, digits: int = 1) -> str:
    """Amount in millions, shown in billions above 1,000."""
    if v is None or not np.isfinite(v):
        return "–"
    return f"{v / 1000:,.2f} bn" if abs(v) >= 1000 else f"{v:,.{digits}f} mm"


# --------------------------------------------------------------------- tiles
def kpi(label: str, value: str, sub: Optional[str] = None, tone: Optional[str] = None,
        note: Optional[str] = None) -> html.Div:
    """
    Key figure tile. `tone` adds a status marker (icon and label), the figure
    itself stays in the text colour.
    """
    children = [html.Div(label, className="kpi-label"), html.Div(value, className="kpi-value")]
    if sub or tone:
        bits = []
        if tone:
            bits.append(html.Span(STATUS_ICON.get(tone, "○"), className=f"dot dot-{tone}"))
        if sub:
            bits.append(html.Span(sub))
        children.append(html.Div(bits, className="kpi-sub"))
    if note:
        children.append(html.Div(note, className="kpi-note"))
    return html.Div(children, className="kpi")


def kpi_row(tiles: Sequence[html.Div]) -> html.Div:
    return html.Div(list(tiles), className="kpi-row")


# --------------------------------------------------------------------- cards
def card(title: str, children, span: int = 6, subtitle: Optional[str] = None) -> html.Div:
    head = [html.Div(title, className="card-title")]
    if subtitle:
        head.append(html.Div(subtitle, className="card-subtitle"))
    body = children if isinstance(children, (list, tuple)) else [children]
    return html.Div([html.Div(head, className="card-head"), *body], className=f"card span-{span}")


def graph(graph_id: str, height: int = 360) -> dcc.Graph:
    """
    Slot for a figure. The height is set here, on the container. In
    responsive mode Plotly fills its container and follows window resizes.
    """
    return dcc.Graph(id=graph_id, config=GRAPH_CONFIG, responsive=True, className="graph",
                     style={"height": f"{height}px"})


def chip(text: str, tone: str = "neutral") -> html.Span:
    return html.Span([html.Span(STATUS_ICON.get(tone, "○"), className=f"dot dot-{tone}"), text],
                     className="chip")


def control(label: str, component, wide: bool = False) -> html.Div:
    return html.Div([html.Div(label, className="control-label"), component],
                    className="control control-wide" if wide else "control")


# -------------------------------------------------------------------- tables
def table(df: pd.DataFrame, formats: Optional[Dict[str, Formatter]] = None,
          rename: Optional[Dict[str, str]] = None, index: bool = False,
          cell: Optional[Callable[[str, object], object]] = None) -> html.Table:
    """
    Plain HTML table. `formats` maps a column to a Python format ("{:.1f}") or
    to a function. `cell` can replace the content of a cell with a component,
    a status chip for instance.
    """
    formats, rename = formats or {}, rename or {}
    d = df.reset_index() if index else df
    numeric = {c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])}

    def render(col, v):
        if cell is not None:
            custom = cell(col, v)
            if custom is not None:
                return custom
        f = formats.get(col)
        if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
            return "–"
        if callable(f):
            return f(v)
        if isinstance(f, str):
            return f.format(v)
        return f"{v:,.2f}" if isinstance(v, (float, np.floating)) else str(v)

    head = html.Thead(html.Tr([html.Th(rename.get(c, str(c)),
                                       className="num" if c in numeric else "")
                               for c in d.columns]))
    rows = [html.Tr([html.Td(render(c, v), className="num" if c in numeric else "")
                     for c, v in zip(d.columns, row)])
            for row in d.itertuples(index=False, name=None)]
    return html.Table([head, html.Tbody(rows)], className="tbl")
