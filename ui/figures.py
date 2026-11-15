"""
Every Plotly figure of the dashboard.

Each function takes data already computed by the engine and a theme ("dark"
or "light"), and returns a figure. No business logic here.

Chart conventions. One vertical axis per panel (measures on different scales
are stacked in synchronised panels), thin marks, a quiet grid, a legend as
soon as there are two series, and direct labels kept for the points that
matter.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import curves, etf_analytics as ea
from ui.theme import (FOURTH, FUND, PREMIUM, REFERENCE, cat, colors, div_scale, empty,
                      log_ticks, ordinal, panel_label, rgba, seq_scale, style,
                      sync_hover)

SYMBOLS = {"Sr Unsecured": "circle", "Sr Secured": "square", "Subordinated": "diamond"}


def _line(x, y, name, color, width=2.0, dash=None, fmt="%{y:.2f}", **kw) -> go.Scatter:
    return go.Scatter(x=x, y=y, name=name, mode="lines",
                      line=dict(color=color, width=width, dash=dash),
                      hovertemplate=fmt + "<extra>" + name + "</extra>", **kw)


def _band(fig, x, upper, lower, name, color, row=None, col=None, alpha=0.13) -> None:
    """Filled band between two series (Bollinger, confidence interval)."""
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), **kw)
    fig.add_trace(go.Scatter(x=x, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba(color, alpha), name=name, hoverinfo="skip"), **kw)


def _colorbar(theme: str, title: str) -> dict:
    c = colors(theme)
    return dict(thickness=10, len=0.85, outlinewidth=0, tickfont=dict(size=10, color=c["muted"]),
                title=dict(text=title, side="right", font=dict(size=11, color=c["muted"])))


# ============================================================================
# 1. Premium and discount
# ============================================================================
def premium_panel(daily: pd.DataFrame, boll: pd.DataFrame, flow_musd: pd.Series,
                  theme: str) -> go.Figure:
    """Four synchronised panels with price and NAV, premium, volume and primary flows."""
    c = colors(theme)
    x = daily.index
    pdb = ea.premium_discount(daily)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        row_heights=[0.34, 0.30, 0.18, 0.18])
    fig.add_trace(_line(x, daily["price"], "ETF price", cat(theme, FUND), fmt="%{y:.3f}"), 1, 1)
    fig.add_trace(_line(x, daily["nav"], "NAV", cat(theme, REFERENCE), fmt="%{y:.3f}"), 1, 1)

    _band(fig, x, boll["upper"], boll["lower"], "Bollinger band, 20d ±2σ", c["muted"], 2, 1)
    fig.add_trace(_line(x, boll["mid"], "20d mean", c["muted"], 1.0, fmt="%{y:.1f} bp",
                        showlegend=False), 2, 1)
    fig.add_trace(_line(x, pdb, "Premium / discount", cat(theme, PREMIUM), fmt="%{y:.1f} bp"), 2, 1)
    out = ((pdb > boll["upper"]) | (pdb < boll["lower"])).fillna(False)
    fig.add_trace(go.Scatter(x=x[out], y=pdb[out], mode="markers", name="Outside the band",
                             marker=dict(size=8, color=c["ink"],
                                         line=dict(width=2, color=c["surface"])),
                             hovertemplate="%{y:.1f} bp<extra>Outside the band</extra>"), 2, 1)
    fig.add_hline(y=0, line_color=c["axis"], line_width=1, row=2, col=1)

    fig.add_trace(go.Bar(x=x, y=daily["value_musd"], name="Traded value",
                         marker=dict(color=rgba(c["ink2"], 0.55), line_width=0),
                         hovertemplate="%{y:.1f} mm<extra>Traded value</extra>"), 3, 1)
    fig.add_trace(go.Bar(x=x, y=flow_musd.clip(lower=0), name="Creations",
                         marker=dict(color=c["pos"], line_width=0),
                         hovertemplate="%{y:.1f} mm<extra>Creations</extra>"), 4, 1)
    fig.add_trace(go.Bar(x=x, y=flow_musd.clip(upper=0), name="Redemptions",
                         marker=dict(color=c["neg"], line_width=0),
                         hovertemplate="%{y:.1f} mm<extra>Redemptions</extra>"), 4, 1)
    style(fig, theme, height=660)
    fig.update_layout(barmode="relative", bargap=0.1)
    sync_hover(fig, 4)
    for row, text in enumerate(["Premium (+) or discount (-) to NAV, bp",
                                "Secondary market traded value, mm",
                                "Primary market flows, mm"], start=2):
        panel_label(fig, text, row, theme)
    return fig


def pd_histogram(pdb: pd.Series, theme: str) -> go.Figure:
    """Distribution of the premium over the window, with today's level."""
    c = colors(theme)
    s = pdb.dropna()
    fig = go.Figure(go.Histogram(x=s, nbinsx=36, marker=dict(color=cat(theme, PREMIUM), line_width=0),
                                 hovertemplate="%{x} bp<br>%{y} days<extra></extra>"))
    fig.add_vline(x=float(s.iloc[-1]), line_color=c["ink"], line_width=1.5,
                  annotation_text=f"Today {s.iloc[-1]:+.0f} bp",
                  annotation_position="top right" if s.iloc[-1] >= s.mean() else "top left")
    fig.add_vline(x=float(s.mean()), line_color=c["muted"], line_width=1, line_dash="dash",
                  annotation_text=f"Mean {s.mean():+.0f} bp",
                  annotation_position="top left" if s.iloc[-1] >= s.mean() else "top right")
    style(fig, theme, height=250, legend=False, hovermode="closest", margin=dict(l=54, r=18, t=30, b=38))
    fig.update_layout(bargap=0.08)
    fig.update_xaxes(title_text="Premium / discount, bp")
    fig.update_yaxes(title_text="Days")
    return fig


def intraday_panel(intra: Optional[pd.DataFrame], theme: str) -> go.Figure:
    """Today's session with price, iNAV and bid/ask, then the intraday premium."""
    if intra is None or len(intra) < 2:
        return empty(theme, "No intraday iNAV feed. Set inav_ric in config.ETFS to enable this panel.", 330)
    c = colors(theme)
    x = intra.index
    prem = (intra["price"] / intra["inav"] - 1.0) * 1e4
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.6, 0.4])
    _band(fig, x, intra["ask"], intra["bid"], "Bid / ask", cat(theme, FUND), 1, 1, 0.14)
    fig.add_trace(_line(x, intra["price"], "ETF price", cat(theme, FUND), fmt="%{y:.3f}"), 1, 1)
    fig.add_trace(_line(x, intra["inav"], "iNAV", cat(theme, REFERENCE), fmt="%{y:.3f}"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=prem, name="Intraday premium", mode="lines",
                             line=dict(color=cat(theme, PREMIUM), width=2), fill="tozeroy",
                             fillcolor=rgba(cat(theme, PREMIUM), 0.10),
                             hovertemplate="%{y:.1f} bp<extra>Intraday premium</extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=[x[-1]], y=[prem.iloc[-1]], mode="markers+text",
                             text=[f"{prem.iloc[-1]:+.1f} bp"], textposition="middle left",
                             textfont=dict(color=c["ink"], size=12), showlegend=False,
                             marker=dict(size=9, color=cat(theme, PREMIUM),
                                         line=dict(width=2, color=c["surface"])),
                             hoverinfo="skip"), 2, 1)
    style(fig, theme, height=350)
    sync_hover(fig, 2)
    panel_label(fig, "Premium to iNAV, bp", 2, theme)
    return fig


# ============================================================================
# 2. Authorized Participant arbitrage
# ============================================================================
def arb_bands(bands: pd.DataFrame, theme: str) -> go.Figure:
    """Premium against creation and redemption thresholds, open windows, net edge."""
    c = colors(theme)
    x = bands.index
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.72, 0.28])
    _band(fig, x, bands["upper"], bands["lower"], "No-arbitrage band", c["muted"], 1, 1, 0.12)
    fig.add_trace(_line(x, bands["upper"], "Creation threshold", c["ink2"], 1.2, "dash",
                        "%{y:.1f} bp"), 1, 1)
    fig.add_trace(_line(x, bands["lower"], "Redemption threshold", c["ink2"], 1.2, "dot",
                        "%{y:.1f} bp"), 1, 1)
    fig.add_trace(_line(x, bands["pd_bps"], "Premium / discount", cat(theme, PREMIUM),
                        fmt="%{y:.1f} bp"), 1, 1)
    for sig, name, col in ((1, "Creation window", c["pos"]), (-1, "Redemption window", c["neg"])):
        m = bands["signal"] == sig
        fig.add_trace(go.Scatter(x=x[m], y=bands.loc[m, "pd_bps"], mode="markers", name=name,
                                 marker=dict(size=8, color=col, line=dict(width=2, color=c["surface"])),
                                 hovertemplate="%{y:.1f} bp<extra>" + name + "</extra>"), 1, 1)
    edge_signed = bands["edge_bps"] * np.sign(bands["signal"])
    fig.add_trace(go.Bar(x=x, y=edge_signed.clip(lower=0), marker=dict(color=c["pos"], line_width=0),
                         showlegend=False, hovertemplate="%{y:.1f} bp<extra>Creation edge</extra>"), 2, 1)
    fig.add_trace(go.Bar(x=x, y=edge_signed.clip(upper=0), marker=dict(color=c["neg"], line_width=0),
                         showlegend=False, hovertemplate="%{y:.1f} bp<extra>Redemption edge</extra>"), 2, 1)
    style(fig, theme, height=520)
    fig.update_layout(barmode="relative", bargap=0.0)
    sync_hover(fig, 2)
    panel_label(fig, "Net edge when a window is open, bp (creation up, redemption down)", 2, theme)
    return fig


def cost_stack(stack: pd.DataFrame, theme: str) -> go.Figure:
    """Cost stack of a creation and of a redemption, in bp."""
    c = colors(theme)
    s = stack.iloc[::-1]
    fig = go.Figure()
    for col, name, color in (("creation", "Creation", c["pos"]), ("redemption", "Redemption", c["neg"])):
        fig.add_trace(go.Bar(y=s.index, x=s[col], name=name, orientation="h", width=0.3,
                             marker=dict(color=color, cornerradius=4, line_width=0),
                             text=[f"{v:.1f}" for v in s[col]], textposition="outside",
                             textfont=dict(color=c["ink2"], size=11), cliponaxis=False,
                             hovertemplate="%{x:.1f} bp<extra>" + name + "</extra>"))
    style(fig, theme, height=330, hovermode="closest", margin=dict(l=150, r=40, t=38, b=38))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.1)
    fig.update_xaxes(title_text="Cost, bp of order notional", showgrid=True, gridcolor=c["grid"])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12, color=c["ink2"]))
    return fig


def arb_grid(grid: pd.DataFrame, theme: str) -> go.Figure:
    """Net edge of the arbitrage by order size and premium level."""
    c = colors(theme)
    z = grid.to_numpy()
    lim = float(np.nanmax(np.abs(z)))
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(s) for s in grid.columns], y=[f"{v:+.0f}" for v in grid.index],
        colorscale=div_scale(theme, red_high=False), zmid=0, zmin=-lim, zmax=lim,
        xgap=2, ygap=2, text=np.round(z).astype(int), texttemplate="%{text}",
        textfont=dict(size=10), colorbar=_colorbar(theme, "Net edge, bp"),
        hovertemplate="Order %{x} CU<br>Premium %{y} bp<br>Net edge %{z:.1f} bp<extra></extra>"))
    style(fig, theme, height=470, legend=False, hovermode="closest", margin=dict(l=58, r=10, t=14, b=44))
    fig.update_xaxes(title_text="Order size, creation units", type="category", ticks="")
    fig.update_yaxes(title_text="Premium (+) / discount (-), bp", type="category", showgrid=False)
    return fig


# ============================================================================
# 3. Liquidity
# ============================================================================
def liquidity_scatter(basket: pd.DataFrame, theme: str) -> go.Figure:
    """Each bond of the basket by daily volume, bid/ask, weight and seniority."""
    c = colors(theme)
    fig = go.Figure()
    sizeref = 2.0 * basket["weight"].max() * 100 / (26.0 ** 2)
    for slot, sen in enumerate(["Sr Unsecured", "Sr Secured", "Subordinated"]):
        g = basket[basket["seniority"] == sen]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g["adv_musd"], y=g["ba_bps"], mode="markers", name=sen,
            marker=dict(size=g["weight"] * 100, sizemode="area", sizeref=sizeref, sizemin=4,
                        color=cat(theme, slot), opacity=0.9, line=dict(width=1.5, color=c["surface"])),
            customdata=np.column_stack([g["bond_id"], g["rating"], g["weight"] * 100,
                                        g["liq_score"], g["amt_out_musd"]]),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>ADV %{x:.1f} mm · "
                          "bid/ask %{y:.0f} bp<br>Weight %{customdata[2]:.2f}% · "
                          "liquidity score %{customdata[3]:.0f}<br>Outstanding %{customdata[4]:,.0f} mm"
                          "<extra></extra>"))
    style(fig, theme, height=400, hovermode="closest")
    fig.update_xaxes(title_text="Average daily volume, mm (log scale)", showgrid=True,
                     gridcolor=c["grid"], **_log_axis(basket["adv_musd"]))
    fig.update_yaxes(title_text="Bid / ask, bp of price")
    return fig


def heatmap(pivot: pd.DataFrame, theme: str, title: str, diverging: bool = False,
            red_high: bool = True, fmt: str = ".0f", height: int = 360,
            x_title: str = "", zlim: Optional[float] = None) -> go.Figure:
    """Generic heatmap, sequential (magnitude) or diverging (sign)."""
    c = colors(theme)
    z = pivot.to_numpy(dtype=float)
    kw: Dict[str, object] = {}
    if diverging:
        lim = zlim or max(float(np.nanmax(np.abs(z))), 1e-9)
        kw = dict(colorscale=div_scale(theme, red_high), zmid=0, zmin=-lim, zmax=lim)
    else:
        kw = dict(colorscale=seq_scale(theme))
    text = np.where(np.isfinite(z), np.vectorize(lambda v: format(v, fmt))(np.nan_to_num(z)), "")
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(v) for v in pivot.columns], y=[str(v) for v in pivot.index],
        xgap=2, ygap=2, text=text, texttemplate="%{text}",
        textfont=dict(size=11), colorbar=_colorbar(theme, title),
        hoverongaps=False,
        hovertemplate="%{y} · %{x}<br>" + title + " %{z:" + fmt + "}<extra></extra>", **kw))
    style(fig, theme, height=height, legend=False, hovermode="closest",
          margin=dict(l=96, r=10, t=14, b=44))
    fig.update_xaxes(type="category", ticks="", title_text=x_title)
    fig.update_yaxes(type="category", showgrid=False, autorange="reversed",
                     tickfont=dict(size=11, color=c["ink2"]))
    return fig


def primary_secondary(ps: pd.DataFrame, theme: str) -> go.Figure:
    """Each session, ETF traded value against basket capacity, coloured by premium."""
    c = colors(theme)
    d = ps.dropna()
    lim = float(np.nanpercentile(np.abs(d["pd_bps"]), 98)) or 1.0
    fig = go.Figure(go.Scatter(
        x=d["etf_value_musd"], y=d["basket_capacity_musd"], mode="markers",
        marker=dict(size=7, color=d["pd_bps"], colorscale=div_scale(theme, red_high=False),
                    cmid=0, cmin=-lim, cmax=lim, opacity=0.85,
                    line=dict(width=1, color=c["surface"]),
                    colorbar=_colorbar(theme, "Premium, bp")),
        customdata=np.column_stack([d.index.strftime("%d %b %Y"), d["pd_bps"]]),
        hovertemplate="%{customdata[0]}<br>ETF traded %{x:.1f} mm<br>Basket capacity %{y:.1f} mm"
                      "<br>Premium %{customdata[1]:.0f} bp<extra></extra>", showlegend=False))
    last = d.iloc[-1]
    fig.add_trace(go.Scatter(x=[last["etf_value_musd"]], y=[last["basket_capacity_musd"]],
                             mode="markers+text", text=["Today"], textposition="top center",
                             textfont=dict(color=c["ink"], size=12), showlegend=False,
                             marker=dict(size=13, color="rgba(0,0,0,0)",
                                         line=dict(width=2, color=c["ink"])), hoverinfo="skip"))
    style(fig, theme, height=400, legend=False, hovermode="closest")
    fig.update_xaxes(title_text="ETF secondary traded value, mm (log scale)", showgrid=True,
                     gridcolor=c["grid"], **_log_axis(d["etf_value_musd"]))
    fig.update_yaxes(title_text="Basket one-day capacity, mm (log scale)",
                     **_log_axis(d["basket_capacity_musd"]))
    return fig


def liquidation_profile(profile: pd.DataFrame, theme: str) -> go.Figure:
    """Share of the basket that trades in 1, 3 or 5 sessions, by order size."""
    c = colors(theme)
    shades = ordinal(theme, len(profile.columns))
    labels = {"1d": "Within 1 day", "3d": "Within 3 days", "5d": "Within 5 days"}
    fig = go.Figure()
    for col, shade in zip(profile.columns, shades):
        fig.add_trace(go.Scatter(x=profile.index, y=profile[col], name=labels.get(col, col),
                                 mode="lines+markers", line=dict(color=shade, width=2),
                                 marker=dict(size=8, color=shade, line=dict(width=2, color=c["surface"])),
                                 hovertemplate="%{y:.0f}%<extra>" + labels.get(col, col) + "</extra>"))
    style(fig, theme, height=360)
    fig.update_xaxes(type="log", title_text="Primary order size, mm (log scale)",
                     tickvals=list(profile.index), ticktext=[f"{v:g}" for v in profile.index],
                     minor=dict(ticks=""))
    fig.update_yaxes(title_text="Share of basket executed, %", range=[0, 104])
    return fig


# ============================================================================
# 4. Capital structure and relative value
# ============================================================================
def merton_scatter(snap: pd.DataFrame, coef: pd.Series, theme: str,
                   selected: Optional[str] = None) -> go.Figure:
    """Distance to default against market spread, coloured by the gap to fair value."""
    c = colors(theme)
    s = snap[~snap["excluded"]]
    grid = np.linspace(max(s["dd"].min() * 0.9, 0.3), s["dd"].max() * 1.05, 80)
    fair = np.exp(coef["a"] + coef["b"] * np.log1p(grid))
    fig = go.Figure()
    fig.add_trace(_line(grid, fair, "Fair value from distance to default", c["ink2"], 1.5,
                        fmt="%{y:.0f} bp"))
    lim = max(float(np.nanmax(np.abs(s["ts_z"]))), 2.0)
    fig.add_trace(go.Scatter(
        x=s["dd"], y=s["mkt_spread"], mode="markers", name="Issuers",
        marker=dict(size=11, color=s["ts_z"], colorscale=div_scale(theme, red_high=True), cmid=0,
                    cmin=-lim, cmax=lim, line=dict(width=1.5, color=c["surface"]),
                    colorbar=_colorbar(theme, "Gap to fair value, z-score")),
        customdata=np.column_stack([s.index, s["name"], s["rating"], s["fair_spread"], s["ts_z"],
                                    s["leverage"] * 100, s["equity_vol"] * 100]),
        hovertemplate="<b>%{customdata[1]}</b> (%{customdata[0]}) · %{customdata[2]}<br>"
                      "DD %{x:.1f} · market %{y:.0f} bp · fair %{customdata[3]:.0f} bp<br>"
                      "z-score %{customdata[4]:+.1f} · leverage %{customdata[5]:.0f}% · "
                      "equity vol %{customdata[6]:.0f}%<extra></extra>"))
    top = s.loc[s["ts_z"].abs().sort_values(ascending=False).index].head(7)
    fig.add_trace(go.Scatter(x=top["dd"], y=top["mkt_spread"], mode="text", text=list(top.index),
                             textposition="top center", textfont=dict(size=11, color=c["ink2"]),
                             showlegend=False, hoverinfo="skip"))
    if selected in s.index:
        fig.add_trace(go.Scatter(x=[s.loc[selected, "dd"]], y=[s.loc[selected, "mkt_spread"]],
                                 mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=19, color="rgba(0,0,0,0)",
                                             line=dict(width=2, color=c["ink"]))))
    style(fig, theme, height=430, hovermode="closest")
    fig.update_xaxes(title_text="Distance to default, 1 year (standard deviations)",
                     showgrid=True, gridcolor=c["grid"])
    fig.update_yaxes(title_text="5Y credit spread, bp (log scale)", **_log_axis(s["mkt_spread"]))
    return fig


def issuer_detail(market: pd.Series, fair: pd.Series, z: pd.Series, theme: str) -> go.Figure:
    """One issuer through time, market spread against fair value, then the gap z-score."""
    c = colors(theme)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.62, 0.38])
    fig.add_trace(_line(market.index, market, "Market 5Y spread", cat(theme, FUND), fmt="%{y:.0f} bp"), 1, 1)
    fig.add_trace(_line(fair.index, fair, "Equity-implied fair spread", cat(theme, REFERENCE),
                        fmt="%{y:.0f} bp"), 1, 1)
    fig.add_trace(_line(z.index, z, "Gap z-score", cat(theme, PREMIUM), fmt="%{y:+.2f}"), 2, 1)
    for lvl in (-2, 2):
        fig.add_hline(y=lvl, line_color=c["muted"], line_width=1, line_dash="dash", row=2, col=1)
    fig.add_hline(y=0, line_color=c["axis"], line_width=1, row=2, col=1)
    style(fig, theme, height=430)
    sync_hover(fig, 2)
    panel_label(fig, "Gap to fair value, z-score over six months (dashes at ±2)", 2, theme)
    return fig


def pair_panel(ph: pd.DataFrame, a: str, b: str, theme: str) -> go.Figure:
    """One pair through time, spread ratio with its bands, then the z-score."""
    c = colors(theme)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.62, 0.38])
    x = ph.index
    _band(fig, x, ph["upper"], ph["lower"], "Six-month band, ±2σ", c["muted"], 1, 1)
    fig.add_trace(_line(x, ph["mean"], "Six-month mean", c["muted"], 1.0, fmt="%{y:.2f}x"), 1, 1)
    fig.add_trace(_line(x, ph["ratio"], f"{a} / {b} spread ratio", cat(theme, FUND), fmt="%{y:.2f}x"), 1, 1)
    fig.add_trace(_line(x, ph["z"], "z-score", cat(theme, PREMIUM), fmt="%{y:+.2f}"), 2, 1)
    for lvl in (-2, 2):
        fig.add_hline(y=lvl, line_color=c["muted"], line_width=1, line_dash="dash", row=2, col=1)
    fig.add_hline(y=0, line_color=c["axis"], line_width=1, row=2, col=1)
    style(fig, theme, height=430)
    sync_hover(fig, 2)
    panel_label(fig, "z-score of the log ratio (dashes at ±2)", 2, theme)
    return fig


# ============================================================================
# 5. Curves and seniority
# ============================================================================
def govt_curve(snaps: Dict[str, pd.Series], fit: Dict[str, float], theme: str, ccy: str) -> go.Figure:
    """Today's sovereign curve (Nelson-Siegel fit) and past curves."""
    c = colors(theme)
    fig = go.Figure()
    labels = list(snaps)
    shades = ordinal(theme, len(labels))[::-1]       # today gets the strongest shade
    taus = np.linspace(0.25, 30, 120)
    for label, shade in list(zip(labels, shades))[::-1]:
        s = snaps[label]
        today = label == labels[0]
        fig.add_trace(go.Scatter(
            x=np.asarray(s.index, dtype=float), y=s.to_numpy(), name=label,
            mode="markers" if today else "lines+markers",
            line=dict(color=shade, width=1.5), marker=dict(size=8 if today else 5, color=shade,
                                                           line=dict(width=2 if today else 0,
                                                                     color=c["surface"])),
            hovertemplate="%{y:.2f}%<extra>" + label + "</extra>"))
    fig.add_trace(_line(taus, curves.ns_curve(fit, taus), "Nelson-Siegel fit, today",
                        ordinal(theme, 4)[-1], 2.0, fmt="%{y:.2f}%"))
    style(fig, theme, height=360)
    fig.update_xaxes(title_text=f"{ccy} sovereign curve, maturity in years")
    fig.update_yaxes(title_text="Yield, %")
    return fig


def slopes_panel(slopes: pd.DataFrame, theme: str) -> go.Figure:
    """2s10s and 5s30s slopes, then the 2s5s10s butterfly."""
    c = colors(theme)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[0.62, 0.38])
    x = slopes.index
    fig.add_trace(_line(x, slopes["2s10s"], "2s10s", cat(theme, FUND), fmt="%{y:.0f} bp"), 1, 1)
    fig.add_trace(_line(x, slopes["5s30s"], "5s30s", cat(theme, REFERENCE), fmt="%{y:.0f} bp"), 1, 1)
    fig.add_trace(_line(x, slopes["2s5s10s fly"], "2s5s10s fly", cat(theme, PREMIUM),
                        fmt="%{y:.0f} bp"), 2, 1)
    fig.add_hline(y=0, line_color=c["axis"], line_width=1, row=1, col=1)
    fig.add_hline(y=0, line_color=c["axis"], line_width=1, row=2, col=1)
    style(fig, theme, height=360)
    sync_hover(fig, 2)
    panel_label(fig, "Butterfly, bp", 2, theme)
    return fig


def credit_curve(basket: pd.DataFrame, fits: Dict[str, Tuple[float, float]], theme: str) -> go.Figure:
    """Spread against maturity by rating bucket, with the fitted curve of each bucket."""
    c = colors(theme)
    fig = go.Figure()
    order = [b for b in ["AA+", "A", "BBB", "BB", "B", "CCC"] if b in set(basket["bucket"])]
    taus = np.linspace(max(basket["maturity_years"].min(), 0.5), basket["maturity_years"].max(), 60)
    for slot, bucket in enumerate(order[:4]):
        g = basket[basket["bucket"] == bucket]
        col = cat(theme, slot)
        fig.add_trace(go.Scatter(
            x=g["maturity_years"], y=g["oas"], mode="markers", name=bucket, legendgroup=bucket,
            marker=dict(size=9, color=col, symbol=g["seniority"].map(SYMBOLS).fillna("circle").tolist(),
                        line=dict(width=1.5, color=c["surface"])),
            customdata=np.column_stack([g["bond_id"], g["seniority"], g["curve_res"], g["rating"]]),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[3]} · %{customdata[1]}<br>"
                          "OAS %{y:.0f} bp at %{x:.1f}y<br>%{customdata[2]:+.0f} bp to the curve"
                          "<extra></extra>"))
        if bucket in fits:
            fig.add_trace(go.Scatter(x=taus, y=curves.credit_curve_line(fits[bucket], taus),
                                     mode="lines", line=dict(color=col, width=2),
                                     legendgroup=bucket, showlegend=False, hoverinfo="skip"))
    style(fig, theme, height=400, hovermode="closest")
    fig.update_xaxes(title_text="Maturity, years", showgrid=True, gridcolor=c["grid"])
    fig.update_yaxes(title_text="OAS, bp")
    return fig


def seniority_dumbbell(pairs: pd.DataFrame, theme: str, n: int = 14) -> go.Figure:
    """By issuer, the spread of the senior bond and of the junior one, joined by a line."""
    if pairs is None or pairs.empty:
        return empty(theme, "No issuer in the basket has bonds on two seniority tiers.", 400)
    c = colors(theme)
    p = pairs.loc[pairs["z"].abs().sort_values(ascending=False).index].head(n).iloc[::-1]
    labels = [f"{t}  ({z:+.1f}σ)" for t, z in zip(p["ticker"], p["z"])]
    fig = go.Figure()
    for lab, lo, hi in zip(labels, p["oas_senior"], p["oas_junior"]):
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lab, lab], mode="lines", showlegend=False,
                                 line=dict(color=c["axis"], width=2), hoverinfo="skip"))
    for col, name, slot, tier in (("oas_senior", "Senior leg", FUND, "senior_tier"),
                                  ("oas_junior", "Junior leg", REFERENCE, "junior_tier")):
        fig.add_trace(go.Scatter(
            x=p[col], y=labels, mode="markers", name=name,
            marker=dict(size=11, color=cat(theme, slot), line=dict(width=2, color=c["surface"])),
            customdata=np.column_stack([p["issuer"], p[tier], p["senior" if col == "oas_senior" else "junior"],
                                        p["ratio"], p["adj_ratio"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[2]} · %{customdata[1]}<br>"
                          "OAS %{x:.0f} bp · ratio %{customdata[3]:.2f}x, "
                          "maturity-adjusted %{customdata[4]:.2f}x<extra></extra>"))
    style(fig, theme, height=max(300, 26 * len(p) + 90), hovermode="closest",
          margin=dict(l=120, r=18, t=38, b=38))
    fig.update_xaxes(title_text="OAS, bp", showgrid=True, gridcolor=c["grid"])
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=c["ink2"]))
    return fig


def seniority_history(hist: pd.DataFrame, theme: str) -> go.Figure:
    """Seniority premium of the portfolio through time (median spread ratios)."""
    if hist is None or hist.empty:
        return empty(theme, "Not enough multi-tier issuers to build a seniority history.", 320)
    fig = go.Figure()
    for slot, col in enumerate(hist.columns):
        fig.add_trace(_line(hist.index, hist[col], col, cat(theme, slot), fmt="%{y:.2f}x"))
    style(fig, theme, height=320)
    fig.update_yaxes(title_text="Junior over senior spread ratio")
    return fig


def carry_scatter(basket: pd.DataFrame, theme: str) -> go.Figure:
    """Three-month carry plus roll-down against duration, by rating bucket."""
    c = colors(theme)
    fig = go.Figure()
    order = [b for b in ["AA+", "A", "BBB", "BB", "B", "CCC"] if b in set(basket["bucket"])]
    sizeref = 2.0 * basket["weight"].max() * 100 / (22.0 ** 2)
    for slot, bucket in enumerate(order[:4]):
        g = basket[basket["bucket"] == bucket]
        fig.add_trace(go.Scatter(
            x=g["mod_dur"], y=g["total_bps"], mode="markers", name=bucket,
            marker=dict(size=g["weight"] * 100, sizemode="area", sizeref=sizeref, sizemin=4,
                        color=cat(theme, slot), line=dict(width=1.5, color=c["surface"])),
            customdata=np.column_stack([g["bond_id"], g["carry_bps"], g["roll_bps"],
                                        g["breakeven_bp"], g["oas"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>Carry %{customdata[1]:.0f} bp + roll "
                          "%{customdata[2]:.0f} bp over 3 months<br>Break-even widening "
                          "%{customdata[3]:.0f} bp · OAS %{customdata[4]:.0f} bp<extra></extra>"))
    top = basket.sort_values("breakeven_bp", ascending=False).head(5)
    fig.add_trace(go.Scatter(x=top["mod_dur"], y=top["total_bps"], mode="text",
                             text=[t.split(" ")[0] for t in top["bond_id"]],
                             textposition=["top center", "bottom center", "middle right",
                                           "top left", "bottom right"][:len(top)],
                             textfont=dict(size=11, color=c["ink2"]),
                             showlegend=False, hoverinfo="skip"))
    style(fig, theme, height=400, hovermode="closest")
    fig.update_xaxes(title_text="Modified duration, years", showgrid=True, gridcolor=c["grid"])
    fig.update_yaxes(title_text="Carry + roll-down over 3 months, bp")
    return fig


# ============================================================================
# 6. Risk
# ============================================================================
def var_histogram(pnl: pd.Series, var: Dict[str, float], theme: str,
                  level: float = 0.99) -> go.Figure:
    """Distribution of the daily P&L of today's portfolio, with the VaR thresholds."""
    c = colors(theme)
    x = pnl.dropna() * 1e4
    fig = go.Figure(go.Histogram(x=x, nbinsx=60, marker=dict(color=cat(theme, FUND), line_width=0),
                                 hovertemplate="%{x} bp<br>%{y} days<extra></extra>"))
    marks = [("hist", f"Historical VaR {level:.0%}", c["ink"], "top left"),
             ("param", f"Parametric VaR {level:.0%}", c["muted"], "top left")]
    for i, (key, label, color, pos) in enumerate(marks):
        fig.add_vline(x=-var[key] * 1e4, line_color=color, line_width=1.5, line_dash="dash",
                      annotation_text=f"{label}  {-var[key] * 1e4:.0f} bp",
                      annotation_position=pos, annotation_yshift=-18 * i)
    style(fig, theme, height=330, legend=False, hovermode="closest")
    fig.update_layout(bargap=0.06)
    fig.update_xaxes(title_text="One-day P&L of today's portfolio under past market moves, bp")
    fig.update_yaxes(title_text="Days")
    return fig


def var_backtest(bt: pd.DataFrame, theme: str) -> go.Figure:
    """Daily P&L against the VaR estimated the day before, exceptions marked."""
    c = colors(theme)
    x = bt.index
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=bt["pnl"] * 1e4, name="Daily P&L",
                         marker=dict(color=rgba(c["ink2"], 0.55), line_width=0),
                         hovertemplate="%{y:.0f} bp<extra>Daily P&L</extra>"))
    fig.add_trace(_line(x, -bt["var"] * 1e4, "VaR 99%, 250-day window", cat(theme, REFERENCE),
                        fmt="%{y:.0f} bp"))
    br = bt[bt["breach"]]
    fig.add_trace(go.Scatter(x=br.index, y=br["pnl"] * 1e4, mode="markers", name="Exception",
                             marker=dict(size=9, color=c["critical"], symbol="x",
                                         line=dict(width=1.5, color=c["surface"])),
                             hovertemplate="%{y:.0f} bp<extra>Exception</extra>"))
    style(fig, theme, height=330)
    fig.update_layout(bargap=0.0)
    fig.update_yaxes(title_text="bp")
    return fig


def grouped_bars(table: pd.DataFrame, theme: str, y_title: str, fmt: str = ".2f",
                 horizontal: bool = False, height: int = 330) -> go.Figure:
    """Grouped bars, fund against benchmark (columns 'fund' and 'bench')."""
    c = colors(theme)
    t = table.iloc[::-1] if horizontal else table
    cats = [f"{i:g}Y" if isinstance(i, (int, float, np.floating)) else str(i) for i in t.index]
    fig = go.Figure()
    for col, name, slot in (("fund", "Fund", FUND), ("bench", "Benchmark", REFERENCE)):
        kw = dict(y=cats, x=t[col], orientation="h") if horizontal else dict(x=cats, y=t[col])
        tpl = ("%{x:" + fmt + "}") if horizontal else ("%{y:" + fmt + "}")
        fig.add_trace(go.Bar(name=name, marker=dict(color=cat(theme, slot), cornerradius=4, line_width=0),
                             hovertemplate=tpl + "<extra>" + name + "</extra>", **kw))
    style(fig, theme, height=height, hovermode="y unified" if horizontal else "x unified",
          spikes=False, margin=dict(l=110 if horizontal else 54, r=18, t=38, b=38))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.12)
    if horizontal:
        fig.update_xaxes(title_text=y_title, showgrid=True, gridcolor=c["grid"])
        fig.update_yaxes(showgrid=False, type="category", tickfont=dict(size=11, color=c["ink2"]))
    else:
        fig.update_xaxes(type="category")
        fig.update_yaxes(title_text=y_title)
    return fig


def tracking_panel(tr: pd.DataFrame, theme: str) -> go.Figure:
    """Fund and benchmark rebased to 100, cumulative gap, rolling tracking error."""
    if tr is None or tr.empty:
        return empty(theme, "No benchmark series. Set bench_ric in config.ETFS to enable tracking.", 420)
    c = colors(theme)
    x = tr.index
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.42, 0.30, 0.28])
    fig.add_trace(_line(x, tr["fund_idx"], "Fund NAV, rebased to 100", cat(theme, FUND)), 1, 1)
    fig.add_trace(_line(x, tr["bench_idx"], "Benchmark, rebased to 100", cat(theme, REFERENCE)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=tr["td_cum_bps"], name="Cumulative tracking difference",
                             mode="lines", line=dict(color=cat(theme, PREMIUM), width=2),
                             fill="tozeroy", fillcolor=rgba(cat(theme, PREMIUM), 0.10),
                             hovertemplate="%{y:+.0f} bp<extra>Tracking difference</extra>"), 2, 1)
    fig.add_trace(_line(x, tr["te_roll_bps"], "Tracking error, 3-month rolling", cat(theme, FOURTH),
                        fmt="%{y:.0f} bp"), 3, 1)
    fig.add_hline(y=0, line_color=c["axis"], line_width=1, row=2, col=1)
    style(fig, theme, height=520)
    sync_hover(fig, 3)
    panel_label(fig, "Cumulative tracking difference, bp", 2, theme)
    panel_label(fig, "Annualised tracking error, bp", 3, theme)
    return fig


# ============================================================================
# 7. Trade ideas
# ============================================================================
def idea_signal(frame: pd.DataFrame, idea: pd.Series, label: str, theme: str) -> go.Figure:
    """Signal history of an idea with its norm, bands and entry, target and stop levels."""
    c = colors(theme)
    f = frame.dropna(subset=["value"]).iloc[-378:]
    fig = go.Figure()
    _band(fig, f.index, f["upper"], f["lower"], "Norm ±2σ", c["muted"])
    fig.add_trace(_line(f.index, f["mean"], "Norm", c["muted"], 1.0))
    fig.add_trace(_line(f.index, f["value"], label, cat(theme, FUND)))
    _level_lines(fig, f.index[0], f.index[-1], idea, theme)
    style(fig, theme, height=340)
    fig.update_yaxes(title_text="Spread ratio" if idea["unit"] == "ratio" else "bp")
    return fig


def idea_fan(sim: Dict[str, object], idea: pd.Series, theme: str) -> go.Figure:
    """Monte Carlo fan of the signal over the horizon of the idea."""
    c = colors(theme)
    fan = sim["fan"]
    x = np.arange(len(fan))
    blue = cat(theme, FUND)
    fig = go.Figure()
    _band(fig, x, fan["p95"], fan["p05"], "5th to 95th percentile", blue, alpha=0.12)
    _band(fig, x, fan["p75"], fan["p25"], "25th to 75th percentile", blue, alpha=0.26)
    fig.add_trace(_line(x, fan["p50"], "Median path", blue))
    _level_lines(fig, x[0], x[-1], idea, theme, keys=("target", "stop"))
    style(fig, theme, height=340)
    fig.update_xaxes(title_text="Trading days from today")
    fig.update_yaxes(title_text="Spread ratio" if idea["unit"] == "ratio" else "bp")
    return fig


def pnl_distribution(pnl: np.ndarray, theme: str) -> go.Figure:
    """Simulated P&L distribution of an idea, gains and losses in separate colours."""
    c = colors(theme)
    fig = go.Figure()
    bins = dict(start=float(np.floor(pnl.min())), end=float(np.ceil(pnl.max())),
                size=max((pnl.max() - pnl.min()) / 36, 1e-6))
    fig.add_trace(go.Histogram(x=pnl[pnl >= 0], xbins=bins, name="Gain",
                               marker=dict(color=c["pos"], line_width=0)))
    fig.add_trace(go.Histogram(x=pnl[pnl < 0], xbins=bins, name="Loss",
                               marker=dict(color=c["neg"], line_width=0)))
    fig.add_vline(x=float(pnl.mean()), line_color=c["ink"], line_width=1.5,
                  annotation_text=f"Mean {pnl.mean():+.0f} bp", annotation_position="top")
    style(fig, theme, height=340, hovermode="closest")
    fig.update_layout(barmode="overlay", bargap=0.06)
    fig.update_xaxes(title_text="Simulated P&L at exit, bp of notional")
    fig.update_yaxes(title_text="Paths")
    return fig


def overview_panel(daily: pd.DataFrame, theme: str) -> go.Figure:
    """Compact view for the Overview tab, price and NAV, then the premium."""
    c = colors(theme)
    x = daily.index
    pdb = ea.premium_discount(daily)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.58, 0.42])
    fig.add_trace(_line(x, daily["price"], "ETF price", cat(theme, FUND), fmt="%{y:.3f}"), 1, 1)
    fig.add_trace(_line(x, daily["nav"], "NAV", cat(theme, REFERENCE), fmt="%{y:.3f}"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=pdb, name="Premium / discount", mode="lines",
                             line=dict(color=cat(theme, PREMIUM), width=2), fill="tozeroy",
                             fillcolor=rgba(cat(theme, PREMIUM), 0.10),
                             hovertemplate="%{y:.1f} bp<extra>Premium / discount</extra>"), 2, 1)
    fig.add_hline(y=0, line_color=c["axis"], line_width=1, row=2, col=1)
    style(fig, theme, height=400)
    sync_hover(fig, 2)
    panel_label(fig, "Premium to NAV, bp", 2, theme)
    return fig


def _log_axis(values) -> dict:
    """Settings of a log axis with 1-2-5 ticks that cover the data."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    ticks = log_ticks(float(v.min()), float(v.max())) if len(v) else []
    return dict(type="log", tickvals=ticks, ticktext=[f"{t:g}" for t in ticks], minor=dict(ticks=""))


def _level_lines(fig, x0, x1, idea, theme, keys=("entry", "target", "stop")) -> None:
    """Entry, target and stop levels drawn as series, so they show in the legend."""
    c = colors(theme)
    spec = {"entry": ("Entry", c["ink"], "dot"), "target": ("Target", c["good"], "dash"),
            "stop": ("Stop", c["critical"], "dash")}
    for key in keys:
        text, color, dash = spec[key]
        v = float(idea[key])
        label = f"{text} {v:.2f}x" if idea["unit"] == "ratio" else f"{text} {v:,.0f} bp"
        fig.add_trace(go.Scatter(x=[x0, x1], y=[v, v], mode="lines", name=label,
                                 line=dict(color=color, width=1.3, dash=dash), hoverinfo="skip"))
