"""
Page structure, with header, global filters and tabs.

The global filters (ETF, lookback, theme) sit on one row above everything they
drive. Each tab declares its slots (tiles, figures, tables) and the callbacks
fill them.
"""
from __future__ import annotations

import dash_ag_grid as dag
from dash import dcc, html

import config as cfg
from ui.components import card, control, graph

LOOKBACKS = {"3M": 63, "6M": 126, "1Y": 252, "3Y": 756}
ORDER_SIZES = [1, 2, 5, 10, 20, 50, 100]
IDEA_TYPES = ["Capital structure", "Credit pair", "Seniority", "Credit curve",
              "Carry and roll", "ETF arbitrage", "ETF premium", "Rates curve"]


def _radio(component_id: str, options, value, persistence: bool = False) -> dcc.RadioItems:
    return dcc.RadioItems(id=component_id, options=options, value=value, inline=True,
                          className="segmented", labelClassName="segmented-item",
                          inputClassName="segmented-input", persistence=persistence,
                          persistence_type="local")


def _slider(component_id: str, lo, hi, step, value, marks=None) -> dcc.Slider:
    return dcc.Slider(id=component_id, min=lo, max=hi, step=step, value=value,
                      marks=marks or {lo: str(lo), hi: str(hi)},
                      tooltip={"placement": "bottom", "always_visible": False},
                      allow_direct_input=False, className="slider")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def _tab_overview() -> html.Div:
    return html.Div([
        html.Div(id="ov-kpis"),
        html.Div([
            card("Price, NAV and premium", graph("ov-chart", 490), 8),
            card("Data sources", html.Div(id="ov-status"), 4,
                 "Each block is live from LSEG or simulated, never a mix"),
            card("Top trade ideas", html.Div(id="ov-ideas"), 7,
                 "Highest conviction across both sleeves"),
            card("Sector relative value", graph("ov-sector", 340), 5,
                 "z-score of each sector's spread against the universe. Red means wide"),
            card("ETF monitor", html.Div(id="ov-monitor"), 12),
        ], className="grid"),
    ])


def _tab_premium() -> html.Div:
    return html.Div([
        html.Div(id="pd-kpis"),
        html.Div([
            card("ETF price against NAV, premium with Bollinger bands, volume and primary flows",
                 graph("pd-panel", 680), 12, "Four synchronised panels. Hover to read all of them at once"),
            card("Today's session", [graph("pd-intraday", 360), html.Div(id="pd-intraday-note",
                                                                     className="caption")], 7),
            card("Distribution of the premium", graph("pd-hist", 384), 5, "Over the selected lookback"),
        ], className="grid"),
    ])


def _tab_ap() -> html.Div:
    p = cfg.ApParams()
    controls = html.Div([
        control("Order size, creation units",
                _radio("ap-size", [{"label": str(s), "value": s} for s in ORDER_SIZES], p.order_cu)),
        control("NAV prices the basket at",
                _radio("ap-basis", [{"label": "Bid", "value": "bid"},
                                    {"label": "Mid", "value": "mid"}], p.nav_basis)),
        control("Basket bid/ask multiplier", _slider("ap-stress", 0.5, 3.0, 0.1, p.stress_mult,
                                                     {0.5: "0.5x", 1: "1x", 2: "2x", 3: "3x"}), True),
        control("Funding rate, %", _slider("ap-funding", 0.0, 8.0, 0.1, p.funding_rate_pct,
                                           {0: "0", 4: "4", 8: "8"}), True),
        control("Hedging cost, bp", _slider("ap-hedge", 0.0, 10.0, 0.5, p.hedge_cost_bps,
                                            {0: "0", 5: "5", 10: "10"}), True),
        control("Market impact coefficient", _slider("ap-impact", 0.0, 2.0, 0.05, p.impact_coeff,
                                                     {0: "0", 1: "1", 2: "2"}), True),
    ], className="controls-inline")
    return html.Div([
        html.Div([card("AP assumptions", controls, 12,
                       "These settings drive every chart on this tab")], className="grid"),
        html.Div(id="ap-kpis"),
        html.Div([
            card("Premium against creation and redemption thresholds", graph("ap-bands", 520), 8,
                 "A dot marks a session where the arbitrage paid after all costs"),
            card("Cost stack of a primary order", graph("ap-stack", 520), 4),
            card("Net edge by order size and premium level", graph("ap-grid", 470), 7,
                 "Blue pays, red does not. Fixed fees dilute with size, impact grows"),
            card("Latest arbitrage windows", html.Div(id="ap-windows"), 5),
        ], className="grid"),
    ])


def _tab_liquidity() -> html.Div:
    controls = html.Div([
        control("Participation in daily volume, %",
                _slider("lq-part", 5, 40, 5, 20, {5: "5", 20: "20", 40: "40"}), True),
        control("Primary order to test, mm",
                _radio("lq-order", [{"label": str(s), "value": s} for s in (5, 10, 25, 50, 100)], 25)),
    ], className="controls-inline")
    return html.Div([
        html.Div([card("Liquidity assumptions", controls, 12)], className="grid"),
        html.Div(id="lq-kpis"),
        html.Div([
            card("Basket bonds by volume, bid/ask and weight", graph("lq-scatter", 400), 7,
                 "Bubble area is the fund weight"),
            card("Liquidity score by sector and maturity", graph("lq-heat", 400), 5,
                 "Weighted by fund weight, on a 0 to 100 scale"),
            card("Primary against secondary liquidity", graph("lq-ps", 400), 6,
                 "One dot per session. Red sessions closed at a discount"),
            card("How much of a primary order clears, by size", graph("lq-profile", 400), 6),
            card("Slowest bonds for the tested order", html.Div(id="lq-table"), 12),
        ], className="grid"),
    ])


def _tab_credit() -> html.Div:
    return html.Div([
        html.Div(id="cr-kpis"),
        html.Div([
            card("Equity against credit, Merton distance to default", graph("cr-scatter", 474), 7,
                 "Click an issuer to open its history. Red trades wide to fair value"),
            card("Issuer history", [
                dcc.Dropdown(id="cr-issuer", clearable=False, className="dd"),
                graph("cr-issuer-fig", 430)], 5),
            card("Most stretched pairs", html.Div(id="cr-pairs-table"), 5,
                 "Same sector, z-score of the spread ratio"),
            card("Pair history", [
                dcc.Dropdown(id="cr-pair", clearable=False, className="dd"),
                graph("cr-pair-fig", 430)], 7),
            card("Sector z-scores by window", graph("cr-sector", 360), 5),
            card("Sector against sector", graph("cr-sector-pairs", 360), 7,
                 "Row minus column, z-score over six months"),
        ], className="grid"),
    ])


def _tab_curves() -> html.Div:
    return html.Div([
        html.Div(id="cv-kpis"),
        html.Div([
            card("Sovereign curve", graph("cv-govt", 360), 6),
            card("Slopes and butterfly", graph("cv-slopes", 360), 6),
            card("Credit curves by rating", graph("cv-credit", 400), 7,
                 "Circle senior unsecured, square secured, diamond subordinated. "
                 "Curves are fitted on senior unsecured bonds"),
            card("Carry and roll-down against duration", graph("cv-carry", 400), 5,
                 "Three-month horizon, curves unchanged"),
            card("Seniority premium by issuer", graph("cv-dumbbell", 470), 6,
                 "Most unusual gaps first, in standard deviations"),
            card("Seniority across the portfolio", [graph("cv-senhist", 280),
                                                    html.Div(id="cv-sen-table")], 6),
        ], className="grid"),
    ])


def _tab_risk() -> html.Div:
    controls = html.Div([
        control("Confidence level", _radio("rk-level", [{"label": "95%", "value": 0.95},
                                                        {"label": "99%", "value": 0.99}], 0.99)),
        control("Horizon", _radio("rk-horizon", [{"label": "1 day", "value": 1},
                                                 {"label": "10 days", "value": 10}], 1)),
    ], className="controls-inline")
    return html.Div([
        html.Div([card("VaR settings", controls, 12)], className="grid"),
        html.Div(id="rk-kpis"),
        html.Div([
            card("P&L distribution of today's portfolio", graph("rk-hist", 330), 6,
                 "Historical simulation on past moves in rates and spreads"),
            card("VaR backtest", graph("rk-backtest", 330), 6, "One-day 99% VaR"),
            card("Key rate durations", graph("rk-krd", 360), 5, "Years, fund against benchmark"),
            card("Duration times spread by sector", graph("rk-dts", 360), 7),
            card("Tracking against the benchmark", graph("rk-tracking", 520), 7),
            card("Exposures and stress tests", [html.Div(id="rk-expo"),
                                                html.Div(id="rk-stress", className="stack-gap")], 5),
        ], className="grid"),
    ])


def _tab_ideas() -> html.Div:
    controls = html.Div([
        control("Scope", _radio("id-scope", [{"label": "Selected ETF", "value": "etf"},
                                             {"label": "All", "value": "all"}], "all")),
        control("Status", _radio("id-status", [{"label": "All", "value": "all"},
                                               {"label": "Live", "value": "Live"},
                                               {"label": "Watch", "value": "Watch"}], "all")),
        control("Type", dcc.Dropdown(id="id-types", options=IDEA_TYPES, value=[], multi=True,
                                     placeholder="All types", className="dd dd-wide"), True),
        control("Minimum conviction", _slider("id-conv", 1, 5, 1, 1, {i: str(i) for i in range(1, 6)}),
                True),
    ], className="controls-inline")
    def col(field, header, width=None, fmt=None, numeric=False, **extra):
        """Definition of an AG Grid column. fmt is a d3 format (such as '+.1f')."""
        d = {"field": field, "headerName": header, **extra}
        if width:
            d["width"] = width
        if numeric:
            d.update(type="rightAligned", filter="agNumberColumnFilter")
        if fmt:
            d["valueFormatter"] = {"function": f"params.value == null ? '' : d3.format('{fmt}')(params.value)"}
        return d

    column_defs = [
        col("id", "ID", 80, pinned="left"), col("type", "Type", 146), col("etf", "ETF", 74),
        col("title", "Idea", None, flex=1, minWidth=240, tooltipField="rationale", wrapText=True,
            autoHeight=True, cellStyle={"lineHeight": "1.35", "paddingTop": "9px",
                                        "paddingBottom": "9px"}),
        col("z", "z-score", 96, "+.1f", True), col("entry", "Entry", 84, ",.2f", True),
        col("target", "Target", 84, ",.2f", True), col("stop", "Stop", 84, ",.2f", True),
        col("horizon_d", "Days", 74, None, True),
        col("exp_pnl_bps", "Exp. P&L bp", 114, ",.0f", True),
        col("risk_bps", "Risk bp", 88, ",.0f", True), col("rr", "R/R", 72, ".1f", True),
        col("conviction", "Conv.", 78, None, True), col("status", "Status", 88),
    ]
    tbl = dag.AgGrid(
        id="ideas-table", columnDefs=column_defs, rowData=[], getRowId="params.data.id",
        defaultColDef={"sortable": True, "filter": True, "floatingFilter": True,
                       "resizable": True, "suppressHeaderMenuButton": True},
        dashGridOptions={"rowSelection": {"mode": "singleRow", "checkboxes": False,
                                          "enableClickSelection": True},
                         "pagination": True, "paginationPageSize": 10,
                         "paginationPageSizeSelector": [10, 20, 50],
                         "domLayout": "autoHeight", "animateRows": False,
                         "tooltipShowDelay": 300},
        className="ideas-grid", style={"width": "100%", "height": None},
    )
    return html.Div([
        html.Div([
            card("Filters", controls, 12, "Each column of the table also has its own filter and sort"),
            card("Trade ideas", tbl, 12, "Select a row to open the idea below"),
            card("Selected idea", html.Div(id="id-detail"), 4),
            card("Signal history with entry, target and stop", graph("id-signal", 360), 8),
            card("Simulated paths of the signal", graph("id-fan", 340), 6,
                 "Mean-reverting process calibrated on one year"),
            card("Simulated P&L at exit", graph("id-pnl", 340), 6,
                 "Exit at target, at stop, or at the horizon"),
        ], className="grid"),
    ])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def build_layout() -> html.Div:
    etf_options = [{"label": f"{e.etf_id} · {e.segment}", "value": e.etf_id}
                   for e in cfg.ETFS.values()]
    tabs = [("overview", "Overview", _tab_overview), ("premium", "Premium / Discount", _tab_premium),
            ("ap", "AP Arbitrage", _tab_ap), ("liquidity", "Liquidity", _tab_liquidity),
            ("credit", "Credit RV", _tab_credit), ("curves", "Curves & Seniority", _tab_curves),
            ("risk", "Risk", _tab_risk), ("ideas", "Trade Ideas", _tab_ideas)]
    return html.Div([
        html.Header([
            html.Div([html.Div(cfg.APP_TITLE, className="app-title"),
                      html.Div(cfg.APP_SUBTITLE, className="app-subtitle")]),
            html.Div([html.Span(id="mode-badge"), html.Span("PROTOTYPE", className="tag"),
                      html.Span(id="asof", className="asof")], className="header-right"),
        ], className="header"),
        html.Div([
            control("ETF", _radio("etf", etf_options, cfg.DEFAULT_ETF)),
            control("Lookback", _radio("lookback", [{"label": k, "value": k} for k in LOOKBACKS], "1Y")),
            control("Theme", _radio("theme", [{"label": "Dark", "value": "dark"},
                                              {"label": "Light", "value": "light"}], "dark", True)),
            html.Div(id="etf-name", className="etf-name"),
        ], className="filters"),
        dcc.Tabs(id="tabs", value="overview", parent_className="tabs-parent", className="tabs",
                 content_className="tabs-content",
                 children=[dcc.Tab(label=label, value=value, className="tab",
                                   selected_className="tab tab-selected", children=build())
                           for value, label, build in tabs]),
        html.Footer("Prototype for discussion. Simulated blocks are flagged in Overview. "
                    "Not investment advice.", className="footer"),
        dcc.Interval(id="tick", interval=cfg.LIVE_REFRESH_MS),
        html.Div(id="theme-sink", style={"display": "none"}),
    ], id="root", className="viz-root")
