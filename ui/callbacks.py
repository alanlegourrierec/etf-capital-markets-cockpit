"""
Links the controls, the engine and the figures.

One callback per tab. Each reads the global filters (ETF, lookback, theme),
asks the Analytics object for its results (it caches the heavy work) and fills
tiles, figures and tables. A tab that is not shown computes nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Input, Output, html, no_update

import config as cfg
from engine import curves, etf_analytics as ea, relative_value as rv, risk
from engine.analytics import Analytics
from engine.trade_ideas import signal_for, signal_frame, simulate_idea
from ui import figures as figs
from ui.components import bp, chip, kpi, kpi_row, money, num, table
from ui.layout import LOOKBACKS, ORDER_SIZES
from ui.theme import empty

WINDOW_LABELS = {21: "1M", 63: "3M", 126: "6M", 252: "1Y"}
SOURCE_CHIP = {"LSEG": ("Live", "good"), "SIM": ("Simulated", "neutral"), "N/A": ("Missing", "warning")}
TABLE_COLS = ["id", "type", "etf", "title", "z", "entry", "target", "stop", "unit", "horizon_d",
              "exp_pnl_bps", "risk_bps", "rr", "carry_bps", "conviction", "liquidity", "status"]


def _records(df: pd.DataFrame) -> list:
    """Turns a frame into a list of dicts without NaN (the grid wants None)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def register_callbacks(app, an: Analytics) -> None:
    provider = an.p

    # ----------------------------------------------------------------- theme
    # Runs in the browser and sets the data-theme attribute the CSS reads.
    app.clientside_callback(
        "function(theme){document.documentElement.setAttribute('data-theme', theme || 'dark');"
        "return '';}",
        Output("theme-sink", "children"), Input("theme", "value"))

    @app.callback(Output("mode-badge", "children"), Output("asof", "children"),
                  Output("etf-name", "children"), Input("etf", "value"))
    def header(etf_id):
        data = an.etf(etf_id)
        mode = provider.mode_label()
        tone = {"LSEG LIVE": "good", "HYBRID": "warning"}.get(mode, "neutral")
        return (chip(mode, tone), f"Data as of {data.daily.index[-1]:%d %b %Y}",
                f"{data.spec.name} · Benchmark {data.spec.benchmark}")

    # -------------------------------------------------------------- overview
    @app.callback(Output("ov-kpis", "children"), Output("ov-chart", "figure"),
                  Output("ov-status", "children"), Output("ov-ideas", "children"),
                  Output("ov-sector", "figure"), Output("ov-monitor", "children"),
                  Input("etf", "value"), Input("lookback", "value"), Input("theme", "value"))
    def overview(etf_id, lookback, theme):
        data = an.etf(etf_id)
        d, spec = data.daily, data.spec
        b, _ = an.basket(etf_id)
        expo, var = risk.exposures(b), an.var(etf_id)
        pdb = ea.premium_discount(d)
        z20 = ea.bollinger(pdb)["z"].iloc[-1]
        last, prev = d.iloc[-1], d.iloc[-2]
        flow5 = float((d["flow_cu"] * spec.cu_shares * d["nav"]).tail(5).sum() / 1e6)
        tiles = kpi_row([
            kpi("ETF price", num(last["price"], 3),
                f"NAV {last['nav']:.3f} · {(last['price'] / prev['price'] - 1) * 100:+.2f}% on the day"),
            kpi("Premium to NAV", bp(pdb.iloc[-1], 0, True), f"{z20:+.1f}σ against 20-day mean",
                "warning" if abs(z20) >= 2 else None),
            kpi("Assets", money(last["aum_musd"]), f"{flow5:+,.1f} mm primary flow, 5 days"),
            kpi("Traded value, 3M average", money(d["value_musd"].tail(63).mean()),
                f"ETF bid/ask {last['etf_ba_bps']:.0f} bp"),
            kpi("OAS", bp(expo.loc["OAS (bp)", "fund"]),
                f"{expo.loc['OAS (bp)', 'active']:+.0f} bp against benchmark"),
            kpi("Modified duration", num(expo.loc["Modified duration (y)", "fund"], 2, suffix=" y"),
                f"{expo.loc['Modified duration (y)', 'active']:+.2f} y against benchmark"),
            kpi("Yield to maturity", num(expo.loc["Yield to maturity (%)", "fund"], 2, suffix="%"),
                f"{len(b)} bonds in the basket"),
            kpi("1-day VaR, 99%", bp(var["hist"] * 1e4), f"Parametric {var['param'] * 1e4:.0f} bp"),
        ])
        chart = figs.overview_panel(d.tail(LOOKBACKS[lookback]), theme)

        def status_cell(col, v):
            if col == "Source":
                label, tone = SOURCE_CHIP.get(v, (str(v), "neutral"))
                return chip(label, tone)
            return None
        status = table(provider.status_table(), cell=status_cell)

        ideas = an.ideas()
        if len(ideas):
            top = ideas.head(6)[["id", "etf", "title", "z", "exp_pnl_bps", "rr", "conviction", "status"]]
            ideas_tbl = table(top, {"z": "{:+.1f}", "exp_pnl_bps": "{:,.0f}", "rr": "{:.1f}",
                                    "conviction": "{:.0f}"},
                              {"id": "ID", "etf": "ETF", "title": "Idea", "z": "z-score",
                               "exp_pnl_bps": "Exp. P&L bp", "rr": "R/R", "conviction": "Conv.",
                               "status": "Status"})
        else:
            ideas_tbl = html.Div("No idea passes the thresholds today.", className="caption")

        iss = an.issuers()
        zmat = rv.sector_z_matrix(iss.spread5y, iss.static, an.universe_of(etf_id)).rename(
            columns=WINDOW_LABELS)
        sector = figs.heatmap(zmat, theme, "z-score", diverging=True, fmt="+.1f", height=330, zlim=3.0)

        rows = []
        for eid, sp in cfg.ETFS.items():
            ed = an.etf(eid).daily
            eb, _ = an.basket(eid)
            ex, ev = risk.exposures(eb), an.var(eid)
            epd = ea.premium_discount(ed)
            _, ts = risk.tracking(ed, sp.ter_bps)
            rows.append({"ETF": eid, "Segment": sp.segment, "Style": "Active" if sp.active else "Index",
                         "Price": ed["price"].iloc[-1], "NAV": ed["nav"].iloc[-1],
                         "Prem. bp": epd.iloc[-1], "Prem. z": ea.bollinger(epd)["z"].iloc[-1],
                         "Assets mm": ed["aum_musd"].iloc[-1],
                         "ADV mm": ed["value_musd"].tail(63).mean(),
                         "OAS bp": ex.loc["OAS (bp)", "fund"],
                         "Dur. y": ex.loc["Modified duration (y)", "fund"],
                         "YTM %": ex.loc["Yield to maturity (%)", "fund"],
                         "VaR bp": ev["hist"] * 1e4, "TE bp": ts.get("te_ann_bps", np.nan),
                         "TD bp/y": ts.get("td_ann_bps", np.nan), "TER bp": sp.ter_bps})
        monitor = table(pd.DataFrame(rows), {
            "Price": "{:.3f}", "NAV": "{:.3f}", "Prem. bp": "{:+.0f}", "Prem. z": "{:+.1f}",
            "Assets mm": "{:,.0f}", "ADV mm": "{:.1f}", "OAS bp": "{:.0f}", "Dur. y": "{:.2f}",
            "YTM %": "{:.2f}", "VaR bp": "{:.0f}", "TE bp": "{:.0f}", "TD bp/y": "{:+.0f}",
            "TER bp": "{:.0f}"})
        return tiles, chart, status, ideas_tbl, sector, monitor

    # ------------------------------------------------- premium and discount
    @app.callback(Output("pd-kpis", "children"), Output("pd-panel", "figure"),
                  Output("pd-hist", "figure"),
                  Input("etf", "value"), Input("lookback", "value"), Input("theme", "value"))
    def premium(etf_id, lookback, theme):
        data = an.etf(etf_id)
        d = data.daily
        n = LOOKBACKS[lookback]
        pdb = ea.premium_discount(d)
        boll = ea.bollinger(pdb)
        st = ea.pd_stats(pdb.tail(n))
        flow = d["flow_cu"] * data.spec.cu_shares * d["nav"] / 1e6
        outside = int(((pdb > boll["upper"]) | (pdb < boll["lower"])).tail(n).sum())
        tiles = kpi_row([
            kpi("Premium today", bp(st["last"], 0, True), f"{boll['z'].iloc[-1]:+.1f}σ against 20-day mean",
                "warning" if abs(boll["z"].iloc[-1]) >= 2 else None),
            kpi(f"Average, {lookback}", bp(st["mean"], 0, True), f"Standard deviation {st['std']:.0f} bp"),
            kpi("Percentile today", num(st["pct_rank"], 0, suffix="%"), f"Within the {lookback} range"),
            kpi("Range", f"{st['min']:+.0f} / {st['max']:+.0f} bp", f"5% to 95%: {st['p05']:+.0f} / {st['p95']:+.0f}"),
            kpi("Days at a premium", num(st["days_premium"], 0, suffix="%"), f"Over {lookback}"),
            kpi("Days outside the band", str(outside), "Bollinger 20 days, ±2σ"),
            kpi("Net primary flow", f"{flow.tail(n).sum():+,.0f} mm",
                f"{int((d['flow_cu'].tail(n) != 0).sum())} sessions with a primary order"),
        ])
        return (tiles, figs.premium_panel(d.tail(n), boll.tail(n), flow.tail(n), theme),
                figs.pd_histogram(pdb.tail(n), theme))

    @app.callback(Output("pd-intraday", "figure"), Output("pd-intraday-note", "children"),
                  Input("tick", "n_intervals"), Input("etf", "value"), Input("theme", "value"))
    def intraday(n_ticks, etf_id, theme):
        data = an.etf(etf_id)
        intra = provider.refresh_intraday(etf_id)
        if intra is None:
            return figs.intraday_panel(None, theme), "Intraday needs an iNAV RIC for this ETF."
        if data.source.get("intraday") == "SIM":
            # Replays the simulated session, each refresh adds three bars
            k = len(intra)
            start = int(0.55 * k)
            shown = start + ((n_ticks or 0) * 3) % (k - start) + 1
            intra = intra.iloc[:shown]
            note = (f"Simulated session replay, {intra.index[-1]:%H:%M}. "
                    f"Three one-minute bars are added every {cfg.LIVE_REFRESH_MS // 1000} seconds.")
        else:
            note = f"Live LSEG one-minute bars, last print {intra.index[-1]:%H:%M}."
        return figs.intraday_panel(intra, theme), note

    # --------------------------------------------------------- AP arbitrage
    @app.callback(Output("ap-kpis", "children"), Output("ap-bands", "figure"),
                  Output("ap-stack", "figure"), Output("ap-grid", "figure"),
                  Output("ap-windows", "children"),
                  Input("etf", "value"), Input("lookback", "value"), Input("theme", "value"),
                  Input("ap-size", "value"), Input("ap-basis", "value"), Input("ap-stress", "value"),
                  Input("ap-funding", "value"), Input("ap-hedge", "value"), Input("ap-impact", "value"))
    def ap(etf_id, lookback, theme, size, basis, stress, funding, hedge, impact):
        data = an.etf(etf_id)
        d, spec = data.daily, data.spec
        b, _ = an.basket(etf_id)
        n = LOOKBACKS[lookback]
        p = cfg.ApParams(order_cu=int(size or 1), nav_basis=basis or "bid",
                         stress_mult=float(stress or 1.0), funding_rate_pct=float(funding or 0.0),
                         hedge_cost_bps=float(hedge or 0.0), impact_coeff=float(impact or 0.0))
        last = d.iloc[-1]
        bands_all = ea.arb_bands(d, spec, b, p)
        bands = bands_all.tail(n)
        stack = ea.ap_cost_stack(spec, b, last["nav"], last["etf_ba_bps"], last["basket_ba_bps"], p)
        step = max(5.0, float(np.round(1.5 * stack["creation"].sum() / 6.0 / 5.0) * 5.0))
        levels = np.arange(-6, 7) * step            # from discount to premium, in round steps
        grid = ea.arb_pnl_grid(spec, b, last["nav"], last["etf_ba_bps"], last["basket_ba_bps"], p,
                               ORDER_SIZES, levels)
        now = bands.iloc[-1]
        notional = ea.order_notional(spec, last["nav"], p.order_cu) / 1e6
        if now["signal"] != 0:
            edge = kpi("Edge today", bp(now["edge_bps"], 1),
                       f"{'Creation' if now['signal'] > 0 else 'Redemption'} window open, "
                       f"about {now['pnl_usd']:,.0f} per order", "good")
        else:
            gap = min(now["upper"] - now["pd_bps"], now["pd_bps"] - now["lower"])
            edge = kpi("Edge today", "No window", f"{gap:.0f} bp from the nearest threshold")
        tiles = kpi_row([
            kpi("Creation threshold", bp(now["upper"], 0, True), "Premium needed to create"),
            kpi("Redemption threshold", bp(now["lower"], 0, True), "Discount needed to redeem"),
            kpi("Premium today", bp(now["pd_bps"], 0, True), "ETF mid against NAV"),
            edge,
            kpi(f"Creation windows, {lookback}", str(int((bands["signal"] > 0).sum())),
                f"Best edge {bands.loc[bands['signal'] > 0, 'edge_bps'].max() if (bands['signal'] > 0).any() else 0:.0f} bp"),
            kpi(f"Redemption windows, {lookback}", str(int((bands["signal"] < 0).sum())),
                f"Best edge {bands.loc[bands['signal'] < 0, 'edge_bps'].max() if (bands['signal'] < 0).any() else 0:.0f} bp"),
            kpi("Order notional", money(notional, 2), f"{p.order_cu} x {spec.cu_shares:,} shares"),
        ])
        win = ea.arb_windows(bands_all)
        if len(win):
            win = win.reset_index().rename(columns={win.index.name or "index": "date"})
            win["date"] = win["date"].dt.strftime("%d %b %Y")
            windows = table(win, {"pd_bps": "{:+.0f}", "threshold": "{:+.0f}", "edge_bps": "{:.1f}",
                                  "pnl_usd": "{:,.0f}"},
                            {"date": "Date", "side": "Side", "pd_bps": "Premium bp",
                             "threshold": "Threshold bp", "edge_bps": "Net edge bp",
                             "pnl_usd": "P&L per order"})
        else:
            windows = html.Div("No window in the history with these assumptions.", className="caption")
        return tiles, figs.arb_bands(bands, theme), figs.cost_stack(stack, theme), \
            figs.arb_grid(grid, theme), windows

    # -------------------------------------------------------------- liquidity
    @app.callback(Output("lq-kpis", "children"), Output("lq-scatter", "figure"),
                  Output("lq-heat", "figure"), Output("lq-ps", "figure"),
                  Output("lq-profile", "figure"), Output("lq-table", "children"),
                  Input("etf", "value"), Input("lookback", "value"), Input("theme", "value"),
                  Input("lq-part", "value"), Input("lq-order", "value"))
    def liquidity(etf_id, lookback, theme, part, order):
        data = an.etf(etf_id)
        d = data.daily
        b, _ = an.basket(etf_id)
        n = LOOKBACKS[lookback]
        part = float(part or 20) / 100.0
        order = float(order or 25)
        cap = ea.basket_capacity(b, part)
        adv_etf = float(d["value_musd"].tail(63).mean())
        w = b["weight"] / b["weight"].sum()
        days = w * order / (part * b["adv_musd"])
        by_day = days.sort_values()
        done90 = float(by_day[(w[by_day.index].cumsum() >= 0.90).to_numpy()].iloc[0])
        illiquid = float(w[b["liq_score"] < 30].sum() * 100.0)
        tiles = kpi_row([
            kpi("ETF traded value, 3M average", money(adv_etf), "Secondary market"),
            kpi("Basket one-day capacity", money(cap), f"90% of the basket at {part:.0%} of volume"),
            kpi("Secondary over primary", num(adv_etf / cap, 2, suffix="x"),
                "Share of basket capacity that trades on screen"),
            kpi("Basket bid/ask", bp(float((w * b["ba_bps"]).sum())), "Weighted by fund weight"),
            kpi("ETF bid/ask", bp(d["etf_ba_bps"].iloc[-1]),
                f"{(w * b['ba_bps']).sum() / d['etf_ba_bps'].iloc[-1]:.1f}x tighter than the basket"),
            kpi(f"Days to place {order:.0f} mm", num(np.ceil(done90), 0), "For 90% of the basket"),
            kpi("Weight in illiquid bonds", num(illiquid, 1, suffix="%"), "Liquidity score under 30"),
        ])
        slow = (b.assign(trade_musd=w * order, days=days)
                .sort_values("days", ascending=False).head(10)
                [["bond_id", "sector", "rating", "seniority", "weight", "amt_out_musd", "adv_musd",
                  "ba_bps", "liq_score", "trade_musd", "days"]])
        slow["weight"] = slow["weight"] * 100.0
        tbl = table(slow, {"weight": "{:.2f}", "amt_out_musd": "{:,.0f}", "adv_musd": "{:.1f}",
                           "ba_bps": "{:.0f}", "liq_score": "{:.0f}", "trade_musd": "{:.2f}",
                           "days": "{:.1f}"},
                    {"bond_id": "Bond", "sector": "Sector", "rating": "Rating", "seniority": "Seniority",
                     "weight": "Weight %", "amt_out_musd": "Outstanding mm", "adv_musd": "ADV mm",
                     "ba_bps": "Bid/ask bp", "liq_score": "Liq. score", "trade_musd": "To trade mm",
                     "days": "Days"})
        sizes = [1, 2, 5, 10, 25, 50, 100, 250]
        return (tiles, figs.liquidity_scatter(b, theme),
                figs.heatmap(ea.liquidity_heatmap(b), theme, "Liquidity score", height=400,
                             x_title="Maturity bucket"),
                figs.primary_secondary(ea.primary_secondary(d, b, part).tail(n), theme),
                figs.liquidation_profile(ea.liquidation_profile(b, sizes, part), theme), tbl)

    # ----------------------------------------------- credit, capital structure
    @app.callback(Output("cr-issuer", "options"), Output("cr-issuer", "value"),
                  Output("cr-pair", "options"), Output("cr-pair", "value"), Input("etf", "value"))
    def credit_options(etf_id):
        uni = an.universe_of(etf_id)
        snap = an.capital_structure()["snap"]
        s = snap[(snap["universe"] == uni) & ~snap["excluded"]]
        s = s.loc[s["ts_z"].abs().sort_values(ascending=False).index]
        issuers = [{"label": f"{t} · {r['name']}  ({r['ts_z']:+.1f}σ)", "value": t}
                   for t, r in s.iterrows()]
        pairs = an.pairs()
        pairs = pairs[pairs["universe"] == uni].head(30) if len(pairs) else pairs
        popt = [{"label": f"{r['pair']} · {r['sector']}  ({r['z']:+.1f}σ)", "value": f"{r['a']}|{r['b']}"}
                for _, r in pairs.iterrows()]
        return (issuers, issuers[0]["value"] if issuers else None,
                popt, popt[0]["value"] if popt else None)

    @app.callback(Output("cr-issuer", "value", allow_duplicate=True),
                  Input("cr-scatter", "clickData"), prevent_initial_call=True)
    def credit_click(click):
        try:
            return click["points"][0]["customdata"][0]
        except (TypeError, KeyError, IndexError):
            return no_update

    @app.callback(Output("cr-kpis", "children"), Output("cr-scatter", "figure"),
                  Output("cr-issuer-fig", "figure"), Output("cr-pairs-table", "children"),
                  Output("cr-sector", "figure"), Output("cr-sector-pairs", "figure"),
                  Input("etf", "value"), Input("lookback", "value"), Input("theme", "value"),
                  Input("cr-issuer", "value"))
    def credit(etf_id, lookback, theme, issuer):
        uni = an.universe_of(etf_id)
        cs = an.capital_structure()
        iss = an.issuers()
        n = LOOKBACKS[lookback]
        snap = cs["snap"][cs["snap"]["universe"] == uni]
        s = snap[~snap["excluded"]]
        coef = cs["fv"]["coef"].iloc[-1]
        r2 = float(np.corrcoef(np.log(s["fair_spread"]), np.log(s["mkt_spread"]))[0, 1] ** 2)
        wide, tight = s["ts_z"].idxmax(), s["ts_z"].idxmin()
        tiles = kpi_row([
            kpi("Issuers screened", str(len(s)), f"{int(snap['excluded'].sum())} financials left out"),
            kpi("Fit of spread on distance to default", num(r2 * 100, 0, suffix="%"),
                "Share of spread dispersion explained"),
            kpi("Median distance to default", num(s["dd"].median(), 1, suffix="σ"),
                f"Median leverage {s['leverage'].median() * 100:.0f}%"),
            kpi("Credit cheapest to equity", wide,
                f"{s.loc[wide, 'ts_z']:+.1f}σ, {s.loc[wide, 'mkt_spread']:.0f} bp against "
                f"{s.loc[wide, 'fair_spread']:.0f} bp fair"),
            kpi("Credit richest to equity", tight,
                f"{s.loc[tight, 'ts_z']:+.1f}σ, {s.loc[tight, 'mkt_spread']:.0f} bp against "
                f"{s.loc[tight, 'fair_spread']:.0f} bp fair"),
            kpi("Names beyond 2σ", str(int((s["ts_z"].abs() >= 2).sum())), "Gap to fair value, six months"),
        ])
        scatter = figs.merton_scatter(snap, coef, theme, issuer)
        if issuer in cs["fv"]["resid"].columns:
            z = rv.rolling_z(cs["fv"]["resid"][issuer], 126)
            detail = figs.issuer_detail(iss.spread5y[issuer].tail(n), cs["fv"]["fair"][issuer].tail(n),
                                        z.tail(n), theme)
        else:
            detail = empty(theme, "Select an issuer.", 430)
        pairs = an.pairs()
        if len(pairs):
            top = pairs[pairs["universe"] == uni].head(10)[
                ["pair", "sector", "spread_a", "spread_b", "ratio", "mean_ratio", "z", "half_life"]]
            ptable = table(top, {"spread_a": "{:.0f}", "spread_b": "{:.0f}", "ratio": "{:.2f}",
                                 "mean_ratio": "{:.2f}", "z": "{:+.1f}", "half_life": "{:.0f}"},
                           {"pair": "Long / short", "sector": "Sector", "spread_a": "Long bp",
                            "spread_b": "Short bp", "ratio": "Ratio", "mean_ratio": "6M mean",
                            "z": "z-score", "half_life": "Half-life d"})
        else:
            ptable = html.Div("No pair available.", className="caption")
        zmat = rv.sector_z_matrix(iss.spread5y, iss.static, uni).rename(columns=WINDOW_LABELS)
        smat = rv.sector_pair_matrix(iss.spread5y, iss.static, uni)
        return (tiles, scatter, detail, ptable,
                figs.heatmap(zmat, theme, "z-score", diverging=True, fmt="+.1f", height=360, zlim=3.0),
                figs.heatmap(smat, theme, "z-score", diverging=True, fmt="+.1f", height=360, zlim=3.0))

    @app.callback(Output("cr-pair-fig", "figure"), Input("cr-pair", "value"),
                  Input("lookback", "value"), Input("theme", "value"))
    def credit_pair(pair, lookback, theme):
        if not pair:
            return empty(theme, "Select a pair.", 430)
        a, b_ = pair.split("|")
        ph = rv.pair_history(an.issuers().spread5y, a, b_)
        return figs.pair_panel(ph.tail(LOOKBACKS[lookback]), a, b_, theme)

    # -------------------------------------------------- curves and seniority
    @app.callback(Output("cv-kpis", "children"), Output("cv-govt", "figure"),
                  Output("cv-slopes", "figure"), Output("cv-credit", "figure"),
                  Output("cv-carry", "figure"), Output("cv-dumbbell", "figure"),
                  Output("cv-senhist", "figure"), Output("cv-sen-table", "children"),
                  Input("etf", "value"), Input("lookback", "value"), Input("theme", "value"))
    def curves_tab(etf_id, lookback, theme):
        data = an.etf(etf_id)
        b, fits = an.basket(etf_id)
        n = LOOKBACKS[lookback]
        curve = an.govt_curve(etf_id)
        fit = curves.fit_nelson_siegel(list(curve.columns), curve.iloc[-1].to_numpy())
        slopes = curves.slope_history(curve)
        st = curves.slope_stats(slopes)
        sen = an.seniority(etf_id)
        sen_hist = curves.seniority_premium_history(b, data.oas_hist)
        tiles = [
            kpi("2Y yield", num(curve[2.0].iloc[-1], 2, suffix="%"),
                f"{(curve[2.0].iloc[-1] - curve[2.0].iloc[-22]) * 100:+.0f} bp in a month"),
            kpi("10Y yield", num(curve[10.0].iloc[-1], 2, suffix="%"),
                f"{(curve[10.0].iloc[-1] - curve[10.0].iloc[-22]) * 100:+.0f} bp in a month"),
            kpi("2s10s", bp(st.loc["2s10s", "last"], 0, True), f"{st.loc['2s10s', 'z']:+.1f}σ against one year"),
            kpi("5s30s", bp(st.loc["5s30s", "last"], 0, True), f"{st.loc['5s30s', 'z']:+.1f}σ against one year"),
        ]
        for bucket, f in [(k, fits[k]) for k in cfg.BUCKET_ORDER if k in fits][:3]:
            slope = float(curves.credit_curve_line(f, 5.0) - curves.credit_curve_line(f, 2.0))
            tiles.append(kpi(f"{bucket} credit curve, 2s5s", bp(slope, 0, True),
                             "Spread pickup from 2 to 5 years"))
        if len(sen):
            tiles.append(kpi("Seniority gaps beyond 1.5σ", str(int((sen["z"].abs() >= 1.5).sum())),
                             f"Out of {len(sen)} issuers on two tiers"))
        summary = curves.seniority_summary(b).reset_index()
        sen_tbl = table(summary, {"bonds": "{:.0f}", "weight": "{:.1f}", "bench_weight": "{:.1f}",
                                  "oas": "{:.0f}", "ytm": "{:.2f}", "mod_dur": "{:.2f}",
                                  "dts_contrib": "{:.0f}"},
                        {"seniority": "Tier", "bonds": "Bonds", "weight": "Fund %",
                         "bench_weight": "Bench %", "oas": "OAS bp", "ytm": "YTM %",
                         "mod_dur": "Dur. y", "dts_contrib": "DTS"})
        return (kpi_row(tiles),
                figs.govt_curve(curves.curve_snapshots(curve), fit, theme, data.spec.currency),
                figs.slopes_panel(slopes.tail(n), theme), figs.credit_curve(b, fits, theme),
                figs.carry_scatter(b, theme), figs.seniority_dumbbell(sen, theme),
                figs.seniority_history(sen_hist.tail(n), theme), sen_tbl)

    # ------------------------------------------------------------------- risk
    @app.callback(Output("rk-kpis", "children"), Output("rk-hist", "figure"),
                  Output("rk-backtest", "figure"), Output("rk-krd", "figure"),
                  Output("rk-dts", "figure"), Output("rk-tracking", "figure"),
                  Output("rk-expo", "children"), Output("rk-stress", "children"),
                  Input("etf", "value"), Input("lookback", "value"), Input("theme", "value"),
                  Input("rk-level", "value"), Input("rk-horizon", "value"))
    def risk_tab(etf_id, lookback, theme, level, horizon):
        data = an.etf(etf_id)
        d, spec = data.daily, data.spec
        b, _ = an.basket(etf_id)
        level, horizon = float(level or 0.99), int(horizon or 1)
        var = an.var(etf_id, level, horizon)
        var_1d = an.var(etf_id, level, 1)
        pnl = an.pnl(etf_id)
        bt, bts = risk.var_backtest(pnl["total"], 0.99)
        tr, ts = risk.tracking(d.tail(LOOKBACKS[lookback]), spec.ter_bps)
        _, ts_full = risk.tracking(d, spec.ter_bps)
        expo = risk.exposures(b)
        h = f"{horizon}-day"
        tiles = kpi_row([
            kpi(f"Historical VaR, {level:.0%}", bp(var["hist"] * 1e4),
                f"{h}, benchmark {var['bench_hist'] * 1e4:.0f} bp"),
            kpi("Parametric VaR", bp(var["param"] * 1e4), f"EWMA-weighted {var['param_ewma'] * 1e4:.0f} bp"),
            kpi("Cornish-Fisher VaR", bp(var["cornish_fisher"] * 1e4), "Adjusted for skew and fat tails"),
            kpi("Expected shortfall, 97.5%", bp(var["es"] * 1e4), f"Average {h} loss beyond the threshold"),
            kpi("VaR by factor", f"{var['hist_rates'] * 1e4:.0f} / {var['hist_spread'] * 1e4:.0f} bp",
                "Rates alone, then spreads alone"),
            kpi("Tracking error", bp(ts_full.get("te_ann_bps", np.nan)),
                f"Ex-ante {var['te_ex_ante'] * 1e4:.0f} bp"),
            kpi("Tracking difference", f"{ts_full.get('td_ann_bps', np.nan):+.0f} bp a year",
                f"TER {spec.ter_bps:.0f} bp, so {ts_full.get('excess_gross_bps', np.nan):+.0f} bp before fees"),
            kpi("Backtest exceptions", f"{bts['exceptions']:.0f} in {bts['n']:.0f} days",
                f"{bts['expected']:.1f} expected, Kupiec p-value {bts['kupiec_p']:.2f}",
                "good" if bts["kupiec_p"] > 0.05 else "warning"),
        ])
        contrib = risk.contribution_table(b, "sector").rename(columns={"dts": "fund", "bench_dts": "bench"})
        smart = lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"           # noqa: E731
        signed = lambda v: f"{v:+,.0f}" if abs(v) >= 100 else f"{v:+,.2f}"       # noqa: E731
        expo_tbl = table(expo.reset_index().rename(columns={"index": "Measure"}),
                         {"fund": smart, "bench": smart, "active": signed},
                         {"fund": "Fund", "bench": "Benchmark", "active": "Active"})
        stress_tbl = table(risk.stress_table(b), {"Fund (%)": "{:+.2f}", "Benchmark (%)": "{:+.2f}",
                                                  "Active (bp)": "{:+.0f}"})
        return (tiles, figs.var_histogram(pnl["total"], var_1d, theme, level),
                figs.var_backtest(bt, theme),
                figs.grouped_bars(risk.krd_table(b), theme, "Key rate duration, years"),
                figs.grouped_bars(contrib, theme, "Duration times spread contribution", ".0f",
                                  horizontal=True, height=360),
                figs.tracking_panel(tr, theme), expo_tbl, stress_tbl)

    # ------------------------------------------------------------ trade ideas
    @app.callback(Output("ideas-table", "rowData"), Output("ideas-table", "selectedRows"),
                  Input("etf", "value"), Input("id-scope", "value"), Input("id-status", "value"),
                  Input("id-types", "value"), Input("id-conv", "value"))
    def ideas_table(etf_id, scope, status, types, conv):
        df = an.ideas()
        if not len(df):
            return [], []
        if scope == "etf":
            df = df[df["etf"].isin([etf_id, "ALL"])]
        if status and status != "all":
            df = df[df["status"] == status]
        if types:
            df = df[df["type"].isin(types)]
        df = df[df["conviction"] >= int(conv or 1)]
        rows = _records(df[TABLE_COLS + ["rationale"]])
        return rows, ({"ids": [rows[0]["id"]]} if rows else [])

    @app.callback(Output("id-detail", "children"), Output("id-signal", "figure"),
                  Output("id-fan", "figure"), Output("id-pnl", "figure"),
                  Input("ideas-table", "selectedRows"), Input("ideas-table", "rowData"),
                  Input("theme", "value"))
    def idea_detail(selected, rows, theme):
        ideas = an.ideas()
        visible = [r["id"] for r in (rows or [])]
        if isinstance(selected, dict):                 # selection set by the callback
            picked = list(selected.get("ids", []))
        else:                                          # selection made with the mouse
            picked = [r.get("id") for r in (selected or [])]
        idea_id = picked[0] if picked and picked[0] in visible else (visible[0] if visible else None)
        if idea_id is None or not len(ideas) or idea_id not in set(ideas["id"]):
            blank = empty(theme, "No idea matches the filters.", 340)
            return html.Div("No idea matches the filters.", className="caption"), blank, blank, blank
        idea = ideas.set_index("id").loc[idea_id]
        sig = signal_for(an, idea["kind"], tuple(idea["ref"].split("|")))
        frame = signal_frame(sig)
        fmt = (lambda v: f"{v:.2f}x") if idea["unit"] == "ratio" else (lambda v: f"{v:,.0f} bp")
        lines = [("Long", idea["long_leg"]), ("Short", idea["short_leg"]),
                 ("Entry / target / stop", f"{fmt(idea['entry'])} / {fmt(idea['target'])} / {fmt(idea['stop'])}"),
                 ("Horizon", f"{int(idea['horizon_d'])} trading days"),
                 ("Expected P&L / risk", f"{idea['exp_pnl_bps']:,.0f} bp / {idea['risk_bps']:,.0f} bp"),
                 ("Carry over the horizon", f"{idea['carry_bps']:+,.0f} bp"),
                 ("Liquidity score", f"{idea['liquidity']:.0f} / 100")]
        simulate = idea["type"] not in ("Carry and roll", "ETF arbitrage")
        if simulate:
            sim = simulate_idea(idea, sig)
            lines += [("Target reached first", f"{sim['p_target']:.0%} of paths"),
                      ("Stop reached first", f"{sim['p_stop']:.0%} of paths"),
                      ("Simulated mean P&L", f"{sim['exp_pnl']:+,.0f} bp"),
                      ("Median holding period", f"{sim['median_days']:.0f} days")]
            fan, dist = figs.idea_fan(sim, idea, theme), figs.pnl_distribution(sim["pnl"], theme)
        else:
            msg = ("A carry position has no reversion signal to simulate."
                   if idea["type"] == "Carry and roll"
                   else "A primary arbitrage is locked in at execution, nothing to simulate.")
            fan = dist = empty(theme, msg, 340)
        detail = html.Div([
            html.Div([chip(idea["type"]), chip(idea["etf"]),
                      chip(idea["status"], "good" if idea["status"] == "Live" else "neutral"),
                      chip(f"Conviction {int(idea['conviction'])} / 5")], className="chips"),
            html.Div(idea["title"], className="idea-title"),
            html.Div(idea["rationale"], className="idea-text"),
            html.Table(html.Tbody([html.Tr([html.Td(k), html.Td(v, className="num")])
                                   for k, v in lines]), className="tbl tbl-kv"),
        ])
        return detail, figs.idea_signal(frame, idea, str(sig["label"]), theme), fan, dist
