"""
ETF dynamics and capital markets.

  1. Premium or discount to NAV, with Bollinger bands
  2. Costs of an Authorized Participant and break-even thresholds of the
     arbitrage (create when the premium is above the upper threshold, redeem
     when the discount is below the lower one)
  3. Liquidity of the creation and redemption basket against the ETF shares

Every cost is in basis points of the order notional.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

import config as cfg

BASE_BA_BPS = {"AA+": 12.0, "A": 15.0, "BBB": 20.0, "BB": 38.0, "B": 55.0, "CCC": 90.0}


# ----------------------------------------------------------------------------
# 1. Premium and discount
# ----------------------------------------------------------------------------
def premium_discount(daily: pd.DataFrame) -> pd.Series:
    """Premium (+) or discount (-) of the ETF price to its NAV, in bp."""
    return ((daily["price"] / daily["nav"] - 1.0) * 1e4).rename("pd_bps")


def bollinger(s: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    roll = s.rolling(window, min_periods=max(5, window // 2))
    mid, sd = roll.mean(), roll.std()
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd,
                         "z": (s - mid) / sd.replace(0.0, np.nan)})


def pd_stats(pd_bps: pd.Series) -> Dict[str, float]:
    """Descriptive statistics of the premium over the window passed in."""
    s = pd_bps.dropna()
    if s.empty:
        return {}
    return {"last": float(s.iloc[-1]), "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
            "p05": float(s.quantile(0.05)), "p95": float(s.quantile(0.95)),
            "pct_rank": float((s <= s.iloc[-1]).mean() * 100.0),
            "days_premium": float((s > 0).mean() * 100.0)}


# ----------------------------------------------------------------------------
# 2. Liquidity of the basket bonds
# ----------------------------------------------------------------------------
def model_liquidity(basket: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Daily volume and bid/ask estimated when the data is missing (LSEG mode).

    A rule of thumb. Liquidity grows with the amount outstanding and fades as
    the bond ages, and the bid/ask depends on rating and duration.
    """
    amt = basket["amt_out_musd"].to_numpy(dtype=float)
    age = basket["age_years"].to_numpy(dtype=float)
    hy = basket["bucket"].isin(["BB", "B", "CCC"]).to_numpy()
    rel = np.log(amt / np.median(amt)) * 0.55 - 0.22 * (age - 3.0)
    adv = np.where(hy, 3.2, 6.0) * np.exp(0.75 * np.clip(rel / 0.6, -2.5, 2.5))
    base = basket["bucket"].map(BASE_BA_BPS).fillna(30.0).to_numpy(dtype=float)
    ba = base * (0.70 + 0.06 * basket["mod_dur"].to_numpy(dtype=float)) * np.exp(
        -0.30 * np.clip(rel / 0.6, -2.5, 2.5))
    return adv, ba


def liquidity_score(basket: pd.DataFrame) -> pd.Series:
    """Score from 0 to 100, from percentile ranks of volume, bid/ask, size and age."""
    r = lambda col, asc: basket[col].rank(pct=True, ascending=asc)     # noqa: E731
    score = (0.40 * r("adv_musd", True) + 0.30 * r("ba_bps", False)
             + 0.20 * r("amt_out_musd", True) + 0.10 * r("age_years", False))
    return (100.0 * score).rename("liq_score")


def maturity_bucket(years) -> pd.Categorical:
    labels = [b[0] for b in cfg.MATURITY_BUCKETS]
    edges = [b[1] for b in cfg.MATURITY_BUCKETS] + [1e9]
    return pd.cut(pd.Series(years, dtype=float), bins=edges, labels=labels, right=False).to_numpy()


def enrich_basket(basket: pd.DataFrame) -> pd.DataFrame:
    """Adds a liquidity score, a maturity bucket and DTS to each bond."""
    b = basket.copy()
    b["liq_score"] = liquidity_score(b)
    b["mat_bucket"] = maturity_bucket(b["maturity_years"].to_numpy())
    b["dts"] = b["spread_dur"] * b["oas"]
    return b


def liquidity_heatmap(basket: pd.DataFrame, rows: str = "sector", cols: str = "mat_bucket",
                      value: str = "liq_score") -> pd.DataFrame:
    """Fund-weighted average of `value` in each rows x cols cell."""
    b = basket.assign(_wv=basket[value] * basket["weight"])
    num = b.pivot_table(index=rows, columns=cols, values="_wv", aggfunc="sum", observed=False)
    den = b.pivot_table(index=rows, columns=cols, values="weight", aggfunc="sum", observed=False)
    out = num / den.where(den > 0)
    order = [c for c in [m[0] for m in cfg.MATURITY_BUCKETS] if c in out.columns]
    return out[order] if cols == "mat_bucket" else out


def basket_capacity(basket: pd.DataFrame, participation: float = 0.20,
                    coverage: float = 0.90) -> float:
    """
    Order size (in millions) for which `coverage` of the basket, by weight,
    trades in one session without taking more than `participation` of any
    bond's daily volume.
    """
    w = (basket["weight"] / basket["weight"].sum()).to_numpy()
    cap = participation * basket["adv_musd"].to_numpy() / np.maximum(w, 1e-9)
    order = np.argsort(-cap)                      # from the easiest bond to the hardest
    cum = np.cumsum(w[order])
    return float(cap[order][np.searchsorted(cum, coverage, side="left").clip(max=len(w) - 1)])


def liquidation_profile(basket: pd.DataFrame, sizes_musd: Sequence[float],
                        participation: float = 0.20,
                        horizons: Sequence[int] = (1, 3, 5)) -> pd.DataFrame:
    """Share of the basket (in %) that trades in 1, 3 or 5 sessions, by order size."""
    w = (basket["weight"] / basket["weight"].sum()).to_numpy()
    daily_cap = participation * basket["adv_musd"].to_numpy()
    out = {}
    for h in horizons:
        out[f"{h}d"] = [float((np.minimum(w * s, h * daily_cap)).sum() / s * 100.0)
                        for s in sizes_musd]
    return pd.DataFrame(out, index=pd.Index(sizes_musd, name="order_musd"))


def primary_secondary(daily: pd.DataFrame, basket: pd.DataFrame,
                      participation: float = 0.20) -> pd.DataFrame:
    """
    Primary liquidity (daily capacity of the basket) and secondary liquidity
    (value traded on the ETF), in millions. Today's capacity is extended to the
    history in proportion to basket volume.
    """
    cap_today = basket_capacity(basket, participation)
    turn = daily["basket_turnover_musd"]
    cap = cap_today * turn / turn.iloc[-1]
    return pd.DataFrame({"etf_value_musd": daily["value_musd"], "basket_capacity_musd": cap,
                         "ratio": daily["value_musd"] / cap,
                         "pd_bps": premium_discount(daily)})


# ----------------------------------------------------------------------------
# 3. Authorized Participant arbitrage
# ----------------------------------------------------------------------------
def order_notional(spec: cfg.EtfSpec, nav: float, order_cu: int) -> float:
    """Notional of the primary order, in dollars."""
    return float(order_cu) * spec.cu_shares * float(nav)


def market_impact_bps(basket: pd.DataFrame, order_usd: float, coeff: float) -> float:
    """
    Market impact of the basket, a square-root model bond by bond.
        impact_i = coeff x daily vol_i x sqrt(amount_i / daily volume_i)
    """
    w = (basket["weight"] / basket["weight"].sum()).to_numpy()
    sigma = basket["mod_dur"].to_numpy() * (4.5 + 0.018 * basket["oas"].to_numpy())
    q = w * order_usd / 1e6
    return float((w * coeff * sigma * np.sqrt(q / np.maximum(basket["adv_musd"].to_numpy(), 1e-6))).sum())


def ap_cost_stack(spec: cfg.EtfSpec, basket: pd.DataFrame, nav: float, etf_ba_bps: float,
                  basket_ba_bps: float, p: cfg.ApParams) -> pd.DataFrame:
    """
    Cost stack of an AP, in bp, for a creation and for a redemption.

    Creation    the AP buys the basket at the offer, delivers it to the issuer,
                receives shares and sells them on the market at the bid.
    Redemption  it buys the shares at the offer, hands them back, receives the
                basket and sells it at the bid.

    When the NAV prices the bonds at the bid, buying the basket costs the whole
    bid/ask above NAV while selling it costs nothing. The arbitrage band is
    therefore lopsided, which is where the structural premium of bond ETFs
    comes from.
    """
    order_usd = order_notional(spec, nav, p.order_cu)
    ba = basket_ba_bps * p.stress_mult
    buy_leg = ba if p.nav_basis == "bid" else ba / 2.0
    sell_leg = 0.0 if p.nav_basis == "bid" else ba / 2.0
    impact = market_impact_bps(basket, order_usd, p.impact_coeff)
    fees = spec.fixed_fee_usd / max(order_usd, 1.0) * 1e4 + spec.var_fee_bps
    funding = p.funding_rate_pct / 100.0 * spec.settle_mismatch_days / 360.0 * 1e4
    half_etf = etf_ba_bps / 2.0
    stack = pd.DataFrame({
        "creation": [buy_leg, impact, half_etf, fees, p.hedge_cost_bps, funding],
        "redemption": [sell_leg, impact, half_etf, fees, p.hedge_cost_bps, funding],
    }, index=["Basket bid/ask vs NAV", "Market impact", "ETF half-spread",
              "Primary fees", "Hedging", "Funding"])
    return stack


def arb_bands(daily: pd.DataFrame, spec: cfg.EtfSpec, basket: pd.DataFrame,
              p: cfg.ApParams) -> pd.DataFrame:
    """
    Arbitrage thresholds through time and the windows that opened.

    upper   premium above which a creation pays
    lower   discount (negative) below which a redemption pays
    edge    net gain in bp when a window is open, 0 otherwise
    signal  +1 creation, -1 redemption, 0 nothing
    """
    pdb = premium_discount(daily)
    ba = daily["basket_ba_bps"] * p.stress_mult
    order_usd = p.order_cu * spec.cu_shares * daily["nav"]
    impact = market_impact_bps(basket, float(order_usd.iloc[-1]), p.impact_coeff)
    fees = spec.fixed_fee_usd / order_usd * 1e4 + spec.var_fee_bps
    funding = p.funding_rate_pct / 100.0 * spec.settle_mismatch_days / 360.0 * 1e4
    common = impact + daily["etf_ba_bps"] / 2.0 + fees + p.hedge_cost_bps + funding
    upper = common + (ba if p.nav_basis == "bid" else ba / 2.0)
    lower = -(common + (0.0 if p.nav_basis == "bid" else ba / 2.0))
    edge_c = (pdb - upper).clip(lower=0.0)
    edge_r = (lower - pdb).clip(lower=0.0)
    signal = np.where(edge_c > 0, 1, np.where(edge_r > 0, -1, 0))
    edge = edge_c + edge_r
    return pd.DataFrame({"pd_bps": pdb, "upper": upper, "lower": lower, "edge_bps": edge,
                         "signal": signal, "pnl_usd": edge / 1e4 * order_usd})


def arb_pnl_grid(spec: cfg.EtfSpec, basket: pd.DataFrame, nav: float, etf_ba_bps: float,
                 basket_ba_bps: float, p: cfg.ApParams, sizes_cu: Sequence[int],
                 levels_bps: Sequence[float]) -> pd.DataFrame:
    """
    Net gain of the arbitrage (in bp) by order size (columns, in creation
    units) and by premium or discount level (rows). Negative means no
    arbitrage. Large orders dilute the fixed fee but pay more impact.
    """
    grid = np.empty((len(levels_bps), len(sizes_cu)))
    for j, size in enumerate(sizes_cu):
        q = cfg.ApParams(**{**p.__dict__, "order_cu": int(size)})
        stack = ap_cost_stack(spec, basket, nav, etf_ba_bps, basket_ba_bps, q)
        c_cost, r_cost = stack["creation"].sum(), stack["redemption"].sum()
        for i, lvl in enumerate(levels_bps):
            grid[i, j] = lvl - c_cost if lvl >= 0 else -lvl - r_cost
    return pd.DataFrame(grid, index=pd.Index(levels_bps, name="pd_bps"),
                        columns=pd.Index(sizes_cu, name="order_cu"))


def arb_windows(bands: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Latest sessions where an arbitrage window was open."""
    w = bands[bands["signal"] != 0].tail(n).iloc[::-1].copy()
    w["side"] = np.where(w["signal"] > 0, "Create", "Redeem")
    w["threshold"] = np.where(w["signal"] > 0, w["upper"], w["lower"])
    return w[["side", "pd_bps", "threshold", "edge_bps", "pnl_usd"]]


def estimate_inav(prev_nav: float, weights, prev_prices, live_prices, cash_weight: float = 0.0) -> float:
    """
    iNAV estimated from the intraday prices of the basket, when no iNAV RIC is
    available. It is yesterday's NAV moved by the weighted performance.
    """
    w = np.asarray(weights, dtype=float)
    perf = np.asarray(live_prices, dtype=float) / np.asarray(prev_prices, dtype=float) - 1.0
    return float(prev_nav * (1.0 + (1.0 - cash_weight) * np.nansum(w * perf) / np.nansum(w)))
