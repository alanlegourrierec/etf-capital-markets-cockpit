"""
Risk and portfolio metrics.

  1. Exposures, with yield, OAS, modified duration, spread duration, DTS,
     key rate durations and contributions by sector, rating or seniority
  2. VaR, historical, parametric (delta-normal) and Cornish-Fisher, expected
     shortfall, contributions by factor and the Kupiec backtest
  3. Index tracking, with tracking difference and ex-post and ex-ante
     tracking error
  4. Deterministic stress scenarios

Fund weights sum to (1 - cash) and benchmark weights to 1. P&L figures are
returns (0.01 is 1%). A VaR is returned as a positive number.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, kurtosis, norm, skew

import config as cfg
from data.models import RatesData
from engine.bondmath import interp_weights, key_rate_durations

SQRT252 = np.sqrt(252.0)


# ----------------------------------------------------------------------------
# 1. Exposures
# ----------------------------------------------------------------------------
def exposures(basket: pd.DataFrame) -> pd.DataFrame:
    """Aggregate characteristics of the fund, the benchmark and the active gap."""
    w, bw = basket["weight"], basket["bench_weight"]
    inv = w.sum()            # invested share (excluding cash)

    def agg(col_fn):
        return float(col_fn(w)), float(col_fn(bw))

    rows = {
        "Yield to maturity (%)": agg(lambda x: (x * basket["ytm"]).sum() / x.sum()),
        "OAS (bp)": agg(lambda x: (x * basket["oas"]).sum() / x.sum()),
        "Modified duration (y)": agg(lambda x: (x * basket["mod_dur"]).sum()),
        "Spread duration (y)": agg(lambda x: (x * basket["spread_dur"]).sum()),
        "DTS (y x bp)": agg(lambda x: (x * basket["spread_dur"] * basket["oas"]).sum()),
        "Convexity": agg(lambda x: (x * basket["convexity"]).sum()),
        "Invested (%)": (float(inv * 100.0), 100.0),
        "Number of bonds": (float((w > 0).sum()), float((bw > 0).sum())),
    }
    out = pd.DataFrame(rows, index=["fund", "bench"]).T
    out["active"] = out["fund"] - out["bench"]
    return out


def krd_table(basket: pd.DataFrame, key_tenors: Sequence[float] = cfg.KEY_TENORS) -> pd.DataFrame:
    """Key rate durations of the fund and the benchmark (in years)."""
    krd = key_rate_durations(basket["mod_dur"].to_numpy(), basket["maturity_years"].to_numpy(),
                             key_tenors)
    out = pd.DataFrame({"fund": basket["weight"].to_numpy() @ krd,
                        "bench": basket["bench_weight"].to_numpy() @ krd},
                       index=pd.Index(key_tenors, name="tenor"))
    out["active"] = out["fund"] - out["bench"]
    return out


def contribution_table(basket: pd.DataFrame, by: str = "sector") -> pd.DataFrame:
    """Weight and DTS contribution by group, fund against benchmark."""
    b = basket.assign(
        _dts=basket["weight"] * basket["spread_dur"] * basket["oas"],
        _bdts=basket["bench_weight"] * basket["spread_dur"] * basket["oas"],
        _dur=basket["weight"] * basket["mod_dur"],
        _bdur=basket["bench_weight"] * basket["mod_dur"])
    g = b.groupby(by, sort=False, observed=True).agg(
        weight=("weight", "sum"), bench_weight=("bench_weight", "sum"),
        dts=("_dts", "sum"), bench_dts=("_bdts", "sum"),
        dur=("_dur", "sum"), bench_dur=("_bdur", "sum"))
    g[["weight", "bench_weight"]] *= 100.0
    g["active_weight"] = g["weight"] - g["bench_weight"]
    g["active_dts"] = g["dts"] - g["bench_dts"]
    g["active_dur"] = g["dur"] - g["bench_dur"]
    return g.sort_values("dts", ascending=False)


# ----------------------------------------------------------------------------
# 2. Historical P&L of today's portfolio and VaR
# ----------------------------------------------------------------------------
def bond_returns(basket: pd.DataFrame, rates: RatesData,
                 oas_hist: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Daily returns each bond of TODAY'S BASKET would have had under past moves
    in rates and spreads (historical simulation).

        r = carry - MD x d(rate) + 1/2 Cx d(rate)^2 - SD x d(spread)
    """
    ids = [i for i in basket.index if i in oas_hist.columns]
    b = basket.loc[ids]
    idx = oas_hist.index
    tau = b["maturity_years"].to_numpy(dtype=float)
    dy = np.zeros((len(idx), len(ids)))                    # change in the risk-free rate, in %
    for ccy, curve in rates.curves.items():
        mask = (b["currency"] == ccy).to_numpy()
        if mask.any():
            c = curve.reindex(idx).ffill().bfill()
            tenors = np.asarray(c.columns, dtype=float)
            dy[:, mask] = c.diff().fillna(0.0).to_numpy() @ interp_weights(tenors, tau[mask]).T
    d_oas = oas_hist[ids].diff().fillna(0.0).to_numpy()    # in bp
    md = b["mod_dur"].to_numpy()[None, :]
    r_rates = -md * dy / 100.0 + 0.5 * b["convexity"].to_numpy()[None, :] * (dy / 100.0) ** 2
    r_spread = -b["spread_dur"].to_numpy()[None, :] * d_oas / 1e4
    carry = np.broadcast_to(b["ytm"].to_numpy()[None, :] / 100.0 / 252.0, r_rates.shape)
    frame = lambda a: pd.DataFrame(a, index=idx, columns=ids).iloc[1:]     # noqa: E731
    return {"rates": frame(r_rates), "spread": frame(r_spread), "carry": frame(carry),
            "total": frame(r_rates + r_spread + carry)}


def portfolio_pnl(basket: pd.DataFrame, parts: Dict[str, pd.DataFrame],
                  weight_col: str = "weight") -> pd.DataFrame:
    """Adds up returns per bond into portfolio P&L, by factor."""
    w = basket.loc[parts["total"].columns, weight_col].to_numpy()
    return pd.DataFrame({k: v.to_numpy() @ w for k, v in parts.items()},
                        index=parts["total"].index)


def var_historical(pnl: pd.Series, level: float = 0.99, horizon: int = 1) -> float:
    return float(-np.quantile(pnl.dropna(), 1.0 - level) * np.sqrt(horizon))


def expected_shortfall(pnl: pd.Series, level: float = 0.975, horizon: int = 1) -> float:
    x = pnl.dropna().to_numpy()
    cut = np.quantile(x, 1.0 - level)
    tail = x[x <= cut]
    return float(-tail.mean() * np.sqrt(horizon)) if len(tail) else 0.0


def covariance(returns: pd.DataFrame, ewma_lambda: float = 0.0) -> np.ndarray:
    """Covariance of returns. ewma_lambda > 0 gives more weight to the recent past."""
    x = returns.to_numpy() - returns.to_numpy().mean(axis=0)
    if ewma_lambda <= 0:
        return x.T @ x / max(len(x) - 1, 1)
    w = ewma_lambda ** np.arange(len(x) - 1, -1, -1)
    w /= w.sum()
    return (x * w[:, None]).T @ x


def var_parametric(weights: np.ndarray, cov: np.ndarray, level: float = 0.99,
                   horizon: int = 1) -> float:
    """Delta-normal VaR, the Gaussian quantile times portfolio volatility."""
    sigma = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    return float(norm.ppf(level) * sigma * np.sqrt(horizon))


def var_cornish_fisher(pnl: pd.Series, level: float = 0.99, horizon: int = 1) -> float:
    """Parametric VaR corrected for skew and fat tails."""
    x = pnl.dropna().to_numpy()
    z = norm.ppf(1.0 - level)
    s, k = float(skew(x)), float(kurtosis(x))
    z_cf = (z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24
            - (2 * z ** 3 - 5 * z) * s ** 2 / 36)
    return float(-(x.mean() + z_cf * x.std()) * np.sqrt(horizon))


def component_var(basket: pd.DataFrame, total: pd.DataFrame, level: float = 0.99,
                  by: str = "sector") -> pd.DataFrame:
    """
    Euler decomposition of the parametric VaR. The contribution of each bond is
    weight x (Sigma w) / sigma_p x quantile, then summed by group.
    """
    ids = list(total.columns)
    w = basket.loc[ids, "weight"].to_numpy()
    cov = covariance(total)
    sigma = np.sqrt(max(w @ cov @ w, 1e-18))
    comp = w * (cov @ w) / sigma * norm.ppf(level)
    out = pd.DataFrame({"component_var": comp, by: basket.loc[ids, by].to_numpy()})
    g = out.groupby(by, sort=False, observed=True)["component_var"].sum().to_frame()
    g["share"] = g["component_var"] / g["component_var"].sum() * 100.0
    return g.sort_values("component_var", ascending=False)


def var_summary(basket: pd.DataFrame, parts: Dict[str, pd.DataFrame],
                level: float = 0.99, horizon: int = 1) -> Dict[str, float]:
    """Every VaR measure of the fund, the benchmark and each factor."""
    pnl = portfolio_pnl(basket, parts, "weight")
    bench = portfolio_pnl(basket, parts, "bench_weight")
    ids = list(parts["total"].columns)
    w = basket.loc[ids, "weight"].to_numpy()
    active = w - basket.loc[ids, "bench_weight"].to_numpy()      # cash counts as an active position
    cov = covariance(parts["total"])
    return {
        "hist": var_historical(pnl["total"], level, horizon),
        "param": var_parametric(w, cov, level, horizon),
        "param_ewma": var_parametric(w, covariance(parts["total"], 0.94), level, horizon),
        "cornish_fisher": var_cornish_fisher(pnl["total"], level, horizon),
        "es": expected_shortfall(pnl["total"], 0.975, horizon),
        "hist_rates": var_historical(pnl["rates"], level, horizon),
        "hist_spread": var_historical(pnl["spread"], level, horizon),
        "bench_hist": var_historical(bench["total"], level, horizon),
        "vol_ann": float(pnl["total"].std() * SQRT252),
        "te_ex_ante": float(np.sqrt(max(active @ cov @ active, 0.0)) * SQRT252),
    }


def var_backtest(pnl: pd.Series, level: float = 0.99, window: int = 250) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Backtest. Each day, the historical VaR estimated on the previous `window`
    days is compared with the realised P&L. Kupiec coverage test.
    """
    var = -pnl.rolling(window, min_periods=window // 2).quantile(1.0 - level).shift(1)
    bt = pd.DataFrame({"pnl": pnl, "var": var}).dropna()
    bt["breach"] = bt["pnl"] < -bt["var"]
    n, x = len(bt), int(bt["breach"].sum())
    p = 1.0 - level
    stats = {"n": float(n), "exceptions": float(x), "expected": float(n * p),
             "kupiec_lr": 0.0, "kupiec_p": 1.0}
    if n > 0:
        rate = min(max(x / n, 1e-9), 1 - 1e-9)
        lr = -2.0 * ((n - x) * np.log(1 - p) + x * np.log(p)
                     - (n - x) * np.log(1 - rate) - x * np.log(rate))
        stats["kupiec_lr"] = float(max(lr, 0.0))
        stats["kupiec_p"] = float(1.0 - chi2.cdf(max(lr, 0.0), 1))
    return bt, stats


# ----------------------------------------------------------------------------
# 3. Index tracking
# ----------------------------------------------------------------------------
def tracking(daily: pd.DataFrame, ter_bps: float, window: int = 63) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Tracking difference (cumulative performance gap, NAV against benchmark) and
    tracking error (annualised volatility of the daily gap).
    """
    d = daily[["nav", "bench"]].dropna()
    if len(d) < 20:
        return pd.DataFrame(), {}
    rf, rb = d["nav"].pct_change(fill_method=None), d["bench"].pct_change(fill_method=None)
    active = (rf - rb).fillna(0.0)
    fund_idx = 100.0 * d["nav"] / d["nav"].iloc[0]
    bench_idx = 100.0 * d["bench"] / d["bench"].iloc[0]
    out = pd.DataFrame({
        "fund_idx": fund_idx, "bench_idx": bench_idx,
        "td_cum_bps": (fund_idx / bench_idx - 1.0) * 1e4,
        "active_bps": active * 1e4,
        "te_roll_bps": active.rolling(window, min_periods=window // 2).std() * SQRT252 * 1e4,
    })
    years = max(len(d) / 252.0, 1e-9)
    td_ann = ((fund_idx.iloc[-1] / bench_idx.iloc[-1]) ** (1.0 / years) - 1.0) * 1e4
    te = float(active.iloc[1:].std() * SQRT252 * 1e4)
    stats = {"td_ann_bps": float(td_ann), "te_ann_bps": te,
             "ir": float(td_ann / te) if te > 0 else 0.0,
             "ter_bps": float(ter_bps), "excess_gross_bps": float(td_ann + ter_bps),
             "td_1y_bps": float((fund_idx.iloc[-1] / fund_idx.iloc[-min(253, len(d))]
                                 / (bench_idx.iloc[-1] / bench_idx.iloc[-min(253, len(d))]) - 1) * 1e4),
             "years": float(years)}
    return out, stats


# ----------------------------------------------------------------------------
# 4. Stress scenarios
# ----------------------------------------------------------------------------
SPREAD_SHOCK_BP = {"AA+": 30.0, "A": 45.0, "BBB": 75.0, "BB": 180.0, "B": 300.0, "CCC": 500.0}


def stress_table(basket: pd.DataFrame) -> pd.DataFrame:
    """Instant impact (in %) of rate and spread scenarios, fund and benchmark."""
    tau = basket["maturity_years"].to_numpy(dtype=float)
    md, sd = basket["mod_dur"].to_numpy(), basket["spread_dur"].to_numpy()
    cx, oas = basket["convexity"].to_numpy(), basket["oas"].to_numpy()
    credit = basket["bucket"].map(SPREAD_SHOCK_BP).fillna(100.0).to_numpy(dtype=float)
    steep = np.interp(tau, [2.0, 30.0], [25.0, 100.0])
    scenarios: List[Tuple[str, np.ndarray, np.ndarray]] = [
        ("Rates +100 bp, parallel", np.full_like(tau, 100.0), np.zeros_like(tau)),
        ("Rates -100 bp, parallel", np.full_like(tau, -100.0), np.zeros_like(tau)),
        ("Bear steepener, 2Y +25 / 30Y +100", steep, np.zeros_like(tau)),
        ("Credit sell-off, by rating (BBB +75 / BB +180 / B +300)", np.zeros_like(tau), credit),
        ("Risk-off, rates -60 bp and spreads x1.6", np.full_like(tau, -60.0), 0.6 * oas),
        ("Spread compression, spreads x0.8", np.zeros_like(tau), -0.2 * oas),
    ]
    rows = []
    for name, dy_bp, ds_bp in scenarios:
        r = -md * dy_bp / 1e4 + 0.5 * cx * (dy_bp / 1e4) ** 2 - sd * ds_bp / 1e4
        fund = float((basket["weight"].to_numpy() * r).sum() * 100.0)
        bench = float((basket["bench_weight"].to_numpy() * r).sum() * 100.0)
        rows.append({"Scenario": name, "Fund (%)": fund, "Benchmark (%)": bench,
                     "Active (bp)": (fund - bench) * 100.0})
    return pd.DataFrame(rows)
