"""
Capital structure, the link between equity and credit (Merton model).

The share is a call option on the firm's assets, struck at the debt. From
the market cap and the equity volatility we back out the value and volatility
of the assets, then the distance to default (DD). Issuer by issuer, the market
spread is then compared with the spread its DD "deserves".

The chain of calculations.
  1. solve_merton          (E, sigma_E, D) -> (V, sigma_V)
  2. distance_to_default   DD and risk-neutral default probability
  3. merton_panel          DD of every issuer at every date
  4. fair_value            cross-sectional fit of ln(spread) on ln(1 + DD).
                           The residual says how cheap credit is to equity
  5. snapshot              today's table with z-scores and the hedge ratio
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

SQRT252 = np.sqrt(252.0)


# ----------------------------------------------------------------------------
# 1. Core of the model
# ----------------------------------------------------------------------------
def solve_merton(E, sigma_E, D, r, T: float = 1.0, n_iter: int = 80):
    """
    Solves the Merton system by damped iterations (vectorised).

        E         = V N(d1) - D exp(-rT) N(d2)
        sigma_E E = N(d1) sigma_V V
    """
    E, sigma_E, D, r = np.broadcast_arrays(*(np.asarray(a, dtype=float)
                                             for a in (E, sigma_E, D, r)))
    disc = D * np.exp(-r * T)
    V = E + disc
    sV = sigma_E * E / V
    sqT = np.sqrt(T)
    for _ in range(n_iter):
        sV = np.clip(sV, 1e-4, 3.0)
        d1 = (np.log(V / D) + (r + 0.5 * sV ** 2) * T) / (sV * sqT)
        nd1 = np.maximum(norm.cdf(d1), 1e-6)
        e_model = V * nd1 - disc * norm.cdf(d1 - sV * sqT)
        V = np.maximum(V - (e_model - E) / nd1, E * 1.0001)      # Newton step on V
        sV = 0.5 * sV + 0.5 * sigma_E * E / (V * nd1)            # damped update
    return V, np.clip(sV, 1e-4, 3.0)


def distance_to_default(V, sV, D, r, T: float = 1.0):
    """DD is d2 of the model. Returns (DD, risk-neutral default probability)."""
    dd = (np.log(V / D) + (r - 0.5 * sV ** 2) * T) / (sV * np.sqrt(T))
    return dd, norm.cdf(-dd)


def merton_spread_bps(V, sV, D, r, T: float = 5.0):
    """Model credit spread of zero-coupon debt of maturity T."""
    sqT = np.sqrt(T)
    d1 = (np.log(V / D) + (r + 0.5 * sV ** 2) * T) / (sV * sqT)
    d2 = d1 - sV * sqT
    quasi_lev = D * np.exp(-r * T) / V
    inner = norm.cdf(d2) + norm.cdf(-d1) / quasi_lev
    return -np.log(np.clip(inner, 1e-12, None)) / T * 1e4


# ----------------------------------------------------------------------------
# 2. Issuers x dates panel
# ----------------------------------------------------------------------------
def equity_vol(equity: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Annualised realised volatility of the share, rolling window."""
    ret = np.log(equity).diff()
    vol = ret.rolling(window, min_periods=max(20, window // 3)).std() * SQRT252
    return vol.bfill().clip(lower=0.05)


def merton_panel(static: pd.DataFrame, equity: pd.DataFrame, rf: pd.Series,
                 vol_window: int = 126) -> Dict[str, pd.DataFrame]:
    """
    For each issuer and each date, computes asset value and volatility, the
    1-year distance to default (KMV-style default point, short debt plus half
    of long debt) and the 5-year Merton spread.
    """
    tick = [t for t in static.index if t in equity.columns]
    eq = equity[tick].ffill()
    st = static.loc[tick]
    mcap = eq / eq.iloc[-1] * st["mktcap_musd"].to_numpy()
    sig = equity_vol(eq, vol_window)
    r = (rf.reindex(eq.index).ffill().bfill().to_numpy() / 100.0)[:, None]
    default_point = (st["st_debt_musd"] + 0.5 * st["lt_debt_musd"]).to_numpy()[None, :]
    debt = st["debt_musd"].to_numpy()[None, :]

    V, sV = solve_merton(mcap.to_numpy(), sig.to_numpy(), default_point, r, 1.0)
    dd, pdef = distance_to_default(V, sV, default_point, r, 1.0)
    V5, sV5 = solve_merton(mcap.to_numpy(), sig.to_numpy(), debt, r, 5.0)
    model = merton_spread_bps(V5, sV5, debt, r, 5.0)

    def frame(a):
        return pd.DataFrame(a, index=eq.index, columns=tick)

    return {"dd": frame(dd), "pd": frame(pdef), "asset_value": frame(V),
            "asset_vol": frame(sV), "equity_vol": sig,
            "leverage": frame(default_point / V), "model_spread": frame(model)}


def fair_value(dd: pd.DataFrame, spread: pd.DataFrame,
               exclude: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Spread "deserved" by the distance to default.

    At each date we fit, across issuers,
        ln(spread) = a + b ln(1 + DD)
    The residual measures how far credit trades above (positive, credit cheap
    to equity) or below what the equity says.
    """
    idx = dd.index.intersection(spread.index)
    cols = [c for c in dd.columns if c in spread.columns]
    fit_cols = [c for c in cols if not exclude or c not in set(exclude)]
    x = np.log1p(dd.loc[idx, cols].clip(lower=0.0))
    y = np.log(spread.loc[idx, cols].clip(lower=1.0))
    xf, yf = x[fit_cols].to_numpy(), y[fit_cols].to_numpy()
    xm, ym = np.nanmean(xf, axis=1, keepdims=True), np.nanmean(yf, axis=1, keepdims=True)
    b = (np.nansum((xf - xm) * (yf - ym), axis=1, keepdims=True)
         / np.maximum(np.nansum((xf - xm) ** 2, axis=1, keepdims=True), 1e-12))
    a = ym - b * xm
    fitted = a + b * x.to_numpy()
    resid = y.to_numpy() - fitted
    return {"fair": pd.DataFrame(np.exp(fitted), index=idx, columns=cols),
            "resid": pd.DataFrame(resid, index=idx, columns=cols),
            "coef": pd.DataFrame({"a": a[:, 0], "b": b[:, 0]}, index=idx)}


def equity_beta(spread: pd.DataFrame, equity: pd.DataFrame, window: int = 126) -> pd.Series:
    """Empirical elasticity, the change in ln(spread) for a return of 1 on the share."""
    cols = [c for c in spread.columns if c in equity.columns]
    ds = np.log(spread[cols]).diff().iloc[-window:]
    de = np.log(equity[cols]).diff().reindex(ds.index)
    cov = ((ds - ds.mean()) * (de - de.mean())).sum()
    var = ((de - de.mean()) ** 2).sum()
    return (cov / var.replace(0.0, np.nan)).fillna(0.0)


def snapshot(static: pd.DataFrame, panel: Dict[str, pd.DataFrame],
             fv: Dict[str, pd.DataFrame], spread: pd.DataFrame, equity: pd.DataFrame,
             z_window: int = 126, spread_dur: float = 4.3) -> pd.DataFrame:
    """Today's table with Merton fundamentals, relative cheapness and the equity hedge."""
    resid = fv["resid"]
    cols = list(resid.columns)
    roll = resid.rolling(z_window, min_periods=max(20, z_window // 3))
    mean, sd = roll.mean().iloc[-1], roll.std().iloc[-1].replace(0.0, np.nan)
    last = resid.iloc[-1]
    mad = (last - last.median()).abs().median() * 1.4826
    beta = equity_beta(spread, equity).reindex(cols)
    mkt = spread[cols].iloc[-1]
    out = static.loc[cols, ["name", "sector", "rating", "bucket", "universe",
                            "mktcap_musd", "debt_musd"]].copy()
    out["equity_vol"] = panel["equity_vol"][cols].iloc[-1]
    out["asset_vol"] = panel["asset_vol"][cols].iloc[-1]
    out["leverage"] = panel["leverage"][cols].iloc[-1]
    out["dd"] = panel["dd"][cols].iloc[-1]
    out["pd_1y"] = panel["pd"][cols].iloc[-1]
    out["model_spread"] = panel["model_spread"][cols].iloc[-1]
    out["mkt_spread"] = mkt
    out["fair_spread"] = fv["fair"].iloc[-1]
    out["resid"] = last
    out["resid_mean"] = mean
    out["resid_sd"] = sd
    out["cs_z"] = (last - last.median()) / (mad if mad > 0 else np.nan)
    out["ts_z"] = (last - mean) / sd
    out["eq_beta"] = beta
    # Amount of equity to sell per 100 of bond bought (in %)
    out["hedge_pct"] = (spread_dur * mkt / 1e4 * beta.abs() * 100.0)
    return out.sort_values("ts_z", ascending=False)
