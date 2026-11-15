"""
Credit relative value, pairs and sectors.

Simple statistical tools that stay readable.
  rolling_z     rolling z-score of a series
  half_life     half-life of mean reversion (AR(1))
  df_tstat      Dickey-Fuller statistic (the more negative, the faster the
                series comes back to its mean)
  scan_pairs    sweep of every pair within a sector
  sector_*      how rich or cheap sectors are to each other
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def rolling_z(s: pd.Series, window: int) -> pd.Series:
    roll = s.rolling(window, min_periods=max(10, window // 3))
    return (s - roll.mean()) / roll.std().replace(0.0, np.nan)


def _ar1(x: np.ndarray) -> Tuple[float, float]:
    """Regression dx_t = a + b x_{t-1}. Returns (b, t-stat of b)."""
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return 0.0, 0.0
    lag, dx = x[:-1], np.diff(x)
    lag_c = lag - lag.mean()
    sxx = float((lag_c ** 2).sum())
    if sxx <= 0:
        return 0.0, 0.0
    b = float((lag_c * (dx - dx.mean())).sum() / sxx)
    resid = dx - dx.mean() - b * lag_c
    se = np.sqrt(float((resid ** 2).sum()) / max(len(dx) - 2, 1) / sxx)
    return b, (b / se if se > 0 else 0.0)


def half_life(x) -> float:
    """Half-life in days. Infinite if the series does not revert to its mean."""
    b, _ = _ar1(np.asarray(x, dtype=float))
    return float(-np.log(2.0) / np.log1p(b)) if -1.0 < b < 0.0 else float("inf")


def df_tstat(x) -> float:
    return _ar1(np.asarray(x, dtype=float))[1]


def ou_fit(x, lookback: int = 252) -> Tuple[float, float, float]:
    """Fits a discrete OU process on the end of the series. Returns (mean, kappa, sigma)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)][-lookback:]
    if len(x) < 30:
        return (float(np.nanmean(x)) if len(x) else 0.0, 0.05, float(np.nanstd(np.diff(x))) if len(x) > 2 else 1.0)
    b, _ = _ar1(x)
    kappa = float(np.clip(-b, 0.005, 0.6))
    resid = np.diff(x) + kappa * (x[:-1] - x.mean())
    return float(x.mean()), kappa, float(resid.std())


def simulate_ou(x0: float, mean: float, kappa: float, sigma: float, horizon: int,
                n_paths: int = 400, seed: int = 11) -> np.ndarray:
    """Monte Carlo paths of a discrete OU. Shape (n_paths, horizon + 1)."""
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, horizon + 1))
    paths[:, 0] = x0
    shocks = sigma * rng.standard_normal((n_paths, horizon))
    for t in range(horizon):
        paths[:, t + 1] = paths[:, t] + kappa * (mean - paths[:, t]) + shocks[:, t]
    return paths


# ----------------------------------------------------------------------------
# Pairs
# ----------------------------------------------------------------------------
def pair_history(spreads: pd.DataFrame, a: str, b: str, window: int = 126) -> pd.DataFrame:
    """Spread ratio A / B with its rolling mean, 2-sigma bands and z-score."""
    q = np.log(spreads[a] / spreads[b]).dropna()
    roll = q.rolling(window, min_periods=max(10, window // 3))
    mean, sd = roll.mean(), roll.std()
    return pd.DataFrame({"ratio": np.exp(q), "mean": np.exp(mean),
                         "upper": np.exp(mean + 2 * sd), "lower": np.exp(mean - 2 * sd),
                         "z": (q - mean) / sd.replace(0.0, np.nan),
                         "log_ratio": q, "log_mean": mean, "log_sd": sd})


def scan_pairs(spreads: pd.DataFrame, static: pd.DataFrame, window: int = 126,
               min_corr: float = 0.15) -> pd.DataFrame:
    """
    Sweeps every pair of issuers within the same universe and sector.

    The signal is the z-score of the log spread ratio. z > 0 means A is wide to
    B against their recent relationship (buy A, sell B).
    """
    rows = []
    ln = np.log(spreads)
    dln = ln.diff()
    groups = static.loc[[t for t in static.index if t in spreads.columns]].groupby(
        ["universe", "sector"], sort=True)
    for (universe, sector), grp in groups:
        for a, b in combinations(list(grp.index), 2):
            q = (ln[a] - ln[b]).dropna()
            if len(q) < 60:
                continue
            tail = q.iloc[-window:]
            sd = float(tail.std())
            if not np.isfinite(sd) or sd <= 0:
                continue
            z = float((q.iloc[-1] - tail.mean()) / sd)
            if z < 0:                        # orient the pair so that z is positive
                a, b, q, tail, z = b, a, -q, -tail, -z
            corr = float(dln[a].iloc[-window:].corr(dln[b].iloc[-window:]))
            if not np.isfinite(corr) or corr < min_corr:
                continue
            x = q.iloc[-2 * window:].to_numpy()
            sa, sb = float(spreads[a].iloc[-1]), float(spreads[b].iloc[-1])
            rows.append({
                "pair": f"{a} / {b}", "a": a, "b": b, "universe": universe, "sector": sector,
                "spread_a": sa, "spread_b": sb, "ratio": sa / sb,
                "mean_ratio": float(np.exp(tail.mean())), "z": z, "log_sd": sd,
                "half_life": half_life(x), "df_t": df_tstat(x), "corr": corr,
                # expected tightening of A (in bp) if the ratio returns to its mean
                "conv_bps": sa * (1.0 - float(np.exp(tail.mean() - q.iloc[-1]))),
            })
    out = pd.DataFrame(rows)
    return out.sort_values("z", ascending=False).reset_index(drop=True) if len(out) else out


# ----------------------------------------------------------------------------
# Sectors
# ----------------------------------------------------------------------------
def sector_relative(spreads: pd.DataFrame, static: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Average log spread of each sector, relative to the universe average."""
    st = static[(static["universe"] == universe) & static.index.isin(spreads.columns)]
    ln = np.log(spreads[st.index])
    by_sector = ln.T.groupby(st["sector"]).mean().T
    return by_sector.sub(ln.mean(axis=1), axis=0)


def sector_z_matrix(spreads: pd.DataFrame, static: pd.DataFrame, universe: str,
                    windows: Sequence[int] = (21, 63, 126, 252)) -> pd.DataFrame:
    """z-score of each sector's relative cheapness over several windows."""
    rel = sector_relative(spreads, static, universe)
    out = {}
    for w in windows:
        tail = rel.iloc[-w:]
        out[w] = (rel.iloc[-1] - tail.mean()) / tail.std().replace(0.0, np.nan)
    return pd.DataFrame(out)


def sector_pair_matrix(spreads: pd.DataFrame, static: pd.DataFrame, universe: str,
                       window: int = 126) -> pd.DataFrame:
    """Sector by sector matrix, z-score of the gap (row minus column)."""
    rel = sector_relative(spreads, static, universe).iloc[-window:]
    secs = list(rel.columns)
    m = pd.DataFrame(0.0, index=secs, columns=secs)
    for a in secs:
        for b in secs:
            if a != b:
                d = rel[a] - rel[b]
                sd = d.std()
                m.loc[a, b] = (d.iloc[-1] - d.mean()) / sd if sd > 0 else 0.0
    return m


def sector_levels(spreads: pd.DataFrame, static: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Today's average spread by sector and its 1-month and 3-month changes."""
    st = static[(static["universe"] == universe) & static.index.isin(spreads.columns)]
    avg = spreads[st.index].T.groupby(st["sector"]).mean().T
    return pd.DataFrame({"spread": avg.iloc[-1],
                         "chg_1m": avg.iloc[-1] - avg.iloc[-min(22, len(avg))],
                         "chg_3m": avg.iloc[-1] - avg.iloc[-min(64, len(avg))]})
