"""
Term structure, credit curves and seniority.

  1. Sovereign curve, with a Nelson-Siegel fit, slopes and butterflies
  2. Credit curves by rating bucket, spread against maturity, and how far
     each bond sits from its curve (rich or cheap)
  3. Carry and roll-down over a given horizon
  4. Seniority premium between Senior Secured, Senior Unsecured and Subordinated
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config as cfg
from engine.bondmath import nelson_siegel

SEN_RANK = {s: i for i, s in enumerate(cfg.SENIORITIES)}


# ----------------------------------------------------------------------------
# 1. Sovereign curve
# ----------------------------------------------------------------------------
def fit_nelson_siegel(tenors: Sequence[float], yields: Sequence[float]) -> Dict[str, float]:
    """Fits Nelson-Siegel with a sweep on lambda and least squares on the betas."""
    t = np.asarray(tenors, dtype=float)
    y = np.asarray(yields, dtype=float)
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    best: Optional[Dict[str, float]] = None
    for lam in np.linspace(0.6, 5.0, 23):
        x = t / lam
        f1 = (1 - np.exp(-x)) / x
        design = np.column_stack([np.ones_like(t), f1, f1 - np.exp(-x)])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        rmse = float(np.sqrt(np.mean((design @ beta - y) ** 2)))
        if best is None or rmse < best["rmse"]:
            best = {"b0": float(beta[0]), "b1": float(beta[1]), "b2": float(beta[2]),
                    "lam": float(lam), "rmse": rmse}
    return best or {"b0": float(np.nanmean(y)), "b1": 0.0, "b2": 0.0, "lam": 1.8, "rmse": 0.0}


def ns_curve(params: Dict[str, float], taus) -> np.ndarray:
    return nelson_siegel(taus, params["b0"], params["b1"], params["b2"], params["lam"])


def curve_snapshots(curve: pd.DataFrame,
                    offsets: Optional[Dict[str, int]] = None) -> Dict[str, pd.Series]:
    """Today's curve and past ones (1 month, 3 months, 1 year in business days)."""
    offsets = offsets or {"Today": 0, "1M ago": 21, "3M ago": 63, "1Y ago": 252}
    return {k: curve.iloc[-1 - min(v, len(curve) - 1)] for k, v in offsets.items()}


def slope_history(curve: pd.DataFrame) -> pd.DataFrame:
    """Slopes and butterfly in bp, 2s10s, 5s30s and 2s5s10s."""
    return pd.DataFrame({
        "2s10s": (curve[10.0] - curve[2.0]) * 100.0,
        "5s30s": (curve[30.0] - curve[5.0]) * 100.0,
        "2s5s10s fly": (2 * curve[5.0] - curve[2.0] - curve[10.0]) * 100.0,
    })


def slope_stats(slopes: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    tail = slopes.iloc[-window:]
    last = slopes.iloc[-1]
    return pd.DataFrame({
        "last": last,
        "chg_1m": last - slopes.iloc[-min(22, len(slopes))],
        "z": (last - tail.mean()) / tail.std().replace(0.0, np.nan),
        "pct_rank": (tail <= last).mean() * 100.0,
    })


# ----------------------------------------------------------------------------
# 2. Credit curves
# ----------------------------------------------------------------------------
def credit_curves(basket: pd.DataFrame) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
    """
    Fits spread = a + b ln(1 + maturity) by rating bucket, on senior unsecured
    bonds only, so that slope and seniority are not mixed.

    Returns the coefficients by bucket and the basket with three more columns.
      fit_oas    spread on the curve of its bucket
      curve_res  gap to the curve in bp (positive means the bond is cheap)
      curve_z    the gap scaled within the bucket
    """
    b = basket.copy()
    x_all = np.log1p(b["maturity_years"].to_numpy(dtype=float))
    senior = (b["seniority"] == "Sr Unsecured").to_numpy()
    pooled = np.polyfit(x_all[senior], b["oas"].to_numpy()[senior], 1) if senior.sum() >= 3 \
        else np.array([0.0, float(b["oas"].mean())])
    fits: Dict[str, Tuple[float, float]] = {}
    fit_oas = np.empty(len(b))
    for bucket, grp in b.groupby("bucket", sort=False):
        pos = b.index.get_indexer(grp.index)
        use = senior[pos]
        if use.sum() >= 5 and np.ptp(x_all[pos][use]) > 0.2:
            slope, inter = np.polyfit(x_all[pos][use], grp["oas"].to_numpy()[use], 1)
        else:                         # too few points, so common slope and bucket level
            slope = pooled[0]
            inter = float(grp["oas"].mean() - slope * x_all[pos].mean())
        fits[str(bucket)] = (float(inter), float(slope))
        fit_oas[pos] = inter + slope * x_all[pos]
    b["fit_oas"] = np.maximum(fit_oas, 1.0)
    b["curve_res"] = b["oas"] - b["fit_oas"]
    sd = b.groupby("bucket")["curve_res"].transform("std").replace(0.0, np.nan)
    b["curve_z"] = (b["curve_res"] / sd).fillna(0.0)
    return fits, b


def credit_curve_line(fit: Tuple[float, float], taus) -> np.ndarray:
    return fit[0] + fit[1] * np.log1p(np.asarray(taus, dtype=float))


# ----------------------------------------------------------------------------
# 3. Carry and roll-down
# ----------------------------------------------------------------------------
def carry_rolldown(basket: pd.DataFrame, govt: Dict[str, pd.Series],
                   fits: Dict[str, Tuple[float, float]], horizon: float = 0.25) -> pd.DataFrame:
    """
    Carry and roll-down over the horizon (in years), in bp of price.

      carry      yield earned over the period
      roll       price gain if the curves (rates and credit) do not move and
                 the bond rolls down its curve
      breakeven  spread widening that wipes out carry plus roll
    """
    tau = basket["maturity_years"].to_numpy(dtype=float)
    tau_h = np.maximum(tau - horizon, 0.05)
    roll_y = np.zeros(len(basket))
    for ccy, curve in govt.items():
        mask = (basket["currency"] == ccy).to_numpy()
        if mask.any():
            ten = np.asarray(curve.index, dtype=float)
            roll_y[mask] = (np.interp(tau[mask], ten, curve.to_numpy())
                            - np.interp(tau_h[mask], ten, curve.to_numpy())) * 100.0
    slope = basket["bucket"].map(lambda k: fits.get(str(k), (0.0, 0.0))[1]).to_numpy(dtype=float)
    roll_s = slope * (np.log1p(tau) - np.log1p(tau_h))
    dur = basket["mod_dur"].to_numpy(dtype=float)
    dur_h = np.maximum(dur - horizon, 0.05)
    out = pd.DataFrame(index=basket.index)
    out["carry_bps"] = basket["ytm"].to_numpy() * horizon * 100.0
    out["roll_bps"] = dur_h * (roll_y + roll_s)
    out["total_bps"] = out["carry_bps"] + out["roll_bps"]
    out["breakeven_bp"] = out["total_bps"] / dur_h
    return out


# ----------------------------------------------------------------------------
# 4. Seniority
# ----------------------------------------------------------------------------
def seniority_summary(basket: pd.DataFrame) -> pd.DataFrame:
    """Weight, spread, duration and DTS contribution by seniority tier."""
    rows = []
    for sen in cfg.SENIORITIES:
        g = basket[basket["seniority"] == sen]
        if g.empty:
            continue
        w = g["weight"]
        rows.append({"seniority": sen, "bonds": len(g), "weight": w.sum() * 100.0,
                     "bench_weight": g["bench_weight"].sum() * 100.0,
                     "oas": float((w * g["oas"]).sum() / w.sum()),
                     "ytm": float((w * g["ytm"]).sum() / w.sum()),
                     "mod_dur": float((w * g["mod_dur"]).sum() / w.sum()),
                     "dts_contrib": float((w * g["spread_dur"] * g["oas"]).sum())})
    return pd.DataFrame(rows).set_index("seniority")


def seniority_pairs(basket: pd.DataFrame, oas_hist: pd.DataFrame,
                    fits: Dict[str, Tuple[float, float]], window: int = 126) -> pd.DataFrame:
    """
    For each issuer present on several tiers, compares the most junior bond
    with the most senior one.

      premium_bp  raw spread gap, junior minus senior
      adj_ratio   spread ratio corrected for the maturity gap, using the
                  credit curve of the bucket
      z           z-score of the log ratio over `window` days (positive means
                  the junior bond is unusually wide, a compression trade)
    """
    rows = []
    for ticker, grp in basket.groupby("ticker", sort=False):
        ranks = grp["seniority"].map(SEN_RANK)
        if ranks.nunique() < 2:
            continue
        sr = grp.loc[ranks.idxmin()]
        jr = grp.loc[ranks.idxmax()]
        if sr["bond_id"] not in oas_hist or jr["bond_id"] not in oas_hist:
            continue
        q = np.log(oas_hist[jr["bond_id"]] / oas_hist[sr["bond_id"]]).dropna()
        if len(q) < 40:
            continue
        tail = q.iloc[-window:]
        sd = float(tail.std())
        fit = fits.get(str(sr["bucket"]), (float(sr["oas"]), 0.0))
        term_adj = (max(credit_curve_line(fit, jr["maturity_years"]), 1.0)
                    / max(credit_curve_line(fit, sr["maturity_years"]), 1.0))
        rows.append({
            "issuer": sr["issuer"], "ticker": ticker, "sector": sr["sector"],
            "rating": sr["rating"], "senior": sr["bond_id"], "junior": jr["bond_id"],
            "senior_tier": sr["seniority"], "junior_tier": jr["seniority"],
            "oas_senior": float(sr["oas"]), "oas_junior": float(jr["oas"]),
            "premium_bp": float(jr["oas"] - sr["oas"]),
            "ratio": float(jr["oas"] / sr["oas"]),
            "adj_ratio": float(jr["oas"] / sr["oas"] / term_adj),
            "mean_ratio": float(np.exp(tail.mean())),
            "z": float((q.iloc[-1] - tail.mean()) / sd) if sd > 0 else 0.0,
            "log_sd": sd,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("z", ascending=False).reset_index(drop=True) if len(out) else out


def seniority_premium_history(basket: pd.DataFrame, oas_hist: pd.DataFrame) -> pd.DataFrame:
    """
    History of the portfolio's seniority premium. It is the median junior over
    senior spread ratio, across issuers present on both tiers.
    """
    series: Dict[str, list] = {"Subordinated / Sr Unsecured": [], "Sr Unsecured / Sr Secured": []}
    for _, grp in basket.groupby("ticker", sort=False):
        by_tier = {s: g["bond_id"].iloc[0] for s, g in grp.groupby("seniority")}
        for label, (jr, sr) in {"Subordinated / Sr Unsecured": ("Subordinated", "Sr Unsecured"),
                                "Sr Unsecured / Sr Secured": ("Sr Unsecured", "Sr Secured")}.items():
            if jr in by_tier and sr in by_tier:
                series[label].append(np.log(oas_hist[by_tier[jr]] / oas_hist[by_tier[sr]]))
    out = {k: np.exp(pd.concat(v, axis=1).median(axis=1)) for k, v in series.items() if v}
    return pd.DataFrame(out)
