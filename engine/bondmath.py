"""
Basic bond maths, vectorised with NumPy.

Rates and coupons are decimals (0.05 is 5%), prices are per 100 of face value
and maturities are in years. Bonds are fixed-rate bullets, which is enough for
an IG and HY credit prototype.
"""
from __future__ import annotations

import numpy as np


def bond_price(ytm, coupon, ttm, freq: int = 2):
    """Price of a fixed-rate bullet bond, closed-form discounting."""
    ytm = np.asarray(ytm, dtype=float)
    coupon = np.asarray(coupon, dtype=float)
    ttm = np.maximum(np.asarray(ttm, dtype=float), 1e-6)
    y = ytm / freq
    n = ttm * freq
    disc = (1.0 + y) ** (-n)
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(np.abs(y) > 1e-10, (1.0 - disc) / y, n)
    return 100.0 * (coupon / freq * annuity + disc)


def risk_measures(ytm, coupon, ttm, freq: int = 2, bump: float = 1e-4):
    """Returns (price, modified duration, convexity) by finite differences."""
    p0 = bond_price(ytm, coupon, ttm, freq)
    p_up = bond_price(np.asarray(ytm, float) + bump, coupon, ttm, freq)
    p_dn = bond_price(np.asarray(ytm, float) - bump, coupon, ttm, freq)
    mod_dur = (p_dn - p_up) / (2.0 * p0 * bump)
    convexity = (p_up + p_dn - 2.0 * p0) / (p0 * bump ** 2)
    return p0, mod_dur, convexity


def ytm_from_price(price, coupon, ttm, freq: int = 2, n_iter: int = 40):
    """Yield to maturity by Newton. Used when LSEG only gives the price."""
    price = np.asarray(price, dtype=float)
    coupon = np.asarray(coupon, dtype=float)
    ytm = np.where(np.isfinite(coupon), coupon, 0.05) + 0.0 * price
    for _ in range(n_iter):
        p, dur, _ = risk_measures(ytm, coupon, ttm, freq)
        step = (p - price) / np.maximum(p * dur, 1e-9)
        ytm = np.clip(ytm + step, -0.02, 1.0)
        if np.nanmax(np.abs(step)) < 1e-10:
            break
    return ytm


def interp_weights(tenors, ttm) -> np.ndarray:
    """
    Matrix (n_bonds x n_tenors) of linear interpolation weights.

    y_bond = W @ curve. The same matrix spreads duration over the key tenors
    (key rate durations), which is why it is exposed.
    """
    t = np.asarray(tenors, dtype=float)
    x = np.clip(np.asarray(ttm, dtype=float), t[0], t[-1])
    hi = np.clip(np.searchsorted(t, x, side="left"), 1, len(t) - 1)
    lo = hi - 1
    w_hi = (x - t[lo]) / (t[hi] - t[lo])
    w = np.zeros((len(x), len(t)))
    rows = np.arange(len(x))
    w[rows, lo] = 1.0 - w_hi
    w[rows, hi] += w_hi
    return w


def key_rate_durations(mod_dur, ttm, key_tenors) -> np.ndarray:
    """Spreads each bond's duration over the two key tenors around it."""
    return np.asarray(mod_dur, float)[:, None] * interp_weights(key_tenors, ttm)


def nelson_siegel(tau, b0, b1, b2, lam):
    """Nelson-Siegel curve. The betas can be vectors, one date per row."""
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-6)
    x = tau / lam
    f1 = (1.0 - np.exp(-x)) / x
    f2 = f1 - np.exp(-x)
    return b0 + b1 * f1 + b2 * f2
