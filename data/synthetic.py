"""
Stochastic generators, the offline fallback.

The simulator builds one consistent market from a single seed, so that every
module tells the same story.

  common factors  calm or stress regime (Markov chain), equity factor,
                  credit factor, sector factors
  rates           USD and EUR curves, Nelson-Siegel with three OU factors
  issuers         equity (factor, own shock, jumps) and a 5Y spread tied to
                  the equity the Merton way, plus a mean-reverting noise that
                  creates the relative value opportunities
  bonds           spread = issuer x curve slope x seniority premium
  ETF             NAV from the basket, premium or discount with an arbitrage
                  that closes the gaps, volumes, creations and redemptions

Nothing here is market data. Levels are plausible, not real.
"""
from __future__ import annotations

import zlib
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

import config as cfg
from data.models import EtfData, IssuerData, RatesData
from engine.bondmath import bond_price, nelson_siegel, risk_measures

SQRT252 = np.sqrt(252.0)

# Central parameters of the sovereign curves (Nelson-Siegel, in %)
NS_PARAMS = {
    "USD": dict(b0=4.35, b1=-0.45, b2=-0.60, lam=1.8),
    "EUR": dict(b0=2.95, b1=-0.80, b2=-0.50, lam=1.8),
}
BASE_BA_BPS = {"AA+": 12.0, "A": 15.0, "BBB": 20.0, "BB": 38.0, "B": 55.0, "CCC": 90.0}


def _ou(shocks: np.ndarray, kappa: float) -> np.ndarray:
    """Discrete mean-reverting process, x_t = (1 - kappa) x_{t-1} + shock_t."""
    out = np.empty_like(shocks, dtype=float)
    x = np.zeros(shocks.shape[1:])
    for t in range(shocks.shape[0]):
        x = (1.0 - kappa) * x + shocks[t]
        out[t] = x
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 0 else 1.0)


class SyntheticMarket:
    """A full simulated market. One instance is one reproducible scenario."""

    def __init__(self, seed: int = cfg.SEED, n_days: int = cfg.HISTORY_DAYS,
                 end: Optional[pd.Timestamp] = None):
        self.seed = int(seed)
        self.n = int(n_days)
        end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
        self.dates = pd.bdate_range(end=end, periods=self.n)
        self._ns: Dict[str, tuple] = {}
        self._etfs: Dict[str, EtfData] = {}
        self._build_factors()
        self.rates = self._build_rates()
        self.issuers = self._build_issuers()

    # ------------------------------------------------------------------ utils
    def _rng(self, name: str) -> np.random.Generator:
        """One generator per component, so the order of calls does not change the draws."""
        return np.random.default_rng([self.seed, zlib.crc32(name.encode())])

    # ---------------------------------------------------------------- factors
    def _build_factors(self) -> None:
        rng = self._rng("factors")
        n = self.n
        # Stress regime, a smoothed two-state Markov chain
        state, raw = 0, np.zeros(n)
        u = rng.random(n)
        for t in range(1, n):
            if state == 0 and u[t] < 0.010:
                state = 1
            elif state == 1 and u[t] < 0.060:
                state = 0
            raw[t] = state
        self.stress = pd.Series(raw).ewm(alpha=0.2).mean().to_numpy()
        self.vol_mult = 1.0 + 1.7 * self.stress

        # Market equity factor
        self.r_mkt = (0.09 / 252 - 0.0016 * self.stress
                      + 0.135 / SQRT252 * self.vol_mult * rng.standard_normal(n))

        # Market credit factor, the log gap to the central level
        eps = rng.standard_normal(n)
        m = np.zeros(n)
        for t in range(1, n):
            m[t] = (0.988 * m[t - 1] - 1.7 * (self.r_mkt[t] - 0.09 / 252)
                    + 0.008 * self.vol_mult[t] * eps[t])
        self.m_credit = m

        # Sector factors, equity then credit (credit follows the sector's equity)
        betas = np.array([1.15, 1.20, 1.05, 1.00, 0.75, 1.05, 1.15, 0.60])
        k = len(cfg.SECTORS)
        sec_idio = (0.075 / SQRT252 * np.sqrt(self.vol_mult)[:, None]
                    * rng.standard_normal((n, k)))
        self.r_sector = self.r_mkt[:, None] * betas + sec_idio
        self.c_sector = _ou(-1.1 * sec_idio + 0.0045 * rng.standard_normal((n, k)), 0.02)

    # ------------------------------------------------------------------ rates
    def _build_rates(self) -> RatesData:
        rng = self._rng("rates")
        n = self.n
        zm = _zscore(self.r_mkt)
        common = rng.standard_normal((n, 3))
        tenors = np.array(cfg.CURVE_TENORS)
        curves = {}
        for ccy, p in NS_PARAMS.items():
            own = rng.standard_normal((n, 3))
            mix = 1.0 if ccy == "USD" else 0.7
            e = mix * common + np.sqrt(1.0 - mix ** 2) * own
            # The level of rates falls when equities sell off
            e[:, 0] = 0.25 * zm + np.sqrt(1.0 - 0.25 ** 2) * e[:, 0]
            sig = np.array([0.050, 0.042, 0.060]) * (1.0 if ccy == "USD" else 0.8)
            f = _ou(e * sig, 0.004)
            b0 = p["b0"] + f[:, 0]
            b1 = p["b1"] + f[:, 1] - 0.35 * self.stress      # steepening under stress
            b2 = p["b2"] + f[:, 2]
            self._ns[ccy] = (b0, b1, b2, p["lam"])
            y = nelson_siegel(tenors[None, :], b0[:, None], b1[:, None], b2[:, None], p["lam"])
            curves[ccy] = pd.DataFrame(np.maximum(y, 0.05), index=self.dates,
                                       columns=cfg.CURVE_TENORS)
        return RatesData(curves=curves, source="SIM")

    # ---------------------------------------------------------------- issuers
    def _build_issuers(self) -> IssuerData:
        rng = self._rng("issuers")
        n = self.n
        rows = [(r, "HY") for r in cfg.HY_ISSUERS] + [(r, "IG") for r in cfg.IG_ISSUERS]
        k = len(rows)
        tickers = [r[0][1] for r in rows]
        ratings = [r[0][4] for r in rows]
        sec_idx = np.array([cfg.SECTORS.index(r[0][3]) for r in rows])
        is_ig = np.array([u == "IG" for _, u in rows])
        score = np.array([cfg.RATING_SCORE[x] for x in ratings], dtype=float)

        # --- Equity, with a sector factor, an own shock and rare jumps
        target_vol = (np.array([cfg.RATING_EQUITY_VOL[x] for x in ratings])
                      * rng.uniform(0.85, 1.10, k))
        beta = rng.uniform(0.8, 1.3, k)
        sec_vol = self.r_sector.std(axis=0)[sec_idx] * SQRT252
        idio_vol = np.sqrt(np.maximum(target_vol ** 2 - (beta * sec_vol) ** 2, 0.12 ** 2))
        e = (idio_vol / SQRT252 * np.sqrt(self.vol_mult)[:, None]
             * rng.standard_normal((n, k)))
        e += (rng.random((n, k)) < 3.0 / 252) * rng.normal(0.0, 0.055, (n, k))
        r = np.clip(self.r_sector[:, sec_idx] * beta + e, -0.35, 0.35)
        equity = rng.uniform(15.0, 250.0, k) * np.cumprod(1.0 + r, axis=0)

        # --- 5Y spread, with rating level, common factors, equity link and OU noise
        theta = (np.array([cfg.RATING_SPREAD_5Y[x] for x in ratings])
                 * np.exp(0.16 * rng.standard_normal(k)))
        load_mkt = 0.70 + 0.040 * score
        gamma = 0.45 + 0.030 * score              # sensitivity of the spread to the equity
        e_leaky = _ou(e, 0.003)                   # long memory of equity shocks
        sig_u = np.where(score >= 11, 0.017, 0.013)
        u = _ou(sig_u * rng.standard_normal((n, k)), 0.035)   # the "mispricing"
        ln_s = (np.log(theta) + load_mkt * self.m_credit[:, None]
                + self.c_sector[:, sec_idx] - gamma * e_leaky + u)

        # --- Balance sheet, leverage of the rating adjusted for the equity's own run
        lev = np.clip(np.array([cfg.RATING_LEVERAGE[x] for x in ratings])
                      * np.exp(0.10 * rng.standard_normal(k)), 0.04, 0.85)
        e_neutral = np.where(is_ig, np.exp(rng.normal(np.log(90e3), 0.7, k)),
                             np.exp(rng.normal(np.log(9e3), 0.6, k)))
        debt = e_neutral * lev / (1.0 - lev)
        mktcap = e_neutral * np.exp(e_leaky[-1])

        static = pd.DataFrame({
            "name": [r[0][0] for r in rows],
            "equity_ric": [r[0][2] for r in rows],
            "sector": [r[0][3] for r in rows],
            "rating": ratings,
            "bucket": [cfg.rating_bucket(x) for x in ratings],
            "currency": [r[0][5] for r in rows],
            "universe": [u for _, u in rows],
            "mktcap_musd": mktcap,
            "debt_musd": debt,
            "st_debt_musd": 0.25 * debt,
            "lt_debt_musd": 0.75 * debt,
        }, index=pd.Index(tickers, name="ticker"))

        self._iss_pos = {t: i for i, t in enumerate(tickers)}
        self._ln_s5 = ln_s
        return IssuerData(
            static=static,
            equity=pd.DataFrame(equity, index=self.dates, columns=tickers),
            spread5y=pd.DataFrame(np.exp(ln_s), index=self.dates, columns=tickers),
            source="SIM",
        )

    # -------------------------------------------------------------------- ETF
    def etf(self, etf_id: str) -> EtfData:
        if etf_id not in self._etfs:
            self._etfs[etf_id] = self._build_etf(cfg.ETFS[etf_id])
        return self._etfs[etf_id]

    def _build_basket_static(self, spec: cfg.EtfSpec, rng) -> pd.DataFrame:
        """Draws the basket composition, with issuer, maturity and seniority."""
        sp = spec.sim
        rows = cfg.issuer_rows(sp.issuers)
        ni = len(rows)
        counts = np.full(ni, sp.n_bonds // ni)
        counts[rng.choice(ni, sp.n_bonds - counts.sum(), replace=False)] += 1
        sen_p = np.array([0.30, 0.62, 0.08] if sp.issuers == "HY" else [0.04, 0.80, 0.16])
        lo, hi = sp.maturity_range
        recs = []
        for i, (name, tkr, _ric, sector, rating, ccy) in enumerate(rows):
            used = []
            edges = np.linspace(lo, hi, counts[i] + 1)
            for j in range(counts[i]):
                p = sen_p.copy()
                if used and rng.random() < 0.5:     # favours a different tier for the second bond
                    for s in used:
                        p[cfg.SENIORITIES.index(s)] = 0.0
                sen = str(rng.choice(cfg.SENIORITIES, p=p / p.sum()))
                used.append(sen)
                recs.append(dict(issuer=name, ticker=tkr, sector=sector, rating=rating,
                                 bucket=cfg.rating_bucket(rating), currency=ccy,
                                 seniority=sen,
                                 maturity_years=float(rng.uniform(edges[j], edges[j + 1]))))
        return pd.DataFrame.from_records(recs)

    def _build_etf(self, spec: cfg.EtfSpec) -> EtfData:
        sp = spec.sim
        rng = self._rng("etf:" + spec.etf_id)
        n = self.n
        hy = sp.issuers == "HY"

        b = self._build_basket_static(spec, rng)
        nb = len(b)
        iss_col = np.array([self._iss_pos[t] for t in b["ticker"]])
        score = b["rating"].map(cfg.RATING_SCORE).to_numpy(dtype=float)
        tau_T = b["maturity_years"].to_numpy()
        age = rng.uniform(0.3, 6.0, nb)
        amt = np.exp(rng.normal(np.log(650.0 if hy else 1100.0), 0.55, nb))
        amt = np.maximum(np.round(amt / 50.0) * 50.0, 300.0)

        # ---- 1. Bond spreads, as issuer x curve x seniority x noise
        # The history is built at constant maturity. The manager keeps the fund's
        # maturity profile, so the bonds are not aged.
        alpha = ((0.12 if hy else 0.28) - 0.10 * self.stress
                 + _ou(0.006 * rng.standard_normal((n, 1)), 0.02)[:, 0])
        term = (np.maximum(tau_T, 0.25)[None, :] / 5.0) ** alpha[:, None]
        sen_base = {"Sr Secured": 0.72, "Sr Unsecured": 1.00,
                    "Subordinated": 1.38 if hy else 1.55}
        sen_stress = {"Sr Secured": -0.06, "Sr Unsecured": 0.0, "Subordinated": 0.12}
        ln_base = np.log(b["seniority"].map(sen_base).to_numpy(dtype=float))
        st_load = b["seniority"].map(sen_stress).to_numpy(dtype=float)
        not_senior = (b["seniority"] != "Sr Unsecured").to_numpy()
        w_sen = _ou(0.012 * rng.standard_normal((n, nb)), 0.03) * not_senior
        v = _ou(0.008 * rng.standard_normal((n, nb)), 0.04)
        oas = np.exp(self._ln_s5[:, iss_col] + ln_base[None, :]
                     + st_load[None, :] * self.stress[:, None] + w_sen + v) * term

        # ---- 2. Yields, coupons, prices, daily performance
        ytm = np.empty((n, nb))
        for ccy in NS_PARAMS:
            mask = (b["currency"] == ccy).to_numpy()
            if mask.any():
                b0, b1, b2, lam = self._ns[ccy]
                g = nelson_siegel(tau_T[mask][None, :], b0[:, None], b1[:, None],
                                  b2[:, None], lam)
                ytm[:, mask] = np.maximum(g, 0.05) + oas[:, mask] / 100.0
        coupon = np.round(np.clip(ytm.mean(axis=0) + rng.normal(0.0, 0.6, nb), 0.5, 13.0) * 8) / 8
        price = bond_price(ytm / 100.0, coupon[None, :] / 100.0, tau_T[None, :])
        ret = np.zeros((n, nb))                      # carry plus the price move of the day
        ret[1:] = (price[1:] - price[:-1]) / price[:-1] + ytm[:-1] / 100.0 / 252.0
        p_T, mod_dur, convexity = risk_measures(ytm[-1] / 100.0, coupon / 100.0, tau_T)

        # ---- 3. Liquidity, daily volume and bid/ask per bond
        latent = (0.55 * np.log(amt / amt.mean()) - 0.22 * (age - age.mean())
                  - 0.05 * (score - score.mean()) + 0.45 * rng.standard_normal(nb))
        z_liq = _zscore(latent)
        adv = (3.2 if hy else 6.0) * np.exp(0.75 * z_liq)
        ba = (b["bucket"].map(BASE_BA_BPS).to_numpy(dtype=float)
              * (0.70 + 0.06 * mod_dur) * np.exp(-0.30 * z_liq))

        # ---- 4. Benchmark weights (amount outstanding) and fund weights (active tilt)
        bench_w = np.minimum(amt / amt.sum(), 0.03)
        bench_w /= bench_w.sum()
        central = b["rating"].map(cfg.RATING_SPREAD_5Y).to_numpy(dtype=float)
        carry_z = _zscore(np.log(oas[-1]) - np.log(central))
        w = bench_w * np.exp(sp.active_tilt * (0.6 * carry_z + 0.8 * rng.standard_normal(nb)))
        w /= w.sum()
        cash_w = 0.015 if spec.active else 0.004

        today = self.dates[-1]
        mat_dates = [today + timedelta(days=int(round(t * 365.25))) for t in tau_T]
        ids, seen = [], {}
        for tkr, c, d in zip(b["ticker"], coupon, mat_dates):
            base = f"{tkr} {c:.3f} {d.year}"
            seen[base] = seen.get(base, 0) + 1
            ids.append(base if seen[base] == 1 else f"{base} #{seen[base]}")
        basket = b.assign(
            bond_id=ids, maturity_date=mat_dates, coupon=coupon, price=p_T, ytm=ytm[-1],
            oas=oas[-1], mod_dur=mod_dur, spread_dur=mod_dur, convexity=convexity,
            amt_out_musd=amt, age_years=age, adv_musd=adv, ba_bps=ba,
            weight=w * (1.0 - cash_w), bench_weight=bench_w,
        ).set_index("bond_id", drop=False)
        basket.index.name = None

        # ---- 5. Fund NAV and benchmark index
        short_rate = self.rates.curves[spec.currency][0.25].to_numpy() / 100.0
        r_bench = ret @ bench_w
        r_fund = ((1.0 - cash_w) * (ret @ w) + cash_w * short_rate / 252.0
                  + (sp.alpha_bps - spec.ter_bps) / 1e4 / 252.0
                  + (2.0e-4 if spec.active else 0.5e-4) * rng.standard_normal(n))
        r_fund[0] = r_bench[0] = 0.0
        growth = np.cumprod(1.0 + r_fund)
        nav = sp.nav_last * growth / growth[-1]
        bench = 100.0 * np.cumprod(1.0 + r_bench)

        # ---- 6. Basket bid/ask through time (it widens under stress)
        ba_T = float(w @ ba)
        mult = 1.0 + 1.5 * self.stress + 1.2 * np.maximum(self.m_credit, 0.0)
        basket_ba = ba_T * mult / mult[-1] * np.exp(0.04 * rng.standard_normal(n))
        basket_ba[-1] = ba_T

        # ---- 7. Premium or discount, with a mean tied to the bid/ask, shocks, arbitrage
        # The premium moves around a positive structural level, since the NAV is
        # struck at the bid. It is capped by the cost of a creation and floored by
        # the cost of a redemption. Beyond those, APs step in and close the gap.
        mu = (sp.pd_mean_bps * np.sqrt(basket_ba / basket_ba.mean())
              - 1.2 * ba_T * self.stress)          # the NAV lags when the market sells off
        cap_up = basket_ba + 0.9 * sp.etf_spread_bps + 4.0
        cap_dn = -(0.9 * sp.etf_spread_bps * (1.0 + 1.2 * self.stress) + 6.0)
        flow_press = _ou(0.30 * sp.pd_vol_bps * rng.standard_normal((n, 1)), 0.10)[:, 0]
        eps = 0.55 * sp.pd_vol_bps * rng.standard_normal(n)
        stale = 0.30 if hy else 0.15            # share of the move the NAV has not seen yet
        x = np.zeros(n)
        for t in range(1, n):
            xt = 0.85 * x[t - 1] + stale * r_fund[t] * 1e4 + 0.5 * flow_press[t] + eps[t]
            level = mu[t] + xt
            if level > cap_up[t]:
                xt = cap_up[t] + 0.4 * (level - cap_up[t]) - mu[t]
            elif level < cap_dn[t]:
                xt = cap_dn[t] + 0.4 * (level - cap_dn[t]) - mu[t]
            x[t] = xt
        band = 0.5 * (cap_up - cap_dn)
        pd_bps = mu + x
        px = nav * (1.0 + pd_bps / 1e4)
        half = (sp.etf_spread_bps / 2.0 * (1.0 + 1.2 * self.stress)
                * np.exp(0.12 * rng.standard_normal(n)))

        # ---- 8. Secondary volumes and primary flows
        sv = 0.55
        value = (sp.adv_musd * np.exp(sv * rng.standard_normal(n) - 0.5 * sv ** 2)
                 * (1.0 + 2.2 * self.stress) * (1.0 + 0.15 * np.abs(x) / band))
        press = x / band
        creations = rng.poisson(0.10 + 0.9 * np.maximum(press, 0.0) ** 1.5) * rng.integers(1, 4, n)
        redemptions = rng.poisson(0.07 + 0.9 * np.maximum(-press, 0.0) ** 1.5) * rng.integers(1, 4, n)
        flow_cu = creations - redemptions
        cum = np.cumsum(flow_cu) * spec.cu_shares
        shares = np.maximum(sp.aum_musd * 1e6 / nav[-1] - (cum[-1] - cum), 20 * spec.cu_shares)
        turnover = adv.sum() * np.exp(0.22 * rng.standard_normal(n)) * (1.0 - 0.30 * self.stress)

        daily = pd.DataFrame({
            "nav": nav, "price": px, "bid": px * (1.0 - half / 1e4),
            "ask": px * (1.0 + half / 1e4), "volume": value * 1e6 / px,
            "value_musd": value, "shares_out": shares, "flow_cu": flow_cu,
            "aum_musd": shares * nav / 1e6, "bench": bench,
            "basket_ba_bps": basket_ba, "etf_ba_bps": 2.0 * half,
            "basket_turnover_musd": turnover,
        }, index=self.dates)

        intraday = self._build_intraday(daily, sp, rng)
        return EtfData(
            spec=spec, daily=daily, intraday=intraday, basket=basket,
            oas_hist=pd.DataFrame(oas, index=self.dates, columns=ids),
            source={"etf": "SIM", "intraday": "SIM", "basket": "SIM"},
        )

    def _build_intraday(self, daily: pd.DataFrame, sp: cfg.SimProfile, rng) -> pd.DataFrame:
        """Last session in one-minute bars, with iNAV, price, bid/ask and volume."""
        day = daily.index[-1]
        idx = pd.date_range(day + timedelta(hours=8),
                            day + timedelta(hours=16, minutes=30), freq="1min")
        k = len(idx)
        ramp = np.linspace(0.0, 1.0, k)
        nav_prev, nav_T = daily["nav"].iloc[-2], daily["nav"].iloc[-1]
        pd_hist = (daily["price"] / daily["nav"] - 1.0) * 1e4
        sig_d = float(daily["nav"].pct_change(fill_method=None).std())

        # iNAV, a Brownian bridge between yesterday's NAV and today's
        path = np.cumsum(sig_d / np.sqrt(k) * rng.standard_normal(k))
        inav = nav_prev * (1.0 + ramp * (nav_T / nav_prev - 1.0) + path - ramp * path[-1])
        # Intraday premium, fast mean reversion anchored on the two closes
        ou = _ou(0.22 * sp.pd_vol_bps * rng.standard_normal((k, 1)), 0.03)[:, 0]
        prem = pd_hist.iloc[-2] + ramp * (pd_hist.iloc[-1] - pd_hist.iloc[-2]) + ou - ramp * ou[-1]
        px = inav * (1.0 + prem / 1e4)
        half = daily["etf_ba_bps"].iloc[-1] / 2.0 * np.exp(0.10 * rng.standard_normal(k))
        shape = 1.6 - 1.2 * np.sin(np.pi * ramp)          # U-shaped volume through the session
        vol = daily["volume"].iloc[-1] * shape / shape.sum() * np.exp(0.5 * rng.standard_normal(k))
        return pd.DataFrame({
            "inav": inav, "price": px, "bid": px * (1.0 - half / 1e4),
            "ask": px * (1.0 + half / 1e4), "volume": vol,
        }, index=idx)
