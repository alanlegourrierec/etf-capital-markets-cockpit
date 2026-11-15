"""
Single entry point of the engine for the interface.

The Analytics class links the data provider to the calculation modules and
caches the expensive results (Merton panel, pair scan, historical returns per
bond). Dash callbacks only call its methods.
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, Tuple

import pandas as pd

import config as cfg
from data.models import EtfData, IssuerData, RatesData
from engine import curves, etf_analytics as ea, merton, relative_value as rv, risk


class Analytics:
    def __init__(self, provider):
        self.p = provider
        self._cache: Dict[str, object] = {}
        self._lock = threading.RLock()

    def _memo(self, key: str, fn: Callable):
        with self._lock:
            if key not in self._cache:
                self._cache[key] = fn()
            return self._cache[key]

    # ------------------------------------------------------------------- data
    def etf(self, etf_id: str) -> EtfData:
        return self.p.etf(etf_id)

    def rates(self) -> RatesData:
        return self.p.rates()

    def issuers(self) -> IssuerData:
        return self.p.issuers()

    @staticmethod
    def universe_of(etf_id: str) -> str:
        return cfg.ETFS[etf_id].sim.issuers

    def govt_curve(self, etf_id: str) -> pd.DataFrame:
        """Sovereign curve in the ETF's currency."""
        curves_ = self.rates().curves
        return curves_.get(cfg.ETFS[etf_id].currency, curves_["USD"])

    # -------------------------------------------------- capital structure, RV
    def capital_structure(self) -> Dict[str, object]:
        def build():
            iss = self.issuers()
            rf = self.rates().curves["USD"][1.0]
            panel = merton.merton_panel(iss.static, iss.equity, rf)
            excluded = list(iss.static.index[iss.static["sector"].isin(cfg.MERTON_EXCLUDE_SECTORS)])
            fv = merton.fair_value(panel["dd"], iss.spread5y, exclude=excluded)
            snap = merton.snapshot(iss.static, panel, fv, iss.spread5y, iss.equity)
            snap["excluded"] = snap.index.isin(excluded)
            return {"panel": panel, "fv": fv, "snap": snap}
        return self._memo("capstruct", build)

    def pairs(self) -> pd.DataFrame:
        return self._memo("pairs", lambda: rv.scan_pairs(self.issuers().spread5y,
                                                         self.issuers().static))

    # ----------------------------------------------------------------- basket
    def basket(self, etf_id: str) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
        """Enriched basket (liquidity, credit curve, carry) and the curve coefficients."""
        def build():
            data = self.etf(etf_id)
            fits, b = curves.credit_curves(ea.enrich_basket(data.basket))
            govt = {ccy: c.iloc[-1] for ccy, c in self.rates().curves.items()}
            b = b.join(curves.carry_rolldown(b, govt, fits))
            return b, fits
        return self._memo(f"basket:{etf_id}", build)

    def seniority(self, etf_id: str) -> pd.DataFrame:
        def build():
            b, fits = self.basket(etf_id)
            return curves.seniority_pairs(b, self.etf(etf_id).oas_hist, fits)
        return self._memo(f"seniority:{etf_id}", build)

    # ------------------------------------------------------------------- risk
    def bond_parts(self, etf_id: str) -> Dict[str, pd.DataFrame]:
        def build():
            data = self.etf(etf_id)
            return risk.bond_returns(data.basket, self.rates(), data.oas_hist)
        return self._memo(f"parts:{etf_id}", build)

    def var(self, etf_id: str, level: float = 0.99, horizon: int = 1) -> Dict[str, float]:
        return self._memo(f"var:{etf_id}:{level}:{horizon}",
                          lambda: risk.var_summary(self.etf(etf_id).basket,
                                                   self.bond_parts(etf_id), level, horizon))

    def pnl(self, etf_id: str) -> pd.DataFrame:
        return self._memo(f"pnl:{etf_id}",
                          lambda: risk.portfolio_pnl(self.etf(etf_id).basket,
                                                     self.bond_parts(etf_id)))

    # ------------------------------------------------------------ trade ideas
    def ideas(self) -> pd.DataFrame:
        from engine.trade_ideas import build_trade_ideas      # late import (cycle)
        return self._memo("ideas", lambda: build_trade_ideas(self))
