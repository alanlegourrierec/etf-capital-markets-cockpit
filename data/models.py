"""
Data containers passed between the data layer and the engine.

Whether the data comes from LSEG or from the simulator, it reaches the engine
in these same shapes. That contract is what makes the hybrid mode work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from config import EtfSpec


@dataclass
class RatesData:
    """Sovereign curves. curves[currency] is a DataFrame of dates by tenors, in %."""
    curves: Dict[str, pd.DataFrame]
    source: str = "SIM"


@dataclass
class IssuerData:
    """
    Issuer panel for the capital structure module.

    static    index is the short ticker. Columns name, equity_ric, sector, rating,
              bucket, currency, universe, mktcap_musd, debt_musd, st_debt_musd,
              lt_debt_musd
    equity    dates by ticker, share price
    spread5y  dates by ticker, 5Y credit spread in bp
    """
    static: pd.DataFrame
    equity: pd.DataFrame
    spread5y: pd.DataFrame
    source: str = "SIM"


@dataclass
class EtfData:
    """
    Everything known about one ETF.

    daily     indexed by date. nav, price, bid, ask, volume, value_musd,
              shares_out, flow_cu, aum_musd, bench, basket_ba_bps, etf_ba_bps,
              basket_turnover_musd
    intraday  indexed by timestamp. inav, price, bid, ask, volume (None if missing)
    basket    one row per bond. bond_id, issuer, ticker, sector, rating, bucket,
              seniority, currency, coupon, maturity_years, price, ytm, oas,
              mod_dur, spread_dur, convexity, amt_out_musd, age_years, adv_musd,
              ba_bps, weight, bench_weight
    oas_hist  dates by bond_id, spread in bp
    source    where each block comes from ("LSEG" or "SIM")
    """
    spec: EtfSpec
    daily: pd.DataFrame
    intraday: Optional[pd.DataFrame]
    basket: pd.DataFrame
    oas_hist: pd.DataFrame
    source: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
