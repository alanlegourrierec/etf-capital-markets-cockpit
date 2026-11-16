"""
Central configuration of the dashboard.

Everything that can change without touching the engine lives here.
  - data mode (auto, lseg or sim) and the LSEG app key
  - ETF universe (RICs, benchmark, fees, creation unit)
  - issuer universe (equity RIC, sector, indicative rating)
  - sovereign curve RICs and candidate LSEG fields
  - default Authorized Participant cost assumptions

Two things to keep in mind.
* RICs and LSEG field names marked "to check" could not be tested away from a
  Workspace desktop. Run `python scripts/lseg_check.py` with Workspace open to
  see which ones answer, then fix them here.
* Ratings, spread levels and amounts outstanding only calibrate the simulator.
  They are not market data.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------
# 1. Application
# ----------------------------------------------------------------------------
APP_TITLE = "ETF Capital Markets Cockpit"
APP_SUBTITLE = "Fixed Income ETFs · Credit relative value · AP arbitrage · Risk"
HOST = os.getenv("ETF_DASH_HOST", "127.0.0.1")
PORT = int(os.getenv("ETF_DASH_PORT", "8050"))

# "auto"  tries LSEG and falls back to the simulator block by block
# "lseg"  same, and prints every block that fell back
# "sim"   does not even try to open an LSEG session (offline demo)
DATA_MODE = os.getenv("ETF_DASH_DATA_MODE", "auto").lower()

def _read_app_key() -> str:
    """
    App key from the App Key Generator in Workspace.

    Read from the LSEG_APP_KEY environment variable (or EIKON_APP_KEY), otherwise
    from a .lseg_app_key file at the root of the project. That file is in
    .gitignore, so the key never reaches a repository.
    """
    key = os.getenv("LSEG_APP_KEY", os.getenv("EIKON_APP_KEY", "")).strip()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lseg_app_key")
    if not key and os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            key = fh.read().strip()
    return key


LSEG_APP_KEY = _read_app_key()

SEED = 7                 # simulator seed, so results can be reproduced
HISTORY_DAYS = 756       # about three years of business days
LIVE_REFRESH_MS = 5000   # refresh pace of the intraday panel

# Last day of the simulated history, as YYYY-MM-DD. Empty means today. The web
# build pins it so the precomputed pages and the engine in the browser agree.
SIM_ASOF = os.getenv("ETF_DASH_ASOF", "").strip() or None

# ----------------------------------------------------------------------------
# 2. Rating scale and simulator calibration
# ----------------------------------------------------------------------------
RATINGS: List[str] = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+",
]
RATING_SCORE: Dict[str, int] = {r: i + 1 for i, r in enumerate(RATINGS)}

# Central 5Y spread by rating, in basis points (simulator calibration)
RATING_SPREAD_5Y: Dict[str, float] = dict(zip(RATINGS, [
    32, 38, 44, 52, 60, 70, 82, 98, 118, 150,
    190, 235, 290, 360, 440, 560, 780,
]))
# Typical debt to asset value and equity volatility by rating
RATING_LEVERAGE: Dict[str, float] = dict(zip(RATINGS, [
    0.08, 0.10, 0.12, 0.14, 0.17, 0.20, 0.23, 0.27, 0.31, 0.36,
    0.41, 0.46, 0.51, 0.56, 0.61, 0.66, 0.74,
]))
RATING_EQUITY_VOL: Dict[str, float] = dict(zip(RATINGS, [
    0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.25, 0.27, 0.29, 0.32,
    0.35, 0.38, 0.42, 0.46, 0.50, 0.56, 0.65,
]))


def rating_bucket(rating: str) -> str:
    """Folds a fine rating into a readable bucket (AA+, A, BBB, BB, B, CCC)."""
    score = RATING_SCORE.get(rating, 9)
    if score <= 4:
        return "AA+"
    if score <= 7:
        return "A"
    if score <= 10:
        return "BBB"
    if score <= 13:
        return "BB"
    if score <= 16:
        return "B"
    return "CCC"


BUCKET_ORDER = ["AA+", "A", "BBB", "BB", "B", "CCC"]
SECTORS = ["Financials", "Energy", "TMT", "Consumer", "Healthcare",
           "Industrials", "Materials", "Utilities"]
SENIORITIES = ["Sr Secured", "Sr Unsecured", "Subordinated"]
MATURITY_BUCKETS: List[Tuple[str, float, float]] = [
    ("0-2Y", 0.0, 2.0), ("2-4Y", 2.0, 4.0), ("4-6Y", 4.0, 6.0),
    ("6-10Y", 6.0, 10.0), ("10Y+", 10.0, 99.0),
]
KEY_TENORS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
CURVE_TENORS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]

# ----------------------------------------------------------------------------
# 3. Issuer universe
#    (name, short ticker, equity RIC, sector, indicative rating, currency)
#    Equity RICs feed the Merton module in LSEG mode. Suffixes to check.
#    CDS_RICS below maps a short ticker to its 5Y CDS RIC. While it is empty
#    the "issuers" block stays simulated.
# ----------------------------------------------------------------------------
IssuerRow = Tuple[str, str, str, str, str, str]

HY_ISSUERS: List[IssuerRow] = [
    ("Carnival", "CCL", "CCL.N", "Consumer", "BB", "USD"),
    ("Caesars Entertainment", "CZR", "CZR.OQ", "Consumer", "B+", "USD"),
    ("Wynn Resorts", "WYNN", "WYNN.OQ", "Consumer", "BB-", "USD"),
    ("Macy's", "M", "M.N", "Consumer", "BB+", "USD"),
    ("Goodyear", "GT", "GT.OQ", "Consumer", "B+", "USD"),
    ("Newell Brands", "NWL", "NWL.OQ", "Consumer", "B+", "USD"),
    ("Hertz", "HTZ", "HTZ.OQ", "Consumer", "B", "USD"),
    ("Tenet Healthcare", "THC", "THC.N", "Healthcare", "BB-", "USD"),
    ("Community Health", "CYH", "CYH.N", "Healthcare", "B-", "USD"),
    ("DaVita", "DVA", "DVA.N", "Healthcare", "BB", "USD"),
    ("Bausch Health", "BHC", "BHC.N", "Healthcare", "B-", "USD"),
    ("Charter (CCO)", "CHTR", "CHTR.OQ", "TMT", "BB+", "USD"),
    ("Sirius XM", "SIRI", "SIRI.OQ", "TMT", "BB", "USD"),
    ("Lumen", "LUMN", "LUMN.N", "TMT", "B-", "USD"),
    ("EchoStar", "SATS", "SATS.OQ", "TMT", "B-", "USD"),
    ("Iron Mountain", "IRM", "IRM.N", "Industrials", "BB-", "USD"),
    ("TransDigm", "TDG", "TDG.N", "Industrials", "B+", "USD"),
    ("United Rentals", "URI", "URI.N", "Industrials", "BB+", "USD"),
    ("American Airlines", "AAL", "AAL.OQ", "Industrials", "B+", "USD"),
    ("United Airlines", "UAL", "UAL.OQ", "Industrials", "BB", "USD"),
    ("XPO", "XPO", "XPO.N", "Industrials", "BB", "USD"),
    ("Cleveland-Cliffs", "CLF", "CLF.N", "Materials", "B+", "USD"),
    ("Olin", "OLN", "OLN.N", "Materials", "BB+", "USD"),
    ("Celanese", "CE", "CE.N", "Materials", "BB+", "USD"),
    ("Venture Global", "VG", "VG.N", "Energy", "BB", "USD"),
    ("SM Energy", "SM", "SM.N", "Energy", "BB-", "USD"),
    ("Transocean", "RIG", "RIG.N", "Energy", "B", "USD"),
    ("Comstock Resources", "CRK", "CRK.N", "Energy", "B+", "USD"),
    ("Sunoco", "SUN", "SUN.N", "Energy", "BB+", "USD"),
    ("NRG Energy", "NRG", "NRG.N", "Utilities", "BB+", "USD"),
    ("Vistra", "VST", "VST.N", "Utilities", "BB+", "USD"),
    ("Talen Energy", "TLN", "TLN.OQ", "Utilities", "BB-", "USD"),
    ("OneMain", "OMF", "OMF.N", "Financials", "BB", "USD"),
    ("Navient", "NAVI", "NAVI.OQ", "Financials", "BB-", "USD"),
    ("Icahn Enterprises", "IEP", "IEP.OQ", "Financials", "B+", "USD"),
    ("Rocket Cos", "RKT", "RKT.N", "Financials", "BB+", "USD"),
]

IG_ISSUERS: List[IssuerRow] = [
    ("JPMorgan", "JPM", "JPM.N", "Financials", "A", "USD"),
    ("Bank of America", "BAC", "BAC.N", "Financials", "A-", "USD"),
    ("Goldman Sachs", "GS", "GS.N", "Financials", "BBB+", "USD"),
    ("Morgan Stanley", "MS", "MS.N", "Financials", "A-", "USD"),
    ("BNP Paribas", "BNP", "BNPP.PA", "Financials", "A-", "EUR"),
    ("HSBC", "HSBA", "HSBA.L", "Financials", "A-", "USD"),
    ("Barclays", "BARC", "BARC.L", "Financials", "BBB+", "USD"),
    ("Santander", "SAN", "SAN.MC", "Financials", "A-", "EUR"),
    ("UniCredit", "UCG", "CRDI.MI", "Financials", "BBB+", "EUR"),
    ("AXA", "AXA", "AXAF.PA", "Financials", "A+", "EUR"),
    ("AT&T", "T", "T.N", "TMT", "BBB", "USD"),
    ("Verizon", "VZ", "VZ.N", "TMT", "BBB+", "USD"),
    ("Comcast", "CMCSA", "CMCSA.OQ", "TMT", "A-", "USD"),
    ("Oracle", "ORCL", "ORCL.N", "TMT", "BBB", "USD"),
    ("Broadcom", "AVGO", "AVGO.OQ", "TMT", "BBB", "USD"),
    ("Orange", "ORA", "ORAN.PA", "TMT", "BBB+", "EUR"),
    ("Telefonica", "TEF", "TEF.MC", "TMT", "BBB-", "EUR"),
    ("Vodafone", "VOD", "VOD.L", "TMT", "BBB", "EUR"),
    ("AbbVie", "ABBV", "ABBV.N", "Healthcare", "A-", "USD"),
    ("Amgen", "AMGN", "AMGN.OQ", "Healthcare", "BBB+", "USD"),
    ("CVS Health", "CVS", "CVS.N", "Healthcare", "BBB", "USD"),
    ("Pfizer", "PFE", "PFE.N", "Healthcare", "A", "USD"),
    ("Bayer", "BAYN", "BAYGn.DE", "Healthcare", "BBB", "EUR"),
    ("Sanofi", "SAN.PA", "SASY.PA", "Healthcare", "AA", "EUR"),
    ("Exxon Mobil", "XOM", "XOM.N", "Energy", "AA-", "USD"),
    ("Occidental", "OXY", "OXY.N", "Energy", "BBB-", "USD"),
    ("Energy Transfer", "ET", "ET.N", "Energy", "BBB", "USD"),
    ("TotalEnergies", "TTE", "TTEF.PA", "Energy", "A+", "EUR"),
    ("BP", "BP", "BP.L", "Energy", "A-", "USD"),
    ("General Motors", "GM", "GM.N", "Consumer", "BBB", "USD"),
    ("Ford Motor", "F", "F.N", "Consumer", "BBB-", "USD"),
    ("Volkswagen", "VOW", "VOWG_p.DE", "Consumer", "BBB+", "EUR"),
    ("Home Depot", "HD", "HD.N", "Consumer", "A", "USD"),
    ("AB InBev", "ABI", "ABI.BR", "Consumer", "A-", "EUR"),
    ("LVMH", "MC", "LVMH.PA", "Consumer", "AA-", "EUR"),
    ("Boeing", "BA", "BA.N", "Industrials", "BBB-", "USD"),
    ("Airbus", "AIR", "AIR.PA", "Industrials", "A", "EUR"),
    ("Siemens", "SIE", "SIEGn.DE", "Industrials", "AA-", "EUR"),
    ("Dow", "DOW", "DOW.N", "Materials", "BBB", "USD"),
    ("Glencore", "GLEN", "GLEN.L", "Materials", "BBB+", "USD"),
    ("ArcelorMittal", "MT", "MT.AS", "Materials", "BBB-", "EUR"),
    ("Holcim", "HOLN", "HOLN.S", "Materials", "BBB+", "EUR"),
    ("Duke Energy", "DUK", "DUK.N", "Utilities", "BBB+", "USD"),
    ("NextEra", "NEE", "NEE.N", "Utilities", "A-", "USD"),
    ("Enel", "ENEL", "ENEI.MI", "Utilities", "BBB+", "EUR"),
    ("Engie", "ENGI", "ENGIE.PA", "Utilities", "BBB+", "EUR"),
]

# 5Y CDS RIC by short ticker, empty by default. Example {"F": "<5Y CDS RIC>"}
CDS_RICS: Dict[str, str] = {}


# ----------------------------------------------------------------------------
# 4. ETF universe
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class SimProfile:
    """Simulator calibration for one ETF."""
    issuers: str                  # "HY" or "IG"
    n_bonds: int
    maturity_range: Tuple[float, float]
    nav_last: float               # today's NAV, on which the simulation is anchored
    aum_musd: float               # simulated assets, in millions
    adv_musd: float               # simulated secondary daily traded value
    etf_spread_bps: float         # ETF bid/ask spread on screen
    pd_mean_bps: float            # long-run average premium
    pd_vol_bps: float             # daily volatility of the premium
    active_tilt: float            # strength of the active tilts (0 = replication)
    alpha_bps: float              # gross yearly alpha the simulator aims for


@dataclass(frozen=True)
class EtfSpec:
    """One ETF followed by the dashboard."""
    etf_id: str
    name: str
    segment: str
    active: bool                  # True for active, False for index tracking
    currency: str
    benchmark: str
    ter_bps: float
    ric: Optional[str] = None             # RIC of the listed line
    inav_ric: Optional[str] = None        # RIC of the iNAV, to fill in
    bench_ric: Optional[str] = None       # RIC of the total return index
    isin: Optional[str] = None
    cu_shares: int = 50_000               # size of a creation unit
    fixed_fee_usd: float = 750.0          # fixed fee per primary order
    var_fee_bps: float = 2.0              # variable fee (duties and charges)
    settle_mismatch_days: float = 1.0     # settlement gap between ETF and basket
    sim: SimProfile = field(default=None)  # type: ignore[assignment]


ETFS: Dict[str, EtfSpec] = {
    # Real product. RIC, ISIN, TER and benchmark come from the Janus Henderson
    # product page (LSE listing). The primary market settings (creation unit,
    # fees) are illustrative and should be set from the prospectus.
    "JSHY": EtfSpec(
        etf_id="JSHY",
        name="Janus Henderson US Short Duration High Yield Active Core UCITS ETF",
        segment="Active High Yield",
        active=True,
        currency="USD",
        benchmark="ICE BofA 0-5Y BB-B US High Yield Constrained",
        ter_bps=49.0,
        ric="JSHY.L",
        isin="IE0007W7MZL0",
        cu_shares=50_000,
        fixed_fee_usd=750.0,
        var_fee_bps=3.0,
        sim=SimProfile(issuers="HY", n_bonds=72, maturity_range=(1.0, 5.5),
                       nav_last=10.20, aum_musd=260.0, adv_musd=3.2,
                       etf_spread_bps=16.0, pd_mean_bps=14.0, pd_vol_bps=9.0,
                       active_tilt=0.45, alpha_bps=60.0),
    ),
    # Model sleeve "Global Corporate IG", with no RIC by default. To plug it
    # into a listed ETF, fill ric, inav_ric and bench_ric.
    "GCIG": EtfSpec(
        etf_id="GCIG",
        name="Global Corporate IG (model sleeve, index-tracking)",
        segment="Global Corporate IG",
        active=False,
        currency="USD",
        benchmark="Global Aggregate Corporate, USD hedged (to be mapped)",
        ter_bps=20.0,
        ric=None,
        cu_shares=100_000,
        fixed_fee_usd=500.0,
        var_fee_bps=1.0,
        sim=SimProfile(issuers="IG", n_bonds=96, maturity_range=(1.0, 14.0),
                       nav_last=25.40, aum_musd=840.0, adv_musd=11.0,
                       etf_spread_bps=7.0, pd_mean_bps=5.0, pd_vol_bps=4.0,
                       active_tilt=0.10, alpha_bps=0.0),
    ),
}
DEFAULT_ETF = "JSHY"


# ----------------------------------------------------------------------------
# 5. Authorized Participant settings (default values of the sliders)
# ----------------------------------------------------------------------------
@dataclass
class ApParams:
    """Cost assumptions of an AP for an in-kind creation or redemption."""
    order_cu: int = 2               # order size, in creation units
    nav_basis: str = "bid"          # the NAV prices the basket at "bid" or "mid"
    hedge_cost_bps: float = 1.5     # rates and credit hedge while the position is carried
    funding_rate_pct: float = 4.6   # yearly funding rate of the AP
    impact_coeff: float = 0.60      # coefficient of the square-root impact model
    participation: float = 0.20     # share of daily volume the order may take
    stress_mult: float = 1.0        # bid/ask multiplier (scenario)


# ----------------------------------------------------------------------------
# 6. LSEG market RICs and candidate fields
#    For each item the client tries the fields in order and keeps the first
#    one that returns something. To confirm in the Data Item Browser.
# ----------------------------------------------------------------------------
GOVT_RICS: Dict[str, Dict[float, str]] = {
    "USD": {0.25: "US3MT=RR", 0.5: "US6MT=RR", 1.0: "US1YT=RR", 2.0: "US2YT=RR",
            3.0: "US3YT=RR", 5.0: "US5YT=RR", 7.0: "US7YT=RR", 10.0: "US10YT=RR",
            20.0: "US20YT=RR", 30.0: "US30YT=RR"},
    "EUR": {0.25: "DE3MT=RR", 0.5: "DE6MT=RR", 1.0: "DE1YT=RR", 2.0: "DE2YT=RR",
            3.0: "DE3YT=RR", 5.0: "DE5YT=RR", 7.0: "DE7YT=RR", 10.0: "DE10YT=RR",
            20.0: "DE20YT=RR", 30.0: "DE30YT=RR"},
}

LSEG_FIELDS: Dict[str, List[str]] = {
    # time series
    "close": ["TRDPRC_1", "CLOSE", "MID_PRICE", "OFF_CLOSE", "HST_CLOSE"],
    "bid": ["BID"],
    "ask": ["ASK"],
    "volume": ["ACVOL_UNS", "VOLUME", "ACVOL_1"],
    "yield": ["MID_YLD_1", "B_YLD_1", "BID_YLD_1", "A_YLD_1", "RT_YIELD_1"],
    "nav": ["TR.NETASSETVAL", "TR.FundNAV"],
    "shares_out": ["TR.ETPSharesOutstanding", "TR.SharesOutstanding"],
    "oas": ["OAS_BID", "TR.OPTIONADJUSTEDSPREADBID", "ZSPREAD", "TR.ZSPREADBID"],
    "cds_spread": ["PRIMACT_1", "MID_SPREAD", "TR.PARMIDSPREAD"],
    # ETF basket (confirmed on the LSEG developer forum)
    "holding_ric": ["TR.FundHoldingRIC"],
    "holding_name": ["TR.FundHoldingName"],
    "holding_weight": ["TR.FundPercentageOfFundAssets"],
    # bond reference data and analytics (to check)
    "bond_issuer": ["TR.FIIssuerName", "TR.IssuerName"],
    "bond_coupon": ["TR.FiCouponRate", "TR.CouponRate", "COUPN_RATE"],
    "bond_maturity": ["TR.FiMaturityDate", "TR.MaturityDate", "MATUR_DATE"],
    "bond_seniority": ["TR.FiSeniorityType", "TR.SeniorityType"],
    "bond_rating": ["TR.FiIssuerSPLongRating", "TR.IssuerRating"],
    "bond_sector": ["TR.TRBCEconomicSector", "TR.FiIndustrySector"],
    "bond_amt_out": ["TR.FiAmountOutstanding", "TR.AmountOutstanding"],
    "bond_currency": ["TR.FiCurrency", "CURRENCY"],
    "bond_ytm": ["TR.YieldToMaturityAnalytics", "BID_YIELD", "TR.BIDYIELD"],
    "bond_duration": ["TR.ModifiedDurationAnalytics", "MOD_DURTN"],
    "bond_convexity": ["TR.ConvexityAnalytics", "CONVEXITY"],
    # issuer fundamentals for Merton
    "mktcap": ["TR.CompanyMarketCap"],
    "total_debt": ["TR.TotalDebtOutstanding", "TR.F.DebtTot"],
    "st_debt": ["TR.F.STDebtCurrPortOfLTDebt", "TR.ShortTermDebt"],
}

# Maps LSEG sectors (TRBC) to the eight sectors of the dashboard
TRBC_TO_SECTOR: Dict[str, str] = {
    "Financials": "Financials", "Real Estate": "Financials",
    "Energy": "Energy", "Technology": "TMT",
    "Telecommunications Services": "TMT", "Consumer Cyclicals": "Consumer",
    "Consumer Non-Cyclicals": "Consumer", "Healthcare": "Healthcare",
    "Industrials": "Industrials", "Basic Materials": "Materials",
    "Utilities": "Utilities",
}

# The Merton model says little about banks, whose liabilities are mostly
# deposits. They are left out of the equity against credit chart by default.
MERTON_EXCLUDE_SECTORS = ("Financials",)


def issuer_rows(kind: str) -> List[IssuerRow]:
    """Returns the issuers of a universe ("HY" or "IG")."""
    return HY_ISSUERS if kind.upper() == "HY" else IG_ISSUERS
