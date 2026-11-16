"""
Hybrid data provider, LSEG or simulator.

The provider cuts the data into consistent blocks. Sovereign curves, the
issuer panel and, for each ETF, price and NAV, basket and bond analytics,
intraday and iNAV.

For each block it tries LSEG, checks that the result is usable, and otherwise
falls back to the simulator for the whole block. Real and simulated data are
never mixed inside a block, since a real price against a simulated NAV would
give a premium that means nothing. The source of each block is logged and
shown on the Overview tab.
"""
from __future__ import annotations

import threading
import time
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as cfg
from data.lseg_client import LsegClient, pick_column
from data.models import EtfData, IssuerData, RatesData
from data.synthetic import SyntheticMarket
from engine.bondmath import bond_price, risk_measures

F = cfg.LSEG_FIELDS


class LiveDataError(RuntimeError):
    """Raised when LSEG answers but the data cannot be used."""


class DataProvider:
    def __init__(self, mode: str = cfg.DATA_MODE, seed: int = cfg.SEED,
                 client: Optional[LsegClient] = None):
        self.mode = mode
        self.sim = SyntheticMarket(seed=seed, end=cfg.SIM_ASOF)
        self.client = client
        self._cache: Dict[str, object] = {}
        self._status: Dict[str, dict] = {}
        self._lock = threading.RLock()
        self._last_intraday_pull: Dict[str, float] = {}

        if mode == "sim":
            self._log("LSEG session", "SIM", "simulation requested (mode sim)")
        else:
            if self.client is None:
                self.client = LsegClient(cfg.LSEG_APP_KEY)
                self.client.connect()
            if self.client.connected:
                self._log("LSEG session", "LSEG", f"session open through {self.client.label}")
            else:
                self._log("LSEG session", "SIM", self.client.error)

    # ---------------------------------------------------------------- status
    @property
    def live(self) -> bool:
        return bool(self.client is not None and self.client.connected)

    def _log(self, block: str, source: str, detail: str) -> None:
        self._status[block] = {"Block": block, "Source": source, "Detail": detail}
        if self.mode == "lseg" and source != "LSEG" and block != "LSEG session":
            print(f"[data] {block} -> {source} ({detail})")

    def status_table(self) -> pd.DataFrame:
        return pd.DataFrame(list(self._status.values()), columns=["Block", "Source", "Detail"])

    def mode_label(self) -> str:
        """SIMULATION, LSEG LIVE or HYBRID, from the source of the blocks loaded so far."""
        src = {s["Source"] for k, s in self._status.items() if k != "LSEG session"}
        src.discard("N/A")
        if src == {"LSEG"}:
            return "LSEG LIVE"
        if "LSEG" in src:
            return "HYBRID"
        return "SIMULATION"

    @staticmethod
    def _window() -> Tuple[str, str]:
        end = pd.Timestamp.today().normalize()
        start = end - timedelta(days=int(cfg.HISTORY_DAYS * 365.25 / 252) + 10)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    def _block(self, key: str, label: str, live_fn: Callable, sim_fn: Callable):
        """Loads a block from LSEG when possible, from the simulator otherwise. Cached."""
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            obj, source = None, "SIM"
            detail = "stochastic simulator"
            if self.live:
                try:
                    obj, detail = live_fn()
                    source = "LSEG"
                except LiveDataError as exc:
                    detail = f"fell back to simulation · {exc}"
                except Exception as exc:        # never take the app down
                    detail = f"fell back to simulation · {type(exc).__name__}: {exc}"[:220]
            if obj is None:
                obj = sim_fn()
                if obj is None:
                    source = "N/A"
            self._log(label, source, detail)
            self._cache[key] = obj
            return obj

    # ----------------------------------------------------------------- blocks
    def rates(self) -> RatesData:
        return self._block("rates", "Sovereign curves", self._live_rates,
                           lambda: self.sim.rates)

    def issuers(self) -> IssuerData:
        return self._block("issuers", "Issuers · equity, 5Y spread, balance sheet",
                           self._live_issuers, lambda: self.sim.issuers)

    def etf(self, etf_id: str) -> EtfData:
        with self._lock:
            key = f"etf:{etf_id}"
            if key in self._cache:
                return self._cache[key]            # type: ignore[return-value]
            spec = cfg.ETFS[etf_id]
            sim = self.sim.etf(etf_id)
            daily = self._block(f"daily:{etf_id}", f"{etf_id} · price, NAV, volume",
                                lambda: self._live_daily(spec), lambda: sim.daily)
            basket, oas_hist = self._block(
                f"basket:{etf_id}", f"{etf_id} · basket and bond analytics",
                lambda: self._live_basket(spec), lambda: (sim.basket, sim.oas_hist))
            daily_is_sim = daily is sim.daily
            intraday = self._block(
                f"intraday:{etf_id}", f"{etf_id} · intraday and iNAV",
                lambda: self._live_intraday(spec),
                lambda: sim.intraday if daily_is_sim else None)
            if not daily_is_sim:
                daily = self._complete_daily(daily, basket, sim, basket is sim.basket)
            src = lambda k: self._status[k]["Source"]          # noqa: E731
            data = EtfData(
                spec=spec, daily=daily, intraday=intraday, basket=basket, oas_hist=oas_hist,
                source={"etf": src(f"{etf_id} · price, NAV, volume"),
                        "basket": src(f"{etf_id} · basket and bond analytics"),
                        "intraday": src(f"{etf_id} · intraday and iNAV")})
            self._cache[key] = data
            return data

    def refresh_intraday(self, etf_id: str) -> Optional[pd.DataFrame]:
        """Refreshes the intraday bars when they come from LSEG, every 30 s at most."""
        data = self.etf(etf_id)
        if data.source.get("intraday") != "LSEG":
            return data.intraday
        now = time.time()
        if now - self._last_intraday_pull.get(etf_id, 0.0) > 30.0:
            self._last_intraday_pull[etf_id] = now
            try:
                data.intraday, _ = self._live_intraday(data.spec)
            except Exception:
                pass                                 # keep the last picture we have
        return data.intraday

    # ------------------------------------------------------------- LSEG, rates
    def _live_rates(self) -> Tuple[RatesData, str]:
        start, end = self._window()
        curves, parts = {}, []
        for ccy, ricmap in cfg.GOVT_RICS.items():
            wide = self.client.wide_history(list(ricmap.values()), F["yield"], start, end)
            if wide.empty:
                raise LiveDataError(f"no {ccy} curve returned (fields {F['yield']})")
            inv = {ric: tenor for tenor, ric in ricmap.items()}
            wide = wide.rename(columns=inv)
            wide = wide[sorted(c for c in wide.columns if c in inv.values())]
            if wide.shape[1] < 4 or len(wide.dropna(how="all")) < 60:
                raise LiveDataError(f"{ccy} curve too sparse ({wide.shape[1]} tenors)")
            # Missing tenors are interpolated along the curve
            full = (wide.reindex(columns=cfg.CURVE_TENORS).astype(float)
                    .interpolate(axis=1, limit_direction="both").ffill().dropna())
            curves[ccy] = full
            parts.append(f"{ccy} {wide.shape[1]} tenors")
        return RatesData(curves=curves, source="LSEG"), ", ".join(parts)

    # ----------------------------------------------------------- LSEG, issuers
    def _live_issuers(self) -> Tuple[IssuerData, str]:
        if len(cfg.CDS_RICS) < 10:
            raise LiveDataError("fill at least 10 RICs in config.CDS_RICS")
        start, end = self._window()
        rows = {r[1]: (r, "HY") for r in cfg.HY_ISSUERS}
        rows.update({r[1]: (r, "IG") for r in cfg.IG_ISSUERS})
        tickers = [t for t in cfg.CDS_RICS if t in rows]
        eq_ric = {rows[t][0][2]: t for t in tickers}
        cds_ric = {cfg.CDS_RICS[t]: t for t in tickers}
        equity = self.client.wide_history(list(eq_ric), F["close"], start, end).rename(columns=eq_ric)
        spread = self.client.wide_history(list(cds_ric), F["cds_spread"], start, end).rename(columns=cds_ric)
        ok = [t for t in tickers if t in equity.columns and t in spread.columns]
        if len(ok) < 10:
            raise LiveDataError(f"only {len(ok)} issuers with both equity and CDS")
        fund, _ = self.client.snapshot_roles(
            [rows[t][0][2] for t in ok],
            {"mktcap": F["mktcap"], "total_debt": F["total_debt"], "st_debt": F["st_debt"]})
        if fund.empty or "mktcap" not in fund or "total_debt" not in fund:
            raise LiveDataError("market cap or debt not available")
        fund["ticker"] = fund["instrument"].map(eq_ric)
        fund = fund.dropna(subset=["ticker"]).drop_duplicates("ticker").set_index("ticker")
        ok = [t for t in ok if t in fund.index and pd.notna(fund.loc[t, "mktcap"])
              and pd.notna(fund.loc[t, "total_debt"])]
        if len(ok) < 10:
            raise LiveDataError("incomplete fundamentals")
        debt = pd.to_numeric(fund.loc[ok, "total_debt"], errors="coerce") / 1e6
        st = (pd.to_numeric(fund["st_debt"], errors="coerce").reindex(ok) / 1e6
              if "st_debt" in fund else 0.25 * debt)
        st = st.fillna(0.25 * debt)
        static = pd.DataFrame({
            "name": [rows[t][0][0] for t in ok],
            "equity_ric": [rows[t][0][2] for t in ok],
            "sector": [rows[t][0][3] for t in ok],
            "rating": [rows[t][0][4] for t in ok],
            "bucket": [cfg.rating_bucket(rows[t][0][4]) for t in ok],
            "currency": [rows[t][0][5] for t in ok],
            "universe": [rows[t][1] for t in ok],
            "mktcap_musd": pd.to_numeric(fund.loc[ok, "mktcap"], errors="coerce").to_numpy() / 1e6,
            "debt_musd": debt.to_numpy(),
            "st_debt_musd": st.to_numpy(),
            "lt_debt_musd": (debt - st).clip(lower=0.0).to_numpy(),
        }, index=pd.Index(ok, name="ticker"))
        idx = equity.index.intersection(spread.index)
        data = IssuerData(static=static, equity=equity.loc[idx, ok].ffill(),
                          spread5y=spread.loc[idx, ok].ffill(), source="LSEG")
        return data, f"{len(ok)} issuers, {len(idx)} days"

    # ----------------------------------------------------- LSEG, price and NAV
    def _live_daily(self, spec: cfg.EtfSpec) -> Tuple[pd.DataFrame, str]:
        if not spec.ric:
            raise LiveDataError("no RIC set for this ETF in config.ETFS")
        start, end = self._window()
        hist = self.client.history([spec.ric], None, start, end).get(spec.ric)
        if hist is None:
            raise LiveDataError(f"no history for {spec.ric}")
        col = {k: pick_column(hist, F[k]) for k in ("close", "bid", "ask", "volume")}
        if col["close"] is None:
            raise LiveDataError(f"no close field for {spec.ric}")
        nav = self.client.wide_history([spec.ric], F["nav"], start, end)
        if nav.empty:
            raise LiveDataError(f"no NAV for {spec.ric} (fields {F['nav']})")
        d = pd.DataFrame({"price": hist[col["close"]]})
        d["nav"] = nav.iloc[:, 0].reindex(d.index).ffill(limit=3)   # NAV is published a day late
        d = d.dropna()
        if len(d) < 40:
            raise LiveDataError(f"price and NAV overlap too short ({len(d)} days)")
        half = spec.sim.etf_spread_bps / 2.0 / 1e4
        for k, default in (("bid", d["price"] * (1 - half)), ("ask", d["price"] * (1 + half)),
                           ("volume", 0.0)):
            d[k] = hist[col[k]].reindex(d.index) if col[k] else np.nan
            d[k] = d[k].fillna(default)
        d["value_musd"] = d["volume"] * d["price"] / 1e6
        shares = self.client.wide_history([spec.ric], F["shares_out"], start, end)
        d["shares_out"] = shares.iloc[:, 0].reindex(d.index).ffill() if not shares.empty else np.nan
        d["flow_cu"] = (d["shares_out"].diff() / spec.cu_shares).round().fillna(0.0)
        d["aum_musd"] = d["shares_out"] * d["nav"] / 1e6
        d["bench"] = np.nan
        if spec.bench_ric:
            bench = self.client.wide_history([spec.bench_ric], F["close"], start, end)
            if not bench.empty:
                d["bench"] = bench.iloc[:, 0].reindex(d.index).ffill()
        d["etf_ba_bps"] = (d["ask"] - d["bid"]) / d["price"] * 1e4
        extra = []
        if shares.empty:
            extra.append("shares outstanding missing")
        if d["bench"].isna().all():
            extra.append("benchmark not mapped")
        detail = f"{spec.ric}, {len(d)} days since {d.index[0].date()}"
        return d, detail + (" · " + ", ".join(extra) if extra else "")

    def _complete_daily(self, daily: pd.DataFrame, basket: pd.DataFrame, sim: EtfData,
                        basket_is_sim: bool) -> pd.DataFrame:
        """Adds the basket liquidity columns to a live price block."""
        d = daily.copy()
        if basket_is_sim:
            for c in ("basket_ba_bps", "basket_turnover_musd"):
                d[c] = sim.daily[c].reindex(d.index).ffill().bfill()
        else:
            w = basket["weight"] / basket["weight"].sum()
            d["basket_ba_bps"] = float((w * basket["ba_bps"]).sum())
            d["basket_turnover_musd"] = float(basket["adv_musd"].sum())
        return d

    # ------------------------------------------------------------- LSEG, basket
    def _live_basket(self, spec: cfg.EtfSpec) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], str]:
        from engine.etf_analytics import model_liquidity      # late import

        if not spec.ric:
            raise LiveDataError("no RIC set for this ETF")
        hold, missing = self.client.snapshot_roles(
            [spec.ric], {"ric": F["holding_ric"], "name": F["holding_name"],
                         "weight": F["holding_weight"]}, {"EndNum": 4000})
        if hold.empty or "ric" in missing or "weight" in missing:
            raise LiveDataError("basket holdings not available (TR.FundHoldingRIC)")
        hold = hold.dropna(subset=["ric"])
        hold["weight"] = pd.to_numeric(hold["weight"], errors="coerce") / 100.0
        hold = hold[hold["weight"] > 0].drop_duplicates("ric").set_index("ric")
        if len(hold) < 20:
            raise LiveDataError(f"basket too short ({len(hold)} lines)")

        roles = {k[5:]: F[k] for k in F if k.startswith("bond_")}
        ana, _ = self.client.snapshot_roles(list(hold.index), roles)
        if ana.empty:
            raise LiveDataError("bond analytics not available")
        ana = ana.drop_duplicates("instrument").set_index("instrument").reindex(hold.index)
        for need in ("coupon", "maturity"):
            if need not in ana:
                raise LiveDataError(f"field {need} missing (see LSEG_FIELDS)")

        today = pd.Timestamp.today().normalize()
        b = pd.DataFrame(index=hold.index)
        b["weight"] = hold["weight"]
        if "issuer" in ana:
            issuer = ana["issuer"]
        elif "name" in hold:
            issuer = hold["name"]
        else:
            issuer = pd.Series("n/a", index=b.index)
        b["issuer"] = issuer.fillna("n/a").astype(str)
        b["ticker"] = b["issuer"].str.replace(r"[^A-Za-z]", "", regex=True).str.upper().str[:6]
        raw_sector = ana["sector"] if "sector" in ana else pd.Series("", index=b.index)
        b["sector"] = raw_sector.map(cfg.TRBC_TO_SECTOR).fillna("Industrials")
        default_rating = "BB" if spec.sim.issuers == "HY" else "BBB"
        rating = (ana["rating"].astype(str).str.strip().str.upper() if "rating" in ana
                  else pd.Series(default_rating, index=b.index))
        b["rating"] = rating.where(rating.isin(cfg.RATINGS), default_rating)
        b["bucket"] = b["rating"].map(cfg.rating_bucket)
        sen = (ana["seniority"].astype(str).str.lower() if "seniority" in ana
               else pd.Series("", index=b.index))
        b["seniority"] = np.select(
            [sen.str.contains("sub|junior|tier", regex=True), sen.str.contains("secured")
             & ~sen.str.contains("unsecured")],
            ["Subordinated", "Sr Secured"], default="Sr Unsecured")
        b["currency"] = (ana["currency"].astype(str).str.upper() if "currency" in ana
                         else spec.currency)
        # Unknown currency or no curve for it, so fall back to the fund currency
        b["currency"] = b["currency"].where(b["currency"].isin(list(self.rates().curves)),
                                            spec.currency)
        b["coupon"] = pd.to_numeric(ana["coupon"], errors="coerce")
        b["maturity_date"] = pd.to_datetime(ana["maturity"], errors="coerce")
        b["maturity_years"] = (b["maturity_date"] - today).dt.days / 365.25
        b = b[(b["maturity_years"] > 0.05) & b["coupon"].notna()]     # drops cash and derivatives
        if len(b) < 20:
            raise LiveDataError(f"too few usable bonds ({len(b)})")

        start, end = self._window()
        oas_hist = self.client.wide_history(list(b.index), F["oas"], start, end)
        keep = [r for r in b.index if r in oas_hist.columns
                and oas_hist[r].notna().sum() >= 60]
        if b.loc[keep, "weight"].sum() < 0.7 * b["weight"].sum():
            raise LiveDataError("spread history covers less than 70% of the basket")
        b = b.loc[keep]
        oas_hist = oas_hist[keep].ffill().dropna(how="all")
        b["oas"] = oas_hist.iloc[-1].reindex(b.index)

        tau = b["maturity_years"].to_numpy()
        ytm = pd.to_numeric(ana["ytm"], errors="coerce").reindex(b.index) if "ytm" in ana else None
        if ytm is None or ytm.isna().any():
            # Yield rebuilt from the sovereign curve at the tenor plus the spread
            curves = self.rates().curves
            govt = np.array([np.interp(t, cfg.CURVE_TENORS,
                                       curves.get(c, curves["USD"]).iloc[-1].to_numpy())
                             for t, c in zip(tau, b["currency"])])
            rebuilt = pd.Series(govt + b["oas"].to_numpy() / 100.0, index=b.index)
            ytm = rebuilt if ytm is None else ytm.fillna(rebuilt)
        b["ytm"] = ytm
        price, dur, cvx = risk_measures(b["ytm"].to_numpy() / 100.0,
                                        b["coupon"].to_numpy() / 100.0, tau)
        b["price"] = price
        lseg_dur = pd.to_numeric(ana["duration"], errors="coerce").reindex(b.index) if "duration" in ana else None
        b["mod_dur"] = dur if lseg_dur is None else lseg_dur.fillna(pd.Series(dur, index=b.index))
        lseg_cvx = pd.to_numeric(ana["convexity"], errors="coerce").reindex(b.index) if "convexity" in ana else None
        b["convexity"] = cvx if lseg_cvx is None else lseg_cvx.fillna(pd.Series(cvx, index=b.index))
        b["spread_dur"] = b["mod_dur"]
        amt = pd.to_numeric(ana["amt_out"], errors="coerce").reindex(b.index) / 1e6 if "amt_out" in ana else None
        b["amt_out_musd"] = 750.0 if amt is None else amt.fillna(amt.median() if amt.notna().any() else 750.0)
        b["age_years"] = 3.0
        b["adv_musd"], b["ba_bps"] = model_liquidity(b)        # modelled liquidity
        bw = np.minimum(b["amt_out_musd"] / b["amt_out_musd"].sum(), 0.03)
        b["bench_weight"] = bw / bw.sum()                       # proxy, weights by amount outstanding

        ids, seen = [], {}
        for tkr, c, d in zip(b["ticker"], b["coupon"], b["maturity_date"]):
            base = f"{tkr} {c:.3f} {d.year}"
            seen[base] = seen.get(base, 0) + 1
            ids.append(base if seen[base] == 1 else f"{base} #{seen[base]}")
        b["ric"] = b.index
        b["bond_id"] = ids
        oas_hist.columns = ids
        b = b.set_index("bond_id", drop=False)
        b.index.name = None
        detail = (f"{len(b)} bonds, {len(oas_hist)} days of spreads · "
                  "volumes, bid/ask and benchmark weights are modelled")
        return (b, oas_hist), detail

    # ----------------------------------------------------------- LSEG, intraday
    def _live_intraday(self, spec: cfg.EtfSpec) -> Tuple[pd.DataFrame, str]:
        if not (spec.ric and spec.inav_ric):
            raise LiveDataError("inav_ric not set in config.ETFS")
        start = pd.Timestamp.today().normalize().strftime("%Y-%m-%dT%H:%M:%S")
        hist = self.client.history([spec.ric, spec.inav_ric], None, start, None, "minute")
        px, inav = hist.get(spec.ric), hist.get(spec.inav_ric)
        if px is None or inav is None:
            raise LiveDataError("no minute bars for the ETF or its iNAV")
        c_px, c_nav = pick_column(px, F["close"]), pick_column(inav, F["close"])
        if c_px is None or c_nav is None:
            raise LiveDataError("no intraday price field")
        d = pd.DataFrame({"price": px[c_px]})
        d["inav"] = inav[c_nav].reindex(d.index).ffill()
        half = spec.sim.etf_spread_bps / 2.0 / 1e4
        for k, default in (("bid", d["price"] * (1 - half)), ("ask", d["price"] * (1 + half)),
                           ("volume", 0.0)):
            c = pick_column(px, F[k])
            d[k] = px[c].reindex(d.index) if c else np.nan
            d[k] = d[k].fillna(default)
        d = d.dropna(subset=["price", "inav"])
        if len(d) < 5:
            raise LiveDataError("intraday session too short")
        return d, f"{len(d)} minute bars"
