"""
The LSEG path, tested with a fake library.

A test cannot open a Workspace session. The client is therefore plugged into
an object that mimics lseg.data (same functions, same return shapes) to check
the plumbing. That covers the clean-up of responses, the choice among
candidate fields, the fallback block by block, and how the engine copes with
a short history (an ETF launched less than a year ago).
"""
import os
import sys
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                   # noqa: E402
from data.lseg_client import LsegClient                # noqa: E402
from data.provider import DataProvider                 # noqa: E402
from data.synthetic import SyntheticMarket             # noqa: E402
from engine.analytics import Analytics                 # noqa: E402

SHORT_HISTORY = 210        # trading days served for the ETF


class FakeLseg:
    """Mimics lseg.data from an independent simulated market (another seed)."""

    class HeaderType:
        NAME, TITLE = 1, 2

    def __init__(self):
        self.m = SyntheticMarket(seed=99)
        self.etf = self.m.etf("JSHY")
        self.rics = {f"BOND{i:03d}=": bid for i, bid in enumerate(self.etf.basket.index)}
        self.calls = []

    def open_session(self, app_key=None):
        return type("Session", (), {"open_state": "OpenState.Opened"})()

    def close_session(self):
        pass

    # --- time series
    def get_history(self, universe, fields=None, interval=None, start=None, end=None,
                    header_type=None):
        self.calls.append(("history", tuple(universe), tuple(fields or ())))
        d = self.etf.daily.iloc[-SHORT_HISTORY:]
        if universe == ["JSHY.L"]:
            if not fields:
                return pd.DataFrame({"TRDPRC_1": d["price"], "BID": d["bid"], "ASK": d["ask"],
                                     "ACVOL_UNS": d["volume"]})
            if fields == ["TR.NETASSETVAL"]:
                return pd.DataFrame({"TR.NETASSETVAL": d["nav"]})
            if fields == ["TR.ETPSharesOutstanding"]:
                return pd.DataFrame({"TR.ETPSHARESOUTSTANDING": d["shares_out"]})
            return pd.DataFrame()
        govt = {ric: (ccy, tenor) for ccy, mp in cfg.GOVT_RICS.items() for tenor, ric in mp.items()}
        if universe[0] in govt:
            cols = {}
            for ric in universe:
                ccy, tenor = govt[ric]
                if tenor in (0.5, 20.0):                 # two tenors missing, to be interpolated
                    continue
                cols[(ric, "B_YLD_1")] = self.m.rates.curves[ccy][tenor]
                cols[(ric, "HIGH_YLD")] = self.m.rates.curves[ccy][tenor] + 0.01
            return pd.DataFrame(cols)
        if universe[0] in self.rics:
            cols = {}
            for ric in universe[:-2]:                     # two bonds without history
                cols[(ric, "OAS_BID")] = self.etf.oas_hist[self.rics[ric]]
                cols[(ric, "BID")] = 100.0
            return pd.DataFrame(cols)
        return pd.DataFrame()

    # --- point-in-time data
    def get_data(self, universe, fields, parameters=None, header_type=None):
        self.calls.append(("data", tuple(universe), tuple(fields)))
        b = self.etf.basket
        if universe == ["JSHY.L"]:
            n = len(b)
            return pd.DataFrame({
                "Instrument": ["JSHY.L"] * (n + 1),
                "TR.FUNDHOLDINGRIC": list(self.rics) + [None],
                "TR.FUNDHOLDINGNAME": list(b["issuer"]) + ["USD CASH"],
                "TR.FUNDPERCENTAGEOFFUNDASSETS": list(b["weight"] * 100) + [1.5],
            })
        if universe[0] in self.rics:
            rows = b.loc[[self.rics[r] for r in universe]]
            tiers = {"Sr Unsecured": "Senior Unsecured", "Sr Secured": "Senior Secured",
                     "Subordinated": "Subordinated Unsecured"}
            trbc = {v: k for k, v in cfg.TRBC_TO_SECTOR.items()}
            return pd.DataFrame({
                "Instrument": universe,
                "TR.FIISSUERNAME": rows["issuer"].to_numpy(),
                "TR.FICOUPONRATE": rows["coupon"].to_numpy(),
                "TR.FIMATURITYDATE": rows["maturity_date"].dt.strftime("%Y-%m-%d").to_numpy(),
                "TR.FISENIORITYTYPE": rows["seniority"].map(tiers).to_numpy(),
                "TR.FIISSUERSPLONGRATING": rows["rating"].to_numpy(),
                "TR.TRBCECONOMICSECTOR": rows["sector"].map(trbc).to_numpy(),
                "TR.FIAMOUNTOUTSTANDING": rows["amt_out_musd"].to_numpy() * 1e6,
                "TR.FICURRENCY": rows["currency"].to_numpy(),
            })
        return pd.DataFrame()


def _mock_client() -> LsegClient:
    fake = FakeLseg()
    client = LsegClient("test-key")
    client.lib, client.kind, client.label = fake, "ld", "fake lseg-data"
    client.connected, client.error = True, ""
    client._hdr = client._header_kwargs(fake)
    return client


class LivePathWithMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("error")
        cls.client = _mock_client()
        cls.provider = DataProvider(mode="auto", client=cls.client)
        cls.an = Analytics(cls.provider)
        cls.data = cls.provider.etf("JSHY")
        cls.status = cls.provider.status_table().set_index("Block")["Source"].to_dict()

    def test_header_request(self):
        self.assertEqual(self.client._hdr, {"header_type": 1})

    def test_blocks_come_from_the_right_source(self):
        self.assertEqual(self.status["LSEG session"], "LSEG")
        self.assertEqual(self.status["Sovereign curves"], "LSEG")
        self.assertEqual(self.status["JSHY · price, NAV, volume"], "LSEG")
        self.assertEqual(self.status["JSHY · basket and bond analytics"], "LSEG")
        self.assertEqual(self.status["JSHY · intraday and iNAV"], "N/A")    # no inav_ric
        self.provider.etf("GCIG")
        status = self.provider.status_table().set_index("Block")["Source"].to_dict()
        self.assertEqual(status["GCIG · price, NAV, volume"], "SIM")          # no RIC
        self.assertEqual(self.provider.mode_label(), "HYBRID")

    def test_daily_block(self):
        d = self.data.daily
        self.assertEqual(len(d), SHORT_HISTORY)
        fake = self.client.lib.etf.daily.iloc[-SHORT_HISTORY:]
        np.testing.assert_allclose(d["price"].to_numpy(), fake["price"].to_numpy())
        np.testing.assert_allclose(d["nav"].to_numpy(), fake["nav"].to_numpy())
        for col in ("bid", "ask", "volume", "value_musd", "shares_out", "flow_cu", "aum_musd",
                    "etf_ba_bps", "basket_ba_bps", "basket_turnover_musd"):
            self.assertIn(col, d.columns)
            self.assertFalse(d[col].isna().any(), col)
        self.assertTrue(d["bench"].isna().all())               # benchmark not mapped
        self.assertIsNone(self.data.intraday)

    def test_rates_block_interpolates_missing_tenors(self):
        usd = self.provider.rates().curves["USD"]
        self.assertEqual(list(usd.columns), cfg.CURVE_TENORS)
        self.assertFalse(usd.isna().any().any())
        truth = self.client.lib.m.rates.curves["USD"]
        np.testing.assert_allclose(usd[10.0].to_numpy(), truth[10.0].reindex(usd.index).to_numpy())

    def test_basket_block(self):
        b, fake = self.data.basket, self.client.lib.etf.basket
        chunks = -(-len(fake) // 40)                           # the client queries in batches of 40
        self.assertEqual(len(b), len(fake) - 2 * chunks)       # two bonds without history per batch
        self.assertEqual(set(b["seniority"]), set(fake["seniority"]))
        self.assertTrue(set(b["sector"]) <= set(cfg.SECTORS))
        self.assertEqual(list(self.data.oas_hist.columns), list(b.index))
        for col in ("ytm", "mod_dur", "convexity", "price", "adv_musd", "ba_bps", "bench_weight"):
            self.assertFalse(b[col].isna().any(), col)
        self.assertAlmostEqual(b["bench_weight"].sum(), 1.0, places=9)
        self.assertTrue((b["mod_dur"] > 0).all() and (b["mod_dur"] < b["maturity_years"] + 0.1).all())

    def test_engine_runs_on_live_shaped_data(self):
        from engine import etf_analytics as ea, risk
        b, _ = self.an.basket("JSHY")
        bands = ea.arb_bands(self.data.daily, self.data.spec, b, cfg.ApParams())
        self.assertEqual(len(bands), SHORT_HISTORY)
        self.assertGreater(self.an.var("JSHY")["hist"], 0)
        tr, stats = risk.tracking(self.data.daily, 49.0)
        self.assertTrue(tr.empty and stats == {})              # no benchmark, nothing to track
        self.assertGreater(len(self.an.ideas()), 0)


class NoSession(unittest.TestCase):
    def test_everything_falls_back_to_simulation(self):
        client = LsegClient("")
        client.connected, client.error = False, "lseg-data: session closed, is Workspace running?"
        provider = DataProvider(mode="auto", client=client)
        provider.etf("JSHY")
        provider.issuers()
        self.assertEqual(provider.mode_label(), "SIMULATION")
        self.assertEqual(set(provider.status_table()["Source"]), {"SIM"})


if __name__ == "__main__":
    unittest.main()
