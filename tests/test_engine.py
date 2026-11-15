"""
Checks of the quantitative engine on the simulated market.

    python -m pytest -q

Each test checks something that can be worked out by hand, such as a model
identity, an order of magnitude or a direction of change.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg                                              # noqa: E402
from data.provider import DataProvider                           # noqa: E402
from engine import curves, etf_analytics as ea, merton, relative_value as rv, risk   # noqa: E402
from engine.analytics import Analytics                           # noqa: E402
from engine.bondmath import bond_price, interp_weights, risk_measures, ytm_from_price  # noqa: E402
from engine.trade_ideas import signal_for, simulate_idea         # noqa: E402


class BondMath(unittest.TestCase):
    def test_par_bond_prices_at_100(self):
        self.assertAlmostEqual(float(bond_price(0.05, 0.05, 7.0)), 100.0, places=8)

    def test_duration_of_a_zero_is_close_to_maturity(self):
        _, dur, cvx = risk_measures(0.04, 0.0, 5.0)
        self.assertAlmostEqual(float(dur), 5.0 / 1.02, places=3)
        self.assertGreater(float(cvx), 0.0)

    def test_yield_inverts_price(self):
        price = bond_price(np.array([0.03, 0.08]), np.array([0.05, 0.06]), np.array([4.0, 9.0]))
        ytm = ytm_from_price(price, np.array([0.05, 0.06]), np.array([4.0, 9.0]))
        np.testing.assert_allclose(ytm, [0.03, 0.08], atol=1e-7)

    def test_interpolation_weights_sum_to_one(self):
        w = interp_weights(cfg.KEY_TENORS, np.array([0.1, 2.5, 8.0, 45.0]))
        np.testing.assert_allclose(w.sum(axis=1), 1.0)


class Merton(unittest.TestCase):
    def test_solution_reprices_equity(self):
        E, sE, D, r = 100.0, 0.40, 100.0, 0.04
        V, sV = merton.solve_merton(E, sE, D, r, 1.0)
        d1 = (np.log(V / D) + (r + 0.5 * sV ** 2)) / sV
        equity = V * merton.norm.cdf(d1) - D * np.exp(-r) * merton.norm.cdf(d1 - sV)
        self.assertAlmostEqual(float(equity), E, places=6)
        self.assertAlmostEqual(float(sE * E), float(merton.norm.cdf(d1) * sV * V), places=6)

    def test_more_leverage_means_less_distance_to_default(self):
        dd = []
        for debt in (50.0, 100.0, 200.0):
            V, sV = merton.solve_merton(100.0, 0.35, debt, 0.04, 1.0)
            dd.append(float(merton.distance_to_default(V, sV, debt, 0.04, 1.0)[0]))
        self.assertTrue(dd[0] > dd[1] > dd[2])


class RelativeValue(unittest.TestCase):
    def test_half_life_of_a_known_process(self):
        rng = np.random.default_rng(0)
        x = np.zeros(4000)
        for t in range(1, len(x)):
            x[t] = 0.9 * x[t - 1] + rng.standard_normal()
        self.assertAlmostEqual(rv.half_life(x), np.log(2) / -np.log(0.9), delta=1.2)
        self.assertLess(rv.df_tstat(x), -5.0)

    def test_random_walk_has_no_half_life(self):
        x = np.cumsum(np.random.default_rng(1).standard_normal(3000))
        self.assertGreater(rv.half_life(x), 150)


class OnSimulatedMarket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.an = Analytics(DataProvider(mode="sim"))

    def test_daily_data_is_complete(self):
        for etf_id in cfg.ETFS:
            d = self.an.etf(etf_id).daily
            self.assertFalse(d.isna().any().any(), etf_id)
            self.assertTrue((d["ask"] > d["bid"]).all())
            self.assertEqual(len(d), cfg.HISTORY_DAYS)

    def test_weights(self):
        for etf_id in cfg.ETFS:
            b = self.an.etf(etf_id).basket
            self.assertAlmostEqual(b["bench_weight"].sum(), 1.0, places=9)
            self.assertTrue(0.97 < b["weight"].sum() < 1.0)

    def test_ap_costs(self):
        data = self.an.etf("JSHY")
        b, _ = self.an.basket("JSHY")
        last = data.daily.iloc[-1]
        args = (data.spec, b, last["nav"], last["etf_ba_bps"], last["basket_ba_bps"])
        bid = ea.ap_cost_stack(*args, cfg.ApParams(nav_basis="bid"))
        mid = ea.ap_cost_stack(*args, cfg.ApParams(nav_basis="mid"))
        # NAV at the bid, so creating costs the whole bid/ask and redeeming nothing
        self.assertAlmostEqual(bid["creation"].sum() - bid["redemption"].sum(),
                               last["basket_ba_bps"], places=6)
        self.assertAlmostEqual(mid["creation"].sum(), mid["redemption"].sum(), places=9)
        small = ea.ap_cost_stack(*args, cfg.ApParams(order_cu=1)).loc["Primary fees", "creation"]
        large = ea.ap_cost_stack(*args, cfg.ApParams(order_cu=50)).loc["Primary fees", "creation"]
        self.assertGreater(small, large)                  # the fixed fee dilutes

    def test_arbitrage_signal_matches_thresholds(self):
        data = self.an.etf("JSHY")
        b, _ = self.an.basket("JSHY")
        bands = ea.arb_bands(data.daily, data.spec, b, cfg.ApParams())
        self.assertTrue((bands["upper"] > 0).all() and (bands["lower"] < 0).all())
        self.assertTrue((bands.loc[bands["signal"] > 0, "pd_bps"] > bands.loc[bands["signal"] > 0, "upper"]).all())
        self.assertTrue((bands.loc[bands["signal"] < 0, "pd_bps"] < bands.loc[bands["signal"] < 0, "lower"]).all())
        self.assertTrue((bands.loc[bands["signal"] == 0, "edge_bps"] == 0).all())

    def test_liquidity(self):
        b, _ = self.an.basket("GCIG")
        self.assertLess(ea.basket_capacity(b, 0.10), ea.basket_capacity(b, 0.30))
        prof = ea.liquidation_profile(b, [1, 10, 100, 1000])
        self.assertTrue((prof.diff().dropna() <= 1e-9).all().all())     # bigger means less of it trades
        self.assertTrue((prof["5d"] >= prof["1d"] - 1e-9).all())
        self.assertTrue(b["liq_score"].between(0, 100).all())

    def test_curve_fit_recovers_the_simulated_curve(self):
        curve = self.an.rates().curves["USD"].iloc[-1]
        fit = curves.fit_nelson_siegel(list(curve.index), curve.to_numpy())
        self.assertLess(fit["rmse"], 1e-6)

    def test_var_measures_are_consistent(self):
        for etf_id in cfg.ETFS:
            var = self.an.var(etf_id)
            pnl = self.an.pnl(etf_id)["total"]
            self.assertGreater(var["hist"], 0)
            self.assertAlmostEqual(var["param"], 2.3263 * pnl.std(), delta=0.15 * var["param"])
            self.assertGreater(var["es"], 0.8 * var["hist"])
            ten_day = self.an.var(etf_id, 0.99, 10)["hist"]
            self.assertAlmostEqual(ten_day, var["hist"] * np.sqrt(10), places=10)
            _, st = risk.var_backtest(pnl)
            self.assertTrue(0.0 <= st["kupiec_p"] <= 1.0)

    def test_tracking(self):
        d = self.an.etf("GCIG").daily
        _, st = risk.tracking(d, cfg.ETFS["GCIG"].ter_bps)
        # an index fund lags by roughly its fees
        self.assertAlmostEqual(st["td_ann_bps"], -cfg.ETFS["GCIG"].ter_bps, delta=15)
        self.assertLess(st["te_ann_bps"], 40)

    def test_exposures_match_a_hand_computation(self):
        b = self.an.etf("JSHY").basket
        expo = risk.exposures(b)
        self.assertAlmostEqual(expo.loc["Modified duration (y)", "fund"],
                               float((b["weight"] * b["mod_dur"]).sum()), places=10)
        krd = risk.krd_table(b)
        self.assertAlmostEqual(krd["fund"].sum(), expo.loc["Modified duration (y)", "fund"], places=8)

    def test_capital_structure(self):
        snap = self.an.capital_structure()["snap"]
        self.assertTrue((snap["dd"] > 0).all())
        ig = snap[snap["universe"] == "IG"]["dd"].median()
        hy = snap[snap["universe"] == "HY"]["dd"].median()
        self.assertGreater(ig, hy)                         # IG sits further from default
        self.assertGreater(np.corrcoef(np.log(snap["fair_spread"]), np.log(snap["mkt_spread"]))[0, 1], 0.8)

    def test_trade_ideas_are_well_formed(self):
        ideas = self.an.ideas()
        self.assertGreater(len(ideas), 5)
        self.assertTrue(ideas["id"].is_unique)
        self.assertTrue(ideas["conviction"].between(1, 5).all())
        self.assertTrue((ideas["exp_pnl_bps"] >= 0).all() and (ideas["risk_bps"] >= 0).all())
        rev = ideas[ideas["type"].isin(["Credit pair", "Seniority", "Capital structure"])]
        # the target is between entry and mean, the stop on the other side
        self.assertTrue((np.sign(rev["entry"] - rev["target"]) == -np.sign(rev["entry"] - rev["stop"])).all())
        row = rev.iloc[0]
        sim = simulate_idea(row, signal_for(self.an, row["kind"], tuple(row["ref"].split("|"))))
        self.assertAlmostEqual(sim["p_target"] + sim["p_stop"] + (1 - sim["p_target"] - sim["p_stop"]), 1.0)
        self.assertEqual(len(sim["fan"]), int(row["horizon_d"]) + 1)


if __name__ == "__main__":
    unittest.main()
