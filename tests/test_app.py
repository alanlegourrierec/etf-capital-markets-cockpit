"""
End-to-end test of the Dash app, without a browser.

Each callback is called through Dash's real HTTP endpoint, with plausible
control values, for both ETFs and both themes. Everything must answer without
error and every figure must hold data.
"""
import json
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app          # noqa: E402

DEFAULTS = {
    ("lookback", "value"): "1Y", ("theme", "value"): "dark", ("tick", "n_intervals"): 4,
    ("ap-size", "value"): 2, ("ap-basis", "value"): "bid", ("ap-stress", "value"): 1.0,
    ("ap-funding", "value"): 4.6, ("ap-hedge", "value"): 1.5, ("ap-impact", "value"): 0.6,
    ("lq-part", "value"): 20, ("lq-order", "value"): 25,
    ("rk-level", "value"): 0.99, ("rk-horizon", "value"): 1,
    ("id-scope", "value"): "all", ("id-status", "value"): "all", ("id-types", "value"): [],
    ("id-conv", "value"): 1,
}


def _parse(output: str):
    cid, prop = output.rsplit(".", 1)
    return {"id": cid, "property": prop}


class AllCallbacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("error")
        cls.app = create_app("sim")
        cls.client = cls.app.server.test_client()

    def call(self, key, values):
        spec = self.app.callback_map[key]
        outputs = ([_parse(o) for o in key.strip(".").split("...")] if key.startswith("..")
                   else _parse(key))
        fill = lambda items: [{**i, "value": values.get((i["id"], i["property"]))} for i in items]   # noqa: E731
        inputs = fill(spec["inputs"])
        payload = {"output": key, "outputs": outputs, "inputs": inputs, "state": fill(spec["state"]),
                   "changedPropIds": [f"{inputs[0]['id']}.{inputs[0]['property']}"]}
        resp = self.client.post("/_dash-update-component", json=payload)
        self.assertEqual(resp.status_code, 200, f"{key}: {resp.get_data(as_text=True)[:300]}")
        return json.loads(resp.get_data(as_text=True))["response"]

    def run_all(self, etf_id, theme, lookback):
        an = self.app.analytics
        values = dict(DEFAULTS)
        values.update({("etf", "value"): etf_id, ("theme", "value"): theme,
                       ("lookback", "value"): lookback})
        snap = an.capital_structure()["snap"]
        uni = an.universe_of(etf_id)
        issuer = snap[(snap["universe"] == uni) & ~snap["excluded"]].index[0]
        pair = an.pairs()
        pair = pair[pair["universe"] == uni].iloc[0]
        values[("cr-issuer", "value")] = issuer
        values[("cr-pair", "value")] = f"{pair['a']}|{pair['b']}"
        values[("cr-scatter", "clickData")] = {"points": [{"customdata": [issuer]}]}
        n_figures = 0
        python_callbacks = [k for k, v in self.app.callback_map.items() if "callback" in v]
        self.assertGreaterEqual(len(python_callbacks), 14)
        # the ideas table first, its result feeds the detail callback
        table_key = next(k for k in python_callbacks if "ideas-table.rowData" in k)
        rows = self.call(table_key, values)["ideas-table"]["rowData"]
        self.assertGreater(len(rows), 3)
        for pick in (rows[0], rows[-1]):
            values[("ideas-table", "rowData")] = rows
            values[("ideas-table", "selectedRows")] = [pick]
            for key in python_callbacks:
                if key == table_key:
                    continue
                out = self.call(key, values)
                for comp in out.values():
                    fig = comp.get("figure")
                    if fig is not None:
                        n_figures += 1
                        self.assertTrue(fig["data"] or fig["layout"].get("annotations"), key)
        return n_figures

    def test_high_yield_dark(self):
        self.assertGreaterEqual(self.run_all("JSHY", "dark", "1Y"), 30)

    def test_investment_grade_light_full_history(self):
        self.assertGreaterEqual(self.run_all("GCIG", "light", "3Y"), 30)

    def test_short_lookback(self):
        self.assertGreaterEqual(self.run_all("JSHY", "light", "3M"), 30)

    def test_every_idea_opens(self):
        """The detail panel must open for every idea, whatever its type."""
        key = next(k for k in self.app.callback_map if k.startswith("..id-detail"))
        ideas = self.app.analytics.ideas()
        rows = ideas[["id"]].to_dict("records")
        for row in rows:
            self.call(key, {("ideas-table", "selectedRows"): [row], ("ideas-table", "rowData"): rows,
                            ("theme", "value"): "dark"})

    def test_layout_is_served(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        layout = self.client.get("/_dash-layout")
        self.assertEqual(layout.status_code, 200)
        self.assertIn("ideas-table", layout.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
