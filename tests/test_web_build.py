"""
Tests of the static web build, without a browser.

The recording of the precomputed views and the check of the built site need
Chromium, so they run in the web build workflow. What is tested here is what
the page relies on: a front end with relative addresses, an engine archive
that Pyodide can load, and a simulated history that ends on the pinned day.
"""
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_site                     # noqa: E402
from app import create_app            # noqa: E402
from data.synthetic import SyntheticMarket   # noqa: E402


class FrontEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        build_site.export_front_end(create_app("sim"), cls.out)
        cls.html = (cls.out / "index.html").read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_addresses_are_relative(self):
        self.assertNotIn('src="/', self.html)
        self.assertNotIn('href="/', self.html)

    def test_every_file_of_the_page_is_there(self):
        import re
        for url in re.findall(r'(?:src|href)="([^"]+)"', self.html):
            self.assertTrue((self.out / url.split("?")[0]).is_file(), url)

    def test_bootstrap_loads_before_the_dash_bundles(self):
        config = self.html.index('id="_dash-config"')
        bootstrap = self.html.index("static/bootstrap.js")
        first_bundle = self.html.index("_dash-component-suites/")
        self.assertLess(config, bootstrap)
        self.assertLess(bootstrap, first_bundle)

    def test_layout_and_callback_map_are_saved(self):
        for name in ("layout.json", "dependencies.json", "engine_worker.js", "cachekey.js"):
            self.assertGreater((self.out / "static" / name).stat().st_size, 500, name)


class Engine(unittest.TestCase):
    def test_bundle_is_pure_python_and_complete(self):
        names = {build_site._norm(d.name) for d in build_site.bundled_distributions()}
        self.assertLessEqual({"dash", "dash-ag-grid", "plotly", "flask", "werkzeug"}, names)
        self.assertFalse(names & build_site.NOT_BUNDLED)

    def test_archives_hold_the_project_and_dash(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_site.write_engine(Path(tmp))
            engine = Path(tmp) / "static" / "engine"
            with zipfile.ZipFile(engine / "app.zip") as zf:
                self.assertLessEqual({"app.py", "config.py", "engine/analytics.py", "ui/callbacks.py"},
                                     set(zf.namelist()))
            with zipfile.ZipFile(engine / "pydeps.zip") as zf:
                files = {i.filename: i.file_size for i in zf.infolist()}
            self.assertIn("dash/__init__.py", files)
            self.assertIn("plotly/graph_objects/__init__.py", files)
            bundles = [n for n in files if n.endswith(".js")]
            self.assertTrue(bundles)                       # kept so Dash can date them
            self.assertEqual(sum(files[n] for n in bundles), 0)       # but empty


class PinnedDay(unittest.TestCase):
    def test_history_ends_on_the_pinned_day(self):
        market = SyntheticMarket(seed=3, n_days=120, end="2026-03-13")
        self.assertEqual(str(market.dates[-1].date()), "2026-03-13")

    def test_environment_variable_reaches_the_provider(self):
        code = ("from data.provider import DataProvider; "
                "print(DataProvider(mode='sim').sim.dates[-1].date())")
        env = dict(os.environ, ETF_DASH_ASOF="2026-03-13")
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True,
                             capture_output=True, check=True).stdout
        self.assertEqual(out.strip(), "2026-03-13")


if __name__ == "__main__":
    unittest.main()
