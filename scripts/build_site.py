"""
Builds the static web version of the dashboard into site/.

    python scripts/build_site.py                   build
    python scripts/build_site.py --check           build, then open the result in a headless browser
    python scripts/build_site.py --check-engine    same, and wait for the Python engine to answer

A Dash app needs a Python server and a static host such as GitHub Pages has
none. The build prepares three things, and web/bootstrap.js ties them together
on the page.

    1. Front end          index.html, the JavaScript bundles and the stylesheet,
                          copied from the running app.
    2. Precomputed views  every callback answer for each ETF, lookback and theme,
                          then the one-click variations of each tab. They are
                          recorded from a real browser session, so the page asks
                          for them with the very same request.
    3. Engine             the project's Python code and its pure Python
                          dependencies, zipped for Pyodide. In the browser it
                          answers whatever was not precomputed.

Needs the packages of requirements-web.txt and a Chromium (playwright install chromium).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import functools
import hashlib
import http.server
import importlib.metadata as md
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
sys.path.insert(0, str(ROOT))

# The Pyodide release the page loads. Its numpy, pandas and scipy are the ones
# pinned in requirements-web.txt, so both sides compute with the same libraries.
PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.8/full/"
# Compiled packages, which Pyodide ships and loads by name
PYODIDE_PACKAGES = ["numpy", "pandas", "scipy", "pydantic", "jinja2", "markupsafe", "click",
                    "narwhals", "packaging", "typing-extensions"]
# What those bring along, and what Dash lists but only uses in a notebook or at install time
NOT_BUNDLED = set(PYODIDE_PACKAGES) | {
    "pydantic-core", "annotated-types", "typing-inspection", "python-dateutil", "pytz", "six",
    "tzdata", "requests", "setuptools"}
# The pure Python packages bundled with the page are these and all they require
BUNDLE_ROOTS = ["dash", "dash-ag-grid", "plotly"]
# The interpreter the page runs, to tell which optional requirements apply there
BROWSER_PYTHON = {"python_version": "3.14", "python_full_version": "3.14.2", "extra": "",
                  "sys_platform": "emscripten", "platform_system": "Emscripten"}
# Files of those packages the Python side never reads
DROP_SUFFIXES = (".js", ".map", ".css", ".ts", ".tsx", ".pyc", ".pyi", ".woff", ".woff2", ".ttf",
                 ".svg", ".png", ".gif", ".ico", ".gz", ".md")
# Dash looks up the date of its bundles to version their addresses, so these stay as empty files
KEEP_EMPTY = (".js", ".css")

PROJECT_FILES = ["app.py", "config.py", "data", "engine", "ui", "assets"]
GLOBAL_CONTROLS = ("theme", "etf", "lookback")
DESCRIPTION = ("Fixed income ETF capital markets dashboard. Premium and discount, AP arbitrage, "
               "basket liquidity, credit relative value, curves, risk and trade ideas.")


def log(text: str) -> None:
    print(text, flush=True)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# ------------------------------------------------------------------ front end
def export_front_end(app, out: Path) -> None:
    """Saves the page and every file it loads, with paths made relative."""
    import dash
    client = app.server.test_client()

    def fetch(url: str) -> bytes:
        reply = client.get(url)
        try:
            if reply.status_code != 200:
                raise RuntimeError(f"{url} answered {reply.status_code}")
            return reply.get_data()
        finally:
            reply.close()

    def save(url: str) -> None:
        target = out / url.split("?")[0].lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fetch(url))

    html = fetch("/").decode("utf-8")
    for url in sorted(set(re.findall(r'(?:src|href)="(/[^"]+)"', html))):
        save(url)
    # Chunks the bundles fetch on demand. MathJax is two megabytes nobody asks for here.
    suites = "/_dash-component-suites/"
    for chunk in sorted(Path(dash.dcc.__file__).parent.glob("async-*.js")):
        if "mathjax" not in chunk.name:
            save(f"{suites}dash/dcc/{chunk.name}")
    save(f"{suites}dash_ag_grid/async-community.js")
    save(f"{suites}plotly/package_data/plotly.min.js")

    html = re.sub(r'((?:src|href)=")/', r"\1", html)
    html = html.replace("<title>", f'<meta name="description" content="{DESCRIPTION}">\n        <title>', 1)
    tags = '\n            <script src="static/cachekey.js"></script>' \
           '\n            <script src="static/bootstrap.js"></script>'
    html, n = re.subn(r'(<script id="_dash-config".*?</script>)', lambda m: m.group(1) + tags, html,
                      count=1, flags=re.S)
    if n != 1:
        raise RuntimeError("the Dash config block was not found in the page")
    (out / "index.html").write_text(html, encoding="utf-8")

    static = out / "static"
    static.mkdir(parents=True, exist_ok=True)
    (static / "layout.json").write_bytes(fetch("/_dash-layout"))
    (static / "dependencies.json").write_bytes(fetch("/_dash-dependencies"))
    for name in ("cachekey.js", "bootstrap.js", "engine_worker.js"):
        shutil.copyfile(WEB / name, static / name)
    (out / ".nojekyll").write_text("")


# --------------------------------------------------------------------- engine
def bundled_distributions() -> list:
    """BUNDLE_ROOTS and what they require in the browser, minus what Pyodide provides."""
    from packaging.requirements import Requirement
    seen, todo = {}, list(BUNDLE_ROOTS)
    while todo:
        name = _norm(todo.pop())
        if name in seen or name in NOT_BUNDLED:
            continue
        dist = md.distribution(name)
        seen[name] = dist
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker is None or req.marker.evaluate(BROWSER_PYTHON):
                todo.append(req.name)
    return [seen[k] for k in sorted(seen)]


def write_engine(out: Path) -> dict:
    """Zips the project and its pure Python dependencies for the browser."""
    target = out / "static" / "engine"
    target.mkdir(parents=True, exist_ok=True)
    versions = {}
    with zipfile.ZipFile(target / "pydeps.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for dist in bundled_distributions():
            files = [f for f in (dist.files or []) if ".." not in f.parts]
            if any(f.suffix in (".so", ".pyd") for f in files):
                raise RuntimeError(f"{dist.name} is compiled. If Pyodide ships it, add it to "
                                   f"PYODIDE_PACKAGES, otherwise the page cannot load it.")
            versions[dist.name] = dist.version
            for f in files:
                if "__pycache__" in f.parts or f.name == "RECORD":
                    continue
                if f.parts[:3] == ("plotly", "package_data", "datasets"):
                    continue
                if f.suffix in DROP_SUFFIXES:
                    if f.suffix in KEEP_EMPTY:
                        zf.writestr(f.as_posix(), "")
                    continue
                path = f.locate()
                if path.is_file():
                    zf.write(path, f.as_posix())
    with zipfile.ZipFile(target / "app.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for entry in PROJECT_FILES:
            path = ROOT / entry
            for f in ([path] if path.is_file() else sorted(path.rglob("*"))):
                if f.is_file() and "__pycache__" not in f.parts:
                    zf.write(f, f.relative_to(ROOT).as_posix())
    return versions


# -------------------------------------------------------------------- browser
def _browser(playwright, chrome: str | None):
    if chrome:
        return playwright.chromium.launch(executable_path=chrome, args=["--no-sandbox"])
    return playwright.chromium.launch()


class _Session:
    """A page of the dashboard, with a way to wait until its callbacks are done."""

    def __init__(self, page, on_reply=None, static=False):
        self.page, self.flying, self.on_reply, self.static = page, set(), on_reply, static
        self.problems: list = []
        page.on("request", self._request)
        page.on("requestfinished", self._finished)
        page.on("requestfailed", self._failed)
        page.on("response", self._response)
        page.on("pageerror", lambda e: self.problems.append(f"page error {str(e)[:300]}"))
        page.on("console", lambda m: self.problems.append(f"console {m.text[:300]}")
                if m.type == "error" else None)

    def _request(self, request):
        if request.method == "POST" and "_dash-update-component" in request.url:
            self.flying.add(request)

    def _finished(self, request):
        if request in self.flying:
            if self.on_reply:
                reply = request.response()
                self.on_reply(request.post_data, reply.status, reply.text() if reply.status == 200 else "")
            self.flying.discard(request)

    def _failed(self, request):
        self.flying.discard(request)
        self.problems.append(f"failed {request.url[-80:]}")

    def _response(self, reply):
        if reply.status >= 400:
            self.problems.append(f"{reply.status} {reply.url[-80:]}")

    def settle(self, quiet_ms: int = 450, timeout_s: float = 90.0) -> None:
        """Returns once no callback has been in flight for quiet_ms."""
        end, calm = time.time() + timeout_s, 0
        while calm < quiet_ms:
            self.page.wait_for_timeout(75)
            # On the static page callbacks never reach the network, the page counts them itself
            busy = self.flying or (self.static and self.page.evaluate("window.cockpit.open") > 0)
            calm = calm + 75 if not busy else 0
            if time.time() > end:
                raise TimeoutError("callbacks did not settle")

    def set(self, component: str, **props) -> None:
        """Changes a control the way a visitor would, callbacks included."""
        self.page.evaluate("([id, props]) => dash_clientside.set_props(id, props)", [component, props])
        self.settle()


def _radio_values(layout, skip=()) -> dict:
    """Every RadioItems of the layout, as {id: (default, [values])}."""
    found = {}
    for node in layout._traverse():
        if type(node).__name__ == "RadioItems" and getattr(node, "id", None) not in skip:
            found[node.id] = (node.value, [o["value"] if isinstance(o, dict) else o for o in node.options])
    return found


def _tab_of(layout, label: str):
    return next(t.value for t in layout["tabs"].children if t.label == label)


# ---------------------------------------------------------- precomputed views
def record_views(app, out: Path, chrome: str | None) -> dict:
    """Walks the controls in a real browser and keeps every callback answer."""
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    layout = app.layout
    radios = _radio_values(layout)
    themes, etfs, lookbacks = (radios[c][1] for c in GLOBAL_CONTROLS)
    defaults = {c: radios[c][0] for c in GLOBAL_CONTROLS}
    layout["tick"].disabled = True        # the intraday clock stays at rest while recording

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cache = out / "static" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    fresh, hit, none = [], set(), set()

    try:
        with sync_playwright() as p:
            browser = _browser(p, chrome)
            page = browser.new_page(viewport={"width": 1500, "height": 950})
            page.add_init_script(path=str(WEB / "cachekey.js"))
            session = _Session(page, lambda body, status, text: fresh.append((body, status, text)))

            def keep() -> None:
                """Names the answers recorded since the last call and writes them."""
                if not fresh:
                    return
                batch = fresh[:]
                del fresh[:]
                keys = page.evaluate("(bodies) => Promise.all(bodies.map(b => cockpitKey(b, false)))",
                                     [b for b, _, _ in batch])
                for key, (_, status, text) in zip(keys, batch):
                    if status == 200:
                        hit.add(key)
                        (cache / f"{key}.json").write_text(text, encoding="utf-8")
                    elif status == 204:
                        none.add(key)

            page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")
            session.settle()
            keep()

            # 1. Every ETF, lookback and theme, with each tab on its default settings
            for theme in themes:
                session.set("theme", value=theme)
                for etf in etfs:
                    session.set("etf", value=etf)
                    for lookback in lookbacks:
                        session.set("lookback", value=lookback)
                        keep()
                log(f"  theme {theme} recorded, {len(hit)} views")

            # 2. One click away from the defaults, each choice of each tab, for both ETFs and themes
            for name in GLOBAL_CONTROLS:
                session.set(name, value=defaults[name])
            local = {k: v for k, v in radios.items() if k not in GLOBAL_CONTROLS}
            for theme in themes:
                session.set("theme", value=theme)
                for etf in etfs:
                    session.set("etf", value=etf)
                    for name, (default, choices) in local.items():
                        for value in choices:
                            if value != default:
                                session.set(name, value=value)
                        session.set(name, value=default)
                    keep()
            log(f"  tab choices recorded, {len(hit)} views")

            # 3. Each trade idea, opened from the table
            session.set("tabs", value=_tab_of(layout, "Trade Ideas"))
            session.set("etf", value=defaults["etf"])
            for theme in themes:
                session.set("theme", value=theme)
                rows = page.evaluate("async () => (await dash_ag_grid.getApiAsync('ideas-table'))"
                                     ".getDisplayedRowCount()")
                for i in list(range(1, rows)) + [0]:
                    page.evaluate("async (i) => (await dash_ag_grid.getApiAsync('ideas-table'))"
                                  ".getDisplayedRowAtIndex(i).setSelected(true, true)", i)
                    session.settle(quiet_ms=250)
                keep()
            log(f"  trade ideas recorded, {len(hit)} views")

            problems = [x for x in session.problems if "favicon" not in x]
            browser.close()
    finally:
        server.shutdown()
        layout["tick"].disabled = False
    if problems:
        raise RuntimeError("the recording session met errors:\n  " + "\n  ".join(problems[:10]))

    (cache / "index.json").write_text(json.dumps({"hit": sorted(hit), "none": sorted(none)}))
    size = sum(f.stat().st_size for f in cache.glob("*.json"))
    return {"views": len(hit), "megabytes": round(size / 1e6, 1)}


# ---------------------------------------------------------------------- check
def _compare(a, b, path: str = "", notes: list | None = None) -> float:
    """
    Largest relative difference between two callback answers.

    Plotly ships arrays as base64 text, so those are decoded before they are
    compared. Anything that is not a number has to match exactly, and what
    does not is listed in notes.
    """
    notes = [] if notes is None else notes

    def differ(what: str) -> float:
        if len(notes) < 8:
            notes.append(f"{path or 'root'} {what}")
        return float("inf")

    if isinstance(a, dict) and isinstance(b, dict):
        if {"dtype", "bdata"} <= a.keys() and {"dtype", "bdata"} <= b.keys():
            import numpy as np
            x, y = (np.frombuffer(base64.b64decode(v["bdata"]), dtype=v["dtype"]).astype(float)
                    for v in (a, b))
            if x.shape != y.shape:
                return differ(f"arrays of {x.size} and {y.size} values")
            scale = np.maximum(1.0, np.maximum(np.abs(x), np.abs(y)))
            gaps = np.abs(x - y) / scale
            return float(np.nanmax(gaps)) if gaps.size and not np.all(np.isnan(gaps)) else 0.0
        if a.keys() != b.keys():
            return differ(f"keys {sorted(a.keys() ^ b.keys())[:6]}")
        return max((_compare(a[k], b[k], f"{path}.{k}", notes) for k in a), default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return differ(f"lists of {len(a)} and {len(b)} items")
        return max((_compare(x, y, f"{path}[{i}]", notes) for i, (x, y) in enumerate(zip(a, b))),
                   default=0.0)
    numbers = (int, float)
    if isinstance(a, numbers) and isinstance(b, numbers) and not isinstance(a, bool) \
            and not isinstance(b, bool):
        return abs(a - b) / max(1.0, abs(a), abs(b))
    return 0.0 if a == b else differ(f"{str(a)[:60]!r} against {str(b)[:60]!r}")


def check_site(out: Path, layout, chrome: str | None, with_engine: bool) -> None:
    """Opens the built site and fails loudly if anything is missing."""
    from playwright.sync_api import sync_playwright

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

    # Served from a sub-folder, the way GitHub Pages serves a project
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                             functools.partial(Quiet, directory=str(out.parent)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}/{out.name}/"
    radios = _radio_values(layout)
    charts = {tab.value: sum(type(n).__name__ == "Graph" for n in tab._traverse())
              for tab in layout["tabs"].children}
    stat = lambda page, key: page.evaluate("(k) => window.cockpit[k]", key)   # noqa: E731

    try:
        with sync_playwright() as p:
            browser = _browser(p, chrome)

            # Precomputed views alone
            page = browser.new_page(viewport={"width": 1500, "height": 950})
            session = _Session(page, static=True)
            page.goto(base + "?engine=off", wait_until="networkidle")
            session.settle()
            for theme in radios["theme"][1]:
                session.set("theme", value=theme)
                for etf in radios["etf"][1]:
                    session.set("etf", value=etf)
                    for lookback in radios["lookback"][1]:
                        session.set("lookback", value=lookback)
            drawn = 0
            for tab, expected in charts.items():
                session.set("tabs", value=tab)
                shown = page.locator(".dash-graph:visible .js-plotly-plot")
                for _ in range(100):
                    if shown.count() >= expected:
                        break
                    page.wait_for_timeout(100)
                drawn += shown.count()
            hits, misses = stat(page, "hits"), stat(page, "misses")
            if session.problems:
                raise RuntimeError("the static page met errors:\n  " + "\n  ".join(session.problems[:10]))
            if misses or drawn != sum(charts.values()):
                raise RuntimeError(f"static page incomplete: {misses} requests not precomputed, "
                                   f"{drawn} of {sum(charts.values())} charts drawn\n  " +
                                   "\n  ".join(stat(page, "missed")))
            log(f"  static page checked, {hits} requests served from the precomputed views, "
                f"{drawn} charts drawn")
            page.close()

            # The engine, which needs the network to fetch Pyodide
            if with_engine:
                page = browser.new_page(viewport={"width": 1500, "height": 950})
                session = _Session(page, static=True)
                page.goto(base, wait_until="networkidle")
                try:
                    page.wait_for_function("window.cockpit.engine !== 'idle' && "
                                           "window.cockpit.engine !== 'loading'", timeout=420_000)
                except Exception:
                    raise RuntimeError("the engine was still loading after seven minutes") from None
                if stat(page, "engine") != "ready":
                    raise RuntimeError("the engine failed to load:\n" + "\n".join(stat(page, "errors")))
                seconds = stat(page, "seconds")
                # A precomputed view, asked again to the engine, must come back the same
                both = page.evaluate("""async () => {
                    const { key, body } = window.cockpit.lastHit;
                    const live = await window.cockpit.ask(body);
                    const kept = await (await fetch("static/cache/" + key + ".json")).text();
                    return [live.status, live.text, kept];
                }""")
                log(f"  engine ready in {seconds} s")
                notes: list = []
                gap = _compare(json.loads(both[1]), json.loads(both[2]), "", notes) if both[0] == 200 \
                    else float("inf")
                if gap > 1e-6:
                    raise RuntimeError(f"engine and precomputed view disagree, status {both[0]}, "
                                       f"gap {gap}\n  " + "\n  ".join(notes or [both[1][:400]]))
                session.set("ap-stress", value=1.7)
                session.set("lq-part", value=30)
                session.set("id-conv", value=3)
                calls, errors = stat(page, "calls"), stat(page, "errors")
                if errors or calls < 3 or session.problems:
                    raise RuntimeError("the engine answered badly:\n  " +
                                       "\n  ".join((errors + session.problems)[:10]))
                log(f"  engine checked, ready in {seconds} s, {calls} requests answered in the browser, "
                    f"largest gap to a precomputed view {gap:.1e}")
                page.close()
            browser.close()
    finally:
        server.shutdown()


# ----------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Static web build of the dashboard")
    parser.add_argument("--out", default=str(ROOT / "site"))
    parser.add_argument("--asof", default=dt.date.today().isoformat(),
                        help="last day of the simulated history, today by default")
    parser.add_argument("--chrome", default=os.getenv("CHROME_PATH"),
                        help="path of a Chrome or Chromium binary, Playwright's own by default")
    parser.add_argument("--check", action="store_true", help="open the built site and verify it")
    parser.add_argument("--check-engine", action="store_true",
                        help="also wait for the Python engine in the browser (needs the network)")
    args = parser.parse_args()

    os.environ["ETF_DASH_ASOF"] = args.asof
    os.environ["ETF_DASH_DATA_MODE"] = "sim"
    from app import create_app            # after the environment is set

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    started = time.time()

    log(f"Building the web version, simulated history ending {args.asof}")
    app = create_app("sim")
    export_front_end(app, out)
    log("  front end exported")
    versions = write_engine(out)
    log(f"  engine bundled, {len(versions)} pure Python packages")
    views = record_views(app, out, args.chrome)
    log(f"  {views['views']} precomputed views, {views['megabytes']} MB")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    info = {"asof": args.asof, "built": stamp, "pyodide": PYODIDE, "packages": PYODIDE_PACKAGES,
            "bundled": versions, "views": views["views"],
            "id": hashlib.sha256(f"{stamp}{sorted(versions.items())}".encode()).hexdigest()[:10]}
    (out / "static" / "build.json").write_text(json.dumps(info, indent=1))

    if args.check or args.check_engine:
        check_site(out, app.layout, args.chrome, args.check_engine)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log(f"Done in {time.time() - started:.0f} s, {total / 1e6:.0f} MB in {out}")


if __name__ == "__main__":
    main()
