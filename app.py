"""
Entry point of the dashboard.

    python app.py                 runs the app on http://127.0.0.1:8050
    python app.py --mode sim      forces the simulator, without LSEG
    python app.py --port 8060     changes the port
    python app.py --no-browser    does not open the browser

Three layers.
    data/    LSEG access, simulator, hybrid provider that falls back block by block
    engine/  the quantitative work, with no dependency on the interface
    ui/      Dash layout, Plotly figures, callbacks
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser

from dash import Dash

import config as cfg
from data.provider import DataProvider
from engine.analytics import Analytics
from ui.callbacks import register_callbacks
from ui.layout import build_layout


def create_app(mode: str = cfg.DATA_MODE) -> Dash:
    """Wires data, engine and interface together. The tests use it too."""
    provider = DataProvider(mode=mode)
    analytics = Analytics(provider)
    app = Dash(__name__, title=cfg.APP_TITLE, update_title=None,
               suppress_callback_exceptions=True)
    app.layout = build_layout()
    register_callbacks(app, analytics)
    app.analytics = analytics            # kept for tests and debugging
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=cfg.APP_TITLE)
    parser.add_argument("--mode", default=cfg.DATA_MODE, choices=["auto", "lseg", "sim"],
                        help="auto uses LSEG when it is available, the simulator otherwise (default)")
    parser.add_argument("--host", default=cfg.HOST)
    parser.add_argument("--port", type=int, default=cfg.PORT)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Dash hot reload")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):       # Windows consoles, no crash on an odd character
        sys.stdout.reconfigure(errors="replace")
    t0 = time.time()
    app = create_app(args.mode)
    provider = app.analytics.p
    for etf_id in cfg.ETFS:              # warm up the data and the heavy calculations
        app.analytics.basket(etf_id)
        app.analytics.var(etf_id)
    app.analytics.ideas()

    url = f"http://{args.host}:{args.port}"
    print("=" * 72)
    print(f" {cfg.APP_TITLE}")
    print(f" Data mode {provider.mode_label()}")
    for row in provider.status_table().itertuples(index=False):
        print(f"   [{row.Source:>4}] {row.Block} · {row.Detail}")
    print(f" Ready in {time.time() - t0:.1f} s  ->  {url}   (Ctrl+C to stop)")
    print("=" * 72)

    if not args.no_browser and not args.debug:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
