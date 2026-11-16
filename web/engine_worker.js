/*
 * The dashboard's Python code, running in the browser.
 *
 * This worker loads Pyodide, unpacks the project and its pure Python
 * dependencies, builds the same Dash app as `python app.py --mode sim` and
 * answers callback requests through Flask's test client. Nothing leaves the
 * visitor's machine and no server is involved.
 *
 * Messages in    {type: "init", base, asof, pyodide, packages}
 *                {type: "call", id, body}
 * Messages out   {type: "step", n}          progress, four steps
 *                {type: "ready", seconds}
 *                {type: "failed", error}
 *                {type: "result", id, status, text}
 */

const BOOT = `
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path[:0] = ["/cockpit/app", "/cockpit/deps"]
os.chdir("/cockpit/app")

import app as cockpit

_dash = cockpit.create_app("sim")
_client = _dash.server.test_client()


def dispatch(body):
    """One callback request in, status and JSON text out."""
    reply = _client.post("/_dash-update-component", data=body, content_type="application/json")
    return str(reply.status_code) + "\\n" + reply.get_data(as_text=True)


dispatch
`;

let dispatch = null;

async function archive(url) {
  const reply = await fetch(url);
  if (!reply.ok) throw new Error(url + " answered " + reply.status);
  return reply.arrayBuffer();
}

async function boot(setup) {
  const started = performance.now();
  const step = (n) => self.postMessage({ type: "step", n });

  step(1);                                              // the Python runtime
  const { loadPyodide } = await import(setup.pyodide + "pyodide.mjs");
  const pyodide = await loadPyodide({
    indexURL: setup.pyodide,
    env: { HOME: "/cockpit", ETF_DASH_DATA_MODE: "sim", ETF_DASH_ASOF: setup.asof },
  });

  step(2);                                              // numpy, pandas, scipy and friends
  const code = Promise.all([archive(setup.base + "engine/pydeps.zip"),
                            archive(setup.base + "engine/app.zip")]);
  await pyodide.loadPackage(setup.packages);

  step(3);                                              // Dash, Plotly and the project
  const [deps, project] = await code;
  pyodide.FS.mkdirTree("/cockpit/deps");
  pyodide.FS.mkdirTree("/cockpit/app");
  pyodide.unpackArchive(deps, "zip", { extractDir: "/cockpit/deps" });
  pyodide.unpackArchive(project, "zip", { extractDir: "/cockpit/app" });

  step(4);                                              // simulate the market, build the app
  dispatch = pyodide.runPython(BOOT);
  return Math.round((performance.now() - started) / 100) / 10;
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (msg.type === "init") {
    try {
      self.postMessage({ type: "ready", seconds: await boot(msg) });
    } catch (error) {
      self.postMessage({ type: "failed", error: String((error && error.stack) || error) });
    }
  } else if (msg.type === "call") {
    let status = 500, text = "";
    try {
      const out = dispatch(msg.body);
      const cut = out.indexOf("\n");
      status = Number(out.slice(0, cut));
      text = out.slice(cut + 1);
    } catch (error) {
      text = String((error && error.message) || error);
    }
    self.postMessage({ type: "result", id: msg.id, status, text });
  }
};
