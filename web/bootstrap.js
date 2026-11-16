/*
 * Runs the dashboard on a static host.
 *
 * The Dash front end asks a Python server for its layout and for the result of
 * every callback. A static host has no server, so this file answers instead.
 *
 *   1. The layout and the callback map are two files written by the build.
 *   2. A callback whose inputs match a precomputed case is read from
 *      static/cache. That covers every ETF, lookback and theme, and the
 *      one-click variations of each tab.
 *   3. Any other request goes to the project's own Python code, which runs in
 *      the browser inside a web worker (engine_worker.js, on Pyodide).
 *
 * Add ?engine=off to the address to stay on the precomputed views.
 */
(function () {
  "use strict";

  var here = location.pathname.replace(/[^/]*$/, "");
  var STATIC = here + "static/";
  var query = new URLSearchParams(location.search);
  var nativeFetch = window.fetch.bind(window);

  // The Dash front end reads its base path from this block
  var configNode = document.getElementById("_dash-config");
  var config = JSON.parse(configNode.textContent);
  config.requests_pathname_prefix = here;
  config.disable_version_check = true;
  configNode.textContent = JSON.stringify(config);

  var build = nativeFetch(STATIC + "build.json").then(function (r) { return r.json(); });
  var cache = nativeFetch(STATIC + "cache/index.json")
    .then(function (r) { return r.json(); })
    .then(function (index) { return { hit: new Set(index.hit), none: new Set(index.none) }; })
    .catch(function () { return { hit: new Set(), none: new Set() }; });

  // Counters, read by the build check and handy in the console
  var stats = window.cockpit = { engine: "idle", open: 0, hits: 0, misses: 0, ticks: 0, calls: 0,
                                 held: 0, errors: [], missed: [], lastHit: null };

  // ------------------------------------------------------------------ status
  var chip = (function () {
    var style = document.createElement("style");
    style.textContent =
      ".engine-chip{position:fixed;right:16px;bottom:16px;z-index:50;max-width:340px;display:none;" +
      "align-items:flex-start;gap:10px;padding:10px 14px;border-radius:10px;cursor:default;" +
      "font:13px/1.4 system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--text-primary,#f0efec);" +
      "background:var(--surface-2,#1c1c1b);border:1px solid var(--border,rgba(255,255,255,.1));" +
      "box-shadow:0 6px 24px rgba(0,0,0,.28)}" +
      ".engine-chip.show{display:flex}" +
      ".engine-chip i{flex:none;width:8px;height:8px;margin-top:5px;border-radius:50%;background:var(--accent,#3987e5)}" +
      ".engine-chip.loading i{animation:engine-pulse 1.2s ease-in-out infinite}" +
      ".engine-chip.ready i{background:var(--good,#0ca30c)}" +
      ".engine-chip.failed i{background:var(--warning,#fab219)}" +
      ".engine-chip small{display:block;margin-top:2px;font-size:12px;color:var(--text-muted,#898781)}" +
      "@keyframes engine-pulse{50%{opacity:.25}}";
    document.head.appendChild(style);

    var node = document.createElement("div");
    node.className = "engine-chip";
    node.setAttribute("role", "status");
    node.title = "Click to hide";
    node.innerHTML = "<i></i><span><b></b><small></small></span>";
    node.addEventListener("click", function () { node.classList.remove("show"); });
    document.body.appendChild(node);

    var timer = null;
    function chip(kind, title, detail, hideAfter) {
      clearTimeout(timer);
      node.className = "engine-chip show " + kind;
      node.querySelector("b").textContent = title;
      node.querySelector("small").textContent = detail || "";
      if (hideAfter) timer = setTimeout(chip.hide, hideAfter);
    }
    chip.hide = function () { clearTimeout(timer); node.classList.remove("show"); };
    return chip;
  })();

  // ------------------------------------------------------------------ engine
  var engine = (function () {
    var worker = null, pending = {}, seq = 0, step = "", settle;
    var ready = new Promise(function (resolve) { settle = resolve; });
    var off = query.get("engine") === "off";
    // Small devices load the engine only when a control needs it
    var lazy = (navigator.deviceMemory && navigator.deviceMemory < 4) || /Mobi|Android/i.test(navigator.userAgent);
    if (off) { stats.engine = "off"; settle(false); }

    function show() {
      if (stats.engine !== "loading") return;
      if (stats.held > 0) chip("loading", "Your change applies as soon as the engine is loaded", step);
      else chip("loading", "Loading the Python engine in your browser", step + " Default views already work.");
    }

    function fail(reason) {
      if (stats.engine === "ready" || stats.engine === "failed") return;
      stats.engine = "failed";
      stats.errors.push(String(reason));
      console.error("Engine failed to load.", reason);
      chip("failed", "The Python engine could not load here", "The page shows its precomputed views only.");
      settle(false);
    }

    function start() {
      if (stats.engine !== "idle") return;
      stats.engine = "loading";
      step = "Step 1 of 4.";
      show();
      build.then(function (info) {
        worker = new Worker(STATIC + "engine_worker.js?v=" + info.id, { type: "module" });
        worker.onerror = function (event) { fail(event.message || "worker error"); };
        worker.onmessage = function (event) {
          var msg = event.data;
          if (msg.type === "step") { step = "Step " + msg.n + " of 4."; show(); }
          else if (msg.type === "ready") {
            stats.engine = "ready";
            stats.seconds = msg.seconds;
            chip("ready", "Engine ready", "Every control now recalculates in your browser.", 7000);
            settle(true);
          }
          else if (msg.type === "failed") fail(msg.error);
          else if (msg.type === "result" && pending[msg.id]) { pending[msg.id](msg); delete pending[msg.id]; }
        };
        worker.postMessage({
          type: "init", base: new URL(STATIC, location.href).href, asof: info.asof,
          pyodide: query.get("pyodide") || info.pyodide, packages: info.packages
        });
      }).catch(fail);
    }

    function call(body) {
      return new Promise(function (resolve) {
        var id = ++seq;
        pending[id] = resolve;
        worker.postMessage({ type: "call", id: id, body: body });
      });
    }

    if (!off && !lazy) window.addEventListener("load", function () { setTimeout(start, 1500); });
    stats.ask = call;          // sends a request body straight to the engine, for checks
    return { ready: ready, start: start, call: call, show: show, lazy: lazy };
  })();

  // --------------------------------------------------------------- callbacks
  var still = {};          // last frozen frame shown, by callback output

  function reply(status, text) {
    return new Response(status === 204 ? null : text,
                        { status: status, headers: { "Content-Type": "application/json" } });
  }

  // Without the engine a clock-driven panel keeps the frame the build recorded
  async function restingFrame(body, known) {
    var output = JSON.parse(body).output;
    var key = await cockpitKey(body, true);
    if (known.hit.has(key) && still[output] !== key) {
      still[output] = key;
      return nativeFetch(STATIC + "cache/" + key + ".json");
    }
    return reply(204);
  }

  async function answer(body) {
    var known = await cache;
    var key = await cockpitKey(body, false);
    if (known.hit.has(key)) {
      stats.hits++;
      stats.lastHit = { key: key, body: body };
      still[JSON.parse(body).output] = key;
      return nativeFetch(STATIC + "cache/" + key + ".json");
    }
    if (known.none.has(key)) { stats.hits++; return reply(204); }

    var moving = cockpitMoving(body);
    if (moving) stats.ticks++;
    else { stats.misses++; if (stats.missed.length < 20) stats.missed.push(body.slice(0, 600)); }
    if (stats.engine !== "ready") {
      if (moving) {                       // the intraday clock ticks before the engine is up
        if (!engine.lazy) engine.start();
        return restingFrame(body, known);
      }
      engine.start();
      stats.held++; engine.show();
      var up = await engine.ready;
      stats.held--; engine.show();
      if (!up) return restingFrame(body, known);
    }

    stats.calls++;
    var shown = false;
    var busy = moving ? null : setTimeout(function () {
      shown = true;
      chip("loading", "Calculating in your browser", "");
    }, 400);
    var out = await engine.call(body);
    clearTimeout(busy);
    if (shown) chip.hide();
    if (out.status >= 400) stats.errors.push(out.text.slice(0, 300));
    return reply(out.status, out.text);
  }

  window.fetch = function (input, init) {
    var url = typeof input === "string" ? input : input.url;
    var path = new URL(url, location.href).pathname;
    if (/\/_dash-layout$/.test(path)) return nativeFetch(STATIC + "layout.json");
    if (/\/_dash-dependencies$/.test(path)) return nativeFetch(STATIC + "dependencies.json");
    if (/\/_dash-update-component$/.test(path)) {
      var body = init && typeof init.body === "string" ? Promise.resolve(init.body) : input.clone().text();
      stats.open++;
      return body.then(answer).finally(function () { stats.open--; });
    }
    return nativeFetch(input, init);
  };
})();
