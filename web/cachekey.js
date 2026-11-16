/*
 * One callback request, one cache key.
 *
 * The page and the build both load this file, so a request recorded at build
 * time and the same request made by a visitor always get the same name.
 * The key covers what the callback computes from (its output and the values
 * of its inputs) and leaves out what only says which control moved.
 */
(function (root) {
  "use strict";

  // Inputs that move on their own. A frozen key ignores them.
  var VOLATILE = { "tick.n_intervals": true };

  function name(item) {
    return (typeof item.id === "string" ? item.id : JSON.stringify(item.id)) + "." + item.property;
  }

  function values(items, freeze) {
    return (items || []).map(function (item) {
      var held = freeze && VOLATILE[name(item)];
      return [name(item), held || item.value === undefined ? null : item.value];
    });
  }

  function hex(buffer) {
    var bytes = new Uint8Array(buffer), out = "";
    for (var i = 0; i < 12; i++) out += (bytes[i] < 16 ? "0" : "") + bytes[i].toString(16);
    return out;
  }

  // freeze = true gives the key of the same request with its clock inputs at rest
  root.cockpitKey = async function (bodyText, freeze) {
    var body = JSON.parse(bodyText);
    var text = JSON.stringify([body.output, values(body.inputs, freeze), values(body.state, freeze)]);
    return hex(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  };

  // True when a clock input has already moved in this request
  root.cockpitMoving = function (bodyText) {
    var body = JSON.parse(bodyText);
    return (body.inputs || []).concat(body.state || []).some(function (item) {
      return VOLATILE[name(item)] && item.value !== undefined && item.value !== null;
    });
  };
})(self);
