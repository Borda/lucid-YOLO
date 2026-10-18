// SPDX-License-Identifier: Apache-2.0
//
// Copy a ```pycon example as code that runs: keep the source after each `>>> ` or
// `... ` prompt, drop the prompts and every output line. Material's copy button hands
// clipboard.js the button's `data-clipboard-text` when it is set and the code block's
// text otherwise, so a capture-phase click listener -- which fires before clipboard.js
// sees the click -- only has to set that attribute on a pycon block's button. A manual
// selection copied from inside a pycon block is rewritten the same way.
(function () {
  "use strict";

  var PROMPT = /^(>>>|\.\.\.)( |$)/;
  var PYCON = ".language-pycon";

  function runnable(text) {
    var kept = [];
    text.split("\n").forEach(function (line) {
      var prompt = PROMPT.exec(line);
      if (prompt) kept.push(line.slice(prompt[0].length));
    });
    return kept.join("\n").replace(/\s+$/, "");
  }

  function pyconCode(button) {
    var target = button.getAttribute("data-clipboard-target");
    var code = target ? document.querySelector(target) : null;
    return code && code.closest(PYCON) ? code : null;
  }

  document.addEventListener(
    "click",
    function (event) {
      var button = event.target.closest("[data-clipboard-target]");
      var code = button && pyconCode(button);
      if (code) button.setAttribute("data-clipboard-text", runnable(code.textContent));
    },
    true
  );

  document.addEventListener("copy", function (event) {
    var selection = window.getSelection();
    if (!selection || selection.rangeCount === 0 || !event.clipboardData) return;
    var node = selection.getRangeAt(0).commonAncestorContainer;
    var element = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
    if (!element || !element.closest(PYCON)) return;
    var text = selection.toString();
    if (!/(^|\n)(>>>|\.\.\.)( |$)/m.test(text)) return;
    event.clipboardData.setData("text/plain", runnable(text));
    event.preventDefault();
  });
})();
