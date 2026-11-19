/*
 * Sidick chat widget: a floating chat bubble for the portfolio.
 * Usage (e.g. in index.html, before </body>):
 *   <script src="/sidick-widget.js" data-api="https://sidick-bot.duckdns.org" defer></script>
 * Optional attributes: data-title, data-color (accent, default #7c3aed).
 */
(function () {
  "use strict";
  var script = document.currentScript;
  var API = ((script && script.dataset.api) || "http://localhost:8000").replace(/\/+$/, "");
  var TITLE = (script && script.dataset.title) || "Chat with Sidick";
  var ACCENT = (script && script.dataset.color) || "#7c3aed";
  var GREETING = "Hi! I'm Sidick's assistant. Ask me about my skills, projects or background. (Je parle aussi français !)";
  var SUGGESTIONS = ["Who are you?", "What projects have you built?", "How can I contact you?"];

  var history = [];
  var busy = false;
  var woke = false;

  var css = [
    ".sw-root{--sw-accent:" + ACCENT + ";--sw-bg:#fff;--sw-fg:#1f2330;--sw-muted:#6b7080;--sw-bot:#f1f2f6;--sw-border:#e3e5ec;",
    "position:fixed;right:20px;bottom:20px;z-index:2147483000;font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}",
    "@media (prefers-color-scheme:dark){.sw-root{--sw-bg:#17181d;--sw-fg:#ececf1;--sw-muted:#9a9db0;--sw-bot:#24262e;--sw-border:#30323c}}",
    ".sw-btn{width:58px;height:58px;border-radius:50%;border:0;background:var(--sw-accent);color:#fff;cursor:pointer;",
    "box-shadow:0 6px 20px rgba(0,0,0,.25);display:flex;align-items:center;justify-content:center;margin-left:auto}",
    ".sw-btn svg{width:28px;height:28px}",
    ".sw-panel{display:none;flex-direction:column;width:min(380px,calc(100vw - 32px));height:min(560px,calc(100vh - 110px));",
    "background:var(--sw-bg);color:var(--sw-fg);border:1px solid var(--sw-border);border-radius:16px;overflow:hidden;",
    "box-shadow:0 12px 40px rgba(0,0,0,.25);margin-bottom:12px}",
    ".sw-open .sw-panel{display:flex}",
    ".sw-head{background:var(--sw-accent);color:#fff;padding:12px 16px;display:flex;align-items:center;justify-content:space-between;font-weight:600}",
    ".sw-head small{display:block;font-weight:400;opacity:.85;font-size:12px}",
    ".sw-x{background:none;border:0;color:#fff;font-size:22px;cursor:pointer;line-height:1}",
    ".sw-msgs{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px}",
    ".sw-m{max-width:85%;padding:9px 12px;border-radius:14px;white-space:pre-wrap;word-wrap:break-word}",
    ".sw-m a{color:inherit;text-decoration:underline}",
    ".sw-bot{background:var(--sw-bot);align-self:flex-start;border-bottom-left-radius:4px}",
    ".sw-user{background:var(--sw-accent);color:#fff;align-self:flex-end;border-bottom-right-radius:4px}",
    ".sw-err{color:#c2410c}",
    ".sw-sugs{display:flex;flex-wrap:wrap;gap:6px}",
    ".sw-sug{border:1px solid var(--sw-border);background:none;color:var(--sw-fg);border-radius:999px;padding:5px 10px;font-size:13px;cursor:pointer}",
    ".sw-form{display:flex;gap:8px;padding:10px;border-top:1px solid var(--sw-border)}",
    ".sw-in{flex:1;border:1px solid var(--sw-border);background:var(--sw-bg);color:var(--sw-fg);border-radius:10px;padding:9px 11px;font:inherit;outline:none}",
    ".sw-in:focus{border-color:var(--sw-accent)}",
    ".sw-send{border:0;background:var(--sw-accent);color:#fff;border-radius:10px;padding:0 14px;font:inherit;cursor:pointer}",
    ".sw-send:disabled{opacity:.5;cursor:default}",
    ".sw-dots::after{content:'…';animation:sw-blink 1s infinite}",
    "@keyframes sw-blink{50%{opacity:.2}}",
  ].join("");

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  }

  // Safe formatting: escape everything, then turn **bold** and URLs into markup.
  function render(node, text) {
    var esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    esc = esc.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    esc = esc.replace(/(https?:\/\/[^\s<)]+[^\s<).,;!?])/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
    node.innerHTML = esc;
  }

  var root = el("div", "sw-root");
  var style = el("style");
  style.textContent = css;
  var panel = el("div", "sw-panel");
  var head = el("div", "sw-head");
  var titleBox = el("div", null, TITLE);
  var status = el("small", null, "Usually replies in a few seconds");
  titleBox.appendChild(status);
  var close = el("button", "sw-x", "×");
  close.setAttribute("aria-label", "Close chat");
  head.appendChild(titleBox);
  head.appendChild(close);
  var msgs = el("div", "sw-msgs");
  var form = el("form", "sw-form");
  var input = el("input", "sw-in");
  input.placeholder = "Ask me anything…";
  input.maxLength = 500;
  var send = el("button", "sw-send", "Send");
  send.type = "submit";
  form.appendChild(input);
  form.appendChild(send);
  panel.appendChild(head);
  panel.appendChild(msgs);
  panel.appendChild(form);
  var btn = el("button", "sw-btn");
  btn.setAttribute("aria-label", "Open chat");
  btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>';
  root.appendChild(style);
  root.appendChild(panel);
  root.appendChild(btn);

  function addMsg(cls, text) {
    var m = el("div", "sw-m " + cls);
    render(m, text);
    msgs.appendChild(m);
    msgs.scrollTop = msgs.scrollHeight;
    return m;
  }

  function wake() {
    if (woke) return;
    woke = true;
    fetch(API + "/health").catch(function () {});
  }

  function toggle(open) {
    root.classList.toggle("sw-open", open);
    if (open) {
      wake();
      setTimeout(function () { input.focus(); }, 50);
    }
  }

  async function ask(text) {
    if (busy || !text.trim()) return;
    busy = true;
    send.disabled = true;
    var sugs = msgs.querySelector(".sw-sugs");
    if (sugs) sugs.remove();
    addMsg("sw-user", text);
    var bubble = addMsg("sw-bot sw-dots", "");
    status.textContent = "Thinking…";
    var answer = "";
    try {
      var res = null;
      var err = null;
      for (var attempt = 0; attempt < 4; attempt++) {
        res = await fetch(API + "/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, history: history.slice(-6) }),
        });
        if (res.ok) break;
        err = await res.json().catch(function () { return {}; });
        // 503 "waking up" = server still loading its models: wait and retry.
        if (res.status !== 503 || !/waking/i.test(err.detail || "") || attempt === 3) break;
        status.textContent = "Waking up…";
        await new Promise(function (r) { setTimeout(r, 8000); });
      }
      if (!res.ok) {
        throw new Error(typeof err.detail === "string" ? err.detail : "Something went wrong. Please try again.");
      }
      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      for (;;) {
        var chunk = await reader.read();
        if (chunk.done) break;
        answer += decoder.decode(chunk.value, { stream: true });
        bubble.classList.remove("sw-dots");
        render(bubble, answer);
        msgs.scrollTop = msgs.scrollHeight;
      }
      answer += decoder.decode();
      render(bubble, answer || "…");
      history.push({ role: "user", content: text }, { role: "assistant", content: answer });
    } catch (e) {
      bubble.classList.remove("sw-dots");
      bubble.classList.add("sw-err");
      render(bubble, e && e.message && !/fetch|network/i.test(e.message)
        ? e.message
        : "I can't reach my server right now. Please try again later, or use the contact form.");
    } finally {
      busy = false;
      send.disabled = false;
      status.textContent = "Usually replies in a few seconds";
      input.focus();
    }
  }

  btn.addEventListener("click", function () { toggle(!root.classList.contains("sw-open")); });
  btn.addEventListener("mouseenter", wake);
  close.addEventListener("click", function () { toggle(false); });
  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = input.value;
    input.value = "";
    ask(text);
  });

  addMsg("sw-bot", GREETING);
  var sugBox = el("div", "sw-sugs");
  SUGGESTIONS.forEach(function (s) {
    var b = el("button", "sw-sug", s);
    b.type = "button";
    b.addEventListener("click", function () { ask(s); });
    sugBox.appendChild(b);
  });
  msgs.appendChild(sugBox);

  function mount() { document.body.appendChild(root); }
  if (document.body) mount();
  else document.addEventListener("DOMContentLoaded", mount);
})();
