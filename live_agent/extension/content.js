// Trading Agent Signals — content script.
// Detects the symbol on the chart page you're viewing (Binance / CoinDCX /
// Groww / Kite / TradingView), asks YOUR local Trading Agent backend for the
// verdict + details, and shows a floating panel. All analysis happens on your
// Mac (http://127.0.0.1:5000) — nothing is sent to any third party.
// Research, not advice.

const API = "http://127.0.0.1:5000";
let lastKey = "";       // symbol+href key to avoid re-rendering
let panel = null;

// ---------------- symbol detection per site ---------------- //
function detect() {
  const h = location.hostname, p = location.pathname;
  let m;

  if (h.includes("binance")) {
    // /en/trade/BTC_USDT   /en/futures/BTCUSDT   /en/trade/BTCUSDT?type=spot
    m = p.match(/\/(?:trade|futures)\/([A-Z0-9_]{5,20})/i);
    if (m) {
      let s = m[1].toUpperCase().replace(/_/g, "");
      if (!s.endsWith("USDT") && !s.endsWith("USD")) return null;
      return { symbol: s.endsWith("USD") && !s.endsWith("USDT") ? s + "T" : s };
    }
  }
  if (h.includes("coindcx")) {
    // /trade/BTCINR  /trade/BTCUSDT  /markets? — take the pair, map INR→USDT
    m = p.match(/\/trade\/([A-Z0-9]{5,20})/i);
    if (m) {
      let s = m[1].toUpperCase();
      for (const q of ["INR", "USDT", "USD", "BTC"]) {
        if (s.endsWith(q)) { s = s.slice(0, -q.length); break; }
      }
      return { symbol: s + "USDT" };
    }
  }
  if (h.includes("kite.zerodha")) {
    // /chart/web/ciq/NSE/RELIANCE/738561   /chart/ext/tvc/NSE/INFY/...
    m = p.match(/\/(NSE|BSE)\/([A-Z0-9&\-]+)\//i);
    if (m) return { symbol: m[2].toUpperCase() + (m[1].toUpperCase() === "BSE" ? ".BO" : ".NS") };
  }
  if (h.includes("groww.in")) {
    // /stocks/reliance-industries-ltd  → resolve slug via backend
    m = p.match(/\/stocks\/([a-z0-9\-]+)/i);
    if (m) return { resolve: m[1] };
    m = p.match(/\/charts\/([a-z0-9\-]+)/i);
    if (m) return { resolve: m[1] };
  }
  if (h.includes("tradingview")) {
    // ?symbol=NSE%3ARELIANCE  or  /symbols/BTCUSDT/  or /chart/xxx/?symbol=...
    const sp = new URLSearchParams(location.search).get("symbol");
    let raw = sp || (p.match(/\/symbols\/([A-Za-z0-9:._\-]+)/) || [])[1];
    if (raw) {
      raw = decodeURIComponent(raw);
      const [ex, sym] = raw.includes(":") ? raw.split(":") : ["", raw];
      if (/BINANCE|BYBIT|OKX|COINBASE/i.test(ex) || /USDT?$/.test(sym))
        return { symbol: sym.toUpperCase().endsWith("USD") ? sym.toUpperCase() + "T" : sym.toUpperCase() };
      if (/NSE|BSE/i.test(ex)) return { symbol: sym.toUpperCase() + (/BSE/i.test(ex) ? ".BO" : ".NS") };
      return { symbol: sym.toUpperCase() };
    }
  }
  return null;
}

// ---------------- backend calls ---------------- //
async function j(path) {
  const r = await fetch(API + path);
  return r.json();
}

async function analyze(det) {
  let symbol = det.symbol;
  if (det.resolve) {
    const r = await j("/api/resolve?q=" + encodeURIComponent(det.resolve));
    if (!r.symbol) throw new Error("Could not identify this stock");
    symbol = r.symbol;
  }
  render({ loading: true, symbol });
  const [ins, mtf] = await Promise.all([
    j("/api/insight?symbol=" + encodeURIComponent(symbol)),
    j("/api/mtf?symbol=" + encodeURIComponent(symbol)).catch(() => ({})),
  ]);
  if (ins.error) throw new Error(ins.error);
  render({ symbol, ins, mtf });
}

// ---------------- UI ---------------- //
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}
const num = x => (x == null || isNaN(x)) ? "—" :
  Number(x).toLocaleString(undefined, { maximumFractionDigits: 2 });
const pc = x => (x == null || isNaN(x)) ? "—" : (x * 100).toFixed(0) + "%";

function ensurePanel() {
  if (panel && document.body.contains(panel)) return panel;
  panel = el("div", "ta-panel ta-min");
  panel.appendChild(el("div", "ta-head",
    `<span class="ta-logo">📈 Agent</span><span class="ta-sym"></span>
     <button class="ta-toggle" title="expand/collapse">▾</button>`));
  panel.appendChild(el("div", "ta-body"));
  panel.querySelector(".ta-toggle").onclick =
    () => panel.classList.toggle("ta-min");
  document.body.appendChild(panel);
  return panel;
}

function render(state) {
  const p = ensurePanel();
  p.querySelector(".ta-sym").textContent = state.symbol || "";
  const body = p.querySelector(".ta-body");

  if (state.loading) {
    body.innerHTML = `<div class="ta-load">◐ analyzing ${state.symbol}…</div>`;
    p.classList.remove("ta-min");
    return;
  }
  if (state.error) {
    body.innerHTML = `<div class="ta-err">${state.error}</div>
      <div class="ta-note">Is the Trading Agent app running on your Mac?<br>
      <code>python webapp.py</code></div>`;
    p.classList.remove("ta-min");
    return;
  }

  const d = state.ins, m = state.mtf || {};
  const side = m.side || "WAIT";
  const cls = side === "BUY" ? "ta-buy" : side === "SELL" ? "ta-sell" : "ta-wait";
  const emoji = side === "BUY" ? "🟢 BUY setup" : side === "SELL" ? "🔴 SELL setup" : "⏸ WAIT";
  const why = (m.verdict || "").replace(/^[🟢🔴⏸]+\s*/, "");
  const f = d.forecast || {}, L = d.levels || {}, S = d.sentiment || {};
  const reads = (m.reads || []).map(r =>
    `<span class="ta-tf ${r.verdict === 'BUY' ? 'ta-g' : r.verdict === 'SELL' ? 'ta-r' : ''}">${r.tf} ${r.verdict === 'BUY' ? '▲' : r.verdict === 'SELL' ? '▼' : '·'}</span>`).join("");

  body.innerHTML = `
    <div class="ta-verdict ${cls}">${emoji}</div>
    <div class="ta-why">${why}</div>
    <div class="ta-tfs">${reads}</div>
    <div class="ta-grid">
      <div><b>${num(d.last)}</b><span>price</span></div>
      <div><b>${pc(f.p_up)}</b><span>P(up) tmrw</span></div>
      <div><b>±${pc(f.exp_move_pct)}</b><span>typical move</span></div>
      <div><b>${num(f.levels && f.levels.p5)}–${num(f.levels && f.levels.p95)}</b><span>likely range</span></div>
      <div><b>${num(L.breakout && L.breakout.level)}</b><span>buy-above</span></div>
      <div><b>${num(L.breakdown && L.breakdown.level)}</b><span>sell-below</span></div>
    </div>
    <div class="ta-note">mood: ${(S.label || "—")}${S.primary_value != null ? " (" + S.primary_value + ")" : ""}
      · <a href="${API}/" target="_blank">open app</a></div>
    <div class="ta-disc">Research, not advice · ~50-55% honest hit-rate · the stop matters most</div>`;
  p.classList.remove("ta-min");
}

// ---------------- main loop (SPA-aware) ---------------- //
async function tick() {
  const det = detect();
  const key = det ? JSON.stringify(det) : "";
  if (!det) { if (panel) panel.remove(), panel = null; lastKey = ""; return; }
  if (key === lastKey) return;
  lastKey = key;
  try { await analyze(det); }
  catch (e) { render({ symbol: det.symbol || det.resolve, error: e.message }); }
}
tick();
setInterval(tick, 1500);
