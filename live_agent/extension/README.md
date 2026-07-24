# Trading Agent Signals — Chrome Extension

Shows your Trading Agent's verdict + full details as a floating panel on any
chart you open — **Binance, CoinDCX, Groww, Kite (Zerodha), TradingView**.
All analysis runs on **your Mac**; the extension only talks to
`http://127.0.0.1:5000`. Nothing goes to any third party.

## Install (2 minutes, one time)

1. Make sure the app is running on your Mac:
   `cd live_agent && python webapp.py`
2. Open Chrome → `chrome://extensions`
3. Toggle **Developer mode** (top-right)
4. Click **Load unpacked** → select this folder
   (`/Users/rohitraj/rl-trading-agent/live_agent/extension`)
5. Open any chart — e.g. `binance.com/en/trade/BTC_USDT` or a Groww stock page —
   and the panel appears bottom-right.

## What it shows

- **One clear verdict**: 🟢 BUY setup / 🔴 SELL setup / ⏸ WAIT — from the
  multi-timeframe alignment engine (5m/15m/4h/1d), with the plain-English reason.
- Per-timeframe reads (which frames are bullish/bearish).
- Price, P(up) tomorrow, typical daily move, tomorrow's likely range.
- The two levels that matter: **buy-above** (breakout) and **sell-below** (breakdown).
- Market mood (crypto Fear & Greed / India VIX / VIX).

Symbol detection per site: Binance & CoinDCX pairs from the URL (INR pairs map
to USDT for analysis), Kite from `/NSE/SYMBOL/`, TradingView from its symbol
parameter, and Groww stock slugs are resolved to NSE tickers by the backend
(`/api/resolve`).

## Honest limits

- **Research, not advice.** Measured hit-rate is ~50–55% — the honest number for
  any directional system. The stop level matters more than the target.
- Requires the local app to be running; if it isn't, the panel says so.
- If a symbol can't be detected/resolved (unusual URL, IPO, new listing), the
  panel simply stays hidden or reports it — no guessing.
