# Live Ensemble Trading Agent

**An autonomous, multi-asset systematic trading agent that runs on your Mac and trades a paper account on its own — once per trading day.**

This is a sibling to the parent [RL trading project](../README.md). Where that one trained a single-asset PPO policy (which underperformed buy-and-hold out-of-sample), this agent takes a more robust, more *advanced* approach: a **multi-signal ensemble** across a **basket of stocks**, with **volatility-targeted position sizing** and a **hard risk overlay** — wired for **live paper execution** via Alpaca.

> [!CAUTION]
> Defaults to simulated / paper money. There is **no real-money code path** in this project. This is a research tool, not investment advice. Backtests overstate live results. Do not point this at real capital without extensive additional validation — and even then, understand you can lose money.

---

## Why this is more advanced than the PPO agent

| | Parent PPO agent | This ensemble agent |
|---|---|---|
| Assets | 1 (AAPL) | 8-name basket (configurable) |
| Brain | Single PPO policy | Blend of 3 orthogonal signals |
| Position sizing | All-in / all-out | Continuous, inverse-volatility weighted |
| Risk controls | Reward penalty only | Hard caps + drawdown kill-switch |
| Execution | Backtest only | Live paper (Alpaca) or local sim |
| Runs by itself | No | Yes — daily via launchd |

Diversifying across **signals** (momentum, mean-reversion, trend) *and* **assets** is the cheapest risk reduction available, and it sidesteps the single-policy fragility that sank the PPO agent.

---

## How it works — one decision cycle

```
 fetch live data ─▶ score the universe ─▶ size positions ─▶ kill-switch ─▶ diff book ─▶ orders ─▶ log
   (yfinance)        (ensemble brain)     (risk overlay)    (drawdown)    (rebalance band)
```

| Module | Role |
|---|---|
| `data_feed.py` | Live OHLCV at any timeframe (yfinance) + technical features (returns, vol, MAs, z-score, RSI) |
| `markets.py` | Resolves a named market/timeframe preset into the working config |
| `strategies.py` | The ensemble brain: momentum + mean-reversion + trend → composite score per asset |
| `risk.py` | Inverse-vol sizing, per-name & gross caps, high-water-mark kill-switch |
| `broker.py` | `SimBroker` (local, no account) or `AlpacaBroker` (paper) behind one interface |
| `trader.py` | Orchestrates the cycle; the autonomous entry point |
| `backtest.py` | Replays the *exact* live code path over history vs. benchmarks |
| `auto_learn.py` | Walk-forward self-learning: re-fits + OOS-validates the weights |
| `forecast.py` | Probabilistic next-day forecaster (any symbol) + calibration |
| `screen.py` | On-demand candidate screener: rank names + forecast + news |
| `news.py` | Headlines as context via Yahoo's official feed (no forum scraping) |
| `sentiment.py` | Market-specific sentiment: crypto Fear&Greed, India VIX, VIX |
| `levels.py` | Breakout/breakdown trigger levels + expected follow-through |
| `webapp.py` | Local Flask dashboard — search any ticker for insights in the browser |
| `universe.py` | Builds/caches the full tradable universe (all NSE + all Binance) |
| `binance_feed.py` | Real OHLCV from Binance's public API for any coin |
| `scan.py` | Offline batch scan of the universe → cached ranking for the dashboard |
| `alerts.py` | Price-level alerts with native macOS notifications |
| `agents.py` | Committee of 45 setup-agents + consensus voting & backtest |
| `signals.py` | Build BUY/SELL signals (entry/target/stop, conviction, leverage) |
| `live_signals.py` | Market-hours-aware best-of-market scan + push STRONG signals |
| `notify.py` | Send signals to Telegram / WhatsApp / macOS |
| `live_trader.py` | Autonomous live paper-trading session (watch it perform) |
| `train_strategy.py` | Walk-forward trainer for the trading policy (OOS-gated) |
| `derivatives.py` | Live Binance futures + leverage / NSE F&O reference |
| `options.py` | Black-Scholes pricing + a paper options book |
| `signal_perf.py` | Measures the real hit-rate / expectancy of the signals |
| `dual_momentum.py` | Research-backed long/short dual/time-series momentum |
| `mtf.py` | Multi-timeframe alignment (Triple-Screen style) + per-TF agent weighting |
| `mcp_server.py` | MCP server — talk to the whole platform through Claude |

### The ensemble brain

Each signal scores every asset in `[-1, +1]`; the composite is a weighted blend (weights in `config.yaml`):

- **Momentum** — skip-adjusted trailing return. Rides persistent trends.
- **Mean-reversion** — negative z-score vs a moving average. Buys oversold names.
- **Trend filter** — fast-vs-slow MA regime. Keeps the book on the right side of the primary trend.

Momentum and mean-reversion deliberately pull opposite ways on short horizons, which dampens whipsaw; the trend filter breaks ties toward the prevailing regime.

### The risk overlay (the seatbelt)

1. **Inverse-volatility sizing** — each name's weight ∝ score / volatility, so a high-vol name like NVDA can't silently dominate portfolio risk.
2. **Hard caps** — no single name above `max_position_weight` (default 25%); total invested ≤ `max_gross_exposure` (default 100% — no leverage).
3. **Kill-switch** — if equity falls `daily_loss_halt` (default 5%) below its high-water mark, the agent **liquidates everything and halts** until you reset it.

---

## Quick start — see it trade in 30 seconds (no account needed)

From the parent project root (reuses its venv):

```bash
source venv/bin/activate
pip install -r live_agent/requirements.txt   # adds alpaca-py + python-dotenv
cd live_agent

# Watch it decide on today's live market — places NO orders:
python trader.py --once --dry-run

# Let it actually trade the local simulated account:
python trader.py --once --broker sim
cat state/sim_account.json     # your simulated cash + positions
```

The `SimBroker` is a real (simulated) account persisted to `state/sim_account.json` — run it day after day and it compounds on its own.

---

## Full universe — every NSE stock + every Binance coin

```bash
python universe.py                 # cache the full list (~2,300 NSE + ~430 Binance)
```

This pulls the official NSE equity list and Binance's `exchangeInfo`, caching them to `state/universe.json`. After that:

- **Search** any of the ~2,700+ symbols in the dashboard (with autocomplete). Crypto symbols (`BTCUSDT`, `BTC-USD`, or bare `BTC`) fetch **real Binance data** via `binance_feed.py`; stocks/indices use yfinance.
- **Rank the whole universe** with the offline scanner (ranking thousands live would take ~an hour, so it's a scheduled job, not a click):

```bash
python scan.py --segment binance              # all ~430 coins
python scan.py --segment nse --limit 500        # first 500 NSE stocks
python scan.py --segment all                     # everything (slow)
```

It writes `state/scan_results.json`, which the dashboard's **Top picks** tab reads instantly. Schedule it nightly with launchd (same pattern as `run_daily.sh`) to keep the shortlist fresh.

> [!CAUTION]
> The scan ranks by the backtested composite — a **research shortlist, not a buy list** (the composite did not beat buy-and-hold). Tiny illiquid coins will appear; most have no tradable edge and brutal spreads. Treat high scores as "look closer", never "buy".

## Talk to it through Claude (MCP server)

`mcp_server.py` exposes the whole platform as **Model-Context-Protocol tools**, so Claude Desktop / Claude Code can answer questions from your own engine — the honest version of "connect Claude to your trading app." It's **read-only insights + paper actions only — no real orders, no real money.**

Tools: `insight`, `signal`, `committee`, `forecast`, `top_picks`, `options_idea`, `market_open`, `hit_rate`, `paper_portfolio`, `paper_buy`.

**Connect it to Claude Desktop** — add this to `~/Library/Application Support/Claude/claude_desktop_config.json` (merge into the existing JSON), then restart Claude Desktop:

```json
{
  "mcpServers": {
    "trading-agent": {
      "command": "/Users/rohitraj/rl-trading-agent/venv/bin/python",
      "args": ["/Users/rohitraj/rl-trading-agent/live_agent/mcp_server.py"]
    }
  }
}
```

Then ask Claude things like *"what's the committee verdict on BTCUSDT?"*, *"scan crypto top picks"*, *"show my paper portfolio"*, *"what's the honest hit rate on crypto?"*.

> Same honesty as everywhere: the tools surface signals, forecasts and *paper* trades — they don't place real orders or claim profit. `paper_buy` moves simulated money only.

## Two modes: Simple (default) and Pro

- **Simple mode** (`/`) — the noob-friendly product: type any stock/coin (even
  plain names like "reliance" or "btc"), get **one clear verdict** — 🟢 BUY setup /
  🔴 SELL setup / ⏸ WAIT — with a plain-language reason, chance-it-rises, typical
  move, likely range, market mood, and (when actionable) buy-above / target / stop.
- **Pro mode** (`/pro`) — the full 17-tab research dashboard for depth.

## Chrome extension — signals on any chart you open

`extension/` overlays your agent's verdict + details on **Binance, CoinDCX,
Groww, Kite (Zerodha), and TradingView** chart pages (symbol auto-detected from
the URL; Groww slugs resolved via `/api/resolve`, CoinDCX INR pairs mapped to
USDT). Install: `chrome://extensions` → Developer mode → **Load unpacked** →
select `live_agent/extension`. Requires the local app running; everything stays
on your machine. See [extension/README.md](extension/README.md).

> **Monetization (honest note):** in India, selling buy/sell recommendations
> requires **SEBI Research Analyst / Investment Adviser registration** —
> unregistered signal-selling is illegal. Legitimate paths: free/open tool,
> education content, or an analytics product with hard disclaimers (and
> registration if it crosses into advice). Also: our measured ~50–55% hit rate
> is the honest number — charging beginners for "signals" would be exactly the
> scam this project exposes.

## Use it as an app (PWA — phone + desktop)

The dashboard is an installable **Progressive Web App**:

- **On your phone (same Wi-Fi):** start the server (`python webapp.py`), open the URL shown in the sidebar (e.g. `http://<your-mac-ip>:5000`), then **"Add to Home Screen"** (iOS Safari share menu / Android Chrome install prompt). You get a real app icon and a standalone fullscreen app — the mobile sidebar works via the ☰ menu.
- **On your Mac:** Chrome/Edge show an "Install app" icon in the address bar for a standalone window.
- **Auto-updating:** the server runs with template auto-reload + a code reloader — **any edit to the project applies on the next refresh**, no restart, no redeploy. (Note: a code-file change restarts the server process, which ends any running live-trading session.)

> ⚠ **LAN-only by design.** The server binds to your local network so your phone can reach it, but do **not** port-forward it to the internet — it has no login. If you ever want access from outside your home, use a private tunnel (e.g. Tailscale) rather than exposing the port.
>
> **Why not cloud-host it?** Possible (Render/Railway auto-deploy on git push), but honestly not worth it here: US-hosted servers are geo-blocked by Binance's API, the app is single-user with local state files, and it adds cost + secrets management. Local + PWA gives you the same "app" experience with none of that.

## Web dashboard (search any stock in your browser)

A local dashboard — runs entirely on your Mac, no cloud:

```bash
pip install -r live_agent/requirements.txt   # includes Flask
cd live_agent
python webapp.py                              # open http://127.0.0.1:5000
```

A **sidebar-driven** app with sections you pick from the left nav:

- **📊 Overview** — a market pulse: indices + major coins with price, daily change, forecast vol, and signal at a glance.
- **🔍 Search any stock** — full insight for any of the ~2,700 symbols: a **price chart** with the breakout/breakdown levels and tomorrow's forecast cone drawn on it, the probabilistic range, P(up), signal score, one-click calibration, **paper buy/sell**, add-to-watchlist, and headlines.
- **🇮🇳 Indian market** / **₿ Crypto market** — separate sections, each with its own cached Top-picks ranking plus a scoped screener.
- **📐 Options lens** — the model's forecast (realised) vol vs the market's **implied** vol (India VIX / VIX): flags when options look rich or cheap. The one genuinely feasible options edge.
- **🧪 Backtest** — backtest the model on any symbol at **daily / weekly / hourly** vs buy-and-hold, with an equity chart. Stocks & crypto.
- **💰 Paper account** — simulated portfolio (cash, positions, equity) traded with fake money.
- **⭐ Watchlist** — your saved symbols, persisted locally, screened into a table.

Re-running re-fetches fresh data, so it stays current.

**Why no Google Drive?** The data is small (a few MB of prices) and re-downloadable from yfinance any time, and the agent's saved state is kilobytes — so cloud storage adds OAuth/quota friction for zero benefit. Everything runs locally. Drive/a VPS would only matter if you later wanted multi-device access or to keep data you *can't* regenerate.

## 🏦 Market desks — India is the primary desk

The platform is separated into three self-contained **desks** (config `desks:`) with a
switcher at the top of the sidebar: **🇮🇳 India (primary)**, ₿ Crypto, 🇺🇸 US. Each desk
scopes the whole dashboard — pulse board, quick chips, option chains, default symbols,
and which tabs are visible. `active_market` is now `india_daily`, so the autonomous
trader and self-learner work the Indian market by default.

The **🏦 India desk** tab covers everything legally tradable in India:

- **Equities & indices** — the full ~2,350-stock NSE universe (existing search/screen/scan).
- **Index & stock options** — NSE chains (NIFTY/BANKNIFTY/FINNIFTY/SENSEX; live on an
  Indian network, Black-Scholes-modeled fallback elsewhere) + the options lens
  (forecast vol vs India VIX).
- **F&O + MCX reference** — indicative leverage/margin reality for index/stock futures,
  option buying vs selling, and MCX commodity contracts (`/api/fno`).
- **Commodities** — live NSE commodity ETF board (gold/silver: GOLDBEES, SETFGOLD,
  HDFCGOLD, SILVERBEES, SILVERIETF…) via `/api/commodities`. Honest note: direct MCX
  futures data needs a broker API (Kite/Dhan/Angel).
- **Mutual funds** — `mf.py`: search all ~40k AMFI schemes and get the honest scorecard
  (NAV history chart, 1m/3m/6m/1y returns, 3y/5y/inception CAGR, volatility, max
  drawdown) via the free mfapi.in API. Funds are allocation vehicles, not trading
  instruments — the note says so on every card.
- **Intraday** — the existing NSE intraday paper session (09:15–15:30 IST gated).

### 🔌 Zerodha Kite — real F&O + MCX data (read-only)

The India desk has a **Broker card** that connects your Zerodha account for real
exchange data (`kite_feed.py`). Once connected, the commodities board switches to
**live MCX futures** (gold, silver, crude, natural gas…), the F&O table shows **live
NFO futures marks**, and option chains become **real strikes/LTP/OI** instead of the
modeled fallback.

Honest, important caveats:
- **Read-only by design.** `kite_feed.py` exposes no order/trade functions — it fetches
  data and cannot place, modify or cancel an order. Trading stays in your hands, in Kite.
- **Paid API (~₹2000/mo)** — subscribe + create an app at developers.kite.trade, then put
  `KITE_API_KEY` / `KITE_API_SECRET` in `live_agent/.env`.
- **You log in, not the app.** Click the login link → authenticate on Zerodha's own site →
  paste the `request_token` from the redirect back into the Broker card. This app never
  sees your Zerodha password.
- **Tokens expire ~6 AM IST daily** — a 20-second re-login each morning is normal, and a
  limitation of Kite, not something we can automate away.
- Not connected? Everything falls back to the free-data paths (NSE ETFs, reference F&O
  table, modeled chains), so the desk always works.

Endpoints: `/api/kite/{status,login_url,connect,logout,fno,mcx,option_chain}`. The
existing `/api/commodities`, `/api/fno` and `/api/option_chain` auto-prefer live Kite
data when connected.

## Markets & timeframes

The agent is **market- and timeframe-agnostic**. `config.yaml` defines named presets under `markets:` — each a basket of symbols, a bar `interval`, the right `periods_per_year` annualisation, and a venue-appropriate `transaction_cost`. `active_market` picks which one the live agent + self-learner use; the backtester can run any one or sweep them all:

```bash
python backtest.py --market crypto_daily --years 3   # one market
python backtest.py --market india_daily              # NSE equities
python backtest.py --all --years 3                   # sweep → ranked leaderboard
```

Built-in presets: `us_tech_daily`, `us_weekly`, `crypto_daily`, `crypto_hourly`, `india_daily`. Add your own by editing `markets:` (any yfinance symbol works — US tickers, `BTC-USD`, `RELIANCE.NS`, …). yfinance caps intraday history (~730d for 1h, ~60d for ≤30m), so intraday presets test on shorter windows.

> [!WARNING]
> **Frequency kills.** In testing, the *weekly* timeframe scored best and *hourly* crypto lost ~80% — almost entirely to transaction costs, which scale with how often you trade. High-frequency / intraday is where a mediocre edge becomes a guaranteed loss. Respect the cost model; if anything, set it higher than you think.

## Validate before you trust it

```bash
python backtest.py --years 3
```

This replays the **exact** live scoring + sizing logic over history and compares against equal-weight buy-and-hold (and SPY for daily US). Output table + an equity-curve PNG per market in `backtest_results/`.

**Honest result (3-yr, as of mid-2026):** the ensemble made money (≈15% CAGR, Sharpe ≈0.9) but **underperformed simply holding the basket** over this tech-heavy bull market — its lower drawdown is the trade-off. A defensive, diversified strategy lags a raging bull by design; its edge is meant to show in choppier or falling markets. **Don't curve-fit the config to "beat" the backtest** — that's how you build something that looks great on history and loses live.

### Walk-forward OOS on the signal backtester (🎯 tab → “Walk-forward OOS”)

The threshold **sweep** answers "which `min_agree` paid *in hindsight*" — which is exactly how
curve-fitting happens. The walk-forward is the honest version: in each fold the threshold is
chosen **only from data before the test segment**, then traded on the unseen segment that follows
(`signal_backtest.walk_forward`, or `wf=1` on `/api/signal_backtest`). Measured example
(BTCUSDT 1d, 4 folds): in-sample the chosen thresholds promised **+0.64%/trade**; out-of-sample
they delivered **+0.07%** — ~89% of the "edge" evaporated on unseen data. That surviving sliver
is the only part worth forward-testing. Expect this. It is not a bug; it is what honest
validation looks like.

---

## 🛡️ Risk desk (the pro layer)

Every measurement in this project says the signals are ~48–55% — **the durable edge is risk
control**, so it now has its own desk (`risk_pro.py` + `regime.py`, 🛡️ tab):

- **Pre-trade checklist** — every trade goes through the desk before the market:
  reward:risk (≥1.5:1 minimum), **regime alignment** (don't fade a freight train),
  45-agent committee agreement, fixed-fractional **position size** (the stop sets the size —
  never the other way around), portfolio heat after the trade, and your daily limits.
  Verdict: **TAKE / TAKE-HALF / SKIP**, with reasons. `/api/risk/checklist?symbol=…&side=BUY`
  (blank entry/stop/target auto-fill from live price + ATR).
- **Daily discipline light** — GREEN / YELLOW / RED from the journal: −3R daily stop,
  3-losses-in-a-row → half size (tilt guard), 5% total open-heat cap, per-market
  concentration guard, and a flag on any open trade without a stop. `/api/risk/discipline`.
- **Regime detector** (`regime.py`, `/api/regime`) — ADX + Kaufman efficiency + realized-vol
  percentile → TRENDING / CALM RANGE / VOLATILE CHOP + which setup family has the tailwind.
  Regime is context, not prediction: it tells you which mistake is most expensive right now.
- **R-curve review** — the journal now tracks cumulative **R-multiples**, win/loss streaks and
  average hold time. Pros review in R, not money: +1R = you made what you risked, and a rising
  R-curve means the process works at any account size.

Limits live in `config.yaml → risk_pro:`. None of this predicts price — it makes losing
streaks survivable, which is the entire pro secret.

---

## Going live on Alpaca (paper money)

1. Create a free account at [alpaca.markets](https://alpaca.markets) and switch to **Paper Trading**.
2. Generate paper API keys.
3. Configure credentials:
   ```bash
   cp .env.example .env
   # edit .env, paste your paper key + secret
   ```
4. Run — it auto-detects the keys and routes to Alpaca paper:
   ```bash
   python trader.py --once --dry-run     # confirm the connection first
   python trader.py --once               # place paper orders
   ```

`AlpacaBroker` is hard-wired to `paper=True`. Moving to real money is intentionally **not** a config toggle in this project.

---

## Probabilistic next-day forecasting

`forecast.py` estimates the **distribution** of tomorrow's move for an index — not the direction (which is ~unforecastable), but the *range* and *probabilities*, which are forecastable because volatility clusters.

```bash
python forecast.py --symbol "^NSEI"   # Nifty (also ^BSESN, ^NSEBANK, ^GSPC, ^NDX)
python forecast.py --all              # every symbol in forecast_universe
```

Method: **Filtered Historical Simulation** — an EWMA volatility forecast plus the empirical distribution of standardised past returns. For each session it prints tomorrow's probabilistic level range (5/50/95%), `P(up)`, `P(move >1%)`, and expected move in points.

**It validates itself by calibration, not by P&L** — the only honest test of a probability. Measured coverage over thousands of days (built-in presets):

| Index | History | 90% coverage | Mean PIT | Vol-target vs B&H |
|---|---|---|---|---|
| Nifty (^NSEI) | 2007– (4.6k d) | 89.5% ✓ | 0.495 | Sharpe 0.72 vs 0.79, **DD 21% vs 38%** |
| Bank Nifty (^NSEBANK) | 2007– | 89.4% ✓ | 0.494 | Sharpe 0.73 vs 0.74, **DD 23% vs 48%** |
| Sensex (^BSESN) | 1997– (6.6k d) | 89.8% ✓ | 0.497 | **Sharpe 0.75 vs 0.64**, DD 34% vs 61% |
| S&P 500 (^GSPC) | 1927– (24k d) | 89.9% ✓ | 0.498 | **Sharpe 0.51 vs 0.40**, DD 61% vs 86% |

The 90% intervals contain the realised move ~90% of the time and the PIT histogram is flat (`backtest_results/forecast_pit_*.png`) — the probabilities are **trustworthy**.

The vol-target backtest is an honest tradable use: scale exposure by `target_vol / forecast_vol`. Over full cycles (Sensex, S&P) it **beats buy-and-hold on Sharpe and roughly halves drawdown** — risk-adjusted edge *without* predicting direction.

> [!NOTE]
> **Direction stays a coin-flip** (`P(up)` ≈ 52–56%, the market's mild upward drift — no directional alpha). The value is in the *range/vol*. **Options:** this vol forecast is the input you'd compare to *implied* vol (India VIX / option IV) to find rich/cheap premium — but a real option-P&L backtest needs historical option chains, which free data lacks (paid NSE/broker data required).

## On-demand screener (give it names → ranked candidates + news)

`screen.py` takes any tickers (or a market preset), pulls **fresh** data each run, and ranks them by the agent's transparent signal stack — attaching each name's next-day forecast and recent headlines.

```bash
python screen.py --symbols "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS"
python screen.py --market india_daily --news 3
python screen.py --symbols "AAPL,NVDA,MSFT"
```

Each row shows the composite score (momentum/mean-rev/trend), the forecast next-day expected move, `P(up)`, and the volatility regime (calm/norm/HIGH). Names clearing the entry threshold are flagged `◀`. Re-running re-scrapes data, so it "updates itself" — schedule it daily with launchd like the trader.

> [!CAUTION]
> **This is a research watchlist, not a buy list.** It ranks by a *defined, backtested* metric — and that metric did **not** beat buy-and-hold in testing. `P(up)` sits near the base rate because direction is unforecastable. It will happily show an empty candidate set when nothing scores well (it does not invent signals). **News is context for you to read, not an alpha source** — and it comes from Yahoo's official feed, *not* Reddit/X/forum scraping, which is ToS-violating, bot-gamed, and has no durable edge. Validate and paper-trade everything.

## Multi-timeframe alignment (🔭 Multi-TF tab)

`mtf.py` implements how professionals actually structure trades (Elder's **Triple Screen** family): the **higher timeframe sets the allowed direction**, lower timeframes time the entry, and **disagreement = no trade**. The 45 agents are **re-weighted per timeframe** — mean-reversion & volume agents lead on 5m/15m (intraday noise mean-reverts), trend & momentum agents lead on 4h/1d (higher frames trend). Output is **one clear verdict** ("BUY — aligned", "SELL — aligned", or "WAIT — frames disagree") instead of 45 raw votes. Crypto reads 5m/15m/4h/1d; stocks read 15m/1h/1d.

> **Honest validation** (2y BTCUSDT 15m, Binance fee): requiring 4h agreement cut trades 13,909→7,706, halved time-in-market, and improved Sharpe (−7.4→−4.8) — but bar-by-bar 15m signal-flipping still loses catastrophically to fees either way (buy-hold was −3.8%). Alignment's real value is **decision clarity and whipsaw reduction** — especially the WAIT verdict (e.g. "1d bearish but 15m bullish = pullback trap") — not turning fast trading profitable.

## Research-backed strategies (📚 Momentum tab)

`dual_momentum.py` implements two well-documented academic strategies faithfully:
**Time-Series Momentum** (Moskowitz-Ooi-Pedersen 2012) and **Dual Momentum**
(Antonacci 2014) — rank a diversified cross-asset universe (SPY/QQQ/GLD/TLT/EEM/Nifty/BTC)
by 12-month momentum, go **long** the strongest with a positive trend and **short**
the weakest with a negative trend, volatility-targeted.

```bash
python dual_momentum.py --years 9
```

> **Honest result:** over 2017–2026 it returned +92% (Sharpe 0.60) vs the basket's
> +272% buy-and-hold (Sharpe 1.07) — it **underperformed.** The literature itself
> warns that much of momentum's apparent edge is *volatility-scaling*, not the
> signal (Kim et al. 2016), and edges decay. Even textbook academic strategies
> don't reliably beat buy-and-hold. It's here as a real, two-sided tool to study —
> not a promise. The tab shows its live long/short positioning.

## Market sentiment (the gauges traders actually watch)

`sentiment.py` adds the *market-specific* mood indicators real traders use, routed by symbol:

- **Crypto** → the **Fear & Greed Index** (alternative.me, free) — a contrarian gauge: extreme fear often marks bottoms, extreme greed marks froth.
- **NSE / Indian equities** → **India VIX**; **US** → **VIX**.
- **Any symbol** → recent **news-headline lean**.

It shows on each insight (🧭 Market sentiment card) and, more importantly, **sharpens the signals**: sentiment that *confirms* a technical BUY/SELL raises its conviction (±12), sentiment that *contradicts* it dulls it. Example seen live: BTC's technicals said SELL, but Fear & Greed at 15 ("Extreme Fear") flagged the short as crowded → conviction dropped 64→52.

> Honest caveat: sentiment is a **weak, noisy** edge — used here as *confirmation/context*, not a standalone signal. It's most useful when price and mood **diverge** (e.g. extreme fear in a downtrend = possible bottom).

## Committee of 45 agents (consensus voting)

Instead of one signal, **45 independent setup-agents** vote on every bar — across six families so the vote reflects genuinely different ideas (trend, momentum, mean-reversion, breakout, volume, statistical). Each agent commits to exactly one side — **BUY (go long) or SELL (go short), never "hold"** — so the committee is always a clean long-vs-short tally. Examples: SMA/EMA/MACD cross, ADX, Supertrend, Ichimoku, Parabolic SAR, Aroon, Vortex, HMA; RSI, Stochastic, Williams %R, CCI, TSI, Ultimate Osc, PPO, Fisher, Elder Ray; Bollinger, z-score, Keltner, RSI(2), %B; Turtle-55, squeeze; OBV, Chaikin MF, Force Index, MFI; linear-regression slope, return z-score, HH/HL structure.

The committee acts only when at least a **threshold** agree. The **🏛️ Committee** tab shows every agent's vote, the per-family net, the verdict, and a one-click consensus backtest with an adjustable threshold slider (18–45).

> **How many agents is enough?** There's no hard cap, but there are only ~6–8 *genuinely independent* edges in price data — past ~45 you mostly add correlated copies that inflate the vote without adding information. Diversity of ideas beats raw count.

## Live paper trading (watch the agents perform)

The **🟢 Live trading** tab runs an **autonomous paper session** you can watch in real time. You set the (paper) capital, market, leverage and duration; a background loop then trades on its own:

- **Crypto** (24/7, real-time Binance data) or **NSE intraday** (only 09:15–15:30 IST, auto square-off at close).
- Opens positions from STRONG signals (committee + ensemble agree), each with a stop and target.
- Marks to the latest price every poll, closes on stop/target/**liquidation**, and updates a live equity curve, P&L, open positions and trade log.

Leverage (**1–125×**, matching Binance) is modelled with a real **liquidation rule** so you *see* the risk — at 125× a ~0.8% adverse move wipes a position. It's a teaching tool, **100% simulated money**, no real orders. A **circuit breaker** auto-squares-off and halts the whole session if it loses your chosen stop-out % (default 25%). NSE intraday positions are force-closed at the 15:30 IST bell.

> Verified: a 125× crypto session swung **+13% → −1.5% in ~70 seconds** with the drawdown racing toward the breaker — exactly the lesson. Use high leverage in the sim to *learn why not to use it live*.

## Live signal chart

The **📉 Live chart** tab streams a symbol's price and drops a **BUY/SELL marker only when ≥ a chosen % of the 45 agents agree** (default **85% = 38/45**). A live consensus meter shows how close it is in between. Near-unanimity fires *rarely by design* — that's the point of a high bar. Crypto streams on 5-minute bars, NSE on 15-minute, refreshing every 6 seconds.

> Verified live: a crypto session opened 4 shorts and tracked equity tick-by-tick; an NSE intraday session (market open) opened RELIANCE/TCS/INFY/LT and showed live P&L. Start one from the dashboard and watch the equity move.

## Live signals to your phone (Telegram / WhatsApp)

The **📡 Signals** tab scans the **open markets** for the strongest setups — *market-hours aware*: crypto 24/7, NSE only on weekdays 09:15–15:30 IST. It ranks the top candidates (from the nightly scan) by **conviction** (blended ensemble + committee, only strong when they *agree*) and shows **🟢 strong long / 🔴 strong short** with **entry, target, stop, conviction, and a risk-based size**.

```bash
python live_signals.py            # scan open markets, cache best signals
python live_signals.py --notify   # also push STRONG ones to your phone
python notify.py "test"           # test your messaging setup
```

The `com.rohitraj.signals.plist` launchd job runs every **30 min** and pushes only *new* STRONG signals (de-duplicated). Configure messaging in `.env`: **Telegram** (free — @BotFather token + chat id) or **WhatsApp** (free personal use via CallMeBot — message their number once to get an API key).

**Suggested size / leverage** is *risk-based*: the size at which hitting the stop costs ~1% of capital (`risk_budget ÷ stop-distance`). For normal stop distances this is **under 1×** — it is a risk *cap*, not a profit dial, and it's capped low (2× stocks / 3× crypto).

> [!CAUTION]
> A "signal" is a high-conviction *alignment of setups*, not a profit prediction. Entry/target/stop are a plan to research and **paper-trade** — the **stop is the most important number, and leverage is how accounts get liquidated.** Intraday signals especially get eaten by costs (proven in the backtests).

## Options ideas (on each stock's insight)

Opening any stock/index shows an **Options idea** card: it compares the model's **forecast vol** to **implied vol** (India VIX / VIX) and, with the directional bias, suggests a *strategy* — buy a call/put when vol looks cheap, sell a defined-risk spread when it looks rich. It's a strategy hint using a market-wide VIX as the IV proxy — **not specific strikes** (per-stock option chains need a paid/broker feed). Crypto is skipped.

> [!CAUTION]
> **Read the families, not the raw count.** Agents within a family are correlated (all oscillators agree in a trend), so "25 of 31" overstates independence. And the threshold is a frequency-vs-conviction dial: backtests showed **25/31 almost never fires** (you sit in cash), while looser thresholds trade more. Across symbols, consensus mainly **cut drawdown and overtrading** — on volatile crypto (BTC) it beat buy-and-hold on *Sharpe* with ~half the drawdown, but rarely beat it on raw return. Tune and backtest; don't assume more agents = more profit.

## Signal hit rate — the honest number (vs "80% correct" channels)

`signal_perf.py` (and the **📊 Hit rate** tab) measures the *real* directional accuracy of the signals over years of history, swept across conviction levels.

```bash
python signal_perf.py --market crypto_daily --years 3 --horizon 5
```

**Measured reality:** hit rates sit around **48–55%**, never the "80% always" that Telegram/WhatsApp channels advertise. Those channels mislead via cherry-picked screenshots, deleted losers, post-hoc entry widening, and pump-and-dump — and they hide *expectancy*, which is what actually decides profit. The tab shows the legitimate edge-lever: **raising the conviction bar** improves hit rate, expectancy and profit factor (crypto: 49%→55% hit, PF 1.19→1.88) at the cost of far fewer trades. **Expectancy and profit factor > 1 matter; hit rate alone does not.**

> The 45-agent committee already *is* the library of real, published setups (trend, momentum, mean-reversion, breakout, volume, statistical). The honest way to raise your odds isn't a secret channel strategy — it's selectivity + risk management, both measurable here.

## Training it to the best

See **[TRAINING.md](TRAINING.md)** for the full, honest playbook: the train →
measure → tune → re-validate loop, which knobs to turn, the selectivity lever,
and the discipline (out-of-sample, expectancy > hit rate, forward-validate on
paper) that separates real improvement from self-deception.

## Strategy training (learning to trade — honestly)

`train_strategy.py` is the closed loop that "learns the best way to trade": it sweeps the committee **consensus threshold** across a basket, **walk-forward**, and keeps the value that performs best **out-of-sample** — not the one that merely looked good in-sample.

```bash
python train_strategy.py --market crypto_daily --years 3
```

It optimises a **risk-adjusted** objective (Sharpe), validates OOS, and only writes `state/learned_policy.json` (adopted live) if the median OOS Sharpe clears a gate.

> [!IMPORTANT]
> It can — and does — **refuse to update.** On crypto it found threshold 24/45 looked best in aggregate, but the median OOS Sharpe didn't clear the gate, so it kept defaults. That refusal *is* the value: it learns the most defensible setting and proves whether it generalises. It optimises a real knob; **it does not (and cannot) guarantee profit.**

## Derivatives, leverage & options

- **📈 Derivatives** tab — the live list of **Binance USDT perpetual futures** (~530) with typical max leverage (BTC 125×, ETH 100×, …) and maintenance margins, plus an NSE F&O reference. Leverage is risk-tiered and broker-dependent — reference data, not advice.
- **Options** — every stock/index insight shows **Black-Scholes ATM premiums** (call/put, breakevens) priced at the model's forecast vol, and you can **paper-buy** a call or put. Open contracts live in a **paper options book** (Paper account tab), marked-to-model with live P&L and decay. It's a simulation (ATM, fixed expiry, European BS) — not a live broker chain.

> [!CAUTION]
> Leverage is the #1 cause of retail blow-ups, and bought options decay (theta). These tools exist so you can *learn the mechanics on paper* — the suggested position size is still risk-based (usually <1×).

## Self-learning (the agent adapts over time)

The agent improves itself with `auto_learn.py` — **walk-forward auto-retraining**, the honest version of "learns by itself":

```bash
python auto_learn.py            # learn + maybe promote new weights
python auto_learn.py --dry-run  # report only, write nothing
```

How it learns without fooling itself:

1. Rolls a window across history: **train** slice → **validate** slice → step forward.
2. On each train slice it searches a grid of ensemble weights and keeps the in-sample best.
3. It then scores those weights on the **next, unseen** slice — the out-of-sample (OOS) Sharpe is the only number that counts.
4. It reports the **overfitting tax** (in-sample minus out-of-sample). A big gap = the model learned noise.
5. **Promotion gate:** only if the *median OOS Sharpe* across folds clears `learning.min_oos_sharpe` does it write `state/learned_weights.json`. Otherwise it **refuses to learn** and keeps the hand-set defaults.

The live trader adopts promoted weights automatically (`use_learned_weights: true`); if nothing was promoted it silently uses defaults. The learning trajectory is logged to `state/learning_history.jsonl` and plotted to `backtest_results/learning_trajectory.png`.

> [!IMPORTANT]
> **Self-training does not create profit.** It keeps the strategy *adapted and validated*, not *guaranteed*. Per-fold OOS results swing wildly (that's markets); the gate just stops the agent from trading a blend that only worked on data it already saw. More training ≠ more money.

Schedule it weekly so fresh weights are ready before Monday:

```bash
cp com.rohitraj.autolearn.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.rohitraj.autolearn.plist
```

## Automation (scheduled jobs on your Mac)

Four optional launchd jobs keep everything fresh and hands-off. Install any of them by copying the plist to `~/Library/LaunchAgents/` and `launchctl load`-ing it:

| Job | Plist | Schedule | What it does |
|---|---|---|---|
| Daily trade | `com.rohitraj.livetradingagent.plist` | weekdays 15:45 | one paper decision cycle |
| Weekly learn | `com.rohitraj.autolearn.plist` | Sun 18:00 | walk-forward re-fit + promote weights |
| Nightly scan | `com.rohitraj.scan.plist` | daily 02:00 | refresh universe + re-rank Crypto & NSE top picks |
| Alerts | `com.rohitraj.alerts.plist` | every 30 min | check price-level alerts, fire macOS notifications |
| Live signals | `com.rohitraj.signals.plist` | every 30 min | scan open markets, push STRONG signals to Telegram/WhatsApp |

```bash
# example: enable the nightly scan + alert checker
cp com.rohitraj.scan.plist com.rohitraj.alerts.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.rohitraj.scan.plist
launchctl load ~/Library/LaunchAgents/com.rohitraj.alerts.plist
```

> The Mac must be awake when a job fires (`caffeinate` or Energy Saver). Alerts use `osascript` notifications — allow notifications for Script Editor/terminal if macOS prompts.

## Alerts

```bash
python alerts.py            # check all alerts once, notify on any that crossed
python alerts.py --list     # show every alert and its status
```

Set alerts from the dashboard's **🔔 Alerts** tab, or one-click from any stock's insight page ("Alert ▲ breakout" / "Alert ▼ breakdown"). They persist in `state/alerts.json`, latch once fired, and pop a native macOS notification.

## Run it autonomously (unattended, daily)

A `launchd` agent fires one cycle every weekday at 15:45 local time:

```bash
cp com.rohitraj.livetradingagent.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.rohitraj.livetradingagent.plist
```

Logs land in `state/launchd.out` / `state/launchd.err`; every decision is appended to `state/decisions.jsonl`. To stop:

```bash
launchctl unload ~/Library/LaunchAgents/com.rohitraj.livetradingagent.plist
```

> The plist times are your Mac's **local** time and assume US Eastern (15:45 ≈ 15 min before the US close). Adjust the `Hour` keys for your timezone. Your Mac must be awake at that time (consider `caffeinate` or Energy Saver settings). Alternatively run `python trader.py --loop` in a terminal to keep it firing daily in-process.

---

## Commands & flags

```bash
python trader.py --once                 # one cycle, then exit (used by launchd)
python trader.py --once --dry-run       # decide + log, place NO orders
python trader.py --loop                 # run forever, fire once per trading day
python trader.py --broker sim|alpaca    # override config's broker choice
python trader.py --reset-killswitch     # clear a tripped halt latch
python backtest.py --years N            # historical validation (active market)
python backtest.py --market NAME        # backtest a specific market preset
python backtest.py --all                # sweep every market -> leaderboard
python auto_learn.py                    # walk-forward self-learning pass
python auto_learn.py --dry-run          # learn + report, write nothing
python forecast.py --symbol "^NSEI"     # probabilistic next-day forecast (any symbol)
python screen.py --symbols "A,B,C"      # rank names + forecast + news context
python universe.py                      # cache all NSE + Binance symbols
python scan.py --segment binance        # offline rank a universe segment -> cache
```

## Configuration

Everything tunable lives in [`config.yaml`](config.yaml): the `universe`, signal `weights` and lookbacks, all `risk` limits, the decision time, and the broker mode. Nothing financial is hardcoded in source.

## State & files (git-ignored)

```
state/
├── sim_account.json    # SimBroker cash + positions
├── killswitch.json     # high-water mark + halt latch
├── decisions.jsonl     # append-only audit log of every decision
├── agent.log           # full run log
└── launchd.{out,err}   # scheduled-run output
backtest_results/equity.png
.env                    # your Alpaca paper keys (never committed)
```

---

## Limitations (read these)

- **Backtest ≠ live**: no slippage, market impact, or borrow costs modeled; the universe is hand-picked (survivorship bias).
- **Long-only, daily**: no shorting, no intraday reaction, fills at next opportunity.
- **Not proven to beat buy-and-hold** — see the backtest section. Its value proposition is risk-adjusted/defensive behavior, not raw return in a bull market.
- **yfinance data** can have gaps/adjustments; a production system would use a paid feed.

This is a learning/research project. **Not financial advice.**
