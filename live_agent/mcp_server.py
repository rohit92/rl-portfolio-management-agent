"""
mcp_server.py — Talk to your trading platform through Claude (MCP server).

This exposes the whole agent as Model-Context-Protocol tools, so an MCP client
(Claude Desktop, Claude Code) can ask things like "what's the signal for BTCUSDT",
"scan crypto for top picks", "show my paper portfolio". It's the honest version of
"connect Claude to your trading app": read-only insights + *paper* actions only —
**no real orders, no real money.**

Run it standalone to sanity-check:
    python mcp_server.py            # starts a stdio MCP server (Ctrl-C to stop)

Connect it to Claude Desktop — add to claude_desktop_config.json:
    {
      "mcpServers": {
        "trading-agent": {
          "command": "/Users/rohitraj/rl-trading-agent/venv/bin/python",
          "args": ["/Users/rohitraj/rl-trading-agent/live_agent/mcp_server.py"]
        }
      }
    }
Then restart Claude Desktop and ask it about your markets.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

import agents as agents_mod
import markets
import options as options_mod
import signals as signals_mod
from broker import Order, SimBroker
from data_feed import fetch
from forecast import forecast_tomorrow
from levels import trigger_levels
from live_signals import market_status
from strategies import Ensemble
from trader import apply_learned_weights, load_config

HERE = Path(__file__).resolve().parent
mcp = FastMCP("trading-agent")

# ---- config (adopt trained weights + policy, like the dashboard) ----
_CFG = markets.resolve(load_config(HERE / "config.yaml"))
apply_learned_weights(_CFG)
_pol = HERE / "state" / "learned_policy.json"
if _pol.exists():
    try:
        p = json.loads(_pol.read_text())
        if p.get("adopted") and p.get("committee_threshold"):
            _CFG.setdefault("committee", {})["threshold"] = int(p["committee_threshold"])
    except Exception:
        pass
_FC = _CFG["forecast"]
_ENS = Ensemble(_CFG)
_PAPER = SimBroker(HERE / "state" / "sim_account.json",
                   float(_CFG.get("initial_cash", 100_000)),
                   float(_CFG["risk"]["transaction_cost"]))
_CACHE: dict = {}


def _sd(symbol: str):
    sym = symbol.strip().upper()
    now = time.time()
    if sym in _CACHE and now - _CACHE[sym][0] < 300:
        return _CACHE[sym][1]
    data = fetch([sym], _CFG, lookback_days=1200)
    sd = data.get(sym)
    _CACHE[sym] = (now, sd)
    return sd


# --------------------------------------------------------------------------- #
#  Read-only insight tools
# --------------------------------------------------------------------------- #

@mcp.tool()
def insight(symbol: str) -> dict:
    """Full snapshot for a symbol: price, next-day probabilistic range, P(up),
    forecast volatility, signal score, volatility regime, and breakout/breakdown
    trigger levels. Works for US/NSE stocks, indices (^NSEI), and crypto (BTCUSDT)."""
    sd = _sd(symbol)
    if sd is None:
        return {"error": f"No data for {symbol}"}
    f = forecast_tomorrow(sd.frame["close"], _FC)
    asc = _ENS.score_one(sd)
    lv = trigger_levels(sd, f["exp_abs_move_pts"] / f["level"])
    return {
        "symbol": symbol.upper(), "last": round(f["level"], 2),
        "forecast_vol_annual_pct": round(f["sigma_next_ann"] * 100, 1),
        "p_up_pct": round(f["p_up"] * 100, 0),
        "expected_move_pct": round(f["exp_abs_move_pts"] / f["level"] * 100, 2),
        "next_day_range": {"low_5pct": round(f["level"] + f["p05"], 2),
                           "mid": round(f["level"] + f["p50"], 2),
                           "high_95pct": round(f["level"] + f["p95"], 2)},
        "signal_score": round(asc.composite, 2),
        "regime": ("HIGH-vol" if sd.frame["vol"].iloc[-1] >
                   1.5 * sd.frame["vol"].median() else "normal"),
        "breakout_level": round(lv["breakout"]["level"], 2),
        "breakdown_level": round(lv["breakdown"]["level"], 2),
        "note": "Direction is ~a coin-flip; the trustworthy part is the range/vol.",
    }


@mcp.tool()
def signal(symbol: str) -> dict:
    """Actionable BUY/SELL signal for a symbol with conviction, entry, target,
    stop and a risk-based suggested size. Returns 'no actionable signal' when the
    setup is weak. Not financial advice — the stop is the key number."""
    sigs = signals_mod.generate([symbol.strip().upper()], _CFG)
    if not sigs:
        return {"symbol": symbol.upper(), "signal": "no actionable signal"}
    s = sigs[0]
    return {"symbol": s["symbol"], "side": s["side"], "conviction_0_100": s["conviction"],
            "entry": s["entry"], "target": s["target"], "stop": s["stop"],
            "suggested_size_x": s["leverage"], "committee": s["committee"],
            "note": "Not advice. High leverage liquidates accounts; respect the stop."}


@mcp.tool()
def committee(symbol: str) -> dict:
    """The 45-agent committee verdict for a symbol: how many of the 45 independent
    setups vote BUY vs SELL, the per-family net, and the consensus verdict."""
    sd = _sd(symbol)
    if sd is None:
        return {"error": f"No data for {symbol}"}
    L = agents_mod.Committee(_CFG).latest(sd.frame)
    return {"symbol": symbol.upper(), "buy_votes": L["long"], "sell_votes": L["short"],
            "of_total": L["total"], "verdict": L["verdict"], "families": L["families"]}


@mcp.tool()
def forecast(symbol: str) -> dict:
    """Probabilistic next-day forecast: expected move, P(up), and the 5/50/95%
    price range. Calibrated — its 90% intervals contain the move ~90% of the time."""
    sd = _sd(symbol)
    if sd is None:
        return {"error": f"No data for {symbol}"}
    f = forecast_tomorrow(sd.frame["close"], _FC)
    return {"symbol": symbol.upper(), "last": round(f["level"], 2),
            "p_up_pct": round(f["p_up"] * 100, 0),
            "expected_move_pct": round(f["exp_abs_move_pts"] / f["level"] * 100, 2),
            "forecast_vol_annual_pct": round(f["sigma_next_ann"] * 100, 1),
            "range_5_50_95": [round(f["level"] + f["p05"], 2),
                              round(f["level"] + f["p50"], 2),
                              round(f["level"] + f["p95"], 2)]}


@mcp.tool()
def top_picks(segment: str = "binance") -> dict:
    """Cached top-ranked candidates for a market segment ('binance' crypto or
    'nse' Indian stocks), produced by the offline scanner. Ranked by signal — a
    research shortlist, NOT a buy list."""
    path = HERE / "state" / f"scan_results_{segment}.json"
    if not path.exists():
        return {"error": f"No scan cached for '{segment}'. Run: python scan.py --segment {segment}"}
    d = json.loads(path.read_text())
    return {"segment": segment, "scanned_at": d.get("scanned_at"),
            "top": [{"symbol": r["symbol"], "score": r["composite"],
                     "candidate": r["is_candidate"]} for r in d.get("rows", [])[:15]]}


@mcp.tool()
def options_idea(symbol: str) -> dict:
    """Options strategy hint for a stock/index: forecast vol vs implied (VIX),
    whether premium looks rich/cheap, and Black-Scholes ATM call/put prices.
    A strategy hint using market VIX as the IV proxy — not specific strikes."""
    sd = _sd(symbol)
    if sd is None:
        return {"error": f"No data for {symbol}"}
    f = forecast_tomorrow(sd.frame["close"], _FC)
    q = options_mod.quote(f["level"], f["sigma_next_ann"], days=30)
    return {"symbol": symbol.upper(), "forecast_vol_pct": round(f["sigma_next_ann"] * 100, 1),
            "atm_call": q["call"], "atm_put": q["put"], "expiry_days": 30,
            "note": "Simulated BS pricing; real options need a broker chain."}


@mcp.tool()
def market_open() -> dict:
    """Which markets are open right now (crypto is 24/7; NSE only weekdays
    09:15-15:30 IST)."""
    return market_status()


# --------------------------------------------------------------------------- #
#  Paper actions (simulated money only)
# --------------------------------------------------------------------------- #

@mcp.tool()
def paper_portfolio() -> dict:
    """Your SIMULATED paper account: cash, positions, and total equity. No real money."""
    positions = _PAPER.get_positions()
    prices, held = {}, []
    for sym, qty in positions.items():
        sd = _sd(sym)
        px = sd.last_close if sd else 0.0
        prices[sym] = px
        held.append({"symbol": sym, "qty": round(qty, 4), "value": round(qty * px, 2)})
    return {"cash": round(_PAPER.get_cash(), 2),
            "equity": round(_PAPER.get_equity(prices), 2), "positions": held,
            "note": "Simulated money only."}


@mcp.tool()
def paper_buy(symbol: str, dollar_amount: float = 1000.0) -> dict:
    """Buy a symbol with SIMULATED money (paper account). No real order is placed."""
    sd = _sd(symbol)
    if sd is None:
        return {"error": f"No data for {symbol}"}
    px = sd.last_close
    qty = float(dollar_amount) / px
    if dollar_amount > _PAPER.get_cash():
        return {"error": "Insufficient paper cash."}
    _PAPER.execute([Order(symbol=symbol.upper(), side="buy", qty=qty)], {symbol.upper(): px})
    return {"ok": True, "bought": symbol.upper(), "qty": round(qty, 4),
            "price": round(px, 2), "note": "PAPER trade — no real money."}


@mcp.tool()
def hit_rate(market: str = "crypto_daily") -> dict:
    """The honest measured directional hit-rate + expectancy of the signal rule
    over history for a market ('crypto_daily', 'us_tech_daily', 'india_daily').
    Reality is ~48-55% — NOT the '80%' that signal channels claim. Slow (~30s)."""
    import signal_perf
    cfg = markets.resolve(load_config(HERE / "config.yaml"), market)
    r = signal_perf.analyze(cfg, years=3, horizon=5, conviction_gate=50)
    return {"market": market, "hit_rate_pct": r.get("hit_rate"),
            "base_up_rate_pct": r.get("base_up_rate"),
            "expectancy_pct": r.get("expectancy_pct"),
            "profit_factor": r.get("profit_factor"), "n_signals": r.get("n_signals"),
            "note": "Expectancy & profit factor decide profit, not hit rate."}


if __name__ == "__main__":
    mcp.run()
