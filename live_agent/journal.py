"""
journal.py — Trade journal + real forward-performance tracking.

This is the discipline layer — and, per everything this project has learned, the
*actual* edge is here, not in the signals. You log a signal when you take it
(entry / target / stop / conviction), close it with the real exit, and the
journal computes your true, forward, out-of-sample record: win rate, expectancy,
profit factor, average R-multiple — broken down by market and signal source.

Why this matters: backtested hit-rates are ~48–55% (coin-flip-ish) and
profitability is decided by *expectancy and profit factor*, not win rate. A
forward journal is the only honest scoreboard — it can't be curve-fit, and it is
how you find out whether the 35–40-agent consensus actually helps *you*.

Persisted to ``state/journal.json``. No money moves here; it is a logbook.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
STORE = HERE / "state" / "journal.json"


# --------------------------------------------------------------------------- #
#  Storage
# --------------------------------------------------------------------------- #
def _load() -> List[Dict]:
    if STORE.exists():
        try:
            return json.loads(STORE.read_text())
        except Exception:
            return []
    return []


def _save(trades: List[Dict]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(trades, indent=2))


def _market_of(symbol: str) -> str:
    u = symbol.upper()
    if u.endswith(".NS") or u.endswith(".BO") or u.startswith("^NSE") or u.startswith("^BSE"):
        return "India"
    if u.endswith("USDT") or u.endswith("-USD") or u in ("BTC", "ETH"):
        return "Crypto"
    if u.startswith("^"):
        return "Index"
    return "US"


# --------------------------------------------------------------------------- #
#  Mutations
# --------------------------------------------------------------------------- #
def log(symbol: str, side: str, entry: float, target: Optional[float] = None,
        stop: Optional[float] = None, conviction: Optional[float] = None,
        source: str = "manual", note: str = "", market: Optional[str] = None) -> Dict:
    """Record a taken signal as an OPEN trade."""
    symbol = symbol.strip().upper()
    side = "BUY" if side.upper().startswith(("B", "L")) else "SELL"
    trade = {
        "id": format(int(time.time() * 1000) % 1_000_000_00, "08d"),
        "opened": time.strftime("%Y-%m-%d %H:%M"),
        "opened_ts": time.time(),
        "symbol": symbol, "market": market or _market_of(symbol),
        "side": side,
        "entry": _num(entry), "target": _num(target), "stop": _num(stop),
        "conviction": _num(conviction), "source": source, "note": note,
        "status": "open",
        "exit": None, "closed": None, "pnl_pct": None, "r_multiple": None,
        "outcome": None,
    }
    trades = _load()
    trades.append(trade)
    _save(trades)
    return trade


def close(trade_id: str, exit_price: float) -> Optional[Dict]:
    """Close a trade at ``exit_price`` and compute realised P&L / R-multiple."""
    trades = _load()
    for t in trades:
        if t["id"] == trade_id and t["status"] == "open":
            entry = t["entry"]
            xp = float(exit_price)
            long = t["side"] == "BUY"
            reward = (xp - entry) if long else (entry - xp)
            t["status"] = "closed"
            t["exit"] = round(xp, 6)
            t["closed"] = time.strftime("%Y-%m-%d %H:%M")
            t["closed_ts"] = time.time()
            t["pnl_pct"] = round(reward / entry * 100.0, 3) if entry else None
            if t["stop"] is not None and entry is not None:
                risk = abs(entry - t["stop"])
                t["r_multiple"] = round(reward / risk, 2) if risk else None
            t["outcome"] = "win" if reward > 0 else ("loss" if reward < 0 else "flat")
            _save(trades)
            return t
    return None


def delete(trade_id: str) -> bool:
    trades = _load()
    n = len(trades)
    trades = [t for t in trades if t["id"] != trade_id]
    _save(trades)
    return len(trades) < n


# --------------------------------------------------------------------------- #
#  Reads
# --------------------------------------------------------------------------- #
def list_trades(status: str = "all", prices: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Trades, newest first. Open trades get a live mark if ``prices`` is given."""
    trades = sorted(_load(), key=lambda t: t.get("opened_ts", 0), reverse=True)
    if status in ("open", "closed"):
        trades = [t for t in trades if t["status"] == status]
    if prices:
        for t in trades:
            if t["status"] != "open":
                continue
            px = prices.get(t["symbol"])
            if px is None or not t["entry"]:
                continue
            long = t["side"] == "BUY"
            reward = (px - t["entry"]) if long else (t["entry"] - px)
            t["live_price"] = round(float(px), 6)
            t["live_pnl_pct"] = round(reward / t["entry"] * 100.0, 3)
            # has the trade hit its target or stop?
            if t["target"] is not None:
                t["target_hit"] = (px >= t["target"]) if long else (px <= t["target"])
            if t["stop"] is not None:
                t["stop_hit"] = (px <= t["stop"]) if long else (px >= t["stop"])
    return trades


def stats() -> Dict:
    """Real forward performance across CLOSED trades."""
    trades = _load()
    closed = [t for t in trades if t["status"] == "closed" and t["pnl_pct"] is not None]
    open_n = sum(1 for t in trades if t["status"] == "open")
    if not closed:
        return {"total": len(trades), "open": open_n, "closed": 0,
                "win_rate": None, "expectancy_pct": None, "profit_factor": None,
                "avg_r": None, "wins": 0, "losses": 0, "best_pct": None,
                "worst_pct": None, "by_market": {}, "by_source": {}, "by_side": {},
                "r_curve": [], "total_r": None, "streak": 0,
                "max_win_streak": 0, "max_loss_streak": 0, "avg_hold_hrs": None,
                "note": "No closed trades yet — log signals and close them to build a real record."}
    wins = [t for t in closed if t["pnl_pct"] > 0]
    losses = [t for t in closed if t["pnl_pct"] < 0]
    gross_win = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))
    rs = [t["r_multiple"] for t in closed if t.get("r_multiple") is not None]

    def _seg(key):
        out: Dict[str, Dict] = {}
        for t in closed:
            k = t.get(key) or "?"
            b = out.setdefault(k, {"n": 0, "wins": 0, "sum_pct": 0.0})
            b["n"] += 1
            b["wins"] += 1 if t["pnl_pct"] > 0 else 0
            b["sum_pct"] += t["pnl_pct"]
        for k, b in out.items():
            b["win_rate"] = round(100.0 * b["wins"] / b["n"], 1)
            b["expectancy_pct"] = round(b["sum_pct"] / b["n"], 3)
        return out

    return {
        "total": len(trades), "open": open_n, "closed": len(closed),
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(100.0 * len(wins) / len(closed), 1),
        "expectancy_pct": round(sum(t["pnl_pct"] for t in closed) / len(closed), 3),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "avg_r": round(sum(rs) / len(rs), 2) if rs else None,
        "best_pct": round(max(t["pnl_pct"] for t in closed), 2),
        "worst_pct": round(min(t["pnl_pct"] for t in closed), 2),
        "by_market": _seg("market"), "by_source": _seg("source"), "by_side": _seg("side"),
        **_analytics(closed),
        "note": ("Forward, out-of-sample record — the only scoreboard that can't be "
                 "curve-fit. Expectancy & profit factor decide profitability, not win rate."),
    }


def _analytics(closed: List[Dict]) -> Dict:
    """Pro-review analytics over closed trades: the cumulative R curve (your real
    equity in risk units), win/loss streaks, and average hold time. This is what
    a trade review actually looks at — R-consistency, not P&L bragging rights."""
    seq = sorted(closed, key=lambda t: t.get("closed_ts") or t.get("opened_ts") or 0)
    # cumulative R (falls back to pnl%-sign when a trade had no stop logged)
    cum, r_curve, r_points = 0.0, [], []
    for t in seq:
        r = t.get("r_multiple")
        if r is None:
            continue
        cum += r
        r_points.append(r)
        r_curve.append({"t": t.get("closed") or t.get("opened"), "r": round(cum, 2)})
    # streaks over ALL closed trades (by pnl sign, chronological)
    cur = best_w = worst_l = 0
    for t in seq:
        if t["pnl_pct"] > 0:
            cur = cur + 1 if cur > 0 else 1
            best_w = max(best_w, cur)
        elif t["pnl_pct"] < 0:
            cur = cur - 1 if cur < 0 else -1
            worst_l = min(worst_l, cur)
    holds = [(t["closed_ts"] - t["opened_ts"]) / 3600.0 for t in seq
             if t.get("closed_ts") and t.get("opened_ts")]
    return {
        "r_curve": r_curve,
        "total_r": round(cum, 2) if r_points else None,
        "streak": cur,                       # +n = winning streak, -n = losing
        "max_win_streak": best_w,
        "max_loss_streak": abs(worst_l),
        "avg_hold_hrs": round(sum(holds) / len(holds), 1) if holds else None,
    }


def _num(x):
    try:
        v = float(x)
        return round(v, 6)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    # tiny self-test on a throwaway store
    STORE = HERE / "state" / "journal_selftest.json"
    if STORE.exists():
        STORE.unlink()
    t1 = log("BTCUSDT", "BUY", 61000, target=63000, stop=60000, conviction=68, source="consensus")
    t2 = log("ETHUSDT", "SELL", 1700, target=1600, stop=1750, conviction=64, source="consensus")
    close(t1["id"], 62500)   # +2.46%, +1.5R win
    close(t2["id"], 1720)    # -1.18% loss
    s = stats()
    print("win_rate", s["win_rate"], "expectancy%", s["expectancy_pct"],
          "profit_factor", s["profit_factor"], "avg_r", s["avg_r"])
    print("by_market", s["by_market"])
    STORE.unlink()
