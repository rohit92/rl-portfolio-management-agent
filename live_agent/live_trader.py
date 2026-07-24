"""
live_trader.py — Autonomous live paper-trading session you can watch perform.

You provide the (paper) capital, pick a market (crypto 24/7 or NSE intraday),
optionally leverage, and a duration. A background loop then, every poll interval:

  1. marks open positions to the latest price (live P&L),
  2. closes any that hit their stop / target / liquidation,
  3. opens new positions from STRONG signals (committee + ensemble agree),
  4. updates the equity curve.

It is 100% simulated money. Leverage is modelled with a real liquidation rule so
you *feel* the risk — it is a teaching tool, not encouragement. NSE intraday is
gated to market hours (09:15–15:30 IST) and force-squared-off at the close.

Run inside the dashboard (Start/Stop on the 🟢 Live trading tab) — this module is
driven by webapp.py, not a CLI.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import signals as sig_mod
from data_feed import fetch, latest_prices

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("live_trader")

_CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
           "ADAUSDT", "AVAXUSDT"]
_NSE = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "SBIN.NS", "BHARTIARTL.NS", "LT.NS"]


def _nse_open() -> bool:
    now = dt.datetime.now(IST)
    return now.weekday() < 5 and dt.time(9, 15) <= now.time() <= dt.time(15, 30)


class LiveSession:
    """A single running paper-trading session (one per process)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.reset()

    def reset(self) -> None:
        self.status = "idle"
        self.cfg: Dict = {}
        self.cash = 0.0
        self.capital = 0.0
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.curve: List[Dict] = []
        self.started_at = None
        self.ends_at = None
        self.last_update = None
        self.message = ""
        self.hwm = 0.0
        self.halt_reason = ""

    # ---- lifecycle ----
    def start(self, app_cfg: dict, capital: float, market: str, leverage: float,
              duration_min: int, poll_seconds: int, symbols: Optional[List[str]],
              max_positions: int = 4, risk_fraction: float = 0.15,
              conviction: int = 55, allow_short: bool = True,
              max_drawdown_pct: float = 25.0,
              fee_rate: Optional[float] = None) -> Dict:
        if self.status == "running":
            return {"error": "A session is already running. Stop it first."}
        self.reset()
        self.capital = self.cash = float(capital)
        self.hwm = float(capital)
        self.cfg = {
            "app_cfg": app_cfg, "market": market,
            # Paper sim allows up to 125x (like Binance) so you can SEE liquidation.
            "leverage": max(1.0, min(float(leverage), 125.0)),
            "poll": max(15, int(poll_seconds)),
            "symbols": symbols or (_CRYPTO if market == "crypto" else _NSE),
            "interval": "5m" if market == "crypto" else "15m",
            "max_positions": int(max_positions), "risk_fraction": float(risk_fraction),
            "conviction": int(conviction), "allow_short": bool(allow_short),
            "max_dd": max(0.02, float(max_drawdown_pct) / 100.0),
            # Binance-style fee charged on each fill's notional (taker ~0.1% spot).
            "fee_rate": float(fee_rate) if fee_rate is not None
                        else float(app_cfg["risk"]["transaction_cost"]),
        }
        self.status = "running"
        self.started_at = time.time()
        self.ends_at = self.started_at + duration_min * 60
        self.curve = [{"t": self._clock(), "equity": round(self.capital, 2)}]
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Live session started: %s, $%.0f, %sx, %dm",
                    market, capital, self.cfg["leverage"], duration_min)
        return self.snapshot()

    def stop(self) -> Dict:
        with self._lock:
            if self.status == "running":
                self.status = "stopping"
                self.message = "Stopping — squaring off positions…"
        return {"ok": True}

    # ---- the loop ----
    def _loop(self) -> None:
        cfg = self.cfg
        app_cfg = cfg["app_cfg"]
        while True:
            if self.status == "stopping" or time.time() >= self.ends_at:
                self._square_off("session ended")
                with self._lock:
                    self.status = "done"
                    self.message = "Session complete."
                return
            try:
                self._tick(app_cfg)
            except Exception as exc:
                logger.warning("tick error: %s", exc)
            # Circuit breaker: drawdown limit hit during the tick → halt.
            if self.status == "halt_pending":
                self._square_off("circuit breaker")
                with self._lock:
                    self.status = "halted"
                    self.message = self.halt_reason
                logger.warning("Circuit breaker tripped: %s", self.halt_reason)
                return
            time.sleep(cfg["poll"])

    def _tick(self, app_cfg: dict) -> None:
        cfg = self.cfg
        nse = cfg["market"] != "crypto"
        # Enough history that indicators (≈165-bar warm-up) + forecast survive:
        # 15m bars need ~45 days, 5m bars ~10 days.
        lb = 45 if cfg["interval"] == "15m" else 10
        data = fetch(cfg["symbols"], app_cfg, lookback_days=lb, interval=cfg["interval"])
        prices = latest_prices(data)

        with self._lock:
            # 1) manage open positions
            for sym in list(self.positions):
                px = prices.get(sym)
                if px is None:
                    continue
                self._maybe_exit(sym, px)
            # NSE: square off everything near the close
            if nse and not _nse_open():
                self._square_off("NSE close")

            # 2) open new positions from STRONG signals (if room + market open)
            can_trade = (not nse) or _nse_open()
            slots = cfg["max_positions"] - len(self.positions)
            if can_trade and slots > 0:
                free = [s for s in data if s not in self.positions]
                sigs = sig_mod.generate(free, app_cfg) if free else []
                for s in sigs:
                    if slots <= 0:
                        break
                    if s["conviction"] < cfg["conviction"]:
                        continue
                    if s["side"] == "SELL" and not cfg["allow_short"]:
                        continue
                    self._open(s, prices.get(s["symbol"], s["price"]))
                    slots -= 1

            # 3) mark equity
            eq = self._equity(prices)
            self.last_update = self._clock()
            self.curve.append({"t": self.last_update, "equity": round(eq, 2)})
            self.curve = self.curve[-400:]

            # 4) circuit breaker — auto-halt if drawdown limit breached.
            self.hwm = max(self.hwm, eq)
            dd = (self.hwm - eq) / self.hwm if self.hwm > 0 else 0.0
            if dd >= cfg["max_dd"]:
                self.status = "halt_pending"
                self.halt_reason = (f"Circuit breaker: −{dd*100:.1f}% from peak "
                                    f"(limit {cfg['max_dd']*100:.0f}%) — squared off & halted.")

    # ---- position mechanics (leverage + liquidation) ----
    def _open(self, sig: Dict, price: float) -> None:
        cfg = self.cfg
        margin = self.capital * cfg["risk_fraction"]
        if margin > self.cash or price <= 0:
            return
        lev = cfg["leverage"]
        qty = (margin * lev) / price
        risk = max(sig.get("exp_move_pct", 0.01), 0.005)
        long = sig["side"] == "BUY"
        sign = 1 if long else -1
        stop = price * (1 - sign * 1.5 * risk)
        target = price * (1 + sign * 2.5 * risk)
        fee = qty * price * cfg["fee_rate"]          # Binance fee on the notional
        self.cash -= margin + fee
        self.positions[sig["symbol"]] = {
            "side": sig["side"], "qty": qty, "entry": price, "margin": margin,
            "leverage": lev, "stop": stop, "target": target, "open_fee": fee,
            "opened": self._clock(), "conviction": sig["conviction"],
        }
        self.trades.append({"t": self._clock(), "symbol": sig["symbol"],
                            "action": f"OPEN {sig['side']}", "price": round(price, 4),
                            "qty": round(qty, 4), "pnl": None})

    def _maybe_exit(self, sym: str, px: float) -> None:
        p = self.positions[sym]
        long = p["side"] == "BUY"
        unrl = (px - p["entry"]) * p["qty"] * (1 if long else -1)
        reason = None
        if unrl <= -p["margin"] * 0.95:
            reason = "LIQUIDATED"
        elif (long and px <= p["stop"]) or (not long and px >= p["stop"]):
            reason = "stop"
        elif (long and px >= p["target"]) or (not long and px <= p["target"]):
            reason = "target"
        if reason:
            self._close(sym, px, reason)

    def _close(self, sym: str, px: float, reason: str) -> None:
        p = self.positions.pop(sym)
        long = p["side"] == "BUY"
        close_fee = p["qty"] * px * self.cfg["fee_rate"]   # Binance fee on exit
        pnl = (px - p["entry"]) * p["qty"] * (1 if long else -1) - close_fee
        pnl = max(pnl, -p["margin"])  # cannot lose more than margin
        self.cash += p["margin"] + pnl
        self.trades.append({"t": self._clock(), "symbol": sym,
                            "action": f"CLOSE ({reason})", "price": round(px, 4),
                            "qty": round(p["qty"], 4), "pnl": round(pnl, 2)})

    def _square_off(self, reason: str) -> None:
        if not self.positions:
            return
        lb = 45 if self.cfg["interval"] == "15m" else 10
        data = fetch(list(self.positions), self.cfg["app_cfg"], lookback_days=lb,
                     interval=self.cfg["interval"])
        prices = latest_prices(data)
        for sym in list(self.positions):
            self._close(sym, prices.get(sym, self.positions[sym]["entry"]), reason)

    def _equity(self, prices: Dict[str, float]) -> float:
        eq = self.cash
        for sym, p in self.positions.items():
            px = prices.get(sym, p["entry"])
            long = p["side"] == "BUY"
            unrl = (px - p["entry"]) * p["qty"] * (1 if long else -1)
            eq += p["margin"] + max(unrl, -p["margin"])
        return eq

    # ---- reporting ----
    def _clock(self) -> str:
        return dt.datetime.now(IST).strftime("%H:%M:%S")

    def snapshot(self) -> Dict:
        with self._lock:
            eq = self.curve[-1]["equity"] if self.curve else self.capital
            open_pos = []
            for sym, p in self.positions.items():
                open_pos.append({"symbol": sym, "side": p["side"],
                                 "entry": round(p["entry"], 4), "qty": round(p["qty"], 4),
                                 "stop": round(p["stop"], 4), "target": round(p["target"], 4),
                                 "leverage": p["leverage"], "conviction": p["conviction"]})
            realized = sum(t["pnl"] for t in self.trades if t["pnl"] is not None)
            secs_left = max(0, int((self.ends_at or 0) - time.time())) if self.status == "running" else 0
            return {
                "status": self.status, "message": self.message,
                "capital": round(self.capital, 2), "cash": round(self.cash, 2),
                "equity": round(eq, 2),
                "pnl": round(eq - self.capital, 2),
                "pnl_pct": round((eq / self.capital - 1) * 100, 2) if self.capital else 0,
                "realized": round(realized, 2),
                "hwm": round(self.hwm, 2),
                "drawdown_pct": round((1 - eq / self.hwm) * 100, 2) if self.hwm else 0,
                "max_dd_pct": round(self.cfg.get("max_dd", 0.25) * 100, 0),
                "market": self.cfg.get("market"), "leverage": self.cfg.get("leverage"),
                "positions": open_pos, "n_positions": len(open_pos),
                "trades": self.trades[-20:][::-1],
                "curve": self.curve[-200:],
                "seconds_left": secs_left,
                "nse_open": _nse_open(),
                "last_update": self.last_update,
            }


# Module singleton
SESSION = LiveSession()
