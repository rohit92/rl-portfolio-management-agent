"""
options.py — Black-Scholes option pricing + a simple paper options book.

Lets the app show real option *premiums* and lets you paper-trade calls/puts.
Prices use the model's forecast volatility (the same calibrated number the
options-lens compares to implied vol). This is a simulation: ATM, a fixed
expiry, European Black-Scholes — not a live broker chain, but enough to learn
the mechanics (premium, decay, breakeven) without risking anything.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
BOOK = HERE / "state" / "options_book.json"
RISK_FREE = 0.06  # ~India 10y; close enough for a teaching sim


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S: float, K: float, T: float, sigma: float,
             kind: str = "call", r: float = RISK_FREE) -> float:
    """European Black-Scholes price. T in years, sigma annualised (fraction)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K) if kind == "call" else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def quote(S: float, sigma_ann: float, days: int = 30) -> Dict:
    """ATM call & put premiums + breakevens for a `days`-to-expiry option."""
    T = days / 365.0
    call = bs_price(S, S, T, sigma_ann, "call")
    put = bs_price(S, S, T, sigma_ann, "put")
    return {
        "spot": round(S, 2), "strike_atm": round(S, 2), "expiry_days": days,
        "iv_used_pct": round(sigma_ann * 100, 1),
        "call": round(call, 2), "put": round(put, 2),
        "call_breakeven": round(S + call, 2), "put_breakeven": round(S - put, 2),
    }


# --------------------------- paper options book ----------------------------- #

def _load() -> List[Dict]:
    return json.loads(BOOK.read_text()) if BOOK.exists() else []


def _save(b: List[Dict]) -> None:
    BOOK.parent.mkdir(parents=True, exist_ok=True)
    BOOK.write_text(json.dumps(b))


def buy(symbol: str, kind: str, S: float, sigma_ann: float,
        qty: int = 1, days: int = 30) -> Dict:
    """Paper-buy an ATM option at its BS premium."""
    T = days / 365.0
    premium = bs_price(S, S, T, sigma_ann, kind)
    pos = {
        "id": format(int(time.time() * 1000) % 100000000, "08d"),
        "symbol": symbol.upper(), "kind": kind, "strike": round(S, 2),
        "entry_premium": round(premium, 2), "qty": int(qty),
        "iv_pct": round(sigma_ann * 100, 1),
        "opened": time.strftime("%Y-%m-%d %H:%M"),
        "expiry_ts": time.time() + days * 86400,
    }
    book = _load(); book.append(pos); _save(book)
    return pos


def book_with_marks(prices: Dict[str, float], vols: Dict[str, float]) -> List[Dict]:
    """Re-price open paper options to current spot/vol (mark-to-model)."""
    out = []
    now = time.time()
    for p in _load():
        S = prices.get(p["symbol"], p["strike"])
        sigma = vols.get(p["symbol"], p["iv_pct"] / 100.0)
        days_left = max(0.0, (p["expiry_ts"] - now) / 86400.0)
        mark = bs_price(S, p["strike"], days_left / 365.0, sigma, p["kind"])
        pnl = (mark - p["entry_premium"]) * p["qty"]
        out.append({**p, "spot": round(S, 2), "mark": round(mark, 2),
                    "days_left": round(days_left, 1), "pnl": round(pnl, 2)})
    return out


def close(opt_id: str) -> None:
    _save([p for p in _load() if p["id"] != opt_id])
