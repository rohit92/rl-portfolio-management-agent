"""
futures_feed.py — Live Binance USDT perpetual futures board.

One place to see the crypto-futures tape: mark price, 24h change, funding rate
(and when the next funding is), open interest, and the typical max leverage.

Data is REAL and live from Binance's public futures API (no key). Two bulk calls
cover price + funding for ~800 symbols; open interest is fetched per-symbol only
for the handful shown (kept small so the board stays snappy).

Honest note: funding rate is the cost of holding a perp, not a signal. A high
positive funding just means longs are crowded (paying shorts) — often a *fade*,
not a follow. High leverage is the #1 cause of blow-ups; the leverage column is
reference, not encouragement.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import requests

from derivatives import _CRYPTO_MAX, _CRYPTO_DEFAULT

logger = logging.getLogger("futures_feed")

_BASE = "https://fapi.binance.com"
_TICKER = _BASE + "/fapi/v1/ticker/24hr"
_PREMIUM = _BASE + "/fapi/v1/premiumIndex"
_OI = _BASE + "/fapi/v1/openInterest"

_CACHE: Dict[str, tuple] = {}
_TTL = 5  # seconds — the whole point is that it is live


def _get(url, params=None):
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def board(limit: int = 25, sort: str = "volume", quote: str = "USDT") -> Dict:
    """Live futures board, top ``limit`` USDT perps.

    sort: ``volume`` (most liquid, default) | ``gainers`` | ``losers`` | ``funding``.
    """
    key = f"{limit}:{sort}:{quote}"
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < _TTL:
        return _CACHE[key][1]

    try:
        tickers = _get(_TICKER)
        premium = _get(_PREMIUM)
    except Exception as exc:
        logger.warning("futures board fetch failed: %s", exc)
        return {"error": f"Binance futures feed unavailable ({exc})", "rows": []}

    prem = {p["symbol"]: p for p in premium}
    rows: List[Dict] = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith(quote):
            continue
        p = prem.get(sym, {})
        try:
            rows.append({
                "symbol": sym,
                "base": sym[: -len(quote)],
                "mark": float(p.get("markPrice", t.get("lastPrice", 0)) or 0),
                "last": float(t.get("lastPrice", 0) or 0),
                "chg24h": float(t.get("priceChangePercent", 0) or 0),
                "high": float(t.get("highPrice", 0) or 0),
                "low": float(t.get("lowPrice", 0) or 0),
                "quote_vol": float(t.get("quoteVolume", 0) or 0),
                "funding": float(p.get("lastFundingRate", 0) or 0) * 100.0,   # %
                "next_funding": int(p.get("nextFundingTime", 0) or 0),
                "max_lev": _CRYPTO_MAX.get(sym[: -len(quote)], _CRYPTO_DEFAULT),
            })
        except (TypeError, ValueError):
            continue

    if sort == "gainers":
        rows.sort(key=lambda r: r["chg24h"], reverse=True)
    elif sort == "losers":
        rows.sort(key=lambda r: r["chg24h"])
    elif sort == "funding":
        rows.sort(key=lambda r: abs(r["funding"]), reverse=True)
    else:  # volume
        rows.sort(key=lambda r: r["quote_vol"], reverse=True)
    rows = rows[:limit]

    # open interest only for the shown rows (per-symbol endpoint) — fetched in
    # parallel so the board stays snappy despite ~20 individual calls.
    def _add_oi(r):
        try:
            oi = _get(_OI, {"symbol": r["symbol"]})
            r["open_interest"] = float(oi.get("openInterest", 0) or 0)
            r["oi_notional"] = r["open_interest"] * r["mark"]
        except Exception:
            r["open_interest"] = None
            r["oi_notional"] = None

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(_add_oi, rows))

    payload = {
        "updated": time.strftime("%H:%M:%S"),
        "sort": sort, "count": len(rows), "rows": rows,
    }
    _CACHE[key] = (now, payload)
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    b = board(limit=10, sort="volume")
    print(f"Top {b['count']} by volume @ {b['updated']}")
    print(f"{'SYM':<12}{'MARK':>12}{'24h%':>8}{'FUND%':>9}{'MAXLEV':>8}")
    for r in b["rows"]:
        print(f"{r['symbol']:<12}{r['mark']:>12.4f}{r['chg24h']:>8.2f}"
              f"{r['funding']:>9.4f}{r['max_lev']:>7}x")
