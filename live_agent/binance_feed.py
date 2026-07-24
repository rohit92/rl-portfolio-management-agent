"""
binance_feed.py — Real OHLCV from Binance's public klines API (no key needed).

So "all Binance coins" genuinely means all of them — not just the handful Yahoo
happens to carry. Any crypto symbol (``BTCUSDT``, ``BTC-USD``, or bare ``BTC``)
is resolved to a Binance USDT pair and fetched directly from Binance, returning
the same lowercase OHLCV frame the rest of the pipeline expects.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_KLINES = "https://api.binance.com/api/v3/klines"
_MAX = 1000  # Binance max bars per request

# yfinance-style interval -> Binance interval
_INTERVAL = {"1d": "1d", "1wk": "1w", "1h": "1h", "1m": "1m",
             "5m": "5m", "15m": "15m", "30m": "30m", "4h": "4h", "1mo": "1M"}


def to_pair(symbol: str) -> str:
    """Resolve a crypto symbol to a Binance USDT pair (e.g. BTC -> BTCUSDT)."""
    s = symbol.upper().replace("-", "")
    if s.endswith("USDT"):
        return s
    if s.endswith("USD"):
        s = s[:-3]
    return f"{s}USDT"


def is_crypto(symbol: str, binance_set: Optional[set] = None) -> bool:
    """Heuristic: does this symbol look like a Binance crypto pair?"""
    u = symbol.upper()
    if u.startswith("^") or u.endswith(".NS") or u.endswith(".BO"):
        return False  # index or Indian equity
    if u.endswith("USDT") or u.endswith("-USD") or u.endswith("-USDT"):
        return True
    if binance_set and to_pair(symbol) in binance_set:
        return True
    return False


def get_ohlcv(symbol: str, interval: str = "1d",
              lookback_days: int = 2500) -> pd.DataFrame:
    """Daily (or other) OHLCV for a Binance coin as a lowercase frame.

    Paginates past Binance's 1000-bar limit. Returns an empty frame on failure.
    """
    pair = to_pair(symbol)
    bint = _INTERVAL.get(interval, "1d")
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=lookback_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    rows = []
    cursor = start_ms
    while cursor < end_ms:
        try:
            r = requests.get(_KLINES, params={
                "symbol": pair, "interval": bint,
                "startTime": cursor, "limit": _MAX}, timeout=20)
            r.raise_for_status()
            batch = r.json()
        except Exception as exc:
            logger.warning("Binance klines failed for %s: %s", pair, exc)
            break
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if len(batch) < _MAX:
            break
        cursor = last_open + 1  # advance past the last bar

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbav", "tqav", "ignore"])
    df.index = pd.to_datetime(df["open_time"], unit="ms")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["open", "high", "low", "close", "volume"]].dropna()
