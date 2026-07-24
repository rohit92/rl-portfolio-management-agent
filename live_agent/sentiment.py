"""
sentiment.py — Market-specific sentiment gauges that real traders watch.

Different markets have different "mood" indicators, and this routes each symbol to
the right one:

  • Crypto → the **Fear & Greed Index** (alternative.me, free). A classic
    *contrarian* gauge: extreme fear (<25) historically marks bottoms, extreme
    greed (>75) marks froth.
  • NSE / Indian equities → **India VIX** (the fear gauge).
  • US equities/indices → **VIX**.
  • Any symbol → recent **news-headline lean** (reuses news.py).

Honest caveat: sentiment is a *weak, noisy* edge — useful as confirmation/context,
not a standalone signal. We use it to *sharpen conviction* (a technical signal that
agrees with sentiment is higher quality), and we surface the raw gauges so you can
see when price and mood diverge (like crypto in 'extreme fear' on a down-trend).
All values cached ~30 min (they move slowly).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)
_CACHE: Dict[str, tuple] = {}
_TTL = 1800  # 30 min


def _cached(key: str, fn):
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < _TTL:
        return _CACHE[key][1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


# --------------------------------------------------------------------------- #
#  Crypto Fear & Greed
# --------------------------------------------------------------------------- #

def crypto_fng() -> Optional[Dict]:
    """Current crypto Fear & Greed: value 0-100, classification, contrarian lean."""
    def _fetch():
        try:
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=12)
            d = r.json()["data"][0]
            v = int(d["value"])
            # Contrarian lean in [-1, +1]: fear -> bullish, greed -> bearish.
            lean = max(-1.0, min(1.0, (50 - v) / 35.0))
            return {"value": v, "label": d["value_classification"],
                    "lean": round(lean, 2), "source": "Crypto Fear & Greed"}
        except Exception as exc:
            logger.warning("F&G fetch failed: %s", exc)
            return None
    return _cached("fng", _fetch)


def crypto_fng_history(limit: int = 400) -> List[Dict]:
    def _fetch():
        try:
            r = requests.get(f"https://api.alternative.me/fng/?limit={limit}", timeout=15)
            return [{"ts": int(x["timestamp"]), "value": int(x["value"])}
                    for x in r.json()["data"]]
        except Exception:
            return []
    return _cached(f"fng_hist_{limit}", _fetch)


# --------------------------------------------------------------------------- #
#  Equity VIX
# --------------------------------------------------------------------------- #

def vix(symbol_is_india: bool) -> Optional[Dict]:
    sym = "^INDIAVIX" if symbol_is_india else "^VIX"

    def _fetch():
        try:
            import yfinance as yf
            v = float(yf.Ticker(sym).history(period="5d")["Close"].iloc[-1])
            # Regime + a mild lean (calm = risk-on +, fear = risk-off -). VIX is
            # ambiguous directionally, so the lean is intentionally small.
            base = 13.0 if symbol_is_india else 18.0
            lean = max(-0.6, min(0.6, (base - v) / 12.0))
            regime = "calm" if v < base * 0.85 else "elevated" if v < base * 1.4 else "FEAR"
            return {"value": round(v, 2), "label": regime, "lean": round(lean, 2),
                    "source": sym}
        except Exception as exc:
            logger.warning("VIX fetch failed: %s", exc)
            return None
    return _cached(f"vix_{sym}", _fetch)


# --------------------------------------------------------------------------- #
#  News lean
# --------------------------------------------------------------------------- #

def news_lean(symbol: str) -> Optional[Dict]:
    def _fetch():
        try:
            import news
            heads = news.get_headlines(symbol, limit=6)
            if not heads:
                return None
            avg = sum(h["lean"] for h in heads) / len(heads)
            return {"value": round(avg, 2), "label": "headline lean",
                    "lean": round(max(-1, min(1, avg)), 2), "source": "Yahoo news",
                    "n": len(heads)}
        except Exception:
            return None
    return _cached(f"news_{symbol.upper()}", _fetch)


# --------------------------------------------------------------------------- #
#  Combined market sentiment for a symbol
# --------------------------------------------------------------------------- #

def _is_crypto(symbol: str) -> bool:
    try:
        import binance_feed
        return binance_feed.is_crypto(symbol)
    except Exception:
        return symbol.upper().endswith("USDT")


def market_sentiment(symbol: str, include_news: bool = True) -> Dict:
    """The right sentiment gauges for this symbol + a blended lean in [-1, +1].

    Set include_news=False in hot loops (per-symbol news fetches are slow); the
    primary gauge (F&G / VIX) is globally cached and fast.
    """
    sym = symbol.strip().upper()
    gauges = []
    if _is_crypto(sym):
        g = crypto_fng()
        if g:
            gauges.append(g)
    else:
        india = sym.endswith(".NS") or sym.startswith("^NSE") or sym == "^BSESN"
        g = vix(india)
        if g:
            gauges.append(g)
    if include_news:
        nl = news_lean(sym)
        if nl:
            gauges.append(nl)

    if not gauges:
        return {"symbol": sym, "lean": 0.0, "gauges": [], "summary": "no sentiment data"}
    # Weight the primary gauge (F&G / VIX) more than news.
    weights = [0.7] + [0.3] * (len(gauges) - 1) if len(gauges) > 1 else [1.0]
    lean = sum(w * g["lean"] for w, g in zip(weights, gauges)) / sum(weights)
    label = gauges[0].get("label", "")
    return {"symbol": sym, "lean": round(lean, 2),
            "primary": gauges[0]["source"], "primary_value": gauges[0].get("value"),
            "label": label, "gauges": gauges,
            "summary": f"{gauges[0]['source']}: {gauges[0].get('value')} ({label})"}
