"""
news.py — Headlines as *context*, via Yahoo Finance's official news feed.

Deliberately NOT a forum/social scraper. Reddit / X / StockTwits scraping is
mostly against those sites' terms, is dominated by bots and pump groups, and has
no durable trading edge — chasing it is how retail traders get played. This
module pulls recent headlines from the official Yahoo feed (the same one the
yfinance library exposes) purely so a human can read the news *alongside* the
quantitative forecast. The naive keyword lean below is a reading aid, not a
validated signal — treat it as such.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import yfinance as yf

logger = logging.getLogger(__name__)

# Tiny finance lexicon for a crude headline lean. NOT a real sentiment model.
_POS = {"surge", "soar", "jump", "gain", "rally", "beat", "beats", "record",
        "high", "growth", "profit", "upgrade", "bullish", "rise", "rises",
        "boost", "strong", "wins", "win", "buy", "outperform"}
_NEG = {"fall", "falls", "plunge", "drop", "slump", "loss", "losses", "miss",
        "misses", "downgrade", "bearish", "cut", "cuts", "weak", "warning",
        "probe", "fraud", "lawsuit", "sell", "slide", "crash", "fear"}


def naive_lean(text: str) -> float:
    """Crude [-1, 1] keyword lean of a headline. A reading aid, not a signal."""
    words = {w.strip(".,!?:'\"()").lower() for w in text.split()}
    pos, neg = len(words & _POS), len(words & _NEG)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _unwrap(item: dict) -> dict:
    """Normalise across yfinance news shapes (old flat vs new 'content' dict)."""
    c = item.get("content", item)
    title = c.get("title") or item.get("title") or ""
    provider = (
        (c.get("provider") or {}).get("displayName")
        or item.get("publisher")
        or "—"
    )
    return {"title": title, "publisher": provider}


def get_headlines(symbol: str, limit: int = 5) -> List[Dict]:
    """Return up to ``limit`` recent headlines for ``symbol`` with a naive lean.

    Returns an empty list (with a warning) if the feed is unavailable — news is
    always optional context, never a hard dependency.
    """
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception as exc:
        logger.warning("News fetch failed for %s: %s", symbol, exc)
        return []

    out: List[Dict] = []
    for item in raw[:limit]:
        h = _unwrap(item)
        if not h["title"]:
            continue
        h["lean"] = naive_lean(h["title"])
        out.append(h)
    return out


def lean_label(score: float) -> str:
    """Human label for a lean score."""
    if score > 0.34:
        return "＋"
    if score < -0.34:
        return "－"
    return "·"
