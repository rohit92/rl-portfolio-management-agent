"""
candles.py — Hand-rolled candlestick pattern recognition (no TA-Lib).

Given a lowercase OHLC frame (``open/high/low/close``) this detects the classic
Japanese candlestick patterns bar-by-bar and summarises the most recent ones for
the live dashboard. Each pattern carries a direction (bullish / bearish /
neutral) and a plain-English description.

Honest note (same line the rest of the project holds): candlestick patterns are
*context clues*, not edges. Backtests of naive pattern-only rules do not beat
buy-and-hold. They are shown here to read the tape, and are confirmed against the
45-agent committee before any signal is called — never traded on their own.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def _geom(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    rng = np.where((h - l) == 0, 1e-9, h - l)
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    bull = c > o
    bear = c < o
    mid = (o + c) / 2.0
    # short-term trend context: close vs its 10-bar average
    sma = pd.Series(c).rolling(10, min_periods=1).mean().to_numpy()
    up_ctx = c < sma            # price below mean -> a bottoming reversal has room
    down_ctx = c > sma          # price above mean -> a topping reversal has room
    return dict(o=o, h=h, l=l, c=c, rng=rng, body=body, upper=upper, lower=lower,
                bull=bull, bear=bear, mid=mid, up_ctx=up_ctx, down_ctx=down_ctx)


# --------------------------------------------------------------------------- #
#  Pattern catalogue.  Each entry: name -> (direction, description, mask fn)
#  The mask fn takes the geometry dict and returns a boolean array (per bar).
# --------------------------------------------------------------------------- #
def _patterns(g: Dict[str, np.ndarray]) -> List[Dict]:
    o, h, l, c = g["o"], g["h"], g["l"], g["c"]
    rng, body, upper, lower = g["rng"], g["body"], g["upper"], g["lower"]
    bull, bear, mid = g["bull"], g["bear"], g["mid"]
    n = len(c)
    small = body < 0.30 * rng          # small real body
    tiny = body <= 0.10 * rng          # doji-like
    long_body = body > 0.60 * rng      # dominant real body

    def shift(a, k=1):
        out = np.empty_like(a, dtype=float)
        out[:] = np.nan
        if k < n:
            out[k:] = a[:-k].astype(float)
        return out

    o1, c1, h1, l1, body1, rng1 = shift(o), shift(c), shift(h), shift(l), shift(body), shift(rng)
    bull1, bear1 = shift(bull.astype(float)) == 1, shift(bear.astype(float)) == 1
    o2, c2, body2 = shift(o, 2), shift(c, 2), shift(body, 2)
    bear2 = shift(bear.astype(float), 2) == 1
    bull2 = shift(bull.astype(float), 2) == 1

    out: List[Dict] = []

    def add(name, direction, desc, mask):
        out.append({"name": name, "dir": direction, "desc": desc,
                    "mask": np.nan_to_num(mask, nan=0.0).astype(bool)})

    # ---- single-bar ----
    add("Doji", "neutral", "Open ≈ close — indecision; watch the next bar.",
        tiny)
    add("Bullish Marubozu", "bull", "Full-body up candle, almost no wicks — strong buying.",
        bull & (upper <= 0.05 * rng) & (lower <= 0.05 * rng) & long_body)
    add("Bearish Marubozu", "bear", "Full-body down candle, almost no wicks — strong selling.",
        bear & (upper <= 0.05 * rng) & (lower <= 0.05 * rng) & long_body)
    add("Hammer", "bull", "Long lower wick after weakness — sellers rejected; possible bottom.",
        (lower >= 2 * body) & (upper <= body) & (body > 0) & g["up_ctx"])
    add("Hanging Man", "bear", "Long lower wick after strength — a topping warning.",
        (lower >= 2 * body) & (upper <= body) & (body > 0) & g["down_ctx"])
    add("Inverted Hammer", "bull", "Long upper wick after weakness — buyers testing; possible bottom.",
        (upper >= 2 * body) & (lower <= body) & (body > 0) & g["up_ctx"])
    add("Shooting Star", "bear", "Long upper wick after strength — buyers rejected; possible top.",
        (upper >= 2 * body) & (lower <= body) & (body > 0) & g["down_ctx"])
    add("Spinning Top", "neutral", "Small body, wicks both sides — balance, momentum fading.",
        small & (upper > body) & (lower > body) & ~tiny)

    # ---- two-bar ----
    add("Bullish Engulfing", "bull", "Up candle fully engulfs the prior down candle — bulls take over.",
        bear1 & bull & (c >= o1) & (o <= c1) & (body > body1))
    add("Bearish Engulfing", "bear", "Down candle fully engulfs the prior up candle — bears take over.",
        bull1 & bear & (o >= c1) & (c <= o1) & (body > body1))
    add("Bullish Harami", "bull", "Small up body inside the prior large down body — selling stalls.",
        bear1 & bull & (long_body_at(body1, rng1)) & (np.maximum(o, c) <= np.maximum(o1, c1)) &
        (np.minimum(o, c) >= np.minimum(o1, c1)) & small)
    add("Bearish Harami", "bear", "Small down body inside the prior large up body — buying stalls.",
        bull1 & bear & (long_body_at(body1, rng1)) & (np.maximum(o, c) <= np.maximum(o1, c1)) &
        (np.minimum(o, c) >= np.minimum(o1, c1)) & small)
    add("Piercing Line", "bull", "Opens below prior low, closes back above its midpoint — reversal up.",
        bear1 & bull & (o < l1) & (c > (o1 + c1) / 2.0) & (c < o1))
    add("Dark Cloud Cover", "bear", "Opens above prior high, closes below its midpoint — reversal down.",
        bull1 & bear & (o > h1) & (c < (o1 + c1) / 2.0) & (c > o1))
    add("Tweezer Bottom", "bull", "Two matching lows after weakness — support holding.",
        (np.abs(l - l1) <= 0.05 * rng) & bear1 & bull & g["up_ctx"])
    add("Tweezer Top", "bear", "Two matching highs after strength — resistance holding.",
        (np.abs(h - h1) <= 0.05 * rng) & bull1 & bear & g["down_ctx"])

    # ---- three-bar ----
    add("Morning Star", "bull", "Big down, small pause, big up — classic bullish reversal.",
        bear2 & (body1 < 0.4 * rng1) & bull & (c > (o2 + c2) / 2.0) & long_body)
    add("Evening Star", "bear", "Big up, small pause, big down — classic bearish reversal.",
        bull2 & (body1 < 0.4 * rng1) & bear & (c < (o2 + c2) / 2.0) & long_body)
    add("Three White Soldiers", "bull", "Three strong up candles in a row — sustained buying.",
        bull & bull1 & bull2 & (c > c1) & (c1 > c2) & (o < c1) & (o > o1) & long_body)
    add("Three Black Crows", "bear", "Three strong down candles in a row — sustained selling.",
        bear & bear1 & bear2 & (c < c1) & (c1 < c2) & (o > c1) & (o < o1) & long_body)

    return out


def long_body_at(body_arr, rng_arr):
    """Prior-bar 'large body' test used by the harami patterns."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(body_arr / rng_arr, nan=0.0) > 0.6


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #
_ICON = {"bull": "▲", "bear": "▼", "neutral": "◆"}


def detect(df: pd.DataFrame, lookback: int = 60) -> List[Dict]:
    """Every pattern instance in the last ``lookback`` bars.

    Returns a list of {i, name, dir, desc, icon} where ``i`` is the bar index
    within the *returned* window (so a chart drawing the last `lookback` bars can
    place a marker directly).
    """
    if df is None or len(df) < 3:
        return []
    df = df.rename(columns=str.lower)
    g = _geom(df)
    n = len(g["c"])
    start = max(0, n - lookback)
    hits: List[Dict] = []
    for p in _patterns(g):
        mask = p["mask"]
        for i in range(start, n):
            if mask[i]:
                hits.append({"i": i - start, "abs": i, "name": p["name"],
                             "dir": p["dir"], "desc": p["desc"],
                             "icon": _ICON[p["dir"]]})
    hits.sort(key=lambda x: x["abs"])
    return hits


def summary(df: pd.DataFrame, recent: int = 5) -> Dict:
    """Latest-bar + last-`recent`-bars candlestick read for a signal card."""
    hits = detect(df, lookback=recent)
    n = len(df)
    latest = [h for h in hits if h["abs"] >= n - 1]      # patterns on the last bar
    bulls = sum(1 for h in hits if h["dir"] == "bull")
    bears = sum(1 for h in hits if h["dir"] == "bear")
    if bulls > bears:
        bias, note = "bullish", f"{bulls} bullish vs {bears} bearish candle signals recently."
    elif bears > bulls:
        bias, note = "bearish", f"{bears} bearish vs {bulls} bullish candle signals recently."
    else:
        bias, note = "neutral", "No clear candlestick bias in recent bars."
    return {
        "latest": [{"name": h["name"], "dir": h["dir"], "desc": h["desc"],
                    "icon": h["icon"]} for h in latest],
        "recent": [{"name": h["name"], "dir": h["dir"], "icon": h["icon"], "i": h["i"]}
                   for h in hits],
        "bull_count": bulls, "bear_count": bears, "bias": bias, "note": note,
    }


if __name__ == "__main__":
    # smoke test on real Binance data
    import binance_feed
    d = binance_feed.get_ohlcv("BTCUSDT", interval="1h", lookback_days=10)
    s = summary(d)
    print("bias:", s["bias"], "-", s["note"])
    for h in detect(d, 40)[-8:]:
        print(f"  bar {h['abs']:>4}  {h['icon']} {h['name']:<22} ({h['dir']})")
