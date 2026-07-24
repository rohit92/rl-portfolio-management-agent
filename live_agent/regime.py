"""
regime.py — Market regime detection: the pro's first question.

Before a professional picks a trade, they ask *what kind of market is this?* —
because every setup family only works in its own weather:

  • Trending market  -> trend / momentum / breakout setups pay; mean-reversion
    gets steamrolled ("don't fade a freight train").
  • Calm range       -> mean-reversion at the edges pays; breakouts fail
    (most break out of nothing and fall back in).
  • Volatile chop    -> almost nothing pays; pros cut size or sit out.

Three independent, hand-rolled measures (no TA-Lib):

  ADX(14)                — Wilder's trend-strength (0..100). >25 = trending.
  Kaufman efficiency     — |net move| / sum of |bar moves| over a window
                           (1.0 = straight line, ~0 = pure noise).
  Realized-vol percentile — current 20-bar annualised vol ranked against its
                           own trailing history (is *this* market hot or quiet
                           by its own standards?).

Direction comes from a fast/slow MA cross plus price location. Output is a
plain-English label + the playbook that historically fits — context, not a
signal. Knowing the regime doesn't predict the future; it tells you which
mistakes are most expensive right now.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data_feed import fetch

logger = logging.getLogger("regime")

# annualisation factors per bar interval (matches signal_backtest)
_PPY = {"5m": 105_120, "15m": 35_040, "30m": 17_520, "1h": 8_760, "4h": 2_190,
        "1d": 252, "1wk": 52}
_LB = {"5m": 20, "15m": 60, "30m": 90, "1h": 180, "4h": 720, "1d": 1500, "1wk": 3000}


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ADX, hand-rolled (EWM with alpha=1/n ≈ Wilder smoothing)."""
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, min_periods=n).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, min_periods=n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, min_periods=n).mean()


def _efficiency(close: pd.Series, n: int = 20) -> float:
    """Kaufman efficiency ratio over the last ``n`` bars."""
    seg = close.iloc[-(n + 1):]
    if len(seg) < n + 1:
        return float("nan")
    net = abs(float(seg.iloc[-1]) - float(seg.iloc[0]))
    path = float(seg.diff().abs().sum())
    return net / path if path > 0 else 0.0


def detect(df: pd.DataFrame, *, ppy: int = 252) -> Dict:
    """Classify the regime of an OHLC frame. Returns label + numbers + playbook."""
    close = df["close"]
    n = len(close)
    if n < 60:
        return {"error": "Not enough bars to read a regime (need 60+)."}

    adx = _adx(df)
    adx_now = float(adx.iloc[-1]) if np.isfinite(adx.iloc[-1]) else None
    er = _efficiency(close, 20)

    # realized vol now vs its own history
    rets = close.pct_change()
    vol_series = rets.rolling(20).std() * math.sqrt(ppy)
    vol_now = float(vol_series.iloc[-1]) if np.isfinite(vol_series.iloc[-1]) else None
    hist = vol_series.dropna()
    vol_pctile = round(float((hist < vol_now).mean() * 100.0), 1) \
        if vol_now is not None and len(hist) > 40 else None

    # direction: fast/slow MA + price location (windows shrink on short frames)
    fast_n, slow_n = (50, 200) if n >= 220 else (20, 100)
    fast = float(close.rolling(fast_n).mean().iloc[-1])
    slow = float(close.rolling(slow_n).mean().iloc[-1])
    px = float(close.iloc[-1])
    if fast > slow and px > fast:
        direction, dir_word = 1, "up"
    elif fast < slow and px < fast:
        direction, dir_word = -1, "down"
    else:
        direction, dir_word = 0, "mixed"

    trending = (adx_now is not None and adx_now >= 25) or er >= 0.35
    weak = adx_now is not None and adx_now < 20
    hot = vol_pctile is not None and vol_pctile >= 80

    if trending and direction != 0:
        label = f"TRENDING {dir_word.upper()}"
        playbook = (f"Trend-following weather: trend / momentum / breakout setups in the "
                    f"{dir_word}-direction have the tailwind. Fading this trend (counter-trend "
                    f"mean-reversion) is the expensive mistake here. Trail stops, let winners run.")
        families = ["Trend", "Momentum", "Breakout"]
        risk_hint = "normal size, with-trend only"
    elif trending:  # strong ADX but MAs disagree — turning market
        label = "TRANSITION / TURNING"
        playbook = ("Strong directional energy but the averages disagree — often a trend "
                    "change in progress. Pros wait for the new direction to confirm rather "
                    "than guess the turn. Smaller size, faster exits.")
        families = ["Breakout"]
        risk_hint = "half size until direction confirms"
    elif weak and hot:
        label = "VOLATILE CHOP"
        playbook = ("Wide, violent, directionless — the regime that eats accounts. Both "
                    "breakouts and fades get stopped. The professional play is the humblest "
                    "one: cut size hard or stand aside until ADX or efficiency picks up.")
        families = []
        risk_hint = "stand aside / quarter size"
    elif weak:
        label = "CALM RANGE"
        playbook = ("Quiet, mean-reverting market: fading moves back to the middle "
                    "(mean-reversion at range edges) historically works; breakout entries "
                    "mostly fail back into the range. Take profits early — no trends to ride.")
        families = ["Mean Reversion"]
        risk_hint = "normal size, quick profits"
    else:
        label = "NEUTRAL / MIXED"
        playbook = ("No clear regime — trend strength and efficiency are middling. "
                    "Edge is thinnest here; be selective and demand higher-conviction setups.")
        families = ["Trend", "Mean Reversion"]
        risk_hint = "be selective"

    return {
        "label": label, "direction": dir_word,
        "adx": round(adx_now, 1) if adx_now is not None else None,
        "efficiency": round(er, 3) if np.isfinite(er) else None,
        "vol_ann_pct": round(vol_now * 100, 1) if vol_now is not None else None,
        "vol_pctile": vol_pctile,
        "trending": bool(trending), "hot_vol": bool(hot),
        "favored_families": families, "playbook": playbook, "risk_hint": risk_hint,
        "note": ("Regime = context, not prediction. It tells you which setup family has "
                 "the tailwind and which mistake is most expensive — not where price goes."),
    }


def analyze(symbol: str, cfg: dict, *, interval: str = "1d",
            lookback_days: Optional[int] = None) -> Dict:
    """Fetch data for ``symbol`` and classify its regime."""
    symbol = symbol.strip().upper()
    lb = lookback_days or _LB.get(interval, 730)
    data = fetch([symbol], cfg, lookback_days=lb, interval=interval)
    sd = data.get(symbol)
    if sd is None or len(sd.frame) < 60:
        return {"error": f"Not enough data for {symbol} @ {interval}."}
    out = detect(sd.frame, ppy=_PPY.get(interval, 252))
    out.update({"symbol": symbol, "interval": interval,
                "last": round(float(sd.frame['close'].iloc[-1]), 6),
                "bars": len(sd.frame)})
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    from pathlib import Path
    from trader import load_config
    cfg = load_config(Path(__file__).resolve().parent / "config.yaml")
    for sym in ("BTCUSDT", "^NSEI", "AAPL"):
        r = analyze(sym, cfg)
        if "error" in r:
            print(sym, "->", r["error"])
            continue
        print(f"{sym}: {r['label']} (dir {r['direction']}, ADX {r['adx']}, "
              f"ER {r['efficiency']}, vol {r['vol_ann_pct']}% @ p{r['vol_pctile']}) "
              f"-> {r['favored_families'] or ['stand aside']}")
