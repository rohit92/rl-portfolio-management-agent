"""
levels.py — Breakout / breakdown trigger levels with expected follow-through.

Turns the forecast into *actionable, conditional* levels:

    "Above <breakout> → momentum long, ~<move> typical follow-through"
    "Below <breakdown> → breakdown, ~<move> to the downside"

The triggers are the recent N-bar high (resistance) and low (support). The
expected follow-through is scaled from the calibrated volatility forecast
(1-day expected move, stretched to a few sessions by the √time rule).

> Honest framing: these are *rules conditioned on a price event*, not
> predictions. Breakouts fail often (false breakouts are common), and the
> level/target are typical magnitudes from volatility — not a promise of
> direction. Use them to define risk, not to guarantee a trade.
"""

from __future__ import annotations

import math
from typing import Dict


def trigger_levels(sd, exp_move_pct_1d: float, lookback: int = 20,
                   horizon: int = 5) -> Dict:
    """Compute breakout/breakdown levels and their expected moves.

    Parameters
    ----------
    sd : SymbolData
        Engineered frame (needs close; uses high/low if present).
    exp_move_pct_1d : float
        Forecast 1-day expected absolute move as a fraction (from forecast.py).
    lookback : int
        Bars used for the high/low channel.
    horizon : int
        Sessions of expected follow-through (scales the 1-day move by √horizon).
    """
    frame = sd.frame
    close = frame["close"]
    high = frame["high"] if "high" in frame else close
    low = frame["low"] if "low" in frame else close

    last = float(close.iloc[-1])
    breakout = float(high.tail(lookback).max())
    breakdown = float(low.tail(lookback).min())

    # Follow-through magnitude: 1-day expected move stretched over `horizon`.
    move = float(exp_move_pct_1d) * math.sqrt(max(horizon, 1)) \
        if exp_move_pct_1d and math.isfinite(exp_move_pct_1d) else 0.0

    return {
        "lookback": lookback,
        "horizon": horizon,
        "last": last,
        "breakout": {
            "level": breakout,
            "dist_pct": breakout / last - 1.0,
            "target": breakout * (1.0 + move),
            "exp_move_pct": move,
        },
        "breakdown": {
            "level": breakdown,
            "dist_pct": breakdown / last - 1.0,
            "target": breakdown * (1.0 - move),
            "exp_move_pct": move,
        },
    }
