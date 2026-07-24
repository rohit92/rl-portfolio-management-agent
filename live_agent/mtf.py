"""
mtf.py — Multi-timeframe analysis: the way professionals actually structure trades.

The core pro ideas implemented here (Elder's "Triple Screen" family of methods):

  1. DIFFERENT STRATEGIES SUIT DIFFERENT TIMEFRAMES. Intraday price action is
     noisy and mean-reverting; higher timeframes carry the persistent trend
     (the same asymmetry the TSMOM literature documents). So the 45 agents are
     re-weighted per timeframe: mean-reversion & volume agents dominate on 5m,
     trend & momentum agents dominate on 4h/1d.

  2. TRADE ONLY ON ALIGNMENT. The higher timeframe sets the allowed direction;
     the lower timeframes time the entry. When frames disagree → WAIT. This is
     the single biggest whipsaw filter pros use.

  3. ONE CLEAR VERDICT. Instead of 45 raw votes, you get a single line:
     "BUY — higher-timeframe uptrend, entry frames aligned" or
     "WAIT — 4h bullish but 15m/5m disagree (pullback; wait for realignment)".

Honesty: alignment reduces trade count and whipsaw — it does NOT conjure a
directional edge that isn't there. ``backtest_mtf`` measures the effect with
real fees so you can see exactly what it buys you.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import agents as agents_mod
from data_feed import fetch

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
logger = logging.getLogger("mtf")

# ---- 1) Strategy-family emphasis per timeframe ------------------------------
# Intraday (5m/15m): fade noise — mean-reversion & order-flow (volume) lead.
# Swing (4h/1h):     ride direction — trend & momentum lead.
# Position (1d):     trend dominates; mean-reversion is mostly noise here.
FAMILY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "5m":  {"Trend": 0.6, "Momentum": 1.0, "MeanRev": 1.5, "Breakout": 1.0,
            "Volume": 1.3, "Statistical": 1.0},
    "15m": {"Trend": 0.8, "Momentum": 1.1, "MeanRev": 1.3, "Breakout": 1.1,
            "Volume": 1.2, "Statistical": 1.0},
    "1h":  {"Trend": 1.2, "Momentum": 1.2, "MeanRev": 0.9, "Breakout": 1.1,
            "Volume": 1.0, "Statistical": 1.0},
    "4h":  {"Trend": 1.4, "Momentum": 1.3, "MeanRev": 0.7, "Breakout": 1.1,
            "Volume": 0.9, "Statistical": 1.0},
    "1d":  {"Trend": 1.5, "Momentum": 1.3, "MeanRev": 0.6, "Breakout": 1.0,
            "Volume": 0.8, "Statistical": 1.0},
}
# Higher timeframe carries more weight in the combined verdict (pro top-down).
TF_WEIGHTS = {"5m": 0.15, "15m": 0.20, "1h": 0.25, "4h": 0.30, "1d": 0.35}
TH = 0.15  # per-timeframe score threshold for a directional read


def _agent_weight_vector(tf: str) -> np.ndarray:
    fam_w = FAMILY_WEIGHTS.get(tf, FAMILY_WEIGHTS["1d"])
    return np.array([fam_w.get(fam, 1.0) for _n, fam, _f in agents_mod.AGENTS])


def weighted_score_series(frame: pd.DataFrame, tf: str) -> pd.Series:
    """Per-bar committee score in [-1, 1], with agents re-weighted for this
    timeframe's dominant strategy family."""
    M = agents_mod.Committee({}).vote_matrix(frame)
    w = _agent_weight_vector(tf)
    return (M.to_numpy() @ w) / w.sum() + pd.Series(0.0, index=M.index)


def _tf_read(frame: pd.DataFrame, tf: str) -> Dict:
    M = agents_mod.Committee({}).vote_matrix(frame)
    last = M.iloc[-1].to_numpy()
    w = _agent_weight_vector(tf)
    score = float((last * w).sum() / w.sum())
    # family leaders for the explanation
    fam_net: Dict[str, int] = {}
    for (name, fam, _f), v in zip(agents_mod.AGENTS, last):
        fam_net[fam] = fam_net.get(fam, 0) + int(v)
    verdict = "BUY" if score > TH else "SELL" if score < -TH else "NEUTRAL"
    leaders = sorted(fam_net.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
    return {"tf": tf, "score": round(score, 2), "verdict": verdict,
            "buy": int((last == 1).sum()), "sell": int((last == -1).sum()),
            "leaders": [f"{k} {v:+d}" for k, v in leaders],
            "families": fam_net}


def analyze(symbol: str, cfg: dict) -> Dict:
    """Live multi-timeframe read for a symbol, with one clear combined verdict."""
    sym = symbol.strip().upper()
    try:
        import binance_feed
        crypto = binance_feed.is_crypto(sym)
    except Exception:
        crypto = False
    plan = ([("5m", 15), ("15m", 55), ("4h", 240), ("1d", 500)] if crypto
            else [("15m", 55), ("1h", 400), ("1d", 500)])

    reads: List[Dict] = []
    for tf, days in plan:
        data = fetch([sym], cfg, lookback_days=days, interval=tf)
        sd = data.get(sym)
        if sd is None or len(sd.frame) < 220:
            continue
        reads.append(_tf_read(sd.frame, tf))
    if not reads:
        return {"error": f"No usable data for {sym}"}

    # Combined, higher-TF-weighted score
    tot_w = sum(TF_WEIGHTS[r["tf"]] for r in reads)
    combined = sum(TF_WEIGHTS[r["tf"]] * r["score"] for r in reads) / tot_w
    higher = reads[-1]                       # the highest timeframe fetched
    entries = reads[:-1] or reads
    aligned_up = higher["score"] > TH and all(r["score"] > 0 for r in entries)
    aligned_dn = higher["score"] < -TH and all(r["score"] < 0 for r in entries)

    if aligned_up:
        side, verdict = "BUY", (f"🟢 BUY — {higher['tf']} uptrend and all entry "
                                f"timeframes aligned long")
    elif aligned_dn:
        side, verdict = "SELL", (f"🔴 SELL — {higher['tf']} downtrend and all entry "
                                 f"timeframes aligned short")
    else:
        side = "WAIT"
        if abs(higher["score"]) > TH:
            hdir = "bullish" if higher["score"] > 0 else "bearish"
            verdict = (f"⏸ WAIT — {higher['tf']} is {hdir} but entry timeframes "
                       f"disagree (pullback/chop). Pros wait for realignment.")
        else:
            verdict = "⏸ WAIT — no clear higher-timeframe direction; stand aside."

    return {"symbol": sym, "combined_score": round(combined, 2), "side": side,
            "verdict": verdict, "reads": reads,
            "rule": "Higher timeframe sets direction; lower timeframes time the "
                    "entry; disagreement = no trade."}


# --------------------------------------------------------------------------- #
#  Honest validation: does alignment actually help?
# --------------------------------------------------------------------------- #

def backtest_mtf(days: int = 730) -> Dict:
    """Trade the 15m score, with vs without requiring 4h agreement (BTCUSDT,
    Binance futures fee). Measures exactly what alignment buys you."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    from utils.metrics import max_drawdown, sharpe_ratio
    import binance_feed

    dfs = {}
    for tf in ("15m", "4h"):
        cache = STATE / f"ohlcv_BTCUSDT_{tf}_{days}d.pkl"
        if cache.exists():
            dfs[tf] = pd.read_pickle(cache)
        else:
            dfs[tf] = binance_feed.get_ohlcv("BTCUSDT", tf, lookback_days=days)
            STATE.mkdir(exist_ok=True)
            dfs[tf].to_pickle(cache)

    s15 = weighted_score_series(dfs["15m"], "15m")
    s4 = weighted_score_series(dfs["4h"], "4h")
    s4a = s4.shift(1).reindex(s15.index, method="ffill")   # last completed 4h read

    close = dfs["15m"]["close"]
    ret = close.pct_change().fillna(0.0).to_numpy()
    sig = s15.shift(1)                                     # act on completed bar
    pos1 = np.where(sig > TH, 1.0, np.where(sig < -TH, -1.0, 0.0))
    agree = ((np.sign(sig) == np.sign(s4a)) & (s4a.abs() > 0.05)).to_numpy()
    pos2 = np.where(agree, pos1, 0.0)

    fee = 0.0004
    ppy = 96 * 365
    out = {"days": days, "bars": len(s15)}
    for name, pos in [("single_15m", pos1), ("mtf_15m_plus_4h", pos2)]:
        turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
        strat = pos * ret - turn * fee
        eq = np.cumprod(1 + strat)
        out[name] = {"total_pct": round((eq[-1] - 1) * 100, 1),
                     "sharpe": round(float(sharpe_ratio(strat, periods_per_year=ppy)), 2),
                     "max_dd_pct": round(max_drawdown(eq) * 100, 1),
                     "trades": int((np.diff(pos) != 0).sum()),
                     "time_in_market_pct": round(100 * float(np.mean(pos != 0)), 1)}
    out["buy_hold_pct"] = round((close.iloc[-1] / close.iloc[0] - 1) * 100, 1)
    return out


if __name__ == "__main__":
    import json
    import yaml
    import markets
    logging.basicConfig(level=logging.WARNING)
    cfg = markets.resolve(yaml.safe_load(open(HERE / "config.yaml")))
    print("LIVE multi-timeframe read — BTCUSDT")
    r = analyze("BTCUSDT", cfg)
    print(" ", r.get("verdict"))
    for t in r.get("reads", []):
        print(f"    {t['tf']:>4}: score {t['score']:+0.2f} → {t['verdict']:<8} "
              f"({t['buy']}B/{t['sell']}S; led by {', '.join(t['leaders'])})")
    print("\nHONEST VALIDATION (2y BTCUSDT 15m, Binance fee) — running…")
    print(json.dumps(backtest_mtf(), indent=2))
