"""
bb_strategy.py — Bollinger middle-line cross strategy on BTC, honest backtest.

Rules (exactly as specified):
  • LONG when a candle CLOSES back above the middle band (20-SMA); exit the long
    when price reaches the UPPER band (+2 std).
  • SHORT when a candle CLOSES below the middle band; cover the short when price
    reaches the LOWER band (−2 std).
  • A middle-line cross the other way flips the position.

Tested on BTCUSDT at 5m / 15m / 4h over 2021-2025 with real Binance klines and
Binance futures taker fees (0.04%). Prints an honest comparison per timeframe.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.metrics import max_drawdown, sharpe_ratio  # noqa: E402

FEE = 0.0004  # Binance USD-M futures taker fee (0.04%) — shorting BTC = futures
BARS_PER_YEAR = {"5m": 288 * 365, "15m": 96 * 365, "4h": 6 * 365}


def klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    cache = Path("state") / f"klines_{symbol}_{interval}_2021_2025.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    url = "https://api.binance.com/api/v3/klines"
    rows, cur = [], start_ms
    while cur < end_ms:
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval,
                             "startTime": cur, "endTime": end_ms, "limit": 1000}, timeout=25)
            b = r.json()
        except Exception:
            time.sleep(1); continue
        if not isinstance(b, list) or not b:
            break
        rows += b
        cur = b[-1][0] + 1
        if len(b) < 1000:
            break
        time.sleep(0.06)
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v", "ct", "qv",
                                     "n", "tb", "tq", "ig"])
    df.index = pd.to_datetime(df["t"], unit="ms")
    df["close"] = pd.to_numeric(df["c"])
    df = df[["close"]]
    cache.parent.mkdir(exist_ok=True)
    df.to_pickle(cache)
    return df


def vol_window(close: pd.Series) -> set:
    """The 4 consecutive UTC hours with the highest average |return| (BTC's most
    volatile session). Entries are restricted to this window."""
    r = close.pct_change().abs()
    by_hour = r.groupby(close.index.hour).mean()
    best_start, best_sum = 0, -1
    for h in range(24):
        block = sum(float(by_hour.get((h + k) % 24, 0)) for k in range(4))
        if block > best_sum:
            best_sum, best_start = block, h
    return {(best_start + k) % 24 for k in range(4)}


def run_breakout(close: pd.Series, interval: str, hours: set) -> dict:
    """Refined rules: enter LONG only when a candle CLOSES above the upper band
    *inside the high-vol window* (fill next bar); exit when it closes back below
    the middle band. Symmetric for shorts. Exits allowed anytime."""
    mid = close.rolling(20).mean(); sd = close.rolling(20).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    c = close.to_numpy(); m = mid.to_numpy(); u = upper.to_numpy(); lo = lower.to_numpy()
    hr = close.index.hour.to_numpy(); n = len(c)
    pos = np.zeros(n); trades = []; state, entry = 0, 0.0
    for t in range(21, n):
        if np.isnan(m[t]):
            pos[t] = state; continue
        if state == 1 and c[t] < m[t]:
            trades.append((1, entry, c[t])); state = 0
        elif state == -1 and c[t] > m[t]:
            trades.append((-1, entry, c[t])); state = 0
        if state == 0 and hr[t] in hours:
            if c[t] > u[t]:
                state, entry = 1, c[t]
            elif c[t] < lo[t]:
                state, entry = -1, c[t]
        pos[t] = state
    ret = np.concatenate([[0.0], np.diff(c) / c[:-1]])
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = np.concatenate([[0.0], pos[:-1] * ret[1:]]) - turn * FEE
    eq = np.cumprod(1 + strat); ppy = BARS_PER_YEAR[interval]
    wins = [d * (x / e - 1) for d, e, x in trades if e]
    return {"interval": interval, "trades": len(trades),
            "win_rate": round(100 * np.mean([w > 0 for w in wins]), 1) if wins else 0,
            "total_pct": round((eq[-1] - 1) * 100, 1),
            "sharpe": round(float(sharpe_ratio(strat, periods_per_year=ppy)), 2),
            "max_dd_pct": round(max_drawdown(eq) * 100, 1),
            "bh_pct": round((c[-1] / c[20] - 1) * 100, 1)}


def run(close: pd.Series, interval: str) -> dict:
    mid = close.rolling(20).mean()
    sd = close.rolling(20).std()
    upper, lower = mid + 2 * sd, mid - 2 * sd
    c = close.to_numpy(); m = mid.to_numpy(); u = upper.to_numpy(); lo = lower.to_numpy()
    n = len(c)
    pos = np.zeros(n)
    trades = []          # (dir, entry, exit)
    state, entry = 0, 0.0
    for t in range(21, n):
        if np.isnan(m[t]) or np.isnan(m[t - 1]):
            pos[t] = state; continue
        up_cross = c[t - 1] <= m[t - 1] and c[t] > m[t]
        dn_cross = c[t - 1] >= m[t - 1] and c[t] < m[t]
        # 1) take-profit exits at the bands
        if state == 1 and c[t] >= u[t]:
            trades.append((1, entry, c[t])); state = 0
        elif state == -1 and c[t] <= lo[t]:
            trades.append((-1, entry, c[t])); state = 0
        # 2) entries / flips on the middle-line cross
        if up_cross and state != 1:
            if state == -1:
                trades.append((-1, entry, c[t]))
            state, entry = 1, c[t]
        elif dn_cross and state != -1:
            if state == 1:
                trades.append((1, entry, c[t]))
            state, entry = -1, c[t]
        pos[t] = state

    ret = np.concatenate([[0.0], np.diff(c) / c[:-1]])
    turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * ret - turn * FEE          # pos held into bar's own return proxy
    # shift: earn next-bar return on the position set this bar
    strat = np.concatenate([[0.0], pos[:-1] * ret[1:]]) - turn * FEE
    eq = np.cumprod(1 + strat)
    ppy = BARS_PER_YEAR[interval]
    wins = [d * (x / e - 1) for d, e, x in trades if e]
    wr = 100 * np.mean([w > 0 for w in wins]) if wins else 0
    return {
        "interval": interval, "bars": n, "trades": len(trades),
        "win_rate": round(wr, 1),
        "total_pct": round((eq[-1] - 1) * 100, 1),
        "sharpe": round(float(sharpe_ratio(strat, periods_per_year=ppy)), 2),
        "max_dd_pct": round(max_drawdown(eq) * 100, 1),
        "bh_pct": round((c[-1] / c[20] - 1) * 100, 1),
    }


def main() -> None:
    start = int(pd.Timestamp("2021-01-01").timestamp() * 1000)
    end = int(pd.Timestamp("2025-12-31").timestamp() * 1000)
    print("Loading BTCUSDT klines 2021-2025 (cached after first fetch)…\n")
    old, new, windows = [], [], {}
    for interval in ["4h", "15m", "5m"]:
        df = klines("BTCUSDT", interval, start, end)
        win = vol_window(df["close"])
        windows[interval] = sorted(win)
        print(f"  {interval}: {len(df)} candles · high-vol window (UTC h): {sorted(win)}")
        old.append(run(df["close"], interval))
        new.append(run_breakout(df["close"], interval, win))

    def table(title, rows):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
        print(f"  {'TF':<6}{'Trades':>8}{'WinRate':>9}{'StratRet':>11}{'Sharpe':>9}"
              f"{'MaxDD':>9}{'Buy&Hold':>11}")
        print("-" * 80)
        for r in rows:
            print(f"  {r['interval']:<6}{r['trades']:>8}{r['win_rate']:>8}%"
                  f"{r['total_pct']:>10}%{r['sharpe']:>9}{r['max_dd_pct']:>8}%"
                  f"{r['bh_pct']:>10}%")
        print("=" * 80)

    table("v1 — middle-cross (original)", old)
    table("v2 — YOUR REFINEMENT: band-close breakout, 4h high-vol window only", new)


if __name__ == "__main__":
    main()
