"""
signal_backtest.py — Backtest the consensus BUY/SELL *points* as real trades.

This answers the question the live dashboard raises: "if I actually took the
35-40-of-45 agent signals — entering with an ATR target and stop — what would
have happened?" It's the historical mirror of the forward trade-journal:

  • Find every bar where >= ``min_agree`` of the 45 agents cross into agreement
    (a BUY when the long count crosses up, a SELL when the short count does).
  • Enter at the NEXT bar's open (no lookahead). Set target = entry ± N·ATR and
    stop = entry ∓ M·ATR.
  • Walk forward bar by bar; exit on stop, target, an opposite signal, or a
    max-hold timeout. One position at a time.
  • Score every trade (P&L %, R-multiple) and aggregate: win rate, expectancy,
    profit factor, avg R, a risk-based equity curve, max drawdown — vs buy&hold.

Honesty baked in: intrabar we assume the STOP is touched before the target
(the pessimistic, realistic convention); fees are charged both sides; and the
result is compared to buy-and-hold, which it usually does NOT beat. A backtest
is a hypothesis, not a promise — validate forward in the journal.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import agents as agents_mod
from data_feed import fetch

logger = logging.getLogger("signal_backtest")

# lookback (days) per interval — matches the live chart's warmup needs
_LB = {"5m": 20, "15m": 60, "30m": 90, "1h": 180, "4h": 720, "1d": 2000, "1wk": 3000}
_PPY = {"5m": 105_120, "15m": 35_040, "30m": 17_520, "1h": 8_760, "4h": 2_190,
        "1d": 252, "1wk": 52}


def _atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, min_periods=n).mean().to_numpy()


def _max_dd(equity: List[float]) -> float:
    peak, mdd = -1e18, 0.0
    for e in equity:
        peak = max(peak, e)
        if peak > 0:
            mdd = min(mdd, e / peak - 1.0)
    return mdd


def _simulate(longs, shorts, o, h, l, c, atr, *, min_agree, target_atr, stop_atr,
              max_hold, allow_short, risk_pct, fee, idx_dates=None) -> Dict:
    n = len(c)
    trades: List[Dict] = []
    equity = 1.0
    curve = [1.0]
    i = 1
    while i < n - 1:
        buy = longs[i] >= min_agree and longs[i - 1] < min_agree
        sell = allow_short and shorts[i] >= min_agree and shorts[i - 1] < min_agree
        if not (buy or sell):
            i += 1
            continue
        side = "BUY" if (buy and not sell) else ("SELL" if (sell and not buy)
               else ("BUY" if longs[i] >= shorts[i] else "SELL"))
        ei = i + 1
        a = atr[ei] if ei < n and np.isfinite(atr[ei]) else (atr[i] if np.isfinite(atr[i]) else np.nan)
        if ei >= n or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        entry = o[ei]
        if side == "BUY":
            stop, target = entry - stop_atr * a, entry + target_atr * a
        else:
            stop, target = entry + stop_atr * a, entry - target_atr * a

        xi, xpx, reason = None, None, None
        for j in range(ei, min(n, ei + max_hold + 1)):
            if side == "BUY":
                if l[j] <= stop:
                    xi, xpx, reason = j, stop, "stop"; break
                if h[j] >= target:
                    xi, xpx, reason = j, target, "target"; break
            else:
                if h[j] >= stop:
                    xi, xpx, reason = j, stop, "stop"; break
                if l[j] <= target:
                    xi, xpx, reason = j, target, "target"; break
            if j > ei:  # opposite consensus flips us out
                opp = (shorts[j] >= min_agree and shorts[j - 1] < min_agree) if side == "BUY" \
                    else (longs[j] >= min_agree and longs[j - 1] < min_agree)
                if opp:
                    xi, xpx, reason = j, c[j], "opposite"; break
        if xi is None:
            xi = min(n - 1, ei + max_hold)
            xpx, reason = c[xi], "time"

        reward = (xpx - entry) if side == "BUY" else (entry - xpx)
        pnl_pct = reward / entry * 100.0 - fee * 200.0   # fee both sides, in %
        risk = abs(entry - stop)
        R = reward / risk if risk else 0.0
        equity *= (1.0 + R * risk_pct)
        curve.append(equity)
        trades.append({
            "side": side, "entry": round(float(entry), 6), "stop": round(float(stop), 6),
            "target": round(float(target), 6), "exit": round(float(xpx), 6),
            "reason": reason, "pnl_pct": round(pnl_pct, 3), "r": round(R, 2),
            "bars_held": int(xi - ei + 1),
            "entry_i": int(ei), "exit_i": int(xi),
            "entry_t": str(idx_dates[ei]) if idx_dates is not None else None,
            "exit_t": str(idx_dates[xi]) if idx_dates is not None else None,
        })
        i = max(xi, i + 1)

    bh = float(c[-1] / c[0] - 1.0) if len(c) and c[0] else 0.0
    return {"stats": _agg(trades, curve, bh), "trades": trades, "equity": curve}


def _agg(trades: List[Dict], curve: List[float], bh: float) -> Dict:
    """Aggregate a trade list + equity curve into the standard stats block."""
    equity = curve[-1] if curve else 1.0
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] < 0]
    gw = sum(t["pnl_pct"] for t in wins)
    gl = abs(sum(t["pnl_pct"] for t in losses))
    rs = [t["r"] for t in trades]
    return {
        "trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(100.0 * len(wins) / len(trades), 1) if trades else None,
        "expectancy_pct": round(sum(t["pnl_pct"] for t in trades) / len(trades), 3) if trades else None,
        "profit_factor": round(gw / gl, 2) if gl else None,
        "avg_r": round(sum(rs) / len(rs), 2) if rs else None,
        "avg_win_pct": round(gw / len(wins), 2) if wins else None,
        "avg_loss_pct": round(-gl / len(losses), 2) if losses else None,
        "total_return_pct": round((equity - 1.0) * 100.0, 2),
        "max_dd_pct": round(_max_dd(curve) * 100.0, 2),
        "buy_hold_pct": round(bh * 100.0, 2),
        "beat_bh": bool((equity - 1.0) > bh),
    }


def backtest(symbol: str, cfg: dict, *, interval: str = "1d", min_agree: int = 38,
             target_atr: float = 2.0, stop_atr: float = 1.0, max_hold: int = 20,
             allow_short: bool = True, risk_pct: float = 0.01, fee: float = 0.0004,
             lookback_days: Optional[int] = None) -> Dict:
    """Full backtest of the consensus buy/sell points for one symbol/interval."""
    symbol = symbol.strip().upper()
    lb = lookback_days or _LB.get(interval, 730)
    data = fetch([symbol], cfg, lookback_days=lb, interval=interval)
    sd = data.get(symbol)
    if sd is None or len(sd.frame) < 60:
        return {"error": f"Not enough data for {symbol} @ {interval}."}
    frame = sd.frame
    com = agents_mod.Committee({"committee": {"threshold": min_agree}})
    M = com.vote_matrix(frame)
    longs = (M == 1).sum(axis=1).to_numpy()
    shorts = (M == -1).sum(axis=1).to_numpy()
    o, h, l, c = (frame["open"].to_numpy(), frame["high"].to_numpy(),
                  frame["low"].to_numpy(), frame["close"].to_numpy())
    atr = _atr(frame)
    dates = [str(x)[:16] for x in frame.index]

    res = _simulate(longs, shorts, o, h, l, c, atr, min_agree=min_agree,
                    target_atr=target_atr, stop_atr=stop_atr, max_hold=max_hold,
                    allow_short=allow_short, risk_pct=risk_pct, fee=fee, idx_dates=dates)
    res["equity_curve"] = _equity_series(res, dates, c)
    res.update({
        "symbol": symbol, "interval": interval, "min_agree": min_agree,
        "target_atr": target_atr, "stop_atr": stop_atr, "max_hold": max_hold,
        "allow_short": allow_short, "risk_pct": risk_pct,
        "bars": len(c), "span": [dates[0], dates[-1]], "total_agents": len(agents_mod.AGENTS),
    })
    return res


def _equity_series(res: Dict, dates: List[str], close: np.ndarray) -> Dict:
    """Down-sampled strategy equity + a buy&hold curve for charting."""
    eq = res["equity"]
    # buy&hold normalised to 1.0
    bh = (close / close[0]).tolist()
    step = max(1, len(bh) // 120)
    bh_s = [{"t": dates[i], "v": round(float(bh[i]), 4)} for i in range(0, len(bh), step)]
    step2 = max(1, len(eq) // 120)
    st_s = [round(float(eq[i]), 4) for i in range(0, len(eq), step2)]
    return {"strategy": st_s, "buy_hold": bh_s}


def sweep(symbol: str, cfg: dict, *, interval: str = "1d",
          thresholds: Optional[List[int]] = None, target_atr: float = 2.0,
          stop_atr: float = 1.0, max_hold: int = 20, allow_short: bool = True,
          risk_pct: float = 0.01, fee: float = 0.0004,
          lookback_days: Optional[int] = None) -> Dict:
    """Backtest across several ``min_agree`` thresholds (vote matrix reused once)
    so you can see which consensus bar historically paid best."""
    symbol = symbol.strip().upper()
    thresholds = thresholds or [32, 35, 38, 40, 42]
    lb = lookback_days or _LB.get(interval, 730)
    data = fetch([symbol], cfg, lookback_days=lb, interval=interval)
    sd = data.get(symbol)
    if sd is None or len(sd.frame) < 60:
        return {"error": f"Not enough data for {symbol} @ {interval}."}
    frame = sd.frame
    M = agents_mod.Committee({"committee": {"threshold": 38}}).vote_matrix(frame)
    longs = (M == 1).sum(axis=1).to_numpy()
    shorts = (M == -1).sum(axis=1).to_numpy()
    o, h, l, c = (frame["open"].to_numpy(), frame["high"].to_numpy(),
                  frame["low"].to_numpy(), frame["close"].to_numpy())
    atr = _atr(frame)
    rows = []
    for thr in thresholds:
        r = _simulate(longs, shorts, o, h, l, c, atr, min_agree=thr,
                      target_atr=target_atr, stop_atr=stop_atr, max_hold=max_hold,
                      allow_short=allow_short, risk_pct=risk_pct, fee=fee)
        rows.append({"min_agree": thr, **r["stats"]})
    best = max((r for r in rows if r["trades"] >= 5),
               key=lambda r: (r["expectancy_pct"] or -9e9), default=None)
    return {"symbol": symbol, "interval": interval, "target_atr": target_atr,
            "stop_atr": stop_atr, "rows": rows,
            "best_min_agree": best["min_agree"] if best else None,
            "buy_hold_pct": rows[0]["buy_hold_pct"] if rows else None,
            "total_agents": len(agents_mod.AGENTS)}


def walk_forward(symbol: str, cfg: dict, *, interval: str = "1d",
                 thresholds: Optional[List[int]] = None, target_atr: float = 2.0,
                 stop_atr: float = 1.0, max_hold: int = 20, allow_short: bool = True,
                 risk_pct: float = 0.01, fee: float = 0.0004, n_folds: int = 4,
                 min_train_trades: int = 5, lookback_days: Optional[int] = None) -> Dict:
    """Walk-forward, out-of-sample test of the consensus signal.

    The sweep answers "which min_agree paid *in hindsight*" — which is exactly
    how curve-fitting happens. This is the honest version: in each fold the
    threshold is chosen ONLY from data before the test segment (anchored,
    expanding train window), then traded on the unseen segment that follows.
    The stitched OOS record is the number a pro would actually trust — expect
    it to be worse than the in-sample sweep. If it isn't positive, the edge
    was probably fitted, not real.
    """
    symbol = symbol.strip().upper()
    thresholds = thresholds or [30, 32, 34, 35, 36, 38, 40]
    lb = lookback_days or _LB.get(interval, 730)
    data = fetch([symbol], cfg, lookback_days=lb, interval=interval)
    sd = data.get(symbol)
    if sd is None or len(sd.frame) < 300:
        return {"error": f"Not enough data for {symbol} @ {interval} to walk forward "
                         f"(need ~300+ bars)."}
    frame = sd.frame
    M = agents_mod.Committee({"committee": {"threshold": 38}}).vote_matrix(frame)
    longs = (M == 1).sum(axis=1).to_numpy()
    shorts = (M == -1).sum(axis=1).to_numpy()
    o, h, l, c = (frame["open"].to_numpy(), frame["high"].to_numpy(),
                  frame["low"].to_numpy(), frame["close"].to_numpy())
    atr = _atr(frame)
    dates = [str(x)[:16] for x in frame.index]
    n = len(c)

    min_train = max(150, n // (n_folds + 2))
    test_len = (n - min_train) // n_folds
    if test_len < 40:
        return {"error": f"Only {n} bars — too short for {n_folds} walk-forward folds."}

    def _sim(i0, i1, thr):
        return _simulate(longs[i0:i1], shorts[i0:i1], o[i0:i1], h[i0:i1], l[i0:i1],
                         c[i0:i1], atr[i0:i1], min_agree=thr, target_atr=target_atr,
                         stop_atr=stop_atr, max_hold=max_hold, allow_short=allow_short,
                         risk_pct=risk_pct, fee=fee, idx_dates=dates[i0:i1])

    folds: List[Dict] = []
    oos_trades: List[Dict] = []
    oos_curve = [1.0]
    s0 = min_train
    for k in range(n_folds):
        s = min_train + k * test_len
        e = n if k == n_folds - 1 else s + test_len
        # --- choose the threshold on TRAIN data only (bars [0, s)) ---
        best_thr, best_exp, best_row = None, None, None
        fallback_thr, fallback_trades = None, 0
        for thr in thresholds:
            r = _sim(0, s, thr)
            st = r["stats"]
            if st["trades"] > fallback_trades:
                fallback_thr, fallback_trades = thr, st["trades"]
            if st["trades"] >= min_train_trades and \
                    (best_exp is None or (st["expectancy_pct"] or -9e9) > best_exp):
                best_thr, best_exp, best_row = thr, st["expectancy_pct"], st
        chosen = best_thr if best_thr is not None else fallback_thr
        fold = {"fold": k + 1,
                "train_span": [dates[0], dates[s - 1]], "train_bars": s,
                "test_span": [dates[s], dates[e - 1]], "test_bars": e - s,
                "chosen_min_agree": chosen,
                "train_trades": (best_row or {}).get("trades", fallback_trades),
                "train_expectancy_pct": (best_row or {}).get("expectancy_pct"),
                "picked_by": "best expectancy" if best_thr is not None else
                             ("most trades (none cleared min-trades gate)"
                              if fallback_thr is not None else "no signals in train")}
        if chosen is None:
            fold.update({"test_trades": 0, "test_expectancy_pct": None,
                         "test_return_pct": 0.0, "note": "no trades — flat this fold"})
            folds.append(fold)
            continue
        # --- trade it on the UNSEEN test segment (start 1 bar early so the
        #     crossing test at the first test bar sees the prior bar's count) ---
        r = _sim(max(0, s - 1), e, chosen)
        base = oos_curve[-1]
        oos_curve.extend(base * v for v in r["equity"][1:])
        for t in r["trades"]:
            t["fold"] = k + 1
            t["min_agree"] = chosen
        oos_trades.extend(r["trades"])
        st = r["stats"]
        fold.update({"test_trades": st["trades"], "test_win_rate": st["win_rate"],
                     "test_expectancy_pct": st["expectancy_pct"],
                     "test_profit_factor": st["profit_factor"],
                     "test_return_pct": st["total_return_pct"]})
        folds.append(fold)

    bh_oos = float(c[-1] / c[s0] - 1.0) if c[s0] else 0.0
    stats = _agg(oos_trades, oos_curve, bh_oos)
    # in-sample expectation vs out-of-sample reality — the degradation check
    tr_exps = [f["train_expectancy_pct"] for f in folds if f.get("train_expectancy_pct") is not None]
    is_exp = round(sum(tr_exps) / len(tr_exps), 3) if tr_exps else None

    # OOS equity + buy&hold series for charting (both start at 1.0 at s0)
    bh_curve = (c[s0:] / c[s0]).tolist()
    step = max(1, len(bh_curve) // 120)
    bh_s = [{"t": dates[s0 + i], "v": round(float(bh_curve[i]), 4)}
            for i in range(0, len(bh_curve), step)]
    step2 = max(1, len(oos_curve) // 120)
    st_s = [round(float(oos_curve[i]), 4) for i in range(0, len(oos_curve), step2)]

    return {"symbol": symbol, "interval": interval, "walk_forward": True,
            "n_folds": n_folds, "thresholds": thresholds,
            "target_atr": target_atr, "stop_atr": stop_atr, "max_hold": max_hold,
            "allow_short": allow_short, "risk_pct": risk_pct,
            "bars": n, "span": [dates[0], dates[-1]],
            "oos_span": [dates[s0], dates[-1]],
            "folds": folds, "stats": stats, "trades": oos_trades,
            "equity_curve": {"strategy": st_s, "buy_hold": bh_s},
            "in_sample_expectancy_pct": is_exp,
            "oos_expectancy_pct": stats["expectancy_pct"],
            "total_agents": len(agents_mod.AGENTS),
            "note": ("Thresholds chosen on PAST data only, traded on unseen data. "
                     "This OOS record is the one to trust — if it is negative while "
                     "the sweep looked great, the sweep was curve-fit.")}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from trader import load_config
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parent / "config.yaml")
    r = backtest("BTCUSDT", cfg, interval="1d", min_agree=38)
    s = r["stats"]
    print(f"BTCUSDT 1d min=38: {s['trades']} trades, win {s['win_rate']}%, "
          f"exp {s['expectancy_pct']}%, PF {s['profit_factor']}, avgR {s['avg_r']}, "
          f"total {s['total_return_pct']}% vs B&H {s['buy_hold_pct']}%, maxDD {s['max_dd_pct']}%")
    print("\nSweep:")
    for row in sweep("BTCUSDT", cfg, interval="1d")["rows"]:
        print(f"  min={row['min_agree']:>2}: {row['trades']:>3} trades  win {row['win_rate']}%  "
              f"exp {row['expectancy_pct']}%  PF {row['profit_factor']}  tot {row['total_return_pct']}%")
