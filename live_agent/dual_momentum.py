"""
dual_momentum.py — Two research-backed strategies, implemented honestly.

  • Time-Series Momentum (TSMOM) — Moskowitz, Ooi & Pedersen (2012): go LONG an
    asset if its 12-month (skip 1) return is positive, SHORT if negative.
  • Dual Momentum — Antonacci (2014): combine *relative* momentum (rank the
    universe) with an *absolute* trend filter (only hold positive-momentum longs
    / negative-momentum shorts), which truncates the left-tail drawdown.

Both are applied cross-asset (equities, gold, bonds, EM, crypto) and
**volatility-targeted** — because the literature (Kim et al. 2016) shows a large
part of TSMOM's benefit comes from vol-scaling, not the raw signal. So we
implement it the honest way and let the backtest show the real, debated edge.

    python dual_momentum.py --years 8

Long AND short, diversified, monthly rebalanced. Still not a guarantee — the
evidence is real but contested, and edges decay. Judge it on risk-adjusted terms.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

import markets
from data_feed import fetch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.metrics import max_drawdown, sharpe_ratio  # noqa: E402

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("dual_momentum")

# Diversified cross-asset universe (the setting where dual momentum shines).
UNIVERSE = ["SPY", "QQQ", "GLD", "TLT", "EEM", "^NSEI", "BTC-USD"]
LOOKBACK, SKIP, REBAL = 252, 21, 21   # 12-month momentum, skip 1m, monthly rebalance


def _closes(cfg: dict, years: int) -> pd.DataFrame:
    data = fetch(UNIVERSE, cfg, lookback_days=years * 365 + 400)
    df = pd.DataFrame({s: sd.frame["close"] for s, sd in data.items()})
    return df.dropna()


def _momentum(close: pd.DataFrame) -> pd.DataFrame:
    """12-month return skipping the most recent month (avoids 1-month reversal)."""
    return close.shift(SKIP) / close.shift(SKIP + LOOKBACK) - 1.0


def backtest(cfg: dict, years: int = 8, top: int = 2, allow_short: bool = True,
             vol_target: float = 0.12, tc: float = 0.001) -> dict:
    close = _closes(cfg, years)
    if len(close) < LOOKBACK + 60:
        raise SystemExit("Not enough overlapping history.")
    rets = close.pct_change().fillna(0.0)
    mom = _momentum(close)
    ann_vol = rets.rolling(20).std() * np.sqrt(252)

    cols = list(close.columns)
    weights = pd.DataFrame(0.0, index=close.index, columns=cols)
    last_w = pd.Series(0.0, index=cols)

    for i, dt in enumerate(close.index):
        if i < LOOKBACK + SKIP:
            weights.iloc[i] = last_w
            continue
        if i % REBAL == 0:
            m = mom.loc[dt].dropna()
            v = ann_vol.loc[dt].replace(0, np.nan)
            w = pd.Series(0.0, index=cols)
            if len(m) >= 2:
                ranked = m.sort_values(ascending=False)
                # DUAL momentum: relative rank + absolute (sign) filter.
                longs = [s for s in ranked.index[:top] if m[s] > 0]
                shorts = [s for s in ranked.index[-top:] if m[s] < 0] if allow_short else []
                for s in longs:
                    w[s] = min(1.5, vol_target / v.get(s, np.nan)) if v.get(s) else 0.0
                for s in shorts:
                    w[s] = -min(1.5, vol_target / v.get(s, np.nan)) if v.get(s) else 0.0
                gross = w.abs().sum()
                if gross > 1.0:            # cap gross exposure at 100%
                    w = w / gross
            last_w = w
        weights.iloc[i] = last_w

    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    strat = (weights.shift(1) * rets).sum(axis=1) - turnover * tc
    bh = rets.mean(axis=1)   # equal-weight buy & hold of the universe
    eq_s = (1 + strat).cumprod(); eq_b = (1 + bh).cumprod()
    return {
        "years": years, "universe": cols,
        "start": str(close.index[0].date()), "end": str(close.index[-1].date()),
        "strat": {"total": float(eq_s.iloc[-1] - 1),
                  "cagr": float(eq_s.iloc[-1] ** (252 / len(eq_s)) - 1),
                  "sharpe": float(sharpe_ratio(strat.to_numpy(), periods_per_year=252)),
                  "max_dd": float(max_drawdown(eq_s.to_numpy()))},
        "bh": {"total": float(eq_b.iloc[-1] - 1),
               "cagr": float(eq_b.iloc[-1] ** (252 / len(eq_b)) - 1),
               "sharpe": float(sharpe_ratio(bh.to_numpy(), periods_per_year=252)),
               "max_dd": float(max_drawdown(eq_b.to_numpy()))},
        "current": _current_positioning(mom, ann_vol, top, allow_short, vol_target),
    }


def _current_positioning(mom, ann_vol, top, allow_short, vol_target) -> List[Dict]:
    m = mom.iloc[-1].dropna(); v = ann_vol.iloc[-1]
    ranked = m.sort_values(ascending=False)
    out = []
    longs = [s for s in ranked.index[:top] if m[s] > 0]
    shorts = [s for s in ranked.index[-top:] if m[s] < 0] if allow_short else []
    for s in ranked.index:
        side = "LONG" if s in longs else "SHORT" if s in shorts else "flat"
        out.append({"symbol": s, "momentum_12m_pct": round(float(m[s]) * 100, 1),
                    "position": side})
    return out


def report(r: dict) -> None:
    print("\n" + "=" * 72)
    print(f"  DUAL / TIME-SERIES MOMENTUM  ({r['start']} → {r['end']})")
    print(f"  Universe: {', '.join(r['universe'])}")
    print("=" * 72)
    print(f"  {'':<16}{'Total':>10}{'CAGR':>9}{'Sharpe':>9}{'MaxDD':>9}")
    for name, k in [("Dual momentum", "strat"), ("Equal-wt buy&hold", "bh")]:
        d = r[k]
        print(f"  {name:<16}{d['total']*100:>9.0f}%{d['cagr']*100:>8.1f}%"
              f"{d['sharpe']:>9.2f}{d['max_dd']*100:>8.0f}%")
    print("-" * 72)
    print("  Current positioning:")
    for p in r["current"]:
        tag = "▲" if p["position"] == "LONG" else "▼" if p["position"] == "SHORT" else " "
        print(f"    {tag} {p['symbol']:<9} 12m mom {p['momentum_12m_pct']:+6.1f}%  → {p['position']}")
    print("-" * 72)
    print("  Honest note: real but *contested* edge — much of TSMOM's benefit is")
    print("  vol-scaling (Kim 2016). Backtests overstate; edges decay. Paper-trade.")
    print("=" * 72 + "\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description="Dual / time-series momentum backtest")
    ap.add_argument("--years", type=int, default=8)
    ap.add_argument("--no-short", action="store_true")
    args = ap.parse_args()
    cfg = markets.resolve(yaml.safe_load(open(HERE / "config.yaml")))
    print("Backtesting dual momentum across a diversified universe…")
    report(backtest(cfg, years=args.years, allow_short=not args.no_short))


if __name__ == "__main__":
    main()
