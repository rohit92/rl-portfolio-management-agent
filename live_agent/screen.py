"""
screen.py — On-demand candidate screener.

Give it stock names (or a market preset) and it ranks them by the agent's
*defined, backtested* signal stack and attaches each name's probabilistic
next-day forecast plus headline context. Re-running re-scrapes fresh data, so it
"updates itself" every time you run it (schedule it daily for hands-off updates).

    python screen.py --symbols "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS"
    python screen.py --market india_daily
    python screen.py --symbols "AAPL,NVDA,MSFT" --news 3

WHAT THIS IS — and is NOT
-------------------------
It ranks names by the ensemble composite (momentum + mean-reversion + trend) and
shows the forecaster's next-day expected move, P(up), and volatility regime.
That composite is the same signal the backtests measured — so the ranking is
*honest about what it is*: a watchlist ordered by a transparent metric.

It is NOT a prediction that the top names will profit. Direction is
~unforecastable (P(up) hovers near the base rate); the value is in the *range/
vol*, and even the composite ranking did not beat buy-and-hold in backtests. Use
this to decide what to *research and paper-trade*, never as a buy list. News is
context for a human to read, not an alpha signal.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

import markets
import news as news_mod
from data_feed import fetch
from forecast import forecast_tomorrow
from strategies import Ensemble

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("screen")


def _vol_regime(sd) -> str:
    """Label current volatility vs the symbol's own recent history."""
    vol = sd.frame["vol"].dropna()
    if len(vol) < 30:
        return "?"
    cur, med = float(vol.iloc[-1]), float(vol.median())
    if cur > 1.5 * med:
        return "HIGH"
    if cur < 0.7 * med:
        return "calm"
    return "norm"


def screen(symbols: List[str], cfg: dict, fc: dict) -> List[Dict]:
    """Score + forecast every symbol; return rows sorted by composite desc."""
    # One fetch per symbol, with enough history for both indicators and the
    # volatility forecast. data_feed handles the active market's interval.
    data = fetch(symbols, cfg, lookback_days=2500)
    if not data:
        raise SystemExit("No data for any requested symbol.")

    ensemble = Ensemble(cfg)
    rows: List[Dict] = []
    for sym, sd in data.items():
        asc = ensemble.score_one(sd)
        try:
            f = forecast_tomorrow(sd.frame["close"], fc)
            exp_move_pct = f["exp_abs_move_pts"] / f["level"]
            p_up, p_up1 = f["p_up"], f["p_up_gt1pct"]
        except Exception as exc:  # forecast needs enough history
            logger.warning("Forecast failed for %s: %s", sym, exc)
            exp_move_pct, p_up, p_up1 = float("nan"), float("nan"), float("nan")
        rows.append({
            "symbol": sym,
            "composite": asc.composite,
            "components": asc.components,
            "last": sd.last_close,
            "exp_move_pct": exp_move_pct,
            "p_up": p_up,
            "p_up1": p_up1,
            "regime": _vol_regime(sd),
        })
    rows.sort(key=lambda r: r["composite"], reverse=True)
    return rows


def report(rows: List[Dict], cfg: dict, top: int, n_news: int) -> None:
    thr = float(cfg["risk"]["entry_threshold"])
    print("\n" + "=" * 86)
    print("  CANDIDATE SCREEN — ranked by composite signal (momentum/mean-rev/trend)")
    print("=" * 86)
    print(f"  {'#':>2} {'Symbol':<12}{'Score':>7}{'mom/rev/trd':>14}"
          f"{'Last':>11}{'ExpMove':>9}{'P(up)':>7}{'Vol':>6}")
    print("-" * 86)
    for i, r in enumerate(rows, 1):
        c = r["components"]
        flag = " ◀ candidate" if r["composite"] >= thr else ""
        em = f"{r['exp_move_pct']:.1%}" if np.isfinite(r["exp_move_pct"]) else "  —"
        pu = f"{r['p_up']:.0%}" if np.isfinite(r["p_up"]) else " —"
        print(f"  {i:>2} {r['symbol']:<12}{r['composite']:>+7.2f}"
              f"{c['momentum']:>5.1f}/{c['mean_reversion']:.1f}/{c['trend']:.1f}"
              f"{r['last']:>11,.1f}{em:>9}{pu:>7}{r['regime']:>6}{flag}")
    print("-" * 86)
    print("  Score ≥ %.2f clears the agent's entry threshold (◀). 'ExpMove' is the"
          % thr)
    print("  forecast |next-day move|; 'P(up)' ≈ base rate (direction is a coin-flip).")
    print("=" * 86)

    # Headlines for the top names — context for a human, not a signal.
    if n_news > 0:
        print("\n  HEADLINES (context only — official Yahoo feed, not forum scraping)\n")
        for r in rows[:top]:
            heads = news_mod.get_headlines(r["symbol"], limit=n_news)
            print(f"  {r['symbol']}:")
            if not heads:
                print("     (no headlines available)")
            for h in heads:
                print(f"     {news_mod.lean_label(h['lean'])} {h['title'][:72]}"
                      f"  — {h['publisher']}")
            print()

    print("  " + "-" * 82)
    print("  ⚠ This is a research watchlist ordered by a transparent metric — NOT a")
    print("    buy list and NOT a profit prediction. The composite did not beat")
    print("    buy-and-hold in backtests. Validate + paper-trade before risking money.")
    print("  " + "-" * 82 + "\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="On-demand candidate screener")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--symbols", help="Comma-separated tickers to screen.")
    parser.add_argument("--market", help="Use a market preset's universe.")
    parser.add_argument("--top", type=int, default=5, help="Rows to attach news to.")
    parser.add_argument("--news", type=int, default=2,
                        help="Headlines per top name (0 to skip).")
    args = parser.parse_args()

    with open(args.config) as fh:
        base = yaml.safe_load(fh)
    cfg = markets.resolve(base, args.market)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list(cfg["universe"])

    print(f"Screening {len(symbols)} symbols (fetching fresh data)...")
    rows = screen(symbols, cfg, base["forecast"])
    report(rows, cfg, top=args.top, n_news=args.news)


if __name__ == "__main__":
    main()
