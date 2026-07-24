"""
backtest.py — Validate the ensemble across markets and timeframes.

The agent is market- and timeframe-agnostic, and so is this backtester. Point it
at any preset from config.yaml's ``markets:`` (US daily, crypto hourly, NSE
daily, weekly, ...) and it replays the *exact* live code path
(``Ensemble.score_all`` + ``risk.target_weights``) bar by bar, with the correct
annualisation factor for that timeframe, against an equal-weight benchmark.

    python backtest.py --market crypto_daily --years 3
    python backtest.py --all                  # sweep every market -> leaderboard

No lookahead: every indicator is strictly trailing; at each bar ``t`` we score
with data up to ``t`` and earn the return from ``t`` to ``t+1``. Turnover is
charged the preset's transaction cost.

> Honest caveat — DATA SNOOPING. Sweeping many markets/timeframes and keeping the
> best is itself a form of overfitting: with enough combinations, one looks great
> by luck. Treat the sweep as a robustness check ("does the edge appear broadly?"),
> not a stock picker ("trade whichever scored highest"). Real confidence comes
> from the walk-forward out-of-sample test in auto_learn.py, not from this table.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import yfinance as yf

import markets
from data_feed import SymbolData, fetch
from risk import target_weights
from strategies import Ensemble

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.metrics import (  # noqa: E402
    annualized_return,
    max_drawdown,
    sharpe_ratio,
    win_rate,
)

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("backtest")


# Bars-per-year by (interval, is_crypto) for correct annualisation.
_PPY = {
    ("1d", False): 252, ("1d", True): 365,
    ("1wk", False): 52, ("1wk", True): 52,
    ("1h", False): 1638, ("1h", True): 8760,
    ("30m", False): 3276, ("30m", True): 17520,
}


def backtest_symbol(symbol: str, interval: str, cfg: dict,
                    lookback_days: int = 1500) -> dict:
    """Backtest the ensemble *timing* on ONE symbol at a given timeframe.

    Goes long (full) when the composite signal clears the entry threshold, else
    flat, charging the preset's transaction cost on each switch. Compared to
    buy-and-hold. Works for stocks (yfinance) and crypto (Binance) alike.
    """
    try:
        import binance_feed
        crypto = binance_feed.is_crypto(symbol)
    except Exception:
        crypto = False
    ppy = _PPY.get((interval, crypto), 252)

    data = fetch([symbol], cfg, lookback_days=lookback_days, interval=interval)
    sd = data.get(symbol)
    if sd is None:
        raise SystemExit(f"No data for {symbol} at {interval}.")

    ensemble = Ensemble(cfg)
    thr = float(cfg["risk"]["entry_threshold"])
    tc = float(cfg["risk"]["transaction_cost"])
    close = sd.frame["close"]
    idx = sd.frame.index

    strat, bh, prev_pos = [], [], 0.0
    for i in range(len(idx) - 1):
        t = idx[i]
        view = SymbolData(symbol=symbol, frame=sd.frame.loc[:t])
        score = ensemble.score_one(view).composite
        pos = 1.0 if score >= thr else 0.0
        r = float(close.iloc[i + 1]) / float(close.iloc[i]) - 1.0
        strat.append(pos * r - abs(pos - prev_pos) * tc)
        bh.append(r)
        prev_pos = pos

    s = np.array(strat); b = np.array(bh)
    eq_s = np.cumprod(1.0 + s); eq_b = np.cumprod(1.0 + b)
    # Downsample equity for a compact chart.
    step = max(1, len(eq_s) // 150)
    spark = [{"i": k, "s": round(float(eq_s[k]), 4), "b": round(float(eq_b[k]), 4)}
             for k in range(0, len(eq_s), step)]
    return {
        "symbol": symbol, "interval": interval, "bars": len(s),
        "span": [str(idx[0].date()), str(idx[-1].date())],
        "strat": {"sharpe": round(float(sharpe_ratio(s, periods_per_year=ppy)), 2),
                  "total": round(float(eq_s[-1] - 1), 4),
                  "max_dd": round(float(max_drawdown(eq_s)), 4)},
        "bh": {"sharpe": round(float(sharpe_ratio(b, periods_per_year=ppy)), 2),
               "total": round(float(eq_b[-1] - 1), 4),
               "max_dd": round(float(max_drawdown(eq_b)), 4)},
        "spark": spark,
    }


def _common_dates(data: Dict[str, SymbolData]) -> pd.DatetimeIndex:
    idx = None
    for sd in data.values():
        idx = sd.frame.index if idx is None else idx.intersection(sd.frame.index)
    return idx.sort_values()


def run_backtest(cfg: dict, years: int) -> dict:
    """Replay the live strategy over ``years`` of history for the active market."""
    universe = list(cfg["universe"])
    interval = cfg.get("interval", "1d")
    ppy = int(cfg.get("periods_per_year", 252))
    lookback_days = years * 365 + 200
    data = fetch(universe, cfg, lookback_days=lookback_days, interval=interval)
    if len(data) < 2:
        raise SystemExit(f"Not enough symbols with data ({interval}) to backtest.")

    dates = _common_dates(data)
    if len(dates) < 40:
        raise SystemExit(f"Not enough overlapping {interval} bars to backtest.")

    ensemble = Ensemble(cfg)
    tc = float(cfg["risk"]["transaction_cost"])
    closes = {s: sd.frame["close"] for s, sd in data.items()}

    strat_rets: List[float] = []
    bench_rets: List[float] = []
    prev_w: Dict[str, float] = {}

    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        view = {s: SymbolData(symbol=s, frame=sd.frame.loc[:t])
                for s, sd in data.items()}
        weights = target_weights(ensemble.score_all(view), cfg)

        day_ret = {s: float(closes[s].loc[t1]) / float(closes[s].loc[t]) - 1.0
                   for s in data}
        port = sum(weights.get(s, 0.0) * day_ret[s] for s in data)
        turnover = sum(abs(weights.get(s, 0.0) - prev_w.get(s, 0.0)) for s in data)
        strat_rets.append(port - turnover * tc)
        prev_w = weights
        bench_rets.append(float(np.mean([day_ret[s] for s in data])))

    strat = np.array(strat_rets, dtype=np.float64)
    bench = np.array(bench_rets, dtype=np.float64)
    spy = _spy_returns(dates) if interval == "1d" else None

    init = float(cfg.get("initial_cash", 100_000))
    return {
        "market": cfg.get("active_market", "custom"),
        "interval": interval,
        "ppy": ppy,
        "dates": dates[1:],
        "strat_equity": init * np.cumprod(1.0 + strat),
        "bench_equity": init * np.cumprod(1.0 + bench),
        "spy_equity": (init * np.cumprod(1.0 + spy)) if spy is not None else None,
        "strat_rets": strat, "bench_rets": bench, "spy_rets": spy,
    }


def _spy_returns(dates: pd.DatetimeIndex) -> np.ndarray | None:
    try:
        raw = yf.Ticker("SPY").history(
            start=dates[0], end=dates[-1] + pd.Timedelta(days=2),
            interval="1d", auto_adjust=True,
        )
        if raw.empty:
            return None
        close = raw["Close"]
        if getattr(close.index, "tz", None) is not None:
            close.index = close.index.tz_localize(None)
        rets = close.reindex(dates, method="ffill").pct_change().to_numpy()[1:]
        return np.nan_to_num(rets)
    except Exception as exc:
        logger.warning("Could not fetch SPY benchmark: %s", exc)
        return None


def _metrics_row(name: str, rets: np.ndarray, equity: np.ndarray, ppy: int) -> dict:
    cagr = float(annualized_return(rets, periods_per_year=ppy))
    mdd = float(max_drawdown(equity))
    return {
        "name": name,
        "total": float(equity[-1] / equity[0] - 1.0),
        "cagr": cagr,
        "sharpe": float(sharpe_ratio(rets, periods_per_year=ppy)),
        "max_dd": mdd,
        "win": float(win_rate(rets)),
        "calmar": (cagr / mdd) if mdd > 0 else 0.0,
    }


def report(res: dict) -> List[dict]:
    ppy = res["ppy"]
    rows = [
        _metrics_row("Ensemble", res["strat_rets"], res["strat_equity"], ppy),
        _metrics_row("EqualWeight", res["bench_rets"], res["bench_equity"], ppy),
    ]
    if res["spy_rets"] is not None and res["spy_equity"] is not None:
        rows.append(_metrics_row("SPY", res["spy_rets"], res["spy_equity"], ppy))

    print("\n" + "=" * 80)
    print(f"  BACKTEST [{res['market']} | {res['interval']}] "
          f"{res['dates'][0]:%Y-%m-%d} → {res['dates'][-1]:%Y-%m-%d} "
          f"({len(res['strat_rets'])} bars)")
    print("=" * 80)
    print(f"  {'Strategy':<14}{'TotalRet':>10}{'CAGR':>9}{'Sharpe':>9}"
          f"{'MaxDD':>9}{'Win':>8}{'Calmar':>9}")
    print("-" * 80)
    for r in rows:
        print(f"  {r['name']:<14}{r['total']:>9.2%}{r['cagr']:>9.2%}"
              f"{r['sharpe']:>9.2f}{r['max_dd']:>8.2%}{r['win']:>8.1%}"
              f"{r['calmar']:>9.2f}")
    print("=" * 80 + "\n")
    return rows


def plot(res: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"equity_{res['market']}.png"
    plt.figure(figsize=(11, 6))
    plt.plot(res["dates"], res["strat_equity"], label="Ensemble agent", lw=2)
    plt.plot(res["dates"], res["bench_equity"], label="Equal-weight B&H",
             lw=1.4, alpha=0.8)
    if res["spy_equity"] is not None:
        plt.plot(res["dates"], res["spy_equity"], label="SPY",
                 lw=1.4, alpha=0.8, ls="--")
    plt.title(f"Ensemble Agent — {res['market']} ({res['interval']})")
    plt.ylabel("Portfolio value ($)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def sweep(base_cfg: dict, years: int) -> None:
    """Backtest every configured market and print a ranked leaderboard."""
    names = markets.list_markets(base_cfg)
    if not names:
        raise SystemExit("No markets configured to sweep.")

    board: List[dict] = []
    for name in names:
        try:
            cfg = markets.resolve(base_cfg, name)
            res = run_backtest(cfg, years)
            rows = {r["name"]: r for r in report(res)}
            plot(res, HERE / "backtest_results")
            ens, bench = rows["Ensemble"], rows["EqualWeight"]
            board.append({
                "market": name, "interval": res["interval"],
                "bars": len(res["strat_rets"]),
                "sharpe": ens["sharpe"], "cagr": ens["cagr"],
                "max_dd": ens["max_dd"],
                "edge": ens["sharpe"] - bench["sharpe"],
            })
        except SystemExit as exc:
            logger.warning("Skipping %s: %s", name, exc)

    board.sort(key=lambda r: r["sharpe"], reverse=True)
    print("\n" + "#" * 80)
    print("  MULTI-MARKET LEADERBOARD (ranked by Ensemble Sharpe)")
    print("#" * 80)
    print(f"  {'Market':<18}{'TF':>5}{'Bars':>7}{'Sharpe':>9}{'CAGR':>9}"
          f"{'MaxDD':>9}{'Edge vs B&H':>13}")
    print("-" * 80)
    for r in board:
        print(f"  {r['market']:<18}{r['interval']:>5}{r['bars']:>7}"
              f"{r['sharpe']:>9.2f}{r['cagr']:>9.2%}{r['max_dd']:>8.2%}"
              f"{r['edge']:>+13.2f}")
    print("#" * 80)
    print("  ⚠ Data-snooping caveat: do NOT just trade the top row. With enough")
    print("    market/timeframe combos, one wins by luck. Use this to ask 'is the")
    print("    edge broad?' — then trust auto_learn.py's out-of-sample gate, not this.")
    print("#" * 80 + "\n")


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Backtest across markets/timeframes")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--market", help="Market preset to test (default: active_market).")
    parser.add_argument("--all", action="store_true", help="Sweep every market.")
    parser.add_argument("--years", type=int, default=3, help="Years of history.")
    args = parser.parse_args()

    with open(args.config) as fh:
        base_cfg = yaml.safe_load(fh)

    if args.all:
        print(f"Sweeping {len(markets.list_markets(base_cfg))} markets "
              f"({args.years}y each)... this fetches a lot of data.")
        sweep(base_cfg, args.years)
        return

    cfg = markets.resolve(base_cfg, args.market)
    print(f"Backtesting '{cfg.get('active_market')}' "
          f"({cfg.get('interval')}, {len(cfg['universe'])} symbols)...")
    res = run_backtest(cfg, args.years)
    report(res)
    png = plot(res, HERE / "backtest_results")
    print(f"Equity curve saved to {png}")


if __name__ == "__main__":
    main()
