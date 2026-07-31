"""
performance.py — turn the trade journal into a performance report.

The journal (journal.py) is a logbook of taken signals: each closed trade has a
realised ``pnl_pct`` and, when a stop was set, an ``r_multiple``. This module
reads those closed trades and computes the metrics a quant actually cares about
— win rate, profit factor, expectancy, per-trade Sharpe, max drawdown of the
equity curve, and a Calmar-style return/risk ratio — then prints a report.

The metric definitions match utils/metrics.py in the parent project (Sharpe =
mean/std, drawdown = max peak-to-trough on the equity curve); they are
re-implemented here as small pure functions so live_agent stays self-contained
and the report can run offline with no dependencies.

Usage:
    python performance.py                 # human-readable report from the journal
    python performance.py --json          # machine-readable metrics
    python performance.py --equity out.csv  # also write the equity curve as CSV
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Optional, Sequence


# --------------------------------------------------------------------------- #
#  Pure helpers — operate on a list of trade dicts, no file/journal needed
# --------------------------------------------------------------------------- #

def closed_trades(trades: Sequence[Dict]) -> List[Dict]:
    """Closed trades with a realised P&L, ordered chronologically by close."""

    done = [
        t for t in trades
        if t.get("status") == "closed" and t.get("pnl_pct") is not None
    ]
    return sorted(done, key=lambda t: t.get("closed_ts", t.get("opened_ts", 0)))


def trade_returns(trades: Sequence[Dict]) -> List[float]:
    """Per-trade simple returns (fractional) from ``pnl_pct``, in close order."""

    return [t["pnl_pct"] / 100.0 for t in closed_trades(trades)]


def equity_curve(trades: Sequence[Dict], starting: float = 1.0) -> List[float]:
    """Compounded equity over the closed trades, seeded at ``starting``."""

    equity = starting
    curve = [equity]
    for r in trade_returns(trades):
        equity *= (1.0 + r)
        curve.append(equity)
    return curve


def sharpe_per_trade(returns: Sequence[float]) -> float:
    """Per-trade Sharpe: mean(return) / std(return). 0.0 if undefined.

    Not annualised — trades are event-spaced, not calendar-spaced — so this is a
    system-quality figure (reward per unit of per-trade variability), directly
    comparable across strategies.
    """

    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    std = math.sqrt(var)
    return 0.0 if std == 0 else mean / std


def max_drawdown(curve: Sequence[float]) -> float:
    """Max peak-to-trough decline of an equity curve, as a positive fraction."""

    if len(curve) < 2:
        return 0.0
    peak = curve[0]
    worst = 0.0
    for value in curve:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


def profit_factor(trades: Sequence[Dict]) -> Optional[float]:
    """Gross profit / gross loss. None if there are no losses (undefined)."""

    gains = sum(t["pnl_pct"] for t in closed_trades(trades) if t["pnl_pct"] > 0)
    losses = -sum(t["pnl_pct"] for t in closed_trades(trades) if t["pnl_pct"] < 0)
    if losses == 0:
        return None
    return gains / losses


def summarize(trades: Sequence[Dict], starting_capital: float = 10_000.0) -> Dict:
    """Compute the full performance summary as a plain dict."""

    done = closed_trades(trades)
    returns = [t["pnl_pct"] / 100.0 for t in done]
    curve = equity_curve(done, starting=starting_capital)

    wins = [t for t in done if t["pnl_pct"] > 0]
    losses = [t for t in done if t["pnl_pct"] < 0]
    r_multiples = [t["r_multiple"] for t in done if t.get("r_multiple") is not None]

    n = len(done)
    total_return_pct = (curve[-1] / starting_capital - 1.0) * 100.0 if n else 0.0
    mdd = max_drawdown(curve)

    return {
        "trades": n,
        "open_trades": sum(1 for t in trades if t.get("status") == "open"),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100.0 * len(wins) / n, 2) if n else 0.0,
        "avg_win_pct": round(sum(t["pnl_pct"] for t in wins) / len(wins), 3) if wins else 0.0,
        "avg_loss_pct": round(sum(t["pnl_pct"] for t in losses) / len(losses), 3) if losses else 0.0,
        "expectancy_pct": round(sum(t["pnl_pct"] for t in done) / n, 3) if n else 0.0,
        "profit_factor": round(profit_factor(done), 3) if profit_factor(done) is not None else None,
        "avg_r_multiple": round(sum(r_multiples) / len(r_multiples), 3) if r_multiples else None,
        "sharpe_per_trade": round(sharpe_per_trade(returns), 3),
        "max_drawdown_pct": round(mdd * 100.0, 2),
        "total_return_pct": round(total_return_pct, 2),
        "calmar": round((total_return_pct / 100.0) / mdd, 3) if mdd > 0 else None,
        "starting_capital": starting_capital,
        "ending_equity": round(curve[-1], 2),
    }


def format_report(summary: Dict) -> str:
    """Render a summary dict as an aligned, human-readable report."""

    def line(label: str, value, suffix: str = "") -> str:
        shown = "—" if value is None else f"{value}{suffix}"
        return f"  {label:<22}{shown}"

    return "\n".join([
        "Trade Journal — Performance Report",
        "=" * 40,
        line("Closed trades", summary["trades"]),
        line("Open trades", summary["open_trades"]),
        line("Win rate", summary["win_rate_pct"], "%"),
        line("Avg win", summary["avg_win_pct"], "%"),
        line("Avg loss", summary["avg_loss_pct"], "%"),
        line("Expectancy / trade", summary["expectancy_pct"], "%"),
        line("Profit factor", summary["profit_factor"]),
        line("Avg R-multiple", summary["avg_r_multiple"]),
        line("Sharpe (per trade)", summary["sharpe_per_trade"]),
        line("Max drawdown", summary["max_drawdown_pct"], "%"),
        line("Total return", summary["total_return_pct"], "%"),
        line("Calmar (ret/DD)", summary["calmar"]),
        "-" * 40,
        line("Starting capital", summary["starting_capital"]),
        line("Ending equity", summary["ending_equity"]),
    ])


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def _load_journal() -> List[Dict]:
    import journal  # local import so the pure helpers above need no journal file
    return journal._load()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Trade-journal performance report.")
    parser.add_argument("--json", action="store_true", help="Emit metrics as JSON.")
    parser.add_argument("--capital", type=float, default=10_000.0,
                        help="Starting capital for the equity curve.")
    parser.add_argument("--equity", metavar="CSV",
                        help="Write the equity curve to this CSV path.")
    args = parser.parse_args(argv)

    trades = _load_journal()
    summary = summarize(trades, starting_capital=args.capital)

    if args.equity:
        curve = equity_curve(closed_trades(trades), starting=args.capital)
        with open(args.equity, "w", encoding="utf-8") as fh:
            fh.write("trade_index,equity\n")
            for i, value in enumerate(curve):
                fh.write(f"{i},{value:.4f}\n")

    print(json.dumps(summary, indent=2) if args.json else format_report(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
