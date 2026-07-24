"""
auto_learn.py — The "learns by itself" engine: walk-forward auto-retraining.

The honest version of self-learning. Instead of letting the agent optimise
against all of history (which just memorises noise), this:

    1. Rolls a window across history: TRAIN slice -> VALIDATE slice -> step.
    2. On each TRAIN slice, searches a grid of ensemble weights and keeps the
       best by in-sample Sharpe.
    3. Scores those weights on the *next* VALIDATE slice — data the optimiser
       never saw. This out-of-sample (OOS) number is the only one that matters.
    4. Reports the IS-vs-OOS gap — the "overfitting tax". A strategy that looks
       great in-sample but collapses out-of-sample is learning noise; we want
       weights whose edge *survives* into unseen data.
    5. Promotion gate: only if the median OOS Sharpe across folds clears
       ``learning.min_oos_sharpe`` do we write the freshly-fitted weights to
       ``state/learned_weights.json`` for the live trader to adopt. Otherwise we
       refuse to "learn" and keep the hand-set defaults.

This is how a systematic strategy adapts to regime change *without* fooling
itself. It does NOT guarantee profit — nothing does. It guarantees discipline.

    python auto_learn.py            # run a learning pass, maybe promote weights
    python auto_learn.py --dry-run  # report only, never write learned weights

Schedule it weekly (see README) so the live agent keeps adapting over time.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import markets
from data_feed import fetch
from risk import target_weights
from strategies import AssetScore, signal_components

# Reuse the parent project's metrics.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.metrics import sharpe_ratio  # noqa: E402

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE / "state"
SIGNAL_NAMES = ["momentum", "mean_reversion", "trend"]
logger = logging.getLogger("auto_learn")


# --------------------------------------------------------------------------- #
#  Build a fast, weight-independent panel of per-day signal components
# --------------------------------------------------------------------------- #

def build_panel(cfg: dict) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Precompute per-symbol signal components, volatility, and forward return.

    The three signals do NOT depend on the ensemble weights, so we compute them
    once. Searching weight combinations afterwards is then just cheap weighted
    sums — making the walk-forward grid search fast.

    Returns
    -------
    panel : dict[str, pd.DataFrame]
        symbol -> frame with columns [momentum, mean_reversion, trend, vol,
        fwd_ret], indexed by date.
    dates : pd.DatetimeIndex
        Dates common to every symbol (ascending), excluding the final day
        (which has no forward return).
    """
    years = int(cfg["learning"]["history_years"])
    data = fetch(list(cfg["universe"]), cfg, lookback_days=years * 365 + 200)
    if len(data) < 2:
        raise SystemExit("Not enough symbols with data to learn from.")

    panel: Dict[str, pd.DataFrame] = {}
    common = None
    for sym, sd in data.items():
        comp = signal_components(sd.frame)
        comp["vol"] = sd.frame["vol"]
        # Forward (next-day) simple return: the reward for today's decision.
        comp["fwd_ret"] = sd.frame["close"].pct_change().shift(-1)
        comp = comp.dropna()
        panel[sym] = comp
        common = comp.index if common is None else common.intersection(comp.index)

    dates = common.sort_values()
    logger.info("Panel built: %d symbols, %d common dates.", len(panel), len(dates))
    return panel, dates


# --------------------------------------------------------------------------- #
#  Evaluate one weight vector over a date slice
# --------------------------------------------------------------------------- #

def weight_grid(step: float) -> List[Dict[str, float]]:
    """Enumerate non-negative weight combos over the 3 signals summing to 1.

    e.g. step=0.25 -> 15 combinations like {momentum:0.5, mean_reversion:0.25,
    trend:0.25}. Pure trend, pure momentum, etc. are included.
    """
    n = round(1.0 / step)
    combos: List[Dict[str, float]] = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            combos.append({
                "momentum": a / n,
                "mean_reversion": b / n,
                "trend": c / n,
            })
    return combos


def evaluate(
    panel: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    weights: Dict[str, float],
    cfg: dict,
) -> float:
    """Sharpe of the strategy using ``weights`` over ``dates``.

    Reuses the live ``target_weights`` sizing for fidelity, so the offline
    objective matches live behaviour exactly.
    """
    tc = float(cfg["risk"]["transaction_cost"])
    ppy = int(cfg.get("periods_per_year", 252))
    rets: List[float] = []
    prev_w: Dict[str, float] = {}

    for t in dates:
        scores: Dict[str, AssetScore] = {}
        for sym, frame in panel.items():
            if t not in frame.index:
                continue
            row = frame.loc[t]
            composite = sum(weights[s] * float(row[s]) for s in SIGNAL_NAMES)
            scores[sym] = AssetScore(
                symbol=sym,
                composite=composite,
                components={s: float(row[s]) for s in SIGNAL_NAMES},
                volatility=float(row["vol"]),
            )
        w = target_weights(scores, cfg)
        port = sum(w.get(s, 0.0) * float(panel[s].loc[t, "fwd_ret"])
                   for s in w)
        turnover = sum(abs(w.get(s, 0.0) - prev_w.get(s, 0.0))
                       for s in set(w) | set(prev_w))
        rets.append(port - turnover * tc)
        prev_w = w

    return float(sharpe_ratio(np.array(rets, dtype=np.float64), periods_per_year=ppy))


# --------------------------------------------------------------------------- #
#  Walk-forward learning
# --------------------------------------------------------------------------- #

def walk_forward(cfg: dict) -> dict:
    """Run the walk-forward search and return the learning record."""
    lc = cfg["learning"]
    panel, dates = build_panel(cfg)
    grid = weight_grid(float(lc["weight_step"]))
    logger.info("Searching %d weight combos per fold.", len(grid))

    train_n = int(lc["train_days"])
    val_n = int(lc["validate_days"])
    step = int(lc["step_days"])

    folds: List[dict] = []
    start = 0
    while start + train_n + val_n <= len(dates):
        train_dates = dates[start:start + train_n]
        val_dates = dates[start + train_n:start + train_n + val_n]

        # In-sample: pick the weights with the best TRAIN Sharpe.
        best_w, best_is = None, -np.inf
        for w in grid:
            s = evaluate(panel, train_dates, w, cfg)
            if s > best_is:
                best_is, best_w = s, w

        # Out-of-sample: judge those weights on unseen forward data.
        oos = evaluate(panel, val_dates, best_w, cfg)
        folds.append({
            "train_end": str(train_dates[-1].date()),
            "val_end": str(val_dates[-1].date()),
            "weights": best_w,
            "is_sharpe": round(best_is, 3),
            "oos_sharpe": round(oos, 3),
        })
        logger.info(
            "Fold to %s | IS Sharpe %.2f -> OOS Sharpe %.2f | w=%s",
            folds[-1]["val_end"], best_is, oos,
            {k: round(v, 2) for k, v in best_w.items()},
        )
        start += step

    if not folds:
        raise SystemExit("Not enough history for even one walk-forward fold.")

    # Fit "current" weights on the most recent train window for live use.
    recent = dates[-(train_n + val_n):-val_n] if len(dates) >= train_n + val_n \
        else dates[:train_n]
    cur_w, cur_is = None, -np.inf
    for w in grid:
        s = evaluate(panel, recent, w, cfg)
        if s > cur_is:
            cur_is, cur_w = s, w

    median_oos = float(np.median([f["oos_sharpe"] for f in folds]))
    mean_is = float(np.mean([f["is_sharpe"] for f in folds]))
    mean_oos = float(np.mean([f["oos_sharpe"] for f in folds]))

    return {
        "folds": folds,
        "current_weights": cur_w,
        "median_oos_sharpe": round(median_oos, 3),
        "mean_is_sharpe": round(mean_is, 3),
        "mean_oos_sharpe": round(mean_oos, 3),
        "overfitting_gap": round(mean_is - mean_oos, 3),
    }


# --------------------------------------------------------------------------- #
#  Promotion + reporting
# --------------------------------------------------------------------------- #

def maybe_promote(result: dict, cfg: dict, dry_run: bool) -> bool:
    """Write learned weights only if they clear the OOS promotion gate."""
    gate = float(cfg["learning"]["min_oos_sharpe"])
    passed = result["median_oos_sharpe"] >= gate
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "adopted": bool(passed and not dry_run),
        "weights": result["current_weights"],
        "median_oos_sharpe": result["median_oos_sharpe"],
        "gate": gate,
        "fitted_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "note": (
            "Promoted: median OOS Sharpe cleared the gate."
            if passed else
            "NOT promoted: OOS Sharpe below gate — keeping config defaults."
        ),
    }
    if dry_run:
        payload["note"] += " (dry-run: nothing written to learned_weights.json)"
    else:
        (STATE_DIR / "learned_weights.json").write_text(json.dumps(payload, indent=2))

    # Append to a learning history for later inspection / plotting.
    if not dry_run:
        with open(STATE_DIR / "learning_history.jsonl", "a") as fh:
            fh.write(json.dumps({
                "at": payload["fitted_at"],
                "median_oos_sharpe": result["median_oos_sharpe"],
                "overfitting_gap": result["overfitting_gap"],
                "adopted": payload["adopted"],
                "weights": result["current_weights"],
            }) + "\n")
    return passed


def report(result: dict, cfg: dict, promoted: bool) -> None:
    print("\n" + "=" * 74)
    print("  WALK-FORWARD SELF-LEARNING REPORT")
    print("=" * 74)
    print(f"  {'Fold ends':<12}{'IS Sharpe':>11}{'OOS Sharpe':>12}   weights (m/r/t)")
    print("-" * 74)
    for f in result["folds"]:
        w = f["weights"]
        print(f"  {f['val_end']:<12}{f['is_sharpe']:>11.2f}{f['oos_sharpe']:>12.2f}"
              f"   {w['momentum']:.2f}/{w['mean_reversion']:.2f}/{w['trend']:.2f}")
    print("-" * 74)
    print(f"  Mean IS Sharpe   : {result['mean_is_sharpe']:.2f}")
    print(f"  Mean OOS Sharpe  : {result['mean_oos_sharpe']:.2f}")
    print(f"  Overfitting tax  : {result['overfitting_gap']:.2f}  (IS - OOS; lower is healthier)")
    print(f"  Median OOS Sharpe: {result['median_oos_sharpe']:.2f}  "
          f"(gate = {cfg['learning']['min_oos_sharpe']})")
    cw = result["current_weights"]
    print(f"  Fitted weights   : momentum {cw['momentum']:.2f} | "
          f"mean_reversion {cw['mean_reversion']:.2f} | trend {cw['trend']:.2f}")
    print("-" * 74)
    if promoted:
        print("  ✅ PROMOTED — live agent will adopt these weights.")
    else:
        print("  ⛔ NOT promoted — edge did not survive out-of-sample. "
              "Keeping hand-set defaults (this is the system refusing to overfit).")
    print("=" * 74 + "\n")


def plot_history(out_dir: Path) -> None:
    """Plot the OOS-Sharpe learning trajectory if history exists."""
    path = STATE_DIR / "learning_history.jsonl"
    if not path.exists():
        return
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(rows) < 2:
        return
    xs = [pd.Timestamp(r["at"]) for r in rows]
    plt.figure(figsize=(10, 5))
    plt.plot(xs, [r["median_oos_sharpe"] for r in rows], "-o", label="Median OOS Sharpe")
    plt.plot(xs, [r["overfitting_gap"] for r in rows], "-o", alpha=0.6, label="Overfitting tax")
    plt.axhline(0, color="gray", lw=0.8)
    plt.title("Self-learning trajectory over time")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "learning_trajectory.png", dpi=120)
    plt.close()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Walk-forward self-learning")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Report only; never write learned weights.")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = markets.resolve(yaml.safe_load(fh))

    print(f"Running walk-forward self-learning on '{cfg.get('active_market')}' "
          f"({cfg.get('interval')})...")
    result = walk_forward(cfg)
    promoted = maybe_promote(result, cfg, args.dry_run)
    report(result, cfg, promoted)
    plot_history(HERE / "backtest_results")


if __name__ == "__main__":
    main()
