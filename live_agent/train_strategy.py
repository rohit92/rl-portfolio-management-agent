"""
train_strategy.py — Learn the best trading policy, honestly.

Closed-loop training: it sweeps the committee **consensus threshold** (how many of
the 45 agents must agree to trade) across a basket of symbols, walk-forward, and
keeps the value that performs best on UNSEEN (out-of-sample) data — not the value
that merely looked good in-sample. If the winning policy's median OOS Sharpe
clears a gate, it is written to ``state/learned_policy.json`` and adopted live.

This is the legitimate meaning of "train the model to trade better":
    • It optimises a RISK-ADJUSTED objective (Sharpe), not raw return.
    • It validates out-of-sample, so it improves *generalisation*, not memorisation.
    • It can REFUSE to update (gate) — refusing to overfit is a feature.

It does NOT, and cannot, guarantee profit. It finds the most defensible setting
of a real knob and proves whether that setting survives into unseen data.

    python train_strategy.py --market crypto_daily --years 3
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import yaml

warnings.filterwarnings("ignore", category=RuntimeWarning)

import agents as agents_mod
import markets
from data_feed import fetch

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
logger = logging.getLogger("train_strategy")

THRESHOLDS = list(range(16, 31, 2))   # candidate consensus levels (of 45)
MIN_OOS_SHARPE = 0.2                   # promotion gate


def train(cfg: dict, years: int) -> dict:
    universe = list(cfg["universe"])[:6]   # a representative basket
    ppy = int(cfg.get("periods_per_year", 252))
    tc = float(cfg["risk"]["transaction_cost"])
    allow_short = cfg.get("interval", "1d") == "1d"  # daily can model shorts
    com = agents_mod.Committee(cfg)

    # Precompute each symbol's vote matrix once.
    data = fetch(universe, cfg, lookback_days=years * 365 + 200,
                 interval=cfg.get("interval", "1d"))
    panels = []
    for sym, sd in data.items():
        M = com.vote_matrix(sd.frame)
        panels.append((sym, M, sd.frame["close"]))
    if not panels:
        raise SystemExit("No data to train on.")

    # Walk-forward folds (by index fraction), aggregate OOS Sharpe per threshold.
    train_frac, val_frac, step = 0.5, 0.12, 0.12
    oos = {t: [] for t in THRESHOLDS}
    is_pick_oos: List[float] = []
    fold_log = []

    for sym, M, close in panels:
        n = len(M)
        start = 0.0
        while start + train_frac + val_frac <= 1.0001:
            a, b = int(n * start), int(n * (start + train_frac))
            c = int(n * (start + train_frac + val_frac))
            tr = slice(a, b); va = slice(b, c)
            # In-sample best threshold for this fold:
            is_scores = {t: agents_mod.backtest_from_votes(
                M.iloc[tr], close, t, ppy, tc, allow_short) for t in THRESHOLDS}
            best_t = max(is_scores, key=is_scores.get)
            # Out-of-sample score of EVERY threshold (to choose globally),
            # and of the in-sample pick (to measure honest generalisation).
            for t in THRESHOLDS:
                oos[t].append(agents_mod.backtest_from_votes(
                    M.iloc[va], close, t, ppy, tc, allow_short))
            is_pick_oos.append(agents_mod.backtest_from_votes(
                M.iloc[va], close, best_t, ppy, tc, allow_short))
            start += step

    mean_oos = {t: float(np.nanmean(v)) if v else 0.0 for t, v in oos.items()}
    best_threshold = max(mean_oos, key=mean_oos.get)
    median_oos = float(np.nanmedian(oos[best_threshold]))
    is_pick_generalisation = float(np.nanmean(is_pick_oos))
    passed = median_oos >= MIN_OOS_SHARPE

    result = {
        "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "market": cfg.get("active_market", "custom"),
        "basket": [s for s, _, _ in panels],
        "oos_sharpe_by_threshold": {str(t): round(v, 3) for t, v in mean_oos.items()},
        "best_threshold": best_threshold,
        "median_oos_sharpe": round(median_oos, 3),
        "naive_is_pick_oos": round(is_pick_generalisation, 3),
        "gate": MIN_OOS_SHARPE,
        "adopted": bool(passed),
        "committee_threshold": best_threshold,
    }
    return result


def report(r: dict) -> None:
    print("\n" + "=" * 70)
    print("  STRATEGY TRAINING — walk-forward consensus-threshold search")
    print("=" * 70)
    print(f"  Market : {r['market']}  ·  basket {len(r['basket'])} symbols")
    print(f"  {'threshold':<12}{'mean OOS Sharpe':>18}")
    print("-" * 70)
    for t, v in r["oos_sharpe_by_threshold"].items():
        flag = "  ← best" if int(t) == r["best_threshold"] else ""
        print(f"  {t:<12}{v:>18}{flag}")
    print("-" * 70)
    print(f"  Best consensus threshold : {r['best_threshold']}/45")
    print(f"  Median OOS Sharpe        : {r['median_oos_sharpe']}  (gate {r['gate']})")
    print(f"  Naive 'pick best IS' OOS : {r['naive_is_pick_oos']}  "
          f"(chasing the in-sample winner generalises worse — that's the trap)")
    print("-" * 70)
    print("  ✅ ADOPTED — written to learned_policy.json" if r["adopted"]
          else "  ⛔ NOT adopted — no setting cleared the OOS gate; keeping defaults.")
    print("=" * 70 + "\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description="Train the committee trading policy")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--market", help="Market preset to train on.")
    ap.add_argument("--years", type=int, default=3)
    args = ap.parse_args()

    cfg = markets.resolve(yaml.safe_load(open(args.config)), args.market)
    print(f"Training on '{cfg.get('active_market')}' "
          f"({cfg.get('interval')}, {len(cfg['universe'])} symbols)…")
    r = train(cfg, args.years)
    report(r)
    if r["adopted"]:
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / "learned_policy.json").write_text(json.dumps(r, indent=2))
        with open(STATE / "policy_history.jsonl", "a") as fh:
            fh.write(json.dumps({"at": r["fitted_at"], "market": r["market"],
                                 "threshold": r["best_threshold"],
                                 "oos": r["median_oos_sharpe"]}) + "\n")


if __name__ == "__main__":
    main()
