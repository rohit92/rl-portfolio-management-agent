"""
signal_perf.py — The honest hit-rate of our own signals.

Telegram/WhatsApp channels claim ">80% correct." This measures the TRUTH for our
signal rule: over real history, when a signal fired, how often did price actually
move the predicted way, and — more importantly — what was the *expectancy*
(average P&L per signal)? A strategy can be "right" 70% of the time and still
lose money; expectancy and profit factor are what matter.

Method (no lookahead): at each bar compute the same blended signal the live
system uses (0.6·ensemble + 0.4·committee-net), gate by conviction, then look at
the realised forward return over ``horizon`` bars. Aggregate across a basket.

    python signal_perf.py --market crypto_daily --years 3 --horizon 5
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

warnings.filterwarnings("ignore", category=RuntimeWarning)

import agents as agents_mod
import markets
from data_feed import fetch
from strategies import signal_components

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("signal_perf")
N_AGENTS = len(agents_mod.AGENTS)


def _blended_series(frame, cfg):
    """Per-bar blended signal + conviction, matching the live signal rule."""
    w = dict(cfg["strategy"]["weights"])
    tot = sum(abs(v) for v in w.values()) or 1.0
    w = {k: v / tot for k, v in w.items()}
    comp = signal_components(frame)                      # momentum/mean_rev/trend
    composite = sum(w[k] * comp[k] for k in w)           # ensemble score series
    M = agents_mod.Committee(cfg).vote_matrix(frame)
    longs = (M == 1).sum(axis=1); shorts = (M == -1).sum(axis=1)
    net = (longs - shorts) / N_AGENTS
    blended = 0.6 * composite + 0.4 * net
    agree = np.where(blended >= 0, longs, shorts) / N_AGENTS
    conviction = np.minimum(1.0, 0.7 * blended.abs() + 0.6 * agree) * 100
    return blended, conviction


def analyze(cfg: dict, years: int, horizon: int, conviction_gate: int) -> Dict:
    basket = list(cfg["universe"])[:8]
    interval = cfg.get("interval", "1d")
    data = fetch(basket, cfg, lookback_days=years * 365 + 200, interval=interval)
    if not data:
        raise SystemExit("No data.")

    wins: List[float] = []      # signed forward return in the signal's direction
    base_up = 0; base_n = 0     # unconditional up-rate (the base rate to beat)
    entry_thr = float(cfg["risk"]["entry_threshold"])

    for sym, sd in data.items():
        frame = sd.frame
        blended, conviction = _blended_series(frame, cfg)
        close = frame["close"]
        fwd = close.shift(-horizon) / close - 1.0
        base_up += int((fwd > 0).sum()); base_n += int(fwd.notna().sum())
        fire = (conviction >= conviction_gate) & (blended.abs() >= entry_thr) & fwd.notna()
        idx = np.where(fire.to_numpy())[0]
        b = blended.to_numpy(); f = fwd.to_numpy()
        for i in idx:
            direction = 1.0 if b[i] >= 0 else -1.0
            wins.append(direction * f[i])     # >0 ⇒ correct direction

    arr = np.array(wins, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n_signals": 0, "note": "No signals cleared the gate."}
    correct = arr > 0
    win_rets = arr[arr > 0]; loss_rets = arr[arr <= 0]
    avg_win = float(win_rets.mean()) if win_rets.size else 0.0
    avg_loss = float(loss_rets.mean()) if loss_rets.size else 0.0
    gross_win = float(win_rets.sum()); gross_loss = float(-loss_rets.sum())
    return {
        "market": cfg.get("active_market"), "interval": interval,
        "horizon": horizon, "conviction_gate": conviction_gate,
        "n_signals": n,
        "hit_rate": round(100 * correct.mean(), 1),
        "base_up_rate": round(100 * base_up / base_n, 1) if base_n else 0,
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "expectancy_pct": round(arr.mean() * 100, 3),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else 0.0,
    }


def analyze_sweep(cfg: dict, years: int, horizon: int,
                  gates: List[int]) -> Dict:
    """Hit rate / expectancy across several conviction gates (one data fetch).

    Shows the real quality-vs-quantity trade-off: raising the bar improves the
    numbers but fires far fewer signals.
    """
    basket = list(cfg["universe"])[:8]
    interval = cfg.get("interval", "1d")
    data = fetch(basket, cfg, lookback_days=years * 365 + 200, interval=interval)
    if not data:
        raise SystemExit("No data.")
    entry = float(cfg["risk"]["entry_threshold"])

    pre = []
    base_up = base_n = 0
    for sym, sd in data.items():
        blended, conviction = _blended_series(sd.frame, cfg)
        close = sd.frame["close"]
        fwd = (close.shift(-horizon) / close - 1.0).to_numpy()
        base_up += int(np.nansum(fwd > 0)); base_n += int((~np.isnan(fwd)).sum())
        pre.append((blended.to_numpy(), conviction.to_numpy(), fwd))

    rows = []
    for g in gates:
        wins = []
        for b, conv, f in pre:
            fire = (conv >= g) & (np.abs(b) >= entry) & ~np.isnan(f)
            for i in np.where(fire)[0]:
                wins.append((1.0 if b[i] >= 0 else -1.0) * f[i])
        arr = np.array(wins, dtype=float)
        if arr.size == 0:
            rows.append({"gate": g, "n": 0})
            continue
        w, l = arr[arr > 0], arr[arr <= 0]
        pf = (w.sum() / -l.sum()) if l.size and l.sum() != 0 else 0.0
        rows.append({
            "gate": g, "n": int(arr.size),
            "hit_rate": round(100 * (arr > 0).mean(), 1),
            "expectancy_pct": round(arr.mean() * 100, 3),
            "avg_win_pct": round(w.mean() * 100, 2) if w.size else 0,
            "avg_loss_pct": round(l.mean() * 100, 2) if l.size else 0,
            "profit_factor": round(pf, 2),
        })
    return {
        "market": cfg.get("active_market"), "interval": interval,
        "horizon": horizon, "base_up_rate": round(100 * base_up / base_n, 1) if base_n else 0,
        "rows": rows,
    }


def report(r: Dict) -> None:
    print("\n" + "=" * 64)
    print("  SIGNAL HIT-RATE — the honest number")
    print("=" * 64)
    if r.get("n_signals", 0) == 0:
        print("  " + r.get("note", "No signals.")); print("=" * 64 + "\n"); return
    print(f"  Market {r['market']} · {r['interval']} · {r['horizon']}-bar horizon · "
          f"conviction ≥ {r['conviction_gate']}")
    print("-" * 64)
    print(f"  Signals fired      : {r['n_signals']}")
    print(f"  Hit rate (correct) : {r['hit_rate']}%   "
          f"(vs base up-rate {r['base_up_rate']}%)")
    print(f"  Avg win / avg loss : +{r['avg_win_pct']}% / {r['avg_loss_pct']}%")
    print(f"  Expectancy / trade : {r['expectancy_pct']}%   "
          f"(THIS decides profit, not hit rate)")
    print(f"  Profit factor      : {r['profit_factor']}  (>1 = edge after the fact)")
    print("-" * 64)
    print("  Reality check: a real edge looks like 52–56% with positive expectancy —")
    print("  NOT the '80%+ always' that signal channels advertise. If you see 80%,")
    print("  ask to see the LOSSES and the expectancy. They won't show you.")
    print("=" * 64 + "\n")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description="Measure real signal hit-rate")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--market")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--conviction", type=int, default=50)
    args = ap.parse_args()
    cfg = markets.resolve(yaml.safe_load(open(args.config)), args.market)
    print(f"Measuring signal hit-rate on '{cfg.get('active_market')}'…")
    report(analyze(cfg, args.years, args.horizon, args.conviction))


if __name__ == "__main__":
    main()
