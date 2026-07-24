"""
scan.py — Batch-scan the full universe and cache a ranked result.

Ranking thousands of symbols can't happen inside a web click (it's an hour of
API calls). So this runs OFFLINE — on a schedule — scores every requested
symbol, and writes ``state/scan_results.json``. The dashboard then loads that
cached ranking instantly.

    python scan.py --segment binance            # ~430 coins
    python scan.py --segment nse --limit 300     # first 300 NSE names
    python scan.py --segment all                 # everything (slow! ~hour)

Throttled and fault-tolerant: a symbol that fails to fetch is skipped, not fatal.
Re-run it nightly via launchd so the dashboard's "Top picks" stays fresh.

Honesty unchanged: this ranks by the backtested composite signal — a research
shortlist, NOT a buy list, and the composite did not beat buy-and-hold.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

import markets
import universe as universe_mod
from data_feed import fetch
from forecast import forecast_tomorrow
from screen import _vol_regime
from strategies import Ensemble
from trader import apply_learned_weights, load_config

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
logger = logging.getLogger("scan")


def _compute(sym: str, cfg: dict, ensemble: Ensemble, fc: dict) -> Dict | None:
    data = fetch([sym], cfg, lookback_days=1000)
    sd = data.get(sym)
    if sd is None:
        return None
    asc = ensemble.score_one(sd)
    try:
        f = forecast_tomorrow(sd.frame["close"], fc)
        exp_move = f["exp_abs_move_pts"] / f["level"]
        p_up = f["p_up"]
    except Exception:
        exp_move, p_up = float("nan"), float("nan")
    return {
        "symbol": sym,
        "composite": round(float(asc.composite), 4),
        "is_candidate": bool(asc.composite >= float(cfg["risk"]["entry_threshold"])),
        "last": round(sd.last_close, 4),
        "exp_move_pct": None if np.isnan(exp_move) else round(float(exp_move), 4),
        "p_up": None if np.isnan(p_up) else round(float(p_up), 4),
        "regime": _vol_regime(sd),
    }


def select_symbols(segment: str, limit: int | None) -> List[str]:
    u = universe_mod.load()
    if segment == "nse":
        syms = u.get("nse", [])
    elif segment == "binance":
        syms = u.get("binance", [])
    else:
        syms = u.get("nse", []) + u.get("binance", [])
    return syms[:limit] if limit else syms


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Batch scan the universe")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--segment", choices=["nse", "binance", "all"],
                        default="binance")
    parser.add_argument("--limit", type=int, help="Cap how many symbols to scan.")
    parser.add_argument("--delay", type=float, default=0.2,
                        help="Seconds between symbols (be kind to the APIs).")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    cfg = markets.resolve(load_config(Path(args.config)))
    apply_learned_weights(cfg)
    ensemble = Ensemble(cfg)
    fc = cfg["forecast"]

    symbols = select_symbols(args.segment, args.limit)
    print(f"Scanning {len(symbols)} symbols [{args.segment}] — this runs offline; "
          f"grab a coffee.\n")

    rows: List[Dict] = []
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        try:
            r = _compute(sym, cfg, ensemble, fc)
            if r:
                rows.append(r)
        except Exception as exc:
            logger.warning("scan %s failed: %s", sym, exc)
        if i % 25 == 0:
            logger.info("…%d/%d (%.0fs elapsed, %d scored)",
                        i, len(symbols), time.time() - t0, len(rows))
        time.sleep(args.delay)

    rows.sort(key=lambda r: r["composite"], reverse=True)
    STATE.mkdir(parents=True, exist_ok=True)
    payload = {
        "scanned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "segment": args.segment,
        "n_scanned": len(rows),
        "threshold": float(cfg["risk"]["entry_threshold"]),
        "rows": rows,
    }
    # Per-segment file (so crypto vs NSE stay separate) + a generic alias.
    (STATE / f"scan_results_{args.segment}.json").write_text(json.dumps(payload))
    (STATE / "scan_results.json").write_text(json.dumps(payload))

    cands = [r for r in rows if r["is_candidate"]]
    print(f"\nScanned {len(rows)} symbols in {time.time()-t0:.0f}s. "
          f"{len(cands)} cleared the signal threshold.")
    print(f"\nTop {args.top} by composite signal:")
    print(f"  {'Symbol':<14}{'Score':>8}{'Last':>14}{'ExpMove':>9}{'P(up)':>7}{'Vol':>6}")
    for r in rows[:args.top]:
        em = f"{r['exp_move_pct']:.1%}" if r["exp_move_pct"] is not None else "—"
        pu = f"{r['p_up']:.0%}" if r["p_up"] is not None else "—"
        flag = " ◀" if r["is_candidate"] else ""
        print(f"  {r['symbol']:<14}{r['composite']:>+8.2f}{r['last']:>14,.2f}"
              f"{em:>9}{pu:>7}{r['regime']:>6}{flag}")
    print(f"\nCached → state/scan_results.json (the dashboard reads this).")
    print("⚠ Research shortlist by a defined metric — NOT a buy list.\n")


if __name__ == "__main__":
    main()
