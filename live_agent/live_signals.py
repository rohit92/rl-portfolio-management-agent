"""
live_signals.py — Best signals across the whole market, market-hours aware.

Every run it figures out which markets are open (crypto 24/7; NSE only on a
weekday 09:15–15:30 IST), pulls the strongest candidates from the cached
nightly scan for those markets, computes fresh signals on them, ranks by
conviction, and pushes the STRONG ones to your phone (Telegram/WhatsApp).

Why "top candidates from the cache" and not "all 2,800 symbols every 30 min":
re-ranking the full universe live would take ~an hour. The nightly scan already
ranks everything by composite; here we refresh signals only on its top names —
fast, and focused on the most attractive setups.

    python live_signals.py                 # compute, cache, print
    python live_signals.py --notify        # also push STRONG signals to phone

Schedule with run_signals.sh / com.rohitraj.signals.plist (every 30 min).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
from typing import Dict, List
from zoneinfo import ZoneInfo

import yaml

import markets
import notify
import signals as sig_mod

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
CACHE = STATE / "live_signals.json"
IST = ZoneInfo("Asia/Kolkata")
STRONG = 60  # conviction threshold to call a signal STRONG / push it
logger = logging.getLogger("live_signals")


def market_status(now_ist: dt.datetime | None = None) -> Dict:
    """Which markets are open right now (IST)."""
    now = now_ist or dt.datetime.now(IST)
    weekday = now.weekday() < 5
    open_t, close_t = dt.time(9, 15), dt.time(15, 30)
    nse_open = weekday and (open_t <= now.time() <= close_t)
    return {
        "ist": now.strftime("%Y-%m-%d %H:%M IST"),
        "nse_open": nse_open,
        "crypto_open": True,  # always
        "segments": (["binance", "nse"] if nse_open else ["binance"]),
    }


def _candidates(segment: str, top: int) -> List[str]:
    """Top symbols by |composite| from the cached nightly scan for a segment."""
    path = STATE / f"scan_results_{segment}.json"
    if not path.exists():
        return []
    rows = json.loads(path.read_text()).get("rows", [])
    rows.sort(key=lambda r: abs(r.get("composite") or 0), reverse=True)
    return [r["symbol"] for r in rows[:top]]


def best_signals(cfg: dict, top_per_segment: int = 30) -> Dict:
    status = market_status()
    syms: List[str] = []
    for seg in status["segments"]:
        syms += _candidates(seg, top_per_segment)
    if not syms:
        logger.warning("No cached candidates — run scan.py first.")
    sigs = sig_mod.generate(syms, cfg) if syms else []
    longs = [s for s in sigs if s["side"] == "BUY"]
    shorts = [s for s in sigs if s["side"] == "SELL"]
    return {
        "updated": status["ist"], "status": status,
        "strong_long": [s for s in longs if s["conviction"] >= STRONG][:10],
        "strong_short": [s for s in shorts if s["conviction"] >= STRONG][:10],
        "all": sigs[:40],
    }


def run(cfg: dict, do_notify: bool) -> Dict:
    res = best_signals(cfg)
    STATE.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(res))
    strong = res["strong_long"] + res["strong_short"]
    if do_notify and strong:
        sig_mod.push(strong)  # de-duplicated inside push()
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Market-wide best signals")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--notify", action="store_true")
    args = ap.parse_args()
    cfg = markets.resolve(yaml.safe_load(open(args.config)))

    res = run(cfg, args.notify)
    st = res["status"]
    print(f"\n{res['updated']} — NSE {'OPEN' if st['nse_open'] else 'closed'}, "
          f"crypto OPEN. Scanning: {', '.join(st['segments'])}\n")
    for s in res["strong_long"] + res["strong_short"]:
        print(sig_mod.format_signal(s).replace("<b>", "").replace("</b>", ""), "\n")
    if not (res["strong_long"] or res["strong_short"]):
        print("No STRONG signals right now (conviction < %d)." % STRONG)
    if args.notify:
        on = [k for k, v in notify.status().items() if v and k != "macos"]
        print(f"Pushed STRONG signals via: {on or 'no phone channel (set .env)'}")


if __name__ == "__main__":
    main()
