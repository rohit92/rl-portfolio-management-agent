"""
signals.py — Generate actionable trade signals and push them to your phone.

Combines the committee consensus + the ensemble score + the calibrated forecast
into a single signal per symbol, with concrete *values to trade*: entry, target,
stop, expected move, and a confidence (vote count). New/changed signals are sent
to Telegram / WhatsApp (via notify.py) and de-duplicated so you are not spammed.

    python signals.py --watchlist            # signals for your saved watchlist
    python signals.py --symbols BTCUSDT,RELIANCE.NS --notify
    python signals.py --segment binance --interval 1h --notify   # live-ish crypto

Schedule with launchd (run_signals.sh) to poll the live market periodically.

Honesty: a "signal" here is a high-conviction *alignment* of setups, not a
prediction of profit. Direction is hard; treat values as a disciplined plan to
research and paper-trade, with the stop being the most important number.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

import agents as agents_mod
import markets
import notify
from data_feed import fetch
from forecast import forecast_tomorrow
from levels import trigger_levels
from strategies import Ensemble

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
SENT = STATE / "signals_sent.json"
logger = logging.getLogger("signals")


def _watchlist() -> List[str]:
    p = STATE / "watchlist.json"
    return json.loads(p.read_text()) if p.exists() else ["^NSEI", "BTCUSDT"]


def suggest_leverage(entry: float, stop: float, asset: str = "stock",
                     risk_pct: float = 0.01):
    """Risk-based position size, expressed as leverage.

    Returns the size at which hitting the stop costs ~``risk_pct`` of capital:
    leverage = risk_budget / stop-distance. This is a *risk cap*, not a profit
    dial — for normal stop distances it is usually < 1x (i.e. no leverage). It is
    capped low on purpose; higher leverage risks liquidation.
    """
    dist = abs(entry - stop) / entry if entry else 0.0
    if dist <= 0:
        return 0.0, "n/a"
    cap = 3.0 if asset == "crypto" else 2.0
    lev = max(0.1, min(risk_pct / dist, cap))
    return round(lev, 2), (f"sizes a ~{int(risk_pct*100)}% loss at the stop "
                           f"({dist*100:.1f}% away). Cap {cap:g}x — more risks liquidation.")


def generate(symbols: List[str], cfg: dict, interval: str = "1d",
             threshold: Optional[int] = None) -> List[Dict]:
    """Build a signal for each symbol that shows an actionable setup."""
    ens = Ensemble(cfg)
    com = agents_mod.Committee(
        {"committee": {"threshold": threshold or cfg.get("committee", {}).get("threshold", 25)}})
    data = fetch(symbols, cfg, lookback_days=1200, interval=interval)
    out: List[Dict] = []

    for sym, sd in data.items():
        try:
            verdict = com.latest(sd.frame)
            asc = ens.score_one(sd)
            f = forecast_tomorrow(sd.frame["close"], cfg["forecast"])
            lv = trigger_levels(sd, f["exp_abs_move_pts"] / f["level"])
        except Exception as exc:
            logger.warning("signal calc failed for %s: %s", sym, exc)
            continue

        thr = float(cfg["risk"]["entry_threshold"])
        # Blend ensemble score, the committee's net vote, AND market sentiment
        # (crypto Fear&Greed / India-VIX / VIX) so direction reflects all three.
        total = max(verdict["total"], 1)
        net = (verdict["long"] - verdict["short"]) / total
        try:
            import sentiment as _sent_mod
            sent = _sent_mod.market_sentiment(sym, include_news=False)
            sent_lean = float(sent.get("lean", 0.0))
        except Exception:
            sent, sent_lean = {}, 0.0
        blended = 0.55 * asc.composite + 0.35 * net + 0.10 * sent_lean
        if blended >= thr:
            side = "BUY"
        elif blended <= -thr:
            side = "SELL"
        else:
            continue  # nothing actionable

        last = float(sd.last_close)
        if side == "BUY":
            entry, target, stop = lv["breakout"]["level"], lv["breakout"]["target"], lv["breakdown"]["level"]
        else:
            entry, target, stop = lv["breakdown"]["level"], lv["breakdown"]["target"], lv["breakout"]["level"]

        # Conviction 0–100: high only when the blended direction is strong AND
        # the committee agrees. Sentiment that CONFIRMS the side sharpens it
        # (+), sentiment that contradicts dulls it (−). Risk-based leverage below.
        agree = (verdict["long"] if side == "BUY" else verdict["short"]) / total
        base_conv = 100 * min(1.0, 0.7 * min(1.0, abs(blended)) + 0.6 * agree)
        sdir = 1.0 if side == "BUY" else -1.0
        conviction = int(round(max(0.0, min(100.0, base_conv + sdir * sent_lean * 12))))
        is_cx = _is_crypto(sym)
        lev, lev_note = suggest_leverage(entry, stop, "crypto" if is_cx else "stock")

        out.append({
            "symbol": sym, "side": side, "price": round(last, 2),
            "entry": round(entry, 2), "target": round(target, 2), "stop": round(stop, 2),
            "exp_move_pct": round(f["exp_abs_move_pts"] / f["level"], 4),
            "committee": f"{verdict['long']}L/{verdict['short']}S of {verdict['total']}",
            "long_votes": verdict["long"], "short_votes": verdict["short"],
            "score": round(asc.composite, 2), "conviction": conviction,
            "leverage": lev, "leverage_note": lev_note,
            "asset": "crypto" if is_cx else "stock",
            "max_leverage": _max_leverage(sym) if is_cx else None,
            "sentiment": sent.get("summary", ""), "sentiment_lean": round(sent_lean, 2),
            "interval": interval, "time": time.strftime("%Y-%m-%d %H:%M"),
        })
    out.sort(key=lambda s: s["conviction"], reverse=True)
    return out


def _is_crypto(sym: str) -> bool:
    try:
        import binance_feed
        return binance_feed.is_crypto(sym)
    except Exception:
        return False


def _max_leverage(sym: str):
    """Typical max leverage for a crypto perp (from the derivatives table)."""
    try:
        import derivatives
        base = sym.upper().replace("-", "").replace("USDT", "").replace("USD", "")
        return derivatives._CRYPTO_MAX.get(base, derivatives._CRYPTO_DEFAULT)
    except Exception:
        return None


def format_signal(s: Dict) -> str:
    arrow = "🟢 STRONG BUY" if s["side"] == "BUY" else "🔴 STRONG SELL"
    return (f"{arrow} <b>{s['symbol']}</b> @ {s['price']}  ({s['interval']})\n"
            f"Conviction {s.get('conviction','?')}/100 · committee {s['committee']}\n"
            f"Entry {s['entry']} · Target {s['target']} · Stop {s['stop']}\n"
            f"Suggested size ~{s.get('leverage','?')}x ({s.get('leverage_note','')})\n"
            f"Exp move ~{s['exp_move_pct']*100:.1f}%\n"
            f"⚠ Not advice — research/paper-trade first. The stop is the key number; "
            f"leverage can liquidate you.")


# --------------------------------------------------------------------------- #
#  De-dup + push
# --------------------------------------------------------------------------- #

def _load_sent() -> Dict[str, str]:
    return json.loads(SENT.read_text()) if SENT.exists() else {}


def _save_sent(d: Dict[str, str]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    SENT.write_text(json.dumps(d))


def push(sigs: List[Dict]) -> List[Dict]:
    """Send only NEW/changed signals (keyed by symbol+side). Returns those sent."""
    sent = _load_sent()
    fresh: List[Dict] = []
    for s in sigs:
        key = s["symbol"]
        token = f"{s['side']}@{s['entry']}"
        if sent.get(key) == token:
            continue  # already notified for this setup
        notify.send(format_signal(s))
        sent[key] = token
        fresh.append(s)
    if fresh:
        _save_sent(sent)
    return fresh


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Generate + push trade signals")
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--watchlist", action="store_true")
    ap.add_argument("--symbols", help="Comma-separated symbols.")
    ap.add_argument("--segment", choices=["nse", "binance"], help="Scan a universe segment.")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--threshold", type=int)
    ap.add_argument("--notify", action="store_true", help="Actually send messages.")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    cfg = markets.resolve(yaml.safe_load(open(args.config)))
    if args.symbols:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    elif args.segment:
        import universe
        u = universe.load()
        syms = (u.get(args.segment, []))[:args.limit]
    else:
        syms = _watchlist()

    sigs = generate(syms, cfg, interval=args.interval, threshold=args.threshold)
    print(f"\n{len(sigs)} actionable signal(s) from {len(syms)} symbols:\n")
    for s in sigs:
        print(format_signal(s).replace("<b>", "").replace("</b>", ""), "\n")

    if args.notify:
        st = notify.status()
        on = [k for k, v in st.items() if v]
        if not on:
            print("⚠ No messaging channel configured — set TELEGRAM_* / WHATSAPP_* in .env")
        fresh = push(sigs)
        print(f"Sent {len(fresh)} new signal(s) via {on or 'nothing'}.")


if __name__ == "__main__":
    main()
