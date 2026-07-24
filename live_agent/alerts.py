"""
alerts.py — Price-level alerts with native macOS notifications.

Set an alert ("notify me if RELIANCE.NS breaks above 1,365") and a scheduled
checker fires a desktop notification when price crosses the level. Alerts persist
in state/alerts.json and latch once triggered (so you are notified once).

    python alerts.py            # run one check pass over all active alerts
    python alerts.py --list     # show all alerts

Wire it to launchd (run_alerts.sh + com.rohitraj.alerts.plist) to check every
~30 min. The dashboard's 🔔 Alerts tab manages them, and every stock's insight
page has one-click "alert on breakout / breakdown" buttons.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
ALERTS = STATE / "alerts.json"
logger = logging.getLogger("alerts")


# --------------------------------------------------------------------------- #
#  Storage
# --------------------------------------------------------------------------- #

def _load() -> List[Dict]:
    if ALERTS.exists():
        return json.loads(ALERTS.read_text())
    return []


def _save(items: List[Dict]) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    ALERTS.write_text(json.dumps(items, indent=0))


def list_alerts() -> List[Dict]:
    return _load()


def add_alert(symbol: str, direction: str, level: float, note: str = "") -> Dict:
    """Add an alert. ``direction`` is 'above' or 'below'."""
    direction = "below" if str(direction).lower().startswith("b") else "above"
    item = {
        "id": uuid.uuid4().hex[:8],
        "symbol": symbol.strip().upper(),
        "direction": direction,
        "level": float(level),
        "note": note,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "triggered": False,
        "triggered_at": None,
    }
    items = _load()
    items.append(item)
    _save(items)
    return item


def remove_alert(alert_id: str) -> None:
    _save([a for a in _load() if a["id"] != alert_id])


# --------------------------------------------------------------------------- #
#  Price + notification
# --------------------------------------------------------------------------- #

def get_price(symbol: str) -> Optional[float]:
    """Latest price — Binance for crypto, yfinance otherwise. Light/fast."""
    try:
        import binance_feed
        if binance_feed.is_crypto(symbol):
            f = binance_feed.get_ohlcv(symbol, "1d", lookback_days=10)
            return float(f["close"].iloc[-1]) if not f.empty else None
    except Exception:
        pass
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period="5d")
        return float(h["Close"].iloc[-1]) if len(h) else None
    except Exception as exc:
        logger.warning("price fetch failed for %s: %s", symbol, exc)
        return None


def notify(title: str, message: str) -> None:
    """Fire a native macOS notification (no-op off macOS)."""
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}" sound name "Glass"'],
            check=False, capture_output=True, timeout=10,
        )
    except Exception as exc:
        logger.debug("notify failed: %s", exc)


# --------------------------------------------------------------------------- #
#  Checking
# --------------------------------------------------------------------------- #

def check_alerts(do_notify: bool = True) -> List[Dict]:
    """Evaluate active alerts; latch + notify any that have crossed. Returns the
    list of alerts that fired in this pass."""
    items = _load()
    fired: List[Dict] = []
    prices: Dict[str, float] = {}

    for a in items:
        if a.get("triggered"):
            continue
        sym = a["symbol"]
        if sym not in prices:
            p = get_price(sym)
            if p is None:
                continue
            prices[sym] = p
        price = prices[sym]
        crossed = (a["direction"] == "above" and price >= a["level"]) or \
                  (a["direction"] == "below" and price <= a["level"])
        if crossed:
            a["triggered"] = True
            a["triggered_at"] = time.strftime("%Y-%m-%d %H:%M")
            a["trigger_price"] = round(price, 2)
            fired.append(a)
            msg = f"{sym} {a['direction']} {a['level']:g} — now {price:g}"
            logger.info("ALERT FIRED: %s", msg)
            if do_notify:
                notify("Trading Agent alert", msg)

    if fired:
        _save(items)
    return fired


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Price-level alert checker")
    ap.add_argument("--list", action="store_true", help="List all alerts.")
    args = ap.parse_args()

    if args.list:
        for a in list_alerts():
            st = "✓ FIRED" if a["triggered"] else "· active"
            print(f"  [{a['id']}] {st}  {a['symbol']} {a['direction']} {a['level']:g}"
                  + (f"  ({a['triggered_at']})" if a["triggered"] else ""))
        return

    fired = check_alerts()
    print(f"Checked {len(list_alerts())} alerts — {len(fired)} fired.")
    for a in fired:
        print(f"  🔔 {a['symbol']} {a['direction']} {a['level']:g} @ {a.get('trigger_price')}")


if __name__ == "__main__":
    main()
