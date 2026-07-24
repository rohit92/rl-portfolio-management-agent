"""
universe.py — Build & cache the full tradable universe.

Pulls every symbol you can actually trade and forecast:
  • NSE  — the official equity list (~2,300 stocks) → yfinance-style ``XXX.NS``.
  • Binance — every spot USDT pair with status TRADING (~430 coins) via the
    public exchangeInfo endpoint → Binance-native ``XXXUSDT`` symbols.

Cached to ``state/universe.json`` so the dashboard and scanner don't re-download
the lists every run. Refresh weekly (listings change).

    python universe.py            # build/refresh the cache, print counts
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import requests

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
CACHE = STATE / "universe.json"
logger = logging.getLogger("universe")

_NSE_CSV = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
_BINANCE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def fetch_nse() -> tuple:
    """All NSE EQ-series symbols as yfinance tickers + {symbol: company name}."""
    try:
        r = requests.get(_NSE_CSV, headers={"User-Agent": _UA,
                         "Accept": "text/csv,*/*"}, timeout=25)
        r.raise_for_status()
    except Exception as exc:
        logger.warning("NSE list fetch failed: %s", exc)
        return [], {}
    out: List[str] = []
    names: Dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(r.text)):
        sym = (row.get("SYMBOL") or "").strip()
        series = (row.get(" SERIES") or row.get("SERIES") or "").strip()
        name = (row.get("NAME OF COMPANY") or "").strip()
        if sym and series in ("EQ", "BE"):
            out.append(f"{sym}.NS")
            if name:
                names[f"{sym}.NS"] = name
    logger.info("NSE: %d symbols", len(out))
    return sorted(set(out)), names


def fetch_binance() -> List[str]:
    """All Binance spot USDT pairs with status TRADING (``XXXUSDT``)."""
    try:
        r = requests.get(_BINANCE_INFO, timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("Binance exchangeInfo fetch failed: %s", exc)
        return []
    out = [
        s["symbol"] for s in data.get("symbols", [])
        if s.get("status") == "TRADING"
        and s.get("quoteAsset") == "USDT"
        and s.get("isSpotTradingAllowed", True)
    ]
    logger.info("Binance: %d USDT pairs", len(out))
    return sorted(set(out))


def build(force: bool = True) -> Dict:
    """Fetch both lists and write the cache. Returns the universe dict."""
    STATE.mkdir(parents=True, exist_ok=True)
    nse_syms, nse_names = fetch_nse()
    universe = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nse": nse_syms,
        "nse_names": nse_names,
        "binance": fetch_binance(),
    }
    universe["counts"] = {k: len(universe[k]) for k in ("nse", "binance")}
    CACHE.write_text(json.dumps(universe, indent=0))
    return universe


def load() -> Dict:
    """Load the cached universe, building it on first use."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return build()


def all_symbols() -> List[str]:
    u = load()
    return u.get("nse", []) + u.get("binance", [])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    u = build()
    print(f"\nUniverse cached to {CACHE}")
    print(f"  NSE stocks    : {u['counts']['nse']:,}")
    print(f"  Binance coins : {u['counts']['binance']:,}")
    print(f"  TOTAL         : {sum(u['counts'].values()):,}")
    print(f"  Updated       : {u['updated']}\n")
