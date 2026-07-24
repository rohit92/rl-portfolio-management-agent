"""
derivatives.py — Leverage & derivatives reference across markets.

Pulls the live list of Binance USDT **perpetual futures** (official public API)
and pairs each with a typical max leverage. Adds a reference table for Indian
(NSE) derivatives. Cached so the dashboard loads instantly.

Honest notes:
  • Max leverage is *risk-tiered and broker-dependent* — it shrinks as your
    position size grows, and exchanges change it. These are typical top-tier
    figures, not promises.
  • For NSE F&O, real margins (SPAN+exposure) come from your broker; the figures
    here are indicative. Per-contract option chains need a broker/paid feed.
  • High leverage is the #1 cause of retail blow-ups. This is reference data, not
    a suggestion to use it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "state" / "derivatives.json"
logger = logging.getLogger("derivatives")

_FUT_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"

# Typical top-tier max leverage for major coins (Binance). Others default below.
_CRYPTO_MAX = {
    "BTC": 125, "ETH": 100, "BNB": 75, "SOL": 50, "XRP": 75, "DOGE": 75,
    "ADA": 75, "AVAX": 75, "LINK": 75, "LTC": 75, "BCH": 75, "DOT": 75,
    "MATIC": 75, "TRX": 75, "ATOM": 50, "NEAR": 50, "ARB": 50, "OP": 50,
}
_CRYPTO_DEFAULT = 25

# Indicative NSE derivatives reference (broker margins vary).
_NSE_DERIVS = [
    {"instrument": "NIFTY futures", "type": "Index future", "approx_leverage": "~6x",
     "note": "SPAN+exposure ≈ 15–18% margin; 1 lot = 50 (varies)."},
    {"instrument": "BANKNIFTY futures", "type": "Index future", "approx_leverage": "~6x",
     "note": "Higher vol → higher margin; 1 lot = 15 (varies)."},
    {"instrument": "Stock futures", "type": "Single-stock future", "approx_leverage": "~5x",
     "note": "Margin ≈ 20–25%; lot size per stock."},
    {"instrument": "Index options (buy)", "type": "Option", "approx_leverage": "premium only",
     "note": "Defined risk = premium paid; cheap but decays (theta)."},
    {"instrument": "Index options (sell)", "type": "Option", "approx_leverage": "margin ≈ future",
     "note": "Undefined risk; needs large margin; not for beginners."},
    {"instrument": "Stock options", "type": "Option", "approx_leverage": "premium / margin",
     "note": "Liquidity thins outside the top names."},
]


def fetch_binance_perps() -> List[Dict]:
    try:
        r = requests.get(_FUT_INFO, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("Binance futures fetch failed: %s", exc)
        return []
    out = []
    for s in data.get("symbols", []):
        if (s.get("contractType") == "PERPETUAL" and s.get("quoteAsset") == "USDT"
                and s.get("status") == "TRADING"):
            base = s.get("baseAsset", "")
            out.append({
                "symbol": s["symbol"], "base": base,
                "max_leverage": _CRYPTO_MAX.get(base, _CRYPTO_DEFAULT),
                "maint_margin_pct": float(s.get("maintMarginPercent", 0) or 0),
            })
    out.sort(key=lambda x: (-x["max_leverage"], x["base"]))
    return out


def build(force: bool = True) -> Dict:
    perps = fetch_binance_perps()
    payload = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "crypto_perps": perps,
        "crypto_count": len(perps),
        "nse_derivatives": _NSE_DERIVS,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload))
    return payload


def load(max_age_hours: int = 24) -> Dict:
    if CACHE.exists():
        d = json.loads(CACHE.read_text())
        return d
    return build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    d = build()
    print(f"\nBinance USDT perpetuals: {d['crypto_count']}")
    print("Top by leverage:")
    for p in d["crypto_perps"][:12]:
        print(f"  {p['symbol']:<14} up to {p['max_leverage']}x")
    print(f"\nNSE derivatives reference: {len(d['nse_derivatives'])} instrument types")
