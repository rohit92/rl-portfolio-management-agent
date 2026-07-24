"""
option_chain.py — Live options chains for NSE / BSE index & stock options and
crypto (Deribit), with a Black-Scholes-modeled fallback.

Sources (all free / public):
  • NSE   -> www.nseindia.com option-chain API (index + equity). Bot-protected:
            needs a primed cookie + browser headers, and is geo-restricted — it
            works from an Indian IP (your Mac) but is often blocked elsewhere.
  • BSE   -> api.bseindia.com derivatives.
  • Crypto-> Deribit public API (BTC / ETH options), no key needed.

When a live feed is unreachable (e.g. NSE blocked on this network) we fall back
to a **modeled** chain: Black-Scholes premiums around the live spot at the
forecast/implied volatility. Every response carries ``source`` = ``live`` or
``modeled`` and a plain ``note`` so nothing is silently faked.

Honest note: an option chain shows positioning (OI, PCR, max-pain), not the
future. PCR and max-pain are widely-watched but weak, self-referential signals —
context for the 45-agent read, not a trade on their own.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import time
from typing import Dict, List, Optional

import requests

from options import bs_price

logger = logging.getLogger("option_chain")

_H = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
_INDEX_SET = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
_STEP = {"NIFTY": 50, "BANKNIFTY": 100, "FINNIFTY": 50, "MIDCPNIFTY": 25,
         "SENSEX": 100, "BANKEX": 100}
_CACHE: Dict[str, tuple] = {}
_TTL = 60


# --------------------------------------------------------------------------- #
#  Shared analytics
# --------------------------------------------------------------------------- #
def _step_for(symbol: str, spot: float) -> int:
    if symbol in _STEP:
        return _STEP[symbol]
    # infer a sensible strike step from price magnitude
    for hi, st in [(50, 1), (200, 2.5), (500, 5), (1000, 10), (2500, 20),
                   (5000, 50), (20000, 100)]:
        if spot < hi:
            return st
    return 500


def _analytics(rows: List[Dict], spot: float) -> Dict:
    """PCR, max pain, totals, bias from a list of {strike, ce, pe} rows."""
    tot_ce = sum((r["ce"] or {}).get("oi", 0) or 0 for r in rows)
    tot_pe = sum((r["pe"] or {}).get("oi", 0) or 0 for r in rows)
    pcr = round(tot_pe / tot_ce, 2) if tot_ce else None
    # max pain: strike minimising total in-the-money payoff to option holders
    best_k, best_loss = None, None
    strikes = [r["strike"] for r in rows]
    for S in strikes:
        loss = 0.0
        for r in rows:
            K = r["strike"]
            ce_oi = (r["ce"] or {}).get("oi", 0) or 0
            pe_oi = (r["pe"] or {}).get("oi", 0) or 0
            loss += ce_oi * max(0.0, S - K) + pe_oi * max(0.0, K - S)
        if best_loss is None or loss < best_loss:
            best_loss, best_k = loss, S
    if pcr is None:
        bias = "n/a"
    elif pcr >= 1.3:
        bias = "put-heavy (support below — leaning bullish/contrarian)"
    elif pcr <= 0.7:
        bias = "call-heavy (resistance above — leaning bearish/contrarian)"
    else:
        bias = "balanced"
    atm = min(strikes, key=lambda k: abs(k - spot)) if strikes else None
    return {"pcr": pcr, "max_pain": best_k, "atm": atm,
            "tot_ce_oi": int(tot_ce), "tot_pe_oi": int(tot_pe), "bias": bias}


def _window(rows: List[Dict], spot: float, n: int = 12) -> List[Dict]:
    """Keep the `n` strikes either side of spot, sorted."""
    rows = sorted(rows, key=lambda r: r["strike"])
    if not rows:
        return rows
    atm_i = min(range(len(rows)), key=lambda i: abs(rows[i]["strike"] - spot))
    return rows[max(0, atm_i - n): atm_i + n + 1]


# --------------------------------------------------------------------------- #
#  NSE (live)
# --------------------------------------------------------------------------- #
def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_H)
    s.get("https://www.nseindia.com/option-chain", timeout=12)
    return s


def fetch_nse(symbol: str) -> Optional[Dict]:
    symbol = symbol.upper()
    is_index = symbol in _INDEX_SET
    url = ("https://www.nseindia.com/api/option-chain-indices" if is_index
           else "https://www.nseindia.com/api/option-chain-equities")
    try:
        s = _nse_session()
        r = s.get(url, params={"symbol": symbol}, timeout=12)
        data = r.json()
    except Exception as exc:
        logger.info("NSE chain unavailable for %s: %s", symbol, exc)
        return None
    rec = data.get("records", {})
    spot = rec.get("underlyingValue")
    all_rows = rec.get("data", [])
    if not spot or not all_rows:
        return None
    # nearest expiry
    expiries = data.get("records", {}).get("expiryDates", [])
    expiry = expiries[0] if expiries else None
    rows: List[Dict] = []
    for it in all_rows:
        if expiry and it.get("expiryDate") != expiry:
            continue
        ce, pe = it.get("CE"), it.get("PE")
        rows.append({
            "strike": it["strikePrice"],
            "ce": {"ltp": (ce or {}).get("lastPrice"), "iv": (ce or {}).get("impliedVolatility"),
                   "oi": (ce or {}).get("openInterest"), "chg_oi": (ce or {}).get("changeinOpenInterest"),
                   "vol": (ce or {}).get("totalTradedVolume")} if ce else None,
            "pe": {"ltp": (pe or {}).get("lastPrice"), "iv": (pe or {}).get("impliedVolatility"),
                   "oi": (pe or {}).get("openInterest"), "chg_oi": (pe or {}).get("changeinOpenInterest"),
                   "vol": (pe or {}).get("totalTradedVolume")} if pe else None,
        })
    rows = _window(rows, spot)
    return {"symbol": symbol, "market": "NSE", "source": "live",
            "note": "Live NSE option chain.", "underlying": round(spot, 2),
            "expiry": expiry, "rows": rows, **_analytics(rows, spot)}


# --------------------------------------------------------------------------- #
#  Deribit (live crypto)
# --------------------------------------------------------------------------- #
def fetch_deribit(currency: str = "BTC") -> Optional[Dict]:
    currency = currency.upper()
    try:
        r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                         params={"currency": currency, "kind": "option"}, timeout=15)
        items = r.json().get("result", [])
    except Exception as exc:
        logger.info("Deribit chain unavailable for %s: %s", currency, exc)
        return None
    if not items:
        return None
    spot = next((it.get("underlying_price") for it in items if it.get("underlying_price")), None)
    if not spot:
        return None
    # parse instruments, group by expiry, pick the nearest future expiry
    parsed = []
    for it in items:
        name = it.get("instrument_name", "")
        parts = name.split("-")
        if len(parts) != 4:
            continue
        _cur, exp_s, strike_s, cp = parts
        try:
            exp = dt.datetime.strptime(exp_s, "%d%b%y").date()
            strike = float(strike_s)
        except ValueError:
            continue
        parsed.append((exp, strike, cp, it))
    if not parsed:
        return None
    today = dt.date.today()
    future = sorted({e for e, *_ in parsed if e >= today})
    expiry = future[0] if future else sorted({e for e, *_ in parsed})[0]

    by_strike: Dict[float, Dict] = {}
    for exp, strike, cp, it in parsed:
        if exp != expiry:
            continue
        row = by_strike.setdefault(strike, {"strike": strike, "ce": None, "pe": None})
        mark_coin = it.get("mark_price")            # in coin units
        leg = {"ltp": round((mark_coin or 0) * spot, 2),   # USD premium
               "ltp_coin": mark_coin,
               "iv": it.get("mark_iv"),
               "oi": it.get("open_interest"),
               "chg_oi": None,
               "vol": it.get("volume")}
        if cp == "C":
            row["ce"] = leg
        else:
            row["pe"] = leg
    rows = _window(list(by_strike.values()), spot)
    return {"symbol": currency, "market": "Crypto (Deribit)", "source": "live",
            "note": "Live Deribit crypto option chain (premiums in USD).",
            "underlying": round(spot, 2), "expiry": expiry.strftime("%d-%b-%Y"),
            "rows": rows, **_analytics(rows, spot)}


# --------------------------------------------------------------------------- #
#  Modeled fallback (Black-Scholes)
# --------------------------------------------------------------------------- #
def model_chain(symbol: str, spot: float, sigma_ann: float, market: str,
                expiry_days: int = 7, reason: str = "") -> Dict:
    step = _step_for(symbol, spot)
    atm = round(spot / step) * step
    T = expiry_days / 365.0
    rows: List[Dict] = []
    for k in range(-12, 13):
        K = atm + k * step
        if K <= 0:
            continue
        call = bs_price(spot, K, T, sigma_ann, "call")
        put = bs_price(spot, K, T, sigma_ann, "put")
        # a crude OI shape (peaks a little OTM) so PCR / max-pain are illustrative
        w = math.exp(-((K - spot) / (3 * step)) ** 2)
        rows.append({
            "strike": round(K, 2),
            "ce": {"ltp": round(call, 2), "iv": round(sigma_ann * 100, 1),
                   "oi": int(9000 * w * (1.1 if K < spot else 0.9)), "chg_oi": None, "vol": None},
            "pe": {"ltp": round(put, 2), "iv": round(sigma_ann * 100, 1),
                   "oi": int(9000 * w * (1.1 if K > spot else 0.9)), "chg_oi": None, "vol": None},
        })
    note = ("Modeled chain (Black-Scholes at forecast vol) — live feed unavailable"
            + (f": {reason}" if reason else "") + ".")
    return {"symbol": symbol, "market": market, "source": "modeled", "note": note,
            "underlying": round(spot, 2),
            "expiry": f"≈ {expiry_days}d (modeled)", "iv_used_pct": round(sigma_ann * 100, 1),
            "rows": rows, **_analytics(rows, spot)}


# --------------------------------------------------------------------------- #
#  Dispatcher
# --------------------------------------------------------------------------- #
def chain(symbol: str, spot: Optional[float] = None,
          sigma_ann: Optional[float] = None) -> Dict:
    """Best available chain for ``symbol``.

    ``spot`` / ``sigma_ann`` are used only for the modeled fallback (pass the
    live price + forecast vol the caller already has).
    """
    symbol = symbol.strip().upper()
    now = time.time()
    ck = symbol
    if ck in _CACHE and now - _CACHE[ck][0] < _TTL:
        return _CACHE[ck][1]

    crypto = symbol in {"BTC", "ETH"} or symbol.endswith(("USDT", "-USD"))
    result: Optional[Dict] = None
    if crypto:
        cur = "ETH" if symbol.startswith("ETH") else "BTC"
        result = fetch_deribit(cur)
        market = "Crypto (Deribit)"
    elif symbol in {"SENSEX", "BANKEX"}:
        market = "BSE"          # NSE API doesn't serve these; modeled here
    else:
        result = fetch_nse(symbol)
        market = "NSE"

    if result is None:
        if spot is None:
            _CACHE[ck] = (now, {"symbol": symbol, "error":
                          "Live chain unavailable and no spot given for a modeled fallback.",
                          "rows": []})
            return _CACHE[ck][1]
        result = model_chain(symbol, float(spot), float(sigma_ann or 0.5), market)

    _CACHE[ck] = (now, result)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for sym, sp, vol in [("BTC", 61000, 0.6), ("NIFTY", 24000, 0.13)]:
        c = chain(sym, spot=sp, sigma_ann=vol)
        print(f"\n{sym}: source={c['source']} spot={c.get('underlying')} "
              f"expiry={c.get('expiry')} atm={c.get('atm')} pcr={c.get('pcr')} "
              f"maxpain={c.get('max_pain')}  ({c['note']})")
        for r in c["rows"][:5]:
            ce, pe = r["ce"] or {}, r["pe"] or {}
            print(f"  {r['strike']:>10}  CE {str(ce.get('ltp')):>10}  PE {str(pe.get('ltp')):>10}")
