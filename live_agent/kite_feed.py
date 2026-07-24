"""
kite_feed.py — Zerodha Kite Connect broker bridge (READ-ONLY, India desk).

Gives the platform *real* NSE/BSE/NFO/MCX data straight from the exchange via
your Zerodha account: live quotes, the real F&O and MCX instrument lists, and
genuine option chains (real strikes / LTP / OI) instead of the modeled fallback.

WHAT THIS IS AND ISN'T
  • Read-only. This module deliberately exposes NO order/trade functions. It
    fetches data. It cannot place, modify or cancel an order.
  • You own the auth. Kite Connect is a paid API (~₹2000/mo). You create an app
    at https://developers.kite.trade, put its api_key/api_secret in .env, then
    log in through Zerodha's own site once a day. This code never sees your
    Zerodha password — only the short-lived request_token you paste back.
  • The access token expires every morning (~6 AM IST). Re-login is a 20-second
    daily ritual, by design of Kite (not something we can automate away).

FLOW
  1. login_url()                     -> the Zerodha login URL (open in browser)
  2. (you log in; Zerodha redirects to your app's redirect URL carrying
     ?request_token=XXXX)
  3. complete_session("XXXX")        -> exchanges it (with api_secret) for an
                                        access_token, persisted to state/
  4. everything else just works until the token expires next morning.

Everything degrades gracefully: no kiteconnect installed, no keys, or no live
session all return a clear ``connected: False`` status rather than crashing, so
the dashboard keeps working on its free-data fallbacks.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("kite_feed")

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
SESSION_FILE = STATE / "kite_session.json"
INSTR_CACHE = STATE / "kite_instruments.json"
_INSTR_TTL = 12 * 3600          # instruments list refreshes twice a day
_INSTR_MEM: Dict[str, tuple] = {}

# Exchanges we care about on the India desk.
EXCHANGES = ("NSE", "BSE", "NFO", "BFO", "MCX", "CDS")


# --------------------------------------------------------------------------- #
#  Credentials + session
# --------------------------------------------------------------------------- #
def _load_env() -> None:
    """Load .env if python-dotenv is around (webapp already may have)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(HERE / ".env")
    except Exception:
        pass


def _api_key() -> Optional[str]:
    _load_env()
    return os.getenv("KITE_API_KEY") or None


def _api_secret() -> Optional[str]:
    _load_env()
    return os.getenv("KITE_API_SECRET") or None


def _read_session() -> Dict:
    if SESSION_FILE.exists():
        try:
            return json.loads(SESSION_FILE.read_text())
        except Exception:
            return {}
    return {}


def _write_session(data: Dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(json.dumps(data, indent=2))


def _kite(require_session: bool = True):
    """A configured KiteConnect client, or None (with the reason logged)."""
    try:
        from kiteconnect import KiteConnect
    except Exception:
        logger.warning("kiteconnect not installed (pip install kiteconnect)")
        return None
    key = _api_key()
    if not key:
        return None
    kc = KiteConnect(api_key=key)
    if require_session:
        sess = _read_session()
        tok = sess.get("access_token")
        if not tok:
            return None
        kc.set_access_token(tok)
    return kc


def status() -> Dict:
    """Is the broker bridge installed / keyed / logged-in / token-fresh?"""
    try:
        import kiteconnect  # noqa: F401
        installed = True
    except Exception:
        installed = False
    key = _api_key()
    sess = _read_session()
    connected = False
    profile = None
    reason = None
    if not installed:
        reason = "kiteconnect SDK not installed (pip install kiteconnect)."
    elif not key:
        reason = "No KITE_API_KEY in .env — create an app at developers.kite.trade."
    elif not sess.get("access_token"):
        reason = "Not logged in — open the login link and paste the request_token."
    else:
        # a cheap authenticated call confirms the token is still valid today
        try:
            kc = _kite()
            profile = kc.profile()
            connected = True
        except Exception as exc:
            reason = (f"Session expired or invalid ({str(exc)[:60]}). "
                      "Kite tokens die ~6 AM IST — log in again.")
    return {
        "installed": installed,
        "has_key": bool(key),
        "connected": connected,
        "user": (profile or {}).get("user_name") if profile else sess.get("user_name"),
        "user_id": (profile or {}).get("user_id") if profile else sess.get("user_id"),
        "login_time": sess.get("login_time"),
        "reason": reason,
        "note": ("Read-only market data. This bridge cannot place orders. "
                 "Access tokens expire every morning — a quick re-login is normal."),
    }


def login_url() -> Dict:
    """The Zerodha login URL to open in your browser."""
    kc = _kite(require_session=False)
    if kc is None:
        return {"error": "Set KITE_API_KEY in .env and install kiteconnect first."}
    return {"login_url": kc.login_url(),
            "note": ("Open this, log in to Zerodha, and you'll be redirected to your "
                     "app's redirect URL carrying ?request_token=... — paste that token "
                     "(or the whole URL) back here.")}


def complete_session(request_token: str) -> Dict:
    """Exchange a request_token (from the post-login redirect) for an access
    token and persist it. Accepts either the bare token or the full redirect URL."""
    rt = (request_token or "").strip()
    if "request_token=" in rt:  # user pasted the whole redirect URL — extract it
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(rt).query)
        rt = (q.get("request_token") or [""])[0]
    if not rt:
        return {"error": "No request_token found in what you pasted."}
    kc = _kite(require_session=False)
    secret = _api_secret()
    if kc is None or not secret:
        return {"error": "Need KITE_API_KEY and KITE_API_SECRET in .env."}
    try:
        data = kc.generate_session(rt, api_secret=secret)
    except Exception as exc:
        return {"error": f"Token exchange failed: {exc}. request_tokens are "
                         f"single-use and expire in minutes — get a fresh one."}
    sess = {
        "access_token": data["access_token"],
        "user_id": data.get("user_id"),
        "user_name": data.get("user_name"),
        "login_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_session(sess)
    logger.info("Kite session established for %s", sess.get("user_id"))
    return {"connected": True, "user": sess.get("user_name"),
            "user_id": sess.get("user_id"), "login_time": sess["login_time"]}


def logout() -> Dict:
    """Forget the local session (does not revoke on Zerodha's side)."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()
    return {"ok": True}


# --------------------------------------------------------------------------- #
#  Instruments (the master list — includes every F&O + MCX contract)
# --------------------------------------------------------------------------- #
def _instruments(exchange: str) -> List[Dict]:
    """Full instrument dump for one exchange, cached to disk for 12h."""
    exchange = exchange.upper()
    now = time.time()
    if exchange in _INSTR_MEM and now - _INSTR_MEM[exchange][0] < _INSTR_TTL:
        return _INSTR_MEM[exchange][1]
    # disk cache
    disk = {}
    if INSTR_CACHE.exists():
        try:
            disk = json.loads(INSTR_CACHE.read_text())
        except Exception:
            disk = {}
    ent = disk.get(exchange)
    if ent and now - ent.get("ts", 0) < _INSTR_TTL:
        _INSTR_MEM[exchange] = (now, ent["rows"])
        return ent["rows"]
    kc = _kite()
    if kc is None:
        return []
    try:
        rows = kc.instruments(exchange)
    except Exception as exc:
        logger.warning("instruments(%s) failed: %s", exchange, exc)
        return []
    # normalise expiry/date objects to strings for JSON
    for r in rows:
        exp = r.get("expiry")
        r["expiry"] = exp.isoformat() if hasattr(exp, "isoformat") else (exp or "")
    disk[exchange] = {"ts": now, "rows": rows}
    try:
        INSTR_CACHE.write_text(json.dumps(disk))
    except Exception:
        pass
    _INSTR_MEM[exchange] = (now, rows)
    return rows


def quote(symbols: List[str]) -> Dict:
    """Live quotes for ``EXCHANGE:TRADINGSYMBOL`` keys (e.g. 'NSE:RELIANCE',
    'MCX:GOLDM26FEBFUT'). Returns {} if not connected."""
    kc = _kite()
    if kc is None or not symbols:
        return {}
    try:
        return kc.quote(symbols)
    except Exception as exc:
        logger.warning("quote failed: %s", exc)
        return {}


# --------------------------------------------------------------------------- #
#  F&O + MCX boards (real contracts, real marks)
# --------------------------------------------------------------------------- #
def _nearest_expiries(rows: List[Dict], name: str, seg: str) -> List[str]:
    exps = sorted({r["expiry"] for r in rows
                   if r.get("name") == name and r.get("segment") == seg and r.get("expiry")})
    today = time.strftime("%Y-%m-%d")
    return [e for e in exps if e >= today] or exps


def fno_board(limit: int = 25) -> Dict:
    """Live NSE index & stock FUTURES board (nearest expiry) with real marks."""
    kc = _kite()
    if kc is None:
        return {"connected": False, "rows": [],
                "note": status().get("reason", "Not connected.")}
    nfo = _instruments("NFO")
    futs = [r for r in nfo if r.get("segment") == "NFO-FUT"]
    # nearest expiry only, indices + most liquid singles first
    if not futs:
        return {"connected": True, "rows": [], "note": "No NFO futures returned."}
    nearest = min(r["expiry"] for r in futs if r.get("expiry"))
    near = [r for r in futs if r.get("expiry") == nearest]
    # indices first
    idx = [r for r in near if r.get("name") in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")]
    rest = [r for r in near if r not in idx]
    picks = (idx + rest)[:limit]
    keys = [f"NFO:{r['tradingsymbol']}" for r in picks]
    q = quote(keys)
    rows = []
    for r in picks:
        k = f"NFO:{r['tradingsymbol']}"
        d = q.get(k, {})
        ltp = d.get("last_price")
        ohlc = d.get("ohlc", {}) or {}
        prev = ohlc.get("close")
        rows.append({
            "symbol": r["tradingsymbol"], "name": r.get("name"),
            "expiry": r.get("expiry"), "lot_size": r.get("lot_size"),
            "ltp": ltp, "prev_close": prev,
            "chg_pct": round((ltp / prev - 1) * 100, 2) if ltp and prev else None,
            "oi": d.get("oi"), "volume": d.get("volume"),
        })
    return {"connected": True, "expiry": nearest, "rows": rows,
            "note": "Live NFO futures — nearest expiry, real marks from your account."}


def mcx_board(limit: int = 25) -> Dict:
    """Live MCX commodity FUTURES board with real marks (gold, silver, crude,
    natural gas, copper …) — the data free feeds do not have."""
    kc = _kite()
    if kc is None:
        return {"connected": False, "rows": [],
                "note": status().get("reason", "Not connected.")}
    mcx = _instruments("MCX")
    futs = [r for r in mcx if r.get("segment") == "MCX-FUT"]
    if not futs:
        return {"connected": True, "rows": [], "note": "No MCX futures returned."}
    # nearest expiry per underlying name
    by_name: Dict[str, Dict] = {}
    for r in sorted(futs, key=lambda x: x.get("expiry") or "9999"):
        nm = r.get("name")
        if nm and nm not in by_name:
            by_name[nm] = r
    picks = list(by_name.values())[:limit]
    keys = [f"MCX:{r['tradingsymbol']}" for r in picks]
    q = quote(keys)
    rows = []
    for r in picks:
        k = f"MCX:{r['tradingsymbol']}"
        d = q.get(k, {})
        ltp = d.get("last_price")
        ohlc = d.get("ohlc", {}) or {}
        prev = ohlc.get("close")
        rows.append({
            "symbol": r["tradingsymbol"], "name": r.get("name"),
            "expiry": r.get("expiry"), "lot_size": r.get("lot_size"),
            "ltp": ltp, "prev_close": prev,
            "chg_pct": round((ltp / prev - 1) * 100, 2) if ltp and prev else None,
            "day_high": ohlc.get("high"), "day_low": ohlc.get("low"),
            "volume": d.get("volume"), "oi": d.get("oi"),
        })
    return {"connected": True, "rows": rows,
            "note": ("Live MCX futures from your Zerodha feed. Commodity segment must be "
                     "activated on your account. High-leverage, high-gap-risk instruments.")}


def option_chain(name: str, expiry: Optional[str] = None,
                 strikes_around: int = 15) -> Dict:
    """Real option chain for an index/stock: actual strikes, LTP, OI, volume,
    pulled live from NFO (or BFO for SENSEX/BANKEX)."""
    kc = _kite()
    if kc is None:
        return {"connected": False, "rows": [],
                "note": status().get("reason", "Not connected.")}
    name = name.strip().upper()
    exch = "BFO" if name in ("SENSEX", "BANKEX") else "NFO"
    rows_all = _instruments(exch)
    opts = [r for r in rows_all
            if r.get("name") == name and r.get("segment", "").endswith("-OPT")]
    if not opts:
        return {"connected": True, "rows": [],
                "note": f"No option instruments found for {name} on {exch}."}
    exps = sorted({r["expiry"] for r in opts if r.get("expiry")})
    today = time.strftime("%Y-%m-%d")
    future_exps = [e for e in exps if e >= today] or exps
    use_exp = expiry if expiry in exps else future_exps[0]
    chain_instr = [r for r in opts if r.get("expiry") == use_exp]

    # find ATM via the underlying's LTP
    und_key = _underlying_key(name)
    spot = None
    if und_key:
        uq = quote([und_key])
        spot = (uq.get(und_key, {}) or {}).get("last_price")
    strikes = sorted({r["strike"] for r in chain_instr})
    if spot and strikes:
        atm = min(strikes, key=lambda s: abs(s - spot))
        i = strikes.index(atm)
        lo, hi = max(0, i - strikes_around), i + strikes_around + 1
        keep = set(strikes[lo:hi])
        chain_instr = [r for r in chain_instr if r["strike"] in keep]

    keys = [f"{exch}:{r['tradingsymbol']}" for r in chain_instr]
    q = quote(keys) if keys else {}
    by_strike: Dict[float, Dict] = {}
    for r in chain_instr:
        d = q.get(f"{exch}:{r['tradingsymbol']}", {})
        side = "CE" if r.get("instrument_type") == "CE" else "PE"
        rec = by_strike.setdefault(r["strike"], {"strike": r["strike"]})
        rec[side] = {"ltp": d.get("last_price"), "oi": d.get("oi"),
                     "volume": d.get("volume"), "symbol": r["tradingsymbol"]}
    rows = [by_strike[s] for s in sorted(by_strike)]
    ce_oi = sum((r.get("CE") or {}).get("oi") or 0 for r in rows)
    pe_oi = sum((r.get("PE") or {}).get("oi") or 0 for r in rows)
    return {
        "connected": True, "source": "live", "symbol": name, "exchange": exch,
        "expiry": use_exp, "expiries": future_exps[:8], "underlying": spot,
        "pcr": round(pe_oi / ce_oi, 2) if ce_oi else None,
        "rows": rows,
        "note": "Live chain (real strikes / LTP / OI) from your Zerodha account.",
    }


def _underlying_key(name: str) -> Optional[str]:
    """Map an F&O underlying name to its spot quote key."""
    idx = {"NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK",
           "FINNIFTY": "NSE:NIFTY FIN SERVICE", "MIDCPNIFTY": "NSE:NIFTY MIDCAP SELECT",
           "SENSEX": "BSE:SENSEX", "BANKEX": "BSE:BANKEX"}
    if name in idx:
        return idx[name]
    return f"NSE:{name}"          # single stock


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = status()
    print("status:", json.dumps({k: s[k] for k in ("installed", "has_key",
          "connected", "reason")}, indent=1))
    if s["connected"]:
        print("F&O board:", len(fno_board().get("rows", [])), "rows")
        print("MCX board:", len(mcx_board().get("rows", [])), "rows")
        oc = option_chain("NIFTY")
        print("NIFTY chain:", len(oc.get("rows", [])), "strikes, PCR", oc.get("pcr"))
    else:
        print("Not connected — the bridge is wired and waiting for your login.")
