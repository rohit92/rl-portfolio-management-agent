"""
tv_bridge.py — Bridge to TradingView Desktop via the tradingview-mcp CLI.

TradingView Desktop (Electron) exposes the Chrome DevTools Protocol when
launched with ``--remote-debugging-port=9222``. The cloned ``tradingview-mcp``
repo (../tradingview-mcp) drives it — same engine that backs the MCP server
Claude uses. This module shells out to its CLI so the *dashboard* can use
TradingView as a data source too:

  • TradingView carries licensed exchange data (incl. NSE/BSE) that our free
    feeds don't — quotes here come from whatever your TV plan streams.
  • ``quote("^NSEI")`` maps to ``NSE:NIFTY``, loads it on the desktop chart and
    returns the quote — so your TradingView follows the dashboard ("sync").

Honesty: quoting a symbol CHANGES the visible TradingView chart (that's how
the CDP bridge reads data). Real-time entitlements depend on your TradingView
login/plan — logged-out or free plans may stream delayed data for some
exchanges. This is a data/UX bridge, not an edge.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import requests

logger = logging.getLogger("tv_bridge")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "tradingview-mcp"
CLI = REPO / "src" / "cli" / "index.js"
NODE_CANDIDATES = ["/opt/homebrew/bin/node", "/usr/local/bin/node", "node"]
CDP_URL = "http://127.0.0.1:9222/json/version"

_CACHE: Dict[str, tuple] = {}
_TTL = 5.0


def _node() -> Optional[str]:
    for n in NODE_CANDIDATES:
        try:
            subprocess.run([n, "--version"], capture_output=True, timeout=5)
            return n
        except Exception:
            continue
    return None


# --- symbol mapping: our tickers -> TradingView symbols ---------------------- #
_TV_MAP = {
    "^NSEI": "NSE:NIFTY", "^NSEBANK": "NSE:BANKNIFTY", "^BSESN": "BSE:SENSEX",
    "^GSPC": "SP:SPX", "^NDX": "NASDAQ:NDX", "^VIX": "CBOE:VIX",
    "^INDIAVIX": "NSE:INDIAVIX",
    "NIFTY": "NSE:NIFTY", "BANKNIFTY": "NSE:BANKNIFTY", "FINNIFTY": "NSE:CNXFINANCE",
    "SENSEX": "BSE:SENSEX",
}


def to_tv_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if s in _TV_MAP:
        return _TV_MAP[s]
    if s.endswith(".NS"):
        return f"NSE:{s[:-3]}"
    if s.endswith(".BO"):
        return f"BSE:{s[:-3]}"
    if s.endswith("USDT"):
        return f"BINANCE:{s}"
    if s.endswith("-USD"):
        return f"BINANCE:{s.replace('-USD', 'USDT')}"
    return s  # pass through (already TV-style or US equity)


# --- core ops ---------------------------------------------------------------- #
def is_running() -> bool:
    """Fast probe: is TradingView Desktop up with the CDP port open?"""
    try:
        r = requests.get(CDP_URL, timeout=1.5)
        return "TradingView" in r.text
    except Exception:
        return False


def _cli(*args: str, timeout: int = 20) -> Dict:
    node = _node()
    if node is None:
        return {"success": False, "error": "node not installed"}
    if not CLI.exists():
        return {"success": False, "error": f"tradingview-mcp not found at {REPO}"}
    try:
        p = subprocess.run([node, str(CLI), *args], capture_output=True,
                           text=True, timeout=timeout)
        out = p.stdout.strip()
        return json.loads(out) if out.startswith("{") else \
            {"success": False, "error": (out or p.stderr.strip())[:300]}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "TradingView bridge timed out"}
    except Exception as exc:
        return {"success": False, "error": str(exc)[:300]}


def status() -> Dict:
    """Bridge status for the dashboard pill."""
    installed = Path("/Applications/TradingView.app").exists()
    running = is_running()
    out = {"installed": installed, "running": running, "connected": False,
           "chart_symbol": None, "repo": str(REPO)}
    if running:
        st = _cli("status", timeout=15)
        out["connected"] = bool(st.get("cdp_connected") or st.get("success"))
        out["chart_symbol"] = st.get("chart_symbol")
    return out


def launch() -> Dict:
    """Launch TradingView Desktop with the CDP port (no-op if running)."""
    if is_running():
        return {"ok": True, "note": "already running"}
    if not Path("/Applications/TradingView.app").exists():
        return {"ok": False, "error": "TradingView.app not installed"}
    try:
        subprocess.run(["open", "-a", "TradingView", "--args",
                        "--remote-debugging-port=9222"], timeout=15)
        return {"ok": True, "note": "launching — give it ~10s"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def quote(symbol: str, sync: bool = True) -> Dict:
    """Quote ``symbol`` through TradingView (maps to a TV symbol first).

    ``sync=True`` loads the symbol on the desktop chart (that's also how the
    quote is read). Cached a few seconds so UI polling doesn't thrash the app.
    """
    tv_sym = to_tv_symbol(symbol)
    now = time.time()
    if tv_sym in _CACHE and now - _CACHE[tv_sym][0] < _TTL:
        return _CACHE[tv_sym][1]
    if not is_running():
        return {"success": False, "error": "TradingView not running"}
    # Switch the chart first and CONFIRM it took (rapid back-to-back switches
    # can be swallowed while the previous chart is still loading), then quote.
    _cli("symbol", "--set", tv_sym, timeout=20)
    switched = False
    for i in range(10):
        time.sleep(1.0)
        cur = _cli("symbol", timeout=15)
        if str(cur.get("symbol", "")).upper() == tv_sym.upper():
            switched = True
            break
        if i == 4:  # one retry mid-way if the first set was swallowed
            _cli("symbol", "--set", tv_sym, timeout=20)
    if not switched:
        return {"success": False, "error": f"chart did not load {tv_sym} in time"}
    q: Dict = {}
    for _ in range(4):
        q = _cli("quote", timeout=20)
        if q.get("success") and q.get("symbol", "").upper() == tv_sym.upper():
            q["tv_symbol"] = tv_sym
            q["source"] = "tradingview"
            _CACHE[tv_sym] = (time.time(), q)
            return q
        time.sleep(1.0)
    q.setdefault("error", f"quote for {tv_sym} unavailable")
    q["success"] = False
    return q


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    st = status()
    print("status:", json.dumps(st, indent=2))
    if st["running"]:
        for sym in ("^NSEI", "RELIANCE.NS", "BTCUSDT"):
            q = quote(sym)
            print(f"{sym:>12} -> {q.get('tv_symbol')}: last={q.get('last') or q.get('close')} "
                  f"(success={q.get('success')})")
