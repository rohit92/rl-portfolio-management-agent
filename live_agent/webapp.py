"""
webapp.py — Local web dashboard for the trading agent.

Runs entirely on your Mac (no cloud, no Google Drive needed — the data is small
and re-fetched from yfinance on demand). Wraps the existing modules
(forecast / strategies / data_feed / news) behind a tiny Flask API and serves a
search-driven dashboard:

    type a ticker  ->  next-day probabilistic forecast + signal scores
                       + volatility regime + headline context

    source ../venv/bin/activate
    python webapp.py            # then open http://127.0.0.1:5000

Honesty carries over from the CLI: the page states plainly that this is research
context, not a buy list, and that direction is a coin-flip.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yfinance as yf
from flask import Flask, jsonify, render_template, request

import agents as agents_mod
import alerts as alerts_mod
import candles as candles_mod
import derivatives as derivatives_mod
import futures_feed as futures_feed_mod
import journal as journal_mod
import option_chain as option_chain_mod
import tv_bridge as tv_bridge_mod
import markets
import options as options_mod
import live_signals as live_signals_mod
import live_trader as live_trader_mod
import kite_feed as kite_mod
import mf as mf_mod
import notify as notify_mod
import regime as regime_mod
import risk_pro as risk_pro_mod
import dual_momentum as dm_mod
import mtf as mtf_mod
import sentiment as sentiment_mod
import signals as signals_mod
import signal_perf as signal_perf_mod
import news as news_mod
import universe as universe_mod
from backtest import backtest_symbol
from broker import Order, SimBroker
from data_feed import fetch
from forecast import calibrate, forecast_tomorrow
from levels import trigger_levels
from screen import _vol_regime
from strategies import Ensemble
from trader import apply_learned_weights, load_config

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("webapp")

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True   # edits to index.html apply on refresh


@app.after_request
def _cors(resp):
    """Allow the Chrome extension (and phone PWA) to call /api/* — local,
    paper-money data only, so an open CORS policy is acceptable here."""
    if request.path.startswith("/api/"):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


def _lan_ip() -> str:
    """This Mac's LAN IP (for phone access on the same Wi-Fi)."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# ---- Config (load once; adopt learned weights + active market) --------------
_CFG = load_config(HERE / "config.yaml")
_CFG = markets.resolve(_CFG)
apply_learned_weights(_CFG)

# Adopt a trained policy (committee threshold) if one cleared the OOS gate.
_POL = HERE / "state" / "learned_policy.json"
if _POL.exists():
    try:
        import json as _json
        _pol = _json.loads(_POL.read_text())
        if _pol.get("adopted") and _pol.get("committee_threshold"):
            _CFG.setdefault("committee", {})["threshold"] = int(_pol["committee_threshold"])
            logger.info("Adopted trained committee threshold %s (OOS Sharpe %s)",
                        _pol["committee_threshold"], _pol.get("median_oos_sharpe"))
    except Exception:
        pass

_FC = _CFG["forecast"]
_ENSEMBLE = Ensemble(_CFG)

# Paper-money account (simulated broker, persisted to state/sim_account.json).
_STATE = HERE / "state"
_PAPER = SimBroker(
    state_path=_STATE / "sim_account.json",
    initial_cash=float(_CFG.get("initial_cash", 100_000)),
    transaction_cost=float(_CFG["risk"]["transaction_cost"]),
)

# ---- Tiny TTL cache so re-searching a symbol is instant ---------------------
_CACHE: Dict[str, tuple] = {}
_TTL = 600  # seconds


def _f(x) -> float:
    """numpy/None-safe float for JSON."""
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _load_symbol(symbol: str):
    """Fetch + cache a symbol's engineered data (one network call)."""
    symbol = symbol.strip().upper()
    now = time.time()
    if symbol in _CACHE and now - _CACHE[symbol][0] < _TTL:
        return _CACHE[symbol][1]
    data = fetch([symbol], _CFG, lookback_days=2500)
    sd = data.get(symbol)
    _CACHE[symbol] = (now, sd)
    return sd


def _insight(symbol: str) -> dict:
    sd = _load_symbol(symbol)
    if sd is None:
        return {"error": f"No data for '{symbol}'. Check the ticker "
                         f"(use e.g. RELIANCE.NS for NSE, BTC-USD for crypto)."}

    asc = _ENSEMBLE.score_one(sd)
    f = forecast_tomorrow(sd.frame["close"], _FC)
    heads = news_mod.get_headlines(symbol, limit=5)
    thr = float(_CFG["risk"]["entry_threshold"])

    return {
        "symbol": symbol.upper(),
        "last": _f(f["level"]),
        "bars": int(len(sd.frame)),
        "span": [str(sd.frame.index[0].date()), str(sd.frame.index[-1].date())],
        "forecast": {
            "sigma_ann": _f(f["sigma_next_ann"]),
            "exp_move_pts": _f(f["exp_abs_move_pts"]),
            "exp_move_pct": _f(f["exp_abs_move_pts"] / f["level"]),
            "p_up": _f(f["p_up"]),
            "p_up1": _f(f["p_up_gt1pct"]),
            "p_down1": _f(f["p_down_gt1pct"]),
            "levels": {"p5": _f(f["level"] + f["p05"]),
                       "p50": _f(f["level"] + f["p50"]),
                       "p95": _f(f["level"] + f["p95"])},
            "moves": {"p5": _f(f["p05"]), "p50": _f(f["p50"]), "p95": _f(f["p95"])},
        },
        "signal": {
            "composite": _f(asc.composite),
            "components": {k: _f(v) for k, v in asc.components.items()},
            "threshold": thr,
            "is_candidate": bool(asc.composite >= thr),
        },
        "regime": _vol_regime(sd),
        "levels": trigger_levels(sd, f["exp_abs_move_pts"] / f["level"]),
        "options": _options_idea(symbol, f["level"], f["sigma_next_ann"], asc.composite),
        "sentiment": sentiment_mod.market_sentiment(symbol),
        "history": [{"t": str(i.date()), "c": round(float(v), 4)}
                    for i, v in sd.frame["close"].tail(140).items()],
        "news": [{"title": h["title"], "publisher": h["publisher"],
                  "lean": _f(h["lean"])} for h in heads],
    }


# ---- Market overview (cheap lite stats, no full forecast) ------------------ #
OVERVIEW_SYMS = ["^NSEI", "^BSESN", "^NSEBANK", "^GSPC", "^NDX",
                 "BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _overview_row(sym: str) -> dict | None:
    sd = _load_symbol(sym)
    if sd is None:
        return None
    close = sd.frame["close"]
    last, prev = float(close.iloc[-1]), float(close.iloc[-2])
    asc = _ENSEMBLE.score_one(sd)
    vol = float(sd.frame["vol"].iloc[-1]) * (252 ** 0.5)
    return {
        "symbol": sym, "last": round(last, 2), "chg": _f(last / prev - 1.0),
        "composite": _f(asc.composite), "vol": _f(vol),
        "is_candidate": bool(asc.composite >= float(_CFG["risk"]["entry_threshold"])),
        "regime": _vol_regime(sd),
    }


# ---- Options lens: forecast (realised) vol vs implied vol (VIX) ------------- #
_VIX_MAP = {"^NSEI": "^INDIAVIX", "^BSESN": "^INDIAVIX", "^NSEBANK": "^INDIAVIX",
            "^GSPC": "^VIX", "^NDX": "^VIX"}
_VIX_CACHE: Dict[str, tuple] = {}


def _vix_value(vix_sym: str):
    """Latest VIX value, cached 10 min."""
    now = time.time()
    if vix_sym in _VIX_CACHE and now - _VIX_CACHE[vix_sym][0] < 600:
        return _VIX_CACHE[vix_sym][1]
    try:
        v = float(yf.Ticker(vix_sym).history(period="5d")["Close"].iloc[-1])
    except Exception:
        v = None
    _VIX_CACHE[vix_sym] = (now, v)
    return v


def _options_idea(sym: str, spot: float, sigma_ann: float, composite: float):
    """An options *strategy* idea from forecast-vs-implied vol + direction,
    with Black-Scholes ATM premiums. Skipped for crypto.
    """
    try:
        import binance_feed
        if binance_feed.is_crypto(sym):
            return None
    except Exception:
        pass
    vix_sym = _VIX_MAP.get(sym, "^INDIAVIX" if sym.endswith(".NS") else "^VIX")
    implied = _vix_value(vix_sym)
    if implied is None:
        return None
    forecast = float(sigma_ann) * 100.0
    bullish = composite >= 0
    direction = "bullish" if bullish else "bearish"
    if forecast > implied * 1.10:
        stance = "vol CHEAP"
        strat = "Buy a CALL (long premium)" if bullish else "Buy a PUT (long premium)"
        note = "Forecast vol > implied: options look underpriced — long premium in the trend direction."
    elif forecast < implied * 0.90:
        stance = "vol RICH"
        strat = ("Sell a bull-put spread / cash-secured put" if bullish
                 else "Sell a bear-call spread")
        note = "Implied vol > forecast: premium looks rich — a defined-risk credit spread."
    else:
        stance = "vol FAIR"
        strat = "Debit call spread" if bullish else "Debit put spread"
        note = "Vol roughly fair — a defined-risk debit spread in the trend direction, or skip."
    return {"vix_symbol": vix_sym, "implied": round(implied, 2),
            "forecast": round(forecast, 2), "stance": stance,
            "direction": direction, "strategy": strat, "note": note,
            "quote": options_mod.quote(spot, float(sigma_ann), days=30)}


@app.route("/")
def index():
    """THE dashboard — single unified app. Live command center is the home
    section; all research tools live in the same sidebar."""
    return render_template("index.html")


@app.route("/pro")
@app.route("/live")
@app.route("/simple")
def legacy_routes():
    """Old entry points all fold into the single dashboard."""
    from flask import redirect
    return redirect("/", code=302)


@app.route("/api/appinfo")
def api_appinfo():
    """App metadata: the LAN URL for phone access (same Wi-Fi only)."""
    return jsonify({"lan_url": f"http://{_lan_ip()}:5000", "version": "1.0"})


_STOP_TOKENS = {"ltd", "limited", "the", "co", "company", "india", "of", "and"}


@app.route("/api/resolve")
def api_resolve():
    """Resolve a free-form name/slug (e.g. Groww's 'reliance-industries-ltd' or
    'tata consultancy services') to a tradable symbol. Crypto resolves too."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "need q"}), 400
    u = universe_mod.load()
    binance = set(u.get("binance", []))
    nse = set(u.get("nse", []))
    qU = q.upper().replace("-", "").replace("_", "").replace(" ", "")

    # 1) direct crypto pair / base asset
    if qU in binance:
        return jsonify({"symbol": qU, "kind": "crypto"})
    for suffix in ("INR", "USD", "USDT"):
        if qU.endswith(suffix) and f"{qU[:-len(suffix)]}USDT" in binance:
            return jsonify({"symbol": f"{qU[:-len(suffix)]}USDT", "kind": "crypto"})
    if f"{qU}USDT" in binance:
        return jsonify({"symbol": f"{qU}USDT", "kind": "crypto"})

    # 2) direct NSE symbol
    if f"{qU}.NS" in nse:
        return jsonify({"symbol": f"{qU}.NS", "kind": "stock"})

    # 3) fuzzy company-name match (for Groww slugs)
    tokens = [t for t in q.lower().replace("-", " ").replace("_", " ").split()
              if t and t not in _STOP_TOKENS]
    if tokens:
        best, best_score = None, 0.0
        for sym, name in (u.get("nse_names") or {}).items():
            nt = set(name.lower().replace(".", " ").split())
            score = sum(1 for t in tokens if t in nt) / len(tokens)
            if score > best_score or (score == best_score and best and
                                      len(name) < len(u["nse_names"].get(best, ""))):
                best, best_score = sym, score
        if best and best_score >= 0.6:
            return jsonify({"symbol": best, "kind": "stock",
                            "name": u["nse_names"][best],
                            "confidence": round(best_score, 2)})
    return jsonify({"error": f"could not resolve '{q}'"}), 200


@app.route("/api/universe")
def api_universe():
    """Full tradable universe (for search autocomplete + counts)."""
    try:
        u = universe_mod.load()
    except Exception as exc:
        return jsonify({"error": str(exc), "symbols": [], "counts": {}}), 200
    return jsonify({
        "counts": u.get("counts", {}),
        "updated": u.get("updated"),
        "symbols": u.get("nse", []) + u.get("binance", []),
    })


@app.route("/api/scan_results")
def api_scan_results():
    """Cached batch-scan ranking (per segment: nse | binance)."""
    import json
    seg = request.args.get("segment", "binance")
    path = _STATE / f"scan_results_{seg}.json"
    if not path.exists():
        path = _STATE / "scan_results.json"
    if not path.exists():
        return jsonify({"error": f"No {seg} scan yet — run: "
                        f"python scan.py --segment {seg}", "rows": []})
    return jsonify(json.loads(path.read_text()))


@app.route("/api/backtest")
def api_backtest():
    """Backtest the model on one symbol at a chosen timeframe."""
    symbol = request.args.get("symbol", "").strip()
    interval = request.args.get("interval", "1d")
    if not symbol:
        return jsonify({"error": "Provide ?symbol=TICKER"}), 400
    try:
        return jsonify(backtest_symbol(symbol, interval, _CFG))
    except Exception as exc:
        logger.exception("backtest failed")
        return jsonify({"error": f"Backtest failed: {exc}"}), 500


# --------------------------- Paper trading ---------------------------------- #

@app.route("/api/paper/portfolio")
def api_paper_portfolio():
    positions = _PAPER.get_positions()
    held = []
    prices = {}
    for sym, qty in positions.items():
        sd = _load_symbol(sym)
        px = sd.last_close if sd else 0.0
        prices[sym] = px
        held.append({"symbol": sym, "qty": round(qty, 4),
                     "price": round(px, 2), "value": round(qty * px, 2)})
    held.sort(key=lambda h: h["value"], reverse=True)
    return jsonify({
        "cash": round(_PAPER.get_cash(), 2),
        "equity": round(_PAPER.get_equity(prices), 2),
        "positions": held,
    })


@app.route("/api/paper/buy")
def api_paper_buy():
    symbol = request.args.get("symbol", "").strip().upper()
    amount = float(request.args.get("amount", 1000))
    sd = _load_symbol(symbol)
    if sd is None:
        return jsonify({"error": f"No price for {symbol}"}), 400
    px = sd.last_close
    qty = amount / px
    if qty <= 0 or amount > _PAPER.get_cash():
        return jsonify({"error": "Insufficient paper cash."}), 400
    _PAPER.execute([Order(symbol=symbol, side="buy", qty=qty)], {symbol: px})
    return jsonify({"ok": True, "filled": {"symbol": symbol, "qty": round(qty, 4),
                    "price": round(px, 2)}})


@app.route("/api/paper/sell")
def api_paper_sell():
    symbol = request.args.get("symbol", "").strip().upper()
    held = _PAPER.get_positions().get(symbol, 0.0)
    if held <= 0:
        return jsonify({"error": f"No {symbol} position to sell."}), 400
    sd = _load_symbol(symbol)
    px = sd.last_close if sd else 0.0
    _PAPER.execute([Order(symbol=symbol, side="sell", qty=held)], {symbol: px})
    return jsonify({"ok": True, "sold": {"symbol": symbol, "qty": round(held, 4),
                    "price": round(px, 2)}})


# ------------------------ Overview / Options / Watchlist -------------------- #

def _desks() -> Dict:
    """Market-desk profiles from config (india primary; crypto/us secondary)."""
    d = _CFG.get("desks") or {}
    if not d:  # fallback if config lacks the desks section
        d = {"india": {"label": "🇮🇳 India", "primary": True,
                       "default_symbol": "RELIANCE.NS",
                       "pulse": OVERVIEW_SYMS, "chips": [], "universe_segment": "nse",
                       "option_chains": ["NIFTY"], "live_market": "nse_intraday",
                       "instruments": []}}
    return d


@app.route("/api/desks")
def api_desks():
    """The market-separation layer: each desk's profile + which is primary."""
    d = _desks()
    primary = next((k for k, v in d.items() if v.get("primary")), "india")
    return jsonify({"desks": d, "primary": primary})


# --------------------- India desk: MF / commodities / F&O -------------------- #

@app.route("/api/mf/search")
def api_mf_search():
    """Search Indian mutual fund schemes (free AMFI data via mfapi.in)."""
    return jsonify({"results": mf_mod.search(request.args.get("q", ""))})


@app.route("/api/mf/nav")
def api_mf_nav():
    """One scheme's honest scorecard: NAV, trailing returns, CAGR, vol, max DD."""
    code = request.args.get("code", "").strip()
    if not code.isdigit():
        return jsonify({"error": "code (numeric scheme code) required"}), 200
    try:
        return jsonify(mf_mod.analyze(code))
    except Exception as exc:
        logger.exception("mf nav failed")
        return jsonify({"error": str(exc)}), 200


_COMMODITY_ETFS = [
    ("GOLDBEES.NS", "Nippon Gold BeES"),
    ("SETFGOLD.NS", "SBI Gold ETF"),
    ("KOTAKGOLD.NS", "Kotak Gold ETF"),
    ("HDFCGOLD.NS", "HDFC Gold ETF"),
    ("SILVERBEES.NS", "Nippon Silver BeES"),
    ("SILVERIETF.NS", "ICICI Pru Silver ETF"),
]


@app.route("/api/commodities")
def api_commodities():
    """India commodity board. If Kite is connected → LIVE MCX futures (gold,
    silver, crude, natural gas…). Otherwise → NSE-listed gold/silver ETFs, the
    legal free-data fallback."""
    # Prefer real MCX futures when the broker bridge is live.
    try:
        if kite_mod.status().get("connected"):
            board = kite_mod.mcx_board()
            if board.get("rows"):
                return jsonify({"source": "kite_mcx", **board})
    except Exception:
        logger.warning("kite mcx_board failed; falling back to ETFs")
    rows = []
    for sym, name in _COMMODITY_ETFS:
        try:
            r = _overview_row(sym)
            if r:
                r["name"] = name
                rows.append(r)
        except Exception:
            continue
    return jsonify({"source": "nse_etf", "rows": rows,
                    "note": ("NSE-listed commodity ETFs — tradable in any demat account, "
                             "full daily data. Connect Zerodha Kite (Broker card) for LIVE "
                             "MCX futures: gold mini, crude, natural gas, silver.")})


@app.route("/api/fno")
def api_fno():
    """NSE F&O + MCX reference: instrument types, indicative leverage/margins."""
    try:
        nse = derivatives_mod.load().get("nse_derivatives", [])
    except Exception:
        nse = []
    mcx = [
        {"instrument": "MCX Gold Mini (GOLDM)", "type": "Commodity future",
         "approx_leverage": "~8–10x", "note": "Margin ≈ 10–12%; needs broker "
         "commodity segment activation. Data via broker API only."},
        {"instrument": "MCX Crude Oil", "type": "Commodity future",
         "approx_leverage": "~6–8x", "note": "High overnight gap risk (global "
         "crude moves while MCX is shut)."},
        {"instrument": "MCX Natural Gas", "type": "Commodity future",
         "approx_leverage": "~5–7x", "note": "The most violent liquid contract "
         "on MCX — beginner accounts die here."},
        {"instrument": "MCX Silver Mini", "type": "Commodity future",
         "approx_leverage": "~7–9x", "note": "Silver vol ≈ 1.5–2× gold vol."},
    ]
    # Live NFO futures board (real marks) if the broker bridge is connected.
    live = None
    try:
        if kite_mod.status().get("connected"):
            b = kite_mod.fno_board()
            if b.get("rows"):
                live = {"expiry": b.get("expiry"), "rows": b["rows"]}
    except Exception:
        logger.warning("kite fno_board failed")
    return jsonify({"nse": nse, "mcx": mcx, "live": live,
                    "live_connected": bool(live),
                    "note": ("Indicative figures — real SPAN+exposure margins come from "
                             "your broker and change with volatility. Options BUYING is "
                             "defined-risk (premium); options SELLING and futures are "
                             "margin instruments where losses exceed the margin."
                             + (" LIVE NFO futures marks shown below are from your Zerodha "
                                "account." if live else ""))})


# --------------------- Zerodha Kite broker bridge (read-only) ---------------- #

@app.route("/api/kite/status")
def api_kite_status():
    """Broker bridge state: installed / keyed / logged-in / token-fresh."""
    return jsonify(kite_mod.status())


@app.route("/api/kite/login_url")
def api_kite_login_url():
    """The Zerodha login URL to open in your browser."""
    return jsonify(kite_mod.login_url())


@app.route("/api/kite/connect")
def api_kite_connect():
    """Complete login: exchange the pasted request_token for an access token.
    (GET with ?request_token=… ; accepts the full redirect URL too.)"""
    rt = request.args.get("request_token", "")
    if not rt:
        return jsonify({"error": "request_token required"}), 200
    return jsonify(kite_mod.complete_session(rt))


@app.route("/api/kite/logout")
def api_kite_logout():
    """Forget the local session (does not revoke on Zerodha's side)."""
    return jsonify(kite_mod.logout())


@app.route("/api/kite/fno")
def api_kite_fno():
    """Live NFO futures board (real marks). Empty if not connected."""
    return jsonify(kite_mod.fno_board())


@app.route("/api/kite/mcx")
def api_kite_mcx():
    """Live MCX commodity futures board. Empty if not connected."""
    return jsonify(kite_mod.mcx_board())


@app.route("/api/kite/option_chain")
def api_kite_option_chain():
    """Real option chain (strikes / LTP / OI) for an index or stock via Kite."""
    name = request.args.get("symbol", "NIFTY")
    expiry = request.args.get("expiry") or None
    return jsonify(kite_mod.option_chain(name, expiry=expiry))


@app.route("/api/overview")
def api_overview():
    """Market pulse, scoped to one desk (?desk=india|crypto|us). Default: primary."""
    d = _desks()
    desk = request.args.get("desk", "").strip().lower()
    if desk not in d:
        desk = next((k for k, v in d.items() if v.get("primary")), None)
    syms = (d.get(desk) or {}).get("pulse") or OVERVIEW_SYMS
    rows = [r for r in (_overview_row(s) for s in syms) if r]
    return jsonify({"rows": rows, "desk": desk})


@app.route("/api/options_lens")
def api_options_lens():
    """Compare the model's forecast (realised) vol to implied vol (VIX)."""
    sym = request.args.get("symbol", "^NSEI").strip()
    vix_sym = _VIX_MAP.get(sym, "^INDIAVIX")
    try:
        implied = float(yf.Ticker(vix_sym).history(period="5d")["Close"].iloc[-1])
    except Exception as exc:
        return jsonify({"error": f"Could not fetch {vix_sym}: {exc}"}), 200
    sd = _load_symbol(sym)
    if sd is None:
        return jsonify({"error": f"No data for {sym}"}), 200
    f = forecast_tomorrow(sd.frame["close"], _FC)
    forecast_pct = float(f["sigma_next_ann"]) * 100.0
    ratio = implied / forecast_pct if forecast_pct else None
    if ratio is None:
        verdict, lean = "n/a", "·"
    elif ratio > 1.10:
        verdict, lean = ("Implied vol is RICH vs forecast — options relatively "
                         "expensive; statistically favours SELLING premium."), "rich"
    elif ratio < 0.90:
        verdict, lean = ("Implied vol is CHEAP vs forecast — options relatively "
                         "underpriced; statistically favours BUYING premium."), "cheap"
    else:
        verdict, lean = "Implied ≈ forecast — options roughly fairly priced.", "fair"
    return jsonify({
        "symbol": sym, "vix_symbol": vix_sym,
        "implied": round(implied, 2), "forecast": round(forecast_pct, 2),
        "ratio": _f(ratio), "verdict": verdict, "lean": lean,
    })


_WATCHLIST = _STATE / "watchlist.json"


def _load_watchlist() -> list:
    import json
    if _WATCHLIST.exists():
        return json.loads(_WATCHLIST.read_text())
    return ["^NSEI", "RELIANCE.NS", "BTCUSDT", "AAPL"]


@app.route("/api/watchlist")
def api_watchlist():
    import json
    wl = _load_watchlist()
    add = request.args.get("add", "").strip().upper()
    rem = request.args.get("remove", "").strip().upper()
    if add and add not in wl:
        wl.append(add)
    if rem in wl:
        wl.remove(rem)
    if add or rem:
        _WATCHLIST.write_text(json.dumps(wl))
    return jsonify({"symbols": wl})


# ------------------------------- Alerts ------------------------------------- #

@app.route("/api/alerts")
def api_alerts():
    return jsonify({"alerts": alerts_mod.list_alerts()})


@app.route("/api/alerts/add")
def api_alerts_add():
    symbol = request.args.get("symbol", "").strip()
    direction = request.args.get("direction", "above")
    level = request.args.get("level", "")
    if not symbol or not level:
        return jsonify({"error": "need symbol & level"}), 400
    try:
        item = alerts_mod.add_alert(symbol, direction, float(level),
                                    request.args.get("note", ""))
    except ValueError:
        return jsonify({"error": "bad level"}), 400
    return jsonify({"ok": True, "alert": item})


@app.route("/api/alerts/remove")
def api_alerts_remove():
    alerts_mod.remove_alert(request.args.get("id", ""))
    return jsonify({"ok": True})


@app.route("/api/alerts/check")
def api_alerts_check():
    fired = alerts_mod.check_alerts(do_notify=True)
    return jsonify({"fired": fired, "alerts": alerts_mod.list_alerts()})


# ----------------------------- Committee ------------------------------------ #

_PPY = {("1d", False): 252, ("1d", True): 365, ("1wk", False): 52,
        ("1wk", True): 52, ("1h", False): 1638, ("1h", True): 8760}


@app.route("/api/committee")
def api_committee():
    sym = request.args.get("symbol", "").strip()
    thr = int(request.args.get("threshold", 25))
    sd = _load_symbol(sym)
    if sd is None:
        return jsonify({"error": f"No data for {sym}"}), 200
    com = agents_mod.Committee({"committee": {"threshold": thr}})
    out = com.latest(sd.frame)
    out["symbol"] = sym.upper()
    out["last"] = round(sd.last_close, 2)
    return jsonify(out)


@app.route("/api/committee_backtest")
def api_committee_backtest():
    sym = request.args.get("symbol", "").strip()
    interval = request.args.get("interval", "1d")
    thr = int(request.args.get("threshold", 25))
    allow_short = request.args.get("short") == "1"
    try:
        import binance_feed
        crypto = binance_feed.is_crypto(sym)
    except Exception:
        crypto = False
    ppy = _PPY.get((interval, crypto), 252)
    data = fetch([sym], _CFG, lookback_days=1500, interval=interval)
    sd = data.get(sym)
    if sd is None:
        return jsonify({"error": f"No data for {sym}"}), 200
    bt = agents_mod.committee_backtest(
        sd.frame, ppy, float(_CFG["risk"]["transaction_cost"]),
        {"committee": {"threshold": thr}}, allow_short)
    bt.update({"symbol": sym.upper(), "interval": interval, "threshold": thr,
               "short": allow_short})
    return jsonify(bt)


# ------------------------ Signals + notifications --------------------------- #

@app.route("/api/signals")
def api_signals():
    """Actionable signals for the watchlist (or given symbols)."""
    raw = request.args.get("symbols", "")
    syms = [s.strip() for s in raw.split(",") if s.strip()] or _load_watchlist()
    thr = request.args.get("threshold")
    interval = request.args.get("interval", "1d")
    sigs = signals_mod.generate(syms, _CFG, interval=interval,
                                threshold=int(thr) if thr else None)
    return jsonify({"signals": sigs, "channels": notify_mod.status()})


@app.route("/api/notify/status")
def api_notify_status():
    return jsonify(notify_mod.status())


@app.route("/api/notify/test")
def api_notify_test():
    res = notify_mod.send("✅ Test signal from your Trading Agent dashboard.")
    return jsonify({"result": res, "channels": notify_mod.status()})


@app.route("/api/signals/send")
def api_signals_send():
    """Generate watchlist signals and push the new ones to Telegram/WhatsApp."""
    syms = _load_watchlist()
    thr = request.args.get("threshold")
    sigs = signals_mod.generate(syms, _CFG, threshold=int(thr) if thr else None)
    fresh = signals_mod.push(sigs)
    return jsonify({"sent": fresh, "total": len(sigs), "channels": notify_mod.status()})


@app.route("/api/live_signals")
def api_live_signals():
    """Best signals across the open markets (cached). Computes on first call."""
    import json
    path = _STATE / "live_signals.json"
    if path.exists():
        out = json.loads(path.read_text())
    else:
        out = live_signals_mod.run(_CFG, do_notify=False)
    out["channels"] = notify_mod.status()
    return jsonify(out)


@app.route("/api/live_signals/run")
def api_live_signals_run():
    """Recompute best signals now; push STRONG ones if ?notify=1."""
    res = live_signals_mod.run(_CFG, do_notify=request.args.get("notify") == "1")
    res["channels"] = notify_mod.status()
    return jsonify(res)


# --------------------- Live autonomous paper trading ------------------------ #

@app.route("/api/live/start")
def api_live_start():
    a = request.args
    syms = [s.strip() for s in a.get("symbols", "").split(",") if s.strip()] or None
    res = live_trader_mod.SESSION.start(
        app_cfg=_CFG,
        capital=float(a.get("capital", 100000)),
        market=a.get("market", "crypto"),
        leverage=float(a.get("leverage", 1)),
        duration_min=int(a.get("duration", 60)),
        poll_seconds=int(a.get("poll", 30)),
        symbols=syms,
        conviction=int(a.get("conviction", 55)),
        allow_short=a.get("short", "1") == "1",
        max_drawdown_pct=float(a.get("max_dd", 25)),
        fee_rate=float(a["fee"]) if a.get("fee") else None,
    )
    return jsonify(res)


@app.route("/api/live/stop")
def api_live_stop():
    return jsonify(live_trader_mod.SESSION.stop())


@app.route("/api/live/status")
def api_live_status():
    return jsonify(live_trader_mod.SESSION.snapshot())


@app.route("/api/live_chart")
def api_live_chart():
    """Recent price series + committee decision markers for a live signal chart.

    A BUY/SELL marker is placed only where >= `pct`% of the 45 agents agree
    (default 85% = 38/45). Returns the live consensus meter too.
    """
    import numpy as np
    sym = request.args.get("symbol", "BTCUSDT").strip()
    pct = float(request.args.get("pct", 85))
    threshold = max(1, round(pct / 100.0 * len(agents_mod.AGENTS)))
    try:
        import binance_feed
        crypto = binance_feed.is_crypto(sym)
    except Exception:
        crypto = False
    interval = "5m" if crypto else "15m"
    lb = 10 if crypto else 45
    data = fetch([sym], _CFG, lookback_days=lb, interval=interval)
    sd = data.get(sym)
    if sd is None:
        return jsonify({"error": f"No data for {sym}"}), 200

    com = agents_mod.Committee({"committee": {"threshold": threshold}})
    M = com.vote_matrix(sd.frame)
    longs = (M == 1).sum(axis=1).to_numpy()
    shorts = (M == -1).sum(axis=1).to_numpy()
    close = sd.frame["close"].to_numpy()
    idx = sd.frame.index

    win = 120
    s0 = max(0, len(close) - win)
    prices = [{"i": i - s0, "c": round(float(close[i]), 4),
               "t": str(idx[i])[-8:-3]} for i in range(s0, len(close))]
    markers = []
    for i in range(max(1, s0), len(close)):
        if longs[i] >= threshold and longs[i - 1] < threshold:
            markers.append({"i": i - s0, "side": "BUY", "c": round(float(close[i]), 4)})
        elif shorts[i] >= threshold and shorts[i - 1] < threshold:
            markers.append({"i": i - s0, "side": "SELL", "c": round(float(close[i]), 4)})

    n = len(agents_mod.AGENTS)
    cur_long, cur_short = int(longs[-1]), int(shorts[-1])
    if cur_long >= threshold:
        verdict, side = f"BUY — {cur_long}/{n} agree", "BUY"
    elif cur_short >= threshold:
        verdict, side = f"SELL — {cur_short}/{n} agree", "SELL"
    else:
        verdict, side = (f"no signal — {max(cur_long, cur_short)}/{n} "
                         f"(need {threshold} for {pct:.0f}%)", "FLAT")
    return jsonify({
        "symbol": sym.upper(), "interval": interval, "last": round(float(close[-1]), 4),
        "prices": prices, "markers": markers, "threshold": threshold, "pct": pct,
        "total": n, "long": cur_long, "short": cur_short,
        "verdict": verdict, "side": side, "updated": idx[-1].strftime("%H:%M"),
    })


# ----------------------- Derivatives & paper options ------------------------ #

@app.route("/api/derivatives")
def api_derivatives():
    return jsonify(derivatives_mod.load())


_MTF_CACHE: Dict[str, tuple] = {}


@app.route("/api/mtf")
def api_mtf():
    """Multi-timeframe read: per-TF specialist scores + one clear aligned verdict."""
    sym = request.args.get("symbol", "BTCUSDT").strip().upper()
    now = time.time()
    if sym in _MTF_CACHE and now - _MTF_CACHE[sym][0] < 300:
        return jsonify(_MTF_CACHE[sym][1])
    try:
        r = mtf_mod.analyze(sym, _CFG)
    except Exception as exc:
        logger.exception("mtf failed")
        return jsonify({"error": str(exc)}), 200
    _MTF_CACHE[sym] = (now, r)
    return jsonify(r)


@app.route("/api/sentiment")
def api_sentiment():
    """Market-specific sentiment (crypto Fear&Greed / India VIX / VIX + news)."""
    sym = request.args.get("symbol", "BTCUSDT").strip()
    return jsonify(sentiment_mod.market_sentiment(sym))


_DM_CACHE: Dict[str, object] = {}


@app.route("/api/dual_momentum")
def api_dual_momentum():
    """Research-backed dual/time-series momentum backtest + current positioning."""
    now = time.time()
    if _DM_CACHE.get("t", 0) and now - float(_DM_CACHE["t"]) < 3600:
        return jsonify(_DM_CACHE["r"])
    try:
        r = dm_mod.backtest(_CFG, years=9)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 200
    _DM_CACHE["t"], _DM_CACHE["r"] = now, r
    return jsonify(r)


@app.route("/api/signal_perf")
def api_signal_perf():
    """Honest hit-rate / expectancy of our signal rule, swept over conviction."""
    cfg = markets.resolve(load_config(HERE / "config.yaml"),
                          request.args.get("market"))
    horizon = int(request.args.get("horizon", 5))
    years = int(request.args.get("years", 3))
    try:
        return jsonify(signal_perf_mod.analyze_sweep(cfg, years, horizon,
                                                     [40, 50, 60, 70]))
    except Exception as exc:
        logger.exception("signal_perf failed")
        return jsonify({"error": str(exc)}), 200


@app.route("/api/options/buy")
def api_options_buy():
    sym = request.args.get("symbol", "").strip()
    kind = "put" if request.args.get("kind", "call").lower().startswith("p") else "call"
    qty = int(request.args.get("qty", 1))
    sd = _load_symbol(sym)
    if sd is None:
        return jsonify({"error": f"No data for {sym}"}), 400
    f = forecast_tomorrow(sd.frame["close"], _FC)
    pos = options_mod.buy(sym, kind, f["level"], f["sigma_next_ann"], qty)
    return jsonify({"ok": True, "position": pos})


@app.route("/api/options/book")
def api_options_book():
    book = options_mod._load()
    prices, vols = {}, {}
    for s in {p["symbol"] for p in book}:
        sd = _load_symbol(s)
        if sd is None:
            continue
        f = forecast_tomorrow(sd.frame["close"], _FC)
        prices[s] = f["level"]; vols[s] = f["sigma_next_ann"]
    return jsonify({"book": options_mod.book_with_marks(prices, vols)})


@app.route("/api/options/close")
def api_options_close():
    options_mod.close(request.args.get("id", ""))
    return jsonify({"ok": True})


@app.route("/api/insight")
def api_insight():
    symbol = request.args.get("symbol", "").strip()
    if not symbol:
        return jsonify({"error": "Provide ?symbol=TICKER"}), 400
    try:
        return jsonify(_insight(symbol))
    except Exception as exc:
        logger.exception("insight failed")
        return jsonify({"error": f"Lookup failed: {exc}"}), 500


@app.route("/api/calibrate")
def api_calibrate():
    """Run the (slower) calibration check so the page can prove trustworthiness."""
    symbol = request.args.get("symbol", "").strip()
    sd = _load_symbol(symbol)
    if sd is None:
        return jsonify({"error": "No data"}), 404
    cal = calibrate(sd.frame["close"], _FC)
    return jsonify({
        "n": int(cal["n"]),
        "cover50": _f(cal["cover50"]),
        "cover90": _f(cal["cover90"]),
        "mean_pit": _f(cal["mean_pit"]),
    })


@app.route("/api/screen")
def api_screen():
    raw = request.args.get("symbols", "")
    symbols = [s.strip() for s in raw.split(",") if s.strip()]
    if not symbols:
        symbols = list(_CFG["universe"])
    rows: List[dict] = []
    for sym in symbols[:25]:  # cap to keep it snappy
        try:
            ins = _insight(sym)
            if "error" in ins:
                continue
            rows.append({
                "symbol": ins["symbol"],
                "composite": ins["signal"]["composite"],
                "is_candidate": ins["signal"]["is_candidate"],
                "last": ins["last"],
                "exp_move_pct": ins["forecast"]["exp_move_pct"],
                "p_up": ins["forecast"]["p_up"],
                "regime": ins["regime"],
            })
        except Exception:
            logger.warning("screen: %s failed", sym)
    rows.sort(key=lambda r: (r["composite"] is not None, r["composite"]),
              reverse=True)
    return jsonify({"rows": rows, "threshold": float(_CFG["risk"]["entry_threshold"])})


# --------------------- Live dashboard endpoints ----------------------------- #

@app.route("/api/futures_feed")
def api_futures_feed():
    """Live Binance USDT-perp board: mark, 24h%, funding, OI, max leverage."""
    limit = int(request.args.get("limit", 25))
    sort = request.args.get("sort", "volume")
    return jsonify(futures_feed_mod.board(limit=limit, sort=sort))


# lookback (days) per interval, sized so intraday indicator warmup survives
_CANDLE_LB = {"5m": 12, "15m": 45, "30m": 60, "1h": 90, "4h": 400,
              "1d": 1200, "1wk": 2500}


@app.route("/api/candles")
def api_candles():
    """OHLC candles + detected candlestick patterns + per-bar committee
    consensus markers (BUY/SELL where >= ``min_agree`` of 45 agents agree)."""
    sym = request.args.get("symbol", "BTCUSDT").strip().upper()
    interval = request.args.get("interval", "15m")
    min_agree = max(1, min(len(agents_mod.AGENTS), int(request.args.get("min_agree", 38))))
    lb = _CANDLE_LB.get(interval, 45)
    data = fetch([sym], _CFG, lookback_days=lb, interval=interval)
    sd = data.get(sym)
    if sd is None or len(sd.frame) < 5:
        return jsonify({"error": f"No data for {sym} @ {interval}"}), 200
    frame = sd.frame
    close = frame["close"].to_numpy()
    n = len(close)
    win = min(120, n)
    s0 = n - win
    idx = frame.index
    intraday = interval not in ("1d", "1wk")

    def _t(i):
        try:
            return idx[i].strftime("%d %H:%M" if intraday else "%Y-%m-%d")
        except Exception:
            return str(idx[i])[:16]

    o, h, l, v = (frame["open"].to_numpy(), frame["high"].to_numpy(),
                  frame["low"].to_numpy(), frame["volume"].to_numpy())
    bars = [{"i": i - s0, "t": _t(i), "o": round(float(o[i]), 4),
             "h": round(float(h[i]), 4), "l": round(float(l[i]), 4),
             "c": round(float(close[i]), 4), "v": round(float(v[i]), 2)}
            for i in range(s0, n)]

    com = agents_mod.Committee({"committee": {"threshold": min_agree}})
    M = com.vote_matrix(frame)
    longs = (M == 1).sum(axis=1).to_numpy()
    shorts = (M == -1).sum(axis=1).to_numpy()
    markers = []
    for i in range(max(1, s0), n):
        if longs[i] >= min_agree and longs[i - 1] < min_agree:
            markers.append({"i": i - s0, "side": "BUY", "c": round(float(close[i]), 4),
                            "n": int(longs[i])})
        elif shorts[i] >= min_agree and shorts[i - 1] < min_agree:
            markers.append({"i": i - s0, "side": "SELL", "c": round(float(close[i]), 4),
                            "n": int(shorts[i])})

    patt = candles_mod.detect(frame, lookback=win)
    csum = candles_mod.summary(frame)
    total = len(agents_mod.AGENTS)
    cl, cs = int(longs[-1]), int(shorts[-1])
    if cl >= min_agree:
        side, verdict = "BUY", f"BUY — {cl}/{total} agents agree"
    elif cs >= min_agree:
        side, verdict = "SELL", f"SELL — {cs}/{total} agents agree"
    else:
        side, verdict = "WAIT", f"WAIT — best {max(cl, cs)}/{total} (need {min_agree})"
    last = M.iloc[-1]
    fam: Dict[str, int] = {}
    for nm, f, _ in agents_mod.AGENTS:
        fam[f] = fam.get(f, 0) + int(last[nm])
    return jsonify({
        "symbol": sym, "interval": interval, "last": round(float(close[-1]), 4),
        "bars": bars, "markers": markers, "patterns": patt, "candle_summary": csum,
        "consensus": {"side": side, "verdict": verdict, "long": cl, "short": cs,
                      "total": total, "min_agree": min_agree, "families": fam},
        "updated": idx[-1].strftime("%H:%M:%S") if hasattr(idx[-1], "strftime") else str(idx[-1]),
    })


_OC_SPOT = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "FINNIFTY": "^NSEI",
            "SENSEX": "^BSESN", "BTC": "BTCUSDT", "ETH": "ETHUSDT"}


@app.route("/api/option_chain")
def api_option_chain():
    """Option chain, best source first: real Kite chain (if the broker bridge is
    connected and the symbol is an Indian F&O name) → NSE/Deribit live feed →
    Black-Scholes-modeled fallback."""
    sym = request.args.get("symbol", "NIFTY").strip().upper()
    # 1) Real broker chain for Indian F&O names when Kite is live.
    if sym not in {"BTC", "ETH"} and not sym.endswith(("USDT", "-USD")):
        try:
            if kite_mod.status().get("connected"):
                kc = kite_mod.option_chain(sym)
                if kc.get("rows"):
                    return jsonify(kc)
        except Exception:
            logger.warning("kite option_chain failed for %s; falling back", sym)
    # 2) Existing NSE/Deribit live feed, else modeled fallback.
    spot = sigma = None
    yf_sym = _OC_SPOT.get(sym, sym)
    try:
        sd = _load_symbol(yf_sym)
        if sd is not None:
            f = forecast_tomorrow(sd.frame["close"], _FC)
            spot, sigma = float(f["level"]), float(f["sigma_next_ann"])
    except Exception:
        logger.warning("option_chain spot/vol fetch failed for %s", sym)
    try:
        return jsonify(option_chain_mod.chain(sym, spot=spot, sigma_ann=sigma))
    except Exception as exc:
        logger.exception("option_chain failed")
        return jsonify({"error": str(exc), "rows": []}), 200


# --------------------- Signal backtester ------------------------------------ #

@app.route("/api/signal_backtest")
def api_signal_backtest():
    """Backtest the consensus BUY/SELL points as ATR-target/stop trades.
    ``sweep=1`` adds a min_agree threshold sweep; ``wf=1`` runs the honest
    walk-forward instead (thresholds picked on past data, traded out-of-sample)."""
    import signal_backtest as sbt
    a = request.args
    sym = a.get("symbol", "BTCUSDT").strip()
    interval = a.get("interval", "1d")
    kw = dict(
        interval=interval,
        min_agree=int(a.get("min_agree", 35)),
        target_atr=float(a.get("target_atr", 2.0)),
        stop_atr=float(a.get("stop_atr", 1.0)),
        max_hold=int(a.get("max_hold", 20)),
        allow_short=a.get("short", "1") != "0",
        risk_pct=float(a.get("risk_pct", 0.01)),
    )
    try:
        if a.get("wf") == "1":
            wf_kw = {k: v for k, v in kw.items() if k != "min_agree"}
            return jsonify(sbt.walk_forward(sym, _CFG,
                                            n_folds=int(a.get("folds", 4)), **wf_kw))
        out = sbt.backtest(sym, _CFG, **kw)
        if a.get("sweep") == "1" and "error" not in out:
            out["sweep"] = sbt.sweep(sym, _CFG, interval=interval,
                                     target_atr=kw["target_atr"], stop_atr=kw["stop_atr"],
                                     max_hold=kw["max_hold"], allow_short=kw["allow_short"],
                                     risk_pct=kw["risk_pct"])
        return jsonify(out)
    except Exception as exc:
        logger.exception("signal_backtest failed")
        return jsonify({"error": str(exc)}), 200


# --------------------- Pro risk desk + regime -------------------------------- #

@app.route("/api/regime")
def api_regime():
    """Market regime read: trending / calm range / volatile chop + playbook."""
    sym = request.args.get("symbol", "BTCUSDT").strip()
    interval = request.args.get("interval", "1d")
    try:
        return jsonify(regime_mod.analyze(sym, _CFG, interval=interval))
    except Exception as exc:
        logger.exception("regime failed")
        return jsonify({"error": str(exc)}), 200


@app.route("/api/risk/discipline")
def api_risk_discipline():
    """GREEN/YELLOW/RED trading-permission light from the journal + heat."""
    try:
        return jsonify(risk_pro_mod.discipline(_CFG))
    except Exception as exc:
        logger.exception("discipline failed")
        return jsonify({"error": str(exc)}), 200


@app.route("/api/risk/checklist")
def api_risk_checklist():
    """Pre-trade checklist: R:R, regime, committee, sizing, heat, daily limits
    → TAKE / HALF SIZE / SKIP. Blank entry/stop/target auto-fill from ATR."""
    a = request.args
    sym = a.get("symbol", "").strip()
    if not sym:
        return jsonify({"error": "symbol required"}), 200
    try:
        return jsonify(risk_pro_mod.checklist(
            sym, _CFG,
            side=a.get("side", "BUY"),
            entry=float(a["entry"]) if a.get("entry") else None,
            stop=float(a["stop"]) if a.get("stop") else None,
            target=float(a["target"]) if a.get("target") else None,
            equity=float(a["equity"]) if a.get("equity") else None,
            risk_pct=float(a["risk_pct"]) if a.get("risk_pct") else None,
            interval=a.get("interval", "1d")))
    except Exception as exc:
        logger.exception("checklist failed")
        return jsonify({"error": str(exc)}), 200


# --------------------- TradingView Desktop bridge --------------------------- #

@app.route("/api/tv/status")
def api_tv_status():
    """Is TradingView Desktop installed / running / CDP-connected?"""
    return jsonify(tv_bridge_mod.status())


@app.route("/api/tv/launch")
def api_tv_launch():
    """Launch TradingView Desktop with the CDP debug port."""
    return jsonify(tv_bridge_mod.launch())


@app.route("/api/tv/quote")
def api_tv_quote():
    """Quote a symbol through TradingView (licensed exchange data, incl. NSE/BSE).
    Side effect by design: the desktop chart follows the dashboard."""
    sym = request.args.get("symbol", "").strip()
    if not sym:
        return jsonify({"success": False, "error": "symbol required"}), 200
    return jsonify(tv_bridge_mod.quote(sym))


# --------------------- Trade journal endpoints ------------------------------ #

@app.route("/api/journal")
def api_journal():
    """List trades (newest first) + live marks on open ones + forward stats."""
    status = request.args.get("status", "all")
    prices: Dict[str, float] = {}
    if request.args.get("mark", "1") != "0":
        for t in journal_mod.list_trades("open"):
            try:
                sd = _load_symbol(t["symbol"])
                if sd is not None:
                    prices[t["symbol"]] = sd.last_close
            except Exception:
                pass
    return jsonify({"trades": journal_mod.list_trades(status, prices or None),
                    "stats": journal_mod.stats()})


@app.route("/api/journal/log")
def api_journal_log():
    a = request.args
    if not a.get("symbol") or not a.get("entry"):
        return jsonify({"ok": False, "error": "symbol and entry required"}), 200
    try:
        t = journal_mod.log(
            symbol=a.get("symbol"), side=a.get("side", "BUY"), entry=float(a.get("entry")),
            target=float(a["target"]) if a.get("target") else None,
            stop=float(a["stop"]) if a.get("stop") else None,
            conviction=float(a["conviction"]) if a.get("conviction") else None,
            source=a.get("source", "manual"), note=a.get("note", ""), market=a.get("market"))
        return jsonify({"ok": True, "trade": t})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 200


@app.route("/api/journal/close")
def api_journal_close():
    tid = request.args.get("id", "")
    exit_price = request.args.get("exit")
    if exit_price:
        px = float(exit_price)
    else:  # default to the current market price
        t = next((x for x in journal_mod.list_trades("open") if x["id"] == tid), None)
        sd = _load_symbol(t["symbol"]) if t else None
        px = sd.last_close if sd else None
    if px is None:
        return jsonify({"ok": False, "error": "No exit price available"}), 200
    t = journal_mod.close(tid, px)
    return jsonify({"ok": bool(t), "trade": t})


@app.route("/api/journal/delete")
def api_journal_delete():
    return jsonify({"ok": journal_mod.delete(request.args.get("id", ""))})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    print("\n  Trading dashboard → http://127.0.0.1:5000   (Ctrl-C to stop)\n")
    lan = _lan_ip()
    print(f"\n  Dashboard:  http://127.0.0.1:5000")
    print(f"  📱 Phone (same Wi-Fi): http://{lan}:5000  → then 'Add to Home Screen'")
    print(f"  ⚠ LAN-only by design — do NOT port-forward this to the internet.\n")
    # 0.0.0.0 = reachable from your phone on the same Wi-Fi (paper money only).
    # TEMPLATES_AUTO_RELOAD (set above) already applies UI edits on refresh — no
    # code reloader here, since its forked subprocess breaks managed launchers.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
