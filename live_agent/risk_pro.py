"""
risk_pro.py — The professional risk desk: sizing, heat, discipline, checklist.

Everything this project has measured says the same thing: the signals are
~48–55% (coin-flip-ish) and the *durable* edge is risk control. This module is
that edge, made concrete. Four tools, exactly what a prop-desk risk manager
does for a trader:

1. position_size()   — "How much?" is a math question, not a feeling. Risk a
                       fixed fraction of equity per trade; the stop distance
                       sets the quantity. Tight stop -> bigger size, wide stop
                       -> smaller size, risk identical either way.

2. open_heat()       — total % of equity at risk across ALL open journal
                       trades right now. Individually-sensible trades add up
                       to a blow-up; heat is the number that catches it.

3. discipline()      — the daily circuit breakers pros are *forced* to obey:
                       max daily loss (in R), max consecutive losses (size
                       down — you are tilted, whatever you think), heat cap.
                       Returns GREEN / YELLOW / RED with reasons.

4. checklist()       — the pre-trade checklist. Every check the desk would
                       run before allowing the order: R:R, regime alignment,
                       committee agreement, heat after this trade, daily
                       limits, streak state. Verdict: TAKE / HALF SIZE / SKIP.

None of this predicts price. All of it decides whether a losing streak is an
ordinary Tuesday or the end of the account. Config lives in ``risk_pro:`` in
config.yaml; journal.json is the source of truth for open risk and daily P&L.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import agents as agents_mod
import journal as journal_mod
import regime as regime_mod
from data_feed import fetch

logger = logging.getLogger("risk_pro")

_LB = {"5m": 20, "15m": 60, "30m": 90, "1h": 180, "4h": 720, "1d": 1500, "1wk": 3000}

_DEFAULTS = {
    "risk_per_trade_pct": 1.0,      # % of equity risked per trade (the "1R")
    "max_open_heat_pct": 5.0,       # total open risk cap across all positions
    "max_daily_loss_r": 3.0,        # stop trading for the day at -3R realised
    "max_consecutive_losses": 3,    # 3 losses in a row -> half size (tilt guard)
    "min_rr": 1.5,                  # reward:risk below this is a warn
    "max_positions_per_market": 3,  # concentration guard (per market, per side)
}


def _cfg(cfg: Optional[dict]) -> Dict:
    out = dict(_DEFAULTS)
    out.update((cfg or {}).get("risk_pro") or {})
    return out


# --------------------------------------------------------------------------- #
#  1) Position sizing
# --------------------------------------------------------------------------- #
def position_size(equity: float, risk_pct: float, entry: float, stop: float,
                  target: Optional[float] = None, fee_rate: float = 0.0) -> Dict:
    """Fixed-fractional sizing: qty = (equity * risk%) / stop-distance."""
    equity, entry, stop = float(equity), float(entry), float(stop)
    dist = abs(entry - stop)
    if equity <= 0 or entry <= 0 or dist <= 0:
        return {"error": "Need equity > 0, entry > 0 and a stop away from entry."}
    risk_amount = equity * risk_pct / 100.0
    qty = risk_amount / dist
    notional = qty * entry
    rr = None
    if target is not None and float(target) > 0:
        rr = round(abs(float(target) - entry) / dist, 2)
    fees = notional * fee_rate * 2.0
    return {
        "risk_amount": round(risk_amount, 2),
        "qty": round(qty, 6),
        "notional": round(notional, 2),
        "pct_of_equity": round(100.0 * notional / equity, 1),
        "leverage_needed": round(notional / equity, 2) if notional > equity else None,
        "stop_distance": round(dist, 6),
        "stop_distance_pct": round(100.0 * dist / entry, 2),
        "rr": rr,
        "round_trip_fees": round(fees, 2) if fee_rate else None,
        "note": (f"Risking {risk_pct}% of equity = {round(risk_amount, 2)}. The stop sets "
                 f"the size — never the other way around."),
    }


# --------------------------------------------------------------------------- #
#  2) Portfolio heat
# --------------------------------------------------------------------------- #
def open_heat(cfg: Optional[dict] = None) -> Dict:
    """Total equity-% at risk across open journal trades (each sized at 1R)."""
    rp = _cfg(cfg)
    per = float(rp["risk_per_trade_pct"])
    opens = journal_mod.list_trades("open")
    by_market: Dict[str, int] = {}
    by_side: Dict[str, int] = {}
    no_stop = 0
    for t in opens:
        by_market[t.get("market") or "?"] = by_market.get(t.get("market") or "?", 0) + 1
        by_side[t.get("side") or "?"] = by_side.get(t.get("side") or "?", 0) + 1
        if t.get("stop") is None:
            no_stop += 1
    heat = round(len(opens) * per, 2)
    cap = float(rp["max_open_heat_pct"])
    conc = {m: n for m, n in by_market.items() if n > int(rp["max_positions_per_market"])}
    return {
        "open_trades": len(opens),
        "risk_per_trade_pct": per,
        "heat_pct": heat, "heat_cap_pct": cap,
        "heat_used": round(100.0 * heat / cap, 0) if cap else None,
        "over_cap": heat > cap,
        "by_market": by_market, "by_side": by_side,
        "concentrated": conc,           # markets holding more than the per-market cap
        "no_stop": no_stop,             # open trades with NO stop = unbounded risk
        "note": (f"Assumes every open trade risks {per}% (1R). {no_stop} open trade(s) "
                 f"have no stop — that risk is unbounded and not in this number."
                 if no_stop else
                 f"Assumes every open trade risks {per}% (1R) of equity."),
    }


# --------------------------------------------------------------------------- #
#  3) Daily discipline
# --------------------------------------------------------------------------- #
def discipline(cfg: Optional[dict] = None) -> Dict:
    """GREEN / YELLOW / RED trading-permission light, from the journal."""
    rp = _cfg(cfg)
    stats = journal_mod.stats()
    heat = open_heat(cfg)

    # today's realised R (and pnl%) from closed trades
    day0 = time.mktime(time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d"))
    closed = [t for t in journal_mod.list_trades("closed")
              if (t.get("closed_ts") or 0) >= day0]
    today_r = sum(t["r_multiple"] for t in closed if t.get("r_multiple") is not None)
    today_pnl = sum(t["pnl_pct"] for t in closed if t.get("pnl_pct") is not None)

    streak = stats.get("streak") or 0
    max_daily = float(rp["max_daily_loss_r"])
    max_losses = int(rp["max_consecutive_losses"])

    reasons: List[str] = []
    status = "GREEN"
    if today_r <= -max_daily:
        status = "RED"
        reasons.append(f"Daily loss limit hit: {round(today_r, 1)}R today "
                       f"(limit −{max_daily}R). Pros stop; revenge trades pay for "
                       f"someone else's yacht. Come back tomorrow.")
    if heat["over_cap"]:
        status = "RED"
        reasons.append(f"Open heat {heat['heat_pct']}% exceeds the {heat['heat_cap_pct']}% "
                       f"cap — no new risk until something is closed.")
    if status != "RED":
        if streak <= -max_losses:
            status = "YELLOW"
            reasons.append(f"{abs(streak)} losses in a row — statistically normal, "
                           f"psychologically not. Half size until a winner resets you.")
        if heat["heat_cap_pct"] and heat["heat_pct"] >= 0.6 * heat["heat_cap_pct"]:
            status = "YELLOW" if status == "GREEN" else status
            reasons.append(f"Heat {heat['heat_pct']}% is over 60% of the cap — "
                           f"be choosy with the remaining risk budget.")
        if heat["no_stop"]:
            status = "YELLOW" if status == "GREEN" else status
            reasons.append(f"{heat['no_stop']} open trade(s) without a stop — "
                           f"define the exit or the market defines it for you.")
        if heat["concentrated"]:
            status = "YELLOW" if status == "GREEN" else status
            reasons.append("Concentration: " + ", ".join(
                f"{n} open in {m}" for m, n in heat["concentrated"].items())
                + f" (cap {rp['max_positions_per_market']}/market) — correlated positions "
                  f"are one position wearing disguises.")
    if not reasons:
        reasons.append("All clear: inside daily loss limit, heat cap and streak rules.")

    return {
        "status": status,
        "reasons": reasons,
        "today_r": round(today_r, 2), "today_pnl_pct": round(today_pnl, 2),
        "today_closed": len(closed),
        "streak": streak,
        "heat": heat,
        "rules": {
            "max_daily_loss_r": max_daily,
            "max_consecutive_losses": max_losses,
            "max_open_heat_pct": heat["heat_cap_pct"],
            "risk_per_trade_pct": heat["risk_per_trade_pct"],
            "max_positions_per_market": int(rp["max_positions_per_market"]),
        },
        "note": ("These limits are the actual pro edge: they make losing streaks "
                 "survivable. GREEN = trade the plan, YELLOW = half size, RED = flat."),
    }


# --------------------------------------------------------------------------- #
#  4) Pre-trade checklist
# --------------------------------------------------------------------------- #
def checklist(symbol: str, cfg: dict, *, side: str = "BUY",
              entry: Optional[float] = None, stop: Optional[float] = None,
              target: Optional[float] = None, equity: Optional[float] = None,
              risk_pct: Optional[float] = None, interval: str = "1d") -> Dict:
    """Run the trade through every desk check and return TAKE / HALF / SKIP.

    Leave entry/stop/target empty to auto-fill from the live price and ATR
    (stop 1×ATR, target 2×ATR — a 2:1 template, not a recommendation).
    """
    rp = _cfg(cfg)
    symbol = symbol.strip().upper()
    side = "BUY" if str(side).upper().startswith(("B", "L")) else "SELL"
    equity = float(equity) if equity else float(cfg.get("initial_cash", 100_000))
    risk_pct = float(risk_pct) if risk_pct else float(rp["risk_per_trade_pct"])

    data = fetch([symbol], cfg, lookback_days=_LB.get(interval, 730), interval=interval)
    sd = data.get(symbol)
    if sd is None or len(sd.frame) < 60:
        return {"error": f"Not enough data for {symbol} @ {interval}."}
    frame = sd.frame
    px = float(frame["close"].iloc[-1])

    # ATR for auto stop/target
    h, l, c = frame["high"], frame["low"], frame["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / 14, min_periods=14).mean().iloc[-1])

    sgn = 1 if side == "BUY" else -1
    entry = float(entry) if entry else px
    stop = float(stop) if stop else entry - sgn * atr
    target = float(target) if target else entry + sgn * 2 * atr
    auto = {"entry": entry == px, "stop_atr": 1.0, "target_atr": 2.0}

    checks: List[Dict] = []

    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})

    # -- stop on the right side, R:R --------------------------------------- #
    dist = (entry - stop) * sgn
    reward = (target - entry) * sgn
    if dist <= 0:
        add("Stop placement", "fail",
            f"Stop {round(stop, 4)} is on the wrong side of entry for a {side} — that is "
            f"not a stop, it is a target wearing a costume.")
        rr = None
    else:
        rr = round(reward / dist, 2) if reward > 0 else 0.0
        min_rr = float(rp["min_rr"])
        if reward <= 0:
            add("Reward:risk", "fail", f"Target {round(target, 4)} is on the wrong side "
                                       f"of entry for a {side}.")
        elif rr < 1.0:
            add("Reward:risk", "fail",
                f"R:R {rr}:1 — risking more than the trade can pay. At a ~50% hit rate "
                f"this loses by construction.")
        elif rr < min_rr:
            add("Reward:risk", "warn",
                f"R:R {rr}:1 is below the {min_rr}:1 desk minimum — needs a >"
                f"{round(100 / (1 + rr), 0):.0f}% win rate just to break even.")
        else:
            add("Reward:risk", "pass", f"R:R {rr}:1 — a {round(100 / (1 + rr), 0):.0f}% "
                                       f"win rate breaks even; our measured ~50% has margin.")

    # -- regime alignment --------------------------------------------------- #
    reg = regime_mod.detect(frame, ppy=regime_mod._PPY.get(interval, 252))
    if "error" not in reg:
        rl, rd = reg["label"], reg["direction"]
        aligned = (side == "BUY" and rd == "up") or (side == "SELL" and rd == "down")
        against = (side == "BUY" and rd == "down") or (side == "SELL" and rd == "up")
        if reg["label"] == "VOLATILE CHOP":
            add("Regime", "warn", f"{rl} — the regime that eats accounts; the desk cuts "
                                  f"size in this weather regardless of setup.")
        elif reg["trending"] and against:
            add("Regime", "warn", f"{rl} and this is a counter-trend {side} — fading a "
                                  f"trend is the most expensive habit in trading.")
        elif reg["trending"] and aligned:
            add("Regime", "pass", f"{rl} — with-trend {side}, the tailwind is yours.")
        else:
            add("Regime", "pass", f"{rl} — no strong headwind for a {side}. "
                                  f"Favored setups: {', '.join(reg['favored_families']) or 'none'}.")
    else:
        reg = None
        add("Regime", "warn", "Could not read the regime (insufficient data).")

    # -- committee agreement ------------------------------------------------ #
    committee = None
    try:
        com = agents_mod.Committee(cfg).latest(frame)
        committee = {"long": com["long"], "short": com["short"], "total": com["total"]}
        my, opp = (com["long"], com["short"]) if side == "BUY" else (com["short"], com["long"])
        if opp >= 30:
            add("Committee (45 agents)", "warn",
                f"{opp}/{com['total']} agents vote AGAINST this {side} (only {my} with you) "
                f"— you are fading your own committee.")
        elif my >= 30:
            add("Committee (45 agents)", "pass",
                f"{my}/{com['total']} agents agree with the {side} — solid confluence.")
        else:
            add("Committee (45 agents)", "pass",
                f"Split committee ({my} with, {opp} against) — no strong read either way.")
    except Exception:
        add("Committee (45 agents)", "warn", "Committee unavailable for this symbol.")

    # -- sizing -------------------------------------------------------------- #
    size = position_size(equity, risk_pct, entry, stop, target)
    if "error" in size:
        add("Position size", "fail", size["error"])
    elif size.get("leverage_needed"):
        add("Position size", "warn",
            f"Stop is so tight the full-risk size needs {size['leverage_needed']}x leverage "
            f"({size['pct_of_equity']}% of equity). Either widen the stop past the noise or "
            f"accept a smaller-than-1R trade.")
    else:
        add("Position size", "pass",
            f"Risk {risk_pct}% = {size['risk_amount']} → {size['qty']} units "
            f"(~{size['pct_of_equity']}% of equity). The stop set the size.")

    # -- discipline + heat --------------------------------------------------- #
    disc = discipline(cfg)
    heat_after = round(disc["heat"]["heat_pct"] + risk_pct, 2)
    if disc["status"] == "RED":
        add("Discipline", "fail", "RED — " + " ".join(disc["reasons"]))
    elif heat_after > disc["heat"]["heat_cap_pct"]:
        add("Discipline", "fail",
            f"This trade takes open heat to {heat_after}% — over the "
            f"{disc['heat']['heat_cap_pct']}% cap. Close something first.")
    elif disc["status"] == "YELLOW":
        add("Discipline", "warn", "YELLOW — " + " ".join(disc["reasons"]))
    else:
        add("Discipline", "pass",
            f"GREEN — today {disc['today_r']:+.1f}R, heat after this trade "
            f"{heat_after}% of {disc['heat']['heat_cap_pct']}% cap.")

    # -- verdict -------------------------------------------------------------- #
    fails = [c for c in checks if c["status"] == "fail"]
    warns = [c for c in checks if c["status"] == "warn"]
    if fails:
        verdict, size_mult = "SKIP", 0.0
        why = "; ".join(c["name"] for c in fails) + " failed."
    elif len(warns) >= 2 or disc["status"] == "YELLOW":
        verdict, size_mult = "TAKE — HALF SIZE", 0.5
        why = "Multiple cautions (" + "; ".join(c["name"] for c in warns) + ") — cut size, keep the lesson cheap."
    elif warns:
        verdict, size_mult = "TAKE — FULL SIZE", 1.0
        why = f"One caution ({warns[0]['name']}) — acceptable; know why you're overriding it."
    else:
        verdict, size_mult = "TAKE — FULL SIZE", 1.0
        why = "Every check passed. Now the hard part: follow the stop."

    return {
        "symbol": symbol, "interval": interval, "side": side,
        "last": round(px, 6), "atr": round(atr, 6),
        "entry": round(entry, 6), "stop": round(stop, 6), "target": round(target, 6),
        "auto_filled": auto, "rr": rr,
        "equity": equity, "risk_pct": risk_pct,
        "checks": checks, "verdict": verdict, "size_mult": size_mult, "why": why,
        "size": None if "error" in size else {**size,
            "qty_final": round(size["qty"] * size_mult, 6),
            "risk_final": round(size["risk_amount"] * size_mult, 2)},
        "regime": None if not reg else {"label": reg["label"], "direction": reg["direction"],
                                        "adx": reg["adx"], "risk_hint": reg["risk_hint"]},
        "committee": committee,
        "discipline": {"status": disc["status"], "today_r": disc["today_r"],
                       "streak": disc["streak"], "heat_pct": disc["heat"]["heat_pct"],
                       "heat_after_pct": heat_after,
                       "heat_cap_pct": disc["heat"]["heat_cap_pct"]},
        "note": ("A checklist can't make a trade win — it makes sure that when it loses, "
                 "it loses 1R and not the account. That is the entire pro secret."),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    from pathlib import Path
    from trader import load_config
    cfg = load_config(Path(__file__).resolve().parent / "config.yaml")
    print("— discipline —")
    d = discipline(cfg)
    print(d["status"], "|", d["reasons"][0], "| today", d["today_r"], "R | heat",
          d["heat"]["heat_pct"], "%")
    print("\n— checklist BTCUSDT BUY (auto levels) —")
    r = checklist("BTCUSDT", cfg, side="BUY")
    for c in r["checks"]:
        print(f"  [{c['status']:^4}] {c['name']}: {c['detail'][:90]}")
    print("VERDICT:", r["verdict"], "→", r["why"])
    print("size:", r["size"] and r["size"]["qty_final"], "units, risk",
          r["size"] and r["size"]["risk_final"])
