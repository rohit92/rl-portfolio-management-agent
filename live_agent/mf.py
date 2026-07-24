"""
mf.py — Indian mutual funds: search, NAV history and honest analytics.

Data source: the free public mfapi.in API, which republishes official AMFI NAV
data for every Indian mutual fund scheme (~40k+ schemes). No key needed.

    search("nifty index")   -> [{code, name}, ...]
    analyze(code)           -> latest NAV, trailing returns (1m/3m/6m/1y),
                               3y/5y/inception CAGR, annualised volatility,
                               max drawdown, and a downsampled NAV series.

Honest notes baked in:
  • NAV is once-a-day and T+1 — mutual funds are not trading instruments; they
    are allocation instruments. No intraday, no stops, no leverage.
  • Debt-fund NAVs are accrual-smoothed — their "volatility" here understates
    real credit/duration risk.
  • Past CAGR is the *starting point* of research, not a promise; sequence risk
    and category cycles dominate 3–5 year windows.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger("mf")

_BASE = "https://api.mfapi.in/mf"
_TTL = 3600  # seconds
_CACHE: Dict[str, tuple] = {}


def _get(url: str):
    now = time.time()
    hit = _CACHE.get(url)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    data = r.json()
    _CACHE[url] = (now, data)
    return data


def search(q: str, limit: int = 25) -> List[Dict]:
    """Search schemes by name fragment (AMFI names, e.g. 'parag parikh flexi')."""
    q = (q or "").strip()
    if len(q) < 3:
        return []
    try:
        rows = _get(f"{_BASE}/search?q={requests.utils.quote(q)}") or []
    except Exception as exc:
        logger.warning("mf search failed: %s", exc)
        return []
    return [{"code": r.get("schemeCode"), "name": r.get("schemeName")}
            for r in rows[:limit]]


def _parse_series(data: List[Dict]) -> List[tuple]:
    """API gives newest-first [{date: 'dd-mm-yyyy', nav: '123.4'}]; return
    oldest-first [(datetime, float_nav)] with junk rows dropped."""
    out = []
    for row in reversed(data or []):
        try:
            nav = float(row["nav"])
            if nav <= 0:
                continue
            out.append((datetime.strptime(row["date"], "%d-%m-%Y"), nav))
        except (KeyError, ValueError):
            continue
    return out


def _nav_at_or_before(series: List[tuple], target: datetime) -> Optional[float]:
    prev = None
    for d, v in series:
        if d > target:
            break
        prev = v
    return prev


def analyze(code: str) -> Dict:
    """Full honest scorecard for one scheme code."""
    try:
        raw = _get(f"{_BASE}/{int(code)}")
    except Exception as exc:
        return {"error": f"Could not fetch scheme {code}: {exc}"}
    meta = raw.get("meta") or {}
    series = _parse_series(raw.get("data"))
    if len(series) < 30:
        return {"error": f"Scheme {code} has too little NAV history."}

    last_d, last_nav = series[-1]
    first_d, first_nav = series[0]

    def trailing(days: int) -> Optional[float]:
        base = _nav_at_or_before(series, last_d - timedelta(days=days))
        return round((last_nav / base - 1.0) * 100.0, 2) if base else None

    def cagr(days: int) -> Optional[float]:
        base = _nav_at_or_before(series, last_d - timedelta(days=days))
        if not base:
            return None
        yrs = days / 365.0
        return round(((last_nav / base) ** (1.0 / yrs) - 1.0) * 100.0, 2)

    span_days = (last_d - first_d).days or 1
    inception_cagr = round(((last_nav / first_nav) ** (365.0 / span_days) - 1.0) * 100.0, 2) \
        if span_days >= 365 else None

    # daily-change vol + max drawdown on NAV
    navs = [v for _, v in series]
    rets = [(navs[i] / navs[i - 1] - 1.0) for i in range(1, len(navs))
            if navs[i - 1] > 0]
    vol_ann = None
    if len(rets) > 40:
        mu = sum(rets) / len(rets)
        var = sum((x - mu) ** 2 for x in rets) / (len(rets) - 1)
        vol_ann = round(math.sqrt(var) * math.sqrt(252) * 100.0, 2)
    peak, mdd = navs[0], 0.0
    for v in navs:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1.0)

    step = max(1, len(series) // 200)
    chart = [{"t": d.strftime("%Y-%m-%d"), "v": round(v, 4)}
             for d, v in series[::step]]

    return {
        "code": int(code),
        "name": meta.get("scheme_name"),
        "fund_house": meta.get("fund_house"),
        "category": meta.get("scheme_category"),
        "type": meta.get("scheme_type"),
        "latest_nav": round(last_nav, 4),
        "nav_date": last_d.strftime("%Y-%m-%d"),
        "since": first_d.strftime("%Y-%m-%d"),
        "returns": {"1m": trailing(30), "3m": trailing(91), "6m": trailing(182),
                    "1y": trailing(365)},
        "cagr": {"3y": cagr(365 * 3), "5y": cagr(365 * 5),
                 "inception": inception_cagr},
        "vol_ann_pct": vol_ann,
        "max_dd_pct": round(mdd * 100.0, 2),
        "points": len(series),
        "chart": chart,
        "note": ("NAV is once-a-day (T+1) — funds are allocation vehicles, not trading "
                 "instruments. Debt-fund vol here understates real risk (accrual "
                 "smoothing). Past CAGR ≠ future returns."),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    res = search("parag parikh flexi")
    print(f"search: {len(res)} hits; first: {res[0] if res else None}")
    if res:
        a = analyze(res[0]["code"])
        if "error" in a:
            print("ERROR", a["error"])
        else:
            print(f"{a['name']} | NAV {a['latest_nav']} ({a['nav_date']}) | "
                  f"1y {a['returns']['1y']}% | 3y CAGR {a['cagr']['3y']}% | "
                  f"5y CAGR {a['cagr']['5y']}% | vol {a['vol_ann_pct']}% | "
                  f"maxDD {a['max_dd_pct']}%")
