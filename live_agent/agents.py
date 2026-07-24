"""
agents.py — A committee of 31 independent trading-setup agents + consensus vote.

Each agent implements one well-known setup (from classic TA, momentum/mean-
reversion literature, breakout systems, volume analysis, and simple statistics)
and votes per bar: +1 long, -1 short, 0 flat. The committee acts ONLY when at
least ``threshold`` agents agree on a direction — a high-conviction consensus.

Honest design notes
-------------------
* The agents are grouped into 6 FAMILIES so the vote reflects genuinely
  different ideas. Within a family agents are correlated (e.g. all oscillators
  agree in a strong trend), so "25 of 31" overstates independence — read the
  family tally, not just the raw count.
* Consensus reduces *variance* (fewer one-signal false starts) and trade
  frequency. Whether that helps net of costs is an empirical question — use
  ``committee_backtest`` to check, never assume.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.metrics import max_drawdown, sharpe_ratio  # noqa: E402


# --------------------------------------------------------------------------- #
#  Indicator helpers (hand-rolled, no TA-Lib dependency)
# --------------------------------------------------------------------------- #
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n).mean()


def _rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    rs = (up.ewm(alpha=1/n, min_periods=n).mean()
          / dn.ewm(alpha=1/n, min_periods=n).mean().replace(0, np.nan))
    return 100 - 100 / (1 + rs)


def _atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, min_periods=n).mean()


def _macd(s, f=12, sl=26, sig=9):
    m = _ema(s, f) - _ema(s, sl); return m, _ema(m, sig)


def _boll(s, n=20, k=2):
    m = _sma(s, n); sd = s.rolling(n).std(); return m, m + k*sd, m - k*sd


def _stoch(df, n=14, d=3):
    ll = df["low"].rolling(n).min(); hh = df["high"].rolling(n).max()
    k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    return k, k.rolling(d).mean()


def _adx(df, n=14):
    h, l = df["high"], df["low"]
    up = h.diff(); dn = -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = _atr(df, n).replace(0, np.nan)
    pdi = 100 * plus.ewm(alpha=1/n, min_periods=n).mean() / atr
    mdi = 100 * minus.ewm(alpha=1/n, min_periods=n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, min_periods=n).mean(), pdi, mdi


def _cci(df, n=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    md = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - tp.rolling(n).mean()) / (0.015 * md.replace(0, np.nan))


def _willr(df, n=14):
    hh = df["high"].rolling(n).max(); ll = df["low"].rolling(n).min()
    return -100 * (hh - df["close"]) / (hh - ll).replace(0, np.nan)


def _mfi(df, n=14):
    tp = (df["high"] + df["low"] + df["close"]) / 3; rmf = tp * df["volume"]
    pos = rmf.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = rmf.where(tp < tp.shift(), 0.0).rolling(n).sum().replace(0, np.nan)
    return 100 - 100 / (1 + pos / neg)


def _obv(df): return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()


def _ad(df):
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    return (clv * df["volume"]).fillna(0).cumsum()


def _cmo(s, n=14):
    d = s.diff(); up = d.clip(lower=0).rolling(n).sum(); dn = (-d.clip(upper=0)).rolling(n).sum()
    return 100 * (up - dn) / (up + dn).replace(0, np.nan)


def _tsi(s, r=25, sm=13):
    m = s.diff()
    return 100 * _ema(_ema(m, r), sm) / _ema(_ema(m.abs(), r), sm).replace(0, np.nan)


def _keltner(df, n=20, k=2):
    m = _ema(df["close"], n); a = _atr(df, n); return m, m + k*a, m - k*a


def _slope(s, n=20):
    x = np.arange(n)
    return s.rolling(n).apply(
        lambda y: np.polyfit(x, y, 1)[0] / (np.mean(y) or 1.0), raw=True)


def _supertrend_dir(df, n=10, mult=3):
    a = _atr(df, n); hl2 = (df["high"] + df["low"]) / 2
    up = (hl2 + mult * a).values; lo = (hl2 - mult * a).values
    c = df["close"].values; d = np.ones(len(c))
    for i in range(1, len(c)):
        if c[i] > up[i-1]: d[i] = 1
        elif c[i] < lo[i-1]: d[i] = -1
        else: d[i] = d[i-1]
    return pd.Series(d, index=df.index)


def _bin(score) -> pd.Series:
    """Binary vote from a directional score: +1 BUY (go long) where score > 0,
    -1 SELL (go short) where score < 0. Each agent commits to a side — no 'hold'.
    The only 0 is where the score is undefined (warm-up NaN / exact tie); at the
    live (latest) bar there is always data, so every agent gives BUY or SELL."""
    arr = np.asarray(score, dtype="float64")
    out = np.where(arr > 0, 1, np.where(arr < 0, -1, 0)).astype("int8")
    idx = score.index if hasattr(score, "index") else None
    return pd.Series(out, index=idx, dtype="int8")


# --------------------------------------------------------------------------- #
#  The 31 agents  (name, family, function: df -> vote series)
# --------------------------------------------------------------------------- #

# Every agent reduces to a directional score; _bin() turns it into BUY/SELL.

# ---- Family 1: TREND ----
def sma_cross(df): return _bin(_sma(df.close, 20) - _sma(df.close, 50))
def ema_cross(df): return _bin(_ema(df.close, 12) - _ema(df.close, 26))
def macd_cross(df): m, sig = _macd(df.close); return _bin(m - sig)
def price_200ma(df): return _bin(df.close - _sma(df.close, 200))
def adx_di(df): _, pdi, mdi = _adx(df); return _bin(pdi - mdi)
def donchian20(df):
    hi = df.high.rolling(20).max(); lo = df.low.rolling(20).min()
    return _bin(df.close - (hi + lo) / 2)
def supertrend(df): return _bin(_supertrend_dir(df))
def ichimoku(df):
    conv = (df.high.rolling(9).max() + df.low.rolling(9).min()) / 2
    base = (df.high.rolling(26).max() + df.low.rolling(26).min()) / 2
    return _bin(conv - base)

# ---- Family 2: MOMENTUM / OSCILLATORS ----
def rsi_mom(df): return _bin(_rsi(df.close, 14) - 50)
def stochastic(df): k, d = _stoch(df); return _bin(k - d)
def williams_r(df): return _bin(_willr(df) + 50)
def cci_setup(df): return _bin(_cci(df))
def roc_mom(df): return _bin(df.close.pct_change(12))
def tsi_setup(df): return _bin(_tsi(df.close))
def awesome_osc(df):
    med = (df.high + df.low) / 2; return _bin(_sma(med, 5) - _sma(med, 34))
def ts_momentum(df): return _bin(df.close.pct_change(126))
def cmo_setup(df): return _bin(_cmo(df.close))

# ---- Family 3: MEAN-REVERSION / VOLATILITY (long when oversold) ----
def bollinger_rev(df): m, u, l = _boll(df.close); return _bin(m - df.close)
def zscore_rev(df):
    z = (df.close - _sma(df.close, 20)) / df.close.rolling(20).std().replace(0, np.nan)
    return _bin(-z)
def keltner_rev(df): m, u, l = _keltner(df); return _bin(m - df.close)
def rsi2_rev(df): return _bin(50 - _rsi(df.close, 2))
def atr_bounce(df): return _bin(_sma(df.close, 20) - df.close)
def vwap_rev(df):
    vw = (df.close * df.volume).rolling(20).sum() / df.volume.rolling(20).sum().replace(0, np.nan)
    return _bin(vw - df.close)

# ---- Family 4: BREAKOUT / RANGE (long above the channel mid) ----
def turtle55(df):
    hi = df.high.rolling(55).max(); lo = df.low.rolling(55).min()
    return _bin(df.close - (hi + lo) / 2)
def range10(df):
    hi = df.high.rolling(10).max(); lo = df.low.rolling(10).min()
    return _bin(df.close - (hi + lo) / 2)
def squeeze_break(df): m, u, l = _boll(df.close); return _bin(df.close - m)

# ---- Family 5: VOLUME ----
def obv_trend(df): o = _obv(df); return _bin(o - _ema(o, 20))
def vol_surge(df): return _bin(df.close - df.close.shift())
def mfi_setup(df): return _bin(_mfi(df) - 50)
def ad_trend(df): a = _ad(df); return _bin(a - _ema(a, 20))

# ---- Family 6: STATISTICAL ----
def linreg_slope(df): return _bin(_slope(df.close, 20))


# ===== Extra indicator helpers for the expanded roster ===== #
def _wma(s, n):
    w = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def _hma(s, n):
    return _wma(2 * _wma(s, n // 2) - _wma(s, n), max(1, int(np.sqrt(n))))


def _psar(df, af=0.02, maxaf=0.2):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(c); ps = c.copy()
    bull = True; a = af; ep = l[0]
    for i in range(1, n):
        ps[i] = ps[i-1] + a * (ep - ps[i-1])
        if bull:
            if l[i] < ps[i]: bull = False; ps[i] = ep; ep = l[i]; a = af
            elif h[i] > ep: ep = h[i]; a = min(a + af, maxaf)
        else:
            if h[i] > ps[i]: bull = True; ps[i] = ep; ep = h[i]; a = af
            elif l[i] < ep: ep = l[i]; a = min(a + af, maxaf)
    return pd.Series(ps, index=df.index)


def _aroon(df, n=25):
    up = df["high"].rolling(n + 1).apply(lambda x: 100 * np.argmax(x) / n, raw=True)
    dn = df["low"].rolling(n + 1).apply(lambda x: 100 * np.argmin(x) / n, raw=True)
    return up, dn


def _vortex(df, n=14):
    tr = pd.concat([df.high - df.low, (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    vip = (df.high - df.low.shift()).abs().rolling(n).sum() / tr.rolling(n).sum().replace(0, np.nan)
    vim = (df.low - df.high.shift()).abs().rolling(n).sum() / tr.rolling(n).sum().replace(0, np.nan)
    return vip, vim


def _uo(df):
    bp = df.close - pd.concat([df.low, df.close.shift()], axis=1).min(axis=1)
    tr = (pd.concat([df.high, df.close.shift()], axis=1).max(axis=1)
          - pd.concat([df.low, df.close.shift()], axis=1).min(axis=1))
    a = lambda m: bp.rolling(m).sum() / tr.rolling(m).sum().replace(0, np.nan)
    return 100 * (4 * a(7) + 2 * a(14) + a(28)) / 7


def _fisher(df, n=10):
    m = (df.high + df.low) / 2
    mn = m.rolling(n).min(); mx = m.rolling(n).max()
    v = (2 * ((m - mn) / (mx - mn).replace(0, np.nan) - 0.5)).clip(-0.999, 0.999)
    return (0.5 * np.log((1 + v) / (1 - v))).ewm(span=3).mean()


# ---- Trend (extra) ----
def psar_agent(df): return _bin(df.close - _psar(df))
def aroon_agent(df): u, d = _aroon(df); return _bin(u - d)
def vortex_agent(df): vp, vm = _vortex(df); return _bin(vp - vm)
def hma_cross(df): h = _hma(df.close, 20); return _bin(h - h.shift())

# ---- Momentum (extra) ----
def ultimate_osc(df): return _bin(_uo(df) - 50)
def ppo_agent(df):
    p = (_ema(df.close, 12) - _ema(df.close, 26)) / _ema(df.close, 26) * 100
    return _bin(p - _ema(p, 9))
def fisher_agent(df): return _bin(_fisher(df))
def elder_ray(df): return _bin(df.close - _ema(df.close, 13))

# ---- MeanRev (extra) ----
def bb_percent(df):
    m, u, l = _boll(df.close); b = (df.close - l) / (u - l).replace(0, np.nan)
    return _bin(0.5 - b)
def pct_from_ma(df): return _bin(_sma(df.close, 50) - df.close)

# ---- Volume (extra) ----
def cmf_agent(df):
    rng = (df.high - df.low).replace(0, np.nan)
    mfv = ((df.close - df.low) - (df.high - df.close)) / rng * df.volume
    cmf = mfv.rolling(20).sum() / df.volume.rolling(20).sum().replace(0, np.nan)
    return _bin(cmf)
def force_index(df): return _bin(_ema((df.close - df.close.shift()) * df.volume, 13))

# ---- Statistical (extra) ----
def ret_zscore(df):
    r = df.close.pct_change()
    z = (r - r.rolling(20).mean()) / r.rolling(20).std().replace(0, np.nan)
    return _bin(-z)
def hh_hl_structure(df):
    hh = df.high.rolling(10).max(); ll = df.low.rolling(10).min()
    return _bin((hh - hh.shift(5)) + (ll - ll.shift(5)))


AGENTS: List[Tuple[str, str, Callable]] = [
    ("SMA cross", "Trend", sma_cross), ("EMA cross", "Trend", ema_cross),
    ("MACD cross", "Trend", macd_cross), ("Price vs 200MA", "Trend", price_200ma),
    ("ADX + DI", "Trend", adx_di), ("Donchian 20", "Trend", donchian20),
    ("Supertrend", "Trend", supertrend), ("Ichimoku", "Trend", ichimoku),
    ("RSI momentum", "Momentum", rsi_mom), ("Stochastic", "Momentum", stochastic),
    ("Williams %R", "Momentum", williams_r), ("CCI", "Momentum", cci_setup),
    ("ROC", "Momentum", roc_mom), ("TSI", "Momentum", tsi_setup),
    ("Awesome Osc", "Momentum", awesome_osc), ("12M momentum", "Momentum", ts_momentum),
    ("CMO", "Momentum", cmo_setup),
    ("Bollinger revert", "MeanRev", bollinger_rev), ("Z-score revert", "MeanRev", zscore_rev),
    ("Keltner revert", "MeanRev", keltner_rev), ("RSI(2) revert", "MeanRev", rsi2_rev),
    ("ATR bounce", "MeanRev", atr_bounce), ("VWAP revert", "MeanRev", vwap_rev),
    ("Turtle 55", "Breakout", turtle55), ("Range 10", "Breakout", range10),
    ("Squeeze break", "Breakout", squeeze_break),
    ("OBV trend", "Volume", obv_trend), ("Volume surge", "Volume", vol_surge),
    ("MFI", "Volume", mfi_setup), ("A/D trend", "Volume", ad_trend),
    ("LinReg slope", "Statistical", linreg_slope),
    # ---- expanded roster ----
    ("Parabolic SAR", "Trend", psar_agent), ("Aroon", "Trend", aroon_agent),
    ("Vortex", "Trend", vortex_agent), ("HMA cross", "Trend", hma_cross),
    ("Ultimate Osc", "Momentum", ultimate_osc), ("PPO", "Momentum", ppo_agent),
    ("Fisher", "Momentum", fisher_agent), ("Elder Ray", "Momentum", elder_ray),
    ("Bollinger %B", "MeanRev", bb_percent), ("% from 50MA", "MeanRev", pct_from_ma),
    ("Chaikin MF", "Volume", cmf_agent), ("Force Index", "Volume", force_index),
    ("Return z-score", "Statistical", ret_zscore),
    ("HH/HL structure", "Statistical", hh_hl_structure),
]
assert len(AGENTS) == 45


# --------------------------------------------------------------------------- #
#  Committee
# --------------------------------------------------------------------------- #

class Committee:
    def __init__(self, cfg: dict | None = None) -> None:
        c = (cfg or {}).get("committee", {})
        self.threshold = int(c.get("threshold", 25))

    def vote_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame of per-bar votes, one column per agent."""
        df = df.rename(columns=str.lower)
        out = {}
        for name, _fam, fn in AGENTS:
            try:
                out[name] = fn(df).reindex(df.index).fillna(0).astype("int8")
            except Exception:
                out[name] = pd.Series(0, index=df.index, dtype="int8")
        return pd.DataFrame(out, index=df.index)

    def decision_series(self, df: pd.DataFrame) -> pd.Series:
        M = self.vote_matrix(df)
        longs = (M == 1).sum(axis=1); shorts = (M == -1).sum(axis=1)
        d = pd.Series(0, index=M.index, dtype="int8")
        d[longs >= self.threshold] = 1
        d[shorts >= self.threshold] = -1
        return d

    def latest(self, df: pd.DataFrame) -> dict:
        """Per-agent latest vote + tallies + verdict for the most recent bar."""
        M = self.vote_matrix(df)
        last = M.iloc[-1]
        agents = [{"name": n, "family": f, "vote": int(last[n])}
                  for n, f, _ in AGENTS]
        longs = int((last == 1).sum()); shorts = int((last == -1).sum())
        flats = len(AGENTS) - longs - shorts
        if longs >= self.threshold:
            verdict, side = f"LONG — {longs}/{len(AGENTS)} agree", "long"
        elif shorts >= self.threshold:
            verdict, side = f"SHORT — {shorts}/{len(AGENTS)} agree", "short"
        else:
            verdict, side = (f"NO TRADE — best is {max(longs, shorts)}/{len(AGENTS)} "
                             f"(need {self.threshold})", "flat")
        # family breakdown of net direction
        fam: Dict[str, int] = {}
        for n, f, _ in AGENTS:
            fam[f] = fam.get(f, 0) + int(last[n])
        return {"agents": agents, "long": longs, "short": shorts, "flat": flats,
                "threshold": self.threshold, "verdict": verdict, "side": side,
                "families": fam, "total": len(AGENTS)}


def backtest_from_votes(M: pd.DataFrame, close: pd.Series, threshold: int,
                        ppy: int, tc: float, allow_short: bool = False):
    """Fast Sharpe for a given consensus threshold, reusing a precomputed vote
    matrix (so the strategy trainer can sweep thresholds cheaply)."""
    longs = (M == 1).sum(axis=1); shorts = (M == -1).sum(axis=1)
    d = pd.Series(0, index=M.index, dtype="int8")
    d[longs >= threshold] = 1
    d[shorts >= threshold] = -1
    pos = d.shift(1).fillna(0).to_numpy().astype(float)
    if not allow_short:
        pos = np.clip(pos, 0, 1)
    ret = close.reindex(M.index).pct_change().fillna(0).to_numpy()
    turn = np.abs(np.diff(np.concatenate([[0], pos])))
    strat = pos * ret - turn * tc
    return float(sharpe_ratio(strat, periods_per_year=ppy))


def committee_backtest(df: pd.DataFrame, ppy: int, tc: float,
                       cfg: dict | None = None, allow_short: bool = False) -> dict:
    """Backtest the consensus: long on LONG, (short on SHORT if allowed), else flat."""
    com = Committee(cfg)
    d = com.decision_series(df).shift(1).fillna(0)  # act next bar (no lookahead)
    close = df["close"].rename("c") if "close" in df else df["Close"]
    ret = close.pct_change().fillna(0).to_numpy()
    pos = d.to_numpy().astype(float)
    if not allow_short:
        pos = np.clip(pos, 0, 1)
    turn = np.abs(np.diff(np.concatenate([[0], pos])))
    strat = pos * ret - turn * tc
    bh = ret
    eq_s = np.cumprod(1 + strat); eq_b = np.cumprod(1 + bh)
    trades = int((np.diff(pos) != 0).sum())
    return {
        "bars": len(strat), "trades": trades,
        "strat": {"total": float(eq_s[-1] - 1), "sharpe": float(sharpe_ratio(strat, periods_per_year=ppy)),
                  "max_dd": float(max_drawdown(eq_s))},
        "bh": {"total": float(eq_b[-1] - 1), "sharpe": float(sharpe_ratio(bh, periods_per_year=ppy)),
               "max_dd": float(max_drawdown(eq_b))},
    }
