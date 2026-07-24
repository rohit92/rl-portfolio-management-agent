"""
data_feed.py — Live market data + technical feature engineering.

The agent's *eyes*. Pulls daily OHLCV bars from Yahoo Finance (free, no API key)
and augments each symbol's frame with the indicators the ensemble consumes:
trailing returns, realised volatility, moving averages, a mean-reversion
z-score, and RSI.

Data source rationale
---------------------
We deliberately decouple *data* from *execution*: data comes from yfinance
(free, reliable for daily bars) while order execution goes through Alpaca paper
(or the local SimBroker). For a once-a-day strategy, end-of-day Yahoo bars are
more than adequate and avoid coupling to any broker's data entitlements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import binance_feed
except Exception:  # binance_feed optional — yfinance still works without it
    binance_feed = None

logger = logging.getLogger(__name__)


@dataclass
class SymbolData:
    """Per-symbol market frame plus convenience accessors.

    Attributes
    ----------
    symbol : str
        Ticker symbol.
    frame : pd.DataFrame
        Daily OHLCV with engineered indicator columns. Indexed by date,
        oldest first.
    """

    symbol: str
    frame: pd.DataFrame

    @property
    def last_close(self) -> float:
        """Most recent closing price."""
        return float(self.frame["close"].iloc[-1])

    @property
    def latest(self) -> pd.Series:
        """Most recent fully-populated row (all indicators present)."""
        return self.frame.iloc[-1]


def _compute_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach technical indicator columns to a raw OHLCV frame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV with lowercase columns: open, high, low, close, volume.
    cfg : dict
        The ``strategy`` + ``risk`` config sections (for window lengths).

    Returns
    -------
    pd.DataFrame
        Same frame with indicator columns appended. Leading rows that lack
        enough history for the longest window are dropped.
    """
    strat = cfg["strategy"]
    risk = cfg["risk"]
    out = df.copy()

    close = out["close"]

    # --- Returns & realised volatility -------------------------------------
    out["daily_return"] = close.pct_change()
    out["vol"] = (
        out["daily_return"].rolling(risk["vol_target_window"]).std()
    )

    # --- Momentum: skip-adjusted trailing return ---------------------------
    # (price_{t-skip} / price_{t-skip-lookback}) - 1. Skipping the most recent
    # few days avoids the well-documented short-term reversal effect.
    look = int(strat["momentum_lookback"])
    skip = int(strat["momentum_skip"])
    out["momentum_raw"] = (
        close.shift(skip) / close.shift(skip + look) - 1.0
    )

    # --- Mean-reversion z-score vs a moving average ------------------------
    win = int(strat["reversion_window"])
    ma = close.rolling(win).mean()
    sd = close.rolling(win).std()
    out["reversion_z"] = (close - ma) / sd.replace(0.0, np.nan)

    # --- Trend: fast vs slow MA ratio --------------------------------------
    fast = close.rolling(int(strat["trend_fast"])).mean()
    slow = close.rolling(int(strat["trend_slow"])).mean()
    out["trend_ratio"] = fast / slow - 1.0

    # --- RSI(14): supplementary momentum/exhaustion gauge ------------------
    out["rsi"] = _rsi(close, period=14)

    # Drop the warm-up rows where the longest window is not yet defined.
    out = out.dropna(
        subset=["momentum_raw", "reversion_z", "trend_ratio", "vol"]
    )
    return out


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Returns a 0–100 series; 50 is neutral, >70 overbought, <30 oversold.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance output to a clean lowercase OHLCV frame.

    yfinance returns Title-case columns (and occasionally a MultiIndex when
    multiple tickers are requested). We normalise to single-level lowercase.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Collapse to the first level (the OHLCV field names).
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    # Normalise to a timezone-naive index so frames from different fetches
    # (symbols, SPY benchmark) align cleanly without tz-dtype clashes.
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    keep = ["open", "high", "low", "close", "volume"]
    return df[[c for c in keep if c in df.columns]]


# yfinance caps how far back intraday bars go. Clamp the request window so we
# don't silently get an empty frame for short-interval presets.
_INTRADAY_MAX_DAYS = {
    "1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "90m": 59,
    "60m": 729, "1h": 729,
}


def fetch(
    symbols: List[str],
    cfg: dict,
    lookback_days: int = 400,
    interval: str | None = None,
) -> Dict[str, SymbolData]:
    """Fetch and feature-engineer bars for each symbol at the given interval.

    Parameters
    ----------
    symbols : list of str
        Tickers to fetch (US, crypto like ``BTC-USD``, or NSE like ``TCS.NS``).
    cfg : dict
        Full config dict (uses ``strategy`` and ``risk`` sections; ``interval``
        is read from here if not passed explicitly).
    lookback_days : int
        Calendar days of history to request. Clamped for intraday intervals.
    interval : str, optional
        Bar size (``1d``, ``1wk``, ``1h``, ``30m`` ...). Defaults to
        ``cfg['interval']`` then ``1d``.

    Returns
    -------
    dict[str, SymbolData]
        Mapping of symbol -> engineered data. Symbols that fail to download
        or lack sufficient history are skipped with a warning.
    """
    interval = interval or cfg.get("interval", "1d")
    cap = _INTRADAY_MAX_DAYS.get(interval)
    if cap is not None and lookback_days > cap:
        logger.info(
            "Interval %s caps history at %d days (requested %d) — clamping.",
            interval, cap, lookback_days,
        )
        lookback_days = cap

    # Explicit start/end is more reliable than yfinance's period strings,
    # which silently ignore arbitrary "Nd" values.
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=lookback_days)
    out: Dict[str, SymbolData] = {}

    for sym in symbols:
        # Route crypto (BTCUSDT / BTC-USD / bare bases) to Binance, everything
        # else (US equities, .NS stocks, ^ indices) to yfinance.
        is_cx = binance_feed is not None and binance_feed.is_crypto(sym)
        try:
            if is_cx:
                frame = binance_feed.get_ohlcv(
                    sym, interval=interval, lookback_days=lookback_days
                )
            else:
                raw = yf.Ticker(sym).history(
                    start=start, end=end, interval=interval, auto_adjust=True
                )
                frame = (_normalise_columns(raw)
                         if raw is not None and not raw.empty else None)
        except Exception as exc:  # network / symbol errors
            logger.warning("Failed to fetch %s: %s", sym, exc)
            continue

        if frame is None or frame.empty:
            logger.warning("No data returned for %s — skipping.", sym)
            continue

        feat = _compute_features(frame, cfg)
        if feat.empty:
            logger.warning(
                "%s has insufficient history for indicators — skipping.", sym
            )
            continue

        out[sym] = SymbolData(symbol=sym, frame=feat)
        logger.debug(
            "%s: %d usable rows, last close %.2f",
            sym,
            len(feat),
            out[sym].last_close,
        )

    logger.info("Fetched data for %d/%d symbols.", len(out), len(symbols))
    return out


def latest_prices(data: Dict[str, SymbolData]) -> Dict[str, float]:
    """Extract the most recent close price for each symbol."""
    return {sym: sd.last_close for sym, sd in data.items()}
