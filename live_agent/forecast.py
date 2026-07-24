"""
forecast.py — Probabilistic next-day forecaster for indices.

You cannot predict tomorrow's *direction* (daily index returns are ~unpredictable).
You CAN estimate the *distribution* of tomorrow's move, because volatility clusters.
This module does exactly that, and — crucially — proves whether the resulting
probabilities are trustworthy.

Method: Filtered Historical Simulation (FHS), a standard, robust approach.
    1. Forecast tomorrow's volatility with an EWMA (RiskMetrics) recursion that
       uses only past data.
    2. Standardise past returns by the volatility that was forecast *for their
       own day*: eps_i = (r_i - mu_i) / sigma_i. These residuals are roughly
       i.i.d. even though the raw returns are not.
    3. Tomorrow's return distribution = mu + sigma_tomorrow * {past eps}. Read
       percentiles and probabilities straight off this simulated sample.

Three things it reports per symbol:
    • FORECAST  — tomorrow's probabilistic range in points + key probabilities.
    • CALIBRATION — does reality match the forecast? (coverage, PIT, vs a naive
      Gaussian baseline). THIS is the part that says whether to trust it.
    • VOL-TARGET BACKTEST — an honest tradable use: scale exposure by 1/forecast
      vol to hold risk constant. Improves risk-adjusted return without predicting
      direction.

    python forecast.py --symbol "^NSEI"
    python forecast.py --all          # every symbol in forecast_universe

Options note: a faithful options backtest needs historical option chains + implied
vol, which free data lacks. The vol forecast here is the *input* an options trader
would compare against implied vol — but pricing real option P&L needs paid data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.metrics import max_drawdown, sharpe_ratio  # noqa: E402

HERE = Path(__file__).resolve().parent
TRADING_DAYS = 252
logger = logging.getLogger("forecast")


# --------------------------------------------------------------------------- #
#  Data
# --------------------------------------------------------------------------- #

def fetch_close(symbol: str) -> pd.Series:
    """Max-history daily close for a symbol, timezone-naive, oldest first.

    Routes crypto symbols to Binance, everything else to yfinance.
    """
    try:
        import binance_feed
        if binance_feed.is_crypto(symbol):
            frame = binance_feed.get_ohlcv(symbol, "1d", lookback_days=4000)
            if frame.empty:
                raise SystemExit(f"No Binance data for {symbol}.")
            return frame["close"].dropna()
    except ImportError:
        pass
    raw = yf.Ticker(symbol).history(period="max", interval="1d", auto_adjust=True)
    if raw.empty:
        raise SystemExit(f"No data for {symbol}.")
    close = raw["Close"].dropna()
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    return close


# --------------------------------------------------------------------------- #
#  Core: EWMA vol + standardised residuals (all strictly out-of-sample)
# --------------------------------------------------------------------------- #

def ewma_sigma(logret: np.ndarray, lam: float, warmup: int) -> np.ndarray:
    """One-step-ahead EWMA volatility forecast.

    ``sigma[t]`` is the vol forecast FOR day t, made using returns up to t-1 —
    so it never peeks at the day it predicts. Seeded from the warmup variance.
    """
    n = len(logret)
    sigma = np.full(n, np.nan)
    var = float(np.var(logret[:warmup]))
    for t in range(warmup, n):
        var = lam * var + (1.0 - lam) * logret[t - 1] ** 2
        sigma[t] = np.sqrt(var)
    return sigma


def _forecast_sample(
    logret: np.ndarray, sigma: np.ndarray, t: int, fc: dict
) -> tuple[float, np.ndarray]:
    """Return (mu_t, simulated next-day log-return sample) for day ``t``.

    Uses only information available before day ``t``: a trailing mean for drift
    and a trailing window of standardised residuals for the shock distribution.
    """
    rw = int(fc["resid_window"])
    dw = int(fc["drift_window"])
    lo = max(int(fc["warmup"]), t - rw)
    hist_r = logret[max(0, t - dw):t]
    mu = float(np.mean(hist_r)) if len(hist_r) else 0.0
    # Standardised residuals for past days (each by its own forecast sigma).
    past = np.arange(lo, t)
    eps = (logret[past] - mu) / sigma[past]
    eps = eps[np.isfinite(eps)]
    return mu, mu + sigma[t] * eps


# --------------------------------------------------------------------------- #
#  Tomorrow's forecast
# --------------------------------------------------------------------------- #

def forecast_tomorrow(close: pd.Series, fc: dict) -> Dict:
    """Probabilistic forecast for the next session given all history."""
    logret = np.log(close.to_numpy()[1:] / close.to_numpy()[:-1])
    sigma = ewma_sigma(logret, float(fc["ewma_lambda"]), int(fc["warmup"]))
    t = len(logret)  # "tomorrow" — sigma for it uses the last observed return
    var = float(np.var(logret[:int(fc["warmup"])]))
    # roll EWMA one more step to get sigma for the unobserved next day
    lam = float(fc["ewma_lambda"])
    v = float(pd.Series(logret ** 2).ewm(alpha=1 - lam, adjust=False).mean().iloc[-1])
    sigma_next = float(np.sqrt(v))
    sigma_full = np.append(sigma, sigma_next)
    logret_pad = np.append(logret, 0.0)
    mu, sample = _forecast_sample(logret_pad, sigma_full, t, fc)

    level = float(close.iloc[-1])
    pct = np.exp(sample) - 1.0           # simple-return draws
    pts = level * pct                    # point moves
    q = lambda p: float(np.percentile(pts, p))
    return {
        "level": level,
        "sigma_next_ann": sigma_next * np.sqrt(TRADING_DAYS),
        "exp_abs_move_pts": float(np.mean(np.abs(pts))),
        "p_up": float(np.mean(pct > 0)),
        "p_down_gt1pct": float(np.mean(pct < -0.01)),
        "p_up_gt1pct": float(np.mean(pct > 0.01)),
        "p05": q(5), "p25": q(25), "p50": q(50), "p75": q(75), "p95": q(95),
    }


# --------------------------------------------------------------------------- #
#  Calibration backtest — the part that earns trust
# --------------------------------------------------------------------------- #

def calibrate(close: pd.Series, fc: dict) -> Dict:
    """Walk history; check realised moves against each day's forecast.

    Reports interval coverage (should match nominal 50%/90%), mean PIT (should
    be ~0.5), and the same coverage for a naive Gaussian baseline for contrast.
    """
    logret = np.log(close.to_numpy()[1:] / close.to_numpy()[:-1])
    sigma = ewma_sigma(logret, float(fc["ewma_lambda"]), int(fc["warmup"]))

    pit, in50, in90, g_in90 = [], [], [], []
    from math import erf, sqrt
    ncdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))

    for t in range(int(fc["warmup"]) + int(fc["drift_window"]), len(logret)):
        if not np.isfinite(sigma[t]):
            continue
        mu, sample = _forecast_sample(logret, sigma, t, fc)
        if sample.size < 50:
            continue
        r = logret[t]
        pit.append(float(np.mean(sample <= r)))
        in50.append(bool(np.percentile(sample, 25) <= r <= np.percentile(sample, 75)))
        in90.append(bool(np.percentile(sample, 5) <= r <= np.percentile(sample, 95)))
        # Gaussian baseline: same mu/sigma, normal shock.
        z = (r - mu) / sigma[t]
        g_in90.append(bool(abs(z) <= 1.645))

    return {
        "n": len(pit),
        "cover50": float(np.mean(in50)),
        "cover90": float(np.mean(in90)),
        "mean_pit": float(np.mean(pit)),
        "gauss_cover90": float(np.mean(g_in90)),
        "pit": np.array(pit),
    }


# --------------------------------------------------------------------------- #
#  Vol-target backtest — an honest tradable use of the vol forecast
# --------------------------------------------------------------------------- #

def voltarget_backtest(close: pd.Series, fc: dict) -> Dict:
    """Scale long exposure by target_vol / forecast_vol; compare to buy & hold."""
    price = close.to_numpy()
    logret = np.log(price[1:] / price[:-1])
    simret = price[1:] / price[:-1] - 1.0
    sigma = ewma_sigma(logret, float(fc["ewma_lambda"]), int(fc["warmup"]))
    ann = sigma * np.sqrt(TRADING_DAYS)

    tgt = float(fc["vol_target"])
    wmax = float(fc["max_leverage"])
    w = np.clip(tgt / ann, 0.0, wmax)        # position size known before the day
    valid = np.isfinite(w)
    strat = w[valid] * simret[valid]
    bh = simret[valid]

    eq_s = np.cumprod(1.0 + strat)
    eq_b = np.cumprod(1.0 + bh)
    return {
        "n": int(valid.sum()),
        "strat_sharpe": float(sharpe_ratio(strat, periods_per_year=TRADING_DAYS)),
        "bh_sharpe": float(sharpe_ratio(bh, periods_per_year=TRADING_DAYS)),
        "strat_dd": float(max_drawdown(eq_s)),
        "bh_dd": float(max_drawdown(eq_b)),
        "strat_total": float(eq_s[-1] - 1.0),
        "bh_total": float(eq_b[-1] - 1.0),
    }


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #

def run_symbol(symbol: str, fc: dict, plots_dir: Path) -> None:
    close = fetch_close(symbol)
    span = f"{close.index[0]:%Y-%m-%d} → {close.index[-1]:%Y-%m-%d}"
    f = forecast_tomorrow(close, fc)
    cal = calibrate(close, fc)
    vt = voltarget_backtest(close, fc)

    print("\n" + "=" * 74)
    print(f"  {symbol}   ({len(close)} daily bars,  {span})")
    print("=" * 74)
    print(f"  TOMORROW'S PROBABILISTIC FORECAST   (last close {f['level']:,.1f})")
    print(f"    Forecast annualised vol : {f['sigma_next_ann']:.1%}")
    print(f"    Expected |move|         : {f['exp_abs_move_pts']:,.0f} pts")
    print(f"    P(up)                   : {f['p_up']:.0%}     "
          f"P(up >1%): {f['p_up_gt1pct']:.0%}   P(down >1%): {f['p_down_gt1pct']:.0%}")
    print(f"    Next-day level range:")
    print(f"       5% : {f['level'] + f['p05']:>10,.0f}   ({f['p05']:+,.0f} pts)")
    print(f"      50% : {f['level'] + f['p50']:>10,.0f}   ({f['p50']:+,.0f} pts)")
    print(f"      95% : {f['level'] + f['p95']:>10,.0f}   ({f['p95']:+,.0f} pts)")
    print("-" * 74)
    print(f"  CALIBRATION  (over {cal['n']:,} days — are the probabilities real?)")
    print(f"    50% interval coverage : {cal['cover50']:.1%}   (target 50%)")
    print(f"    90% interval coverage : {cal['cover90']:.1%}   (target 90%)   "
          f"[Gaussian baseline: {cal['gauss_cover90']:.1%}]")
    print(f"    Mean PIT              : {cal['mean_pit']:.3f}   (target 0.500)")
    print("-" * 74)
    print(f"  VOL-TARGET BACKTEST  (tradable use of the vol forecast, {vt['n']:,} days)")
    print(f"    {'':<16}{'Sharpe':>9}{'MaxDD':>9}{'TotalRet':>12}")
    print(f"    {'Vol-targeted':<16}{vt['strat_sharpe']:>9.2f}"
          f"{vt['strat_dd']:>8.1%}{vt['strat_total']:>11.1%}")
    print(f"    {'Buy & hold':<16}{vt['bh_sharpe']:>9.2f}"
          f"{vt['bh_dd']:>8.1%}{vt['bh_total']:>11.1%}")
    print("=" * 74 + "\n")

    # PIT histogram — flat = well calibrated.
    plots_dir.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("^", "").replace("/", "_")
    plt.figure(figsize=(6, 4))
    plt.hist(cal["pit"], bins=20, range=(0, 1), color="steelblue",
             edgecolor="white", alpha=0.85)
    plt.axhline(cal["n"] / 20, color="crimson", ls="--", lw=1.2, label="ideal (uniform)")
    plt.title(f"PIT calibration — {symbol}\n(flat = trustworthy probabilities)")
    plt.xlabel("Probability Integral Transform"); plt.ylabel("count")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir / f"forecast_pit_{safe}.png", dpi=120)
    plt.close()


def main() -> None:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Probabilistic next-day forecaster")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--symbol", help="Single symbol (e.g. '^NSEI').")
    parser.add_argument("--all", action="store_true",
                        help="Run every symbol in forecast_universe.")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    fc = cfg["forecast"]
    plots = HERE / "backtest_results"

    if args.all:
        for sym in cfg["forecast_universe"]:
            try:
                run_symbol(sym, fc, plots)
            except SystemExit as exc:
                logger.warning("Skipping %s: %s", sym, exc)
    else:
        run_symbol(args.symbol or "^NSEI", fc, plots)


if __name__ == "__main__":
    main()
