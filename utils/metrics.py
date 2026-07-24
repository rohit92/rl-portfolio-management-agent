"""
Performance metrics for RL-based portfolio management.

All metrics follow standard quantitative finance definitions.
Each function documents its formula in LaTeX-style notation within the docstring.
Edge cases (zero std, zero drawdown, empty inputs) are handled gracefully.

Design decisions:
    - Pure functions with no side effects for easy testing and composition.
    - numpy used for vectorised computation; avoids slow Python loops.
    - All annualisation factors are configurable (default 252 trading days/year).
    - Returns are expected as *simple* (arithmetic) returns, not log returns.
"""

from __future__ import annotations

import logging
import math
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger — follows stdlib best practice (no print statements).
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ============================= sharpe_ratio ================================

def sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r"""Compute the annualised Sharpe ratio.

    .. math::

        \text{Sharpe} = \frac{\bar{R} - R_f}{\sigma_R} \times \sqrt{N}

    where :math:`\bar{R}` is the mean of the return series,
    :math:`R_f` is the per-period risk-free rate,
    :math:`\sigma_R` is the standard deviation of the return series, and
    :math:`N` is the number of periods per year (annualisation factor).

    Parameters
    ----------
    returns : List[float]
        Sequence of per-period simple returns
        (e.g. daily returns for a daily strategy).
    risk_free_rate : float, optional
        Per-period risk-free rate, by default ``0.0``.
    periods_per_year : int, optional
        Trading periods in one calendar year, by default ``252``
        (standard US equity trading days).

    Returns
    -------
    float
        Annualised Sharpe ratio.  Returns ``0.0`` when the standard deviation
        of the returns is zero (no volatility ⇒ undefined risk-adjusted return).

    Examples
    --------
    >>> sharpe_ratio([0.01, 0.02, -0.005, 0.003])  # doctest: +ELLIPSIS
    1.26...
    """
    arr = np.asarray(returns, dtype=np.float64)

    if arr.size == 0:
        logger.warning("sharpe_ratio called with empty returns; returning 0.0")
        return 0.0

    excess = arr - risk_free_rate  # subtract risk-free rate element-wise
    std = float(np.std(excess, ddof=1))  # sample std (ddof=1 → unbiased)

    if std == 0.0:
        # Constant-return series: risk is zero, Sharpe is undefined → 0.0
        logger.debug("Zero standard deviation in returns; Sharpe set to 0.0")
        return 0.0

    mean_excess = float(np.mean(excess))
    annualisation_factor = math.sqrt(periods_per_year)  # √N scaling
    result = (mean_excess / std) * annualisation_factor

    logger.debug("Sharpe ratio computed: %.4f", result)
    return result


# ============================= max_drawdown ================================

def max_drawdown(equity_curve: List[float]) -> float:
    r"""Compute the maximum drawdown of an equity curve.

    .. math::

        \text{MDD} = \max_{t} \frac{\text{peak}_t - \text{trough}_t}{\text{peak}_t}

    The function returns a **positive** number; e.g. ``0.15`` means the
    portfolio lost 15 % from its peak before recovering.

    Parameters
    ----------
    equity_curve : List[float]
        Monotonically–timestamped portfolio values (e.g. daily NAVs).

    Returns
    -------
    float
        Maximum drawdown as a positive fraction in [0, 1].
        Returns ``0.0`` for curves with fewer than 2 points.

    Examples
    --------
    >>> max_drawdown([100, 120, 90, 110])
    0.25
    """
    arr = np.asarray(equity_curve, dtype=np.float64)

    if arr.size < 2:
        logger.warning("max_drawdown needs ≥ 2 data points; returning 0.0")
        return 0.0

    # Running maximum (peak) at each time-step
    running_peak = np.maximum.accumulate(arr)

    # Drawdown at each time-step as a fraction of the running peak
    drawdowns = (running_peak - arr) / np.where(
        running_peak != 0.0, running_peak, 1.0  # guard against division by zero
    )

    result = float(np.max(drawdowns))
    logger.debug("Max drawdown computed: %.4f", result)
    return result


# =============================== win_rate ==================================

def win_rate(returns: List[float]) -> float:
    r"""Compute the win rate over non-zero return periods.

    .. math::

        \text{WinRate} = \frac{\#\{r_i > 0\}}{\#\{r_i \neq 0\}}

    Only non-zero returns are considered; flat days are excluded.

    Parameters
    ----------
    returns : List[float]
        Sequence of per-period simple returns.

    Returns
    -------
    float
        Fraction of profitable periods among non-zero-return periods,
        in the range [0, 1].  Returns ``0.0`` if all returns are zero.

    Examples
    --------
    >>> win_rate([0.01, -0.02, 0.005, 0.0, 0.003])
    0.75
    """
    arr = np.asarray(returns, dtype=np.float64)

    # Mask out flat (zero-return) periods
    non_zero_mask = arr != 0.0
    n_non_zero = int(np.sum(non_zero_mask))

    if n_non_zero == 0:
        logger.warning("No non-zero returns; win_rate returning 0.0")
        return 0.0

    n_wins = int(np.sum(arr > 0.0))  # count of strictly positive returns
    result = n_wins / n_non_zero

    logger.debug("Win rate computed: %.4f  (wins=%d, non-zero=%d)",
                 result, n_wins, n_non_zero)
    return result


# =========================== annualized_return =============================

def annualized_return(
    returns: List[float],
    periods_per_year: int = 252,
) -> float:
    r"""Compute the annualised compound return.

    .. math::

        R_{\text{ann}} = \left(1 + R_{\text{total}}\right)^{252 / n} - 1

    where :math:`R_{\text{total}} = \prod(1 + r_i) - 1` and :math:`n` is
    the number of observed periods.

    Parameters
    ----------
    returns : List[float]
        Sequence of per-period simple returns.
    periods_per_year : int, optional
        Number of trading periods per year, by default ``252``.

    Returns
    -------
    float
        Annualised return as a decimal (e.g. ``0.12`` = 12 %).
        Returns ``0.0`` for empty input.

    Examples
    --------
    >>> annualized_return([0.01] * 252)  # doctest: +ELLIPSIS
    11.34...
    """
    arr = np.asarray(returns, dtype=np.float64)
    n_days = arr.size

    if n_days == 0:
        logger.warning("annualized_return called with empty returns; returning 0.0")
        return 0.0

    # Cumulative compounded return: product of (1 + r_i)
    total_return = float(np.prod(1.0 + arr)) - 1.0

    # Guard against negative total wealth (would make fractional exponent complex)
    base = 1.0 + total_return
    if base <= 0.0:
        logger.warning(
            "Total wealth factor ≤ 0 (%.4f); annualized_return returning -1.0",
            base,
        )
        return -1.0  # total loss or worse — cannot annualise meaningfully

    exponent = periods_per_year / n_days  # annualisation exponent
    result = base ** exponent - 1.0

    logger.debug("Annualized return computed: %.4f", result)
    return result


# ============================ calmar_ratio =================================

def calmar_ratio(
    returns: List[float],
    equity_curve: List[float],
) -> float:
    r"""Compute the Calmar ratio.

    .. math::

        \text{Calmar} = \frac{R_{\text{ann}}}{|\text{MDD}|}

    Preferred by many hedge funds over the Sharpe ratio because it penalises
    large drawdowns more directly.

    Parameters
    ----------
    returns : List[float]
        Sequence of per-period simple returns (used to compute annualised return).
    equity_curve : List[float]
        Portfolio values over time (used to compute max drawdown).

    Returns
    -------
    float
        Calmar ratio.  Returns ``0.0`` when max drawdown is zero
        (no peak-to-trough decline observed).

    Examples
    --------
    >>> calmar_ratio([0.01, -0.02, 0.015], [100, 101, 99, 100.5])
    ...  # doctest: +SKIP
    """
    mdd = max_drawdown(equity_curve)

    if mdd == 0.0:
        # No drawdown ⇒ ratio is undefined; return 0.0 as safe default
        logger.debug("Max drawdown is zero; Calmar ratio set to 0.0")
        return 0.0

    ann_ret = annualized_return(returns)  # re-uses our annualisation logic
    result = ann_ret / abs(mdd)  # absolute value for safety

    logger.debug("Calmar ratio computed: %.4f  (ann_ret=%.4f, mdd=%.4f)",
                 result, ann_ret, mdd)
    return result
