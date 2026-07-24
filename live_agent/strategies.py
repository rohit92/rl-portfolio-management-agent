"""
strategies.py — The ensemble "brain".

Instead of a single (and, as the parent project showed, fragile) PPO policy,
the live agent blends three orthogonal, well-understood signals. Each scores a
symbol in [-1, +1]; the composite is their weighted average. Diversifying across
*signals* is the same idea as diversifying across *assets*: when one edge decays,
the others carry the portfolio.

Signals
-------
1. Momentum      — skip-adjusted trailing return. Rides persistent trends.
2. Mean-reversion— negative z-score vs a moving average. Buys oversold names.
3. Trend filter  — fast-vs-slow MA regime. Confirms the broader direction.

These are intentionally *complementary*: momentum and mean-reversion pull in
opposite directions on short horizons, so blending them dampens whipsaw while
the trend filter keeps the book on the right side of the primary regime.

Extensibility
-------------
``Ensemble`` is signal-agnostic — it just calls each registered ``Signal``.
To plug the parent project's trained PPO model in as a fourth signal, wrap its
deterministic action (Buy=+1 / Hold=0 / Sell=-1) in a ``Signal`` and add it to
the registry with a weight in config.yaml. No other code changes needed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict

import pandas as pd

from data_feed import SymbolData

logger = logging.getLogger(__name__)


# Signal scales: the raw input magnitude that saturates each squashed signal.
# These are the single source of truth, shared by the live scoring functions
# below and the vectorised ``signal_components`` used by the auto-learner — so
# the thing we optimise offline is exactly the thing we trade live.
MOMENTUM_SCALE = 0.20    # a ~20% trailing move is a strong momentum reading
REVERSION_SCALE = 2.0    # a 2-sigma deviation is a strong reversion reading
TREND_SCALE = 0.05       # a 5% fast/slow MA separation is a strong trend reading


def _squash(x: float, scale: float) -> float:
    """Map an unbounded raw signal to [-1, 1] smoothly via tanh.

    ``scale`` sets the input magnitude that saturates the output to ~±0.76
    (tanh(1)). Larger scale => gentler, more linear response.
    """
    if x is None or math.isnan(x):
        return 0.0
    return math.tanh(x / scale)


def signal_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Vectorised per-day signal values for a full feature frame.

    Returns a DataFrame (indexed like ``frame``) with one column per signal,
    each in [-1, 1]. This is the batch equivalent of the scalar signal
    functions below and MUST stay numerically identical to them — both paths
    use the same scale constants.
    """
    import numpy as np
    return pd.DataFrame(
        {
            "momentum": np.tanh(frame["momentum_raw"] / MOMENTUM_SCALE),
            "mean_reversion": np.tanh(-frame["reversion_z"] / REVERSION_SCALE),
            "trend": np.tanh(frame["trend_ratio"] / TREND_SCALE),
        },
        index=frame.index,
    )


# --------------------------------------------------------------------------- #
#  Individual signals: each maps a SymbolData -> score in [-1, 1]
# --------------------------------------------------------------------------- #

def momentum_signal(sd: SymbolData, cfg: dict) -> float:
    """Trailing-return momentum, squashed to [-1, 1].

    Positive when the symbol has risen over the lookback window. Scaled so that
    a ~20% trailing move saturates toward the top of the range.
    """
    raw = float(sd.latest["momentum_raw"])
    return _squash(raw, scale=MOMENTUM_SCALE)


def mean_reversion_signal(sd: SymbolData, cfg: dict) -> float:
    """Mean-reversion from a moving-average z-score.

    A *negative* z-score (price below its average) is a *positive* buy signal,
    hence the sign flip. Scaled so a 2-sigma deviation is a strong signal.
    """
    z = float(sd.latest["reversion_z"])
    return _squash(-z, scale=REVERSION_SCALE)


def trend_signal(sd: SymbolData, cfg: dict) -> float:
    """Fast-vs-slow moving-average trend filter.

    Positive when the fast MA sits above the slow MA (up-regime). Scaled so a
    5% MA separation is a strong reading.
    """
    ratio = float(sd.latest["trend_ratio"])
    return _squash(ratio, scale=TREND_SCALE)


# Registry: name -> scoring function. Add new signals here.
SIGNALS: Dict[str, Callable[[SymbolData, dict], float]] = {
    "momentum": momentum_signal,
    "mean_reversion": mean_reversion_signal,
    "trend": trend_signal,
}


@dataclass
class AssetScore:
    """Composite score for one asset plus the per-signal breakdown.

    The breakdown is kept for transparency — every decision the agent makes can
    be explained by inspecting which signals drove it.
    """

    symbol: str
    composite: float
    components: Dict[str, float]
    volatility: float

    def explain(self) -> str:
        parts = ", ".join(f"{k}={v:+.2f}" for k, v in self.components.items())
        return f"{self.symbol}: composite={self.composite:+.2f} ({parts})"


class Ensemble:
    """Weighted blend of the registered signals.

    Parameters
    ----------
    cfg : dict
        Full config. Uses ``strategy.weights`` for the blend.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        weights: Dict[str, float] = dict(cfg["strategy"]["weights"])
        total = sum(abs(w) for w in weights.values())
        if total <= 0:
            raise ValueError("strategy.weights must not all be zero.")
        # Normalise so the composite stays within [-1, 1].
        self.weights = {k: w / total for k, w in weights.items()}
        # Validate that every weighted signal actually exists in the registry.
        unknown = set(self.weights) - set(SIGNALS)
        if unknown:
            raise ValueError(f"Unknown signals in config weights: {unknown}")
        logger.info("Ensemble weights (normalised): %s", self.weights)

    def score_one(self, sd: SymbolData) -> AssetScore:
        """Score a single symbol, returning composite + component breakdown."""
        components: Dict[str, float] = {}
        composite = 0.0
        for name, weight in self.weights.items():
            s = SIGNALS[name](sd, self.cfg)
            components[name] = s
            composite += weight * s
        return AssetScore(
            symbol=sd.symbol,
            composite=composite,
            components=components,
            volatility=float(sd.latest["vol"]),
        )

    def score_all(self, data: Dict[str, SymbolData]) -> Dict[str, AssetScore]:
        """Score every symbol in the universe.

        Returns a dict keyed by symbol, ordered by descending composite score
        so callers can rank/select the strongest names first.
        """
        scores = {sym: self.score_one(sd) for sym, sd in data.items()}
        ranked = dict(
            sorted(scores.items(), key=lambda kv: kv[1].composite, reverse=True)
        )
        for asc in ranked.values():
            logger.debug(asc.explain())
        return ranked
