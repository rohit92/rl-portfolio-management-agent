# =============================================================================
# Baseline Agents — non-learned benchmarks for RL agent comparison
# =============================================================================
# Provides two deterministic baselines that produce the *same* output dict
# as ``PPOTrader.evaluate()``, enabling apples-to-apples comparison:
#
#   1. BuyAndHoldAgent — buys on the first step, holds until episode end.
#   2. RandomAgent — samples uniformly from the action space each step.
#
# Both agents are stateless (no training) and operate purely on the env API.
# =============================================================================

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from utils.metrics import (
    returns_from_values,
    annualized_return,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    win_rate,
)

# Module-level logger — consistent with the rest of the codebase
logger = logging.getLogger(__name__)


# =============================================================================
# Buy-and-Hold Baseline
# =============================================================================
class BuyAndHoldAgent:
    """Deterministic buy-and-hold baseline agent.

    Strategy:
        Step 0  → action 1 (BUY)
        Step 1+ → action 0 (HOLD) until the episode terminates

    This represents the simplest long-only strategy and serves as the
    minimum performance bar that any learned agent must beat.

    The environment is responsible for tracking position state — we merely
    emit the buy signal once and then do nothing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Store configuration for metric computation.

        Args:
            config: Optional configuration dictionary. Used to read
                    ``initial_portfolio_value`` for equity curve construction.
                    If *None*, defaults to 10 000.
        """
        self.config: Dict[str, Any] = config if config is not None else {}

    # ------------------------------------------------------------------
    def evaluate(self, env: gym.Env) -> Dict[str, Any]:
        """Run one full buy-and-hold episode and compute performance metrics.

        Args:
            env: Gymnasium-compatible trading environment.

        Returns:
            Dictionary with keys: ``return``, ``sharpe``, ``max_dd``,
            ``win_rate``, ``calmar``, ``equity_curve``, ``actions``,
            ``daily_returns``.  Identical schema to ``PPOTrader.evaluate()``.
        """
        obs, info = env.reset()
        portfolio_values: List[float] = []
        actions: List[int] = []
        terminated, truncated = False, False
        step: int = 0

        while not (terminated or truncated):
            if step == 0:
                action = 1  # BUY on the very first step
            else:
                action = 0  # HOLD for every subsequent step

            obs, reward, terminated, truncated, info = env.step(action)
            portfolio_values.append(float(info["portfolio_value"]))
            actions.append(action)
            step += 1

        # --- Convert to numpy for metric functions ---
        values_array = np.array([float(env.unwrapped.initial_portfolio_value)] + portfolio_values, dtype=np.float64)
        returns_array = returns_from_values(values_array)
        actions_array = np.array(actions, dtype=np.int32)

        # --- Equity curve: cumulative product of (1 + daily_return) ---
        initial_value: float = float(
            self.config.get("initial_portfolio_value", 10_000)
        )
        equity_curve = values_array[1:]  # actual portfolio value after each step

        # --- Compute risk/return metrics ---
        total_return = float(annualized_return(returns_array))
        sharpe_val = float(sharpe_ratio(returns_array))
        mdd = float(max_drawdown(equity_curve))
        wr = float(win_rate(returns_array))
        calmar = float(calmar_ratio(returns_array, equity_curve))

        metrics: Dict[str, Any] = {
            "return": total_return,
            "sharpe": sharpe_val,
            "max_dd": mdd,
            "win_rate": wr,
            "calmar": calmar,
            "equity_curve": equity_curve,
            "actions": actions_array,
            "daily_returns": returns_array,
        }

        logger.info(
            "BuyAndHold — Return: %.4f | Sharpe: %.4f | MaxDD: %.4f | "
            "WinRate: %.4f | Calmar: %.4f",
            total_return,
            sharpe_val,
            mdd,
            wr,
            calmar,
        )

        return metrics


# =============================================================================
# Random Agent Baseline
# =============================================================================
class RandomAgent:
    """Uniformly random baseline agent.

    At each step, samples an action uniformly at random from {0, 1, 2}
    (HOLD, BUY, SELL) using a seeded RNG for reproducibility.

    This establishes the *lower* performance bound — any agent performing
    below this level is worse than chance.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        """Initialise the random agent with a seeded RNG.

        Args:
            config: Optional configuration dictionary. Used to read
                    ``initial_portfolio_value`` for equity curve construction.
            seed: Random seed for the numpy Generator, ensuring reproducible
                  action sequences across runs.
        """
        self.config: Dict[str, Any] = config if config is not None else {}
        self.seed = seed
        # Use the modern numpy Generator API for cleaner seeding
        self._rng: np.random.Generator = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def evaluate(self, env: gym.Env) -> Dict[str, Any]:
        """Run one full random-action episode and compute performance metrics.

        Args:
            env: Gymnasium-compatible trading environment.

        Returns:
            Dictionary with keys: ``return``, ``sharpe``, ``max_dd``,
            ``win_rate``, ``calmar``, ``equity_curve``, ``actions``,
            ``daily_returns``.  Identical schema to ``PPOTrader.evaluate()``.
        """
        # Re-seed the RNG so repeated evaluate() calls are deterministic
        self._rng = np.random.default_rng(self.seed)

        obs, info = env.reset()
        portfolio_values: List[float] = []
        actions: List[int] = []
        terminated, truncated = False, False

        # Number of discrete actions: HOLD(0), BUY(1), SELL(2)
        n_actions: int = 3

        while not (terminated or truncated):
            # Sample uniformly from {0, 1, 2}
            action: int = int(self._rng.integers(low=0, high=n_actions))
            obs, reward, terminated, truncated, info = env.step(action)
            portfolio_values.append(float(info["portfolio_value"]))
            actions.append(action)

        # --- Convert to numpy for metric functions ---
        values_array = np.array([float(env.unwrapped.initial_portfolio_value)] + portfolio_values, dtype=np.float64)
        returns_array = returns_from_values(values_array)
        actions_array = np.array(actions, dtype=np.int32)

        # --- Equity curve: cumulative product of (1 + daily_return) ---
        initial_value: float = float(
            self.config.get("initial_portfolio_value", 10_000)
        )
        equity_curve = values_array[1:]  # actual portfolio value after each step

        # --- Compute risk/return metrics ---
        total_return = float(annualized_return(returns_array))
        sharpe_val = float(sharpe_ratio(returns_array))
        mdd = float(max_drawdown(equity_curve))
        wr = float(win_rate(returns_array))
        calmar = float(calmar_ratio(returns_array, equity_curve))

        metrics: Dict[str, Any] = {
            "return": total_return,
            "sharpe": sharpe_val,
            "max_dd": mdd,
            "win_rate": wr,
            "calmar": calmar,
            "equity_curve": equity_curve,
            "actions": actions_array,
            "daily_returns": returns_array,
        }

        logger.info(
            "RandomAgent — Return: %.4f | Sharpe: %.4f | MaxDD: %.4f | "
            "WinRate: %.4f | Calmar: %.4f",
            total_return,
            sharpe_val,
            mdd,
            wr,
            calmar,
        )

        return metrics
