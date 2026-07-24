"""
Custom Gymnasium environment for single-asset RL-based portfolio management.

This module defines ``TradingEnv``, the core environment consumed by the PPO
agent.  It wraps a historical price DataFrame into the standard Gymnasium
``step`` / ``reset`` API with:

* **Observation space** — a flattened 1-D vector of the last *lookback_window*
  days of normalised market features concatenated with two portfolio-state
  scalars (current position flag and unrealised PnL percentage).
* **Action space** — ``Discrete(3)`` mapping to Hold (0), Buy (1), Sell (2).
* **Reward function** — risk-adjusted log-return minus transaction costs and a
  rolling-volatility penalty, conceptually analogous to Sharpe-ratio
  optimisation.

Design note — *Reward shaping & autonomous-driving analogy*:
    The explicit penalty terms (transaction cost, volatility) serve the same
    role as *guidance rewards* in autonomous-driving RL (e.g., lane-keeping
    bonuses, jerk penalties).  Both encode domain-expert knowledge directly
    into the reward signal so the policy converges to *safe* / *profitable*
    behaviour without needing an impractically large number of environment
    interactions.

References:
    - Schulman et al., 2017 (PPO)
    - Gymnasium API — https://gymnasium.farama.org/
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Module-level logger — library code never uses ``print``.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical feature column names (Title-case and lower-case variants).
# yfinance >= 0.2.31 returns lowercase; older versions use Title-case.
# We detect which convention the caller's DataFrame uses at init time.
# ---------------------------------------------------------------------------
_FEATURES_TITLE: List[str] = [
    "Open", "High", "Low", "Close", "Volume",
    "daily_return", "rolling_vol_5d", "ma_ratio_10", "ma_ratio_20",
]
_FEATURES_LOWER: List[str] = [
    "open", "high", "low", "close", "volume",
    "daily_return", "rolling_vol_5d", "ma_ratio_10", "ma_ratio_20",
]

# Number of portfolio-state features appended to each observation.
_N_PORTFOLIO_FEATURES: int = 2  # (current_position, unrealized_pnl_pct)


def _resolve_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of 9 feature column names present in *df*.

    Tries Title-case first, then lower-case.  Raises ``ValueError`` if
    neither convention matches.

    Parameters
    ----------
    df : pd.DataFrame
        The market-data DataFrame to inspect.

    Returns
    -------
    List[str]
        Ordered list of 9 feature column names that exist in *df*.

    Raises
    ------
    ValueError
        If the required columns cannot be found under either naming
        convention.
    """
    # Prefer Title-case (matches older yfinance / user-normalised data)
    if all(col in df.columns for col in _FEATURES_TITLE):
        return list(_FEATURES_TITLE)

    # Fall back to lower-case (newer yfinance default)
    if all(col in df.columns for col in _FEATURES_LOWER):
        return list(_FEATURES_LOWER)

    # Build a helpful diagnostic message listing what is missing.
    missing_title = [c for c in _FEATURES_TITLE if c not in df.columns]
    missing_lower = [c for c in _FEATURES_LOWER if c not in df.columns]
    raise ValueError(
        f"DataFrame is missing required feature columns.\n"
        f"  Title-case missing: {missing_title}\n"
        f"  Lower-case missing: {missing_lower}\n"
        f"  Available columns : {list(df.columns)}"
    )


# ============================================================================
# TradingEnv
# ============================================================================

class TradingEnv(gym.Env):
    """Gymnasium environment for single-asset long-only trading.

    The agent observes a rolling window of normalised market features plus
    two portfolio-state scalars and chooses among three discrete actions
    (Hold / Buy / Sell) at each daily time-step.

    Parameters
    ----------
    data : pd.DataFrame
        Historical market data with at least the 9 required feature columns
        (normalised OHLCV + engineered features).  Must contain enough rows
        for at least ``lookback_window + 1`` steps.
    config : dict
        Run configuration.  Recognised keys (with defaults):

        * ``lookback_window`` (int, 20) — number of past days in each obs.
        * ``initial_portfolio_value`` (float, 10 000) — starting capital (USD).
        * ``transaction_cost`` (float, 0.001) — proportional cost per trade.
        * ``volatility_penalty_weight`` (float, 0.1) — scaling factor for the
          rolling-volatility penalty term.

    Attributes
    ----------
    action_space : gym.spaces.Discrete
        ``Discrete(3)`` — 0=Hold, 1=Buy, 2=Sell.
    observation_space : gym.spaces.Box
        Flat ``float32`` vector of length ``lookback_window * n_features + 2``.
    """

    # Gymnasium metadata — we support only a simple text render mode.
    metadata: Dict[str, Any] = {"render_modes": ["human"]}

    # ------------------------------------------------------------------ #
    #  Construction & validation                                          #
    # ------------------------------------------------------------------ #

    def __init__(self, data: pd.DataFrame, config: dict) -> None:
        """Initialise the environment from pre-processed market data.

        Parameters
        ----------
        data : pd.DataFrame
            Pre-normalised market data.
        config : dict
            Run-level configuration dictionary.
        """
        super().__init__()

        # ---- Extract config with sensible defaults ----
        self.lookback_window: int = int(config.get("lookback_window", 20))
        self.initial_portfolio_value: float = float(
            config.get("initial_portfolio_value", 10_000)
        )
        self.transaction_cost: float = float(
            config.get("transaction_cost", 0.001)
        )
        self.volatility_penalty_weight: float = float(
            config.get("volatility_penalty_weight", 0.1)
        )

        # ---- Resolve column names (handles yfinance casing) ----
        self.feature_columns: List[str] = _resolve_feature_columns(data)
        self.n_features: int = len(self.feature_columns)  # always 9

        # Identify which column holds the close price for PnL tracking.
        # Works with both "Close" and "close".
        self._close_col: str = (
            "Close" if "Close" in self.feature_columns else "close"
        )

        # ---- Store data as NumPy for fast indexing ----
        self._data: np.ndarray = data[self.feature_columns].to_numpy(
            dtype=np.float32
        )  # shape: (n_days, n_features)

        # For reward/PnL computation we need UN-NORMALISED close prices.
        # The data pipeline saves these in a 'raw_close' column.
        # If that column is missing (e.g. user-supplied data), fall back
        # to the normalized Close column with a warning.
        if "raw_close" in data.columns:
            self._close_prices: np.ndarray = data["raw_close"].to_numpy(
                dtype=np.float64
            )
        else:
            logger.warning(
                "'raw_close' column not found — using normalized '%s' for "
                "reward computation.  Log-returns may be incorrect if data "
                "is z-score normalized.",
                self._close_col,
            )
            self._close_prices = data[self._close_col].to_numpy(
                dtype=np.float64
            )  # 1-D array of close prices (may be normalised)

        # ---- Validate data length ----
        self._n_rows: int = len(self._data)
        min_required = self.lookback_window + 1  # need ≥1 actionable step
        if self._n_rows < min_required:
            raise ValueError(
                f"Data has {self._n_rows} rows but at least "
                f"{min_required} are required (lookback_window="
                f"{self.lookback_window})."
            )

        # Maximum steps the agent can take in a single episode.
        self._max_steps: int = self._n_rows - self.lookback_window

        # ---- Define Gymnasium spaces ----
        obs_size: int = self.lookback_window * self.n_features + _N_PORTFOLIO_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )  # flat vector of length 182 with default settings

        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell

        # ---- Episode state (initialised properly in reset()) ----
        self._current_step: int = 0        # index into _data
        self._position: int = 0            # 0 = flat, 1 = long
        self._entry_price: float = 0.0     # price at which we entered
        self._portfolio_value: float = self.initial_portfolio_value
        self._step_count: int = 0          # steps taken in current episode
        self._last_reward: float = 0.0     # cached for render()
        self._daily_returns: List[float] = []  # rolling buffer for vol penalty

        logger.info(
            "TradingEnv initialised — %d rows, lookback=%d, obs_size=%d, "
            "max_steps=%d",
            self._n_rows,
            self.lookback_window,
            obs_size,
            self._max_steps,
        )

    # ------------------------------------------------------------------ #
    #  Observation builder                                                #
    # ------------------------------------------------------------------ #

    def _build_observation(self) -> np.ndarray:
        """Construct the flattened observation vector for the current step.

        Layout::

            [ market_features(lookback)  |  current_position  |  unrealized_pnl_pct ]
              ← lookback * n_features →    ←       2 scalars                       →

        Returns
        -------
        np.ndarray
            1-D ``float32`` vector of length ``lookback_window * n_features + 2``.
        """
        # Slice the lookback window ending at (and including) current step.
        start: int = self._current_step - self.lookback_window + 1
        end: int = self._current_step + 1  # exclusive upper bound
        market_features: np.ndarray = self._data[start:end].flatten()

        # Portfolio-state features ------------------------------------------
        current_position: float = float(self._position)

        # Unrealised PnL as a fraction of entry price (0.0 if flat).
        if self._position == 1 and self._entry_price > 0.0:
            current_price = self._close_prices[self._current_step]
            unrealized_pnl_pct: float = (
                (current_price - self._entry_price) / self._entry_price
            )
        else:
            unrealized_pnl_pct = 0.0

        portfolio_state = np.array(
            [current_position, unrealized_pnl_pct], dtype=np.float32
        )

        return np.concatenate([market_features, portfolio_state])

    # ------------------------------------------------------------------ #
    #  reset()                                                            #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the beginning of an episode.

        Parameters
        ----------
        seed : int, optional
            RNG seed for reproducibility (passed to ``super().reset``).
        options : dict, optional
            Additional reset options (unused, kept for Gymnasium compliance).

        Returns
        -------
        observation : np.ndarray
            Initial observation vector.
        info : dict
            Empty auxiliary information dictionary.
        """
        super().reset(seed=seed)  # seeds self.np_random

        # Start at the first index where a full lookback window is available.
        self._current_step = self.lookback_window - 1
        self._position = 0                          # flat (no holdings)
        self._entry_price = 0.0                     # no entry yet
        self._portfolio_value = self.initial_portfolio_value
        self._step_count = 0
        self._last_reward = 0.0
        self._daily_returns = []                    # clear rolling buffer

        obs = self._build_observation()

        logger.debug(
            "Environment reset — starting at index %d", self._current_step
        )
        return obs, {}

    # ------------------------------------------------------------------ #
    #  step()                                                             #
    # ------------------------------------------------------------------ #

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time-step in the environment.

        Parameters
        ----------
        action : int
            Agent's chosen action: 0=Hold, 1=Buy, 2=Sell.

        Returns
        -------
        observation : np.ndarray
            Observation *after* the action is applied and time advances.
        reward : float
            Scalar reward for this transition.
        terminated : bool
            ``True`` when the episode ends naturally (data exhausted).
        truncated : bool
            Always ``False`` (no external truncation criterion).
        info : dict
            Diagnostic information for logging / evaluation.
        """
        # Advance time — the new step is the one we are *transitioning to*.
        prev_step: int = self._current_step
        self._current_step += 1
        self._step_count += 1

        # Current and previous close prices for return calculation.
        price_now: float = float(self._close_prices[self._current_step])
        price_prev: float = float(self._close_prices[prev_step])

        # ---- Resolve action & detect invalid moves ----
        invalid_action_penalty: float = 0.0
        action_taken: int = action  # may be overridden below

        if action == 1:  # Buy
            if self._position == 1:
                # Already long — treat as Hold, apply small penalty.
                invalid_action_penalty = -0.001
                action_taken = 0  # effective action is Hold
            else:
                # Enter long position at current price.
                self._position = 1
                self._entry_price = price_now
        elif action == 2:  # Sell
            if self._position == 0:
                # Not holding — treat as Hold, apply small penalty.
                invalid_action_penalty = -0.001
                action_taken = 0  # effective action is Hold
            else:
                # Close long position.
                self._position = 0
                self._entry_price = 0.0
        # action == 0 (Hold) requires no state change.

        # ---- Compute reward components ----

        # 1) Log-return: captures *compounding* correctly.
        #    Arithmetic returns are non-additive over time; log-returns are,
        #    making them the natural unit for multi-period optimisation.
        if self._position == 1:
            # We are (still or newly) long — earn the market return.
            log_return: float = math.log(price_now / price_prev)
        else:
            log_return = 0.0

        # 2) Transaction cost: penalises over-trading and reflects real
        #    market friction (commissions + slippage ≈ 10 bps per trade).
        #    Only charged when the agent *changes* its position.
        position_changed: bool = (
            (action == 1 and action_taken == 1)   # valid buy
            or (action == 2 and action_taken == 2)  # valid sell (action_taken stays 2 for sell)
        )
        # Re-check: for sell, action_taken was not overwritten when valid.
        position_changed = (
            (action == 1 and invalid_action_penalty == 0.0)
            or (action == 2 and invalid_action_penalty == 0.0)
        )
        tc: float = self.transaction_cost if position_changed else 0.0

        # 3) Volatility penalty: encourages the policy to avoid high-
        #    variance return streams.  Conceptually equivalent to
        #    maximising a Sharpe-like objective (return / risk).
        self._daily_returns.append(log_return)
        vol_window: int = 5  # use last 5 daily returns for rolling std
        if len(self._daily_returns) >= vol_window:
            recent = self._daily_returns[-vol_window:]
            rolling_std: float = float(np.std(recent, ddof=0))
        else:
            # Not enough history yet — no penalty.
            rolling_std = 0.0
        vol_penalty: float = self.volatility_penalty_weight * rolling_std

        # 4) Combine all reward components.
        #    CONNECTION TO THESIS: this reward-shaping mirrors the
        #    *guidance reward* pattern in autonomous-driving RL —
        #    domain knowledge (risk aversion, cost awareness) is
        #    baked directly into the scalar reward to accelerate
        #    convergence toward safe/profitable policies.
        reward: float = (
            log_return
            - tc
            - vol_penalty
            + invalid_action_penalty  # negative when action was invalid
        )

        # ---- Update portfolio value ----
        if self._position == 1:
            # Mark-to-market: portfolio value follows the stock price.
            daily_mult: float = price_now / price_prev
            self._portfolio_value *= daily_mult
        # If flat, portfolio value stays constant (cash earns 0).

        # Deduct transaction cost from portfolio value when trading.
        if position_changed:
            self._portfolio_value *= (1.0 - self.transaction_cost)

        # ---- Termination ----
        # Episode ends when we run out of data.
        terminated: bool = self._current_step >= self._n_rows - 1
        truncated: bool = False  # no external truncation in this env

        # Cache reward for render().
        self._last_reward = reward

        # ---- Build next observation ----
        obs = self._build_observation()

        # ---- Info dict for logging / evaluation ----
        info: Dict[str, Any] = {
            "portfolio_value": self._portfolio_value,
            "position": self._position,
            "daily_return": log_return,
            "action_taken": action_taken if action_taken != 0 or action == 0 else 0,
            "price": price_now,
            "step": self._step_count,
            "transaction_cost": tc,
            "volatility_penalty": vol_penalty,
            "invalid_action_penalty": invalid_action_penalty,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  render()                                                           #
    # ------------------------------------------------------------------ #

    def render(self) -> None:
        """Print a single-line status summary of the current state.

        Output format::

            Step 42 | Price: 152.30 | Position: LONG | Portfolio: $10,234.56 | Reward: 0.0023
        """
        position_str: str = "LONG" if self._position == 1 else "FLAT"
        price: float = float(self._close_prices[self._current_step])
        logger.info(
            "Step %d | Price: %.2f | Position: %s | Portfolio: $%.2f | "
            "Reward: %.6f",
            self._step_count,
            price,
            position_str,
            self._portfolio_value,
            self._last_reward,
        )
        # Also print to stdout for interactive use / debugging.
        print(
            f"Step {self._step_count} | Price: {price:.2f} | "
            f"Position: {position_str} | "
            f"Portfolio: ${self._portfolio_value:,.2f} | "
            f"Reward: {self._last_reward:.6f}"
        )

    # ------------------------------------------------------------------ #
    #  Convenience properties                                             #
    # ------------------------------------------------------------------ #

    @property
    def portfolio_value(self) -> float:
        """Return the current portfolio value in USD."""
        return self._portfolio_value

    @property
    def position(self) -> int:
        """Return current position: 0 = flat, 1 = long."""
        return self._position

    def __repr__(self) -> str:
        return (
            f"TradingEnv(rows={self._n_rows}, lookback={self.lookback_window}, "
            f"features={self.n_features}, obs_size="
            f"{self.observation_space.shape[0]})"
        )
