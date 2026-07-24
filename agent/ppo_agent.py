# =============================================================================
# PPO Trading Agent — Proximal Policy Optimization wrapper for portfolio management
# =============================================================================
# Wraps stable-baselines3 PPO with:
#   - Full config-driven hyperparameter injection (zero hardcoded values)
#   - Periodic validation via custom callback with Sharpe-based checkpointing
#   - Deterministic evaluation with comprehensive risk/return metrics
#   - Automatic GPU/CPU device selection
#   - Reproducible seeding across all random components
# =============================================================================

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from utils.metrics import (
    annualized_return,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    win_rate,
)

# Module-level logger — avoids print() and respects caller's logging config
logger = logging.getLogger(__name__)


# =============================================================================
# Validation Callback — runs a full evaluation episode at fixed intervals
# =============================================================================
class ValidationCallback(BaseCallback):
    """Periodic validation callback that evaluates the agent on a held-out env.

    At every ``eval_freq`` training steps, runs one full deterministic episode
    on ``val_env``, computes the Sharpe ratio, logs it, and saves a checkpoint
    if the Sharpe exceeds the previous best.

    Attributes:
        val_env: Gymnasium environment for validation (must NOT overlap with train data).
        eval_freq: Number of training steps between consecutive evaluations.
        checkpoint_dir: Directory to write ``ppo_best.zip`` checkpoints.
        best_val_sharpe: Running maximum of validation Sharpe ratios.
        val_sharpes: History of ``(step, sharpe)`` tuples for downstream plotting.
    """

    def __init__(
        self,
        val_env: gym.Env,
        eval_freq: int,
        checkpoint_dir: str,
        verbose: int = 0,
    ) -> None:
        """Initialise the validation callback.

        Args:
            val_env: Held-out gymnasium environment for periodic evaluation.
            eval_freq: Evaluate every this many *training* timesteps.
            checkpoint_dir: Filesystem path where best checkpoints are saved.
            verbose: Verbosity level forwarded to ``BaseCallback``.
        """
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.checkpoint_dir = checkpoint_dir
        self.best_val_sharpe: float = -np.inf  # initialise to worst possible
        self.val_sharpes: List[Tuple[int, float]] = []

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        """Called after every environment step during training.

        Returns:
            True to continue training (we never request early stopping here).
        """
        # Only evaluate at the configured frequency
        if self.num_timesteps % self.eval_freq != 0:
            return True  # not an eval step — continue training

        # --- Run one full deterministic episode on the validation env ---
        obs, info = self.val_env.reset()
        daily_returns: List[float] = []
        terminated, truncated = False, False

        while not (terminated or truncated):
            # Deterministic action selection — no exploration noise
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.val_env.step(action)
            daily_returns.append(float(reward))

        # --- Compute Sharpe on the validation episode ---
        returns_array = np.array(daily_returns, dtype=np.float64)
        val_sharpe = float(sharpe_ratio(returns_array))

        # Record for later analysis / plotting
        self.val_sharpes.append((self.num_timesteps, val_sharpe))

        logger.info(
            "Step %d | Val Sharpe: %.4f", self.num_timesteps, val_sharpe
        )

        # --- Checkpoint if this is the best validation Sharpe so far ---
        if val_sharpe > self.best_val_sharpe:
            self.best_val_sharpe = val_sharpe
            save_path = os.path.join(self.checkpoint_dir, "ppo_best")
            self.model.save(save_path)  # writes ppo_best.zip
            logger.info(
                "New best val Sharpe %.4f — checkpoint saved to %s.zip",
                val_sharpe,
                save_path,
            )

        return True  # never early-stop from callback


# =============================================================================
# PPOTrader — main agent class
# =============================================================================
class PPOTrader:
    """PPO-based trading agent for single-asset portfolio management.

    Encapsulates the full lifecycle: initialisation from a config dict,
    training with periodic validation, deterministic evaluation, and
    checkpoint save/load.

    Design decisions:
        - ``MlpPolicy`` is used because the observation space is a flat
          feature vector (OHLCV + indicators), not image-like.
        - All hyperparameters originate from a config dict so that
          experiments are fully reproducible from a single YAML file.
        - Device selection is automatic (``"auto"``): PyTorch picks GPU
          when CUDA/MPS is available, CPU otherwise.

    Attributes:
        config: Frozen copy of the configuration dictionary.
        model: The underlying ``stable_baselines3.PPO`` instance.
        training_rewards: Episode rewards collected during training.
        val_sharpes: ``(step, sharpe)`` history from validation callback.
        best_val_sharpe: Maximum Sharpe observed during validation.
    """

    def __init__(self, config: Dict[str, Any], env: gym.Env) -> None:
        """Create a PPOTrader from a config dict and a training environment.

        Args:
            config: Dictionary of hyperparameters (typically loaded from YAML).
                    Required keys: learning_rate, n_steps, batch_size, n_epochs,
                    gamma, gae_lambda, clip_range, seed.
                    Optional keys: net_arch, tensorboard_log_dir, checkpoint_dir,
                    eval_freq.
            env: Gymnasium-compatible training environment.
        """
        self.config = config

        # --- Resolve directories (create if missing) ---
        self._checkpoint_dir: str = config.get("checkpoint_dir", "checkpoints/")
        self._tensorboard_dir: str = config.get("tensorboard_log_dir", "runs/")
        Path(self._checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self._tensorboard_dir).mkdir(parents=True, exist_ok=True)

        # --- Network architecture from config ---
        net_arch: List[int] = config.get("net_arch", [256, 256])
        policy_kwargs: Dict[str, Any] = {"net_arch": net_arch}

        # --- Seed for reproducibility ---
        seed: int = config.get("seed", 42)

        # --- Instantiate SB3 PPO ---
        # device="auto" lets PyTorch pick GPU (CUDA/MPS) when available
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=float(config["learning_rate"]),
            n_steps=int(config["n_steps"]),
            batch_size=int(config["batch_size"]),
            n_epochs=int(config["n_epochs"]),
            gamma=float(config["gamma"]),
            gae_lambda=float(config["gae_lambda"]),
            clip_range=float(config["clip_range"]),
            policy_kwargs=policy_kwargs,
            tensorboard_log=self._tensorboard_dir,
            seed=seed,
            device="auto",  # GPU if available, CPU otherwise
            verbose=0,  # SB3 internal logging off; we log ourselves
        )

        logger.info(
            "PPOTrader initialised — device=%s, seed=%d, arch=%s",
            self.model.device,
            seed,
            net_arch,
        )

        # --- Tracking containers (populated during training) ---
        self.training_rewards: List[float] = []
        self.val_sharpes: List[Tuple[int, float]] = []
        self.best_val_sharpe: float = -np.inf

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, total_timesteps: int, val_env: gym.Env) -> None:
        """Train the PPO agent with periodic validation.

        Args:
            total_timesteps: Total number of environment interactions.
            val_env: Held-out gymnasium environment used for periodic Sharpe
                     evaluation and best-model checkpointing.
        """
        eval_freq: int = int(self.config.get("eval_freq", 10_000))

        # Set up the validation callback
        val_callback = ValidationCallback(
            val_env=val_env,
            eval_freq=eval_freq,
            checkpoint_dir=self._checkpoint_dir,
            verbose=0,
        )

        logger.info(
            "Starting training — %d timesteps, eval every %d steps",
            total_timesteps,
            eval_freq,
        )

        # --- Run PPO training loop ---
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=val_callback,
            log_interval=10,  # SB3 logs every 10 rollouts to tensorboard
            progress_bar=False,  # avoid tqdm noise in production
        )

        # --- Persist validation metrics from the callback ---
        self.val_sharpes = val_callback.val_sharpes
        self.best_val_sharpe = val_callback.best_val_sharpe

        # --- Collect episode rewards from the SB3 monitor ---
        # ep_info_buffer stores the most recent episodes
        if self.model.ep_info_buffer:
            self.training_rewards = [
                ep_info["r"] for ep_info in self.model.ep_info_buffer
            ]

        # --- Save final checkpoint ---
        final_path = os.path.join(self._checkpoint_dir, "ppo_final")
        self.model.save(final_path)
        logger.info("Training complete — final model saved to %s.zip", final_path)
        logger.info(
            "Best validation Sharpe during training: %.4f", self.best_val_sharpe
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, env: gym.Env) -> Dict[str, Any]:
        """Run one full deterministic episode and compute performance metrics.

        Args:
            env: Gymnasium environment to evaluate on (e.g. test set).

        Returns:
            Dictionary with keys:
                - ``return``: total cumulative return (float)
                - ``sharpe``: annualised Sharpe ratio (float)
                - ``max_dd``: maximum drawdown as a negative fraction (float)
                - ``win_rate``: fraction of positive-return days (float)
                - ``calmar``: Calmar ratio (float)
                - ``equity_curve``: cumulative portfolio values (np.ndarray)
                - ``actions``: sequence of discrete actions taken (np.ndarray)
                - ``daily_returns``: per-step returns (np.ndarray)
        """
        obs, info = env.reset()
        daily_returns: List[float] = []
        actions: List[int] = []
        terminated, truncated = False, False

        while not (terminated or truncated):
            # Deterministic policy — no stochastic exploration
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            daily_returns.append(float(reward))
            actions.append(int(action))

        # --- Convert to numpy for metric computation ---
        returns_array = np.array(daily_returns, dtype=np.float64)
        actions_array = np.array(actions, dtype=np.int32)

        # --- Build equity curve from daily returns ---
        initial_value: float = float(
            self.config.get("initial_portfolio_value", 10_000)
        )
        # Cumulative product of (1 + r_t) scaled by initial capital
        equity_curve = initial_value * np.cumprod(1.0 + returns_array)

        # --- Compute all risk/return metrics ---
        total_return = float(annualized_return(returns_array))
        sharpe = float(sharpe_ratio(returns_array))
        mdd = float(max_drawdown(equity_curve))
        wr = float(win_rate(returns_array))
        calmar = float(calmar_ratio(returns_array, equity_curve))

        metrics: Dict[str, Any] = {
            "return": total_return,
            "sharpe": sharpe,
            "max_dd": mdd,
            "win_rate": wr,
            "calmar": calmar,
            "equity_curve": equity_curve,
            "actions": actions_array,
            "daily_returns": returns_array,
        }

        logger.info(
            "Evaluation — Return: %.4f | Sharpe: %.4f | MaxDD: %.4f | "
            "WinRate: %.4f | Calmar: %.4f",
            total_return,
            sharpe,
            mdd,
            wr,
            calmar,
        )

        return metrics

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        """Load a previously saved PPO checkpoint.

        Args:
            path: Filesystem path to the ``.zip`` checkpoint (with or without
                  the ``.zip`` extension — SB3 handles both).
        """
        self.model = PPO.load(path, device="auto")  # auto-detect device
        logger.info("Loaded PPO checkpoint from %s", path)
