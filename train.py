"""
train.py — Entry point for training the PPO portfolio management agent.

Usage:
    python train.py --config configs/default.yaml --ticker AAPL

Workflow:
    1. Parse CLI arguments and load YAML config.
    2. Load train/val CSV data from data/raw/.
    3. Create TradingEnv instances for training and validation.
    4. Initialize PPOTrader with the config and train environment.
    5. Train for `total_timesteps` specified in config.
    6. Report best validation Sharpe ratio and save training curve plot.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict

import pandas as pd
import yaml

from env.trading_env import TradingEnv
from agent.ppo_agent import PPOTrader
from utils.plot import plot_training_curve


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the YAML configuration dictionary.

    Args:
        config_path: Filesystem path to the YAML config file.

    Returns:
        Parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh)
    return config


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with basic validation.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the loaded DataFrame is empty.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath, parse_dates=True)
    if df.empty:
        raise ValueError(f"Data file is empty: {filepath}")
    return df


def main() -> None:
    """Parse arguments, train the PPO agent, and save results."""
    # ── Logging setup ─────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── CLI ───────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train PPO agent for portfolio management"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker symbol to train on",
    )
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────
    config = load_config(args.config)

    # Ensure output directories exist before any writes
    results_dir: str = config.get("results_dir", "results/")
    checkpoint_dir: str = config.get("checkpoint_dir", "checkpoints/")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────
    data_dir = "data/raw"
    train_path = os.path.join(data_dir, f"{args.ticker}_train.csv")
    val_path = os.path.join(data_dir, f"{args.ticker}_val.csv")

    print(f"Loading training data from {train_path} ...")
    train_df = load_csv_data(train_path)

    print(f"Loading validation data from {val_path} ...")
    val_df = load_csv_data(val_path)

    # ── Environments ──────────────────────────────────────────────────
    train_env = TradingEnv(data=train_df, config=config)
    val_env = TradingEnv(data=val_df, config=config)

    # ── Agent ─────────────────────────────────────────────────────────
    trader = PPOTrader(config=config, env=train_env)

    total_timesteps: int = config.get("total_timesteps", 100_000)
    print(f"Starting training for {total_timesteps:,} timesteps ...")

    # Train — validation metrics are tracked internally
    trader.train(
        total_timesteps=total_timesteps,
        val_env=val_env,
    )

    # Report best validation Sharpe
    best_sharpe = trader.best_val_sharpe
    # Find the step at which the best Sharpe was achieved
    best_step = 0
    if trader.val_sharpes:
        best_entry = max(trader.val_sharpes, key=lambda x: x[1])
        best_step = best_entry[0]

    print(
        f"Training complete. Best val Sharpe: {best_sharpe:.4f} "
        f"at step {best_step}"
    )

    # ── Plots ─────────────────────────────────────────────────────────
    curve_path = os.path.join(results_dir, "training_curve.png")

    # Extract just the Sharpe values from (step, sharpe) tuples
    val_sharpe_values = [s for _, s in trader.val_sharpes] if trader.val_sharpes else []

    plot_training_curve(
        rewards=trader.training_rewards,
        val_sharpes=val_sharpe_values,
        save_path=curve_path,
    )
    print(f"Training curve saved to {curve_path}")


if __name__ == "__main__":
    main()
