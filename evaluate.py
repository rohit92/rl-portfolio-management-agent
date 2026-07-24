"""
evaluate.py — Evaluate a trained PPO agent against baseline strategies.

Usage:
    python evaluate.py --config configs/default.yaml --ticker AAPL \
                       --checkpoint checkpoints/ppo_best.zip

Workflow:
    1. Load config and test data.
    2. Evaluate three agents (PPO, BuyAndHold, Random) on the test set.
    3. Print a formatted comparison table of key performance metrics.
    4. Save equity curve and action distribution plots.
"""

import argparse
import logging
import os
from typing import Any, Dict, List

import pandas as pd
import yaml

from env.trading_env import TradingEnv
from agent.ppo_agent import PPOTrader
from agent.baselines import BuyAndHoldAgent, RandomAgent
from utils.plot import plot_equity_curves, plot_action_distribution


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


def main() -> None:
    """Parse arguments, evaluate agents, and print/save results."""
    # ── Logging setup ─────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── CLI ───────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Evaluate trained agent vs baselines"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Ticker symbol to evaluate on",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ppo_best.zip",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────
    config = load_config(args.config)

    # Ensure output directories exist
    results_dir: str = config.get("results_dir", "results/")
    os.makedirs(results_dir, exist_ok=True)

    # ── Test data ─────────────────────────────────────────────────────
    data_dir = "data/raw"
    test_path = os.path.join(data_dir, f"{args.ticker}_test.csv")
    print(f"Loading test data from {test_path} ...")
    test_df = pd.read_csv(test_path, parse_dates=True)

    if test_df.empty:
        raise ValueError(f"Test data file is empty: {test_path}")

    # ── Evaluate all three agents ─────────────────────────────────────
    # Each agent gets a freshly-reset environment so results are comparable.

    results: Dict[str, Dict[str, Any]] = {}
    equity_curves: Dict[str, List[float]] = {}
    ppo_actions: List[int] = []

    # --- 1. PPO Agent ---
    print("Evaluating PPO agent ...")
    ppo_env = TradingEnv(data=test_df.copy(), config=config)
    ppo_trader = PPOTrader(config=config, env=ppo_env)

    # Load the trained checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.isfile(checkpoint_path):
        # Try without .zip extension (SB3 may auto-append it)
        alt_path = checkpoint_path.replace(".zip", "")
        if os.path.isfile(alt_path + ".zip"):
            checkpoint_path = alt_path + ".zip"
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {args.checkpoint}\n"
                f"Run train.py first to generate a checkpoint."
            )

    ppo_trader.load(checkpoint_path)
    ppo_metrics = ppo_trader.evaluate(ppo_env)
    results["PPO"] = ppo_metrics
    equity_curves["PPO"] = ppo_metrics["equity_curve"].tolist()
    ppo_actions = ppo_metrics["actions"].tolist()
    print("  ✓ PPO evaluation complete")

    # --- 2. Buy-and-Hold Agent ---
    print("Evaluating Buy-and-Hold agent ...")
    bh_env = TradingEnv(data=test_df.copy(), config=config)
    bh_agent = BuyAndHoldAgent(config=config)
    bh_metrics = bh_agent.evaluate(bh_env)
    results["BuyAndHold"] = bh_metrics
    equity_curves["BuyAndHold"] = bh_metrics["equity_curve"].tolist()
    print("  ✓ BuyAndHold evaluation complete")

    # --- 3. Random Agent ---
    print("Evaluating Random agent ...")
    rand_env = TradingEnv(data=test_df.copy(), config=config)
    rand_agent = RandomAgent(config=config, seed=config.get("seed", 42))
    rand_metrics = rand_agent.evaluate(rand_env)
    results["Random"] = rand_metrics
    equity_curves["Random"] = rand_metrics["equity_curve"].tolist()
    print("  ✓ Random evaluation complete")

    # ── Comparison table ──────────────────────────────────────────────
    print()
    print("=" * 65)
    print(
        f"{'Agent':<20} {'Return':>8} {'Sharpe':>8} "
        f"{'MaxDD':>8} {'WinRate':>8} {'Calmar':>8}"
    )
    print("=" * 65)
    for agent_name, metrics in results.items():
        print(
            f"{agent_name:<20} {metrics['return']:>7.2%} "
            f"{metrics['sharpe']:>8.3f} {metrics['max_dd']:>7.2%} "
            f"{metrics['win_rate']:>7.2%} {metrics['calmar']:>8.3f}"
        )
    print("=" * 65)

    # ── Plots ─────────────────────────────────────────────────────────
    equity_path = os.path.join(results_dir, "equity_curves.png")
    plot_equity_curves(
        results=equity_curves,
        save_path=equity_path,
    )

    action_dist_path = os.path.join(results_dir, "action_distribution.png")
    plot_action_distribution(
        actions=ppo_actions,
        save_path=action_dist_path,
    )

    print(f"\nPlots saved to {results_dir}")


if __name__ == "__main__":
    main()
