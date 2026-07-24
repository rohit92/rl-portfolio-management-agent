# RL-Based Portfolio Management Agent

**A reinforcement learning system that learns to trade a single equity using Proximal Policy Optimization (PPO).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2](https://img.shields.io/badge/pytorch-2.2-ee4c2c.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.3-green.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## Overview

This project applies the **Actor-Critic policy gradient framework** — the same algorithmic family used in model-based RL for autonomous driving — to sequential financial decision-making. The core insight is that portfolio management is a **Markov Decision Process**: the agent observes market state, takes actions (buy/sell/hold), and receives risk-adjusted rewards, exactly analogous to an autonomous agent observing sensor inputs and receiving driving rewards. Where a self-driving car must balance lane-keeping reward against collision penalty under partial observability, our trading agent must balance return against volatility and transaction costs under noisy market dynamics. Both problems demand policies that generalize from historical trajectories without overfitting to any single episode, and both benefit from PPO's stable, clipped-objective training — making this a natural bridge between robotics RL and quantitative finance.

---

## MDP Formulation

The trading problem is cast as a finite-horizon, episodic MDP:

| Component | Definition |
|---|---|
| **State *S*** | 20-day sliding window of OHLCV + technical indicators (daily return, 5-day rolling volatility, 10-day and 20-day MA ratios) + portfolio state (current position, unrealized PnL) |
| **Action *A*** | `Discrete(3)`: Hold = 0, Buy = 1, Sell = 2 |
| **Reward *R*** | `log(portfolio_value_t / portfolio_value_{t-1}) − transaction_cost × |trade| − λ × realized_volatility` |
| **Transition *T*** | Deterministic execution at next day's open; market dynamics are exogenous and stochastic |
| **Episode** | One calendar year of trading data (≈ 252 trading days) |
| **Discount *γ*** | 0.99 — standard for finite-horizon MDPs with daily steps |

> **Why this formulation works:** The state captures enough market microstructure (via the lookback window and indicators) to approximate the Markov property. The reward function directly encodes the risk-return trade-off that any rational portfolio manager optimizes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │  AAPL OHLCV  │────▶│  TradingEnv      │────▶│  Observation     │
  │  (CSV/yfinance)│    │  .step(action)   │     │  (182-dim float) │
  └──────────────┘     │  .reset()        │     └────────┬─────────┘
                       └──────────────────┘              │
                              ▲                          ▼
                              │               ┌──────────────────┐
                              │               │  PPO Agent       │
                              │               │  ┌────────────┐  │
                              │               │  │ Actor  MLP  │  │
                              │               │  │ 182→256→   │  │
                              │               │  │ 256→3      │  │
                              │               │  │ (policy π) │  │
                              │               │  └────────────┘  │
                              │               │  ┌────────────┐  │
                              │               │  │ Critic MLP  │  │
                              │               │  │ 182→256→   │  │
                              │               │  │ 256→1      │  │
                              │               │  │ (value V)  │  │
                              │               │  └────────────┘  │
                              │               └────────┬─────────┘
                              │                        │
                              │                        ▼
                       ┌──────┴───────┐     ┌──────────────────┐
                       │  Market Exec │◀────│  Action          │
                       │  (next-day   │     │  {Hold,Buy,Sell} │
                       │   open fill) │     └──────────────────┘
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐     ┌──────────────────┐
                       │   Reward     │────▶│  Rollout Buffer  │
                       │  log-return  │     │  (2048 steps)    │
                       │  − costs     │     └────────┬─────────┘
                       │  − vol pen.  │              │
                       └──────────────┘              ▼
                                            ┌──────────────────┐
                                            │  PPO Update      │
                                            │  • 10 epochs     │
                                            │  • 64 minibatch  │
                                            │  • Clipped obj.  │
                                            │  • GAE(λ=0.95)   │
                                            └──────────────────┘
```

**Observation Space Breakdown (182 dimensions):**

| Feature Group | Dims per Day | × Window | Total |
|---|---|---|---|
| OHLCV (Open, High, Low, Close, Volume) | 5 | 20 | 100 |
| Technical Indicators (daily_return, rolling_vol_5d, ma_ratio_10, ma_ratio_20) | 4 | 20 | 80 |
| Portfolio State (position, unrealized PnL) | 2 | 1 | 2 |
| | | **Total** | **182** |

---

## Why PPO over DQN

This project deliberately chooses **Proximal Policy Optimization (PPO)** — an Actor-Critic algorithm — over **Deep Q-Networks (DQN)**, a value-based method. The reasoning is both technical and thematic:

### 1. Actor-Critic Architecture
PPO maintains two networks: a **policy (actor)** that directly outputs action probabilities, and a **value function (critic)** that estimates expected future returns. This decomposition maps naturally onto trading: the actor learns *what to do*, and the critic learns *how good the current state is* — analogous to a portfolio manager's intuition paired with a risk model.

### 2. DQN's Action-Aliasing Problem
DQN learns a single Q-value for each state-action pair. In trading, many states have nearly identical optimal Q-values for different actions (e.g., holding and buying may be equally good in a flat market). This **action aliasing** causes DQN to oscillate between actions, leading to excessive trading and poor convergence. PPO's policy gradient naturally handles this by learning smooth action distributions.

### 3. Clipped Surrogate Objective
PPO's defining innovation is its clipped objective function:

```
L_CLIP(θ) = E[min(r_t(θ) · A_t,  clip(r_t(θ), 1−ε, 1+ε) · A_t)]
```

This prevents **destructive policy updates** — a single bad batch of financial data cannot catastrophically change the policy. This stability is essential in non-stationary environments like financial markets, where distribution shift is the norm rather than the exception.

### 4. Thematic Alignment with Autonomous Driving
PPO belongs to the same family of policy gradient algorithms used in state-of-the-art autonomous driving RL research (e.g., model-based Actor-Critic methods for end-to-end driving). Using PPO for trading creates a natural comparison point: both domains require sequential decision-making under uncertainty, both penalize catastrophic failures (crashes / portfolio blow-ups), and both benefit from the variance reduction that GAE provides.

### 5. Sample Efficiency Trade-off
DQN is often more sample-efficient due to its replay buffer, but PPO compensates with **on-policy stability**. In trading, where the non-stationarity of market data means old experiences quickly become misleading, PPO's on-policy nature is arguably an advantage — it only learns from data generated by the current policy.

---

## Project Structure

```
rl-trading-agent/
│
├── README.md                      # This file
├── requirements.txt               # Pinned dependencies for reproducibility
├── .gitignore                     # Excludes data, checkpoints, and artifacts
│
├── configs/
│   ├── __init__.py                # Config loader utilities
│   └── default.yaml               # All hyperparameters (nothing hardcoded)
│
├── data/
│   ├── __init__.py                # Data module init
│   ├── download_data.py           # Downloads & preprocesses OHLCV data via yfinance
│   └── raw/                       # Downloaded CSV files (git-ignored)
│       ├── AAPL_train.csv
│       ├── AAPL_val.csv
│       └── AAPL_test.csv
│
├── env/
│   ├── __init__.py                # Environment module init
│   └── trading_env.py             # Custom Gymnasium TradingEnv (core of project)
│
├── agent/
│   ├── __init__.py                # Agent module init
│   ├── ppo_agent.py               # PPO wrapper using Stable-Baselines3
│   └── baselines.py               # BuyAndHold + RandomAgent baselines
│
├── utils/
│   ├── __init__.py                # Utility module init
│   ├── metrics.py                 # Sharpe, max drawdown, win rate, Calmar
│   └── plot.py                    # Training curves, equity curves, action dist.
│
├── train.py                       # Training entry point
├── evaluate.py                    # Evaluation and benchmarking entry point
│
├── checkpoints/                   # Saved model weights (git-ignored)
├── results/                       # Output plots and metric tables (git-ignored)
└── runs/                          # TensorBoard logs (git-ignored)
```

---

## Results

> *Populate this table after running `python evaluate.py`. All metrics are computed on the **out-of-sample test set** (2023-01-01 to 2023-12-31).*

| Agent | Annual Return | Sharpe Ratio | Max Drawdown | Win Rate | Calmar Ratio |
|---|---|---|---|---|---|
| **PPO** | — | — | — | — | — |
| **Buy-and-Hold** | — | — | — | — | — |
| **Random** | — | — | — | — | — |

**Metric Definitions:**

| Metric | Formula | Interpretation |
|---|---|---|
| Annual Return | `(V_final / V_initial) − 1` | Total percentage gain over the test year |
| Sharpe Ratio | `mean(daily_returns) / std(daily_returns) × √252` | Risk-adjusted return (> 1.0 is good, > 2.0 is excellent) |
| Max Drawdown | `max(peak − trough) / peak` | Worst peak-to-trough loss (lower is better) |
| Win Rate | `count(profitable_days) / count(total_days)` | Fraction of days with positive returns |
| Calmar Ratio | `Annual Return / Max Drawdown` | Return per unit of tail risk |

---

## Design Decisions

Every non-obvious choice in this project is documented here.

### 1. Log Returns in Reward (not Arithmetic Returns)

```python
reward = log(portfolio_value_t / portfolio_value_{t-1})
```

Log returns are **time-additive** (they sum across periods), which aligns with how the RL discount factor aggregates rewards. Arithmetic returns are multiplicative and can mislead the value function — a +50% gain followed by a −50% loss is a net −25% arithmetically but 0 in log space, which correctly reflects the asymmetry. Log returns also have better statistical properties (closer to normally distributed), which stabilizes gradient estimation.

### 2. Transaction Cost Penalty (Prevents Overtrading)

Without a transaction cost term, the agent learns to trade on every tiny price fluctuation — a policy that looks profitable in training (no costs) but is catastrophic in production (death by a thousand cuts). The 10-basis-point cost (`transaction_cost: 0.001`) is realistic for retail equity trading and forces the agent to only trade when the expected move exceeds the round-trip cost.

### 3. Test Set is 2023 Only (Strict Out-of-Sample)

| Split | Period | Purpose |
|---|---|---|
| Train | 2010–2020 | Policy learning |
| Validation | 2021–2022 | Hyperparameter tuning, early stopping |
| Test | 2023 | Final evaluation (touched **once**) |

The test set is a single calendar year that the model never sees during training or tuning. This prevents **lookahead bias** — the most common and most damaging mistake in quantitative finance backtesting. The year 2023 includes both the banking crisis (March) and the AI-driven rally (H2), providing a diverse macro environment.

### 4. Normalization from Train Set Only

Feature normalization (z-scores) is computed **exclusively** on training data. Using test-set statistics would leak future information into the model — a subtle but critical form of lookahead bias. At evaluation time, the same training-set mean and standard deviation are applied to normalize test observations.

### 5. PPO over Other RL Algorithms

| Algorithm | Type | Why Not |
|---|---|---|
| DQN | Value-based | Action aliasing, off-policy staleness (see above) |
| A2C | Actor-Critic | No clipping — prone to destructive updates |
| SAC | Actor-Critic (continuous) | Designed for continuous action spaces; overkill for `Discrete(3)` |
| TD3 | Actor-Critic (continuous) | Same as SAC — continuous action assumption |
| TRPO | Policy gradient | Computationally expensive second-order optimization |
| **PPO** | **Actor-Critic** | **Clipped objective, on-policy stability, SB3 support, same family as autonomous driving RL** |

### 6. 20-Day Lookback Window

The lookback window of 20 trading days (≈ 1 calendar month) is chosen for three reasons:
- **Technical analysis convention:** Most standard indicators (20-day SMA, 14-day RSI, Bollinger Bands) are defined on this timescale.
- **Observation space manageability:** 20 × 9 features + 2 portfolio features = 182 dimensions — large enough to capture patterns, small enough for a 2-layer MLP to learn efficiently.
- **Stationarity window:** Financial data becomes increasingly non-stationary over longer horizons. Twenty days captures short-term momentum and mean-reversion signals without introducing excessive regime-change noise.

---

## Known Limitations

| Limitation | Impact | Potential Mitigation |
|---|---|---|
| **Single asset** | No diversification benefit; fully exposed to idiosyncratic risk | Extend to multi-asset with portfolio weight vector actions |
| **Daily granularity** | Misses intraday dynamics, gaps at open | Use minute-level data with appropriate infrastructure |
| **No short selling** | Cannot profit from downturns; can only hold or exit | Add `Short` action and margin accounting |
| **Binary position sizing** | Always 100% in or 100% out; no fractional positions | Continuous action space with SAC/TD3 |
| **Single-ticker evaluation** | AAPL 2023 had specific macro tailwinds (AI narrative) | Test across sectors, cap sizes, and market regimes |
| **No market impact model** | Assumes infinite liquidity and zero slippage | Add volume-dependent slippage model |
| **Research project** | Not battle-tested in live markets | **This is not financial advice** |

> [!CAUTION]
> This is an academic research project demonstrating reinforcement learning concepts applied to financial data. It is **not** investment advice. Past performance on historical data does not guarantee future results. Do not use this system for real trading without extensive additional validation.

---

## How to Run

### Prerequisites

- Python 3.10 or later
- macOS (Apple Silicon or Intel), Linux, or Windows
- ~2 GB disk space for data and checkpoints

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd rl-trading-agent

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download market data
python data/download_data.py

# 5. Train the PPO agent
python train.py --config configs/default.yaml

# 6. Evaluate and generate results
python evaluate.py
```

### Monitor Training

```bash
# Launch TensorBoard to visualize training curves
tensorboard --logdir runs/
```

Then open [http://localhost:6006](http://localhost:6006) in your browser to monitor:
- Episode reward over time
- Policy entropy (exploration vs. exploitation)
- Value function loss
- Explained variance

---

## Configuration

**All** hyperparameters are centralized in [`configs/default.yaml`](configs/default.yaml). No magic numbers exist in the source code — every configurable value is read from this file at runtime.

```yaml
# PPO Hyperparameters (Schulman et al., 2017 defaults with minor tuning)
learning_rate: 0.0003          # Adam optimizer LR
n_steps: 2048                  # Rollout buffer size per update
batch_size: 64                 # Minibatch size per gradient step
n_epochs: 10                   # Passes over buffer per update
gamma: 0.99                    # Discount factor
gae_lambda: 0.95               # GAE lambda for advantage estimation
clip_range: 0.2                # PPO clipping range
total_timesteps: 500000        # Total environment steps

# Environment
lookback_window: 20            # Days of history per observation
initial_portfolio_value: 10000 # Starting capital (USD)
transaction_cost: 0.001        # 10 bps per trade
volatility_penalty_weight: 0.1 # Volatility penalty in reward

# Data Splits
ticker: "AAPL"
train_start: "2010-01-01"
train_end: "2020-12-31"
val_start: "2021-01-01"
val_end: "2022-12-31"
test_start: "2023-01-01"
test_end: "2023-12-31"

# Reproducibility
seed: 42
```

To experiment with different configurations, copy `default.yaml`, modify values, and pass the new file:

```bash
python train.py --config configs/my_experiment.yaml
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| **stable-baselines3** | 2.3.2 | PPO implementation with Gymnasium integration |
| **gymnasium** | 0.29.1 | RL environment API (successor to OpenAI Gym) |
| **torch** | 2.2.2 | Neural network backend; auto-detects GPU/CPU |
| **yfinance** | 0.2.38 | Downloads historical OHLCV data from Yahoo Finance |
| **pandas** | 2.2.2 | Time-series data manipulation and alignment |
| **numpy** | 1.26.4 | Numerical computation and array operations |
| **matplotlib** | 3.8.4 | Performance visualization (equity curves, drawdowns) |
| **pyyaml** | 6.0.1 | Configuration file parsing |
| **scikit-learn** | 1.4.2 | Feature normalization (StandardScaler) |
| **tensorboard** | 2.16.2 | Real-time training metrics visualization |

All versions are pinned in `requirements.txt` for full reproducibility. The project is compatible with **macOS** (Apple Silicon M-series and Intel), **Linux**, and **Windows**. PyTorch automatically uses GPU acceleration when CUDA is available and falls back to CPU otherwise — no code changes required.

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438.
3. Mnih, V., et al. (2013). *Playing Atari with Deep Reinforcement Learning*. arXiv:1312.5602.
4. Jiang, Z., Xu, D., & Liang, J. (2017). *A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem*. arXiv:1706.10059.

---

<p align="center">
  <em>Built as a research project exploring the intersection of reinforcement learning and quantitative finance.</em>
</p>
