"""
Publication-quality plotting utilities for RL-based portfolio management.

All plots follow a clean, minimal style suitable for research papers and
quant-firm presentations.  Chart-junk (gratuitous gridlines, 3-D effects,
excessive decoration) is strictly avoided.

Design decisions:
    - matplotlib is the sole dependency (no seaborn import needed; the
      built-in ``seaborn-v0_8-darkgrid`` stylesheet is sufficient).
    - Every function saves to disk and explicitly closes the figure to avoid
      memory leaks during long training runs.
    - Colour palette is colour-blind-friendly (tab10 default).
    - DPI set to 150 for a good balance between file size and print quality.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt  # noqa: E402  (import after backend set)
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Module-level logger — all diagnostics go through stdlib logging, not print.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------
# Use a clean, publication-grade style.  "seaborn-v0_8-darkgrid" ships with
# matplotlib ≥ 3.6.  We fall back gracefully if the style is unavailable.
_PREFERRED_STYLE = "seaborn-v0_8-darkgrid"
try:
    plt.style.use(_PREFERRED_STYLE)
    logger.debug("Matplotlib style set to '%s'", _PREFERRED_STYLE)
except OSError:
    logger.warning(
        "Style '%s' not found; falling back to 'ggplot'", _PREFERRED_STYLE
    )
    plt.style.use("ggplot")

# Shared constants — single source of truth for visual consistency.
_DPI: int = 150                  # resolution for saved figures
_FIGURE_WIDTH: float = 12.0     # inches
_FIGURE_HEIGHT_DUAL: float = 8.0   # inches for two-subplot layouts
_FIGURE_HEIGHT_SINGLE: float = 5.0  # inches for single-subplot layouts
_TITLE_FONTSIZE: int = 14
_LABEL_FONTSIZE: int = 12
_LEGEND_FONTSIZE: int = 10


# ========================= plot_equity_curves ==============================

def plot_equity_curves(
    results: Dict[str, List[float]],
    save_path: str,
) -> None:
    """Plot equity curves and drawdowns for multiple strategies.

    Creates a vertically-stacked two-panel figure:

    * **Top panel** — Equity curves for every agent/baseline on a shared
      time axis, allowing direct visual comparison.
    * **Bottom panel** — Per-strategy drawdown from peak, plotted as
      negative fractions for intuitive reading (deeper = worse).

    Parameters
    ----------
    results : Dict[str, List[float]]
        Mapping of strategy name → portfolio value time-series.
        Example: ``{"PPO": [100, 101, ...], "BuyAndHold": [100, 99, ...]}``.
    save_path : str
        Filesystem path (including extension, e.g. ``.png``) where the
        figure will be saved.

    Returns
    -------
    None
        Side effect: writes a figure to *save_path*.

    Raises
    ------
    ValueError
        If *results* is empty.
    """
    if not results:
        raise ValueError("results dict must contain at least one strategy")

    logger.info("Plotting equity curves for %d strategies → %s",
                len(results), save_path)

    fig, (ax_equity, ax_dd) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT_DUAL),
        sharex=True,                          # shared time axis
        gridspec_kw={"height_ratios": [3, 1]},  # equity panel 3× taller
    )

    for name, equity_values in results.items():
        equity = np.asarray(equity_values, dtype=np.float64)
        timesteps = np.arange(len(equity))

        # ---- Equity curve (top panel) ----
        ax_equity.plot(timesteps, equity, label=name, linewidth=1.4)

        # ---- Drawdown curve (bottom panel) ----
        running_peak = np.maximum.accumulate(equity)
        drawdown = (equity - running_peak) / np.where(
            running_peak != 0.0, running_peak, 1.0  # guard div-by-zero
        )
        # drawdown is ≤ 0; plot it that way so "deeper" means worse
        ax_dd.fill_between(timesteps, drawdown, alpha=0.35, label=name)
        ax_dd.plot(timesteps, drawdown, linewidth=0.8)

    # ---- Top-panel cosmetics ----
    ax_equity.set_title(
        "Equity Curves — Strategy Comparison",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax_equity.set_ylabel("Portfolio Value", fontsize=_LABEL_FONTSIZE)
    ax_equity.legend(fontsize=_LEGEND_FONTSIZE, loc="upper left")
    ax_equity.tick_params(labelsize=_LABEL_FONTSIZE - 1)

    # ---- Bottom-panel cosmetics ----
    ax_dd.set_title("Drawdown from Peak", fontsize=_TITLE_FONTSIZE - 1)
    ax_dd.set_ylabel("Drawdown", fontsize=_LABEL_FONTSIZE)
    ax_dd.set_xlabel("Time Step", fontsize=_LABEL_FONTSIZE)
    ax_dd.tick_params(labelsize=_LABEL_FONTSIZE - 1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)  # release memory immediately

    logger.info("Equity-curve figure saved to %s", save_path)


# ========================= plot_training_curve =============================

def plot_training_curve(
    rewards: List[float],
    val_sharpes: List[float],
    save_path: str,
    rolling_window: int = 100,
) -> None:
    """Plot training reward and validation Sharpe ratio over time.

    Two vertically-stacked subplots:

    * **Top** — Rolling mean of episode rewards (smoothed with a centred
      window of size *rolling_window*), revealing learning progress.
    * **Bottom** — Validation-set Sharpe ratio at each evaluation
      checkpoint, showing generalisation quality.

    Parameters
    ----------
    rewards : List[float]
        Raw per-episode (or per-step) reward values collected during
        training.
    val_sharpes : List[float]
        Sharpe ratios measured on the validation set at periodic
        evaluation checkpoints.
    save_path : str
        Filesystem path where the figure will be saved.
    rolling_window : int, optional
        Window size for the rolling mean of rewards, by default ``100``.

    Returns
    -------
    None
        Side effect: writes a figure to *save_path*.
    """
    logger.info(
        "Plotting training curve (%d reward samples, %d val checkpoints) → %s",
        len(rewards), len(val_sharpes), save_path,
    )

    fig, (ax_reward, ax_sharpe) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT_DUAL),
    )

    # ---- Top panel: rolling mean reward ----
    reward_arr = np.asarray(rewards, dtype=np.float64)

    if reward_arr.size >= rolling_window:
        # Convolve with a uniform kernel for a centred rolling mean
        kernel = np.ones(rolling_window) / rolling_window
        smoothed = np.convolve(reward_arr, kernel, mode="valid")
        # x-axis offset so the smoothed line aligns with original indices
        x_offset = rolling_window // 2
        x_smooth = np.arange(x_offset, x_offset + len(smoothed))
    else:
        # Not enough data points to smooth — show raw series
        smoothed = reward_arr
        x_smooth = np.arange(len(smoothed))
        logger.warning(
            "Reward series shorter than rolling_window (%d < %d); "
            "showing raw rewards",
            reward_arr.size, rolling_window,
        )

    # Faint raw rewards in background for context
    ax_reward.plot(
        np.arange(len(reward_arr)),
        reward_arr,
        alpha=0.15,
        color="steelblue",
        linewidth=0.5,
        label="Raw reward",
    )
    # Bold smoothed line
    ax_reward.plot(
        x_smooth,
        smoothed,
        color="steelblue",
        linewidth=1.6,
        label=f"Rolling mean (w={rolling_window})",
    )

    ax_reward.set_title(
        "Training Reward", fontsize=_TITLE_FONTSIZE, fontweight="bold",
    )
    ax_reward.set_ylabel("Reward", fontsize=_LABEL_FONTSIZE)
    ax_reward.set_xlabel("Training Step / Episode", fontsize=_LABEL_FONTSIZE)
    ax_reward.legend(fontsize=_LEGEND_FONTSIZE, loc="lower right")
    ax_reward.tick_params(labelsize=_LABEL_FONTSIZE - 1)

    # ---- Bottom panel: validation Sharpe ----
    sharpe_arr = np.asarray(val_sharpes, dtype=np.float64)
    checkpoint_indices = np.arange(len(sharpe_arr))

    ax_sharpe.plot(
        checkpoint_indices,
        sharpe_arr,
        marker="o",
        markersize=4,
        linewidth=1.4,
        color="darkorange",
    )

    ax_sharpe.set_title(
        "Validation Sharpe Ratio", fontsize=_TITLE_FONTSIZE - 1,
    )
    ax_sharpe.set_ylabel("Sharpe Ratio", fontsize=_LABEL_FONTSIZE)
    ax_sharpe.set_xlabel("Evaluation Checkpoint", fontsize=_LABEL_FONTSIZE)
    ax_sharpe.axhline(
        y=0.0, color="grey", linestyle="--", linewidth=0.8,
        label="Sharpe = 0",
    )  # zero-line reference
    ax_sharpe.legend(fontsize=_LEGEND_FONTSIZE, loc="lower right")
    ax_sharpe.tick_params(labelsize=_LABEL_FONTSIZE - 1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("Training-curve figure saved to %s", save_path)


# ======================= plot_action_distribution ==========================

def plot_action_distribution(
    actions: List[int],
    save_path: str,
) -> None:
    """Plot a bar chart of agent action frequencies.

    Visualises how often the agent chose each discrete action
    (Hold / Buy / Sell).  Useful for detecting degenerate policies such as
    "always hold" or excessive overtrading.

    Parameters
    ----------
    actions : List[int]
        Sequence of integer action codes.  Expected mapping:
        ``{0: "Hold", 1: "Buy", 2: "Sell"}``.
    save_path : str
        Filesystem path where the figure will be saved.

    Returns
    -------
    None
        Side effect: writes a figure to *save_path*.
    """
    logger.info(
        "Plotting action distribution (%d actions) → %s",
        len(actions), save_path,
    )

    # Canonical action labels — extend this map if the action space grows
    action_labels: Dict[int, str] = {0: "Hold", 1: "Buy", 2: "Sell"}

    arr = np.asarray(actions, dtype=np.int64)

    # Count occurrences of each known action
    unique_actions, counts = np.unique(arr, return_counts=True)
    action_count_map = dict(zip(unique_actions.tolist(), counts.tolist()))

    # Build ordered bar data aligned with the canonical label order
    labels: List[str] = []
    frequencies: List[int] = []
    for code in sorted(action_labels.keys()):
        labels.append(action_labels.get(code, f"Action {code}"))
        frequencies.append(action_count_map.get(code, 0))

    # Also include any unexpected action codes for robustness
    for code in sorted(action_count_map.keys()):
        if code not in action_labels:
            labels.append(f"Action {code}")
            frequencies.append(action_count_map[code])
            logger.warning("Unexpected action code %d found in actions", code)

    total = sum(frequencies)
    # Compute percentages for annotation (avoid div-by-zero)
    percentages = [
        (f / total * 100.0) if total > 0 else 0.0 for f in frequencies
    ]

    # ---- Create figure ----
    fig, ax = plt.subplots(
        figsize=(_FIGURE_WIDTH * 0.6, _FIGURE_HEIGHT_SINGLE),
    )

    bar_colours = ["#4C72B0", "#55A868", "#C44E52"]  # blue, green, red
    # Extend palette if more bars than colours
    while len(bar_colours) < len(labels):
        bar_colours.append("#8172B2")  # muted purple fallback

    bars = ax.bar(
        labels,
        frequencies,
        color=bar_colours[: len(labels)],
        edgecolor="white",
        linewidth=0.8,
    )

    # Annotate each bar with count and percentage
    for bar, freq, pct in zip(bars, frequencies, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,  # centre of bar
            bar.get_height() + total * 0.008,       # slightly above top
            f"{freq:,}  ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=_LABEL_FONTSIZE - 1,
            fontweight="bold",
        )

    ax.set_title(
        "Agent Action Distribution",
        fontsize=_TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_ylabel("Frequency", fontsize=_LABEL_FONTSIZE)
    ax.set_xlabel("Action", fontsize=_LABEL_FONTSIZE)
    ax.tick_params(labelsize=_LABEL_FONTSIZE - 1)

    # Remove top and right spines for a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)

    logger.info("Action-distribution figure saved to %s", save_path)
