"""
Download and preprocess daily OHLCV data for the RL trading agent.

Pipeline stages:
    1. Download raw OHLCV data from Yahoo Finance via yfinance.
    2. Compute derived features (return, volatility, momentum signals).
    3. Split into train / validation / test sets by calendar date.
    4. Normalize using statistics computed **only** on training data.
    5. Save CSVs and scaler statistics to disk.

Design decisions:
    - All configurable values (tickers, dates, paths) are read from
      ``configs/default.yaml`` so nothing is hardcoded.
    - Normalization statistics are computed exclusively on training data
      to prevent data leakage.  An explicit assertion guards this.
    - Derived features use forward-fill for NaN (first few rows lost to
      rolling windows) rather than zero-fill, because zero-fill distorts
      the distribution of scaled features.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging — library code uses the logging module, never print().
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ====================================================================
# Configuration helpers
# ====================================================================

def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        config_path: Explicit path to a YAML config.  When ``None``,
            defaults to ``configs/default.yaml`` relative to the project
            root (two levels up from this file).

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        # Navigate from data/ up to project root, then into configs/
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "configs" / "default.yaml"

    with open(config_path, "r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    logger.info("Loaded config from %s", config_path)
    return cfg


# ====================================================================
# Data download
# ====================================================================

# All tickers the pipeline supports — used for batch download validation.
SUPPORTED_TICKERS: list[str] = ["AAPL", "MSFT", "GOOGL"]


def download_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance.

    Args:
        ticker: Stock symbol (e.g. ``"AAPL"``).
        start_date: ISO-format start date inclusive (e.g. ``"2010-01-01"``).
        end_date: ISO-format end date inclusive (e.g. ``"2023-12-31"``).

    Returns:
        DataFrame indexed by ``Date`` with columns
        ``[Open, High, Low, Close, Volume]``.

    Raises:
        ValueError: If the download returns an empty DataFrame.
    """
    logger.info(
        "Downloading %s OHLCV data from %s to %s …",
        ticker,
        start_date,
        end_date,
    )

    # yfinance `end` is exclusive, so add one day to include end_date.
    df: pd.DataFrame = yf.download(
        ticker,
        start=start_date,
        end=pd.Timestamp(end_date) + pd.Timedelta(days=1),
        auto_adjust=True,   # adjusts for splits/dividends automatically
        progress=False,      # suppress noisy progress bars in pipelines
    )

    if df.empty:
        raise ValueError(
            f"yfinance returned no data for {ticker} "
            f"({start_date} – {end_date}). "
            "Check ticker symbol and date range."
        )

    # Handle potential MultiIndex columns from yfinance
    # (occurs when downloading a single ticker — columns may be tuples)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardise column names to match downstream expectations.
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    logger.info("Downloaded %d rows for %s.", len(df), ticker)
    return df


# ====================================================================
# Feature engineering
# ====================================================================

# Rolling window sizes — read from one place so grep-ability is trivial.
_VOLATILITY_WINDOW: int = 5    # 1-week rolling standard deviation
_MA_SHORT_WINDOW: int = 10     # 2-week moving average
_MA_LONG_WINDOW: int = 20      # 4-week (1 month) moving average


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features on top of raw OHLCV data.

    Features added:
        - ``daily_return``:  ``(Close_t / Close_{t-1}) - 1``
        - ``rolling_vol_5d``: Rolling 5-day std of daily returns.
        - ``ma_ratio_10``:    ``Close / MA_10`` — short-term momentum.
        - ``ma_ratio_20``:    ``Close / MA_20`` — medium-term momentum.

    NaN rows (caused by rolling lookbacks) are **dropped** rather than
    forward-filled so that the model never trains on fabricated values.

    Args:
        df: DataFrame with at least a ``Close`` column, indexed by date.

    Returns:
        DataFrame with original + derived columns, NaN rows removed.
    """
    df = df.copy()  # avoid mutating caller's data

    # Daily simple return: pct change from previous close.
    df["daily_return"] = df["Close"].pct_change()

    # Annualisation not needed — we want the *realised* daily vol window.
    df["rolling_vol_5d"] = df["daily_return"].rolling(
        window=_VOLATILITY_WINDOW
    ).std()

    # Moving-average ratios act as mean-reversion / momentum signals.
    # Values > 1 ⟹ price above its moving average (bullish).
    df["ma_ratio_10"] = df["Close"] / df["Close"].rolling(
        window=_MA_SHORT_WINDOW
    ).mean()

    df["ma_ratio_20"] = df["Close"] / df["Close"].rolling(
        window=_MA_LONG_WINDOW
    ).mean()

    # Drop rows where any feature is NaN (first ~20 rows due to MA_20).
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    logger.info(
        "Feature engineering: dropped %d NaN rows (rolling warm-up).",
        rows_before - rows_after,
    )

    return df


# Canonical ordered list of all feature columns.
FEATURE_COLUMNS: list[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "daily_return",
    "rolling_vol_5d",
    "ma_ratio_10",
    "ma_ratio_20",
]


# ====================================================================
# Temporal splitting
# ====================================================================


def split_by_date(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame into train / val / test by date.

    Splits are performed by calendar date — **never** by random
    shuffling — because financial time-series have strong temporal
    autocorrelation and random splits would cause data leakage.

    Args:
        df: Full DataFrame indexed by ``Date``.
        train_start: Inclusive start of training period.
        train_end: Inclusive end of training period.
        val_start: Inclusive start of validation period.
        val_end: Inclusive end of validation period.
        test_start: Inclusive start of test period.
        test_end: Inclusive end of test period.

    Returns:
        ``(train_df, val_df, test_df)`` — three DataFrames.

    Raises:
        AssertionError: If any split is empty or if periods overlap.
    """
    train = df.loc[train_start:train_end]
    val = df.loc[val_start:val_end]
    test = df.loc[test_start:test_end]

    # ---- Safety checks ----
    assert len(train) > 0, "Training split is empty — check date range."
    assert len(val) > 0, "Validation split is empty — check date range."
    assert len(test) > 0, "Test split is empty — check date range."

    # Ensure no temporal overlap between splits.
    assert train.index.max() < val.index.min(), (
        "Train/Val overlap detected — dates must be strictly sequential."
    )
    assert val.index.max() < test.index.min(), (
        "Val/Test overlap detected — dates must be strictly sequential."
    )

    logger.info(
        "Split sizes → Train: %d | Val: %d | Test: %d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


# ====================================================================
# Normalisation (z-score, train-only statistics)
# ====================================================================

# ⚠️  CRITICAL DATA LEAKAGE WARNING ⚠️
# The scaler (mean / std) MUST be computed ONLY on the training set.
# Applying statistics derived from validation or test data would leak
# future information into the model and produce unrealistically
# optimistic backtest results.  The assertions below enforce this
# invariant at runtime.


def compute_scaler_stats(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-feature mean and std from training data **only**.

    Args:
        train_df: Training split DataFrame.
        feature_columns: Ordered list of feature column names.

    Returns:
        Dict mapping feature name → ``{"mean": …, "std": …}``.
    """
    stats: dict[str, dict[str, float]] = {}
    for col in feature_columns:
        col_mean: float = float(train_df[col].mean())
        col_std: float = float(train_df[col].std())

        # Guard against zero std (constant feature) — would cause div/0.
        if col_std == 0.0:
            logger.warning(
                "Feature '%s' has zero std in training data; "
                "setting std to 1.0 to avoid division by zero.",
                col,
            )
            col_std = 1.0

        stats[col] = {"mean": col_mean, "std": col_std}

    logger.info("Computed scaler stats for %d features.", len(stats))
    return stats


def apply_normalisation(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Apply z-score normalisation using pre-computed statistics.

    ``z = (x - mean) / std``

    The statistics **must** come from :func:`compute_scaler_stats`
    fitted on training data only.

    Args:
        df: DataFrame to normalise (may be train, val, or test).
        stats: Per-feature mean/std dict from training data.
        feature_columns: Ordered list of feature column names.

    Returns:
        Normalised DataFrame (copy — original is not mutated).
    """
    df = df.copy()
    for col in feature_columns:
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        df[col] = (df[col] - mean) / std  # standard z-score transform
    return df


def _assert_no_data_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    feature_columns: list[str],
) -> None:
    """Runtime assertions that scaler stats match training data only.

    Verifies that the stored mean/std values correspond to the training
    DataFrame and **not** to the full dataset or to val/test.

    Args:
        train_df: Un-normalised training DataFrame.
        val_df: Un-normalised validation DataFrame.
        test_df: Un-normalised test DataFrame.
        stats: Scaler statistics dictionary.
        feature_columns: Ordered list of feature column names.

    Raises:
        AssertionError: If any statistic does not match the training
            set within floating-point tolerance.
    """
    # ⚠️  DATA LEAKAGE CHECK — DO NOT REMOVE ⚠️
    for col in feature_columns:
        expected_mean = float(train_df[col].mean())
        expected_std = float(train_df[col].std())

        # Tolerance for floating-point comparison.
        assert np.isclose(stats[col]["mean"], expected_mean, rtol=1e-6), (
            f"Scaler mean for '{col}' does not match training data. "
            f"Possible data leakage!"
        )
        assert np.isclose(stats[col]["std"], expected_std, rtol=1e-6), (
            f"Scaler std for '{col}' does not match training data. "
            f"Possible data leakage!"
        )

    # Verify stats do NOT match full dataset (train+val+test combined).
    full_df = pd.concat([train_df, val_df, test_df])
    mismatch_count = 0
    for col in feature_columns:
        full_mean = float(full_df[col].mean())
        if not np.isclose(stats[col]["mean"], full_mean, rtol=1e-6):
            mismatch_count += 1

    # At least some features should differ between train-only and full.
    # (Volume and price levels shift substantially year-over-year.)
    assert mismatch_count > 0, (
        "Scaler stats match full dataset — this suggests they were "
        "computed on val/test data as well.  DATA LEAKAGE DETECTED."
    )

    logger.info("✓ Data leakage assertions passed.")


# ====================================================================
# I/O helpers
# ====================================================================


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ticker: str,
    output_dir: Path,
) -> None:
    """Save normalised train/val/test DataFrames to CSV.

    Args:
        train_df: Normalised training DataFrame.
        val_df: Normalised validation DataFrame.
        test_df: Normalised test DataFrame.
        ticker: Stock symbol used in filenames.
        output_dir: Directory for CSV output (e.g. ``data/raw/``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        path = output_dir / f"{ticker}_{split_name}.csv"
        split_df.to_csv(path)
        logger.info("Saved %s → %s (%d rows)", split_name, path, len(split_df))


def save_scaler_stats(
    stats: dict[str, dict[str, float]],
    path: Path,
) -> None:
    """Persist scaler statistics to a JSON file.

    Args:
        stats: Per-feature mean/std dictionary.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Scaler stats saved to %s", path)


# ====================================================================
# Main pipeline
# ====================================================================


def main(config_path: Path | None = None) -> None:
    """Run the full data download-and-preprocessing pipeline.

    Steps:
        1. Load configuration.
        2. Download OHLCV data for the configured ticker.
        3. Compute derived features.
        4. Split by calendar date into train / val / test.
        5. Compute scaler stats on training data **only**.
        6. Assert no data leakage.
        7. Normalise all splits with training statistics.
        8. Save CSVs and scaler stats to disk.
        9. Print human-readable summary.

    Args:
        config_path: Optional override for the YAML config file.
    """
    # ---- Logging setup (root level for script entry point) ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = _load_config(config_path)

    # ---- Extract data params from config ----
    ticker: str = cfg["ticker"]
    train_start: str = cfg["train_start"]
    train_end: str = cfg["train_end"]
    val_start: str = cfg["val_start"]
    val_end: str = cfg["val_end"]
    test_start: str = cfg["test_start"]
    test_end: str = cfg["test_end"]
    seed: int = cfg["seed"]

    # Set random seed for reproducibility (numpy used in feature eng).
    np.random.seed(seed)

    # ---- Resolve output paths relative to this file's directory ----
    data_dir = Path(__file__).resolve().parent
    raw_dir = data_dir / "raw"
    scaler_path = data_dir / "scaler_stats.json"

    # ---- Step 1: Download ----
    # Use full date range (earliest start to latest end) in one call
    # to avoid gaps from separate downloads per split.
    full_df = download_ohlcv(
        ticker=ticker,
        start_date=train_start,  # earliest date across all splits
        end_date=test_end,       # latest date across all splits
    )

    # ---- Step 2: Feature engineering ----
    full_df = compute_features(full_df)

    # ---- Step 3: Temporal split ----
    train_df, val_df, test_df = split_by_date(
        full_df,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )

    # ---- Step 4: Compute scaler on TRAINING DATA ONLY ----
    # ⚠️  NEVER fit the scaler on val or test — this is data leakage. ⚠️
    stats = compute_scaler_stats(train_df, FEATURE_COLUMNS)

    # ---- Step 5: Verify no leakage ----
    _assert_no_data_leakage(
        train_df, val_df, test_df, stats, FEATURE_COLUMNS
    )

    # ---- Step 6: Normalise all splits with train-only stats ----
    train_norm = apply_normalisation(train_df, stats, FEATURE_COLUMNS)
    val_norm = apply_normalisation(val_df, stats, FEATURE_COLUMNS)
    test_norm = apply_normalisation(test_df, stats, FEATURE_COLUMNS)

    # ---- Step 6b: Preserve raw Close prices for reward computation ----
    # The TradingEnv needs un-normalised close prices to compute log-returns
    # and portfolio value correctly.  Normalised close prices can be negative
    # (z-score), which breaks log(price_t / price_{t-1}).
    train_norm["raw_close"] = train_df["Close"].values
    val_norm["raw_close"] = val_df["Close"].values
    test_norm["raw_close"] = test_df["Close"].values

    # ---- Step 7: Save to disk ----
    save_splits(train_norm, val_norm, test_norm, ticker, raw_dir)
    save_scaler_stats(stats, scaler_path)

    # ---- Step 8: Print summary (visible even if logging is muted) ----
    summary_lines = [
        f"Train: {len(train_norm)} rows | "
        f"Val: {len(val_norm)} rows | "
        f"Test: {len(test_norm)} rows",
        f"Features: {FEATURE_COLUMNS}",
        f"Scaler stats saved to {scaler_path}",
    ]
    for line in summary_lines:
        logger.info(line)
        print(line)


if __name__ == "__main__":
    main()
