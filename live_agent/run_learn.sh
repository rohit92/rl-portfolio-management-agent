#!/bin/bash
# Weekly self-learning pass (invoked by launchd). Full retraining on the latest
# data: refresh the universe, re-fit ensemble weights, and search the committee
# policy per market — each adopted ONLY if it clears the out-of-sample gate.
# This keeps the model ADAPTED over time; it does not guarantee profit.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ../venv/bin/activate
mkdir -p state
{
  echo "=== weekly retrain $(date) ==="
  python universe.py
  python auto_learn.py
  python train_strategy.py --market crypto_daily --years 3
  python train_strategy.py --market us_tech_daily --years 3
  python train_strategy.py --market india_daily --years 3
  python train_strategy.py --market us_weekly --years 5
  echo "=== retrain complete $(date) ==="
} >> state/learn.out 2>> state/learn.err
