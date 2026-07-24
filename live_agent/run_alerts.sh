#!/bin/bash
# Check price-level alerts and fire macOS notifications for any that crossed.
# Scheduled by launchd every ~30 minutes.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ../venv/bin/activate
mkdir -p state
exec python alerts.py >> state/alerts.out 2>> state/alerts.err
