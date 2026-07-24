#!/bin/bash
# Nightly: refresh the universe and re-rank both segments so the dashboard's
# Top-picks (Crypto + Indian market) stay fresh. Scheduled by launchd.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ../venv/bin/activate
mkdir -p state
python universe.py            >> state/scan.out 2>> state/scan.err
python scan.py --segment binance >> state/scan.out 2>> state/scan.err
python scan.py --segment nse     >> state/scan.out 2>> state/scan.err
