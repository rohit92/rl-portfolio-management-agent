#!/bin/bash
# Live-ish signal poller: generate watchlist signals and push NEW ones to
# Telegram/WhatsApp. Scheduled by launchd (e.g. every 15 min). De-dups so you
# only get a message when a setup first appears.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ../venv/bin/activate
mkdir -p state
exec python live_signals.py --notify >> state/signals.out 2>> state/signals.err
