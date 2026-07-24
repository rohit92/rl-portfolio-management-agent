#!/bin/bash
# Wrapper invoked by launchd once per trading day. Activates the project venv
# and runs a single decision cycle, appending output to the state logs.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source ../venv/bin/activate
mkdir -p state
exec python trader.py --once >> state/launchd.out 2>> state/launchd.err
