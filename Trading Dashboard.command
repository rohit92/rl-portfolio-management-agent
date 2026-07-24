#!/bin/bash
# =========================================================================
#  Trading Dashboard — one-click launcher
#  Double-click this file in Finder, OR run it from a terminal:
#      ~/rl-trading-agent/"Trading Dashboard.command"
#  It launches TradingView's data bridge, starts the dashboard, and opens
#  it in your browser. Close this window (or press Ctrl-C) to stop.
# =========================================================================
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/live_agent" || exit 1

PY="$ROOT/venv/bin/python"
[ -x "$PY" ] || PY="python3"

printf '\n\033[1;31m●\033[0m  Trading Dashboard — starting…\n\n'

# 1) TradingView Desktop with the CDP data bridge (optional, if installed)
if [ -d "/Applications/TradingView.app" ]; then
  if ! curl -s --max-time 2 http://127.0.0.1:9222/json/version >/dev/null 2>&1; then
    echo "📡  Launching TradingView (real-time bridge on :9222)…"
    open -a TradingView --args --remote-debugging-port=9222 >/dev/null 2>&1 || true
  else
    echo "📡  TradingView bridge already connected."
  fi
else
  echo "ℹ️   TradingView Desktop not installed — dashboard uses its own live feeds."
fi

# 2) Already running? Just open it.
if curl -s --max-time 2 http://127.0.0.1:5000/ >/dev/null 2>&1; then
  echo "✅  Dashboard already running → opening browser."
  open "http://127.0.0.1:5000"
  echo "    (a server is already up in another window; this one will exit)"
  exit 0
fi

# 3) Open the browser once the server answers, then run the server here.
(
  for _ in $(seq 1 40); do
    curl -s --max-time 1 http://127.0.0.1:5000/ >/dev/null 2>&1 && break
    sleep 0.5
  done
  open "http://127.0.0.1:5000"
) &

echo "🚀  Dashboard → http://127.0.0.1:5000"
echo "    (leave this window open; press Ctrl-C or close it to stop)"
echo
exec "$PY" webapp.py
