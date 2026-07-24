"""
trader.py — The autonomous agent. This is the entry point.

One decision cycle:

    fetch live data ─▶ score the universe ─▶ size positions (risk overlay)
        ─▶ kill-switch check ─▶ diff vs. current book ─▶ place orders ─▶ log

Run modes
---------
    python trader.py --once              # one cycle, then exit (use with launchd)
    python trader.py --once --dry-run    # decide + log, but place NO orders
    python trader.py --loop              # run forever, fire once per trading day
    python trader.py --reset-killswitch  # clear a tripped halt latch

Broker is chosen by config.yaml (``broker: auto|sim|alpaca``) or overridden with
``--broker``. With no Alpaca keys it uses the local SimBroker, so it works the
instant you run it.

Safety posture
--------------
* Defaults to paper/sim — never real money.
* Hard risk caps + a drawdown kill-switch (see risk.py).
* ``--dry-run`` lets you watch decisions before letting it trade.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import yaml
from dotenv import load_dotenv

import markets
from broker import Order, make_broker
from data_feed import fetch, latest_prices
from risk import KillSwitch, target_weights
from strategies import Ensemble

HERE = Path(__file__).resolve().parent
STATE_DIR = HERE / "state"
logger = logging.getLogger("live_agent")


# --------------------------------------------------------------------------- #
#  Setup helpers
# --------------------------------------------------------------------------- #

def load_config(path: Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def apply_learned_weights(cfg: dict) -> None:
    """Adopt weights produced by auto_learn.py — but only if they were promoted.

    The self-learner writes ``state/learned_weights.json`` with ``adopted: true``
    only when its out-of-sample validation gate passed. If that file is missing,
    not adopted, or malformed, we silently keep the hand-set config defaults.
    This is the safe handshake between offline learning and live trading.
    """
    if not cfg.get("use_learned_weights"):
        return
    path = STATE_DIR / "learned_weights.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("learned_weights.json unreadable — using config defaults.")
        return
    if payload.get("adopted") and isinstance(payload.get("weights"), dict):
        cfg["strategy"]["weights"] = payload["weights"]
        logger.info(
            "Adopted learned weights (median OOS Sharpe %.2f): %s",
            payload.get("median_oos_sharpe", float("nan")), payload["weights"],
        )
    else:
        logger.info("Learned weights present but not promoted — using defaults.")


def setup_logging() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(STATE_DIR / "agent.log"),
        ],
    )


# --------------------------------------------------------------------------- #
#  Order construction: target weights -> concrete orders
# --------------------------------------------------------------------------- #

def compute_orders(
    current: Dict[str, float],
    targets: Dict[str, float],
    equity: float,
    prices: Dict[str, float],
    cfg: dict,
) -> List[Order]:
    """Diff the current book against target weights and emit orders.

    A ``rebalance_band`` deadband suppresses trivial rebalances (and their
    transaction costs). Full exits (target weight 0) always execute regardless
    of the band, so we never leave dust behind.
    """
    band = float(cfg["risk"]["rebalance_band"]) * equity
    symbols = set(current) | set(targets)
    orders: List[Order] = []

    for sym in sorted(symbols):
        price = prices.get(sym)
        if price is None or price <= 0:
            continue
        cur_qty = current.get(sym, 0.0)
        cur_val = cur_qty * price
        tgt_val = targets.get(sym, 0.0) * equity
        diff_val = tgt_val - cur_val

        full_exit = targets.get(sym, 0.0) == 0.0 and cur_qty > 0
        if not full_exit and abs(diff_val) < band:
            continue  # inside the deadband — leave it alone

        qty = diff_val / price
        if abs(qty) < 1e-6:
            continue
        side = "buy" if qty > 0 else "sell"
        # Never sell more than we hold (long-only book).
        if side == "sell":
            qty = min(abs(qty), cur_qty)
            if qty < 1e-6:
                continue
        orders.append(Order(symbol=sym, side=side, qty=abs(qty)))

    return orders


# --------------------------------------------------------------------------- #
#  One decision cycle
# --------------------------------------------------------------------------- #

def run_cycle(cfg: dict, broker, dry_run: bool) -> dict:
    """Execute a single decision cycle. Returns a JSON-able decision record."""
    universe = list(cfg["universe"])

    # 1) Data + features
    data = fetch(universe, cfg)
    if not data:
        logger.error("No market data available — aborting cycle.")
        return {"error": "no_data"}
    prices = latest_prices(data)

    # 2) Score the universe
    ensemble = Ensemble(cfg)
    scores = ensemble.score_all(data)

    # 3) Account snapshot
    equity = broker.get_equity(prices)
    current = broker.get_positions()

    # 4) Kill-switch — liquidate + halt on a drawdown breach
    ks = KillSwitch(
        state_path=STATE_DIR / "killswitch.json",
        threshold=float(cfg["risk"]["daily_loss_halt"]),
    )
    ks_res = ks.check(equity)
    if ks_res.tripped:
        if not dry_run:
            broker.liquidate_all(prices)
        record = _record(broker, dry_run, equity, scores, {}, [], halted=True)
        _print_summary(equity, scores, {}, [], halted=True, dry_run=dry_run)
        _append_decision(record)
        return record

    # 5) Target weights (risk overlay) + orders
    targets = target_weights(scores, cfg)
    orders = compute_orders(current, targets, equity, prices, cfg)

    # 6) Execute
    if dry_run:
        logger.info("[DRY-RUN] Would place %d orders (none sent).", len(orders))
    elif orders:
        broker.execute(orders, prices)
    else:
        logger.info("No orders this cycle — book already within tolerance.")

    record = _record(broker, dry_run, equity, scores, targets, orders, halted=False)
    _print_summary(equity, scores, targets, orders, halted=False, dry_run=dry_run)
    _append_decision(record)
    return record


# --------------------------------------------------------------------------- #
#  Logging / reporting
# --------------------------------------------------------------------------- #

def _record(broker, dry_run, equity, scores, targets, orders, halted) -> dict:
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "broker": broker.name,
        "dry_run": dry_run,
        "halted": halted,
        "equity": round(equity, 2),
        "scores": {s: round(a.composite, 4) for s, a in scores.items()},
        "targets": {s: round(w, 4) for s, w in targets.items()},
        "orders": [{"symbol": o.symbol, "side": o.side, "qty": round(o.qty, 4)}
                   for o in orders],
    }


def _append_decision(record: dict) -> None:
    path = STATE_DIR / "decisions.jsonl"
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def _print_summary(equity, scores, targets, orders, halted, dry_run) -> None:
    tag = "  [DRY-RUN]" if dry_run else ""
    print("\n" + "=" * 64)
    print(f"  DECISION @ {dt.datetime.now():%Y-%m-%d %H:%M}{tag}")
    print(f"  Equity: ${equity:,.2f}")
    if halted:
        print("  *** KILL-SWITCH TRIPPED — liquidated & halted ***")
        print("=" * 64 + "\n")
        return
    print("-" * 64)
    print(f"  {'Symbol':<8}{'Score':>9}{'Target wt':>12}")
    for sym, asc in scores.items():
        w = targets.get(sym, 0.0)
        flag = "  <" if w > 0 else ""
        print(f"  {sym:<8}{asc.composite:>+9.2f}{w:>11.1%}{flag}")
    print("-" * 64)
    if orders:
        print(f"  Orders ({len(orders)}):")
        for o in orders:
            print(f"    {o.side.upper():<4} {o.qty:>9.4f}  {o.symbol}")
    else:
        print("  Orders: none (within rebalance band)")
    print("=" * 64 + "\n")


# --------------------------------------------------------------------------- #
#  Loop scheduling
# --------------------------------------------------------------------------- #

def run_loop(cfg: dict, broker, dry_run: bool) -> None:
    """Run forever; fire one cycle per weekday at the configured local time."""
    decision_time = dt.datetime.strptime(
        cfg["cadence"]["decision_time_local"], "%H:%M"
    ).time()
    poll = int(cfg["cadence"]["poll_seconds"])
    last_run_date = None

    logger.info(
        "Loop started — firing daily at %s local time (Mon–Fri). Ctrl-C to stop.",
        decision_time.strftime("%H:%M"),
    )
    while True:
        now = dt.datetime.now()
        is_weekday = now.weekday() < 5
        due = now.time() >= decision_time and last_run_date != now.date()
        if is_weekday and due:
            logger.info("Scheduled decision time reached — running cycle.")
            run_cycle(cfg, broker, dry_run)
            last_run_date = now.date()
        time.sleep(poll)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Live ensemble trading agent")
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--once", action="store_true",
                        help="Run a single decision cycle and exit.")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously, once per trading day.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute + log decisions but place NO orders.")
    parser.add_argument("--broker", choices=["auto", "sim", "alpaca"],
                        help="Override the broker from config.")
    parser.add_argument("--reset-killswitch", action="store_true",
                        help="Clear a tripped kill-switch and exit.")
    args = parser.parse_args()

    setup_logging()
    load_dotenv(HERE / ".env")  # load Alpaca keys if present
    cfg = load_config(Path(args.config))
    cfg = markets.resolve(cfg)  # apply active market preset (symbols/interval/cost)
    apply_learned_weights(cfg)  # adopt promoted weights from the self-learner
    if args.broker:
        cfg["broker"] = args.broker

    if args.reset_killswitch:
        KillSwitch(STATE_DIR / "killswitch.json",
                   float(cfg["risk"]["daily_loss_halt"])).reset()
        return

    broker = make_broker(cfg, STATE_DIR)
    logger.info("Broker: %s | dry_run=%s", broker.name, args.dry_run)

    if args.loop:
        run_loop(cfg, broker, args.dry_run)
    else:
        # Default to a single cycle if neither flag is given.
        run_cycle(cfg, broker, args.dry_run)


if __name__ == "__main__":
    main()
