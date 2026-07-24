"""
broker.py — Execution backends behind a single interface.

Two implementations, chosen at runtime:

* ``SimBroker``   — a local simulated broker. Holds cash + positions in a JSON
                    file and fills orders immediately at the supplied price
                    (minus transaction costs). Needs no account, no keys, no
                    network for execution. This is the default so the agent can
                    "trade on its own" the moment you run it.

* ``AlpacaBroker``— a thin wrapper over Alpaca's PAPER trading API. Same
                    interface; real (fake-money) order routing and fills.

Safety: ``AlpacaBroker`` hard-codes ``paper=True``. There is no live-money code
path in this file. Going live would require deliberately constructing the client
with ``paper=False`` — which this project never does.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """A single order. ``qty`` may be fractional; ``side`` is 'buy' or 'sell'."""

    symbol: str
    side: str   # "buy" | "sell"
    qty: float


class Broker(Protocol):
    """Minimal execution interface the trader relies on."""

    name: str

    def get_equity(self, prices: Dict[str, float]) -> float: ...
    def get_positions(self) -> Dict[str, float]: ...
    def execute(self, orders: List[Order], prices: Dict[str, float]) -> None: ...
    def liquidate_all(self, prices: Dict[str, float]) -> None: ...


# --------------------------------------------------------------------------- #
#  SimBroker — local, no account required
# --------------------------------------------------------------------------- #

class SimBroker:
    """Simulated broker with JSON-persisted cash and positions.

    Parameters
    ----------
    state_path : Path
        Where to persist ``{cash, positions}`` between runs.
    initial_cash : float
        Starting cash if no state file exists yet.
    transaction_cost : float
        Proportional cost charged on every fill (each side of a round-trip).
    """

    name = "sim"

    def __init__(
        self, state_path: Path, initial_cash: float, transaction_cost: float
    ) -> None:
        self.state_path = state_path
        self.transaction_cost = float(transaction_cost)
        self._state = self._load(initial_cash)

    # ---- persistence ------------------------------------------------------
    def _load(self, initial_cash: float) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        logger.info("SimBroker: new account funded with $%.2f", initial_cash)
        return {"cash": float(initial_cash), "positions": {}}

    def _save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2))

    # ---- reads ------------------------------------------------------------
    def get_positions(self) -> Dict[str, float]:
        return dict(self._state["positions"])

    def get_cash(self) -> float:
        return float(self._state["cash"])

    def get_equity(self, prices: Dict[str, float]) -> float:
        equity = self.get_cash()
        for sym, qty in self._state["positions"].items():
            equity += qty * prices.get(sym, 0.0)
        return equity

    # ---- writes -----------------------------------------------------------
    def execute(self, orders: List[Order], prices: Dict[str, float]) -> None:
        for o in orders:
            price = prices.get(o.symbol)
            if price is None or price <= 0:
                logger.warning("SimBroker: no price for %s — skipping.", o.symbol)
                continue
            notional = o.qty * price
            cost = notional * self.transaction_cost
            pos = self._state["positions"]
            if o.side == "buy":
                self._state["cash"] -= notional + cost
                pos[o.symbol] = pos.get(o.symbol, 0.0) + o.qty
            else:  # sell
                self._state["cash"] += notional - cost
                pos[o.symbol] = pos.get(o.symbol, 0.0) - o.qty
                if abs(pos[o.symbol]) < 1e-9:
                    pos.pop(o.symbol, None)
            logger.info(
                "SimBroker FILL: %-4s %8.4f %s @ %.2f (cost $%.2f)",
                o.side.upper(), o.qty, o.symbol, price, cost,
            )
        self._save()

    def liquidate_all(self, prices: Dict[str, float]) -> None:
        orders = [
            Order(symbol=s, side="sell", qty=q)
            for s, q in self._state["positions"].items()
            if q > 0
        ]
        if orders:
            logger.warning("SimBroker: liquidating all %d positions.", len(orders))
            self.execute(orders, prices)


# --------------------------------------------------------------------------- #
#  AlpacaBroker — paper trading via Alpaca
# --------------------------------------------------------------------------- #

class AlpacaBroker:
    """Alpaca PAPER trading adapter. Real order routing, fake money.

    Constructed only when API keys are present. ``paper=True`` is fixed.
    """

    name = "alpaca"

    def __init__(self, api_key: str, secret_key: str) -> None:
        # Imported lazily so the project runs without alpaca-py installed.
        from alpaca.trading.client import TradingClient

        self._client = TradingClient(api_key, secret_key, paper=True)
        acct = self._client.get_account()
        logger.info(
            "AlpacaBroker connected (PAPER) — equity $%.2f, status=%s",
            float(acct.equity), acct.status,
        )

    def get_positions(self) -> Dict[str, float]:
        return {
            p.symbol: float(p.qty) for p in self._client.get_all_positions()
        }

    def get_equity(self, prices: Dict[str, float]) -> float:
        # prices ignored — Alpaca reports authoritative account equity.
        return float(self._client.get_account().equity)

    def is_market_open(self) -> bool:
        return bool(self._client.get_clock().is_open)

    def execute(self, orders: List[Order], prices: Dict[str, float]) -> None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        for o in orders:
            side = OrderSide.BUY if o.side == "buy" else OrderSide.SELL
            req = MarketOrderRequest(
                symbol=o.symbol,
                qty=round(abs(o.qty), 4),     # fractional shares allowed
                side=side,
                time_in_force=TimeInForce.DAY,  # required for fractional
            )
            try:
                self._client.submit_order(req)
                logger.info(
                    "AlpacaBroker SUBMIT: %-4s %.4f %s (market, DAY)",
                    o.side.upper(), abs(o.qty), o.symbol,
                )
            except Exception as exc:
                logger.error("AlpacaBroker order failed for %s: %s", o.symbol, exc)

    def liquidate_all(self, prices: Dict[str, float]) -> None:
        logger.warning("AlpacaBroker: closing ALL positions + cancelling orders.")
        self._client.close_all_positions(cancel_orders=True)


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #

def make_broker(cfg: dict, state_dir: Path) -> Broker:
    """Construct the broker selected by config + environment.

    Selection (``cfg['broker']``):
        * "alpaca" -> require keys, build AlpacaBroker.
        * "sim"    -> always SimBroker.
        * "auto"   -> AlpacaBroker if keys present, else SimBroker.

    Falls back to SimBroker (with a warning) if Alpaca is requested but
    unavailable, so an unattended run never crashes for want of credentials.
    """
    choice = str(cfg.get("broker", "auto")).lower()
    api_key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_SECRET_KEY")
    have_keys = bool(api_key and secret and "your_paper" not in (api_key or ""))

    def _sim() -> Broker:
        return SimBroker(
            state_path=state_dir / "sim_account.json",
            initial_cash=float(cfg.get("initial_cash", 100_000)),
            transaction_cost=float(cfg["risk"]["transaction_cost"]),
        )

    if choice == "sim":
        return _sim()

    if choice in ("alpaca", "auto") and have_keys:
        try:
            return AlpacaBroker(api_key, secret)  # type: ignore[arg-type]
        except Exception as exc:
            logger.error("Alpaca init failed (%s) — falling back to SimBroker.", exc)
            return _sim()

    if choice == "alpaca":
        logger.error(
            "broker=alpaca but no API keys found in environment. "
            "Set ALPACA_API_KEY / ALPACA_SECRET_KEY (see .env.example). "
            "Falling back to SimBroker for now."
        )
    else:
        logger.info("No Alpaca keys found — using SimBroker (local simulation).")
    return _sim()
