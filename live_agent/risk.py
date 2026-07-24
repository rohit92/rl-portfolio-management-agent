"""
risk.py — Position sizing and the risk overlay.

Turns raw ensemble scores into concrete target portfolio weights, then enforces
hard limits. This is the layer that keeps an autonomous agent from doing
something stupid with (paper) money. Three independent guardrails:

1. Inverse-volatility sizing — risk parity, lite. A name's weight is scaled by
   1/volatility so each position contributes a *similar* amount of risk. Without
   this, the highest-vol name (e.g. NVDA) silently dominates portfolio risk.

2. Hard caps — no single name above ``max_position_weight``; total invested
   never exceeds ``max_gross_exposure`` (1.0 == no leverage).

3. Kill-switch — a high-water-mark drawdown trip. If equity falls more than
   ``daily_loss_halt`` below its peak, the agent liquidates everything and
   refuses to trade until manually reset. This is the seatbelt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from strategies import AssetScore

logger = logging.getLogger(__name__)


def target_weights(
    scores: Dict[str, AssetScore], cfg: dict
) -> Dict[str, float]:
    """Convert ensemble scores into target portfolio weights.

    Pipeline: threshold -> inverse-vol scaling -> per-name cap -> gross cap.

    Parameters
    ----------
    scores : dict[str, AssetScore]
        Composite scores from the ensemble.
    cfg : dict
        Full config (uses the ``risk`` section).

    Returns
    -------
    dict[str, float]
        Symbol -> target weight in [0, max_position_weight], summing to at most
        ``max_gross_exposure``. Long-only: names below threshold get 0.
    """
    risk = cfg["risk"]
    entry = float(risk["entry_threshold"])
    max_pos = float(risk["max_position_weight"])
    max_gross = float(risk["max_gross_exposure"])

    # 1) Keep only sufficiently attractive long candidates.
    raw: Dict[str, float] = {}
    for sym, asc in scores.items():
        if asc.composite < entry:
            continue
        vol = max(asc.volatility, 1e-4)  # floor to avoid divide-by-zero blowups
        # Conviction (score) scaled by inverse volatility.
        raw[sym] = asc.composite / vol

    if not raw:
        logger.info("No symbols cleared the entry threshold — target is cash.")
        return {}

    # 2) Normalise raw allocations to sum to gross exposure.
    total = sum(raw.values())
    weights = {s: (v / total) * max_gross for s, v in raw.items()}

    # 3) Apply the per-name cap, then redistribute any spilled weight to
    #    uncapped names (iteratively, so the gross target is preserved).
    weights = _apply_position_cap(weights, max_pos, max_gross)

    logger.info(
        "Target weights: %s (gross %.2f)",
        {k: round(v, 3) for k, v in weights.items()},
        sum(weights.values()),
    )
    return weights


def _apply_position_cap(
    weights: Dict[str, float], max_pos: float, max_gross: float
) -> Dict[str, float]:
    """Cap each weight at ``max_pos``, redistributing excess to uncapped names.

    Iterates to a fixed point so the total stays as close to ``max_gross`` as
    the cap allows. If every name is capped, gross may fall below the target
    (that's correct — the caps bind).
    """
    capped = dict(weights)
    for _ in range(100):  # converges in a few passes; bound for safety
        over = {s: w for s, w in capped.items() if w > max_pos + 1e-12}
        if not over:
            break
        # Spill = total weight above the cap, to be redistributed.
        spill = sum(w - max_pos for s, w in over.items())
        for s in over:
            capped[s] = max_pos
        uncapped = [s for s in capped if capped[s] < max_pos - 1e-12]
        if not uncapped:
            break  # everything is at the cap; nothing to redistribute into
        base = sum(capped[s] for s in uncapped)
        if base <= 0:
            # Distribute evenly if the uncapped names have zero base weight.
            for s in uncapped:
                capped[s] = min(max_pos, capped[s] + spill / len(uncapped))
        else:
            for s in uncapped:
                capped[s] = min(max_pos, capped[s] + spill * capped[s] / base)
    return capped


@dataclass
class KillSwitch:
    """High-water-mark drawdown kill-switch with persistent state.

    Tracks peak equity across runs in a small JSON file. When current equity
    drops ``threshold`` below the peak, ``check`` returns ``tripped=True`` and
    latches — the agent should liquidate and stop trading until the state file
    is reset (delete it, or call :meth:`reset`).
    """

    state_path: Path
    threshold: float

    def _load(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"high_water_mark": 0.0, "halted": False}

    def _save(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    def check(self, equity: float) -> "KillSwitchResult":
        """Update the high-water mark and evaluate the drawdown trip.

        Parameters
        ----------
        equity : float
            Current total account equity.

        Returns
        -------
        KillSwitchResult
            Whether the switch is tripped and the supporting numbers.
        """
        state = self._load()
        hwm = max(float(state.get("high_water_mark", 0.0)), equity)

        drawdown = 0.0 if hwm <= 0 else (hwm - equity) / hwm
        tripped = bool(state.get("halted", False)) or drawdown >= self.threshold

        state["high_water_mark"] = hwm
        state["halted"] = tripped
        self._save(state)

        if tripped:
            logger.error(
                "KILL-SWITCH TRIPPED — equity %.2f is %.1f%% below peak %.2f "
                "(limit %.1f%%). Liquidate + halt.",
                equity, drawdown * 100, hwm, self.threshold * 100,
            )
        return KillSwitchResult(tripped=tripped, drawdown=drawdown, hwm=hwm)

    def reset(self) -> None:
        """Clear the halt latch and reset the high-water mark."""
        self._save({"high_water_mark": 0.0, "halted": False})
        logger.info("Kill-switch reset.")


@dataclass
class KillSwitchResult:
    tripped: bool
    drawdown: float
    hwm: float
