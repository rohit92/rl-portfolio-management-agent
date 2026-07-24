"""
markets.py — Resolve a named market/timeframe preset into the working config.

Keeps the rest of the codebase market-agnostic: every module reads
``cfg['universe']``, ``cfg['interval']``, ``cfg['periods_per_year']`` and
``cfg['risk']['transaction_cost']`` without caring whether it's US daily, crypto
hourly, or NSE daily. This function is the single place that maps a preset name
(from ``config.yaml`` ``markets:``) onto those keys.
"""

from __future__ import annotations

import copy
import logging
from typing import List

logger = logging.getLogger(__name__)


def list_markets(cfg: dict) -> List[str]:
    """Names of all configured market presets."""
    return list((cfg.get("markets") or {}).keys())


def resolve(cfg: dict, name: str | None = None) -> dict:
    """Return a copy of ``cfg`` with the chosen market preset applied.

    Parameters
    ----------
    cfg : dict
        Full config (must contain a ``markets`` mapping to do anything).
    name : str, optional
        Preset to apply. Defaults to ``cfg['active_market']``.

    Returns
    -------
    dict
        Deep-copied config with ``universe``, ``interval``,
        ``periods_per_year`` and ``risk.transaction_cost`` set from the preset.
        If no markets are configured the original config is returned unchanged
        (with sane interval/annualisation defaults filled in).
    """
    out = copy.deepcopy(cfg)
    markets = out.get("markets") or {}
    name = name or out.get("active_market")

    if not markets or name not in markets:
        # No presets — fall back to whatever universe is present, daily.
        out.setdefault("interval", "1d")
        out.setdefault("periods_per_year", 252)
        if name and markets:
            logger.warning("Unknown market '%s' — using existing config.", name)
        return out

    preset = markets[name]
    out["active_market"] = name
    out["universe"] = list(preset["symbols"])
    out["interval"] = preset.get("interval", "1d")
    out["periods_per_year"] = int(preset.get("periods_per_year", 252))
    if "transaction_cost" in preset:
        out["risk"]["transaction_cost"] = float(preset["transaction_cost"])

    logger.info(
        "Market '%s': %d symbols, interval=%s, periods/yr=%d, cost=%.4f",
        name, len(out["universe"]), out["interval"],
        out["periods_per_year"], out["risk"]["transaction_cost"],
    )
    return out
