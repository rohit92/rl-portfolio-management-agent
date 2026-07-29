"""
Tests for the risk and execution core of the live trading agent.

These cover the code paths where a bug costs (paper) money:

  * broker.SimBroker  — fills, cash accounting, transaction costs, equity
                        mark-to-market, missing-price safety, persistence,
                        and full liquidation.
  * risk.target_weights / _apply_position_cap — position sizing, the per-name
                        cap, and gross-exposure limits.
  * risk.KillSwitch   — the high-water-mark drawdown halt and its latch.

All tests are offline and deterministic: SimBroker and KillSwitch persist to a
temp directory, and AssetScore objects are constructed directly, so nothing
here touches Alpaca, the network, or real market data.

Run (from the live_agent directory):
    ../venv/bin/python -m pytest tests/ -q
    ../venv/bin/python -m unittest discover -s tests -q
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

# live_agent modules import each other by bare name (e.g. `import risk`,
# `from strategies import AssetScore`), so the package dir must be on sys.path.
LIVE_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LIVE_AGENT_DIR not in sys.path:
    sys.path.insert(0, LIVE_AGENT_DIR)

import risk  # noqa: E402
from broker import Order, SimBroker  # noqa: E402
from strategies import AssetScore  # noqa: E402


def _score(symbol, composite, volatility):
    return AssetScore(symbol=symbol, composite=composite,
                      components={}, volatility=volatility)


# =========================================================================== #
#  SimBroker — the paper-trading execution engine
# =========================================================================== #

class TestSimBroker(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "account.json"

    def _broker(self, cash=1000.0, cost=0.0, path=None):
        return SimBroker(state_path=path or self.path,
                         initial_cash=cash, transaction_cost=cost)

    def test_new_account_starts_with_initial_cash(self):
        b = self._broker(cash=5000.0)
        self.assertEqual(b.get_cash(), 5000.0)
        self.assertEqual(b.get_positions(), {})
        self.assertEqual(b.get_equity({}), 5000.0)

    def test_buy_decrements_cash_and_adds_position(self):
        b = self._broker(cash=1000.0)
        b.execute([Order("AAA", "buy", 10)], {"AAA": 10.0})
        self.assertEqual(b.get_cash(), 900.0)          # spent 10 * 10
        self.assertEqual(b.get_positions(), {"AAA": 10.0})

    def test_equity_marks_positions_to_market(self):
        b = self._broker(cash=1000.0)
        b.execute([Order("AAA", "buy", 10)], {"AAA": 10.0})
        # cash 900 + 10 shares * 12 = 1020
        self.assertEqual(b.get_equity({"AAA": 12.0}), 1020.0)

    def test_sell_closes_and_removes_flat_position(self):
        b = self._broker(cash=1000.0)
        b.execute([Order("AAA", "buy", 10)], {"AAA": 10.0})
        b.execute([Order("AAA", "sell", 10)], {"AAA": 11.0})
        self.assertEqual(b.get_cash(), 1010.0)         # -100 then +110
        self.assertNotIn("AAA", b.get_positions())     # flat -> pruned

    def test_transaction_cost_charged_on_each_fill(self):
        b = self._broker(cash=1000.0, cost=0.01)       # 1%
        b.execute([Order("BBB", "buy", 10)], {"BBB": 10.0})
        # notional 100, cost 1 -> cash 899
        self.assertAlmostEqual(b.get_cash(), 899.0)

    def test_round_trip_is_flat_when_costless(self):
        b = self._broker(cash=1000.0, cost=0.0)
        b.execute([Order("AAA", "buy", 3)], {"AAA": 20.0})
        b.execute([Order("AAA", "sell", 3)], {"AAA": 20.0})
        self.assertAlmostEqual(b.get_cash(), 1000.0)
        self.assertEqual(b.get_positions(), {})

    def test_missing_or_zero_price_is_skipped_safely(self):
        b = self._broker(cash=1000.0)
        b.execute([Order("CCC", "buy", 5)], {})            # no price at all
        b.execute([Order("DDD", "buy", 5)], {"DDD": 0.0})  # non-positive price
        self.assertEqual(b.get_cash(), 1000.0)             # untouched
        self.assertEqual(b.get_positions(), {})

    def test_state_persists_across_instances(self):
        b1 = self._broker(cash=1000.0)
        b1.execute([Order("AAA", "buy", 4)], {"AAA": 25.0})
        # A fresh broker on the same file must see the same account.
        b2 = SimBroker(state_path=self.path, initial_cash=999999.0,
                       transaction_cost=0.0)
        self.assertEqual(b2.get_cash(), 900.0)
        self.assertEqual(b2.get_positions(), {"AAA": 4.0})

    def test_liquidate_all_closes_longs_and_returns_cash(self):
        b = self._broker(cash=1000.0)
        b.execute([Order("AAA", "buy", 10)], {"AAA": 10.0})   # cash 900
        b.liquidate_all({"AAA": 10.0})                        # +100
        self.assertEqual(b.get_positions(), {})
        self.assertAlmostEqual(b.get_cash(), 1000.0)


# =========================================================================== #
#  risk._apply_position_cap — per-name cap with redistribution
# =========================================================================== #

class TestApplyPositionCap(unittest.TestCase):
    def test_no_change_when_within_cap(self):
        w = {"A": 0.4, "B": 0.3, "C": 0.3}
        out = risk._apply_position_cap(dict(w), max_pos=0.5, max_gross=1.0)
        self.assertEqual(out, w)

    def test_excess_is_redistributed_preserving_gross(self):
        out = risk._apply_position_cap({"A": 0.7, "B": 0.2, "C": 0.1},
                                       max_pos=0.5, max_gross=1.0)
        self.assertAlmostEqual(out["A"], 0.5)                 # capped
        self.assertAlmostEqual(sum(out.values()), 1.0)        # gross preserved
        # spill (0.2) redistributed 2:1 by base weight (B:0.2, C:0.1)
        self.assertAlmostEqual(out["B"], 0.3333, places=3)
        self.assertAlmostEqual(out["C"], 0.1667, places=3)

    def test_no_name_ever_exceeds_cap(self):
        out = risk._apply_position_cap({"A": 0.9, "B": 0.05, "C": 0.05},
                                       max_pos=0.4, max_gross=1.0)
        for weight in out.values():
            self.assertLessEqual(weight, 0.4 + 1e-9)

    def test_all_capped_lets_gross_fall_below_target(self):
        # Both names want more than the cap allows -> caps bind, gross < 1.0.
        out = risk._apply_position_cap({"A": 0.6, "B": 0.6},
                                       max_pos=0.3, max_gross=1.0)
        self.assertEqual(out, {"A": 0.3, "B": 0.3})
        self.assertAlmostEqual(sum(out.values()), 0.6)


# =========================================================================== #
#  risk.target_weights — full sizing pipeline
# =========================================================================== #

class TestTargetWeights(unittest.TestCase):
    def _cfg(self, entry=0.1, max_pos=0.6, max_gross=1.0):
        return {"risk": {"entry_threshold": entry,
                         "max_position_weight": max_pos,
                         "max_gross_exposure": max_gross}}

    def test_below_threshold_goes_to_cash(self):
        scores = {"X": _score("X", composite=0.05, volatility=0.02)}
        self.assertEqual(risk.target_weights(scores, self._cfg(entry=0.1)), {})

    def test_lower_vol_gets_more_weight_for_equal_conviction(self):
        scores = {
            "LOWVOL": _score("LOWVOL", composite=0.5, volatility=0.01),
            "HIGHVOL": _score("HIGHVOL", composite=0.5, volatility=0.05),
        }
        w = risk.target_weights(scores, self._cfg())
        self.assertGreater(w["LOWVOL"], w["HIGHVOL"])   # inverse-vol sizing

    def test_respects_per_name_cap_and_gross(self):
        scores = {
            "LOWVOL": _score("LOWVOL", composite=0.5, volatility=0.01),
            "HIGHVOL": _score("HIGHVOL", composite=0.5, volatility=0.05),
        }
        w = risk.target_weights(scores, self._cfg(max_pos=0.6, max_gross=1.0))
        self.assertLessEqual(max(w.values()), 0.6 + 1e-9)
        self.assertLessEqual(sum(w.values()), 1.0 + 1e-9)

    def test_sub_threshold_names_are_dropped(self):
        scores = {
            "KEEP": _score("KEEP", composite=0.5, volatility=0.02),
            "DROP": _score("DROP", composite=0.05, volatility=0.02),
        }
        w = risk.target_weights(scores, self._cfg(entry=0.1))
        self.assertIn("KEEP", w)
        self.assertNotIn("DROP", w)


# =========================================================================== #
#  risk.KillSwitch — drawdown halt + latch
# =========================================================================== #

class TestKillSwitch(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "killswitch.json"

    def _ks(self, threshold=0.1):
        return risk.KillSwitch(state_path=self.path, threshold=threshold)

    def test_first_check_sets_hwm_and_does_not_trip(self):
        res = self._ks().check(1000.0)
        self.assertFalse(res.tripped)
        self.assertEqual(res.hwm, 1000.0)
        self.assertAlmostEqual(res.drawdown, 0.0)

    def test_high_water_mark_ratchets_up(self):
        ks = self._ks()
        ks.check(1000.0)
        res = ks.check(1200.0)
        self.assertEqual(res.hwm, 1200.0)
        self.assertFalse(res.tripped)

    def test_trips_when_drawdown_reaches_threshold(self):
        ks = self._ks(threshold=0.1)
        ks.check(1000.0)
        ks.check(1100.0)                 # peak 1100
        res = ks.check(990.0)            # (1100-990)/1100 = 0.10
        self.assertTrue(res.tripped)
        self.assertAlmostEqual(res.drawdown, 0.1)

    def test_latches_after_tripping(self):
        ks = self._ks(threshold=0.1)
        ks.check(1000.0)
        ks.check(800.0)                  # trips (20% down)
        # Even after equity fully recovers, the halt stays latched.
        res = ks.check(1000.0)
        self.assertTrue(res.tripped)

    def test_reset_clears_the_latch(self):
        ks = self._ks(threshold=0.1)
        ks.check(1000.0)
        ks.check(800.0)                  # trips
        ks.reset()
        res = ks.check(1000.0)
        self.assertFalse(res.tripped)


if __name__ == "__main__":
    unittest.main()
