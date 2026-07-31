"""
Tests for performance.py — the trade-journal analytics/report.

All pure and offline: the metrics operate on hand-built trade dicts, so nothing
here reads the real journal file, hits the network, or needs a broker. Expected
values are hand-computed from a 4-trade fixture (+2%, -1%, +3%, -2%).
"""
import os
import sys
import unittest

LIVE_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LIVE_AGENT_DIR not in sys.path:
    sys.path.insert(0, LIVE_AGENT_DIR)

import performance as perf  # noqa: E402


FIXTURE = [
    {"status": "closed", "pnl_pct": 2.0, "r_multiple": 1.5, "closed_ts": 1},
    {"status": "closed", "pnl_pct": -1.0, "r_multiple": -1.0, "closed_ts": 2},
    {"status": "closed", "pnl_pct": 3.0, "r_multiple": 2.0, "closed_ts": 3},
    {"status": "closed", "pnl_pct": -2.0, "r_multiple": -1.0, "closed_ts": 4},
    {"status": "open"},  # ignored by everything except the open-trade count
]


class TestSelection(unittest.TestCase):
    def test_only_closed_trades_with_pnl_count(self):
        self.assertEqual(len(perf.closed_trades(FIXTURE)), 4)

    def test_closed_trades_sorted_by_close_time(self):
        shuffled = [FIXTURE[2], FIXTURE[0], FIXTURE[3], FIXTURE[1]]
        order = [t["closed_ts"] for t in perf.closed_trades(shuffled)]
        self.assertEqual(order, [1, 2, 3, 4])


class TestEquityCurve(unittest.TestCase):
    def test_compounds_from_starting_capital(self):
        curve = perf.equity_curve(perf.closed_trades(FIXTURE), starting=10_000.0)
        self.assertEqual(len(curve), 5)                 # start + 4 trades
        self.assertEqual(curve[0], 10_000.0)
        self.assertAlmostEqual(curve[-1], 10_192.9212, places=3)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.s = perf.summarize(FIXTURE, starting_capital=10_000.0)

    def test_counts(self):
        self.assertEqual((self.s["trades"], self.s["wins"], self.s["losses"]), (4, 2, 2))
        self.assertEqual(self.s["open_trades"], 1)

    def test_win_rate_and_averages(self):
        self.assertEqual(self.s["win_rate_pct"], 50.0)
        self.assertEqual(self.s["avg_win_pct"], 2.5)
        self.assertEqual(self.s["avg_loss_pct"], -1.5)
        self.assertEqual(self.s["expectancy_pct"], 0.5)

    def test_profit_factor(self):
        # gross win 5 / gross loss 3
        self.assertAlmostEqual(self.s["profit_factor"], 1.667, places=3)

    def test_sharpe_max_dd_and_return(self):
        self.assertAlmostEqual(self.s["sharpe_per_trade"], 0.21, places=2)
        self.assertAlmostEqual(self.s["max_drawdown_pct"], 2.0, places=2)
        self.assertAlmostEqual(self.s["total_return_pct"], 1.93, places=2)

    def test_avg_r_multiple_and_calmar(self):
        self.assertAlmostEqual(self.s["avg_r_multiple"], 0.375, places=3)
        self.assertAlmostEqual(self.s["calmar"], 0.965, places=3)


class TestEdgeCases(unittest.TestCase):
    def test_empty_journal_is_all_zeros(self):
        s = perf.summarize([])
        self.assertEqual(s["trades"], 0)
        self.assertEqual(s["win_rate_pct"], 0.0)
        self.assertEqual(s["total_return_pct"], 0.0)
        self.assertIsNone(s["calmar"])                  # no drawdown -> undefined

    def test_all_wins_profit_factor_is_undefined(self):
        s = perf.summarize([
            {"status": "closed", "pnl_pct": 1.0, "closed_ts": 1},
            {"status": "closed", "pnl_pct": 2.0, "closed_ts": 2},
        ])
        self.assertIsNone(s["profit_factor"])           # no losses -> None
        self.assertEqual(s["max_drawdown_pct"], 0.0)

    def test_single_trade_sharpe_is_zero(self):
        s = perf.summarize([{"status": "closed", "pnl_pct": 5.0, "closed_ts": 1}])
        self.assertEqual(s["sharpe_per_trade"], 0.0)    # std undefined for n<2

    def test_report_renders_without_error(self):
        text = perf.format_report(perf.summarize(FIXTURE))
        self.assertIn("Performance Report", text)
        self.assertIn("Win rate", text)


if __name__ == "__main__":
    unittest.main()
