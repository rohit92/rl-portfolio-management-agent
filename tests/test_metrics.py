"""
Tests for utils/metrics.py — the quantitative performance metrics used to
evaluate the RL agent. These are pure functions, so the tests are fast and
deterministic (no network, no model, no data download).

Expected values are taken from the functions' own docstring examples and
verified against the reference implementation.

Run:  ./venv/bin/python -m pytest tests/ -q
  or: ./venv/bin/python -m unittest -q
"""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils import metrics  # noqa: E402


class TestMaxDrawdown(unittest.TestCase):
    def test_docstring_example(self):
        self.assertAlmostEqual(metrics.max_drawdown([100, 120, 90, 110]), 0.25)

    def test_returns_positive_fraction(self):
        self.assertGreaterEqual(metrics.max_drawdown([100, 50, 200]), 0.0)

    def test_too_few_points_is_zero(self):
        self.assertEqual(metrics.max_drawdown([100]), 0.0)
        self.assertEqual(metrics.max_drawdown([]), 0.0)

    def test_monotonic_rise_has_no_drawdown(self):
        self.assertEqual(metrics.max_drawdown([100, 101, 102, 103]), 0.0)


class TestWinRate(unittest.TestCase):
    def test_docstring_example_excludes_flat_days(self):
        self.assertAlmostEqual(metrics.win_rate([0.01, -0.02, 0.005, 0.0, 0.003]), 0.75)

    def test_all_zero_returns_zero(self):
        self.assertEqual(metrics.win_rate([0.0, 0.0]), 0.0)

    def test_bounds(self):
        self.assertTrue(0.0 <= metrics.win_rate([0.1, -0.1, 0.2, -0.3]) <= 1.0)


class TestSharpeRatio(unittest.TestCase):
    def test_zero_std_is_handled(self):
        # Constant returns -> zero volatility -> guarded to 0.0, not NaN/inf.
        val = metrics.sharpe_ratio([0.01, 0.01, 0.01])
        self.assertEqual(val, 0.0)
        self.assertFalse(math.isnan(val))

    def test_positive_series_is_positive(self):
        self.assertGreater(metrics.sharpe_ratio([0.01, -0.005, 0.02, 0.0, 0.011]), 0.0)


class TestAnnualizedReturn(unittest.TestCase):
    def test_one_percent_daily_for_a_year(self):
        # (1.01 ** 252) - 1  ~= 11.27
        self.assertAlmostEqual(metrics.annualized_return([0.01] * 252), 11.274, places=2)

    def test_empty_is_zero(self):
        self.assertEqual(metrics.annualized_return([]), 0.0)

    def test_total_loss_is_floored(self):
        # A -100% period drives wealth to zero; must not raise / go complex.
        self.assertEqual(metrics.annualized_return([-1.0, 0.02]), -1.0)


class TestCalmarRatio(unittest.TestCase):
    def test_no_drawdown_is_zero(self):
        self.assertEqual(metrics.calmar_ratio([0.01, 0.01], [100, 101, 102]), 0.0)

    def test_uses_annret_over_mdd(self):
        eq = [100, 120, 90, 110]
        ret = [0.02, 0.01, -0.01]
        expected = metrics.annualized_return(ret) / abs(metrics.max_drawdown(eq))
        self.assertAlmostEqual(metrics.calmar_ratio(ret, eq), expected)


if __name__ == "__main__":
    unittest.main()
