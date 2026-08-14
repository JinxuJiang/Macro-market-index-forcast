from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = next(path for path in ROOT.iterdir() if path.is_dir() and path.name.startswith("02"))
sys.path.insert(0, str(FEATURE_DIR))

from feature_engine.common import (
    annualized_volatility,
    moving_average_distance,
    rolling_drawdown,
    simple_return,
)


class FeatureFormulaTest(unittest.TestCase):
    def test_simple_return(self):
        values = pd.Series([100.0, 101.0, 103.0, 102.0])
        self.assertAlmostEqual(simple_return(values, 2).iloc[2], 0.03)

    def test_moving_average_distance(self):
        values = pd.Series([1.0, 2.0, 3.0])
        self.assertAlmostEqual(moving_average_distance(values, 3).iloc[-1], 0.5)

    def test_rolling_drawdown_is_non_positive(self):
        values = pd.Series([100.0, 120.0, 90.0])
        self.assertAlmostEqual(rolling_drawdown(values, 3).iloc[-1], -0.25)

    def test_volatility_does_not_use_future(self):
        values = pd.Series(np.linspace(100.0, 140.0, 50))
        original = annualized_volatility(values, 20)
        changed = values.copy()
        changed.iloc[-1] = 1000.0
        pd.testing.assert_series_equal(original.iloc[:-1], annualized_volatility(changed, 20).iloc[:-1])


if __name__ == "__main__":
    unittest.main()

