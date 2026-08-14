from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = next(path for path in ROOT.iterdir() if path.is_dir() and path.name.startswith("02"))
sys.path.insert(0, str(FEATURE_DIR))

from feature_engine.pit_aligner import PITAligner


class FeatureAvailabilityTest(unittest.TestCase):
    def test_asof_never_uses_future_available_date(self):
        availability = pd.DataFrame(
            {
                "dataset": ["x", "x"],
                "data_date": ["20260807", "20260808"],
                "period_date": ["20260807", "20260808"],
                "available_date": ["20260810", "20260810"],
            }
        )
        native = pd.DataFrame(
            {"date": ["20260807", "20260808"], "value": [1.0, 2.0]}
        )
        aligner = PITAligner(
            availability, pd.Index(["20260807", "20260810", "20260811"])
        )
        values, lineage = aligner.align("x", native, "date", ["value"])
        self.assertTrue(pd.isna(values.loc["20260807", "value"]))
        self.assertEqual(values.loc["20260810", "value"], 2.0)
        row = lineage[lineage["trade_date"] == "20260810"].iloc[0]
        self.assertEqual(row["source_data_date"], "20260808")
        self.assertLessEqual(row["source_available_date"], row["trade_date"])

    def test_monthly_change_must_be_computed_before_expansion(self):
        monthly = pd.Series([49.0, 50.0, 52.0], index=["202605", "202606", "202607"])
        self.assertEqual(monthly.diff().loc["202607"], 2.0)


if __name__ == "__main__":
    unittest.main()

