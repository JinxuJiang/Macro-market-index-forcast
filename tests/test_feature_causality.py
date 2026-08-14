from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = next(path for path in ROOT.iterdir() if path.is_dir() and path.name.startswith("02"))
sys.path.insert(0, str(FEATURE_DIR))

from feature_engine import FeatureEngine, load_config


class FeatureCausalityTest(unittest.TestCase):
    def test_truncated_rebuild_matches_full_history(self):
        config = load_config(FEATURE_DIR / "config" / "features.yaml")
        engine = FeatureEngine(config)
        cutoff = "20260731"
        full, _, _, _ = engine.build(write=False)
        target = pd.read_parquet(
            engine.project_root / config["sources"]["target"],
            columns=["trade_date"],
        )
        self.assertEqual(
            str(full["trade_date"].max()), str(target["trade_date"].astype(str).max())
        )
        truncated, _, _, _ = engine.build(end_date=cutoff, write=False)
        expected = full[full["trade_date"].astype(str) <= cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, truncated.reset_index(drop=True))

    def test_future_end_date_is_rejected(self):
        config = load_config(FEATURE_DIR / "config" / "features.yaml")
        engine = FeatureEngine(config)
        target = pd.read_parquet(
            engine.project_root / config["sources"]["target"],
            columns=["trade_date"],
        )
        target_end = pd.to_datetime(
            target["trade_date"].astype(str).max(), format="%Y%m%d"
        )
        future_end = (target_end + pd.Timedelta(days=365)).strftime("%Y%m%d")
        with self.assertRaisesRegex(ValueError, "晚于目标指数最后真实行情日"):
            engine.build(end_date=future_end, write=False)


if __name__ == "__main__":
    unittest.main()
