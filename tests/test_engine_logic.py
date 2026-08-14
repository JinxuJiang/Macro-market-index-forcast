from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = next(path for path in ROOT.iterdir() if path.is_dir() and path.name.startswith("01"))

import sys

sys.path.insert(0, str(DATA_DIR))

from tushare_engine import MarketDataEngine, load_config


class EngineLogicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        config = load_config(DATA_DIR / "config" / "data_sources.yaml")
        config["_project_root"] = self.temp.name
        self.engine = MarketDataEngine(config, pro_client=object())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_incremental_merge_keeps_latest_value(self) -> None:
        path = Path(self.temp.name) / "existing.parquet"
        pd.DataFrame(
            {
                "ts_code": ["TEST", "TEST"],
                "trade_date": ["20260803", "20260804"],
                "close": [10.0, 11.0],
            }
        ).to_parquet(path, index=False)
        update = pd.DataFrame(
            {
                "ts_code": ["TEST", "TEST"],
                "trade_date": ["20260804", "20260805"],
                "close": [12.0, 13.0],
            }
        )

        result = self.engine._merge(
            path,
            [update],
            ["ts_code", "trade_date"],
            ["trade_date"],
        )

        self.assertEqual(result["trade_date"].tolist(), ["20260803", "20260804", "20260805"])
        self.assertEqual(result.loc[result["trade_date"] == "20260804", "close"].item(), 12.0)

    def test_daily_availability_rules(self) -> None:
        path = Path(self.temp.name) / "daily.parquet"
        pd.DataFrame({"trade_date": ["20260807"]}).to_parquet(path, index=False)
        china_dates = ["20260803", "20260804", "20260805", "20260806", "20260807", "20260810"]

        margin = self.engine._daily_availability_rows(
            "margin", path, "trade_date", china_dates, lag_trade_days=1
        )
        fx = self.engine._daily_availability_rows(
            "fx", path, "trade_date", china_dates, strictly_after=True
        )

        self.assertEqual(margin[0]["available_date"], "20260810")
        self.assertEqual(fx[0]["available_date"], "20260810")
        self.assertGreater(margin[0]["available_date"], margin[0]["period_date"])

    def test_monthly_next_month_end_availability_rules(self) -> None:
        path = Path(self.temp.name) / "monthly.parquet"
        pd.DataFrame({"month": ["202606", "202607"]}).to_parquet(path, index=False)
        china_dates = [
            "20260701", "20260702", "20260703", "20260706", "20260707",
            "20260708", "20260709", "20260710", "20260713", "20260714",
            "20260715", "20260716", "20260717", "20260720", "20260721",
            "20260722", "20260723", "20260724", "20260727", "20260728",
            "20260729", "20260730", "20260731", "20260803", "20260804",
            "20260805", "20260806", "20260807", "20260810", "20260811",
            "20260812",
        ]

        pmi = self.engine._monthly_availability_rows(
            "pmi", path, china_dates, note=""
        )
        cpi = self.engine._monthly_availability_rows(
            "cpi", path, china_dates, note=""
        )

        self.assertEqual(len(pmi), 1)
        self.assertEqual(pmi[0]["period_date"], "20260630")
        self.assertEqual(pmi[0]["available_date"], "20260731")
        self.assertEqual(pmi[0]["availability_method"], "next_month_last_trade_date")
        self.assertEqual(cpi[0]["available_date"], "20260731")

        complete_calendar = china_dates + [
            "20260813", "20260814", "20260817", "20260818", "20260819",
            "20260820", "20260821", "20260824", "20260825", "20260826",
            "20260827", "20260828", "20260831",
        ]
        completed = self.engine._monthly_availability_rows(
            "pmi", path, complete_calendar, note=""
        )
        self.assertEqual(len(completed), 2)
        self.assertEqual(completed[1]["period_date"], "20260731")
        self.assertEqual(completed[1]["available_date"], "20260831")

    def test_price_validation_detects_invalid_rows(self) -> None:
        data = pd.DataFrame(
            {
                "ts_code": ["TEST", "TEST"],
                "trade_date": ["20260803", "20260803"],
                "open": [10.0, 0.0],
                "high": [11.0, 11.0],
                "low": [9.0, 9.0],
                "close": [10.5, 10.5],
            }
        )
        missing, duplicate, bad_price = self.engine._price_check(
            data, ["ts_code", "trade_date"]
        )
        self.assertEqual(missing, [])
        self.assertEqual(duplicate, 1)
        self.assertEqual(bad_price, 1)


if __name__ == "__main__":
    unittest.main()
