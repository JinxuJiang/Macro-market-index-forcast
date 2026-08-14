from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = next(path for path in ROOT.iterdir() if path.is_dir() and path.name.startswith("01"))
sys.path.insert(0, str(DATA_DIR))

from tushare_engine import MarketDataEngine, load_config


@unittest.skipUnless(
    os.getenv("RUN_TUSHARE_INTEGRATION") == "1",
    "设置 RUN_TUSHARE_INTEGRATION=1 后运行小样本真实接口测试",
)
class RealTushareSampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_config(DATA_DIR / "config" / "data_sources.yaml")
        cls.engine = MarketDataEngine(config)

    def test_new_daily_sources_return_expected_schema(self) -> None:
        samples = [
            ("index_global", {"ts_code": "SPX", "start_date": "20260701", "end_date": "20260707"}, {"ts_code", "trade_date", "open", "close"}),
            ("index_global", {"ts_code": "IXIC", "start_date": "20260701", "end_date": "20260707"}, {"ts_code", "trade_date", "open", "close"}),
            ("fx_daily", {"ts_code": "USDCNH.FXCM", "start_date": "20260701", "end_date": "20260707"}, {"ts_code", "trade_date", "bid_open", "bid_close"}),
            ("sge_daily", {"ts_code": "Au99.99", "start_date": "20260701", "end_date": "20260707"}, {"ts_code", "trade_date", "open", "close"}),
        ]
        for api_name, kwargs, required in samples:
            with self.subTest(api=api_name, code=kwargs["ts_code"]):
                frame = self.engine._call(api_name, **kwargs)
                self.assertFalse(frame.empty)
                self.assertTrue(required.issubset(frame.columns))

    def test_monthly_sources_return_selected_fields(self) -> None:
        pmi = self.engine._call("cn_pmi")
        cpi = self.engine._call("cn_cpi")
        self.assertFalse(pmi.empty)
        self.assertFalse(cpi.empty)
        self.assertTrue({"MONTH", "PMI010000"}.issubset(pmi.columns))
        self.assertTrue({"month", "nt_yoy"}.issubset(cpi.columns))


if __name__ == "__main__":
    unittest.main()
