#!/usr/bin/env python
"""独立运行数据层质量校验。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))

from tushare_engine import MarketDataEngine, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="市场状态模型数据层验收")
    parser.add_argument(
        "--config", default=str(DATA_DIR / "config" / "data_sources.yaml")
    )
    parser.add_argument("--end-date", default="", help="验收截止日 YYYYMMDD")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = MarketDataEngine(config, pro_client=object())
    report = engine.validate(end_date=args.end_date or None)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["summary"]["fail"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
