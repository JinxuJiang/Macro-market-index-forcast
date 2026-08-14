#!/usr/bin/env python
"""下载或增量更新市场状态模型的原始数据。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))

from tushare_engine import MarketDataEngine, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="市场状态模型 Tushare 数据下载")
    parser.add_argument(
        "--config",
        default=str(DATA_DIR / "config" / "data_sources.yaml"),
        help="数据源配置文件",
    )
    parser.add_argument("--start-date", default="", help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default="", help="结束日期 YYYYMMDD")
    parser.add_argument(
        "--skip-validation", action="store_true", help="下载后不运行质量校验"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    engine = MarketDataEngine(config)
    manifest = engine.download_all(
        start_date=args.start_date or None,
        end_date=args.end_date or None,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.skip_validation:
        return 0
    report = engine.validate(end_date=manifest["requested_end_date"])
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 1 if report["summary"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

