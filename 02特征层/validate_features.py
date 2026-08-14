#!/usr/bin/env python
"""只读复查磁盘上的正式特征结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAYER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAYER_DIR))

from feature_engine import FeatureEngine, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="市场状态模型V1特征验收")
    parser.add_argument("--config", default=str(LAYER_DIR / "config" / "features.yaml"))
    args = parser.parse_args()
    report = FeatureEngine(load_config(args.config)).validate_saved()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["summary"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

