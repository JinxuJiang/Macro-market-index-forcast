#!/usr/bin/env python
"""构建市场状态V1正式日频特征。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAYER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAYER_DIR))

from feature_engine import FeatureEngine, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="市场状态模型V1特征构建")
    parser.add_argument("--config", default=str(LAYER_DIR / "config" / "features.yaml"))
    args = parser.parse_args()
    engine = FeatureEngine(load_config(args.config))
    _, _, manifest, report = engine.build(write=True)
    print(json.dumps({"manifest": manifest, "validation": report["summary"]}, ensure_ascii=False, indent=2))
    return 1 if report["summary"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
