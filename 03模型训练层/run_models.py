#!/usr/bin/env python
"""构建共享模型数据，并运行Ridge或CNN-GRU季度Walk-forward。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAYER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAYER_DIR))

from models.cnn_gru.trainer import CnnGruTrainer
from models.ridge.trainer import RidgeTrainer
from pipeline.dataset import ModelDataset, load_yaml, resolve_layer_path
from pipeline.quarterly_runner import QuarterlyRunner
from pipeline.walk_forward import ExpandingQuarterlySplitter


def build_shared(config: dict, rebuild: bool = False):
    dataset = ModelDataset(config)
    frame = dataset.load_or_build(rebuild=rebuild)
    splitter = ExpandingQuarterlySplitter(dataset.model_frame(), dataset.target_name, config["walk_forward"])
    splitter.write(dataset.processed_dir)
    return dataset, splitter


def write_model_comparison(experiments_root: Path) -> None:
    rows = []
    for model_name in ["ridge", "cnn_gru"]:
        report_path = experiments_root / model_name / "evaluation_report.json"
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "model_name": model_name,
                "prediction_start": report["prediction_start"],
                "prediction_end": report["prediction_end"],
                "labelled_prediction_end": report["labelled_prediction_end"],
                **report["overall"],
            }
        )
    if rows:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(experiments_root / "model_comparison.parquet", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="市场收益预测模型季度训练")
    parser.add_argument("--config", default=str(LAYER_DIR / "config.yaml"))
    parser.add_argument("--model", choices=["ridge", "cnn_gru", "all"], default="ridge")
    parser.add_argument("--prepare-only", action="store_true", help="只构建标签和fold，不训练")
    parser.add_argument("--rebuild-data", action="store_true", help="重新构建共享标签和fold")
    parser.add_argument("--start-quarter")
    parser.add_argument("--end-quarter")
    parser.add_argument("--max-folds", type=int, help="仅运行筛选后的前N个fold，用于基准检查")
    args = parser.parse_args()

    common = load_yaml(Path(args.config))
    dataset, splitter = build_shared(common, rebuild=args.rebuild_data)
    print(json.dumps({**dataset.manifest(), "fold_count": len(splitter.folds)}, ensure_ascii=False, indent=2))
    if args.prepare_only:
        return 0

    experiments_root = resolve_layer_path(common["output"]["experiments_dir"])
    requested = ["ridge", "cnn_gru"] if args.model == "all" else [args.model]
    for model_name in requested:
        model_config = load_yaml(LAYER_DIR / "models" / model_name / "config.yaml")
        trainer_class = RidgeTrainer if model_name == "ridge" else CnnGruTrainer
        trainer = trainer_class(dataset, model_config, common, experiments_root / model_name)
        runner = QuarterlyRunner(dataset, splitter, trainer, common)
        runner.run(args.start_quarter, args.end_quarter, args.max_folds)
        write_model_comparison(experiments_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
