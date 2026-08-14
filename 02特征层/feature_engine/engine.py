"""特征构建、验收、manifest与原子发布总引擎。"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .common import FeatureDataLoader
from .layer1_target import build_layer1
from .layer2_macro_cross_asset import build_layer2
from .layer3_market_structure import build_layer3
from .pit_aligner import PITAligner


def load_config(path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parents[2])
    return config


class FeatureEngine:
    def __init__(self, config: dict):
        self.config = config
        self.project_root = Path(config["_project_root"])
        self.output_root = self.project_root / config["project"]["output_root"]
        self.processed_dir = self.output_root / "processed"
        self.log_dir = self.output_root / "logs"

    @property
    def feature_file(self) -> Path:
        return self.processed_dir / "feature_table.parquet"

    @property
    def lineage_file(self) -> Path:
        return self.processed_dir / "feature_lineage.parquet"

    @property
    def manifest_file(self) -> Path:
        return self.log_dir / "feature_manifest.json"

    @property
    def validation_file(self) -> Path:
        return self.log_dir / "feature_validation_report.json"

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)

    @staticmethod
    def _atomic_json(payload: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary.replace(path)

    def build(self, end_date: Optional[str] = None, write: bool = True):
        # 交易日历可能为了PIT月末判断而预取到未来。未显式指定截止日时，
        # 以目标指数的最后一个真实行情日为准，不能以交易日历末日为准。
        target_path = self.project_root / self.config["sources"]["target"]
        target_dates = pd.read_parquet(target_path, columns=["trade_date"])[
            "trade_date"
        ].astype(str)
        if target_dates.empty:
            raise RuntimeError("目标指数没有可用行情日期")
        target_end_date = str(target_dates.max())
        if end_date and str(end_date) > target_end_date:
            raise ValueError(
                "特征截止日{}晚于目标指数最后真实行情日{}".format(
                    end_date, target_end_date
                )
            )
        effective_end_date = str(end_date) if end_date else target_end_date

        loader = FeatureDataLoader(
            self.project_root, self.config, end_date=effective_end_date
        )
        trade_dates = loader.trade_dates()
        if trade_dates.empty:
            raise RuntimeError("没有可用中国交易日")
        aligner = PITAligner(loader.availability(), trade_dates)
        annualization_days = int(self.config["project"].get("annualization_days", 252))

        print("[1/5] 构建第一层：中证1000自身状态", flush=True)
        layer1 = build_layer1(loader, aligner, annualization_days)
        print("[2/5] 构建第二层：资金、宏观与跨资产", flush=True)
        layer2 = build_layer2(loader, aligner, annualization_days)
        print("[3/5] 构建第三层：市场结构与海外风险", flush=True)
        layer3 = build_layer3(loader, aligner, annualization_days, layer1)

        specs = {**layer1.specs, **layer2.specs, **layer3.specs}
        expected_columns = list(specs)
        features = pd.concat(
            [layer1.features, layer2.features, layer3.features], axis=1
        ).reindex(trade_dates)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.reindex(columns=expected_columns)
        features.index.name = "trade_date"
        feature_output = features.reset_index()

        lineage = pd.concat(
            [layer1.lineage, layer2.lineage, layer3.lineage], ignore_index=True
        )
        lineage = lineage.sort_values(["trade_date", "dataset"]).reset_index(drop=True)

        print("[4/5] 执行特征与PIT验收", flush=True)
        report = self.validate(feature_output, lineage, specs)
        manifest = self._manifest(feature_output, lineage, specs, end_date, report)
        if report["summary"]["fail"]:
            if write:
                self._atomic_json(report, self.validation_file)
            raise RuntimeError("特征验收失败，不覆盖正式输出")

        if write:
            print("[5/5] 原子发布正式特征表", flush=True)
            self._atomic_parquet(feature_output, self.feature_file)
            self._atomic_parquet(lineage, self.lineage_file)
            self._atomic_json(manifest, self.manifest_file)
            self._atomic_json(report, self.validation_file)
        return feature_output, lineage, manifest, report

    def validate_saved(self) -> dict:
        if not self.feature_file.exists() or not self.lineage_file.exists():
            raise FileNotFoundError("正式特征表或lineage不存在")
        features = pd.read_parquet(self.feature_file)
        lineage = pd.read_parquet(self.lineage_file)
        if not self.manifest_file.exists():
            raise FileNotFoundError(self.manifest_file)
        manifest = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        specs = {item["name"]: {k: v for k, v in item.items() if k != "name"} for item in manifest["features"]}
        return self.validate(features, lineage, specs)

    def validate(self, features: pd.DataFrame, lineage: pd.DataFrame, specs: dict) -> dict:
        checks = []

        def add(item, status, detail):
            checks.append({"item": item, "status": status, "detail": detail})

        expected = list(specs)
        actual = [column for column in features.columns if column != "trade_date"]
        add(
            "feature_contract",
            "PASS" if actual == expected else "FAIL",
            "expected={} actual={} missing={} extra={}".format(
                len(expected), len(actual), sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
            ),
        )
        dates = features["trade_date"].astype(str)
        duplicates = int(dates.duplicated().sum())
        add("trade_dates", "PASS" if not duplicates and dates.is_monotonic_increasing else "FAIL", "rows={} duplicates={} sorted={}".format(len(dates), duplicates, dates.is_monotonic_increasing))

        numeric = features[expected]
        infinite = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
        add("finite_values", "PASS" if infinite == 0 else "FAIL", "infinite={}".format(infinite))
        forbidden = [name for name in actual if name.startswith("label_") or name.startswith("future_")]
        add("no_labels", "PASS" if not forbidden else "FAIL", "forbidden={}".format(forbidden))

        all_empty = numeric.columns[numeric.isna().all()].tolist()
        add("not_all_empty", "PASS" if not all_empty else "FAIL", "all_empty={}".format(all_empty))
        latest_missing = numeric.columns[numeric.iloc[-1].isna()].tolist() if len(numeric) else expected
        add("latest_row_complete", "PASS" if not latest_missing else "FAIL", "latest={} missing={}".format(dates.iloc[-1] if len(dates) else "", latest_missing))

        bad_drawdown = int((numeric[["drawdown_20d", "drawdown_250d"]] > 1e-12).sum().sum())
        add("drawdown_bounds", "PASS" if bad_drawdown == 0 else "FAIL", "positive_values={}".format(bad_drawdown))
        volatility_columns = [name for name in expected if name.startswith("vol_") or "_vol_" in name]
        negative_volatility = int((numeric[volatility_columns] < 0).sum().sum())
        add("volatility_bounds", "PASS" if negative_volatility == 0 else "FAIL", "negative_values={}".format(negative_volatility))

        lineage_duplicates = int(lineage.duplicated(["trade_date", "dataset"]).sum())
        add("lineage_unique", "PASS" if lineage_duplicates == 0 else "FAIL", "duplicates={}".format(lineage_duplicates))
        valid = lineage["source_available_date"].notna()
        early = int((lineage.loc[valid, "source_available_date"].astype(str) > lineage.loc[valid, "trade_date"].astype(str)).sum())
        add("pit_not_early", "PASS" if early == 0 else "FAIL", "violations={}".format(early))

        latest_lineage = lineage[lineage["trade_date"].astype(str) == dates.iloc[-1]].copy()
        thresholds = self.config.get("staleness_trade_days", {})
        stale = []
        for _, row in latest_lineage.iterrows():
            threshold = thresholds.get(str(row["dataset"]))
            age = row["age_trade_days"]
            if threshold is not None and pd.notna(age) and int(age) > int(threshold):
                stale.append("{}:{}>{}".format(row["dataset"], int(age), threshold))
        add("latest_staleness", "PASS" if not stale else "WARN", "stale={}".format(stale))

        complete_mask = numeric.notna().all(axis=1)
        complete_rows = int(complete_mask.sum())
        first_complete = dates[complete_mask].iloc[0] if complete_rows else ""
        add("complete_case_coverage", "PASS" if complete_rows else "FAIL", "complete_rows={} first_complete={}".format(complete_rows, first_complete))

        missing_ratio = numeric.isna().mean().sort_values(ascending=False)
        summary = {
            "pass": sum(item["status"] == "PASS" for item in checks),
            "warn": sum(item["status"] == "WARN" for item in checks),
            "fail": sum(item["status"] == "FAIL" for item in checks),
        }
        return {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "summary": summary,
            "rows": len(features),
            "feature_count": len(expected),
            "date_start": dates.iloc[0] if len(dates) else "",
            "date_end": dates.iloc[-1] if len(dates) else "",
            "complete_rows": complete_rows,
            "first_complete_date": first_complete,
            "missing_ratio": {key: float(value) for key, value in missing_ratio.items()},
            "checks": checks,
        }

    def _manifest(self, features, lineage, specs, end_date, report):
        return {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "requested_end_date": end_date or "",
            "actual_end_date": str(features["trade_date"].iloc[-1]),
            "rows": len(features),
            "feature_count": len(specs),
            "lineage_rows": len(lineage),
            "annualization_days": int(self.config["project"].get("annualization_days", 252)),
            "features": [
                {"name": name, "layer": 1 if name in list(specs)[:13] else (2 if name in list(specs)[13:26] else 3), **spec}
                for name, spec in specs.items()
            ],
            "first_complete_date": report["first_complete_date"],
            "outputs": {
                "feature_table": str(self.feature_file),
                "feature_lineage": str(self.lineage_file),
                "validation_report": str(self.validation_file),
            },
        }
