#!/usr/bin/env python
"""只读验收共享标签、季度fold和已有模型预测。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

LAYER_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LAYER_DIR))

from pipeline.dataset import ModelDataset, load_yaml, regression_metrics, resolve_layer_path


def check(statuses, item, passed, detail):
    statuses.append({"item": item, "status": "PASS" if passed else "FAIL", "detail": detail})


def main() -> int:
    parser = argparse.ArgumentParser(description="模型训练层只读验收")
    parser.add_argument("--config", default=str(LAYER_DIR / "config.yaml"))
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    dataset = ModelDataset(config)
    dataset_path = dataset.processed_dir / "dataset_with_label.parquet"
    manifest_path = dataset.processed_dir / "fold_manifest.parquet"
    membership_path = dataset.processed_dir / "fold_membership.parquet"
    statuses = []
    for path in [dataset_path, manifest_path, membership_path]:
        check(statuses, path.name, path.exists(), str(path))
    if any(row["status"] == "FAIL" for row in statuses):
        print(json.dumps({"checks": statuses}, ensure_ascii=False, indent=2))
        return 1

    frame = dataset.load_or_build(rebuild=False)
    manifest = pd.read_parquet(manifest_path)
    membership = pd.read_parquet(membership_path)
    target = dataset.target_name
    check(statuses, "feature_count", len(dataset.feature_names) == 39, f"actual={len(dataset.feature_names)}")
    check(statuses, "signal_dates", frame["signal_date"].is_monotonic_increasing and not frame["signal_date"].duplicated().any(), f"rows={len(frame)}")
    complete = frame[target].notna()
    tail_only = not complete[complete.idxmin() :].any() if (~complete).any() else True
    first_missing = int(np.flatnonzero((~complete).to_numpy())[0]) if (~complete).any() else len(frame)
    tail_only = bool(complete.iloc[:first_missing].all() and (~complete.iloc[first_missing:]).all())
    check(statuses, "label_missing_tail_only", tail_only, f"last_complete={frame.loc[complete, 'signal_date'].max().date()}")

    prices = pd.read_parquet(dataset.target_price_path).copy()
    prices["signal_date"] = pd.to_datetime(prices["trade_date"].astype(str), format="%Y%m%d")
    prices = prices.sort_values("signal_date").reset_index(drop=True)
    recomputed = prices["open"].shift(-21) / prices["open"].shift(-1) - 1.0
    difference = np.nanmax(np.abs(frame[target].to_numpy(dtype=float) - recomputed.to_numpy(dtype=float)))
    check(statuses, "label_formula", difference < 1e-12, f"max_abs_diff={difference:.3e}")
    check(statuses, "first_prediction_quarter", str(manifest.iloc[0]["model_period"]) == "2018Q2", str(manifest.iloc[0]["model_period"]))
    check(statuses, "validation_days", bool((manifest["n_inner_valid"] == 252).all()), f"folds={len(manifest)}")
    check(statuses, "prediction_non_overlap", not membership.loc[membership.sample_role == "prediction", "signal_date"].duplicated().any(), "all prediction dates unique")

    indexed = frame.set_index("signal_date")
    purge_ok = True
    final_ok = True
    for row in manifest.itertuples(index=False):
        train_dates = membership.loc[(membership.fold_id == row.fold_id) & (membership.sample_role == "inner_train"), "signal_date"]
        final_dates = membership.loc[(membership.fold_id == row.fold_id) & (membership.sample_role == "final_train"), "signal_date"]
        purge_ok &= bool((indexed.loc[pd.to_datetime(train_dates), "exit_date"] <= row.inner_valid_start).all())
        final_ok &= bool((indexed.loc[pd.to_datetime(final_dates), "exit_date"] <= row.as_of_date).all())
    check(statuses, "purge_pit", purge_ok, "inner_train.exit_date <= inner_valid_start")
    check(statuses, "final_train_pit", final_ok, "final_train.exit_date <= as_of_date")

    experiments = resolve_layer_path(config["output"]["experiments_dir"])
    delta = float(config["evaluation"]["huber_delta"])
    for model_name in ["ridge", "cnn_gru"]:
        prediction_path = experiments / model_name / "oos_predictions.parquet"
        if not prediction_path.exists():
            statuses.append({"item": f"{model_name}_predictions", "status": "SKIP", "detail": "尚未训练"})
            continue
        prediction = pd.read_parquet(prediction_path)
        prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
        unique = not prediction["signal_date"].duplicated().any()
        check(statuses, f"{model_name}_prediction_unique", unique, f"rows={len(prediction)}")
        metrics = regression_metrics(prediction, delta)
        statuses.append({"item": f"{model_name}_metrics", "status": "PASS", "detail": metrics})
        smoothing = config.get("prediction_smoothing", {})
        if bool(smoothing.get("enabled", False)):
            smooth_path = experiments / model_name / "oos_predictions_smoothed.parquet"
            check(statuses, f"{model_name}_smoothed_output", smooth_path.exists(), str(smooth_path))
            if smooth_path.exists():
                ordered = prediction.sort_values("signal_date").reset_index(drop=True)
                smooth = pd.read_parquet(smooth_path).sort_values("signal_date").reset_index(drop=True)
                smooth["signal_date"] = pd.to_datetime(smooth["signal_date"])
                expected = ordered["prediction"].astype(float).ewm(
                    halflife=float(smoothing["halflife_days"]),
                    adjust=bool(smoothing.get("adjust", False)),
                    min_periods=int(smoothing.get("min_periods", 1)),
                ).mean().to_numpy()
                exact = (
                    smooth["signal_date"].tolist() == ordered["signal_date"].tolist()
                    and np.allclose(smooth["prediction_raw"].to_numpy(dtype=float), ordered["prediction"].to_numpy(dtype=float))
                    and np.allclose(smooth["prediction"].to_numpy(dtype=float), expected)
                )
                check(
                    statuses,
                    f"{model_name}_smoothed_exact",
                    exact,
                    f"ewm_halflife={smoothing['halflife_days']}, causal_adjust={smoothing.get('adjust', False)}",
                )

    ridge_path = experiments / "ridge" / "oos_predictions.parquet"
    cnn_path = experiments / "cnn_gru" / "oos_predictions.parquet"
    if ridge_path.exists() and cnn_path.exists():
        ridge_dates = set(pd.to_datetime(pd.read_parquet(ridge_path)["signal_date"]))
        cnn_dates = set(pd.to_datetime(pd.read_parquet(cnn_path)["signal_date"]))
        common_dates = ridge_dates & cnn_dates
        check(statuses, "model_common_dates", len(common_dates) > 0, f"common={len(common_dates)}")

    summary = {
        "pass": sum(row["status"] == "PASS" for row in statuses),
        "skip": sum(row["status"] == "SKIP" for row in statuses),
        "fail": sum(row["status"] == "FAIL" for row in statuses),
    }
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "summary": summary, "checks": statuses}
    dataset.logs_dir.mkdir(parents=True, exist_ok=True)
    (dataset.logs_dir / "dataset_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    return 1 if summary["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
