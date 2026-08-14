#!/usr/bin/env python
"""只读验收第四层交易时点、仓位、净值和绩效。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import (
    LAYER_DIR,
    build_index_curve,
    build_decision_dates,
    build_rebalance_plan,
    build_signal_dates,
    calculate_metrics,
    common_prediction_dates,
    load_predictions,
    load_price,
    load_yaml,
    next_trade_date,
    resolve_layer_path,
    slice_backtest_period,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="第四层回测验收")
    parser.add_argument("--config", default=str(LAYER_DIR / "config.yaml"))
    args = parser.parse_args()
    config = load_yaml(Path(args.config))
    price = load_price(resolve_layer_path(config["data"]["target_price"]))
    predictions = {
        model: load_predictions(resolve_layer_path(path), model)
        for model, path in config["data"]["predictions"].items()
    }
    shared_prediction_dates = common_prediction_dates(predictions)
    price = slice_backtest_period(
        price,
        config["backtest"],
        default_end_date=shared_prediction_dates.max(),
    )
    expected_signals = build_signal_dates(
        shared_prediction_dates,
        price.index,
        str(config["backtest"]["rebalance_frequency"]),
    )
    expected_decisions = build_decision_dates(
        shared_prediction_dates,
        price.index,
        str(config["backtest"]["rebalance_frequency"]),
    )
    data_root = resolve_layer_path(config["output"]["data_dir"]) / "processed"
    report_root = resolve_layer_path(config["output"]["reports_dir"])
    checks = []

    def check(item, passed, detail):
        checks.append({"item": item, "status": "PASS" if passed else "FAIL", "detail": detail})

    curves = {}
    for model in predictions:
        model_dir = data_root / model
        paths = {
            "equity": model_dir / "equity_curve.parquet",
            "signals": model_dir / "signals.parquet",
            "orders": model_dir / "orders.parquet",
            "trades": model_dir / "trades.parquet",
            "metrics": report_root / model / "performance.json",
            "rebalance_signals": report_root / model / "rebalance_signals.csv",
        }
        check(f"{model}_outputs", all(path.exists() for path in paths.values()), str(model_dir))
        if not all(path.exists() for path in paths.values()):
            continue
        equity = pd.read_parquet(paths["equity"])
        signals = pd.read_parquet(paths["signals"])
        orders = pd.read_parquet(paths["orders"])
        trades = pd.read_parquet(paths["trades"])
        stored_metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        plan = pd.read_csv(paths["rebalance_signals"], encoding="utf-8-sig", parse_dates=["signal_date", "planned_execution_date"])
        for column in ["date"]:
            equity[column] = pd.to_datetime(equity[column])
        signals["signal_date"] = pd.to_datetime(signals["signal_date"])
        if len(orders):
            orders["created_date"] = pd.to_datetime(orders["created_date"])
            orders["executed_date"] = pd.to_datetime(orders["executed_date"])

        check(
            f"{model}_signal_schedule",
            signals["signal_date"].tolist() == expected_signals,
            f"actual={len(signals)}, expected={len(expected_signals)}",
        )
        expected_values = predictions[model].set_index("signal_date").loc[expected_signals, "prediction"].to_numpy()
        check(
            f"{model}_signal_values",
            np.allclose(signals["prediction"].to_numpy(dtype=float), expected_values),
            "signals equal formal OOS predictions",
        )
        expected_plan = build_rebalance_plan(
            model,
            predictions[model],
            expected_decisions,
            price.index,
            config["backtest"],
        )
        plan_columns = [
            "signal_date", "model_name", "instrument", "ts_code", "prediction_raw",
            "prediction_smoothed", "prediction_history_mean", "prediction_history_std",
            "prediction_zscore", "signal_threshold", "state_z_threshold", "previous_target_state",
            "desired_state", "action", "trade_required", "previous_target_exposure",
            "target_exposure", "exposure_change",
            "execution_rule", "planned_execution_date",
        ]
        try:
            pd.testing.assert_frame_equal(
                plan[plan_columns],
                expected_plan[plan_columns],
                check_exact=False,
                rtol=1e-14,
                atol=1e-14,
            )
            plan_exact = True
        except AssertionError:
            plan_exact = False
        check(f"{model}_rebalance_plan_exact", plan_exact, f"rows={len(plan)}, independent_of_backtrader")
        valid_states = set(plan["desired_state"]).issubset({"bear", "neutral", "bull"})
        valid_targets = set(np.round(plan["target_exposure"].astype(float), 12)).issubset(
            {0.0, float(config["backtest"]["neutral_exposure"]), float(config["backtest"]["target_exposure"])}
        )
        check(
            f"{model}_three_state_targets",
            valid_states and valid_targets,
            plan["desired_state"].value_counts().to_dict(),
        )
        executable_plan = expected_plan.set_index("signal_date").loc[expected_signals]
        backtrader_plan_exact = (
            signals["desired_state"].tolist() == executable_plan["desired_state"].tolist()
            and np.allclose(
                signals["prediction_zscore"].to_numpy(dtype=float),
                executable_plan["prediction_zscore"].to_numpy(dtype=float),
            )
            and np.allclose(
                signals["target_exposure"].to_numpy(dtype=float),
                executable_plan["target_exposure"].to_numpy(dtype=float),
            )
        )
        check(f"{model}_backtrader_uses_plan", backtrader_plan_exact, "state, z-score and target exposure")
        completed = orders[orders["status"] == "Completed"] if len(orders) else orders
        execution_ok = True
        price_ok = True
        commission_ok = True
        for row in completed.itertuples(index=False):
            execution_ok &= pd.Timestamp(row.executed_date) == next_trade_date(row.created_date, price.index)
            expected_open = float(price.loc[pd.Timestamp(row.executed_date), "open"])
            price_ok &= np.isclose(float(row.executed_price), expected_open, rtol=0.0, atol=1e-10)
            expected_commission = abs(float(row.executed_size) * float(row.executed_price)) * float(
                config["backtest"]["commission"]
            )
            commission_ok &= np.isclose(float(row.commission), expected_commission, rtol=1e-12, atol=1e-8)
        check(f"{model}_t_plus_one_execution", execution_ok, f"completed_orders={len(completed)}")
        check(
            f"{model}_all_orders_completed",
            len(completed) == len(orders),
            orders["status"].value_counts().to_dict() if len(orders) else {},
        )
        check(f"{model}_execution_at_open", price_ok, "executed price equals next trading-day open")
        check(f"{model}_commission_exact", commission_ok, f"rate={config['backtest']['commission']}")
        no_short = bool((equity["position_size"] >= -1e-12).all())
        solvent = bool((equity["value"] > 0).all() and (equity["cash"] >= -1e-6).all())
        check(f"{model}_long_cash_only", no_short, f"min_position={equity['position_size'].min():.6f}")
        check(f"{model}_solvent", solvent, f"min_cash={equity['cash'].min():.2f}")
        recomputed = calculate_metrics(equity, trades, config["backtest"])
        keys = [
            "total_return", "annual_return", "annual_volatility", "sharpe",
            "sortino", "max_drawdown", "calmar", "win_rate",
        ]
        metric_ok = all(
            (pd.isna(recomputed[key]) and pd.isna(stored_metrics[key]))
            or np.isclose(recomputed[key], stored_metrics[key], rtol=1e-12, atol=1e-12)
            for key in keys
        )
        check(f"{model}_metrics_recomputed", metric_ok, "8 metrics")
        nav = equity["value"].to_numpy(dtype=float) / float(config["backtest"]["initial_cash"])
        check(f"{model}_nav_finite", np.isfinite(nav).all(), f"rows={len(nav)}")
        curves[model] = equity

    comparison_path = data_root / "comparison" / "equity_comparison.parquet"
    check("comparison_output", comparison_path.exists(), str(comparison_path))
    html_path = report_root / "comparison" / "backtest_report.html"
    html_ok = html_path.exists()
    if html_ok:
        html_text = html_path.read_text(encoding="utf-8")
        html_ok = all(marker in html_text for marker in ["中证1000多头/现金择时回测报告", "data:image/png;base64,", "绩效指标"])
    check("html_report", html_ok, str(html_path))
    combined_signal_path = report_root / "comparison" / "rebalance_signals.csv"
    combined_ok = combined_signal_path.exists()
    if combined_ok:
        combined_signals = pd.read_csv(combined_signal_path, encoding="utf-8-sig")
        combined_ok = (
            len(combined_signals) == len(expected_decisions) * len(predictions)
            and list(combined_signals.columns) == [
                "信号日期", "模型", "平滑预测", "z值", "市场状态",
                "是否调仓", "操作", "目标仓位", "计划执行日",
            ]
            and set(combined_signals["模型"]) == {"Ridge", "CNN-GRU"}
        )
    check("combined_rebalance_signals", combined_ok, str(combined_signal_path))
    if comparison_path.exists() and len(curves) == len(predictions):
        comparison = pd.read_parquet(comparison_path)
        same_dates = all(curves[model]["date"].tolist() == comparison["date"].tolist() for model in curves)
        check("common_backtest_dates", same_dates, f"rows={len(comparison)}")
        index_curve, index_sharpe = build_index_curve(price, comparison["date"])
        index_ok = np.allclose(comparison["index_nav"], index_curve["index_nav"])
        benchmark = json.loads((report_root / "comparison" / "benchmark.json").read_text(encoding="utf-8"))
        check("index_curve_exact", index_ok, "direct CSI1000 normalized curve")
        check("index_sharpe_exact", np.isclose(index_sharpe, benchmark["sharpe"]), benchmark["sharpe"])

    summary = {
        "pass": sum(row["status"] == "PASS" for row in checks),
        "fail": sum(row["status"] == "FAIL" for row in checks),
    }
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"), "summary": summary, "checks": checks}
    log_dir = resolve_layer_path(config["output"]["data_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "backtest_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    return 1 if summary["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
