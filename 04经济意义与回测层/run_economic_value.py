#!/usr/bin/env python
"""运行Ridge和CNN-GRU的中证1000多头/现金经济意义回测。"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtest import (
    LAYER_DIR,
    build_index_curve,
    build_decision_dates,
    build_rebalance_plan,
    build_signal_dates,
    common_prediction_dates,
    load_predictions,
    load_price,
    load_yaml,
    resolve_layer_path,
    run_model_backtest,
    save_rebalance_plan,
    save_result,
    slice_backtest_period,
)


def write_html_report(results, benchmark, chart_path: Path, output_path: Path, config: dict) -> None:
    """生成可独立打开的回测报告，图片以内嵌方式保存。"""
    image_data = base64.b64encode(chart_path.read_bytes()).decode("ascii")
    model_labels = {"ridge": "Ridge", "cnn_gru": "CNN-GRU"}
    metric_rows = [
        ("累计收益", "total_return", "pct"),
        ("年化收益", "annual_return", "pct"),
        ("年化波动率", "annual_volatility", "pct"),
        ("Sharpe", "sharpe", "num"),
        ("Sortino", "sortino", "num"),
        ("最大回撤", "max_drawdown", "pct"),
        ("Calmar", "calmar", "num"),
        ("胜率", "win_rate", "pct"),
        ("已平仓交易数", "closed_trades", "int"),
    ]

    def format_value(value, kind: str) -> str:
        if pd.isna(value):
            return "—"
        if kind == "pct":
            return f"{float(value) * 100:.2f}%"
        if kind == "int":
            return f"{int(value)}"
        return f"{float(value):.3f}"

    headers = "".join(f"<th>{escape(model_labels.get(r.model_name, r.model_name))}</th>" for r in results)
    rows = []
    for label, key, kind in metric_rows:
        values = "".join(f"<td>{format_value(r.metrics[key], kind)}</td>" for r in results)
        rows.append(f"<tr><th>{label}</th>{values}</tr>")

    cards = []
    for result in results:
        label = escape(model_labels.get(result.model_name, result.model_name))
        cards.append(
            f"<div class='metric-card'><div class='metric-label'>{label} 累计收益</div>"
            f"<div class='metric-value'>{format_value(result.metrics['total_return'], 'pct')}</div>"
            f"<div class='metric-sub'>Sharpe {format_value(result.metrics['sharpe'], 'num')}</div></div>"
        )
    cards.append(
        f"<div class='metric-card benchmark'><div class='metric-label'>中证1000指数</div>"
        f"<div class='metric-value'>{float(benchmark['sharpe']):.3f}</div>"
        "<div class='metric-sub'>同期直接持有 Sharpe</div></div>"
    )

    best = max(results, key=lambda item: float(item.metrics["sharpe"]))
    best_label = escape(model_labels.get(best.model_name, best.model_name))
    comparison = (
        f"本次回测中，{best_label} 的策略 Sharpe（{float(best.metrics['sharpe']):.3f}）高于另一模型；"
        f"同期中证1000直接持有 Sharpe 为 {float(benchmark['sharpe']):.3f}。"
        "这些结果只描述当前样本外预测在既定交易规则下的历史表现。"
    )
    bt = config["backtest"]
    frequency_label = {
        "monthly_first_trading_day": "每月首个交易日收盘后读取预测",
        "weekly_first_trading_day": "每周首个交易日收盘后读取预测",
        "weekly_last_trading_day": "每周最后一个交易日收盘后读取预测",
    }.get(str(bt["rebalance_frequency"]), str(bt["rebalance_frequency"]))
    report_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>中证1000多头/现金择时回测报告</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 38px 20px; background: #f3f6f4; color: #24312a;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; }}
    .container {{ max-width: 1180px; margin: auto; background: #fff; border-radius: 12px;
                  padding: 34px; box-shadow: 0 4px 24px rgba(25, 60, 40, .10); }}
    h1 {{ margin: 0 0 8px; padding-bottom: 14px; border-bottom: 3px solid #2f855a; font-size: 29px; }}
    h2 {{ margin-top: 34px; font-size: 21px; color: #245c40; }}
    .meta {{ color: #6b7b72; font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 24px; }}
    .metric-card {{ padding: 19px; border: 1px solid #dce8e0; border-radius: 9px; background: #fbfdfb; }}
    .metric-card.benchmark {{ background: #f4f5f7; border-color: #d9dde2; }}
    .metric-label {{ color: #607068; font-size: 14px; }}
    .metric-value {{ margin: 7px 0 3px; color: #217a4b; font-size: 30px; font-weight: 700; }}
    .benchmark .metric-value {{ color: #343b42; }}
    .metric-sub {{ color: #79867f; font-size: 13px; }}
    .summary {{ padding: 17px 20px; border-left: 4px solid #2f855a; background: #edf8f1; line-height: 1.8; }}
    table {{ width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 8px; }}
    th, td {{ padding: 12px 15px; border-bottom: 1px solid #e4ebe7; text-align: right; }}
    thead th {{ color: #fff; background: #327457; }}
    th:first-child {{ text-align: left; }}
    tbody tr:nth-child(even) {{ background: #f8faf9; }}
    .settings {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(245px, 1fr)); gap: 10px 24px;
                 padding: 18px 20px; background: #f7faf8; border-radius: 8px; line-height: 1.65; }}
    .settings b {{ color: #2d5d44; }}
    figure {{ margin: 18px 0 0; }}
    img {{ width: 100%; height: auto; border: 1px solid #e2e8e4; border-radius: 8px; }}
    figcaption, .note {{ color: #718078; font-size: 13px; line-height: 1.65; }}
    @media (max-width: 640px) {{ .container {{ padding: 22px 16px; }} body {{ padding: 12px; }}
      th, td {{ padding: 10px 7px; font-size: 13px; }} h1 {{ font-size: 24px; }} }}
  </style>
</head>
<body>
<main class="container">
  <h1>中证1000多头/现金择时回测报告</h1>
  <div class="meta">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ 回测区间：{benchmark['start_date']} 至 {benchmark['end_date']}</div>
  <section class="cards">{''.join(cards)}</section>
  <h2>结果摘要</h2>
  <div class="summary">{comparison}</div>
  <h2>绩效指标</h2>
  <table><thead><tr><th>指标</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>
  <h2>净值曲线</h2>
  <figure><img src="data:image/png;base64,{image_data}" alt="Ridge、CNN-GRU与中证1000指数净值对比图">
    <figcaption>所有曲线以共同回测起点归一化为 1；指数曲线为直接持有中证1000的同期对照。</figcaption></figure>
  <h2>回测设置</h2>
  <div class="settings">
    <div><b>信号：</b>{escape(frequency_label)}</div><div><b>预测版本：</b>{escape(str(config['data'].get('prediction_variant', 'raw')))}</div>
    <div><b>成交：</b>下一交易日开盘价</div>
    <div><b>状态：</b>过去{int(bt['standardization_window'])}日预测标准化，阈值 ±{float(bt['state_z_threshold']):g}</div>
    <div><b>仓位：</b>熊市 0%｜震荡 {float(bt['neutral_exposure']):.0%}｜牛市 {float(bt['target_exposure']):.0%}</div>
    <div><b>初始资金：</b>¥{float(bt['initial_cash']):,.0f}</div>
    <div><b>手续费：</b>单边 {float(bt['commission']):.2%}</div><div><b>现金及无风险利率：</b>0</div>
  </div>
  <p class="note">说明：中证1000指数点位在 Backtrader 中作为合成资产单位价格使用；未模拟 IM 期货乘数、保证金、滑点、现金利息或止损。本报告用于检验模型的经济意义，不代表可直接执行的实盘收益。</p>
</main>
</body>
</html>"""
    output_path.write_text(report_html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="中证1000多头/现金经济意义回测")
    parser.add_argument("--config", default=str(LAYER_DIR / "config.yaml"))
    parser.add_argument("--model", choices=["ridge", "cnn_gru", "all"], default="all")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    price = load_price(resolve_layer_path(config["data"]["target_price"]))
    all_predictions = {
        model: load_predictions(resolve_layer_path(path), model)
        for model, path in config["data"]["predictions"].items()
    }
    common_dates = common_prediction_dates(all_predictions)
    price = slice_backtest_period(
        price,
        config["backtest"],
        default_end_date=common_dates.max(),
    )
    decision_dates = build_decision_dates(
        common_dates,
        price.index,
        str(config["backtest"]["rebalance_frequency"]),
    )
    signal_dates = build_signal_dates(
        common_dates,
        price.index,
        str(config["backtest"]["rebalance_frequency"]),
    )
    requested = list(all_predictions) if args.model == "all" else [args.model]
    data_root = resolve_layer_path(config["output"]["data_dir"])
    report_root = resolve_layer_path(config["output"]["reports_dir"])
    results = []
    rebalance_plans = []
    for model in requested:
        plan = build_rebalance_plan(
            model,
            all_predictions[model],
            decision_dates,
            price.index,
            config["backtest"],
        )
        save_rebalance_plan(plan, report_root, model)
        rebalance_plans.append(plan)
        result = run_model_backtest(
            model,
            price,
            all_predictions[model],
            signal_dates,
            config["backtest"],
            plan,
        )
        save_result(result, data_root, report_root)
        results.append(result)
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2, allow_nan=True))

    comparison_dir = data_root / "processed" / "comparison"
    comparison_report = report_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_report.mkdir(parents=True, exist_ok=True)
    combined_plan = pd.concat(rebalance_plans, ignore_index=True).sort_values(
        ["signal_date", "model_name"]
    ).reset_index(drop=True)
    state_labels = {"bear": "熊市", "neutral": "震荡", "bull": "牛市"}
    model_labels = {"ridge": "Ridge", "cnn_gru": "CNN-GRU"}
    display_plan = pd.DataFrame(
        {
            "信号日期": combined_plan["signal_date"].dt.strftime("%Y-%m-%d"),
            "模型": combined_plan["model_name"].map(model_labels).fillna(combined_plan["model_name"]),
            "平滑预测": combined_plan["prediction_smoothed"].map(lambda value: f"{float(value):.4f}"),
            "z值": combined_plan["prediction_zscore"].map(lambda value: f"{float(value):.2f}"),
            "市场状态": combined_plan["desired_state"].map(state_labels),
            "是否调仓": combined_plan["trade_required"].map({True: "是", False: "否"}),
            "操作": combined_plan.apply(
                lambda row: "买入/加仓"
                if float(row["exposure_change"]) > 0
                else ("卖出/减仓" if float(row["exposure_change"]) < 0 else "维持"),
                axis=1,
            ),
            "目标仓位": combined_plan["target_exposure"].map(lambda value: f"{float(value):.0%}"),
            "计划执行日": combined_plan["planned_execution_date"].map(
                lambda value: "" if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")
            ),
        }
    )
    display_plan.to_csv(
        comparison_report / "rebalance_signals.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics = pd.DataFrame([result.metrics for result in results])
    metrics.to_parquet(comparison_dir / "performance_summary.parquet", index=False)

    equity = None
    for result in results:
        frame = result.equity[["date", "value"]].copy()
        frame[result.model_name] = frame["value"] / float(result.metrics["initial_cash"])
        frame = frame.drop(columns="value")
        equity = frame if equity is None else equity.merge(frame, on="date", how="inner")
    index_curve, index_sharpe = build_index_curve(price, equity["date"])
    equity = equity.merge(index_curve, on="date", how="left")
    equity.to_parquet(comparison_dir / "equity_comparison.parquet", index=False)
    benchmark = {
        "name": "CSI1000_index",
        "sharpe": index_sharpe,
        "start_date": equity["date"].min().strftime("%Y-%m-%d"),
        "end_date": equity["date"].max().strftime("%Y-%m-%d"),
    }
    (comparison_report / "benchmark.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(12, 6))
    for result in results:
        axis.plot(equity["date"], equity[result.model_name], label=result.model_name)
    axis.plot(equity["date"], equity["index_nav"], label="CSI1000 index", color="black", alpha=0.75)
    axis.set_title("Long/Cash Timing vs CSI1000 Index")
    axis.set_ylabel("Normalized NAV")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    chart_path = comparison_report / "equity_comparison.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    write_html_report(results, benchmark, chart_path, comparison_report / "backtest_report.html", config)

    print(json.dumps({"benchmark": benchmark, "signal_count": len(signal_dates)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
