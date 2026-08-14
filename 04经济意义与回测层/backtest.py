"""中证1000指数代理资产的多头/现金 Backtrader 回测。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import backtrader as bt
import numpy as np
import pandas as pd
import yaml


LAYER_DIR = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_layer_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (LAYER_DIR / path).resolve()


def load_price(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = ["trade_date", "open", "high", "low", "close", "vol"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"中证1000行情缺少字段: {missing}")
    frame["date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="raise")
    frame = frame.sort_values("date").drop_duplicates("date", keep=False)
    for column in ["open", "high", "low", "close", "vol"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("中证1000OHLC包含缺失值")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("中证1000OHLC包含非正值")
    return frame.set_index("date")[["open", "high", "low", "close", "vol"]]


def slice_backtest_period(price: pd.DataFrame, config: dict) -> pd.DataFrame:
    """按正式回测配置截取行情；配置日期为日历边界。"""
    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])
    if end < start:
        raise ValueError("backtest.end_date不能早于start_date")
    result = price.loc[start:end].copy()
    if result.empty:
        raise ValueError(f"回测区间没有行情: {start.date()} ~ {end.date()}")
    return result


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = ["signal_date", "prediction"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"{model_name}预测文件缺少字段: {missing}")
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame = frame.sort_values("signal_date").reset_index(drop=True)
    if frame["signal_date"].duplicated().any():
        raise ValueError(f"{model_name}预测日期重复")
    if not np.isfinite(frame["prediction"].to_numpy(dtype=float)).all():
        raise ValueError(f"{model_name}预测包含NaN或无穷值")
    optional = [column for column in ["prediction_raw", "smoothing_method", "smoothing_halflife_days"] if column in frame]
    return frame[["signal_date", "prediction", *optional]]


def common_prediction_dates(predictions: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    date_sets = [set(pd.to_datetime(frame["signal_date"])) for frame in predictions.values()]
    common = set.intersection(*date_sets)
    if not common:
        raise ValueError("模型之间没有共同预测日期")
    return pd.DatetimeIndex(sorted(common))


def build_signal_dates(
    prediction_dates: Iterable[pd.Timestamp],
    price_dates: pd.DatetimeIndex,
    frequency: str,
) -> List[pd.Timestamp]:
    prediction_dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(prediction_dates))))
    price_dates = pd.DatetimeIndex(price_dates)
    available = prediction_dates[prediction_dates.isin(price_dates)]
    if frequency != "monthly_first_trading_day":
        raise ValueError(f"不支持的调仓频率: {frequency}")
    schedule_frame = pd.DataFrame({"signal_date": available})
    schedule_frame["month"] = schedule_frame["signal_date"].dt.to_period("M")
    scheduled = pd.DatetimeIndex(schedule_frame.groupby("month", sort=True)["signal_date"].first())
    result = []
    for date in scheduled:
        position = price_dates.searchsorted(date, side="right")
        if position < len(price_dates):
            result.append(pd.Timestamp(date))
    if not result:
        raise ValueError("没有可执行的调仓信号")
    return result


def build_decision_dates(
    prediction_dates: Iterable[pd.Timestamp],
    price_dates: pd.DatetimeIndex,
    frequency: str,
) -> List[pd.Timestamp]:
    """生成信号决策日；不要求T+1行情已经到达。"""
    prediction_dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(prediction_dates))))
    available = prediction_dates[prediction_dates.isin(pd.DatetimeIndex(price_dates))]
    if frequency != "monthly_first_trading_day":
        raise ValueError(f"不支持的调仓频率: {frequency}")
    schedule = pd.DataFrame({"signal_date": available})
    schedule["month"] = schedule["signal_date"].dt.to_period("M")
    return [pd.Timestamp(value) for value in schedule.groupby("month", sort=True)["signal_date"].first()]


def build_rebalance_plan(
    model_name: str,
    predictions: pd.DataFrame,
    decision_dates: Iterable[pd.Timestamp],
    price_dates: pd.DatetimeIndex,
    config: dict,
) -> pd.DataFrame:
    """只根据截至T日的预测生成调仓计划，不依赖Backtrader或T+1成交结果。"""
    dates = [pd.Timestamp(value) for value in decision_dates]
    indexed = predictions.set_index("signal_date")
    missing = pd.DatetimeIndex(dates).difference(indexed.index)
    if len(missing):
        raise ValueError(f"{model_name}缺少决策日预测: {missing[:5].tolist()}")
    threshold = float(config["signal_threshold"])
    z_threshold = float(config["state_z_threshold"])
    window = int(config["standardization_window"])
    bull_exposure = float(config["target_exposure"])
    neutral_exposure = float(config["neutral_exposure"])
    if not 0.0 <= neutral_exposure <= bull_exposure <= 1.0:
        raise ValueError("状态仓位必须满足0 <= neutral_exposure <= target_exposure <= 1")
    ordered = predictions.sort_values("signal_date").reset_index(drop=True).copy()
    history = ordered["prediction"].astype(float).shift(1)
    ordered["prediction_history_mean"] = history.rolling(window, min_periods=window).mean()
    ordered["prediction_history_std"] = history.rolling(window, min_periods=window).std(ddof=1)
    ordered["prediction_zscore"] = (
        ordered["prediction"] - ordered["prediction_history_mean"]
    ) / ordered["prediction_history_std"]
    indexed = ordered.set_index("signal_date")
    prior_state = "bear"
    prior_exposure = 0.0
    rows = []
    for date in dates:
        source = indexed.loc[date]
        prediction = float(source["prediction"])
        history_mean = float(source["prediction_history_mean"])
        history_std = float(source["prediction_history_std"])
        zscore = float(source["prediction_zscore"])
        if not np.isfinite([history_mean, history_std, zscore]).all() or history_std <= 0:
            raise ValueError(f"{model_name}在{date.date()}没有{window}日有效标准化历史")
        if prediction < threshold and zscore <= -z_threshold:
            desired_state, target_exposure = "bear", 0.0
        elif prediction > threshold and zscore >= z_threshold:
            desired_state, target_exposure = "bull", bull_exposure
        else:
            desired_state, target_exposure = "neutral", neutral_exposure
        exposure_change = target_exposure - prior_exposure
        if exposure_change > 1e-12:
            action, trade_required = "BUY", True
        elif exposure_change < -1e-12:
            action, trade_required = "SELL", True
        else:
            action, trade_required = f"HOLD_{desired_state.upper()}", False
        next_position = pd.DatetimeIndex(price_dates).searchsorted(date, side="right")
        planned_date = (
            pd.Timestamp(price_dates[next_position])
            if next_position < len(price_dates)
            else pd.NaT
        )
        raw_value = source.get("prediction_raw", np.nan)
        rows.append(
            {
                "signal_date": date,
                "model_name": model_name,
                "instrument": "CSI1000_PROXY",
                "ts_code": "000852.SH",
                "prediction_raw": float(raw_value) if pd.notna(raw_value) else np.nan,
                "prediction_smoothed": prediction,
                "prediction_history_mean": history_mean,
                "prediction_history_std": history_std,
                "prediction_zscore": zscore,
                "signal_threshold": threshold,
                "state_z_threshold": z_threshold,
                "previous_target_state": prior_state,
                "desired_state": desired_state,
                "action": action,
                "trade_required": trade_required,
                "previous_target_exposure": prior_exposure,
                "target_exposure": target_exposure,
                "exposure_change": exposure_change,
                "execution_rule": "NEXT_TRADING_DAY_OPEN",
                "planned_execution_date": planned_date,
            }
        )
        prior_state = desired_state
        prior_exposure = target_exposure
    return pd.DataFrame(rows)


def save_rebalance_plan(plan: pd.DataFrame, report_root: Path, model_name: str) -> None:
    output_dir = Path(report_root) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_dir / "rebalance_signals.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")


def next_trade_date(signal_date: pd.Timestamp, price_dates: pd.DatetimeIndex) -> pd.Timestamp:
    position = price_dates.searchsorted(pd.Timestamp(signal_date), side="right")
    if position >= len(price_dates):
        raise ValueError(f"{signal_date.date()}之后没有下一交易日")
    return pd.Timestamp(price_dates[position])


class IndexLongCashStrategy(bt.Strategy):
    """T日收盘读信号，Backtrader默认在T+1开盘成交。"""

    params = (
        ("plan_map", None),
    )

    def __init__(self):
        self.pending_order = None
        self.equity_records: List[dict] = []
        self.signal_records: List[dict] = []
        self.order_records: List[dict] = []
        self.trade_records: List[dict] = []
        self.current_target_exposure = 0.0

    def next(self):
        date = pd.Timestamp(self.datas[0].datetime.date(0))
        plan = self.p.plan_map.get(date)
        if plan is not None:
            prediction = float(plan["prediction_smoothed"])
            desired_state = str(plan["desired_state"])
            target_exposure = float(plan["target_exposure"])
            action = f"hold_{desired_state}"
            if self.pending_order is not None:
                action = "pending_order"
            elif not np.isclose(target_exposure, self.current_target_exposure, atol=1e-12):
                self.pending_order = self.order_target_percent(target=target_exposure)
                action = "increase_exposure" if target_exposure > self.current_target_exposure else "decrease_exposure"
                self.current_target_exposure = target_exposure
            self.signal_records.append(
                {
                    "signal_date": date,
                    "prediction": prediction,
                    "prediction_zscore": float(plan["prediction_zscore"]),
                    "desired_state": desired_state,
                    "target_exposure": target_exposure,
                    "action": action,
                }
            )

        value = float(self.broker.getvalue())
        position_value = float(self.position.size * self.data.close[0])
        self.equity_records.append(
            {
                "date": date,
                "value": value,
                "cash": float(self.broker.getcash()),
                "position_size": float(self.position.size),
                "position_value": position_value,
                "exposure": position_value / value if value else np.nan,
            }
        )

    @staticmethod
    def _bt_date(value) -> pd.Timestamp:
        return pd.Timestamp(bt.num2date(value).date()) if value else pd.NaT

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        record = {
            "order_ref": int(order.ref),
            "status": order.getstatusname(),
            "created_date": self._bt_date(order.created.dt),
            "executed_date": self._bt_date(order.executed.dt),
            "is_buy": bool(order.isbuy()),
            "created_size": float(order.created.size),
            "executed_size": float(order.executed.size),
            "executed_price": float(order.executed.price),
            "executed_value": float(order.executed.value),
            "commission": float(order.executed.comm),
        }
        self.order_records.append(record)
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.pending_order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.trade_records.append(
            {
                "entry_date": self._bt_date(trade.dtopen),
                "exit_date": self._bt_date(trade.dtclose),
                "bars": int(trade.barlen),
                "gross_pnl": float(trade.pnl),
                "net_pnl": float(trade.pnlcomm),
                "won": bool(trade.pnlcomm > 0),
            }
        )


@dataclass
class BacktestResult:
    model_name: str
    equity: pd.DataFrame
    signals: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def calculate_metrics(equity: pd.DataFrame, trades: pd.DataFrame, config: dict) -> dict:
    annualization = int(config["annualization_days"])
    risk_free = float(config.get("risk_free_rate", 0.0))
    values = equity.sort_values("date")["value"].astype(float)
    returns = values.pct_change().dropna()
    if len(returns) < 2:
        raise ValueError("净值序列太短，无法计算绩效")
    total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
    annual_return = float((1.0 + total_return) ** (annualization / len(returns)) - 1.0)
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(annualization))
    daily_rf = risk_free / annualization
    excess = returns - daily_rf
    sharpe = float(excess.mean() / returns.std(ddof=1) * np.sqrt(annualization)) if annual_volatility > 0 else np.nan
    downside = np.minimum(excess.to_numpy(dtype=float), 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * np.sqrt(annualization))
    sortino = float(excess.mean() * annualization / downside_deviation) if downside_deviation > 0 else np.nan
    drawdown = values / values.cummax() - 1.0
    max_drawdown = float(-drawdown.min())
    calmar = float(annual_return / max_drawdown) if max_drawdown > 0 else np.nan
    closed_trades = int(len(trades))
    win_rate = float(trades["won"].mean()) if closed_trades else np.nan
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "closed_trades": closed_trades,
        "start_date": equity["date"].min().strftime("%Y-%m-%d"),
        "end_date": equity["date"].max().strftime("%Y-%m-%d"),
    }


def run_model_backtest(
    model_name: str,
    price: pd.DataFrame,
    predictions: pd.DataFrame,
    signal_dates: Iterable[pd.Timestamp],
    config: dict,
    rebalance_plan: pd.DataFrame,
) -> BacktestResult:
    signal_dates = [pd.Timestamp(value) for value in signal_dates]
    prediction_map = predictions.set_index("signal_date")["prediction"]
    missing = pd.DatetimeIndex(signal_dates).difference(prediction_map.index)
    if len(missing):
        raise ValueError(f"{model_name}缺少调仓预测: {missing[:5].tolist()}")
    executable_plan = rebalance_plan.set_index("signal_date").loc[signal_dates]
    plan_map = {date: row.to_dict() for date, row in executable_plan.iterrows()}
    feed_frame = price.loc[signal_dates[0] :].copy()
    feed_frame["openinterest"] = 0.0

    cerebro = bt.Cerebro(stdstats=False)
    data = bt.feeds.PandasData(
        dataname=feed_frame,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="vol",
        openinterest="openinterest",
    )
    cerebro.adddata(data, name="CSI1000_PROXY")
    cerebro.addstrategy(
        IndexLongCashStrategy,
        plan_map=plan_map,
    )
    cerebro.broker.setcash(float(config["initial_cash"]))
    cerebro.broker.setcommission(commission=float(config["commission"]))
    strategies = cerebro.run(runonce=True, preload=True)
    strategy = strategies[0]

    equity = pd.DataFrame(strategy.equity_records).sort_values("date").reset_index(drop=True)
    signals = pd.DataFrame(strategy.signal_records).sort_values("signal_date").reset_index(drop=True)
    orders = pd.DataFrame(strategy.order_records)
    trades = pd.DataFrame(strategy.trade_records)
    if trades.empty:
        trades = pd.DataFrame(columns=["entry_date", "exit_date", "bars", "gross_pnl", "net_pnl", "won"])
    metrics = calculate_metrics(equity, trades, config)
    metrics.update(
        {
            "model_name": model_name,
            "initial_cash": float(config["initial_cash"]),
            "target_exposure": float(config["target_exposure"]),
            "commission": float(config["commission"]),
            "signal_count": int(len(signals)),
            "completed_orders": int((orders.get("status", pd.Series(dtype=str)) == "Completed").sum()),
        }
    )
    return BacktestResult(model_name, equity, signals, orders, trades, metrics)


def build_index_curve(price: pd.DataFrame, equity_dates: Iterable[pd.Timestamp]) -> Tuple[pd.DataFrame, float]:
    dates = pd.DatetimeIndex(pd.to_datetime(list(equity_dates)))
    if len(dates) < 2:
        raise ValueError("基准日期太短")
    signal_date = dates[0]
    entry_date = next_trade_date(signal_date, price.index)
    aligned = price.loc[dates, ["close"]].copy()
    aligned["index_nav"] = 1.0
    invested = aligned.index >= entry_date
    aligned.loc[invested, "index_nav"] = aligned.loc[invested, "close"] / float(price.loc[entry_date, "open"])
    returns = aligned["index_nav"].pct_change().dropna()
    volatility = returns.std(ddof=1)
    sharpe = float(returns.mean() / volatility * np.sqrt(252)) if volatility > 0 else np.nan
    result = aligned[["index_nav"]].reset_index().rename(columns={"index": "date"})
    return result, sharpe


def save_result(result: BacktestResult, data_root: Path, report_root: Path) -> None:
    model_data = Path(data_root) / "processed" / result.model_name
    model_report = Path(report_root) / result.model_name
    model_data.mkdir(parents=True, exist_ok=True)
    model_report.mkdir(parents=True, exist_ok=True)
    result.equity.to_parquet(model_data / "equity_curve.parquet", index=False)
    result.signals.to_parquet(model_data / "signals.parquet", index=False)
    result.orders.to_parquet(model_data / "orders.parquet", index=False)
    result.trades.to_parquet(model_data / "trades.parquet", index=False)
    (model_report / "performance.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
