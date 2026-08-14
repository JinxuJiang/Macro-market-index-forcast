"""特征层公共数据加载、数学公式和数据结构。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class LayerResult:
    """单个信息层的统一输出。"""

    features: pd.DataFrame
    lineage: pd.DataFrame
    specs: Dict[str, dict]


class FeatureDataLoader:
    """只读加载01数据层正式文件，并执行输入契约检查。"""

    def __init__(self, project_root: Path, config: dict, end_date: Optional[str] = None):
        self.project_root = Path(project_root)
        self.config = config
        self.end_date = str(end_date) if end_date else None

    def path(self, source_name: str) -> Path:
        return self.project_root / self.config["sources"][source_name]

    def load(
        self,
        source_name: str,
        required: Iterable[str],
        date_column: Optional[str] = None,
        monthly: bool = False,
        keys: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        path = self.path(source_name)
        if not path.exists():
            raise FileNotFoundError(path)
        data = pd.read_parquet(path)
        missing = sorted(set(required) - set(data.columns))
        if missing:
            raise ValueError("{} 缺少字段: {}".format(source_name, missing))
        if keys:
            duplicate = int(data.duplicated(list(keys)).sum())
            if duplicate:
                raise ValueError("{} 主键重复: {}".format(source_name, duplicate))
        if date_column:
            data = data.copy()
            data[date_column] = data[date_column].astype(str)
            if self.end_date:
                cutoff = self.end_date[:6] if monthly else self.end_date
                data = data[data[date_column] <= cutoff]
            data = data.sort_values(date_column).reset_index(drop=True)
        return data

    def trade_dates(self) -> pd.Index:
        calendar = self.load(
            "trade_calendar",
            ["exchange", "cal_date", "is_open"],
            date_column="cal_date",
            keys=["exchange", "cal_date"],
        )
        opened = calendar.loc[calendar["is_open"].astype(int) == 1, "cal_date"]
        return pd.Index(opened.astype(str).drop_duplicates().tolist(), name="trade_date")

    def availability(self) -> pd.DataFrame:
        required = [
            "dataset",
            "data_date",
            "period_date",
            "available_date",
            "availability_method",
        ]
        data = self.load("availability", required)
        data = data.copy()
        for column in ("dataset", "data_date", "period_date", "available_date"):
            data[column] = data[column].astype(str)
        if data.duplicated(["dataset", "data_date"]).any():
            raise ValueError("data_availability存在重复(dataset, data_date)")
        return data


def safe_divide(numerator, denominator):
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def simple_return(values: pd.Series, window: int) -> pd.Series:
    return safe_divide(values, values.shift(window)) - 1.0


def log_difference(values: pd.Series, window: int = 1) -> pd.Series:
    positive = values.where(values > 0)
    return np.log(positive).diff(window)


def annualized_volatility(
    values: pd.Series, window: int, annualization_days: int = 252
) -> pd.Series:
    returns = values.pct_change(fill_method=None)
    return returns.rolling(window, min_periods=window).std(ddof=1) * math.sqrt(
        annualization_days
    )


def moving_average_distance(values: pd.Series, window: int) -> pd.Series:
    average = values.rolling(window, min_periods=window).mean()
    return safe_divide(values, average) - 1.0


def rolling_drawdown(values: pd.Series, window: int) -> pd.Series:
    peak = values.rolling(window, min_periods=window).max()
    return safe_divide(values, peak) - 1.0


def rolling_mean_ratio(values: pd.Series, window: int) -> pd.Series:
    average = values.rolling(window, min_periods=window).mean()
    return safe_divide(values, average)


def price_features(
    values: pd.Series,
    prefix: str,
    return_windows: Iterable[int],
    volatility_windows: Iterable[int],
    annualization_days: int,
) -> pd.DataFrame:
    result = pd.DataFrame(index=values.index)
    for window in return_windows:
        result["{}_ret_{}d".format(prefix, window)] = simple_return(values, window)
    for window in volatility_windows:
        result["{}_vol_{}d".format(prefix, window)] = annualized_volatility(
            values, window, annualization_days
        )
    return result

