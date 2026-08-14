"""将原生频率特征按实际可用日映射到中国交易日。"""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


class PITAligner:
    """data_availability是可用规则的唯一事实来源。"""

    def __init__(self, availability: pd.DataFrame, trade_dates: pd.Index):
        self.availability = availability.copy()
        self.trade_dates = pd.Index(trade_dates.astype(str), name="trade_date")
        self._trade_position = {
            date: position for position, date in enumerate(self.trade_dates.tolist())
        }

    def align(
        self,
        dataset: str,
        native_features: pd.DataFrame,
        data_date_column: str,
        feature_columns: Iterable[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """按available_date向后as-of展开，同时返回来源日期长表。"""
        feature_columns = list(feature_columns)
        if native_features.duplicated([data_date_column]).any():
            raise ValueError("{} 原生特征数据日期重复".format(dataset))

        source = native_features[[data_date_column] + feature_columns].copy()
        source[data_date_column] = source[data_date_column].astype(str)
        source = source.rename(columns={data_date_column: "data_date"})

        mapping = self.availability[self.availability["dataset"] == dataset][
            ["data_date", "period_date", "available_date"]
        ].copy()
        if mapping.empty:
            raise ValueError("可用日期表缺少数据集: {}".format(dataset))

        events = source.merge(mapping, on="data_date", how="inner", validate="one_to_one")
        if events.empty:
            raise ValueError("{} 原始数据与可用日期表无法匹配".format(dataset))

        last_trade_date = self.trade_dates[-1]
        events = events[events["available_date"] <= last_trade_date].copy()
        events = events.sort_values(["available_date", "data_date"])
        # 项目起点前的多期低频数据可能被压到同一个可用日，取最新统计期。
        events = events.drop_duplicates("available_date", keep="last")

        left = pd.DataFrame({"trade_date": self.trade_dates})
        left["_trade_dt"] = pd.to_datetime(left["trade_date"], format="%Y%m%d")
        right = events.copy()
        right["_available_dt"] = pd.to_datetime(
            right["available_date"], format="%Y%m%d"
        )
        right = right.sort_values("_available_dt")

        aligned = pd.merge_asof(
            left.sort_values("_trade_dt"),
            right,
            left_on="_trade_dt",
            right_on="_available_dt",
            direction="backward",
            allow_exact_matches=True,
        )
        aligned = aligned.set_index("trade_date")
        result = aligned[feature_columns].copy()
        result.index.name = "trade_date"

        lineage = aligned[["data_date", "period_date", "available_date"]].copy()
        lineage = lineage.rename(
            columns={
                "data_date": "source_data_date",
                "period_date": "source_period_date",
                "available_date": "source_available_date",
            }
        )
        lineage.insert(0, "dataset", dataset)
        lineage = lineage.reset_index()
        lineage["age_trade_days"] = lineage.apply(self._age_trade_days, axis=1)

        valid = lineage["source_available_date"].notna()
        early = lineage.loc[valid, "source_available_date"].astype(str) > lineage.loc[
            valid, "trade_date"
        ].astype(str)
        if early.any():
            raise AssertionError("{} PIT映射提前使用数据".format(dataset))
        return result, lineage

    def _age_trade_days(self, row):
        available = row.get("source_available_date")
        trade_date = str(row.get("trade_date"))
        if pd.isna(available):
            return pd.NA
        available = str(available)
        if available not in self._trade_position or trade_date not in self._trade_position:
            return pd.NA
        return self._trade_position[trade_date] - self._trade_position[available]

