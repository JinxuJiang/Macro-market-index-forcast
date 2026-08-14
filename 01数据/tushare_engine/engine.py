"""Tushare 市场状态数据引擎。

负责三层原始数据的下载、增量合并、PIT 可用时点映射和质量验收。
本层不计算模型特征。
"""

from __future__ import annotations

import bisect
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

DATE_FMT = "%Y%m%d"
INDEX_FIELDS = (
    "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
)
GLOBAL_INDEX_FIELDS = (
    "ts_code,trade_date,open,close,high,low,pre_close,change,pct_chg,swing,vol"
)


def load_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parents[2])
    return config


class MarketDataEngine:
    """市场状态模型专用的轻量 Tushare 数据引擎。"""

    def __init__(
        self,
        config: dict,
        token: str | None = None,
        pro_client: Any | None = None,
    ) -> None:
        self.config = config
        self.project_root = Path(config["_project_root"])
        self.data_dir = self.project_root / "01数据"
        self.root = self.data_dir / config["project"].get("data_root", "data")
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.layer1_dir = self.raw_dir / "layer1_target"
        self.layer2_dir = self.raw_dir / "layer2_macro_cross_asset"
        self.layer3_dir = self.raw_dir / "layer3_market_structure"
        self.log_dir = self.root / "logs"
        for path in (
            self.raw_dir,
            self.processed_dir,
            self.layer1_dir,
            self.layer2_dir,
            self.layer3_dir,
            self.log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        tushare_cfg = config.get("tushare", {})
        self.request_interval_sec = float(
            os.getenv(
                "TUSHARE_INTERVAL_SEC",
                tushare_cfg.get("request_interval_sec", 0.15),
            )
        )
        self.max_retries = int(tushare_cfg.get("max_retries", 3))
        self.retry_sleep_sec = float(tushare_cfg.get("retry_sleep_sec", 2.0))
        self._last_call_at = 0.0
        self._pro = pro_client
        self.token = token or (None if pro_client is not None else self._read_token())

    @property
    def target_index(self) -> dict:
        return self.config["layer1_target"]["index"]

    @property
    def layer2_config(self) -> dict:
        return self.config.get("layer2_macro_cross_asset", {})

    @property
    def layer3_config(self) -> dict:
        return self.config.get("layer3_market_structure", {})

    @property
    def trade_cal_file(self) -> Path:
        return self.raw_dir / "trade_cal.parquet"

    @property
    def shibor_file(self) -> Path:
        return self.layer2_dir / "shibor.parquet"

    @property
    def margin_file(self) -> Path:
        return self.layer2_dir / "margin.parquet"

    @property
    def pmi_file(self) -> Path:
        return self.layer2_dir / "pmi.parquet"

    @property
    def cpi_file(self) -> Path:
        return self.layer2_dir / "cpi.parquet"

    @property
    def fx_file(self) -> Path:
        return self.layer2_dir / f"{self.layer2_config['fx']['ts_code']}.parquet"

    @property
    def gold_file(self) -> Path:
        return self.layer2_dir / f"{self.layer2_config['gold']['ts_code']}.parquet"

    @property
    def availability_file(self) -> Path:
        return self.processed_dir / "data_availability.parquet"

    @property
    def pro(self) -> Any:
        if self._pro is None:
            import tushare as ts

            self._pro = ts.pro_api(self.token)
        return self._pro

    def domestic_index_path(self, ts_code: str) -> Path:
        if ts_code == self.target_index["ts_code"]:
            return self.layer1_dir / f"{ts_code}.parquet"
        return self.layer3_dir / f"{ts_code}.parquet"

    def global_index_path(self, ts_code: str) -> Path:
        return self.layer3_dir / f"{ts_code}.parquet"

    def configured_domestic_indices(self) -> list[dict]:
        return [self.target_index, *self.layer3_config.get("domestic_indices", [])]

    def configured_global_indices(self) -> list[dict]:
        return list(self.layer3_config.get("global_indices", []))

    def _resolve_from_project(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _read_token(self) -> str:
        env_token = os.getenv("TUSHARE_TOKEN", "").strip()
        if env_token:
            return env_token
        cfg = self.config.get("tushare", {})
        candidates = [
            self.data_dir / cfg.get("token_file", "tushare_token.txt"),
            self._resolve_from_project(cfg.get("fallback_token_file", "")),
        ]
        for path in candidates:
            if path.is_file():
                token = path.read_text(encoding="utf-8").strip()
                if token:
                    return token
        raise FileNotFoundError(
            "未找到 Tushare token。请设置 TUSHARE_TOKEN，或创建 "
            f"{candidates[0]}"
        )

    def _call(self, api_name: str, **kwargs: Any) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            elapsed = time.monotonic() - self._last_call_at
            if elapsed < self.request_interval_sec:
                time.sleep(self.request_interval_sec - elapsed)
            try:
                result = getattr(self.pro, api_name)(**kwargs)
                self._last_call_at = time.monotonic()
                return result if result is not None else pd.DataFrame()
            except Exception as exc:  # Tushare 异常类型不稳定，统一重试
                self._last_call_at = time.monotonic()
                last_error = exc
                if attempt < self.max_retries:
                    wait_seconds = (
                        20.0 * attempt
                        if "频率超限" in str(exc)
                        else self.retry_sleep_sec * attempt
                    )
                    time.sleep(wait_seconds)
        raise RuntimeError(f"Tushare {api_name} 调用失败: {last_error}") from last_error

    @staticmethod
    def _year_ranges(start_date: str, end_date: str) -> Iterable[tuple[str, str]]:
        start = dt.datetime.strptime(start_date, DATE_FMT).date()
        end = dt.datetime.strptime(end_date, DATE_FMT).date()
        if end < start:
            return
        for year in range(start.year, end.year + 1):
            left = max(start, dt.date(year, 1, 1))
            right = min(end, dt.date(year, 12, 31))
            yield left.strftime(DATE_FMT), right.strftime(DATE_FMT)

    @staticmethod
    def _multi_year_ranges(
        start_date: str, end_date: str, years_per_chunk: int
    ) -> Iterable[tuple[str, str]]:
        """按多年分段，减少受总行数限制接口的请求次数。"""
        start = dt.datetime.strptime(start_date, DATE_FMT).date()
        end = dt.datetime.strptime(end_date, DATE_FMT).date()
        left = start
        while left <= end:
            chunk_end_year = min(left.year + years_per_chunk - 1, end.year)
            right = min(end, dt.date(chunk_end_year, 12, 31))
            yield left.strftime(DATE_FMT), right.strftime(DATE_FMT)
            left = right + dt.timedelta(days=1)

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temp, index=False, compression="zstd")
        temp.replace(path)

    @staticmethod
    def _atomic_json(payload: dict, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp.replace(path)

    @staticmethod
    def _merge(
        existing_path: Path,
        new_frames: list[pd.DataFrame],
        keys: list[str],
        sort_by: list[str],
        legacy_path: Path | None = None,
    ) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        if existing_path.exists():
            parts.append(pd.read_parquet(existing_path))
        elif legacy_path is not None and legacy_path.exists():
            parts.append(pd.read_parquet(legacy_path))
        parts.extend(frame for frame in new_frames if not frame.empty)
        if not parts:
            return pd.DataFrame()
        # Pandas 对“部分分片整列为空”的 dtype 推断将发生变化。先从每个
        # 分片移除其全空列，合并后再补回全局全空列，保持字段且消除告警。
        all_columns = list(dict.fromkeys(col for frame in parts for col in frame.columns))
        prepared = [
            frame.drop(columns=[col for col in frame.columns if frame[col].isna().all()])
            for frame in parts
        ]
        result = pd.concat(prepared, ignore_index=True)
        for column in all_columns:
            if column not in result.columns:
                result[column] = pd.NA
        result = result.reindex(columns=all_columns)
        result = result.drop_duplicates(keys, keep="last")
        return result.sort_values(sort_by).reset_index(drop=True)

    @staticmethod
    def _describe_output(path: Path) -> str:
        data = pd.read_parquet(path)
        date_col = next(
            (name for name in ("trade_date", "date", "month", "cal_date") if name in data),
            None,
        )
        coverage = ""
        if date_col and not data.empty:
            coverage = f"，范围 {data[date_col].min()}..{data[date_col].max()}"
        return f"{len(data)} 行{coverage}"

    def _progress(self, position: int, total: int, label: str, path: Path) -> None:
        print(
            f"[{position:02d}/{total:02d}] {label}：{self._describe_output(path)}",
            flush=True,
        )

    @staticmethod
    def default_end_date() -> str:
        """盘中默认截至昨天；收盘数据稳定后允许截至今天。"""
        # 中国标准时间全年固定为 UTC+8；兼容当前 Python 3.8 环境。
        now = dt.datetime.utcnow() + dt.timedelta(hours=8)
        day = now.date() if now.hour >= 18 else now.date() - dt.timedelta(days=1)
        return day.strftime(DATE_FMT)

    def download_trade_calendar(self, start_date: str, end_date: str) -> Path:
        frames = [
            self._call(
                "trade_cal",
                exchange="SSE",
                start_date=left,
                end_date=right,
                fields="exchange,cal_date,is_open,pretrade_date",
            )
            for left, right in self._year_ranges(start_date, end_date)
        ]
        data = self._merge(
            self.trade_cal_file,
            frames,
            keys=["exchange", "cal_date"],
            sort_by=["cal_date"],
        )
        self._atomic_parquet(data, self.trade_cal_file)
        return self.trade_cal_file

    def open_trade_dates(self, start_date: str, end_date: str) -> list[str]:
        if not self.trade_cal_file.exists():
            raise FileNotFoundError(self.trade_cal_file)
        data = pd.read_parquet(self.trade_cal_file)
        mask = (
            (data["is_open"].astype(int) == 1)
            & data["cal_date"].astype(str).between(start_date, end_date)
        )
        return data.loc[mask, "cal_date"].astype(str).sort_values().tolist()

    def _incremental_start(
        self,
        path: Path,
        date_col: str,
        default: str,
        legacy_path: Path | None = None,
    ) -> str:
        source = path if path.exists() else legacy_path
        if source is None or not source.exists():
            return default
        data = pd.read_parquet(source, columns=[date_col])
        if data.empty:
            return default
        latest = dt.datetime.strptime(str(data[date_col].max()), DATE_FMT).date()
        return max(
            dt.datetime.strptime(default, DATE_FMT).date(),
            latest - dt.timedelta(days=10),
        ).strftime(DATE_FMT)

    def download_index(self, ts_code: str, start_date: str, end_date: str) -> Path:
        path = self.domestic_index_path(ts_code)
        legacy = self.raw_dir / "index_daily" / f"{ts_code}.parquet"
        fetch_start = self._incremental_start(path, "trade_date", start_date, legacy)
        frames = [
            self._call(
                "index_daily",
                ts_code=ts_code,
                start_date=left,
                end_date=right,
                fields=INDEX_FIELDS,
            )
            for left, right in self._multi_year_ranges(fetch_start, end_date, 8)
        ]
        data = self._merge(
            path,
            frames,
            keys=["ts_code", "trade_date"],
            sort_by=["trade_date"],
            legacy_path=legacy,
        )
        if data.empty:
            raise RuntimeError(f"指数 {ts_code} 未返回任何数据，请检查代码和权限")
        self._atomic_parquet(data, path)
        return path

    def download_global_index(
        self, ts_code: str, start_date: str, end_date: str
    ) -> Path:
        path = self.global_index_path(ts_code)
        fetch_start = self._incremental_start(path, "trade_date", start_date)
        frames = [
            self._call(
                "index_global",
                ts_code=ts_code,
                start_date=left,
                end_date=right,
                fields=GLOBAL_INDEX_FIELDS,
            )
            for left, right in self._multi_year_ranges(fetch_start, end_date, 8)
        ]
        data = self._merge(path, frames, ["ts_code", "trade_date"], ["trade_date"])
        if data.empty:
            raise RuntimeError(f"国际指数 {ts_code} 未返回任何数据")
        self._atomic_parquet(data, path)
        return path

    def download_shibor(self, start_date: str, end_date: str) -> Path:
        legacy = self.raw_dir / "macro" / "shibor.parquet"
        fetch_start = self._incremental_start(
            self.shibor_file, "date", start_date, legacy
        )
        frames = [
            self._call("shibor", start_date=left, end_date=right)
            for left, right in self._multi_year_ranges(fetch_start, end_date, 3)
        ]
        data = self._merge(
            self.shibor_file, frames, ["date"], ["date"], legacy_path=legacy
        )
        self._atomic_parquet(data, self.shibor_file)
        return self.shibor_file

    def download_margin(self, start_date: str, end_date: str) -> Path:
        legacy = self.raw_dir / "margin" / "margin.parquet"
        fetch_start = self._incremental_start(
            self.margin_file, "trade_date", start_date, legacy
        )
        frames = [
            self._call("margin", start_date=left, end_date=right)
            for left, right in self._year_ranges(fetch_start, end_date)
        ]
        data = self._merge(
            self.margin_file,
            frames,
            ["trade_date", "exchange_id"],
            ["trade_date", "exchange_id"],
            legacy_path=legacy,
        )
        self._atomic_parquet(data, self.margin_file)
        return self.margin_file

    def download_monthly_macro(self, api_name: str, path: Path) -> Path:
        frame = self._call(api_name)
        if "month" not in frame.columns and "MONTH" in frame.columns:
            frame = frame.rename(columns={"MONTH": "month"})
        if frame.empty or "month" not in frame.columns:
            raise RuntimeError(f"{api_name} 未返回有效的 month 数据")
        frame["month"] = frame["month"].astype(str)
        data = self._merge(path, [frame], ["month"], ["month"])
        self._atomic_parquet(data, path)
        return path

    def download_fx(self, ts_code: str, start_date: str, end_date: str) -> Path:
        path = self.fx_file
        fetch_start = self._incremental_start(path, "trade_date", start_date)
        frames = [
            self._call(
                "fx_daily", ts_code=ts_code, start_date=left, end_date=right
            )
            for left, right in self._multi_year_ranges(fetch_start, end_date, 3)
        ]
        data = self._merge(path, frames, ["ts_code", "trade_date"], ["trade_date"])
        if data.empty:
            raise RuntimeError(f"外汇 {ts_code} 未返回任何数据")
        self._atomic_parquet(data, path)
        return path

    def download_gold(self, ts_code: str, start_date: str, end_date: str) -> Path:
        path = self.gold_file
        fetch_start = self._incremental_start(path, "trade_date", start_date)
        frames = [
            self._call(
                "sge_daily", ts_code=ts_code, start_date=left, end_date=right
            )
            for left, right in self._multi_year_ranges(fetch_start, end_date, 3)
        ]
        data = self._merge(path, frames, ["ts_code", "trade_date"], ["trade_date"])
        if data.empty:
            raise RuntimeError(f"黄金 {ts_code} 未返回任何数据")
        self._atomic_parquet(data, path)
        return path

    @staticmethod
    def _month_end(month: str) -> str:
        value = str(month)
        year, mon = int(value[:4]), int(value[4:6])
        if mon == 12:
            next_month = dt.date(year + 1, 1, 1)
        else:
            next_month = dt.date(year, mon + 1, 1)
        return (next_month - dt.timedelta(days=1)).strftime(DATE_FMT)

    @staticmethod
    def _map_to_trade_date(
        base_date: str, trade_dates: list[str], strictly_after: bool
    ) -> str:
        position = (
            bisect.bisect_right(trade_dates, base_date)
            if strictly_after
            else bisect.bisect_left(trade_dates, base_date)
        )
        return trade_dates[position] if position < len(trade_dates) else ""

    def _daily_availability_rows(
        self,
        dataset: str,
        path: Path,
        date_col: str,
        trade_dates: list[str],
        lag_trade_days: int = 0,
        strictly_after: bool = False,
        note: str = "",
    ) -> list[dict]:
        if not path.exists():
            return []
        dates = sorted(pd.read_parquet(path, columns=[date_col])[date_col].astype(str).unique())
        rows = []
        for data_date in dates:
            if strictly_after:
                available = self._map_to_trade_date(data_date, trade_dates, True)
                method = "next_china_trade_date"
                lag_value = 1
            else:
                position = bisect.bisect_left(trade_dates, data_date)
                target = position + lag_trade_days
                available = trade_dates[target] if target < len(trade_dates) else ""
                method = "trade_day_lag"
                lag_value = lag_trade_days
            if available:
                rows.append(
                    {
                        "dataset": dataset,
                        "period_date": data_date,
                        "data_date": data_date,
                        "publish_date": "",
                        "available_date": available,
                        "availability_method": method,
                        "lag_value": lag_value,
                        "note": note,
                    }
                )
        return rows

    def _monthly_availability_rows(
        self,
        dataset: str,
        path: Path,
        trade_dates: list[str],
        note: str,
    ) -> list[dict]:
        """月频数据统一从统计期下一自然月的最后一个中国交易日使用。"""
        if not path.exists():
            return []
        months = sorted(pd.read_parquet(path, columns=["month"])["month"].astype(str).unique())
        rows = []
        for month in months:
            period_date = self._month_end(month)
            year, mon = int(month[:4]), int(month[4:6])
            if mon == 12:
                next_month = "{:04d}01".format(year + 1)
            else:
                next_month = "{:04d}{:02d}".format(year, mon + 1)
            next_month_end = self._month_end(next_month)
            candidates = [date for date in trade_dates if date.startswith(next_month)]
            # 交易日历若只覆盖到月中，不能把当前最后一行误认为月末交易日。
            calendar_complete = bool(trade_dates) and trade_dates[-1] >= next_month_end
            available = candidates[-1] if candidates and calendar_complete else ""
            if available:
                rows.append(
                    {
                        "dataset": dataset,
                        "period_date": period_date,
                        "data_date": month,
                        "publish_date": "",
                        "available_date": available,
                        "availability_method": "next_month_last_trade_date",
                        "lag_value": 0,
                        "note": note,
                    }
                )
        return rows

    def build_data_availability(self, start_date: str, end_date: str) -> Path:
        """按真实观测生成每条数据最早允许进入模型的中国交易日。"""
        calendar = pd.read_parquet(self.trade_cal_file, columns=["cal_date"])
        calendar_end = str(calendar["cal_date"].astype(str).max())
        trade_dates = self.open_trade_dates(start_date, calendar_end)
        rules = self.config.get("availability", {})
        rows: list[dict] = []

        domestic_rule = rules.get("domestic_index", {})
        for item in self.configured_domestic_indices():
            rows.extend(
                self._daily_availability_rows(
                    f"index:{item['ts_code']}",
                    self.domestic_index_path(item["ts_code"]),
                    "trade_date",
                    trade_dates,
                    lag_trade_days=int(domestic_rule.get("lag_trade_days", 0)),
                    note=str(domestic_rule.get("note", "")),
                )
            )

        global_rule = rules.get("global_index", {})
        for item in self.configured_global_indices():
            rows.extend(
                self._daily_availability_rows(
                    f"global_index:{item['ts_code']}",
                    self.global_index_path(item["ts_code"]),
                    "trade_date",
                    trade_dates,
                    strictly_after=True,
                    note=str(global_rule.get("note", "")),
                )
            )

        for dataset, path, date_col in (
            ("shibor", self.shibor_file, "date"),
            ("margin", self.margin_file, "trade_date"),
        ):
            rule = rules.get(dataset, {})
            rows.extend(
                self._daily_availability_rows(
                    dataset,
                    path,
                    date_col,
                    trade_dates,
                    lag_trade_days=int(rule.get("lag_trade_days", 0)),
                    note=str(rule.get("note", "")),
                )
            )

        for dataset, path in (("pmi", self.pmi_file), ("cpi", self.cpi_file)):
            rule = rules.get(dataset, {})
            rows.extend(
                self._monthly_availability_rows(
                    dataset,
                    path,
                    trade_dates,
                    str(rule.get("note", "")),
                )
            )

        for dataset, path in (("fx", self.fx_file), ("gold", self.gold_file)):
            rule = rules.get(dataset, {})
            rows.extend(
                self._daily_availability_rows(
                    dataset,
                    path,
                    "trade_date",
                    trade_dates,
                    strictly_after=True,
                    note=str(rule.get("note", "")),
                )
            )

        columns = [
            "dataset",
            "period_date",
            "data_date",
            "publish_date",
            "available_date",
            "availability_method",
            "lag_value",
            "note",
        ]
        data = pd.DataFrame(rows, columns=columns)
        if not data.empty:
            data = data.drop_duplicates(["dataset", "data_date"], keep="last")
            data = data.sort_values(["dataset", "period_date"]).reset_index(drop=True)
        self._atomic_parquet(data, self.availability_file)
        return self.availability_file

    # 兼容旧调用名称。
    def build_availability_calendar(self, start_date: str, end_date: str) -> Path:
        return self.build_data_availability(start_date, end_date)

    def download_all(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        start = start_date or str(self.config["project"]["start_date"])
        end = end_date or self.default_end_date()
        started = dt.datetime.now().isoformat(timespec="seconds")
        outputs: dict[str, str] = {}
        total_steps = 14
        step = 0

        # 交易所日历是预先公布的。取到当年年末，才能在月末最后一个交易日
        # 当天准确识别“本月最后一个交易日”，不会造成事后回填。
        calendar_end = f"{end[:4]}1231"
        path = self.download_trade_calendar(start, calendar_end)
        outputs["trade_calendar"] = str(path)
        step += 1
        self._progress(step, total_steps, "交易日历", path)
        for item in self.configured_domestic_indices():
            item_start = str(item.get("start_date", start))
            path = self.download_index(item["ts_code"], item_start, end)
            outputs[f"index:{item['ts_code']}"] = str(path)
            step += 1
            self._progress(step, total_steps, item["name"], path)
        for item in self.configured_global_indices():
            item_start = str(item.get("start_date", start))
            path = self.download_global_index(item["ts_code"], item_start, end)
            outputs[f"global_index:{item['ts_code']}"] = str(path)
            step += 1
            self._progress(step, total_steps, item["name"], path)

        cfg = self.layer2_config
        if cfg.get("shibor", {}).get("enabled", True):
            path = self.download_shibor(str(cfg["shibor"].get("start_date", start)), end)
            outputs["shibor"] = str(path)
            step += 1
            self._progress(step, total_steps, "1个月Shibor", path)
        if cfg.get("margin", {}).get("enabled", True):
            path = self.download_margin(str(cfg["margin"].get("start_date", start)), end)
            outputs["margin"] = str(path)
            step += 1
            self._progress(step, total_steps, "沪深融资余额", path)
        if cfg.get("pmi", {}).get("enabled", True):
            path = self.download_monthly_macro("cn_pmi", self.pmi_file)
            outputs["pmi"] = str(path)
            step += 1
            self._progress(step, total_steps, "制造业PMI", path)
        if cfg.get("cpi", {}).get("enabled", True):
            path = self.download_monthly_macro("cn_cpi", self.cpi_file)
            outputs["cpi"] = str(path)
            step += 1
            self._progress(step, total_steps, "全国CPI", path)
        if cfg.get("fx", {}).get("enabled", True):
            path = self.download_fx(
                cfg["fx"]["ts_code"], str(cfg["fx"].get("start_date", start)), end
            )
            outputs["fx"] = str(path)
            step += 1
            self._progress(step, total_steps, "离岸人民币", path)
        if cfg.get("gold", {}).get("enabled", True):
            path = self.download_gold(
                cfg["gold"]["ts_code"],
                str(cfg["gold"].get("start_date", start)),
                end,
            )
            outputs["gold"] = str(path)
            step += 1
            self._progress(step, total_steps, "上海金", path)

        path = self.build_data_availability(start, end)
        outputs["data_availability"] = str(path)
        step += 1
        self._progress(step, total_steps, "数据可用日期表", path)
        manifest = {
            "started_at": started,
            "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
            "requested_start_date": start,
            "requested_end_date": end,
            "target_index": self.target_index,
            "outputs": outputs,
        }
        self._atomic_json(manifest, self.log_dir / "download_manifest.json")
        return manifest

    @staticmethod
    def _price_check(data: pd.DataFrame, keys: list[str]) -> tuple[list[str], int, int]:
        required = {"trade_date", "open", "high", "low", "close", *keys}
        missing = sorted(required - set(data.columns))
        duplicate = int(data.duplicated(keys).sum()) if not missing else 0
        bad_price = 0
        if not missing:
            bad_price = int(
                (data[["open", "high", "low", "close"]].min(axis=1) <= 0).sum()
            )
        return missing, duplicate, bad_price

    def validate(self, end_date: str | None = None) -> dict:
        requested_end = end_date or self.default_end_date()
        checks: list[dict] = []

        def add(category: str, item: str, status: str, detail: str) -> None:
            checks.append(
                {"category": category, "item": item, "status": status, "detail": detail}
            )

        latest_open = ""
        cal = pd.DataFrame()
        if not self.trade_cal_file.exists():
            add("calendar", "trade_cal", "FAIL", "文件不存在")
        else:
            cal = pd.read_parquet(self.trade_cal_file)
            duplicate = int(cal.duplicated(["exchange", "cal_date"]).sum())
            eligible = cal[
                (cal["is_open"].astype(int) == 1)
                & (cal["cal_date"].astype(str) <= requested_end)
            ]
            latest_open = str(eligible["cal_date"].max()) if not eligible.empty else ""
            add(
                "calendar",
                "trade_cal",
                "PASS" if duplicate == 0 and latest_open else "FAIL",
                f"rows={len(cal)} duplicate={duplicate} latest_open={latest_open}",
            )

        for item in self.configured_domestic_indices():
            code = item["ts_code"]
            path = self.domestic_index_path(code)
            if not path.exists():
                add("layer1/3", code, "FAIL", "文件不存在")
                continue
            data = pd.read_parquet(path)
            missing, duplicate, bad_price = self._price_check(
                data, ["ts_code", "trade_date"]
            )
            latest = str(data["trade_date"].max()) if not data.empty else ""
            status = (
                "PASS"
                if not missing and not duplicate and not bad_price and latest == latest_open
                else "FAIL"
            )
            add(
                "layer1/3",
                code,
                status,
                f"rows={len(data)} latest={latest} missing_cols={missing} "
                f"duplicate={duplicate} bad_price={bad_price}",
            )

        for item in self.configured_global_indices():
            code = item["ts_code"]
            path = self.global_index_path(code)
            if not path.exists():
                add("layer3", code, "FAIL", "文件不存在")
                continue
            data = pd.read_parquet(path)
            missing, duplicate, bad_price = self._price_check(
                data, ["ts_code", "trade_date"]
            )
            latest = str(data["trade_date"].max()) if not data.empty else ""
            stale = (dt.datetime.strptime(latest_open, DATE_FMT) - dt.datetime.strptime(latest, DATE_FMT)).days if latest and latest_open else 999
            status = "PASS" if not missing and not duplicate and not bad_price and stale <= 10 else "FAIL"
            add("layer3", code, status, f"rows={len(data)} latest={latest} stale_days={stale} missing_cols={missing} duplicate={duplicate} bad_price={bad_price}")

        daily_specs = [
            ("shibor", self.shibor_file, "date", ["1m"]),
            ("margin", self.margin_file, "trade_date", ["exchange_id", "rzye"]),
        ]
        for name, path, date_col, required_cols in daily_specs:
            if not path.exists():
                add("layer2", name, "FAIL", "文件不存在")
                continue
            data = pd.read_parquet(path)
            missing = sorted(set(required_cols + [date_col]) - set(data.columns))
            latest = str(data[date_col].max()) if not data.empty and not missing else ""
            stale = (dt.datetime.strptime(latest_open, DATE_FMT) - dt.datetime.strptime(latest, DATE_FMT)).days if latest and latest_open else 999
            status = "PASS" if not missing and stale <= 10 else "FAIL"
            add("layer2", name, status, f"rows={len(data)} latest={latest} stale_days={stale} missing_cols={missing}")

        monthly_specs = [
            ("pmi", self.pmi_file, self.layer2_config["pmi"]["field"], (0, 100)),
            ("cpi", self.cpi_file, self.layer2_config["cpi"]["field"], (-20, 30)),
        ]
        for name, path, field, limits in monthly_specs:
            if not path.exists():
                add("layer2", name, "FAIL", "文件不存在")
                continue
            data = pd.read_parquet(path)
            missing = sorted({"month", field} - set(data.columns))
            duplicate = int(data.duplicated(["month"]).sum()) if not missing else 0
            invalid = int((~pd.to_numeric(data[field], errors="coerce").between(*limits)).sum()) if not missing else 0
            status = "PASS" if not missing and not duplicate and not invalid else "FAIL"
            latest = str(data["month"].max()) if "month" in data and len(data) else ""
            add("layer2", name, status, f"rows={len(data)} latest={latest} field={field} missing_cols={missing} duplicate={duplicate} invalid={invalid}")

        market_specs = [
            ("fx", self.fx_file, ["bid_open", "bid_close", "bid_high", "bid_low"]),
            ("gold", self.gold_file, ["open", "close", "high", "low"]),
        ]
        for name, path, price_cols in market_specs:
            if not path.exists():
                add("layer2", name, "FAIL", "文件不存在")
                continue
            data = pd.read_parquet(path)
            missing = sorted({"ts_code", "trade_date", *price_cols} - set(data.columns))
            duplicate = int(data.duplicated(["ts_code", "trade_date"]).sum()) if not missing else 0
            bad_price = int((data[price_cols].min(axis=1) <= 0).sum()) if not missing else 0
            latest = str(data["trade_date"].max()) if not data.empty and not missing else ""
            stale = (dt.datetime.strptime(latest_open, DATE_FMT) - dt.datetime.strptime(latest, DATE_FMT)).days if latest and latest_open else 999
            status = "PASS" if not missing and not duplicate and not bad_price and stale <= 10 else "FAIL"
            add("layer2", name, status, f"rows={len(data)} latest={latest} stale_days={stale} missing_cols={missing} duplicate={duplicate} bad_price={bad_price}")

        if not self.availability_file.exists():
            add("pit", "data_availability", "FAIL", "文件不存在")
        else:
            data = pd.read_parquet(self.availability_file)
            required = {
                "dataset", "period_date", "data_date", "available_date",
                "availability_method", "lag_value",
            }
            missing = sorted(required - set(data.columns))
            duplicate = int(data.duplicated(["dataset", "data_date"]).sum()) if not missing else 0
            reversed_dates = int((data["available_date"].astype(str) < data["period_date"].astype(str)).sum()) if not missing else 0
            expected = {
                *(f"index:{x['ts_code']}" for x in self.configured_domestic_indices()),
                *(f"global_index:{x['ts_code']}" for x in self.configured_global_indices()),
                "shibor", "margin", "pmi", "cpi", "fx", "gold",
            }
            actual = set(data["dataset"].astype(str)) if "dataset" in data else set()
            missing_datasets = sorted(expected - actual)
            margin = data[data.get("dataset", "") == "margin"] if not missing else pd.DataFrame()
            bad_margin_lag = int((margin["available_date"] <= margin["period_date"]).sum()) if not margin.empty else 0
            macro = data[data.get("dataset", "").isin(["pmi", "cpi"])] if not missing else pd.DataFrame()
            bad_macro_rule = 0
            if not macro.empty and not cal.empty:
                open_cal = cal[cal["is_open"].astype(int) == 1].copy()
                open_cal["month"] = open_cal["cal_date"].astype(str).str[:6]
                last_trade_by_month = open_cal.groupby("month")["cal_date"].max().astype(str)
                next_month = (
                    pd.to_datetime(macro["data_date"].astype(str) + "01", format="%Y%m%d")
                    + pd.offsets.MonthBegin(1)
                ).dt.strftime("%Y%m")
                expected_macro_date = next_month.map(last_trade_by_month)
                bad_macro_rule = int(
                    (
                        macro["availability_method"].astype(str).ne("next_month_last_trade_date")
                        | macro["available_date"].astype(str).ne(expected_macro_date.astype(str))
                    ).sum()
                )
            status = "PASS" if not missing and not duplicate and not reversed_dates and not missing_datasets and not bad_margin_lag and not bad_macro_rule else "FAIL"
            add("pit", "data_availability", status, f"rows={len(data)} datasets={len(actual)} missing_cols={missing} duplicate={duplicate} reversed_dates={reversed_dates} missing_datasets={missing_datasets} bad_margin_lag={bad_margin_lag} bad_macro_rule={bad_macro_rule}")

        summary = {
            "pass": sum(c["status"] == "PASS" for c in checks),
            "warn": sum(c["status"] == "WARN" for c in checks),
            "fail": sum(c["status"] == "FAIL" for c in checks),
        }
        report = {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "requested_end_date": requested_end,
            "latest_open_date": latest_open,
            "target_index": self.target_index,
            "summary": summary,
            "checks": checks,
        }
        self._atomic_json(report, self.log_dir / "validation_report.json")
        return report
