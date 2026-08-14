"""截面仓库数据的只读适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


class CrossSectionalDataAdapter:
    """只读访问截面仓库的全 A 行情与市值数据。

    本类不在截面仓库写入任何文件。市场宽度特征由后续特征层通过该接口读取，
    避免在两个仓库重复保存全 A 日行情。
    """

    MARKET_FIELDS = ("open", "high", "low", "close", "volume", "amount")

    def __init__(
        self,
        repository: str | Path,
        market_data_dir: str | Path,
        daily_basic_file: str | Path,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.market_data_dir = self.repository / Path(market_data_dir)
        self.daily_basic_file = self.repository / Path(daily_basic_file)

    def market_field_path(self, field: str) -> Path:
        if field not in self.MARKET_FIELDS:
            raise ValueError(f"不支持的行情字段: {field}")
        return self.market_data_dir / f"{field}.parquet"

    def validate_paths(self) -> dict:
        fields = {
            field: self.market_field_path(field).exists()
            for field in self.MARKET_FIELDS
        }
        return {
            "repository": str(self.repository),
            "repository_exists": self.repository.exists(),
            "market_fields": fields,
            "daily_basic_file": str(self.daily_basic_file),
            "daily_basic_exists": self.daily_basic_file.exists(),
            "ok": all(fields.values()) and self.daily_basic_file.exists(),
        }

    def read_market_field(
        self,
        field: str,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        path = self.market_field_path(field)
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(path, columns=list(columns) if columns else None)

    def read_daily_basic(
        self,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        if not self.daily_basic_file.exists():
            raise FileNotFoundError(self.daily_basic_file)
        return pd.read_parquet(
            self.daily_basic_file,
            columns=list(columns) if columns else None,
        )

