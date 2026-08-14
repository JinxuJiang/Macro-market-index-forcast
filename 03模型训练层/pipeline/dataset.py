"""共享监督学习数据集、预处理器与评价指标。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score


LAYER_DIR = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_layer_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (LAYER_DIR / path).resolve()


def config_hash(*configs: dict) -> str:
    payload = json.dumps(configs, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def huber_loss(y_true: Sequence[float], y_pred: Sequence[float], delta: float) -> float:
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    absolute = np.abs(error)
    loss = np.where(absolute <= delta, 0.5 * error**2, delta * (absolute - 0.5 * delta))
    return float(np.mean(loss))


def direction_accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.sign(actual) == np.sign(predicted)))


def regression_metrics(frame: pd.DataFrame, delta: float) -> Dict[str, float]:
    valid = frame.dropna(subset=["actual", "prediction"])
    if len(valid) < 2:
        return {"n": int(len(valid)), "r2": np.nan, "huber_loss": np.nan, "direction_accuracy": np.nan}
    actual = valid["actual"].to_numpy(dtype=float)
    prediction = valid["prediction"].to_numpy(dtype=float)
    return {
        "n": int(len(valid)),
        "r2": float(r2_score(actual, prediction)),
        "huber_loss": huber_loss(actual, prediction, delta),
        "direction_accuracy": direction_accuracy(actual, prediction),
    }


@dataclass
class FeaturePreprocessor:
    """只用传入训练样本拟合的中位数、缩尾与标准化器。"""

    feature_names: List[str]
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    standardize: bool = True
    medians_: Optional[pd.Series] = None
    lower_: Optional[pd.Series] = None
    upper_: Optional[pd.Series] = None
    means_: Optional[pd.Series] = None
    scales_: Optional[pd.Series] = None

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        values = frame.loc[:, self.feature_names].astype(float)
        self.medians_ = values.median(axis=0)
        if self.medians_.isna().any():
            missing = self.medians_[self.medians_.isna()].index.tolist()
            raise ValueError(f"训练集中整列缺失，无法拟合预处理器: {missing}")
        filled = values.fillna(self.medians_)
        self.lower_ = filled.quantile(self.lower_quantile)
        self.upper_ = filled.quantile(self.upper_quantile)
        clipped = filled.clip(lower=self.lower_, upper=self.upper_, axis=1)
        self.means_ = clipped.mean(axis=0)
        scales = clipped.std(axis=0, ddof=0)
        self.scales_ = scales.mask(scales <= 1e-12, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("预处理器尚未拟合")
        result = frame.loc[:, self.feature_names].astype(float).fillna(self.medians_)
        result = result.clip(lower=self.lower_, upper=self.upper_, axis=1)
        if self.standardize:
            result = (result - self.means_) / self.scales_
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("预处理结果包含NaN或无穷值")
        result.index = frame.index
        return result

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


class ModelDataset:
    """构造并持有日频特征、标签和标签可用日期。"""

    ID_COLUMNS = ["signal_date", "entry_date", "exit_date"]

    def __init__(self, config: dict):
        self.config = config
        self.target_name = config["label"]["name"]
        self.model_start = pd.Timestamp(config["data"]["model_sample_start"])
        self.feature_table_path = resolve_layer_path(config["data"]["feature_table"])
        self.target_price_path = resolve_layer_path(config["data"]["target_price"])
        self.data_dir = resolve_layer_path(config["output"]["data_dir"])
        self.processed_dir = self.data_dir / "processed"
        self.logs_dir = self.data_dir / "logs"
        self.frame: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []

    @staticmethod
    def _parse_trade_date(series: pd.Series) -> pd.Series:
        return pd.to_datetime(series.astype(str), format="%Y%m%d", errors="raise")

    def build(self, write: bool = True) -> pd.DataFrame:
        features = pd.read_parquet(self.feature_table_path).copy()
        prices = pd.read_parquet(self.target_price_path).copy()
        if "trade_date" not in features or "trade_date" not in prices:
            raise ValueError("特征表和目标行情都必须包含trade_date")

        features["signal_date"] = self._parse_trade_date(features["trade_date"])
        features = features.drop(columns=["trade_date"]).sort_values("signal_date")
        prices["signal_date"] = self._parse_trade_date(prices["trade_date"])
        prices = prices.sort_values("signal_date").reset_index(drop=True)

        if features["signal_date"].duplicated().any() or prices["signal_date"].duplicated().any():
            raise ValueError("输入数据存在重复交易日")
        if not features["signal_date"].reset_index(drop=True).equals(prices["signal_date"]):
            raise ValueError("特征表与中证1000行情交易日不完全一致")

        entry_offset = int(self.config["label"]["entry_offset"])
        exit_offset = int(self.config["label"]["exit_offset"])
        open_price = pd.to_numeric(prices["open"], errors="coerce")
        if open_price.isna().any() or (open_price <= 0).any():
            raise ValueError("中证1000 open包含缺失或非正值")

        frame = features.copy()
        frame.insert(1, "entry_date", prices["signal_date"].shift(-entry_offset))
        frame.insert(2, "exit_date", prices["signal_date"].shift(-exit_offset))
        frame[self.target_name] = open_price.shift(-exit_offset) / open_price.shift(-entry_offset) - 1.0
        frame = frame.reset_index(drop=True)

        self.feature_names = [
            column for column in frame.columns if column not in self.ID_COLUMNS + [self.target_name]
        ]
        self.frame = frame

        if write:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(self.processed_dir / "dataset_with_label.parquet", index=False)
            manifest = self.manifest()
            with (self.logs_dir / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
        return frame

    def load_or_build(self, rebuild: bool = False) -> pd.DataFrame:
        output = self.processed_dir / "dataset_with_label.parquet"
        if rebuild or not output.exists():
            return self.build(write=True)
        frame = pd.read_parquet(output)
        for column in self.ID_COLUMNS:
            frame[column] = pd.to_datetime(frame[column])
        self.feature_names = [
            column for column in frame.columns if column not in self.ID_COLUMNS + [self.target_name]
        ]
        self.frame = frame
        return frame

    def model_frame(self) -> pd.DataFrame:
        if self.frame is None:
            raise RuntimeError("数据集尚未加载")
        return self.frame.loc[self.frame["signal_date"] >= self.model_start].copy()

    def make_preprocessor(self) -> FeaturePreprocessor:
        cfg = self.config["preprocessing"]
        return FeaturePreprocessor(
            feature_names=list(self.feature_names),
            lower_quantile=float(cfg["winsor_lower"]),
            upper_quantile=float(cfg["winsor_upper"]),
            standardize=bool(cfg["standardize"]),
        )

    def rows_for_dates(self, dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
        if self.frame is None:
            raise RuntimeError("数据集尚未加载")
        wanted = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        indexed = self.frame.set_index("signal_date", drop=False)
        missing = wanted.difference(indexed.index)
        if len(missing):
            raise KeyError(f"数据集缺少交易日: {missing[:5].tolist()}")
        return indexed.loc[wanted].copy()

    def manifest(self) -> dict:
        if self.frame is None:
            raise RuntimeError("数据集尚未构建")
        complete = self.frame[self.target_name].notna()
        model_rows = self.frame["signal_date"] >= self.model_start
        return {
            "source_feature_table": str(self.feature_table_path),
            "source_target_price": str(self.target_price_path),
            "rows": int(len(self.frame)),
            "feature_count": int(len(self.feature_names)),
            "date_start": self.frame["signal_date"].min().strftime("%Y-%m-%d"),
            "date_end": self.frame["signal_date"].max().strftime("%Y-%m-%d"),
            "model_sample_start": self.model_start.strftime("%Y-%m-%d"),
            "model_rows": int(model_rows.sum()),
            "label_name": self.target_name,
            "label_formula": "open[T+21] / open[T+1] - 1",
            "last_complete_label_date": self.frame.loc[complete, "signal_date"].max().strftime("%Y-%m-%d"),
        }
