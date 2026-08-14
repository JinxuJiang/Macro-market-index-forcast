"""逐季度统一执行：寻参 -> 全历史重训 -> 当季预测 -> 汇总评价。"""

from __future__ import annotations

import json
import os
import traceback
import hashlib
import ctypes
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from pipeline.dataset import ModelDataset, config_hash, regression_metrics
from pipeline.walk_forward import ExpandingQuarterlySplitter, QuarterlyFold


class QuarterlyRunner:
    def __init__(self, dataset: ModelDataset, splitter: ExpandingQuarterlySplitter, trainer, common_config: dict):
        self.dataset = dataset
        self.splitter = splitter
        self.trainer = trainer
        self.common_config = common_config
        self.experiment_dir = Path(trainer.experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        # 预测平滑是训练完成后的确定性后处理，不应让已有冻结模型失效。
        training_protocol = dict(common_config)
        training_protocol.pop("prediction_smoothing", None)
        self.protocol_hash = config_hash(training_protocol, trainer.model_config)

    def _state_path(self, fold: QuarterlyFold) -> Path:
        return self.trainer.fold_dir(fold) / "manifest.json"

    def _write_state(self, fold: QuarterlyFold, status: str, **extra) -> None:
        path = self._state_path(fold)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fold_id": fold.fold_id,
            "model_period": fold.model_period,
            "model_name": self.trainer.model_name,
            "protocol_hash": self.protocol_hash,
            "fold_data_hash": self._fold_data_hash(fold),
            "status": status,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "as_of_date": fold.as_of_date.strftime("%Y-%m-%d"),
            "n_inner_train": len(fold.inner_train_dates),
            "n_purge": len(fold.purge_dates),
            "n_inner_valid": len(fold.inner_valid_dates),
            "n_final_train": len(fold.final_train_dates),
            "n_prediction": len(fold.prediction_dates),
            **extra,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fold_data_hash(self, fold: QuarterlyFold) -> str:
        rows = self.dataset.rows_for_dates(fold.final_train_dates)
        columns = ["signal_date", "entry_date", "exit_date", *self.dataset.feature_names, self.dataset.target_name]
        hashed = pd.util.hash_pandas_object(rows.loc[:, columns], index=False).to_numpy().tobytes()
        return hashlib.sha256(hashed).hexdigest()[:16]

    def _validate_existing_state(self, fold: QuarterlyFold) -> None:
        path = self._state_path(fold)
        if not path.exists():
            return
        state = json.loads(path.read_text(encoding="utf-8"))
        stored_protocol = state.get("protocol_hash")
        stored_data = state.get("fold_data_hash")
        if stored_protocol and stored_protocol != self.protocol_hash:
            raise RuntimeError(
                f"{fold.model_period}已有产物的配置哈希为{stored_protocol}，当前为{self.protocol_hash}；"
                "拒绝静默复用，请使用新的实验目录"
            )
        current_data = self._fold_data_hash(fold)
        if stored_data and stored_data != current_data:
            raise RuntimeError(
                f"{fold.model_period}训练历史数据已变化（{stored_data} -> {current_data}），拒绝改写冻结产物"
            )

    @staticmethod
    def _pid_is_running(pid: int) -> bool:
        if os.name == "nt":
            process_query_limited_information = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @contextmanager
    def _fold_lock(self, fold: QuarterlyFold):
        lock_path = self.trainer.fold_dir(fold) / ".run.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if lock_path.exists():
            try:
                owner_pid = int(lock_path.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                owner_pid = -1
            if owner_pid > 0 and self._pid_is_running(owner_pid):
                raise RuntimeError(f"{fold.model_period}正在由进程{owner_pid}运行，拒绝重复启动")
            lock_path.unlink(missing_ok=True)
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
        finally:
            os.close(descriptor)
        try:
            yield
        finally:
            lock_path.unlink(missing_ok=True)

    def run(
        self,
        start_quarter: Optional[str] = None,
        end_quarter: Optional[str] = None,
        max_folds: Optional[int] = None,
    ) -> pd.DataFrame:
        folds = list(self.splitter)
        if start_quarter:
            folds = [fold for fold in folds if fold.model_period >= start_quarter]
        if end_quarter:
            folds = [fold for fold in folds if fold.model_period <= end_quarter]
        if max_folds is not None:
            folds = folds[: int(max_folds)]

        outputs = []
        for index, fold in enumerate(folds, start=1):
            with self._fold_lock(fold):
                self._validate_existing_state(fold)
                fold_dir = self.trainer.fold_dir(fold)
                tuning_action = "复用已有寻参结果" if (fold_dir / "selected_params.json").exists() else "寻参"
                prediction_path = fold_dir / "predictions.parquet"
                prediction_action = "全历史重训与预测"
                if prediction_path.exists():
                    existing_dates = set(pd.to_datetime(pd.read_parquet(prediction_path, columns=["signal_date"])["signal_date"]))
                    prediction_action = (
                        "复用已有预测"
                        if set(fold.prediction_dates).issubset(existing_dates)
                        else "使用冻结模型追加最新预测"
                    )
                print(
                    f"[{self.trainer.model_name}] {index}/{len(folds)} {fold.model_period}: {tuning_action}",
                    flush=True,
                )
                try:
                    tuning = self.trainer.tune(fold)
                    self._write_state(fold, "tuned", selected_parameters=tuning)
                    print(f"[{self.trainer.model_name}] {fold.model_period}: {prediction_action}", flush=True)
                    prediction = self.trainer.refit_and_predict(fold, tuning)
                    outputs.append(prediction)
                    self._write_state(fold, "completed", selected_parameters=tuning)
                except Exception as exc:
                    self._write_state(fold, "failed", error=str(exc), traceback=traceback.format_exc())
                    raise

        return self.aggregate()

    def aggregate(self) -> pd.DataFrame:
        paths = sorted((self.experiment_dir / "folds").glob("*/predictions.parquet"))
        if not paths:
            return pd.DataFrame()
        combined = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        combined["signal_date"] = pd.to_datetime(combined["signal_date"])
        if combined["signal_date"].duplicated().any():
            duplicates = combined.loc[combined["signal_date"].duplicated(), "signal_date"].head().tolist()
            raise ValueError(f"正式季度预测日期重复: {duplicates}")

        actual_map = self.dataset.frame.set_index("signal_date")[self.dataset.target_name]
        combined["actual"] = combined["signal_date"].map(actual_map)
        combined = combined.sort_values("signal_date").reset_index(drop=True)
        combined.to_parquet(self.experiment_dir / "oos_predictions.parquet", index=False)
        self._write_evaluation(combined)
        smoothing = self.common_config.get("prediction_smoothing", {})
        if bool(smoothing.get("enabled", False)):
            smoothed = self._smooth_predictions(combined, smoothing)
            smoothed.to_parquet(self.experiment_dir / "oos_predictions_smoothed.parquet", index=False)
            self._write_evaluation(smoothed, prefix="smoothed_")
        return combined

    @staticmethod
    def _smooth_predictions(predictions: pd.DataFrame, smoothing: dict) -> pd.DataFrame:
        method = str(smoothing.get("method", "")).lower()
        if method != "ewm":
            raise ValueError(f"不支持的预测平滑方法: {method}")
        halflife = float(smoothing["halflife_days"])
        if halflife <= 0:
            raise ValueError("prediction_smoothing.halflife_days必须大于0")
        result = predictions.copy().sort_values("signal_date").reset_index(drop=True)
        result["prediction_raw"] = result["prediction"].astype(float)
        result["prediction"] = result["prediction_raw"].ewm(
            halflife=halflife,
            adjust=bool(smoothing.get("adjust", False)),
            min_periods=int(smoothing.get("min_periods", 1)),
        ).mean()
        result["smoothing_method"] = method
        result["smoothing_halflife_days"] = halflife
        return result

    def _write_evaluation(self, predictions: pd.DataFrame, prefix: str = "") -> None:
        delta = float(self.common_config["evaluation"]["huber_delta"])
        valid = predictions.dropna(subset=["actual", "prediction"]).copy()
        overall = regression_metrics(valid, delta)
        yearly_rows = []
        for year, group in valid.groupby(valid["signal_date"].dt.year):
            yearly_rows.append({"year": int(year), **regression_metrics(group, delta)})
        quarterly_rows = []
        for period, group in valid.groupby(valid["signal_date"].dt.to_period("Q")):
            quarterly_rows.append({"quarter": str(period), **regression_metrics(group, delta)})
        pd.DataFrame(yearly_rows).to_parquet(self.experiment_dir / f"{prefix}yearly_metrics.parquet", index=False)
        pd.DataFrame(quarterly_rows).to_parquet(self.experiment_dir / f"{prefix}quarterly_metrics.parquet", index=False)
        report = {
            "model_name": self.trainer.model_name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "prediction_start": predictions["signal_date"].min().strftime("%Y-%m-%d"),
            "prediction_end": predictions["signal_date"].max().strftime("%Y-%m-%d"),
            "labelled_prediction_end": valid["signal_date"].max().strftime("%Y-%m-%d") if len(valid) else None,
            "overall": overall,
        }
        if prefix:
            report["prediction_variant"] = "ewm_smoothed"
            report["smoothing"] = self.common_config["prediction_smoothing"]
        (self.experiment_dir / f"{prefix}evaluation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
        )
