"""单个季度fold的CNN-GRU Optuna寻参、双种子重训与预测。"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd

from models.cnn_gru.model import build_cnn_gru
from pipeline.dataset import ModelDataset
from pipeline.walk_forward import QuarterlyFold


class CnnGruTrainer:
    model_name = "cnn_gru"

    def __init__(self, dataset: ModelDataset, model_config: dict, common_config: dict, experiment_dir: Path):
        self.dataset = dataset
        self.model_config = model_config
        self.common_config = common_config
        self.experiment_dir = Path(experiment_dir)
        self.sequence_length = int(model_config["sequence_length"])
        self.delta = float(model_config["fixed"]["huber_delta"])
        self.seeds = [int(value) for value in model_config["seeds"]]

    def fold_dir(self, fold: QuarterlyFold) -> Path:
        return self.experiment_dir / "folds" / fold.model_period

    @staticmethod
    def _set_seed(seed: int) -> None:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)

    @staticmethod
    def _clear_session() -> None:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        gc.collect()

    def _processed_context(self, fit_rows: pd.DataFrame, end_date: pd.Timestamp):
        model_frame = self.dataset.model_frame()
        context = model_frame[model_frame["signal_date"] <= end_date].copy()
        preprocessor = self.dataset.make_preprocessor().fit(fit_rows)
        processed = preprocessor.transform(context)
        processed.index = pd.DatetimeIndex(context["signal_date"])
        return context.set_index("signal_date", drop=False), processed, preprocessor

    def _sequences(
        self,
        raw_context: pd.DataFrame,
        processed: pd.DataFrame,
        end_dates: Iterable[pd.Timestamp],
        require_target: bool,
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        date_index = pd.DatetimeIndex(processed.index)
        positions = {date: index for index, date in enumerate(date_index)}
        x_values, y_values, kept_dates = [], [], []
        for value in pd.to_datetime(list(end_dates)):
            date = pd.Timestamp(value)
            position = positions.get(date)
            if position is None or position + 1 < self.sequence_length:
                continue
            row = raw_context.loc[date]
            target = row[self.dataset.target_name]
            if require_target and pd.isna(target):
                continue
            window = processed.iloc[position - self.sequence_length + 1 : position + 1].to_numpy(dtype=np.float32)
            if window.shape != (self.sequence_length, len(self.dataset.feature_names)):
                continue
            x_values.append(window)
            y_values.append(float(target) if pd.notna(target) else np.nan)
            kept_dates.append(date)
        if not x_values:
            raise ValueError("无法构造CNN-GRU序列样本")
        return np.stack(x_values), np.asarray(y_values, dtype=np.float32), kept_dates

    def _trial_params(self, trial) -> dict:
        space = self.model_config["search_space"]
        return {
            "cnn_filters": trial.suggest_categorical("cnn_filters", space["cnn_filters"]),
            "kernel_size": trial.suggest_categorical("kernel_size", space["kernel_size"]),
            "gru_hidden_size": trial.suggest_categorical("gru_hidden_size", space["gru_hidden_size"]),
            "dropout": trial.suggest_categorical("dropout", space["dropout"]),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                float(space["learning_rate"]["low"]),
                float(space["learning_rate"]["high"]),
                log=bool(space["learning_rate"].get("log", True)),
            ),
        }

    def _fit_with_validation(self, params, seed, x_train, y_train, x_valid, y_valid):
        import tensorflow as tf

        self._clear_session()
        self._set_seed(seed)
        model = build_cnn_gru(x_train.shape[1:], params, self.delta)
        fixed = self.model_config["fixed"]
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_valid, y_valid),
            epochs=int(fixed["max_epochs"]),
            batch_size=int(fixed["batch_size"]),
            shuffle=True,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=int(fixed["patience"]),
                    min_delta=float(fixed["min_delta"]),
                    restore_best_weights=True,
                )
            ],
        )
        validation_losses = history.history["val_loss"]
        best_index = int(np.argmin(validation_losses))
        result = float(validation_losses[best_index]), best_index + 1
        del model
        self._clear_session()
        return result

    def tune(self, fold: QuarterlyFold) -> Dict:
        import optuna
        from optuna.trial import TrialState

        output_dir = self.fold_dir(fold)
        output_dir.mkdir(parents=True, exist_ok=True)
        selected_path = output_dir / "selected_params.json"
        if selected_path.exists():
            return json.loads(selected_path.read_text(encoding="utf-8"))

        train_rows = self.dataset.rows_for_dates(fold.inner_train_dates)
        valid_rows = self.dataset.rows_for_dates(fold.inner_valid_dates)
        raw, processed, _ = self._processed_context(train_rows, fold.inner_valid_dates[-1])
        x_train, y_train, _ = self._sequences(raw, processed, fold.inner_train_dates, require_target=True)
        x_valid, y_valid, _ = self._sequences(raw, processed, fold.inner_valid_dates, require_target=True)

        storage = f"sqlite:///{(output_dir / 'optuna.db').as_posix()}"
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name=f"cnn_gru_{fold.model_period}",
            direction="minimize",
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )

        def objective(trial):
            params = self._trial_params(trial)
            losses, epochs = [], {}
            for seed in self.seeds:
                loss, best_epoch = self._fit_with_validation(params, seed, x_train, y_train, x_valid, y_valid)
                losses.append(loss)
                epochs[str(seed)] = int(best_epoch)
            trial.set_user_attr("best_epochs", epochs)
            trial.set_user_attr("seed_losses", {str(seed): float(loss) for seed, loss in zip(self.seeds, losses)})
            return float(np.mean(losses))

        completed_trials = sum(trial.state == TrialState.COMPLETE for trial in study.trials)
        remaining = max(0, int(self.model_config["trials_per_fold"]) - completed_trials)
        if remaining:
            study.optimize(objective, n_trials=remaining, gc_after_trial=True)

        best = study.best_trial
        selected = {
            "params": {key: (float(value) if isinstance(value, np.floating) else value) for key, value in best.params.items()},
            "best_epochs": best.user_attrs["best_epochs"],
            "seed_validation_losses": best.user_attrs["seed_losses"],
            "validation_huber_loss": float(best.value),
            "trial_number": int(best.number),
        }
        trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
        trials.to_parquet(output_dir / "tuning_results.parquet", index=False)
        selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        self._clear_session()
        return selected

    def refit_and_predict(self, fold: QuarterlyFold, tuning_result: Dict) -> pd.DataFrame:
        import tensorflow as tf

        output_dir = self.fold_dir(fold)
        prediction_path = output_dir / "predictions.parquet"
        if prediction_path.exists():
            existing = pd.read_parquet(prediction_path)
            existing["signal_date"] = pd.to_datetime(existing["signal_date"])
            missing_dates = sorted(set(fold.prediction_dates) - set(existing["signal_date"]))
            if not missing_dates:
                return existing
            preprocessor = joblib.load(output_dir / "preprocessor.joblib")
            model_frame = self.dataset.model_frame()
            context = model_frame[model_frame["signal_date"] <= max(missing_dates)].copy()
            processed = preprocessor.transform(context)
            processed.index = pd.DatetimeIndex(context["signal_date"])
            raw = context.set_index("signal_date", drop=False)
            x_prediction, actual, prediction_dates = self._sequences(
                raw, processed, missing_dates, require_target=False
            )
            predictions, seed_rows = [], []
            for seed in self.seeds:
                model = tf.keras.models.load_model(str(output_dir / f"model_seed_{seed}.keras"))
                predicted = model.predict(
                    x_prediction,
                    batch_size=int(self.model_config["fixed"]["batch_size"]),
                    verbose=0,
                ).reshape(-1)
                predictions.append(predicted)
                seed_rows.extend(
                    {
                        "signal_date": date,
                        "model_period": fold.model_period,
                        "seed": seed,
                        "prediction": float(value),
                        "actual": float(y) if np.isfinite(y) else np.nan,
                    }
                    for date, value, y in zip(prediction_dates, predicted, actual)
                )
                del model
                self._clear_session()
            seed_path = output_dir / "predictions_by_seed.parquet"
            old_seed = pd.read_parquet(seed_path)
            pd.concat([old_seed, pd.DataFrame(seed_rows)], ignore_index=True).sort_values(
                ["signal_date", "seed"]
            ).to_parquet(seed_path, index=False)
            appended = pd.DataFrame(
                {
                    "signal_date": prediction_dates,
                    "model_period": fold.model_period,
                    "model_name": self.model_name,
                    "prediction": np.mean(np.stack(predictions), axis=0),
                    "actual": actual,
                }
            )
            result = pd.concat([existing, appended], ignore_index=True).sort_values("signal_date")
            result.to_parquet(prediction_path, index=False)
            return result

        final_rows = self.dataset.rows_for_dates(fold.final_train_dates)
        raw, processed, preprocessor = self._processed_context(final_rows, fold.prediction_dates[-1])
        x_train, y_train, _ = self._sequences(raw, processed, fold.final_train_dates, require_target=True)
        x_prediction, actual, prediction_dates = self._sequences(
            raw, processed, fold.prediction_dates, require_target=False
        )
        params = tuning_result["params"]
        fixed = self.model_config["fixed"]
        predictions = []
        seed_rows = []
        for seed in self.seeds:
            self._clear_session()
            self._set_seed(seed)
            model = build_cnn_gru(x_train.shape[1:], params, self.delta)
            epochs = int(tuning_result["best_epochs"][str(seed)])
            model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=int(fixed["batch_size"]),
                shuffle=True,
                verbose=0,
            )
            predicted = model.predict(x_prediction, batch_size=int(fixed["batch_size"]), verbose=0).reshape(-1)
            predictions.append(predicted)
            seed_rows.extend(
                {
                    "signal_date": date,
                    "model_period": fold.model_period,
                    "seed": seed,
                    "prediction": float(value),
                    "actual": float(y) if np.isfinite(y) else np.nan,
                }
                for date, value, y in zip(prediction_dates, predicted, actual)
            )
            model.save(str(output_dir / f"model_seed_{seed}.keras"))
            del model
            self._clear_session()

        preprocessor.save(output_dir / "preprocessor.joblib")
        pd.DataFrame(seed_rows).to_parquet(output_dir / "predictions_by_seed.parquet", index=False)
        averaged = np.mean(np.stack(predictions), axis=0)
        result = pd.DataFrame(
            {
                "signal_date": prediction_dates,
                "model_period": fold.model_period,
                "model_name": self.model_name,
                "prediction": averaged,
                "actual": actual,
            }
        )
        result.to_parquet(prediction_path, index=False)
        return result
