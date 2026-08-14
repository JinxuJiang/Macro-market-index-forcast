"""单个季度fold的Ridge寻参、最终重训与预测。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from models.ridge.model import RidgeRegressor
from pipeline.dataset import ModelDataset, huber_loss
from pipeline.walk_forward import QuarterlyFold


class RidgeTrainer:
    model_name = "ridge"

    def __init__(self, dataset: ModelDataset, model_config: dict, common_config: dict, experiment_dir: Path):
        self.dataset = dataset
        self.model_config = model_config
        self.common_config = common_config
        self.experiment_dir = Path(experiment_dir)
        self.delta = float(common_config["evaluation"]["huber_delta"])

    def fold_dir(self, fold: QuarterlyFold) -> Path:
        return self.experiment_dir / "folds" / fold.model_period

    def tune(self, fold: QuarterlyFold) -> Dict:
        output_dir = self.fold_dir(fold)
        output_dir.mkdir(parents=True, exist_ok=True)
        selected_path = output_dir / "selected_params.json"
        if selected_path.exists():
            return json.loads(selected_path.read_text(encoding="utf-8"))

        train = self.dataset.rows_for_dates(fold.inner_train_dates)
        valid = self.dataset.rows_for_dates(fold.inner_valid_dates)
        preprocessor = self.dataset.make_preprocessor().fit(train)
        x_train = preprocessor.transform(train).to_numpy(dtype=float)
        x_valid = preprocessor.transform(valid).to_numpy(dtype=float)
        y_train = train[self.dataset.target_name].to_numpy(dtype=float)
        y_valid = valid[self.dataset.target_name].to_numpy(dtype=float)

        results = []
        for alpha in sorted(float(value) for value in self.model_config["alpha_candidates"]):
            model = RidgeRegressor(alpha=alpha, fit_intercept=self.model_config["fit_intercept"])
            model.fit(x_train, y_train)
            prediction = model.predict(x_valid)
            results.append({"alpha": alpha, "validation_huber_loss": huber_loss(y_valid, prediction, self.delta)})

        result_frame = pd.DataFrame(results).sort_values(
            ["validation_huber_loss", "alpha"], ascending=[True, False]
        )
        best = result_frame.iloc[0]
        selected = {
            "alpha": float(best["alpha"]),
            "validation_huber_loss": float(best["validation_huber_loss"]),
        }
        result_frame["is_best"] = result_frame["alpha"] == selected["alpha"]
        result_frame.to_parquet(output_dir / "tuning_results.parquet", index=False)
        selected_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
        return selected

    def refit_and_predict(self, fold: QuarterlyFold, tuning_result: Dict) -> pd.DataFrame:
        output_dir = self.fold_dir(fold)
        prediction_path = output_dir / "predictions.parquet"
        if prediction_path.exists():
            existing = pd.read_parquet(prediction_path)
            existing["signal_date"] = pd.to_datetime(existing["signal_date"])
            missing_dates = sorted(set(fold.prediction_dates) - set(existing["signal_date"]))
            if not missing_dates:
                return existing
            preprocessor = joblib.load(output_dir / "preprocessor.joblib")
            model = joblib.load(output_dir / "model.joblib")
            prediction_rows = self.dataset.rows_for_dates(missing_dates)
            predicted = model.predict(preprocessor.transform(prediction_rows).to_numpy(dtype=float))
            appended = pd.DataFrame(
                {
                    "signal_date": prediction_rows["signal_date"].to_numpy(),
                    "model_period": fold.model_period,
                    "model_name": self.model_name,
                    "prediction": predicted,
                    "actual": prediction_rows[self.dataset.target_name].to_numpy(dtype=float),
                }
            )
            result = pd.concat([existing, appended], ignore_index=True).sort_values("signal_date")
            result.to_parquet(prediction_path, index=False)
            return result

        final_train = self.dataset.rows_for_dates(fold.final_train_dates)
        prediction_rows = self.dataset.rows_for_dates(fold.prediction_dates)
        preprocessor = self.dataset.make_preprocessor().fit(final_train)
        x_train = preprocessor.transform(final_train).to_numpy(dtype=float)
        x_prediction = preprocessor.transform(prediction_rows).to_numpy(dtype=float)
        y_train = final_train[self.dataset.target_name].to_numpy(dtype=float)

        model = RidgeRegressor(
            alpha=float(tuning_result["alpha"]),
            fit_intercept=self.model_config["fit_intercept"],
        ).fit(x_train, y_train)
        predicted = model.predict(x_prediction)

        preprocessor.save(output_dir / "preprocessor.joblib")
        model.save(output_dir / "model.joblib")
        result = pd.DataFrame(
            {
                "signal_date": prediction_rows["signal_date"].to_numpy(),
                "model_period": fold.model_period,
                "model_name": self.model_name,
                "prediction": predicted,
                "actual": prediction_rows[self.dataset.target_name].to_numpy(dtype=float),
            }
        )
        result.to_parquet(prediction_path, index=False)
        return result
