"""标准scikit-learn Ridge封装。"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge


class RidgeRegressor:
    def __init__(self, alpha: float, fit_intercept: bool = True):
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        self.model.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(x), dtype=float)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
