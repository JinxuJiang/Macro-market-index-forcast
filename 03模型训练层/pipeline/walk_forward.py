"""季度Expanding Walk-forward与基于真实exit_date的Purge。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import pandas as pd


@dataclass(frozen=True)
class QuarterlyFold:
    fold_id: int
    model_period: str
    as_of_date: pd.Timestamp
    inner_train_dates: List[pd.Timestamp]
    purge_dates: List[pd.Timestamp]
    inner_valid_dates: List[pd.Timestamp]
    final_train_dates: List[pd.Timestamp]
    prediction_dates: List[pd.Timestamp]

    @property
    def validation_start_date(self) -> pd.Timestamp:
        return self.inner_valid_dates[0]


def quarter_label(period: pd.Period) -> str:
    return f"{period.year}Q{period.quarter}"


class ExpandingQuarterlySplitter:
    def __init__(self, frame: pd.DataFrame, target_name: str, config: dict):
        self.frame = frame.sort_values("signal_date").reset_index(drop=True)
        self.target_name = target_name
        self.validation_days = int(config["validation_days"])
        self.first_period = pd.Period(config["first_prediction_quarter"], freq="Q")
        self.folds = self._build_folds()

    def _build_folds(self) -> List[QuarterlyFold]:
        dates = pd.DatetimeIndex(self.frame["signal_date"])
        periods = dates.to_period("Q")
        available_periods = sorted(set(periods[periods >= self.first_period]))
        folds: List[QuarterlyFold] = []

        for period in available_periods:
            prediction_mask = periods == period
            prediction_dates = dates[prediction_mask].tolist()
            if not prediction_dates:
                continue
            prediction_start = pd.Timestamp(prediction_dates[0])
            prior_dates = dates[dates < prediction_start]
            if len(prior_dates) == 0:
                continue
            as_of_date = pd.Timestamp(prior_dates[-1])

            eligible = self.frame[
                self.frame[self.target_name].notna()
                & self.frame["exit_date"].notna()
                & (self.frame["exit_date"] <= as_of_date)
                & (self.frame["signal_date"] < prediction_start)
            ].copy()
            if len(eligible) <= self.validation_days:
                continue

            final_train_dates = eligible["signal_date"].tolist()
            inner_valid = eligible.tail(self.validation_days)
            validation_start = pd.Timestamp(inner_valid["signal_date"].iloc[0])
            before_valid = eligible[eligible["signal_date"] < validation_start]
            inner_train = before_valid[before_valid["exit_date"] <= validation_start]
            purge = before_valid[before_valid["exit_date"] > validation_start]
            if len(inner_train) == 0 or len(inner_valid) != self.validation_days:
                continue

            folds.append(
                QuarterlyFold(
                    fold_id=len(folds),
                    model_period=quarter_label(period),
                    as_of_date=as_of_date,
                    inner_train_dates=inner_train["signal_date"].tolist(),
                    purge_dates=purge["signal_date"].tolist(),
                    inner_valid_dates=inner_valid["signal_date"].tolist(),
                    final_train_dates=final_train_dates,
                    prediction_dates=[pd.Timestamp(value) for value in prediction_dates],
                )
            )
        return folds

    def __iter__(self) -> Iterator[QuarterlyFold]:
        yield from self.folds

    def manifest_frame(self) -> pd.DataFrame:
        rows = []
        for fold in self.folds:
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model_period": fold.model_period,
                    "as_of_date": fold.as_of_date,
                    "inner_train_start": fold.inner_train_dates[0],
                    "inner_train_end": fold.inner_train_dates[-1],
                    "purge_start": fold.purge_dates[0] if fold.purge_dates else pd.NaT,
                    "purge_end": fold.purge_dates[-1] if fold.purge_dates else pd.NaT,
                    "inner_valid_start": fold.inner_valid_dates[0],
                    "inner_valid_end": fold.inner_valid_dates[-1],
                    "final_train_start": fold.final_train_dates[0],
                    "final_train_end": fold.final_train_dates[-1],
                    "prediction_start": fold.prediction_dates[0],
                    "prediction_end": fold.prediction_dates[-1],
                    "n_inner_train": len(fold.inner_train_dates),
                    "n_purge": len(fold.purge_dates),
                    "n_inner_valid": len(fold.inner_valid_dates),
                    "n_final_train": len(fold.final_train_dates),
                    "n_prediction": len(fold.prediction_dates),
                }
            )
        return pd.DataFrame(rows)

    def membership_frame(self) -> pd.DataFrame:
        records = []
        for fold in self.folds:
            roles = {
                "inner_train": fold.inner_train_dates,
                "purge": fold.purge_dates,
                "inner_valid": fold.inner_valid_dates,
                "final_train": fold.final_train_dates,
                "prediction": fold.prediction_dates,
            }
            for role, dates in roles.items():
                records.extend(
                    {
                        "fold_id": fold.fold_id,
                        "model_period": fold.model_period,
                        "signal_date": date,
                        "sample_role": role,
                    }
                    for date in dates
                )
        return pd.DataFrame.from_records(records)

    def write(self, processed_dir: Path) -> None:
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        self.manifest_frame().to_parquet(Path(processed_dir) / "fold_manifest.parquet", index=False)
        self.membership_frame().to_parquet(Path(processed_dir) / "fold_membership.parquet", index=False)
