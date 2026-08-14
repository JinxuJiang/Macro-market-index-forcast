"""第一层：中证1000自身量价与趋势状态。"""

from __future__ import annotations

import pandas as pd

from .common import (
    LayerResult,
    annualized_volatility,
    moving_average_distance,
    rolling_drawdown,
    rolling_mean_ratio,
    safe_divide,
    simple_return,
)


FEATURE_SPECS = {
    "ret_5d": {"source": ["index:000852.SH"], "formula": "close/close.shift(5)-1"},
    "ret_20d": {"source": ["index:000852.SH"], "formula": "close/close.shift(20)-1"},
    "ret_60d": {"source": ["index:000852.SH"], "formula": "close/close.shift(60)-1"},
    "ret_120d": {"source": ["index:000852.SH"], "formula": "close/close.shift(120)-1"},
    "vol_20d": {"source": ["index:000852.SH"], "formula": "std(daily_return,20)*sqrt(252)"},
    "vol_60d": {"source": ["index:000852.SH"], "formula": "std(daily_return,60)*sqrt(252)"},
    "price_to_ma20": {"source": ["index:000852.SH"], "formula": "close/mean(close,20)-1"},
    "price_to_ma60": {"source": ["index:000852.SH"], "formula": "close/mean(close,60)-1"},
    "price_to_ma250": {"source": ["index:000852.SH"], "formula": "close/mean(close,250)-1"},
    "drawdown_20d": {"source": ["index:000852.SH"], "formula": "close/max(close,20)-1"},
    "drawdown_250d": {"source": ["index:000852.SH"], "formula": "close/max(close,250)-1"},
    "intraday_range_20d": {"source": ["index:000852.SH"], "formula": "mean((high-low)/pre_close,20)"},
    "amount_ratio_20d": {"source": ["index:000852.SH"], "formula": "amount/mean(amount,20)"},
}


def build_layer1(loader, aligner, annualization_days: int) -> LayerResult:
    required = [
        "ts_code",
        "trade_date",
        "high",
        "low",
        "close",
        "pre_close",
        "amount",
    ]
    data = loader.load(
        "target",
        required,
        date_column="trade_date",
        keys=["ts_code", "trade_date"],
    ).set_index("trade_date")
    close = pd.to_numeric(data["close"], errors="coerce")
    native = pd.DataFrame(index=data.index)
    for window in (5, 20, 60, 120):
        native["ret_{}d".format(window)] = simple_return(close, window)
    for window in (20, 60):
        native["vol_{}d".format(window)] = annualized_volatility(
            close, window, annualization_days
        )
    for window in (20, 60, 250):
        native["price_to_ma{}".format(window)] = moving_average_distance(
            close, window
        )
    for window in (20, 250):
        native["drawdown_{}d".format(window)] = rolling_drawdown(close, window)
    daily_range = safe_divide(
        pd.to_numeric(data["high"], errors="coerce")
        - pd.to_numeric(data["low"], errors="coerce"),
        pd.to_numeric(data["pre_close"], errors="coerce"),
    )
    native["intraday_range_20d"] = daily_range.rolling(20, min_periods=20).mean()
    native["amount_ratio_20d"] = rolling_mean_ratio(
        pd.to_numeric(data["amount"], errors="coerce"), 20
    )
    native = native.reset_index()
    features, lineage = aligner.align(
        "index:000852.SH", native, "trade_date", FEATURE_SPECS.keys()
    )
    return LayerResult(features, lineage, FEATURE_SPECS)

