"""第三层：国内市场结构与海外风险偏好。"""

from __future__ import annotations

import pandas as pd

from .common import LayerResult, price_features


FEATURE_SPECS = {
    "csi300_ret_20d": {"source": ["index:000300.SH"], "formula": "close/close.shift(20)-1"},
    "csi300_vol_20d": {"source": ["index:000300.SH"], "formula": "std(daily_return,20)*sqrt(252)"},
    "csi500_ret_20d": {"source": ["index:000905.SH"], "formula": "close/close.shift(20)-1"},
    "csi500_vol_20d": {"source": ["index:000905.SH"], "formula": "std(daily_return,20)*sqrt(252)"},
    "csi_all_ret_20d": {"source": ["index:000985.CSI"], "formula": "close/close.shift(20)-1"},
    "csi_all_vol_20d": {"source": ["index:000985.CSI"], "formula": "std(daily_return,20)*sqrt(252)"},
    "csi1000_vs_csi300_ret_20d": {"source": ["index:000852.SH", "index:000300.SH"], "formula": "target_ret_20d-csi300_ret_20d"},
    "csi1000_vs_csi_all_ret_20d": {"source": ["index:000852.SH", "index:000985.CSI"], "formula": "target_ret_20d-csi_all_ret_20d"},
    "spx_ret_20d": {"source": ["global_index:SPX"], "formula": "native close/close.shift(20)-1"},
    "spx_vol_20d": {"source": ["global_index:SPX"], "formula": "std(native_daily_return,20)*sqrt(252)"},
    "ixic_ret_20d": {"source": ["global_index:IXIC"], "formula": "native close/close.shift(20)-1"},
    "ixic_vol_20d": {"source": ["global_index:IXIC"], "formula": "std(native_daily_return,20)*sqrt(252)"},
    "ixic_vs_spx_ret_20d": {"source": ["global_index:IXIC", "global_index:SPX"], "formula": "ixic_ret_20d-spx_ret_20d on China signal date"},
}


def _build_index(loader, aligner, source_name, dataset, prefix, annualization_days):
    data = loader.load(
        source_name,
        ["ts_code", "trade_date", "close"],
        date_column="trade_date",
        keys=["ts_code", "trade_date"],
    )
    native = price_features(
        pd.to_numeric(data.set_index("trade_date")["close"], errors="coerce"),
        prefix,
        [20],
        [20],
        annualization_days,
    ).reset_index()
    columns = ["{}_ret_20d".format(prefix), "{}_vol_20d".format(prefix)]
    return aligner.align(dataset, native, "trade_date", columns)


def build_layer3(loader, aligner, annualization_days: int, layer1: LayerResult) -> LayerResult:
    pieces = []
    lineages = []
    specifications = [
        ("csi300", "index:000300.SH", "csi300"),
        ("csi500", "index:000905.SH", "csi500"),
        ("csi_all", "index:000985.CSI", "csi_all"),
        ("spx", "global_index:SPX", "spx"),
        ("ixic", "global_index:IXIC", "ixic"),
    ]
    for source_name, dataset, prefix in specifications:
        feature, lineage = _build_index(
            loader, aligner, source_name, dataset, prefix, annualization_days
        )
        pieces.append(feature)
        lineages.append(lineage)

    features = pd.concat(pieces, axis=1)
    features["csi1000_vs_csi300_ret_20d"] = (
        layer1.features["ret_20d"] - features["csi300_ret_20d"]
    )
    features["csi1000_vs_csi_all_ret_20d"] = (
        layer1.features["ret_20d"] - features["csi_all_ret_20d"]
    )
    features["ixic_vs_spx_ret_20d"] = (
        features["ixic_ret_20d"] - features["spx_ret_20d"]
    )
    lineage = pd.concat(lineages, ignore_index=True)
    return LayerResult(features, lineage, FEATURE_SPECS)

