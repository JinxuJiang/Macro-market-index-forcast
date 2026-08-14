"""第二层：短期资金、宏观、汇率与黄金。"""

from __future__ import annotations

import pandas as pd

from .common import LayerResult, price_features, simple_return


FEATURE_SPECS = {
    "shibor_1m_level": {"source": ["shibor"], "formula": "1m Shibor level"},
    "shibor_1m_change_20d": {"source": ["shibor"], "formula": "shibor-shibor.shift(20)"},
    "margin_balance": {"source": ["margin"], "formula": "sum(rzye,SSE+SZSE)"},
    "margin_growth_20d": {"source": ["margin"], "formula": "balance/balance.shift(20)-1"},
    "pmi_gap": {"source": ["pmi"], "formula": "PMI010000-50"},
    "pmi_change_1m": {"source": ["pmi"], "formula": "PMI010000.diff(1)"},
    "cpi_yoy": {"source": ["cpi"], "formula": "nt_yoy"},
    "cpi_yoy_change_1m": {"source": ["cpi"], "formula": "nt_yoy.diff(1)"},
    "usdcnh_ret_5d": {"source": ["fx"], "formula": "mid_close/mid_close.shift(5)-1"},
    "usdcnh_ret_20d": {"source": ["fx"], "formula": "mid_close/mid_close.shift(20)-1"},
    "usdcnh_vol_20d": {"source": ["fx"], "formula": "std(native_daily_return,20)*sqrt(252)"},
    "gold_ret_20d": {"source": ["gold"], "formula": "close/close.shift(20)-1"},
    "gold_vol_20d": {"source": ["gold"], "formula": "std(native_daily_return,20)*sqrt(252)"},
}


def _align(aligner, dataset, native, date_column, columns):
    return aligner.align(dataset, native, date_column, columns)


def build_layer2(loader, aligner, annualization_days: int) -> LayerResult:
    pieces = []
    lineages = []

    shibor = loader.load(
        "shibor", ["date", "1m"], date_column="date", keys=["date"]
    ).copy()
    shibor["shibor_1m_level"] = pd.to_numeric(shibor["1m"], errors="coerce")
    shibor["shibor_1m_change_20d"] = shibor["shibor_1m_level"].diff(20)
    columns = ["shibor_1m_level", "shibor_1m_change_20d"]
    feature, lineage = _align(aligner, "shibor", shibor, "date", columns)
    pieces.append(feature)
    lineages.append(lineage)

    margin = loader.load(
        "margin",
        ["trade_date", "exchange_id", "rzye"],
        date_column="trade_date",
        keys=["trade_date", "exchange_id"],
    )
    margin = margin[margin["exchange_id"].isin(["SSE", "SZSE"])]
    margin = margin.groupby("trade_date", as_index=False)["rzye"].sum(min_count=1)
    margin = margin.sort_values("trade_date")
    margin["margin_balance"] = pd.to_numeric(margin["rzye"], errors="coerce")
    margin["margin_growth_20d"] = simple_return(margin["margin_balance"], 20)
    columns = ["margin_balance", "margin_growth_20d"]
    feature, lineage = _align(
        aligner, "margin", margin, "trade_date", columns
    )
    pieces.append(feature)
    lineages.append(lineage)

    pmi = loader.load(
        "pmi", ["month", "PMI010000"], date_column="month", monthly=True, keys=["month"]
    ).copy()
    pmi["pmi_gap"] = pd.to_numeric(pmi["PMI010000"], errors="coerce") - 50.0
    pmi["pmi_change_1m"] = pd.to_numeric(pmi["PMI010000"], errors="coerce").diff()
    columns = ["pmi_gap", "pmi_change_1m"]
    feature, lineage = _align(aligner, "pmi", pmi, "month", columns)
    pieces.append(feature)
    lineages.append(lineage)

    cpi = loader.load(
        "cpi", ["month", "nt_yoy"], date_column="month", monthly=True, keys=["month"]
    ).copy()
    cpi["cpi_yoy"] = pd.to_numeric(cpi["nt_yoy"], errors="coerce")
    cpi["cpi_yoy_change_1m"] = cpi["cpi_yoy"].diff()
    columns = ["cpi_yoy", "cpi_yoy_change_1m"]
    feature, lineage = _align(aligner, "cpi", cpi, "month", columns)
    pieces.append(feature)
    lineages.append(lineage)

    fx = loader.load(
        "fx",
        ["ts_code", "trade_date", "bid_close", "ask_close"],
        date_column="trade_date",
        keys=["ts_code", "trade_date"],
    ).copy()
    fx["mid_close"] = (
        pd.to_numeric(fx["bid_close"], errors="coerce")
        + pd.to_numeric(fx["ask_close"], errors="coerce")
    ) / 2.0
    native_fx = price_features(
        fx.set_index("trade_date")["mid_close"],
        "usdcnh",
        [5, 20],
        [20],
        annualization_days,
    ).reset_index()
    columns = ["usdcnh_ret_5d", "usdcnh_ret_20d", "usdcnh_vol_20d"]
    feature, lineage = _align(aligner, "fx", native_fx, "trade_date", columns)
    pieces.append(feature)
    lineages.append(lineage)

    gold = loader.load(
        "gold",
        ["ts_code", "trade_date", "close"],
        date_column="trade_date",
        keys=["ts_code", "trade_date"],
    )
    native_gold = price_features(
        pd.to_numeric(gold.set_index("trade_date")["close"], errors="coerce"),
        "gold",
        [20],
        [20],
        annualization_days,
    ).reset_index()
    columns = ["gold_ret_20d", "gold_vol_20d"]
    feature, lineage = _align(
        aligner, "gold", native_gold, "trade_date", columns
    )
    pieces.append(feature)
    lineages.append(lineage)

    features = pd.concat(pieces, axis=1)
    features = features.loc[:, ~features.columns.duplicated()]
    lineage = pd.concat(lineages, ignore_index=True)
    return LayerResult(features, lineage, FEATURE_SPECS)

