# 01 数据层

本层只负责把可追溯的原始数据稳定落盘、记录数据可用时点并执行质量验收，不计算模型特征。

## V1 数据范围

V1 使用 13 类原始数据：1份交易日历负责时间对齐，其余12类数据按三层信息结构组织。

| 层级 | 原始数据 | Tushare接口/代码 | 状态 |
|---|---|---|---|
| 时间基础 | 中国交易日历 | `trade_cal` | 已接入 |
| 第一层 | 中证1000 | `index_daily`, `000852.SH` | 已接入 |
| 第二层 | 1个月Shibor | `shibor.1m` | 已接入 |
| 第二层 | 沪深融资余额 | `margin.rzye` | 已接入 |
| 第二层 | 制造业PMI | `cn_pmi.PMI010000` | 已接入 |
| 第二层 | 全国CPI | `cn_cpi.nt_yoy` | 已接入 |
| 第二层 | 离岸人民币 | `fx_daily`, `USDCNH.FXCM` | 已接入 |
| 第二层 | 上海金 | `sge_daily`, `Au99.99` | 已接入 |
| 第三层 | 沪深300 | `index_daily`, `000300.SH` | 已接入 |
| 第三层 | 中证500 | `index_daily`, `000905.SH` | 已接入 |
| 第三层 | 中证全指 | `index_daily`, `000985.CSI` | 已接入 |
| 第三层 | 标普500 | `index_global`, `SPX` | 已接入 |
| 第三层 | 纳斯达克综合指数 | `index_global`, `IXIC` | 已接入 |

上证指数旧文件可以保留，但不再更新、验收或进入V1特征。原油、PPI、GDP、长期利率和信用利差不进入V1。

## 目录

```text
01数据/
├── config/data_sources.yaml
├── tushare_engine/
│   ├── engine.py
│   └── adapter.py
├── download_data.py
├── validate_data.py
└── data/                         # 运行时生成，不提交
    ├── raw/
    │   ├── trade_cal.parquet
    │   ├── layer1_target/
    │   │   └── 000852.SH.parquet
    │   ├── layer2_macro_cross_asset/
    │   │   ├── shibor.parquet
    │   │   ├── margin.parquet
    │   │   ├── pmi.parquet
    │   │   ├── cpi.parquet
    │   │   ├── USDCNH.FXCM.parquet
    │   │   └── Au99.99.parquet
    │   └── layer3_market_structure/
    │       ├── 000300.SH.parquet
    │       ├── 000905.SH.parquet
    │       ├── 000985.CSI.parquet
    │       ├── SPX.parquet
    │       └── IXIC.parquet
    ├── processed/
    │   └── data_availability.parquet
    └── logs/
        ├── download_manifest.json
        └── validation_report.json
```

首次运行新结构时，国内指数、Shibor和两融会读取旧目录中的已有文件作为迁移来源；旧文件不会被删除。

## 数据可用时点

`data_availability.parquet` 记录每条原始观测最早可以进入模型的中国交易日。它不是模型特征，而是PIT审计依据。

| 数据 | V1可用规则 |
|---|---|
| 国内指数 | T日收盘后可用 |
| 1个月Shibor | T日收盘后可用 |
| 两融 | T日数据从T+1交易日开始使用 |
| PMI | 统计期下一自然月的最后一个中国交易日开始使用 |
| CPI | 统计期下一自然月的最后一个中国交易日开始使用 |
| SPX、IXIC | 数据日期后的首个中国交易日使用 |
| USDCNH | GMT数据日期后的首个中国交易日使用 |
| Au99.99 | 完整日行情从下一中国交易日使用 |

主要字段为 `dataset`、`period_date`、`data_date`、`publish_date`、`available_date`、`availability_method` 和 `lag_value`。

PMI和CPI暂不依赖不完整的历史发布日期表，统一采用上述简单、保守且可延续到未来年份的规则。交易日历预取到当年年末，以便在月末当天准确判断最后一个交易日；特征输出仍严格截断在用户指定的截止日。

## Token

优先顺序：

1. 环境变量 `TUSHARE_TOKEN`；
2. 本仓库 `01数据/tushare_token.txt`；
3. 截面仓库已有的 `01数据/tushare_token.txt`，只读复用。

Token文件已加入 `.gitignore`。

## 使用

首次下载与后续更新使用同一个命令。日频文件会重抓末尾10个自然日并按主键去重；PMI和CPI体量小，每次获取完整接口结果后按月份合并。

```powershell
python 01数据/download_data.py
python 01数据/download_data.py --end-date 20260812
python 01数据/validate_data.py --end-date 20260812
```

默认截止日期按 `Asia/Shanghai` 时间计算：18:00前截至昨天，18:00后允许截至今天。

## 测试

模拟数据逻辑测试不访问网络：

```powershell
python -m unittest discover -s tests -p "test_engine_logic.py" -v
```

小样本真实Tushare测试只请求少量固定日期数据，不写入正式数据目录：

```powershell
$env:RUN_TUSHARE_INTEGRATION="1"
python -m unittest discover -s tests -p "test_real_tushare_sample.py" -v
Remove-Item Env:RUN_TUSHARE_INTEGRATION
```

`validate_data.py` 检查磁盘上的真实完整数据；`tests/` 检查增量合并、PIT映射和接口结构等代码逻辑，二者用途不同。

## 数据规则

- 原始接口字段尽量原样保存，标准日期保留为 `YYYYMMDD` 字符串；
- 文件采用Parquet + Zstandard，并通过临时文件原子替换；
- 国内外指数、外汇和黄金主键为 `(ts_code, trade_date)`；
- Shibor主键为 `date`，两融主键为 `(trade_date, exchange_id)`；
- PMI和CPI主键统一为小写字段 `month`；
- 下载结果写入 `download_manifest.json`，验收结果写入 `validation_report.json`；
- 正式数据下载失败时命令返回失败，不静默跳过，也不会用空表覆盖已有文件。
