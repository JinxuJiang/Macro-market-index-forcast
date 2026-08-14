# 02 特征层

本层只读使用 `01数据` 的正式Parquet，将12类有效原始数据转换为中国交易日日频特征。本层执行PIT对齐、原生频率差分/收益/增长率和因果滚动计算；不构造未来标签，不做依赖训练样本统计量的缺失填补、异常值缩尾、标准化、PCA或特征选择。

## 代码结构

```text
feature_engine/
├── common.py                         # 数据加载、输入检查、公共公式与数据结构
├── pit_aligner.py                    # available_date映射、as-of展开与lineage
├── layer1_target.py                  # 中证1000自身状态
├── layer2_macro_cross_asset.py       # 资金、宏观、汇率与黄金
├── layer3_market_structure.py        # 国内结构与海外风险偏好
└── engine.py                         # 调度、验收、manifest与原子发布
```

## 计算顺序

```text
01层正式数据
→ 在每项数据的原生频率上计算特征
→ 读取data_availability绑定可用日
→ as-of映射到中国交易日
→ 合并三个信息层
→ 验收
→ 原子发布
```

PMI/CPI的月度变化在月度序列上计算；美股、USDCNH和黄金的收益与波动在各自原生行情序列上计算；两融先汇总沪深余额并计算变化，再按数据层给出的T+1可用日映射。

PMI和CPI均按数据层的统一规则做PIT对齐：统计期数据从下一自然月的最后一个中国交易日起进入特征表。在此之前继续使用上一期已可用数据，避免利用尚未到达约定可用时点的月份。

## 输出

```text
data/
├── processed/
│   ├── feature_table.parquet          # 模型层唯一正式特征输入
│   └── feature_lineage.parquet        # 每日各数据源实际使用日期
└── logs/
    ├── feature_manifest.json
    └── feature_validation_report.json
```

`feature_table.parquet`不含标签及统计预处理结果。模型层必须先完成时间切分，再仅用训练集拟合缺失填补器、异常值边界和Scaler。

## 使用

```powershell
python 02特征层/build_features.py
python 02特征层/validate_features.py
```

构建截止日自动取中证1000原始行情的最后日期，不提供手工截止日参数，也不会因预取的未来交易日历生成未来特征行。历史截断能力仅保留在引擎内部用于因果性测试，且不能晚于目标指数最后真实行情日。
