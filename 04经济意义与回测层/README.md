# 04经济意义与回测层

本层只读取模型训练层第二次训练后形成的正式样本外预测，使用 Backtrader 对中证1000指数代理资产执行多头/现金择时回测。

## 回测口径

- T日收盘后读取预测，Backtrader市价单在T+1开盘成交。
- 每月第一个交易日收盘读取一次信号，与截面模型的月初调仓日程一致。
- Backtrader市价单在信号后的下一交易日开盘成交。
- Ridge与CNN-GRU分别使用自己的10日平滑预测，并以此前252个交易日预测做滞后标准化。
- `prediction < 0`且`z <= -0.5`为熊市，目标仓位0%。
- `prediction > 0`且`z >= 0.5`为牛市，目标仓位90%。
- 其余为震荡状态，目标仓位45%。
- 连续相同状态不重复交易；状态改变时在下一交易日开盘调整至目标仓位。
- 初始资金100万元，单边手续费0.2%，无显式滑点，现金收益与无风险利率均为0。
- 中证1000点位直接作为代理资产单位价格，不模拟IM期货乘数、保证金或真实指数产品。

这是一项指数代理经济意义检验，不应表述为可直接执行的实盘收益。

## 指标

- 累计收益
- 年化收益
- 年化波动率
- Sharpe
- Sortino
- 最大回撤
- Calmar
- 胜率（Backtrader已平仓多头交易中净利润为正的比例）

中证1000指数不作为另一套Backtrader策略运行，只在对比图中展示直接归一化净值，并单独计算Sharpe。

## 运行

```powershell
python 04经济意义与回测层/run_economic_value.py
python 04经济意义与回测层/validate_backtest.py
```

单独运行一个模型：

```powershell
python 04经济意义与回测层/run_economic_value.py --model ridge
```

## 产物

```text
data/processed/{model}/
├── equity_curve.parquet
├── signals.parquet
├── orders.parquet
└── trades.parquet

reports/{model}/performance.json
reports/{model}/rebalance_signals.csv

data/processed/comparison/
├── performance_summary.parquet
└── equity_comparison.parquet

reports/comparison/
├── benchmark.json
├── equity_comparison.png
├── rebalance_signals.csv
└── backtest_report.html
```

`reports/comparison/rebalance_signals.csv` 是面向查看的精简中文表；单模型目录中的同名CSV保留完整计算与审计字段。
