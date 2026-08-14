# 03模型训练层

本层使用中证1000未来20日 open-to-open 收益作为唯一回归标签，以季度 Expanding Walk-forward 分别训练 Ridge 和 CNN-GRU。

完整设计见 [模型数据集与Walk-forward设计.md](模型数据集与Walk-forward设计.md)。

## 运行环境

使用 `qf` Conda 环境。CNN-GRU依赖该环境中的 TensorFlow 2.10、Keras 2.10、Optuna和可用的NVIDIA GPU。

## 运行顺序

先构建并验收共享标签与fold：

```powershell
python 03模型训练层/run_models.py --prepare-only --rebuild-data
python 03模型训练层/validate_models.py
```

运行全部Ridge季度：

```powershell
python 03模型训练层/run_models.py --model ridge
```

先用一个CNN-GRU季度检查环境和耗时：

```powershell
python 03模型训练层/run_models.py --model cnn_gru --start-quarter 2018Q2 --end-quarter 2018Q2
```

确认后运行全部CNN-GRU季度：

```powershell
python 03模型训练层/run_models.py --model cnn_gru
```

训练完成后只读验收：

```powershell
python 03模型训练层/validate_models.py
```

默认采用断点续跑：已经完成的寻参和季度预测会复用；同一fold有原子运行锁，拒绝两个进程同时训练。配置或该fold训练历史发生变化时，程序拒绝静默复用冻结产物。

## 常用限制参数

```text
--start-quarter 2022Q1
--end-quarter 2023Q4
--max-folds 1
```

`--max-folds` 用于代表fold基准检查，不会改变该fold的正式参数规则。

## 预测平滑

每个模型同时保留两条正式样本外预测链：

- `experiments/{model}/oos_predictions.parquet`：未经后处理的原始预测。
- `experiments/{model}/oos_predictions_smoothed.parquet`：按日期因果计算的EWM平滑预测，固定半衰期10个交易日。

平滑参数依据2018Q2—2021Q4开发期的预测效力延迟曲线固定；2022年以后的样本不参与参数选择。平滑是确定性的预测后处理，不改变或解冻已经训练完成的模型。
