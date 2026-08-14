# 模型数据集与 Walk-forward 设计

> 状态：标签、时间切分、PIT、预处理、评价口径以及 Ridge 和 CNN-GRU 的 V1 训练规则均已讨论确认；下一步讨论工程架构。
>
> 本文只定义两个模型共同遵守的数据与训练协议，不包含模型实现。

## 1. 任务定义

本项目采用单目标回归，不构造上涨/下跌分类标签，也不构造牛市、熊市、震荡市等离散状态标签。

```yaml
task: regression
target_index: 000852.SH
signal_frequency: daily
model_update_frequency: quarterly
```

两个计划模型使用完全相同的标签、fold、预处理边界和正式预测日期：

- Ridge：受 L2 约束的线性回归基准；
- CNN-GRU：使用一段历史特征序列的非线性时序模型。

两种模型各自按照适合自身的样本形式训练和评价，不要求为了样本形式一致而让 Ridge 放弃可用训练行。2018Q2 以后的正式季度预测日期保持一致，最终直接比较相同指标。

## 2. 当前数据范围

截至 2026-08-12，当前产物为：

| 项目 | 范围 |
|---|---|
| 中证1000原始行情 | 2010-01-04 至 2026-08-12，共 4033 个交易日 |
| 日频特征表 | 2010-01-04 至 2026-08-12，共 4033 行、39 个特征 |
| 第一条39特征全部完整的记录 | 2012-03-19 |
| 正式模型样本起点 | 2012-03-19 |
| 当前最后一条完整20日 open-to-open 标签 | 2026-07-14 |

最后一条完整标签日期必须在每次运行时根据交易日历和真实行情动态计算，不能把 `2026-07-14` 写死在代码中。最新约 21 个尚未实现标签的信号日可以生成预测，但不能进入训练或评价。

## 3. 唯一回归标签

样本归属日为信号日 `T`。T 日收盘后获得当日全部特征，最早在下一交易日开盘执行：

```text
entry_date(T) = T 后第 1 个交易日
exit_date(T)  = T 后第 21 个交易日

target_20d_o2o(T)
    = open[exit_date(T)] / open[entry_date(T)] - 1
    = open[T+21] / open[T+1] - 1
```

标签表示从 T+1 开盘到 T+21 开盘、共 20 个交易日区间的简单收益率。

约束：

- 使用中证1000原始 `open`，不使用 `close`；
- 使用简单收益率，不使用对数收益率；
- 不扣除 Shibor 或其他无风险利率；
- 标签不填充、不缩尾、不标准化；
- 数据集中显式保存 `signal_date`、`entry_date` 和 `exit_date`，PIT 判断使用真实日期而不是只依赖行号。

## 4. 外层季度 Expanding Walk-forward

正式回测从 `2018Q2` 开始。每个自然季度是一个外层 fold，也是该季度模型唯一负责的预测区间。

```text
2018Q2 fold：使用当时可见历史 → 寻参 → 最终重训 → 只预测2018Q2
2018Q3 fold：扩展历史数据     → 寻参 → 最终重训 → 只预测2018Q3
2018Q4 fold：继续扩展历史     → 寻参 → 最终重训 → 只预测2018Q4
……
```

Expanding 的左端始终固定在 `2012-03-19`，右端随季度推进而扩展；不采用会丢弃早期样本的 rolling window。

```yaml
outer_walk_forward:
  type: expanding
  data_start: 2012-03-19
  first_prediction_quarter: 2018Q2
  prediction_period: natural_quarter
  model_update: quarterly
  model_fixed_within_quarter: true
```

每个月可以更新上游数据并追加当月日频预测，但同一自然季度内不得重新拟合预处理器、模型权重或超参数。进入下一自然季度后，才建立新的 fold 并重新寻参、训练。

### 4.1 Fold 的训练时点

预测季度开始前的最后一个中国交易日为该 fold 的 `as_of_date`。季度最终训练样本必须同时满足：

```text
feature signal_date <= as_of_date
label exit_date     <= as_of_date
```

第二条规则保证模型训练时，样本的 T+21 卖出开盘价已经真实发生。不能仅按 `signal_date < prediction_quarter` 判断训练资格。

## 5. Fold 内部验证集

每个外层季度 fold 从其合法历史样本中取最后 252 个交易日作为内部验证集：

```yaml
inner_validation:
  type: chronological_holdout
  length: 252 trading days
```

固定一年而不采用“历史最后15%”，可以保证 expanding 过程中各季度的调参尺度基本一致。

内部验证集只负责该 fold 的超参数选择和 CNN-GRU 早停控制，不作为最终 OOS 预测结果。不同外层 fold 的内部验证窗口允许重叠，但不得把这些重叠验证行直接拼接成总体评价样本。

## 6. 训练集与验证集之间的 Purge

20日标签高度重叠。若内部验证集第一条信号日为 `validation_start_date`，内部训练集中只允许保留在该时点已经完整实现标签的样本：

```text
train_sample.exit_date <= validation_start_date
```

验证集开始前不满足该条件的历史信号日进入 purge 区，通常约为 20 个交易日：

```text
内部训练集 | purge（约20日） | 内部验证集（252日）
```

必须根据每行真实 `exit_date` 判断 purge，避免因为 Python 切片端点不同而混淆配置中的 20 或 21。

Purge 的作用是模拟内部验证开始时的真实信息集：当时这些样本的未来20日标签还没有实现，因此不能用于候选模型训练。

## 7. 每个 Fold 的标准执行顺序

采用逐 fold 完整执行，不采用“全部fold统一寻参后再整体重跑”的两遍式编排。

```text
Fold 1：内部寻参 → 最终重训 → 预测本季度 → 保存
Fold 2：内部寻参 → 最终重训 → 预测本季度 → 保存
Fold 3：内部寻参 → 最终重训 → 预测本季度 → 保存
……
```

单个 fold 的生命周期为：

1. 根据季度 `as_of_date` 取得全部合法历史样本；
2. 取最后 252 个交易日作为内部验证集；
3. 根据 `exit_date <= validation_start_date` 建立内部训练集和 purge 区；
4. 在内部训练集上训练每组候选超参数；
5. 在内部验证集上计算 Huber Loss，选择该 fold 的最佳超参数；
6. 将内部训练集、purge 区和内部验证集重新合并为季度开始前的全部合法历史；
7. 在全部合法历史上重新拟合预处理器；
8. 使用最佳超参数从头训练该季度最终模型；
9. 固定最终预处理器和模型，预测该自然季度的全部交易日；
10. 保存该 fold 的边界、候选结果、最佳参数、最终预处理器、最终模型和正式预测。

各季度正式预测日期互不重叠，因此可以按日期拼接为一条完整 OOS 预测链。每个信号日只能出现一次。

## 8. 为什么最终重训要重新合并数据

内部寻参时，验证集不能参与候选模型权重拟合；否则它不能公正地比较超参数。

最佳超参数确定后，外层预测季度才是尚未接触的 test。此时季度开始前的验证数据和 purge 数据均已成为合法历史，可以用于最终模型：

```text
内部训练集 + purge区 + 内部验证集
              ↓
       全部合法历史样本
              ↓
       训练季度最终模型
```

若不重新合并，正式模型会持续丢弃最近一年的验证数据和约20日的 purge 数据，不符合 expanding 尽量利用历史信息的目标。

这不会造成未来泄漏，因为所有重新加入样本都必须满足 `label exit_date <= fold as_of_date`，而外层预测季度从未参与寻参或重训。

## 9. Fold 内预处理

基础预处理顺序暂定为：

```text
训练集统计量中位数填充
→ 训练集1%/99%分位缩尾
→ 训练集均值与标准差标准化
```

同一个 fold 内部需要拟合两次预处理器。

### 9.1 内部寻参预处理器

```text
fit：内部训练集
transform：内部训练集、内部验证集
```

验证集不能参与缺失值中位数、缩尾边界、均值或标准差的估计。

### 9.2 季度最终预处理器

选出最佳超参数后：

```text
fit：季度开始前全部合法历史
transform：全部合法历史、当季预测特征
```

加入最近一年验证数据和 purge 数据后，统计量发生变化是正常现象。寻参选择的是预处理方法与模型超参数，不是永久冻结内部训练集的具体均值或分位点。

季度最终预处理器必须和最终模型绑定保存；同一季度内不得随着月度数据更新而修改。

所有模型必须遵循相同的时间边界。Ridge 和 CNN-GRU 均执行上述预处理。CNN-GRU 在完成缺失值填补、缩尾和标准化后，再从连续交易日构造60日输入序列。

## 10. Huber 口径

MAE 不再用于寻参或最终评价，统一改用 Huber Loss：

```yaml
huber:
  delta: 0.05
```

原始标签以小数收益率表示，因此 `delta=0.05` 表示预测误差在 5 个百分点以内采用平方惩罚，超过后转为线性惩罚。

使用位置：

- 每个 fold 以内部验证集 Huber Loss 最小选择最佳超参数；
- CNN-GRU 使用 Huber Loss 作为训练损失；
- 最终 OOS 评价报告 Huber Loss。

Ridge 仍保留标准目标，即平方误差加 L2 惩罚。若将 Ridge 的训练目标也改为 Huber，它将变成 Huber 回归而不再是标准 Ridge。Ridge 的 `alpha` 仅通过验证集 Huber Loss 选择。

## 11. 最终评价指标

正式评价只使用拼接后的、不重叠的季度 OOS 预测：

```yaml
metrics:
  - r2
  - huber_loss
  - direction_accuracy
```

### 11.1 普通测试集 R²

使用标准回归 `r2_score`：

```text
R² = 1 - Σ(y - ŷ)² / Σ(y - mean(y))²
```

不建设实时历史平均收益 HA 基准，也不使用论文定义的相对 HA 的 OOS R²。

### 11.2 Huber Loss

使用固定 `delta=0.05`，与 fold 内寻参口径一致。

### 11.3 方向准确率

```text
direction_accuracy = mean(sign(prediction) == sign(actual))
```

除全时期总体指标外，同样按自然年和自然季度输出这三个指标，以检查成绩是否集中在少数行情阶段。由于日频20日标签相互重叠，这些指标是描述性 OOS 评价；V1 暂不增加显著性检验。

## 12. PIT 与验收要求

每个 fold 至少验收：

- 训练、purge、验证和季度预测日期严格按时间递增；
- 内部训练样本全部满足 `exit_date <= validation_start_date`；
- 季度最终训练样本全部满足 `exit_date <= as_of_date`；
- 内部验证集恰为 252 个交易日，数据不足的 fold 不得生成；
- 内部验证数据不参与内部预处理器拟合；
- 外层预测季度不参与寻参、预处理或模型拟合；
- 不同季度正式预测日期没有重复；
- 同季度最终预处理器、超参数和模型权重保持固定；
- 最新未实现标签的预测行不得进入评价；
- Ridge 与 CNN-GRU 使用同一组外层预测日期。

建议保留以下审计字段：

```text
fold_id
model_period
as_of_date
train_start / train_end
purge_start / purge_end
valid_start / valid_end
prediction_start / prediction_end
n_train / n_purge / n_valid / n_prediction
selected_parameters
validation_huber_loss
```

## 13. 计划产物

具体文件格式可在实现前微调，但职责应分开：

```text
dataset_with_label.parquet   # 特征、唯一回归标签及三个关键日期
fold_manifest.parquet        # 每个季度的时间边界与样本数量
tuning_results.parquet       # 各fold候选参数、验证Huber与最佳参数
oos_predictions.parquet      # 日期唯一的正式季度OOS预测链
evaluation_report.json       # 总体、逐年和逐季度三个指标
```

模型文件与最终预处理器按 `model_name/model_period` 保存，不覆盖历史季度产物。

## 14. Ridge V1 训练规则

Ridge 使用信号日 T 当天的39维特征预测唯一的未来20日收益标签。当天特征中已经包含5日、20日、60日、120日和250日等历史窗口信息，但模型本身不读取额外序列。

```yaml
ridge:
  fit_intercept: true
  alpha_candidates: [0.01, 0.1, 1, 10, 100, 1000]
  training_objective: squared_error_plus_l2
  selection_metric: validation_huber_loss
  tie_break: larger_alpha
  prediction_clipping: none
```

Ridge 不使用 Optuna。每个 fold 依次测试六个固定 `alpha`，以内部验证集 Huber Loss 最小者为最佳参数；完全并列时选择更大的 `alpha`。选定后按照第7节的规则，使用全部合法历史重新预处理和训练。

## 15. CNN-GRU V1 训练规则

### 15.1 技术栈

```yaml
framework: TensorFlow 2.10.0
api: tf.keras
gpu: NVIDIA GeForce RTX 3060 Laptop GPU 6GB
tuner: Optuna TPE
```

当前 `qf` 环境已经安装 TensorFlow 2.10、Keras 2.10 和 Optuna，且 TensorFlow 可以识别 GPU；不切换到 PyTorch。

### 15.2 输入与序列

CNN-GRU 使用以信号日 T 结束的连续60个交易日特征：

```text
X[T-59 : T]，形状为 60 × 39
                  ↓
       target_20d_o2o(T)
```

第一条CNN-GRU训练样本比第一条完整特征晚59个交易日，但正式预测仍从2018Q2开始，不额外推迟。每个新季度的首日序列允许读取上一季度已经发生的59日特征；季度内后续预测允许读取当季截至信号日已经发生的特征。

预处理和序列构造顺序为：

```text
按fold拟合并执行缺失值填补、缩尾、标准化
→ 保持交易日连续
→ 构造60日窗口
```

验证集首日的输入窗口允许包含验证期以前的历史特征。Purge 限制的是训练标签不能跨入验证期，不禁止验证序列读取当时已知的历史特征。

模型采用非 stateful GRU。训练窗口之间可以打乱，但每个窗口内部的60日顺序不得改变；验证和预测不打乱。

### 15.3 固定结构与搜索空间

网络固定为小型单层结构：

```text
Input(60, 39)
→ Conv1D(ReLU, padding="same")
→ Dropout
→ GRU
→ Dropout
→ Dense(1, linear)
```

```yaml
cnn_gru:
  fixed:
    sequence_length: 60
    cnn_layers: 1
    gru_layers: 1
    batch_size: 32
    bidirectional: false
    stateful: false
    optimizer: Adam
    loss: Huber
    huber_delta: 0.05
    learning_rate_scheduler: none
    training_shuffle: true
    validation_shuffle: false
    prediction_shuffle: false

  optuna_search_space:
    cnn_filters: [16, 32]
    kernel_size: [3, 5]
    gru_hidden_size: [16, 32, 64]
    dropout: [0.0, 0.1, 0.2]
    learning_rate:
      low: 0.0003
      high: 0.003
      log: true
```

V1 不使用 BatchNormalization、多层CNN、多层GRU、池化、双向GRU或Attention，也不搜索序列长度、batch size、优化器、Huber delta和网络层数。

### 15.4 Optuna与双随机种子

```yaml
optuna:
  sampler: TPE
  trials_per_fold: 12
  pruner: none
  objective: mean_validation_huber_loss
random_seeds: [42, 2026]
```

每个 fold 建立独立的 Optuna study。每个 trial 的同一组超参数分别使用两个固定随机种子训练，两个种子的最佳验证 Huber Loss 平均值作为该 trial 的分数。12个trial中平均验证 Huber 最小者为该 fold 的最佳超参数。

V1 不使用 Optuna pruning；Keras early stopping 负责提前结束不再改善的训练。两个种子的调参与最终预测明细均分别保存，正式预测取两个最终模型预测的算术平均。

### 15.5 Early stopping与最终重训epoch

内部寻参阶段：

```yaml
early_stopping:
  monitor: val_loss
  mode: min
  max_epochs: 150
  patience: 12
  min_delta: 0.00001
  restore_best_weights: true
```

最佳trial为两个种子分别保存各自的 `best_epoch`。最终合并全部合法历史后不再划验证集、不再 early stopping，也不读取预测季度标签：

```text
seed 42最终模型   → 按seed 42内部寻参记录的best_epoch训练
seed 2026最终模型 → 按seed 2026内部寻参记录的best_epoch训练
正式预测          → 两个模型预测取平均
```

最终重训不使用 `ReduceLROnPlateau`，以避免在不存在独立验证集时改变学习率规则。若最佳epoch达到150轮上限，记录验收警告但不自动扩大上限。

## 16. 运行规模预估

从2018Q2至当前约34个季度fold。CNN-GRU按每fold 12个trial、每trial 2个种子估算，寻参约需816次小模型训练，最终重训约需68次。RTX 3060 Laptop GPU上的实际耗时应先用一个代表fold实测后外推；当前粗略估计为4至8小时，异常慢时可能达到10至15小时。训练流程必须支持按fold保存和断点续跑。

## 17. 工程架构待讨论事项

工程架构已确认如下：

```text
03模型训练层/
├── config.yaml
├── pipeline/
│   ├── dataset.py
│   ├── walk_forward.py
│   └── quarterly_runner.py
├── models/
│   ├── ridge/
│   │   ├── config.yaml
│   │   ├── model.py
│   │   └── trainer.py
│   └── cnn_gru/
│       ├── config.yaml
│       ├── model.py
│       └── trainer.py
├── run_models.py
├── validate_models.py
├── data/
│   ├── processed/
│   └── logs/
└── experiments/
    ├── ridge/
    └── cnn_gru/
```

公共 `walk_forward.py` 只生成季度fold的日期成员；公共 `quarterly_runner.py` 统一控制每个fold依次执行 `tune()` 和 `refit_and_predict()`，完成当前fold后才进入下一个fold。两个模型目录分别实现自身训练细节，不复制PIT和季度编排逻辑。

共享 `data/processed/` 保存：

```text
dataset_with_label.parquet
fold_manifest.parquet
fold_membership.parquet
```

模型专属的预处理器、调参结果、模型、季度预测和评价报告全部保存在各自的 `experiments/<model>/` 下，不进入共享data目录。

运行入口统一为 `run_models.py`；`validate_models.py` 只读验收，不触发训练。历史季度产物默认冻结，运行支持按fold保存和断点续跑。月度数据更新只追加同季度预测，进入新季度才重新寻参和训练。

### 正式预测平滑

模型训练结束后，第三层对按日期拼接的完整日频OOS预测链执行因果EWM平滑。原始预测不覆盖，分别输出：

```text
oos_predictions.parquet
oos_predictions_smoothed.parquet
```

固定参数为 `halflife=10`、`adjust=false`、`min_periods=1`，Ridge和CNN-GRU共用同一规则。参数仅使用2018Q2—2021Q4开发期预测效力延迟曲线确定，2022年以后样本及第四层回测收益不参与选择。

开发期中，CNN-GRU的Pearson预测效力峰值位于延迟13日，早期与后期子区间分别位于14日和11日；Ridge完整开发期峰值位于14日，但分阶段稳定性较弱。EWM半衰期10日对应约13.9个交易日的加权平均信息年龄，与较稳定模型的11—14日效力峰值吻合。因此固定10日，不再随季度或模型单独寻参。

平滑仅使用信号日当天及此前的原始OOS预测，是确定性后处理；它不进入模型拟合、不读取未来标签，也不改变冻结模型的训练协议哈希。

### 项目命名

当前文件夹仍为“市场状态模型”。由于正式任务是连续收益回归，后续可统一改名为“市场收益预测模型”；本阶段不修改项目根目录名称和既有路径。
