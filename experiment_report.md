# RecBole 推荐模型实验报告

## 一、实验基本信息

| 项目 | 内容 |
| --- | --- |
| 实验任务 | RecBole 推荐系统入门实验 |
| 实验日期 | 2026-09-29 |
| 训练平台 | AutoDL |
| Python | 3.10.8 |
| PyTorch | 2.1.2+cu118 |
| RecBole | 1.2.1 |
| 计算设备 | CUDA GPU |

## 二、实验目的

本实验使用 RecBole 在 MovieLens 100K 数据集上训练和评估 LightGCN 与 SASRec，熟悉数据加载、模型训练、验证、测试和 Top-K 推荐指标计算流程。

## 三、数据与评估设置

实验使用 RecBole 内置的 `ml-100k` 数据集，包含 943 个用户、1682 个物品和 100000 条交互。

- LightGCN 使用随机顺序和 `8:1:1` 划分，进行按用户的 full-ranking 评估；本次采用 RecBole 官方 LightGCN 默认模型参数；
- SASRec 按时间顺序构造行为序列，采用 leave-one-out 验证集和测试集，并进行 full-ranking 评估；
- 两个模型均报告 Recall、Hit、NDCG 和 MRR 的 `@10`、`@20` 指标，以验证集 `NDCG@10` 选择最优配置。

## 四、实验结果

### 4.1 两个模型的验证集最优结果

| 模型 | Recall@10 | Hit@10 | NDCG@10 | MRR@10 | Recall@20 | Hit@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGCN | 0.2297 | 0.7731 | 0.2504 | 0.4249 | 0.3448 | 0.8844 | 0.2774 | 0.4328 |
| SASRec | 0.1442 | 0.1442 | 0.0700 | 0.0477 | 0.2344 | 0.2344 | 0.0925 | 0.0537 |

上述验证集指标与参数一一对应：LightGCN 的最优指标为第一行，SASRec 的最优指标为第二行。

对应最优参数：

- LightGCN：embedding size=64，图传播层数=2，learning rate=`0.001`，regularization=`0.00001`，loss=BPR，early stopping step=10，seed=2026。
- SASRec：hidden size=64，inner size=256，Transformer layers=1，attention heads=2，max sequence length=50，dropout=0.2，weight decay=`0.000001`，learning rate=`0.0005`，train batch size=512，loss=CE，early stopping step=5，seed=2026。

### 4.2 两个模型的测试集最优结果

| 模型 | Recall@10 | Hit@10 | NDCG@10 | MRR@10 | Recall@20 | Hit@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGCN | 0.2518 | 0.8049 | 0.2967 | 0.4846 | 0.3663 | 0.8940 | 0.3149 | 0.4909 |
| SASRec | 0.1241 | 0.1241 | 0.0592 | 0.0395 | 0.2185 | 0.2185 | 0.0829 | 0.0460 |

以上测试集指标同样按模型分别对应：LightGCN 为第一行，SASRec 为第二行；两行均使用各模型在验证集上选出的最优配置，不使用测试集指标进行调参。

## 五、模型调参过程与最优配置

### 5.1 LightGCN 最优配置

本次 LightGCN 使用 RecBole 官方文档给出的默认模型参数作为基线。官方文档提供了学习率、图传播层数和正则化强度的搜索范围，但没有发布 ML-100K 的固定官方指标。本次实验先运行官方默认配置；由于其结果已经明显优于原配置，因此停止后续调参。

| 参数 | 最优设置 |
| --- | --- |
| embedding size | 64 |
| 图传播层数 | 2 |
| learning rate | 0.001 |
| regularization | 0.00001 |
| loss | BPR |
| epoch 上限 | 300 |
| early stopping step | 10 |
| seed | 2026 |

### 5.2 SASRec 最优配置

SASRec 采用逐阶段调参策略，并始终使用验证集 `NDCG@10` 选择下一阶段配置：先确定 learning rate，再确定 train batch size，随后比较 dropout 和 weight decay，最后比较 attention heads、Transformer 层数、最大序列长度和隐藏层容量。

| 参数 | 最优设置 |
| --- | --- |
| hidden size | 64 |
| inner size | 256 |
| Transformer layers | 1 |
| attention heads | 2 |
| max sequence length | 50 |
| dropout | 0.2 |
| weight decay | 0.000001 |
| learning rate | 0.0005 |
| train batch size | 512 |
| loss | CE |
| early stopping step | 5 |
| seed | 2026 |

## 六、结果分析

LightGCN 在测试集的 Recall、Hit、NDCG 和 MRR 均高于 SASRec，说明在本次 MovieLens 100K 设置下，LightGCN 对目标物品的排序位置更靠前。SASRec 的 `Recall@20` 略高，但由于其 leave-one-out 评估中每名用户只有一个目标物品，因此 Hit 与 Recall 的数值相同。需要注意，两个模型采用不同的数据划分和任务设定，指标不应被视为严格同条件下的模型排名。本实验结果主要用于验证两类推荐模型在 RecBole 中的训练、调参和评估流程。

## 七、结论

本实验完成了 LightGCN 和 SASRec 的端到端训练与评估。LightGCN 使用 RecBole 官方默认模型参数后，测试集 `NDCG@10` 达到 `0.2967`，高于原配置的 `0.1457`；SASRec 通过学习率、batch size、正则化和模型结构的分阶段调参，验证集 `NDCG@10` 从调参前的 `0.0636` 提升到 `0.0700`。本次结果说明，LightGCN 的学习率、图传播层数、正则化强度和训练轮数对 ML-100K 结果影响明显；不同切分协议下的模型指标仍不应直接进行严格横向比较。

## 八、自定义数据集：Amazon Electronics

### 8.1 数据集说明

本实验进一步使用 Amazon Electronics 评论数据集作为自定义数据集。数据来自 Amazon Electronics 商品评论，采用 5-core 版本，即保留交互次数不少于 5 次的用户和商品，以减少极端稀疏用户/商品对训练的影响。原始评论记录包含用户、商品、评分和时间戳等字段，经过清洗、字段映射和按时间排序后，转换为 RecBole 的 atomic `.inter` 格式。

处理后的数据集统计如下：

| 统计项 | 数值 |
| --- | ---: |
| 用户数 | 192,403 |
| 商品数 | 63,001 |
| 交互数 | 1,689,188 |
| 平均每用户交互数 | 8.78 |

LightGCN 使用 `user_id`、`item_id`、`rating` 和 `timestamp` 字段构建用户—商品图；SASRec 使用 `user_id`、`item_id` 和 `timestamp`，将每名用户的交互按时间升序组织为行为序列。两种模型均在该数据集上完成了训练和评估。

### 8.2 评估设置

两个模型均使用随机种子 `2026`、训练 `10` 个 epoch、`eval_batch_size=16384`、每 5 个 epoch 评估一次，并设置 `stopping_step=2`。评估模式为 `uni100`，即为每个用户采样 100 个负例进行排序评估。因此以下指标是 100 个负例采样下的结果，不等同于对全部商品进行全量排序的结果。

### 8.3 模型参数

| 参数 | LightGCN | SASRec |
| --- | --- | --- |
| train batch size | 2048 | 1024 |
| learning rate | 0.0002 | 0.0005 |
| embedding/hidden size | 128 | 64 |
| 图传播层数 / Transformer 层数 | 1 | 1 |
| inner size | — | 256 |
| attention heads | — | 2 |
| dropout | — | 0.2 |
| weight decay / reg weight | 0.0001 | 0.000001 |
| loss | BPR | CE |
| 最大序列长度 | — | 50 |
| 数据划分 | RS `[8,1,1]`，RO | LS，TO |
| 评估模式 | `uni100` | `uni100` |

### 8.4 实验指标

验证集结果如下：

| 模型 | Recall@10 | Recall@20 | Hit@10 | Hit@20 | NDCG@10 | NDCG@20 | MRR@10 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGCN | 0.5300 | 0.6570 | 0.5457 | 0.6729 | 0.3445 | 0.3771 | 0.2925 | 0.3013 |
| SASRec | 0.6540 | 0.7714 | 0.6540 | 0.7714 | 0.4535 | 0.4832 | 0.3911 | 0.3993 |

测试集结果如下：

| 模型 | Recall@10 | Recall@20 | Hit@10 | Hit@20 | NDCG@10 | NDCG@20 | MRR@10 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGCN | 0.5290 | 0.6564 | 0.5447 | 0.6723 | 0.3435 | 0.3763 | 0.2918 | 0.3006 |
| SASRec | 0.6220 | 0.7422 | 0.6220 | 0.7422 | 0.4261 | 0.4565 | 0.3653 | 0.3736 |

在本次 `uni100` 采样评估下，SASRec 的各项指标高于 LightGCN，说明序列模型在该 Amazon Electronics 数据上的短期行为预测中表现较好。但 LightGCN 使用随机划分 `RS/RO`，SASRec 使用按时间划分 `LS/TO`，两者的任务协议并不完全一致，因此该结果主要用于记录自定义数据集上的跑通结果，不能作为严格公平的模型排名结论。
