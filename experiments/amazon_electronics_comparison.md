# Amazon Electronics：LightGCN 与 SASRec 跑通实验

## 实验设置

- 数据集：Amazon Electronics 5-core，192,403 位用户、63,001 个物品、1,689,188 条交互。
- 随机种子：`2026`。
- 评估：`uni100`，即每个正样本与 100 个均匀采样的负样本共同排序；结果为采样评估，不能与全排序（`full`）指标直接比较。
- 指标：Recall、Hit、NDCG、MRR，均报告 `@10` 和 `@20`。

## 配置

| 模型 | 训练配置 | 数据划分与评估 |
| --- | --- | --- |
| LightGCN | 10 epochs；batch size 2048；learning rate 0.0002；embedding size 128；1 层图卷积；reg weight 0.0001；BPR loss | `RS: [8,1,1]`、`RO`、`uni100`；eval batch size 16384 |
| SASRec | 10 epochs；batch size 1024；learning rate 0.0005；embedding/hidden size 64；inner size 256；1 层、2 heads；dropout 0.2；最大序列长度 50；CE loss | `LS: valid_and_test`、`TO`、`uni100`；eval batch size 16384 |

两者均每 5 epoch 验证一次，`stopping_step=2`。

## 结果

### 验证集最佳结果

| 模型 | Recall@10 | Recall@20 | Hit@10 | Hit@20 | NDCG@10 | NDCG@20 | MRR@10 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGCN | 0.5300 | 0.6570 | 0.5457 | 0.6729 | 0.3445 | 0.3771 | 0.2925 | 0.3013 |
| SASRec | 0.6540 | 0.7714 | 0.6540 | 0.7714 | 0.4535 | 0.4832 | 0.3911 | 0.3993 |

### 测试集结果

| 模型 | Recall@10 | Recall@20 | Hit@10 | Hit@20 | NDCG@10 | NDCG@20 | MRR@10 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGCN | 0.5290 | 0.6564 | 0.5447 | 0.6723 | 0.3435 | 0.3763 | 0.2918 | 0.3006 |
| SASRec | 0.6220 | 0.7422 | 0.6220 | 0.7422 | 0.4261 | 0.4565 | 0.3653 | 0.3736 |

LightGCN 的总运行时间约 40 分钟；SASRec 的总运行时间约 18 分钟。

## 说明

SASRec 在本次运行的各项采样指标中更高，这与 Amazon Electronics 的时间序列交互可供序列模型建模相符。但两模型使用的切分协议不同：LightGCN 使用随机切分（RS/RO），SASRec 使用按时间的 leave-one-out 切分（LS/TO）。因此这些数值不能作为严格的模型优劣比较，只能证明两个模型均已在自定义数据集上成功完成训练和评估。
