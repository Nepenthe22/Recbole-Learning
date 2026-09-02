# SASRec 实验记录

## 1. 实验目的

在 `ml-100k` 上跑通 SASRec，理解时间排序、历史序列、位置 embedding、Transformer Encoder 和 CE loss。

## 2. 实验配置

配置文件：[`../config/ml-100k.yaml`](../config/ml-100k.yaml)

| 项目 | 设置 |
| --- | --- |
| 模型 | SASRec |
| 数据集 | ml-100k |
| 序列最大长度 | 50 |
| 划分 | LS，valid_and_test |
| 排序 | TO，按时间排序 |
| 评估模式 | 按用户分组、全量排序 |
| 损失 | CE |
| hidden size | 64 |
| Transformer 层数/头数 | 2 / 2 |
| epoch | 10 |
| seed | 2026 |

## 3. 结果

运行后，从 `outputs/results/sasrec_ml-100k.json` 摘录结果。

| 数据集 | Recall@10 | Hit@10 | NDCG@10 | MRR@10 | Recall@20 | Hit@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ml-100k | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

## 4. 学习要点

RecBole 会按时间构造用户历史序列，SASRec 为物品 embedding 加上 position embedding，再通过带因果 mask 的 self-attention 建模历史行为关系。CE loss 将目标物品视为全量物品分类标签，因此训练阶段不需要负采样。
