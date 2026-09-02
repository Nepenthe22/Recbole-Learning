# LightGCN 实验记录

## 1. 实验目的

在 `ml-100k` 上跑通 RecBole 的 LightGCN，记录验证集和测试集的排序指标，熟悉从数据准备、训练、验证到测试的完整流程。

## 2. 实验配置

配置文件：[`../config/ml-100k.yaml`](../config/ml-100k.yaml)

| 项目 | 设置 |
| --- | --- |
| 模型 | LightGCN |
| 数据集 | ml-100k |
| 划分 | RS，8:1:1 |
| 评估模式 | 按用户分组、全量排序 |
| 训练损失 | BPR |
| embedding size | 64 |
| LightGCN 层数 | 3 |
| epoch | 10 |
| seed | 2026 |

## 3. 结果

运行后，将 `outputs/results/lightgcn_ml-100k.json` 中的结果摘录到下表。JSON 保留完整的 RecBole 返回值，避免手工抄写时丢失信息。

| 数据集 | Recall@10 | Hit@10 | NDCG@10 | MRR@10 | Recall@20 | Hit@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ml-100k | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

## 4. 运行检查

- [ ] 数据文件为 `data/ml-100k/ml-100k.inter`，表头包含字段类型；
- [ ] 日志中出现 `best valid result` 和 `test result`；
- [ ] 生成 `outputs/results/lightgcn_ml-100k.json`；
- [ ] 如启用保存，`outputs/checkpoints/` 中存在 LightGCN checkpoint；
- [ ] 记录 Python、PyTorch、RecBole 版本以及 CPU/GPU 环境。

## 5. 学习要点

LightGCN 的核心是用户—物品二部图上的邻居聚合：初始用户和物品 embedding 在归一化邻接矩阵上进行多层传播，再对各层表示做加权平均。训练阶段使用 BPR，让正样本的预测分数高于负样本；评估阶段按用户对候选物品做排序，并计算 Recall、Hit、NDCG 和 MRR。
