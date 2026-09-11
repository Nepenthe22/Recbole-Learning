# RecBole 推荐模型实验报告

## 一、实验基本信息

| 项目 | 内容 |
| --- | --- |
| 实验任务 | RecBole 推荐系统入门实验 |
| 实验日期 | 2026-09-03 |
| 训练平台 | AutoDL |
| Python | 3.10.8 |
| PyTorch | 2.1.2+cu118 |
| RecBole | 1.2.1 |
| 计算设备 | CUDA GPU，日志显示约 11.63 GiB 可见显存 |

## 二、实验目的

本实验使用 RecBole 在 MovieLens 100K 数据集上训练和评估两种典型推荐模型：

1. 使用 LightGCN 学习用户—物品交互图中的协同过滤表示；
2. 使用 SASRec 根据用户历史行为序列预测下一次交互；
3. 熟悉数据加载、训练、验证和测试流程；
4. 理解 Recall、Hit、NDCG 和 MRR 等 Top-K 排序指标。

## 三、数据与评估设置

实验使用 RecBole 内置的 `ml-100k` 示例数据集。数据包含 943 个用户、1682 个物品和 100000 条交互。

- LightGCN 使用随机顺序和 `8:1:1` 划分，按用户进行 full-ranking 评估；
- SASRec 按时间排序构造行为序列，使用 leave-one-out 的验证集和测试集，并进行 full-ranking 评估；
- 两个模型均报告 Recall、Hit、NDCG、MRR 的 `@10` 和 `@20` 指标，以 `NDCG@10` 选择最佳模型。

## 四、实现与配置

两个模型共用 `experiments/code/run_experiment.py`。脚本负责检查配置、数据路径和 CUDA，再调用 RecBole 的 `run_recbole`。RecBole 按默认行为将文本日志写入根目录 `log/`，TensorBoard 文件写入 `log_tensorboard/`，checkpoint 写入 `saved/`。

### 4.1 LightGCN

| 参数 | 设置 |
| --- | --- |
| embedding size | 32 |
| 图传播层数 | 2 |
| loss | BPR |
| epoch 上限 | 100 |
| learning rate | 0.0002 |
| regularization | 0.0001 |
| early stopping step | 20 |
| seed | 2026 |

### 4.2 SASRec

| 参数 | 设置 |
| --- | --- |
| embedding/hidden size | 64 |
| Transformer 层数 | 2 |
| attention heads | 2 |
| 最大序列长度 | 50 |
| dropout | 0.2 |
| loss | CE |
| epoch 上限 | 60 |
| learning rate | 0.0005 |
| early stopping step | 8 |
| seed | 2026 |

## 五、实验结果

### 5.1 验证集最佳结果

| 模型 | Recall@10 | Hit@10 | NDCG@10 | MRR@10 | Recall@20 | Hit@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGCN | 0.1125 | 0.5514 | 0.1243 | 0.2349 | 0.1786 | 0.6957 | 0.1402 | 0.2449 |
| SASRec | 0.1357 | 0.1357 | 0.0636 | 0.0420 | 0.2291 | 0.2291 | 0.0865 | 0.0480 |

### 5.2 测试集结果

| 模型 | Recall@10 | Hit@10 | NDCG@10 | MRR@10 | Recall@20 | Hit@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGCN | 0.1246 | 0.5779 | 0.1457 | 0.2822 | 0.1858 | 0.6903 | 0.1572 | 0.2900 |
| SASRec | 0.1198 | 0.1198 | 0.0577 | 0.0388 | 0.1930 | 0.1930 | 0.0760 | 0.0438 |

## 六、结果分析

LightGCN 在本次测试中的 NDCG 和 MRR 明显高于 SASRec，说明其命中物品通常出现在更靠前的位置。SASRec 在 `Recall@20` 上略高，但其 leave-one-out 评估中每名用户只有一个目标物品，因此 Hit 与 Recall 数值相同。

两种模型采用不同的数据划分方式和任务设定，指标不能视为严格的同条件模型排名。本次结果主要用于验证两类推荐模型均可在 RecBole 中完成端到端训练和评估。

## 七、运行命令

```bash
python experiments/code/run_experiment.py --model LightGCN --gpu-id 0
python experiments/code/run_experiment.py --model SASRec --gpu-id 0
```

原始日志、TensorBoard 文件和 checkpoint 保存在服务器默认运行目录中，不纳入 GitHub 版本控制。

## 八、SASRec 四阶段调参

调参固定使用 `seed=2026`、`stopping_step=8`，以验证集 `NDCG@10` 选择进入下一阶段的配置。测试集指标只用于最终比较，不参与参数选择。

### 8.1 第一阶段：学习率

固定 `n_layers=2`、`n_heads=2`、`hidden_size=64`、`train_batch_size=2048`、dropout `0.2` 和最大序列长度 `50`。

| learning rate | 最佳 epoch | Valid Recall@10 | Valid NDCG@10 | Valid MRR@10 | Valid Recall@20 | Valid NDCG@20 | Valid MRR@20 | Test Recall@10 | Test NDCG@10 | Test MRR@10 | Test Recall@20 | Test NDCG@20 | Test MRR@20 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0003 | 19 | 0.1410 | 0.0657 | 0.0431 | 0.2333 | 0.0889 | 0.0494 | 0.1177 | 0.0580 | 0.0399 | 0.1941 | 0.0771 | 0.0450 |
| **0.0005** | **12** | **0.1485** | **0.0665** | 0.0422 | 0.2248 | 0.0856 | 0.0473 | **0.1273** | **0.0615** | **0.0418** | 0.2004 | 0.0797 | 0.0467 |
| 0.0007 | 17 | 0.1400 | 0.0634 | 0.0406 | 0.2312 | 0.0863 | 0.0468 | 0.1273 | 0.0559 | 0.0347 | **0.2227** | **0.0798** | 0.0411 |
| 0.0010 | 5 | 0.1304 | 0.0627 | 0.0423 | 0.2322 | 0.0882 | 0.0491 | 0.1124 | 0.0563 | 0.0391 | 0.1962 | 0.0775 | 0.0450 |

`learning_rate=0.0005` 的验证集 `NDCG@10` 最高，因此作为第二阶段基线。
