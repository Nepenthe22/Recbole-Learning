# RecBole 推荐模型实验报告

## 一、实验基本信息

| 项目 | 内容 |
| --- | --- |
| 课程/任务 | RecBole 推荐系统入门实验 |
| 实验人 | 待填写 |
| 指导老师 | 待填写 |
| 实验日期 | 待填写 |
| 训练平台 | AutoDL |
| GPU 型号 | 训练后根据日志填写 |

## 二、实验目的

本实验使用 RecBole 框架在 MovieLens 100K 数据集上完成两种典型推荐模型的训练与评估：

1. 使用 LightGCN 学习用户—物品交互图中的协同过滤表示；
2. 使用 SASRec 根据用户历史行为序列预测下一次可能交互的物品；
3. 熟悉推荐系统从数据准备、模型训练、验证到测试的完整实验流程；
4. 理解 Recall、Hit、NDCG 和 MRR 等 Top-K 排序指标的含义。

## 三、实验环境

本实验在 AutoDL 的 NVIDIA GPU 环境中进行，使用 `d2l` Conda 环境。

| 软件/硬件 | 版本或信息 |
| --- | --- |
| 操作系统 | 待填写 |
| Python | 待填写，可执行 `python --version` 查看 |
| PyTorch | 待填写，可执行 `python -c "import torch; print(torch.__version__)"` 查看 |
| CUDA | 待填写，可执行 `nvidia-smi` 查看 |
| RecBole | 1.2.1 |
| GPU | 待填写 |

## 四、数据集与预处理

实验使用 MovieLens 100K 数据集。数据记录包含用户编号、物品编号、评分和时间戳四个字段，转换后的 RecBole atomic file 表头为：

```text
user_id:token    item_id:token    rating:float    timestamp:float
```

数据准备脚本为 `experiments/code/prepare_ml100k.py`，生成文件：

```text
experiments/data/ml-100k/ml-100k.inter
```

LightGCN 使用随机比例 `8:1:1` 划分训练集、验证集和测试集；SASRec 按时间顺序生成用户行为序列，并使用 leave-one-out 方式保留验证行为和测试行为。

## 五、实验实现与配置

两个实验共用 `experiments/code/run_experiment.py`，通过 `--model` 参数选择模型。运行器统一完成以下工作：

- 检查配置文件和数据目录；
- 检查 CUDA 和指定 NVIDIA GPU 是否可用；
- 调用 RecBole 的 `run_recbole` 完成训练与评估；
- 保存完整日志、模型 checkpoint 和 JSON 结果；
- 记录模型、数据集、GPU、配置覆盖项和评估结果。

### 5.1 LightGCN

LightGCN 将用户和物品看作二部图中的节点，通过归一化邻接矩阵进行多层邻居传播，并对各层 embedding 聚合得到最终表示。模型使用 BPR loss，使用户对正样本物品的预测分数高于负样本物品。

主要配置：

| 参数 | 设置 |
| --- | --- |
| embedding size | 64 |
| 图传播层数 | 3 |
| loss | BPR |
| 训练 epoch | 10 |
| 评估划分 | RS，8:1:1 |
| 排序方式 | RO |
| 评估模式 | full |

运行命令：

```bash
python experiments/code/run_experiment.py --model LightGCN --gpu-id 0
```

### 5.2 SASRec

SASRec 将用户历史物品序列输入 Transformer Encoder。物品 embedding 与 position embedding 相加后，通过带因果 mask 的 self-attention 建模历史行为之间的依赖关系，最后预测下一个物品。由于使用 CE loss，训练时将目标物品视为全量物品分类标签，不使用训练负采样。

主要配置：

| 参数 | 设置 |
| --- | --- |
| hidden size | 64 |
| Transformer 层数 | 2 |
| attention heads | 2 |
| 最大历史序列长度 | 50 |
| loss | CE |
| 训练 epoch | 10 |
| 评估划分 | LS，valid_and_test |
| 排序方式 | TO |
| 评估模式 | full |

运行命令：

```bash
python experiments/code/run_experiment.py --model SASRec --gpu-id 0
```

## 六、评价指标

设推荐列表前 `K` 个位置对应的命中情况为 `0/1` 序列。

- **Hit@K**：前 K 个推荐中至少命中一个真实物品则记为 1，否则为 0；
- **Recall@K**：前 K 个推荐命中的真实物品数，占用户全部真实正样本数的比例；
- **MRR@K**：第一个命中的物品排名的倒数，越早命中得分越高；
- **NDCG@K**：考虑命中位置的折损收益，越靠前的命中贡献越大。

本实验同时记录 `@10` 和 `@20`，并使用 `NDCG@10` 作为验证集模型选择指标。

## 七、实验结果

训练完成后，将 `experiments/outputs/results/` 中对应 JSON 文件的结果填写到下表。不要提前填写未经训练得到的数值。

| 模型 | Recall@10 | Hit@10 | NDCG@10 | MRR@10 | Recall@20 | Hit@20 | NDCG@20 | MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGCN | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| SASRec | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

最佳验证集结果文件：

- LightGCN：`experiments/outputs/results/lightgcn_ml-100k.json`
- SASRec：`experiments/outputs/results/sasrec_ml-100k.json`

## 八、结果分析

### 8.1 LightGCN 分析

待训练完成后填写：结合 Recall、Hit、NDCG 和 MRR，说明 LightGCN 在协同过滤任务上的表现，并分析图传播层数和 embedding 表示对结果的影响。

### 8.2 SASRec 分析

待训练完成后填写：结合序列建模特点，分析 SASRec 是否利用了用户行为顺序，并说明 position embedding、self-attention 和 CE loss 对模型效果的作用。

### 8.3 模型对比

LightGCN 主要利用用户—物品交互图中的结构信息，SASRec 主要利用用户行为的时间顺序。两者的结果差异可以从数据划分方式、行为序列信息和模型归纳偏置三个方面解释。由于两个模型的评估划分策略不同，比较时应同时报告配置，不能只比较单一指标数值。

## 九、实验结论

本实验完成了 RecBole 推荐实验的基本流程，并分别实现了图协同过滤模型 LightGCN 和序列推荐模型 SASRec。LightGCN 通过图上的邻居传播学习用户和物品表示；SASRec 通过位置 embedding 和 self-attention 建模用户历史行为序列。训练结果和最终结论将在 AutoDL 完成 GPU 训练后补充。

## 十、文件说明

```text
experiments/
├── code/
│   ├── prepare_ml100k.py
│   └── run_experiment.py
├── config/
│   ├── lightgcn_ml-100k.yaml
│   └── sasrec_ml-100k.yaml
├── data/
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── results/
└── experiment_report.md
```

## 十一、训练完成检查清单

- [ ] `nvidia-smi` 能正常显示 GPU；
- [ ] LightGCN 和 SASRec 均成功完成训练；
- [ ] 两个日志中均出现验证集和测试集结果；
- [ ] 将 JSON 结果填写到第七节；
- [ ] 补充 Python、PyTorch、CUDA 和 GPU 型号；
- [ ] 根据实际结果完成第八节分析；
- [ ] 检查报告中的“待填写”内容已全部处理。
