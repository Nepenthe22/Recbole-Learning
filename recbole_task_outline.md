# RecBole 入门任务纲要

## 任务背景

导师布置的任务主要围绕 RecBole 推荐系统框架展开，目标是初步掌握推荐系统实验的完整流程：模型运行、数据接入、评测指标理解和源码阅读。

本任务适合按照“先跑通，再理解，再总结”的顺序推进。不要一开始就陷入源码细节，先让 LightGCN 和 SASRec 在内置数据集上跑起来，再逐步深入数据格式、评测逻辑和模型实现。

## 总体目标

1. 使用 RecBole 跑通 LightGCN 模型，理解协同过滤场景的基本实验流程。
2. 使用 RecBole 跑通 SASRec 模型，理解序列推荐场景的基本实验流程。
3. 尝试使用自定义数据加载方式，将第三方数据集接入 RecBole。
4. 阅读 RecBole 评测指标源码，理解 Recall、Hit、NDCG、MRR 的计算方式。
5. 阅读 LightGCN 和 SASRec 的模型源码，理解 embedding、GNN、Transformer 和 loss 的基本实现。
6. 最终整理一份实验记录或学习报告，用于向导师汇报。

## 第一阶段：跑通内置数据集

推荐先使用 RecBole 自带或官方支持的数据集，例如 `ml-100k`。该数据集规模较小，适合快速验证代码和流程。

### 1. 安装 RecBole

```bash
pip install recbole
```

如果环境中已有 PyTorch，需要确认 PyTorch 版本和 CUDA 是否匹配。

### 2. 跑通 LightGCN

LightGCN 属于协同过滤模型，重点是学习用户和物品在交互图上的 embedding。

示例代码：

```python
from recbole.quick_start import run_recbole

run_recbole(
    model='LightGCN',
    dataset='ml-100k',
    config_dict={
        'epochs': 10,
        'metrics': ['Recall', 'Hit', 'NDCG', 'MRR'],
        'topk': [10, 20],
        'valid_metric': 'NDCG@10',
    }
)
```

需要记录：

- 使用的数据集
- 模型名称
- 训练轮数
- 验证集和测试集上的 Recall、Hit、NDCG、MRR
- 是否成功生成日志和 checkpoint

### 3. 跑通 SASRec

SASRec 属于序列推荐模型，核心任务是根据用户历史交互序列预测下一个可能交互的物品。

示例代码：

```python
from recbole.quick_start import run_recbole

run_recbole(
    model='SASRec',
    dataset='ml-100k',
    config_dict={
        'epochs': 10,
        'metrics': ['Recall', 'Hit', 'NDCG', 'MRR'],
        'topk': [10, 20],
        'valid_metric': 'NDCG@10',
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full'
        }
    }
)
```

需要注意：

- SASRec 依赖用户行为序列，因此需要时间顺序。
- `order: TO` 表示按照时间排序。
- `LS` 是 leave-one-out 风格的序列划分，适合序列推荐。

## 第二阶段：自定义数据集接入

RecBole 推荐使用 atomic file 格式接入自定义数据。初学阶段不建议直接重写 dataloader，除非任务明确要求。

### 1. 数据目录结构

推荐结构：

```text
dataset/
  mydata/
    mydata.inter
```

其中目录名、数据集名和文件名前缀需要保持一致。

### 2. 最小交互文件格式

`mydata.inter` 示例：

```text
user_id:token	item_id:token	timestamp:float
u1	i1	100
u1	i2	200
u1	i3	300
u2	i2	100
u2	i4	200
```

字段说明：

- `user_id:token`：用户 ID，离散 token 类型。
- `item_id:token`：物品 ID，离散 token 类型。
- `timestamp:float`：时间戳，序列推荐通常需要。

### 3. 配置文件示例

`mydata.yaml`：

```yaml
data_path: ./dataset

USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
TIME_FIELD: timestamp

load_col:
    inter: [user_id, item_id, timestamp]

metrics: [Recall, Hit, NDCG, MRR]
topk: [10]
valid_metric: NDCG@10
```

LightGCN 可使用：

```yaml
eval_args:
    split: {'RS': [8,1,1]}
    group_by: user
    order: RO
    mode: full

train_neg_sample_args:
    distribution: uniform
    sample_num: 1
```

SASRec 可使用：

```yaml
eval_args:
    split: {'LS': 'valid_and_test'}
    group_by: user
    order: TO
    mode: full

MAX_ITEM_LIST_LENGTH: 50
loss_type: CE
train_neg_sample_args: ~
```

### 4. 常见问题

- 数据集目录名、文件名前缀和运行时传入的 dataset 名必须一致。
- `.inter` 文件表头必须带字段类型，例如 `user_id:token`。
- SASRec 需要时间字段，并且需要按时间排序。
- 用户交互过少会导致序列推荐样本不足。
- 如果使用 benchmark 文件，需要注意 train、valid、test 文件划分方式。
- 不要一开始就修改 RecBole 源码，优先检查配置和数据格式。

## 第三阶段：评测指标源码阅读

源码位置：

```text
recbole/evaluator/metrics.py
```

重点理解 ranking 指标的输入：

```text
pos_index: shape = [用户数, max_topk]
```

可以把它理解成一个 0/1 矩阵，表示推荐列表中每个位置是否命中了真实正样本。

### 1. Hit@K

含义：前 K 个推荐结果中，只要有一个命中，就记为 1，否则为 0。

直观理解：

```text
推荐命中情况: [0, 1, 0, 0, 0]
Hit@5 = 1
```

### 2. Recall@K

含义：前 K 个推荐结果命中的正样本数，占该用户所有真实正样本数的比例。

公式：

```text
Recall@K = 前 K 个命中的正样本数量 / 用户真实正样本总数
```

### 3. MRR@K

含义：第一个命中物品排名位置的倒数。

例子：

```text
推荐命中情况: [0, 1, 0, 1, 0]
第一个命中在第 2 位
MRR@5 = 1 / 2
```

### 4. NDCG@K

含义：考虑命中位置的排序质量。命中越靠前，贡献越大。

核心思想：

```text
DCG = sum(命中值 / log2(rank + 1))
NDCG = DCG / IDCG
```

其中 IDCG 表示理想排序下的 DCG。

### 5. 建议手算例子

```text
推荐列表前 5 命中情况: [0, 1, 0, 1, 0]
真实正样本数: 3

Hit@5 = 1
Recall@5 = 2 / 3
MRR@5 = 1 / 2
DCG@5 = 1 / log2(3) + 1 / log2(5)
NDCG@5 = DCG@5 / IDCG@5
```

## 第四阶段：LightGCN 源码阅读

源码位置：

```text
recbole/model/general_recommender/lightgcn.py
```

阅读重点：

1. 模型初始化
   - 从 config 中读取 embedding size、层数、正则权重等参数。
   - 从 dataset 中获取用户数、物品数和交互矩阵。

2. embedding 定义
   - `user_embedding`
   - `item_embedding`

3. 图结构构建
   - 根据用户-物品交互矩阵构造二部图。
   - 对邻接矩阵做归一化，通常是 `D^-0.5 A D^-0.5`。

4. 图传播
   - 将 user embedding 和 item embedding 拼接。
   - 每一层通过 sparse matrix multiplication 在图上传播。
   - 最后对不同层的 embedding 求平均。

5. 预测方式
   - 用户 embedding 和物品 embedding 做点积。

6. loss 计算
   - 通常使用 BPR loss。
   - 对用户、正样本物品、负样本物品的 embedding 加正则。

需要重点回答的问题：

- LightGCN 为什么不使用复杂的非线性变换？
- 用户和物品 embedding 是如何在图上传播的？
- 正样本和负样本分别从哪里来？
- BPR loss 的目标是什么？

## 第五阶段：SASRec 源码阅读

源码位置：

```text
recbole/model/sequential_recommender/sasrec.py
```

阅读重点：

1. 模型初始化
   - 读取 hidden size、层数、attention heads、dropout 等参数。

2. embedding 定义
   - `item_embedding`
   - `position_embedding`

3. 序列输入
   - 输入是用户历史物品序列。
   - RecBole 会将历史行为整理为 `item_id_list` 和 `item_length`。

4. Transformer 调用
   - item embedding 加 position embedding。
   - 构造 attention mask，避免看到不该看的未来信息。
   - 调用 `TransformerEncoder` 得到序列表示。

5. 序列表示抽取
   - 通常取最后一个有效位置的 hidden state 作为用户当前兴趣表示。

6. loss 计算
   - 支持 BPR loss 和 CE loss。
   - CE loss 中，序列输出会与所有 item embedding 做匹配，然后做多分类交叉熵。

需要重点回答的问题：

- SASRec 为什么需要 position embedding？
- self-attention 如何建模用户历史行为之间的关系？
- 最后一个有效位置的 hidden state 表示什么？
- CE loss 和 BPR loss 有什么区别？

## 第六阶段：最终交付材料

建议整理一份学习报告或实验记录，结构如下：

```text
1. 实验环境
Python 版本、PyTorch 版本、RecBole 版本、CPU/GPU 环境。

2. 数据集
内置数据集 ml-100k，以及自定义数据集的字段说明。

3. LightGCN 实验结果
记录 Recall@10、Hit@10、NDCG@10、MRR@10。

4. SASRec 实验结果
记录 Recall@10、Hit@10、NDCG@10、MRR@10。

5. 自定义数据加载过程
说明 atomic file 格式、配置文件、遇到的问题和解决方式。

6. 指标源码理解
解释 Recall、Hit、NDCG、MRR，并给出手算例子。

7. 模型源码理解
分别总结 LightGCN 和 SASRec 的输入、核心模块、forward 流程和 loss。

8. 总结与下一步计划
说明目前掌握了什么，以及后续可以继续做什么。
```

## 给导师的阶段性回复模板

```text
老师您好，我已经初步完成了 RecBole 的入门任务。

1. 我使用 ml-100k 数据集分别跑通了 LightGCN 和 SASRec，并记录了 Recall、Hit、NDCG、MRR 等指标结果。
2. 我尝试了自定义数据集接入，主要采用 RecBole 的 atomic file 格式，将第三方交互数据整理为 user_id、item_id、timestamp 三列，并配置 load_col、USER_ID_FIELD、ITEM_ID_FIELD、TIME_FIELD 后完成加载。
3. 我阅读了 recbole/evaluator/metrics.py，理解了 Hit、Recall、NDCG、MRR 的计算逻辑，并用一个小例子手算验证。
4. 我阅读了 LightGCN 和 SASRec 的源码。LightGCN 主要是构造用户-物品图并进行多层邻接矩阵传播，使用 BPR loss；SASRec 使用 item embedding 和 position embedding，经 TransformerEncoder 得到序列表示，再用 BPR 或 CE 计算 loss。

目前我对 RecBole 的基本使用、数据格式、评测流程和两个经典模型的实现方式有了初步理解。下一步我想继续深入比较不同模型在同一数据集上的表现，并尝试在自己的研究数据上复现实验。
```

## 当前最优先行动

先完成以下三个动作：

1. 跑通 LightGCN on `ml-100k`。
2. 跑通 SASRec on `ml-100k`。
3. 把两个模型的日志和指标结果填入实验记录。

完成这三步后，再继续做自定义数据和源码阅读，会更容易理解 RecBole 的整体流程。
