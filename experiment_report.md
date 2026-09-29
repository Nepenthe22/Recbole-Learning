# RecBole 推荐模型实验报告

## 一、实验基本信息

| 项目 | 内容 |
| --- | --- |
| 实验任务 | RecBole 实验 |
| Python | 3.10.8 |
| PyTorch | 2.1.2+cu118 |
| RecBole | 1.2.1 |
| 计算设备 | CUDA GPU 2080Ti |

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

## 七、自定义数据集：Amazon Electronics

### 7.1 数据集说明

本实验进一步使用 Amazon Electronics 评论数据集作为自定义数据集。数据来自 Amazon Electronics 商品评论，采用 5-core 版本，即保留交互次数不少于 5 次的用户和商品，以减少极端稀疏用户/商品对训练的影响。原始评论记录包含用户、商品、评分和时间戳等字段，经过清洗、字段映射和按时间排序后，转换为 RecBole 的 atomic `.inter` 格式。

处理后的数据集统计如下：

| 统计项 | 数值 |
| --- | ---: |
| 用户数 | 192,403 |
| 商品数 | 63,001 |
| 交互数 | 1,689,188 |
| 平均每用户交互数 | 8.78 |

LightGCN 使用 `user_id`、`item_id`、`rating` 和 `timestamp` 字段构建用户—商品图；SASRec 使用 `user_id`、`item_id` 和 `timestamp`，将每名用户的交互按时间升序组织为行为序列。两种模型均在该数据集上完成了训练和评估。

### 7.2 评估设置

两个模型均使用随机种子 `2026`、训练 `10` 个 epoch、`eval_batch_size=16384`、每 5 个 epoch 评估一次，并设置 `stopping_step=2`。评估模式为 `uni100`，即为每个用户采样 100 个负例进行排序评估。因此以下指标是 100 个负例采样下的结果，不等同于对全部商品进行全量排序的结果。

### 7.3 模型参数

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

### 7.4 实验指标

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

## 八、评价指标说明

本实验主要使用 Recall、Hit、NDCG 和 MRR 衡量推荐结果质量。四个指标都在 Top-K 推荐列表上计算，但关注点不同。

### 8.1 Recall

Recall@K 表示用户真实相关物品中，有多少被推荐列表前 K 个位置找回，反映推荐结果对正确答案的覆盖程度。对于每名用户只有一个测试目标的设置，Recall@K 可以理解为目标物品是否出现在前 K 个推荐中；当每名用户存在多个测试目标时，Recall 还反映找回了多少个目标物品。

设用户集合为 $U$，用户 $u$ 的真实相关物品集合为 $R_u$，模型给出的前 K 个推荐集合为 $P_u^K$，则：

$$
\mathrm{Recall@K} = \frac{1}{|U|}\sum_{u\in U}\frac{|R_u\cap P_u^K|}{|R_u|}
$$

### 8.2 Hit

Hit@K 判断前 K 个推荐中是否至少命中一个真实相关物品，再对所有用户取平均，反映有多少用户至少获得了一次有效推荐。Hit 更关注“是否命中”，不区分同一用户命中一个还是多个相关物品。

使用指示函数 $\mathbb{I}(\cdot)$ 表示是否满足条件，则：

$$
\mathrm{Hit@K} = \frac{1}{|U|}\sum_{u\in U}\mathbb{I}\left(R_u\cap P_u^K\neq\varnothing\right)
$$

### 8.3 NDCG

NDCG@K 不仅关注是否命中，还会根据相关物品在推荐列表中的位置进行折损。正确物品排名越靠前，NDCG 得分越高；如果相关物品排在较后位置，得分会降低。因此 NDCG 更适合衡量整体排序质量。

令 $rel_{u,i}$ 表示用户 $u$ 的第 $i$ 个推荐结果的相关性，通常采用二值相关性（命中为 1，否则为 0），则：

$$
\mathrm{DCG@K}(u)=\sum_{i=1}^{K}\frac{2^{rel_{u,i}}-1}{\log_2(i+1)}
$$

将实际排序结果的 DCG 与理想排序结果的 DCG（IDCG）相比，可得：

$$
\mathrm{NDCG@K}=\frac{1}{|U|}\sum_{u\in U}\frac{\mathrm{DCG@K}(u)}{\mathrm{IDCG@K}(u)}
$$

### 8.4 MRR

MRR@K 取用户在前 K 个推荐中第一个相关物品排名的倒数，再对用户求平均。例如第一个相关物品排在第 1 位时得分为 1，排在第 5 位时得分为 0.2。MRR 主要关注用户看到第一个正确结果需要等待多久。

令 $rank_u$ 为用户 $u$ 在前 K 个推荐中第一个相关物品的排名；如果前 K 个结果没有命中，则令 $rank_u=\infty$，其倒数记为 0，则：

$$
\mathrm{MRR@K}=\frac{1}{|U|}\sum_{u\in U}\frac{1}{rank_u}
$$

其中，`K` 表示只查看推荐列表的前 K 个结果，例如 `Recall@10` 只考虑前 10 个推荐，`@20` 则考虑前 20 个推荐。通常随着 K 增大，Recall 和 Hit 不会降低，因为更长的推荐列表有更大机会包含真实物品；NDCG 和 MRR 则进一步反映正确物品在列表中的具体位置。

对于每名用户只有一个测试目标的 leave-one-out 设置，Recall@K 和 Hit@K 在理论上会非常接近，甚至相等；当每名用户存在多个测试目标、采用不同数据划分方式或使用不同指标实现时，两者可能出现差异。

## 九、模型源码

### 9.1 LightGCN 源码结构与核心流程

RecBole 中 LightGCN 位于 `recbole/model/general_recommender/lightgcn.py`，属于一般推荐模型。模型从交互数据中读取用户 ID 和物品 ID，并将用户—物品交互关系构造成二部图。源码首先为用户和物品建立可学习的 embedding，然后通过图传播层在相邻节点之间传递 embedding 信息。

LightGCN 的核心特点是只保留 embedding 传播和邻居聚合，不使用特征变换矩阵和非线性激活函数。第 $l+1$ 层的节点表示可以概括为：

$$
\mathbf{e}^{(l+1)}_v=\sum_{u\in\mathcal{N}(v)}\frac{1}{\sqrt{|\mathcal{N}(v)|}\sqrt{|\mathcal{N}(u)|}}\mathbf{e}^{(l)}_u
$$

源码中通常将初始 embedding 和各层传播结果保存下来，再进行平均或加权融合，得到最终的用户和物品表示。这样可以同时利用用户自身的协同表示和多跳邻居信息。`n_layers` 决定图传播的层数，层数越大，模型能够利用更远的协同关系，但也可能带来过平滑问题。

训练阶段，LightGCN 使用 BPR 损失。对用户 $u$、正样本物品 $i$ 和负样本物品 $j$，模型先计算用户与物品 embedding 的内积作为偏好分数：

$$
\hat{y}_{ui}=\mathbf{e}_u^\top\mathbf{e}_i
$$

然后优化正样本分数高于负样本分数的差值：

$$
\mathcal{L}_{BPR}=-\sum\log\sigma(\hat{y}_{ui}-\hat{y}_{uj})+\lambda\lVert\Theta\rVert^2
$$

其中，负样本由 RecBole 的采样器产生，正则化项用于限制 embedding 的规模。评估阶段，源码使用用户 embedding 与候选物品 embedding 的矩阵乘法得到所有候选物品分数，再通过 Top-K 排序计算 Recall、Hit、NDCG 和 MRR。

### 9.2 SASRec 源码结构与核心流程

RecBole 中 SASRec 位于 `recbole/model/sequential_recommender/sasrec.py`，属于序列推荐模型。与 LightGCN 使用全局用户—物品图不同，SASRec 输入的是用户历史交互序列。源码首先根据时间顺序取出用户最近的行为，并截断或补齐到 `MAX_ITEM_LIST_LENGTH`，本实验设置为 50。

每个物品 ID 先经过 item embedding，并加上位置 embedding，以区分不同历史位置。随后，序列表示进入由多头自注意力和前馈网络组成的 Transformer block。因果 attention mask 保证当前位置只能使用当前及之前的行为，不能看到未来物品，从而避免训练时的信息泄漏。模型通过自注意力学习序列中不同物品之间的依赖关系，例如某些商品组合、相邻购买行为和较长期的兴趣变化。

对序列位置 $t$，自注意力的基本计算形式为：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}+M\right)V
$$

其中 $M$ 是因果 mask，被遮挡的未来位置不会参与注意力计算。经过 Transformer 层后，源码取目标位置的隐藏状态与候选物品 embedding 进行匹配，得到下一个物品的预测分数。

本实验的 SASRec 使用交叉熵损失。训练时，模型根据历史序列预测下一个真实交互物品，优化目标物品在候选物品中的概率。与 LightGCN 的 BPR 成对排序损失相比，SASRec 的 CE 损失直接进行分类式预测，因此需要明确序列中的目标位置和 padding mask。源码还会使用 padding mask 忽略补齐位置，避免无效位置影响梯度。

### 9.3 两个模型源码的主要差异

| 对比项 | LightGCN | SASRec |
| --- | --- | --- |
| 模型类型 | 一般推荐模型 | 序列推荐模型 |
| 核心输入 | 用户—物品交互图 | 按时间排列的用户行为序列 |
| 主要建模对象 | 用户与物品的协同关系 | 用户行为的顺序依赖 |
| 核心结构 | 二部图 embedding 传播 | Transformer 自注意力 |
| 训练目标 | BPR 成对排序 | CE 下一物品预测 |
| 关键超参数 | embedding size、`n_layers`、`reg_weight` | hidden size、`n_heads`、`MAX_ITEM_LIST_LENGTH`、dropout |
| 适合的行为假设 | 用户相似性和物品共现关系稳定 | 近期行为顺序包含较强预测信息 |

从源码执行流程看，LightGCN 的主要计算开销来自图传播和用户—物品分数计算；SASRec 的主要计算开销来自序列 Transformer，尤其是序列长度、attention heads 和 hidden size。两者虽然都最终输出用户对物品的排序分数，但输入组织方式、损失函数和模型假设不同，因此实验比较时需要同时报告数据划分和训练目标。
