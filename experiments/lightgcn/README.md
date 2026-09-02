# LightGCN 实验

这个目录只负责任务大纲中的第一个任务：使用 RecBole 在 MovieLens 100K（`ml-100k`）上运行 LightGCN。
RecBole 已经提供了 LightGCN 模型实现，本项目的代码负责数据准备、实验配置、运行入口和结果归档，不修改第三方库源码。

## 目录说明

```text
experiments/lightgcn/
├── code/                  # 可执行脚本
│   ├── prepare_ml100k.py  # 下载并转换 MovieLens 100K
│   └── run_lightgcn.py    # LightGCN 实验入口
├── config/                # RecBole 配置
│   └── ml-100k.yaml
├── data/                  # 本任务数据，运行准备脚本后生成
├── docs/                  # 实验记录
│   └── experiment_record.md
└── outputs/               # 运行产物
    ├── checkpoints/       # RecBole 模型 checkpoint
    ├── logs/              # 运行日志
    └── results/           # JSON 结果摘要
```

## 运行

在 AutoDL 中选择带 NVIDIA GPU 的实例，并在仓库根目录执行：

```bash
conda activate d2l
nvidia-smi
python experiments/lightgcn/code/prepare_ml100k.py
python experiments/lightgcn/code/run_lightgcn.py --gpu-id 0
```

如果换到其他环境，再按该环境的 PyTorch/CUDA 情况安装
`experiments/lightgcn/requirements.txt` 中的依赖即可；`d2l` 环境无需重复安装 RecBole。

运行脚本默认强制检查 CUDA，并在启动日志中记录 NVIDIA 显卡名称。显卡未挂载、编号不存在或 PyTorch 不是 CUDA 版本时，脚本会在训练前直接报错，不会悄悄改用 CPU。

默认设置为 10 个 epoch、64 维 embedding、3 层 LightGCN 传播，使用按用户分组的全量排序评估，报告 `Recall`、`Hit`、`NDCG` 和 `MRR` 的 `@10`、`@20` 结果。

多卡环境可以指定可见 GPU，例如：

```bash
python experiments/lightgcn/code/run_lightgcn.py \
  --gpu-id 0,1 \
  --epochs 20 \
  --seed 2026 \
  --run-name lightgcn_ml100k_20epoch
```

运行结束后，重点查看：

- `outputs/logs/<run-name>.log`：完整运行日志；
- `outputs/results/<run-name>.json`：验证集最优结果和测试集结果；
- `outputs/checkpoints/`：保存的最佳模型。

如果只想检查脚本参数和环境，而暂时不保存 checkpoint，可使用 `--no-save`。结果文件中的指标以 RecBole 实际输出为准，不在代码中硬编码实验数值。
