# SASRec 实验

这个目录对应任务大纲中的 SASRec 序列推荐实验，目录结构与 LightGCN 保持一致。
SASRec 使用用户历史交互序列预测下一个物品，因此配置采用时间排序（`TO`）和 leave-one-out（`LS`）评估，并使用 CE loss。

## 目录说明

```text
experiments/sasrec/
├── code/run_sasrec.py       # CUDA 检查、训练入口、日志和结果归档
├── config/ml-100k.yaml      # SASRec 序列模型配置
├── data/                    # ml-100k atomic file
├── docs/experiment_record.md
└── outputs/                 # checkpoints、logs、results
```

## AutoDL NVIDIA GPU 运行

```bash
conda activate d2l
nvidia-smi

# 将数据直接准备到本实验目录
python experiments/lightgcn/code/prepare_ml100k.py \
  --data-path experiments/sasrec/data

python experiments/sasrec/code/run_sasrec.py --gpu-id 0
```

运行脚本默认强制检查 CUDA，并记录 NVIDIA 显卡名称。多卡环境可使用 `--gpu-id 0,1`。
训练结果分别写入 `outputs/checkpoints/`、`outputs/logs/` 和 `outputs/results/`。
