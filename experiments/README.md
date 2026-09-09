# RecBole 推荐模型实验

本目录统一管理 LightGCN 和 SASRec 两个实验。两个模型共享 GPU 检查、训练入口和评价指标，模型差异保存在各自的 YAML 配置中。

## 目录结构

```text
experiments/
├── code/
│   └── run_experiment.py       # 通过 --model 选择模型
├── config/
│   ├── lightgcn_ml-100k.yaml   # LightGCN 配置
│   └── sasrec_ml-100k.yaml     # SASRec 配置
├── data/                       # 自定义数据集，不提交数据文件
├── experiment_report.md        # 实验结果与分析
└── requirements.txt            # Python 依赖
```

RecBole 使用仓库根目录下的默认运行产物目录：

- `log/`：文本训练日志；
- `log_tensorboard/`：TensorBoard 事件文件；
- `saved/`：模型 checkpoint。

这些目录均为本地或服务器运行产物，不提交到 GitHub。

## AutoDL 环境

如果基础环境的 Python 版本不兼容 RecBole，建议创建 Python 3.10 或 3.11 环境：

```bash
conda create -n recbole python=3.10 -y
conda activate recbole
python -m pip install -r experiments/requirements.txt
```

运行前确认 CUDA 可用：

```bash
nvidia-smi
```

## 运行实验

从仓库根目录执行：

```bash
python experiments/code/run_experiment.py --model LightGCN --gpu-id 0
python experiments/code/run_experiment.py --model SASRec --gpu-id 0
```

运行器默认使用 RecBole 内置的 `ml-100k` 数据集，并强制检查 CUDA，避免意外退回 CPU。使用其他数据集时，需要通过 `--data-path` 指定 `<data-path>/<dataset>/` 的父目录。

可使用 `--epochs`、`--seed`、`--config` 覆盖默认配置，完整参数见：

```bash
python experiments/code/run_experiment.py --help
```
