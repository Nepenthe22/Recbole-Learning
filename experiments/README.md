# RecBole 推荐模型实验

本目录统一管理任务大纲中的两个实验：LightGCN 和 SASRec。
两个模型共享数据准备、GPU 检查、训练入口、日志归档和结果格式；模型差异只保留在各自的 YAML 配置中。

## 目录结构

```text
experiments/
├── code/
│   ├── prepare_ml100k.py       # 共享：下载并生成 RecBole atomic file
│   └── run_experiment.py       # 共享：通过 --model 选择模型
├── config/
│   ├── lightgcn_ml-100k.yaml   # LightGCN 配置
│   └── sasrec_ml-100k.yaml     # SASRec 配置
├── data/                       # 数据集，不提交大文件
├── outputs/                    # checkpoint、日志、JSON 结果
├── experiment_report.md        # 最终实验报告
└── requirements.txt            # 依赖说明
```

## AutoDL 运行

```bash
conda activate d2l
nvidia-smi

python experiments/code/prepare_ml100k.py
python experiments/code/run_experiment.py --model LightGCN --gpu-id 0
python experiments/code/run_experiment.py --model SASRec --gpu-id 0
```

也可以先把数据放到自定义目录，再通过 `--data-path` 指定。运行器默认强制检查 CUDA，避免在 AutoDL 显卡未挂载时误用 CPU。

结果位置：

- `outputs/logs/`：完整运行日志；
- `outputs/results/`：验证集最优结果和测试集指标；
- `outputs/checkpoints/`：RecBole 保存的模型。

## 模型选择

```bash
python experiments/code/run_experiment.py --model LightGCN --epochs 20
python experiments/code/run_experiment.py --model SASRec --epochs 20
```
