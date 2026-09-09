# Recbole-Learning

本仓库记录基于 [RecBole](https://recbole.io/) 的推荐系统入门实验，目前包含 LightGCN 与 SASRec 在 MovieLens 100K 数据集上的训练、评估和结果分析。

## 项目结构

```text
.
├── experiments/
│   ├── code/                    # 统一实验入口
│   ├── config/                  # LightGCN、SASRec 配置
│   ├── data/                    # 自定义数据占位；数据文件不提交
│   ├── README.md                # 实验环境与运行方法
│   ├── experiment_report.md     # 实验结果和分析
│   └── requirements.txt
├── log/                         # RecBole 默认文本日志，不提交
├── log_tensorboard/             # RecBole 默认 TensorBoard 文件，不提交
├── saved/                       # RecBole 默认 checkpoint，不提交
└── recbole_task_outline.md      # 学习任务大纲
```

`log/`、`log_tensorboard/` 和 `saved/` 由 RecBole 运行时自动创建。仓库只保存代码、配置、说明文档和汇总结果，不保存运行产物与模型权重。

运行方法和实验配置见 [`experiments/README.md`](experiments/README.md)，最终结果见 [`experiments/experiment_report.md`](experiments/experiment_report.md)。
