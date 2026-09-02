"""运行 RecBole LightGCN，并把日志和指标归档到本任务目录。

RecBole 负责模型、数据集划分、训练和评估；本脚本只负责实验编排：

1. 校验数据目录和配置文件是否存在；
2. 用绝对路径覆盖 data_path 和 checkpoint_dir；
3. 将标准输出、标准错误和 RecBole 日志同步保存到 outputs/logs；
4. 将验证集最优结果、测试集结果和运行参数写入 outputs/results。

用法（在仓库根目录执行）：
    python experiments/lightgcn/code/prepare_ml100k.py
    python experiments/lightgcn/code/run_lightgcn.py
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config" / "ml-100k.yaml"
DEFAULT_DATA_PATH = EXPERIMENT_DIR / "data"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


class Tee:
    """把一个输出流同时写到终端和日志文件。"""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        """将文本写入所有目标流，并返回写入长度。"""
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        """刷新所有目标流，兼容 logging 和 tqdm 的调用方式。"""
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        """保持终端输出的交互属性，避免进度条被错误地完全禁用。"""
        return any(stream.isatty() for stream in self.streams)


def parse_args() -> argparse.Namespace:
    """读取实验入口参数。"""
    parser = argparse.ArgumentParser(description="Run LightGCN with RecBole")
    parser.add_argument("--dataset", default="ml-100k", help="RecBole dataset 名称")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="RecBole YAML 配置文件"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="RecBole data_path；其中应包含 <dataset>/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出根目录，下面自动创建 checkpoints/logs/results",
    )
    parser.add_argument("--epochs", type=int, help="覆盖 YAML 中的训练轮数")
    parser.add_argument("--seed", type=int, help="覆盖 YAML 中的随机种子")
    parser.add_argument(
        "--gpu-id",
        default="0",
        help="NVIDIA GPU 编号，支持单卡或逗号分隔的多卡编号，默认 0",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="日志和结果文件名（不含扩展名），默认 lightgcn_<dataset>",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存最佳模型 checkpoint，仅运行训练和评估",
    )
    return parser.parse_args()


def make_config_dict(args: argparse.Namespace, checkpoint_dir: Path) -> dict[str, Any]:
    """组装需要由脚本动态覆盖的 RecBole 参数。"""
    # 路径使用绝对路径，避免 RecBole 因启动目录不同而找错数据或输出位置。
    config_dict: dict[str, Any] = {
        "data_path": str(args.data_path.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        # RecBole 使用 gpu_id 选择 CUDA 设备；默认 0 对应 AutoDL 的第一张显卡。
        "gpu_id": str(args.gpu_id),
    }
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.seed is not None:
        config_dict["seed"] = args.seed
    return config_dict


def validate_cuda(gpu_id: str) -> str:
    """检查 NVIDIA CUDA 是否可用，并返回当前显卡名称。

    本任务面向 AutoDL GPU 环境。训练前主动检查可以避免 CUDA 未挂载时
    RecBole 静默退回 CPU，导致训练速度和实验记录不符合预期。
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前环境未安装 PyTorch，无法使用 NVIDIA GPU。") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到可用 CUDA/NVIDIA GPU。请在 AutoDL 中选择 GPU 实例，"
            "并确认安装了 CUDA 版 PyTorch。"
        )

    try:
        gpu_ids = [int(part.strip()) for part in gpu_id.split(",")]
    except ValueError as exc:
        raise ValueError(f"--gpu-id 必须是类似 0 或 0,1 的编号，实际为: {gpu_id}") from exc

    if not gpu_ids or any(index < 0 for index in gpu_ids):
        raise ValueError(f"--gpu-id 必须包含非负整数，实际为: {gpu_id}")
    device_count = torch.cuda.device_count()
    if max(gpu_ids) >= device_count:
        raise ValueError(
            f"请求 GPU {gpu_id}，但当前环境只检测到 {device_count} 张可见 GPU。"
        )
    return torch.cuda.get_device_name(gpu_ids[0])


def to_jsonable(value: Any) -> Any:
    """将 RecBole/PyTorch 可能返回的标量转换为可序列化对象。"""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def check_inputs(config_file: Path, dataset_dir: Path) -> None:
    """在导入 RecBole 前做清晰的输入检查。"""
    if not config_file.is_file():
        raise FileNotFoundError(f"找不到配置文件: {config_file}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"找不到数据目录: {dataset_dir}\n"
            "请先执行 prepare_ml100k.py，或把已准备好的 RecBole 数据放入该目录。"
        )


def run_experiment(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path]:
    """运行 LightGCN，并返回结果字典、日志路径和结果路径。"""
    config_file = args.config.resolve()
    data_path = args.data_path.resolve()
    output_dir = args.output_dir.resolve()
    dataset_dir = data_path / args.dataset
    check_inputs(config_file, dataset_dir)

    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    result_dir = output_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = validate_cuda(args.gpu_id)

    run_name = args.run_name or f"lightgcn_{args.dataset}"
    log_file = log_dir / f"{run_name}.log"
    result_file = result_dir / f"{run_name}.json"
    config_dict = make_config_dict(args, checkpoint_dir)

    # 延迟导入，保证即使未安装 RecBole，--help 和输入检查也能正常使用。
    try:
        from recbole.quick_start import run_recbole
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "当前 Python 环境未安装 RecBole，请先执行："
            "python -m pip install -r experiments/lightgcn/requirements.txt"
        ) from exc

    print(f"配置文件: {config_file}")
    print(f"数据目录: {dataset_dir}")
    print(f"NVIDIA GPU: {args.gpu_id} ({gpu_name})")
    print(f"日志文件: {log_file}")
    print(f"结果文件: {result_file}")

    # RecBole 会读取 sys.argv 作为自身的命令行覆盖项。这里已经由本脚本解析了
    # 所有参数，清空剩余参数可避免 --run-name 等脚本参数被 RecBole 误解析。
    original_argv = sys.argv[:]
    sys.argv[:] = [sys.argv[0]]
    try:
        with log_file.open("w", encoding="utf-8") as log_stream:
            stdout_tee = Tee(sys.stdout, log_stream)
            stderr_tee = Tee(sys.stderr, log_stream)
            with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                result = run_recbole(
                    model="LightGCN",
                    dataset=args.dataset,
                    config_file_list=[str(config_file)],
                    config_dict=config_dict,
                    saved=not args.no_save,
                )
    finally:
        sys.argv[:] = original_argv

    result = to_jsonable(result)
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "LightGCN",
        "dataset": args.dataset,
        "config_file": str(config_file),
        "config_overrides": to_jsonable(config_dict),
        "saved": not args.no_save,
        "result": result,
    }
    result_file.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record, log_file, result_file


def main() -> int:
    """脚本入口。"""
    args = parse_args()
    record, _, result_file = run_experiment(args)
    print(f"实验完成，结果已写入: {result_file}")
    print(json.dumps(record["result"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
