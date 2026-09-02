"""使用 RecBole 运行 SASRec 序列推荐实验。

脚本默认面向 AutoDL NVIDIA GPU 环境：运行前会检查 CUDA 和 GPU 编号，
训练过程会同步保存终端日志，并把 RecBole 返回的验证集、测试集指标写入 JSON。

用法（在仓库根目录执行）：
    python experiments/sasrec/code/run_sasrec.py --gpu-id 0
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
    """同时向终端和日志文件写入文本。"""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        """写入所有目标流。"""
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        """刷新所有目标流。"""
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        """保留终端属性，避免进度条行为异常。"""
        return any(stream.isatty() for stream in self.streams)


def parse_args() -> argparse.Namespace:
    """解析 SASRec 实验参数。"""
    parser = argparse.ArgumentParser(description="Run SASRec with RecBole")
    parser.add_argument("--dataset", default="ml-100k", help="RecBole dataset 名称")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="RecBole YAML 配置文件"
    )
    parser.add_argument(
        "--data-path", type=Path, default=DEFAULT_DATA_PATH, help="包含 <dataset>/ 的目录"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出根目录"
    )
    parser.add_argument("--epochs", type=int, help="覆盖 YAML 中的训练轮数")
    parser.add_argument("--seed", type=int, help="覆盖 YAML 中的随机种子")
    parser.add_argument(
        "--gpu-id", default="0", help="NVIDIA GPU 编号，支持 0 或 0,1，默认 0"
    )
    parser.add_argument(
        "--run-name", default=None, help="日志和结果文件名，默认 sasrec_<dataset>"
    )
    parser.add_argument("--no-save", action="store_true", help="不保存模型 checkpoint")
    return parser.parse_args()


def validate_cuda(gpu_id: str) -> str:
    """确认当前环境能使用指定的 NVIDIA GPU。"""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前环境未安装 PyTorch，无法使用 NVIDIA GPU。") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "未检测到 CUDA/NVIDIA GPU。请在 AutoDL 选择 GPU 实例，并安装 CUDA 版 PyTorch。"
        )
    try:
        gpu_ids = [int(part.strip()) for part in gpu_id.split(",")]
    except ValueError as exc:
        raise ValueError(f"--gpu-id 必须是 0 或 0,1 形式，实际为: {gpu_id}") from exc
    if not gpu_ids or any(index < 0 for index in gpu_ids):
        raise ValueError(f"--gpu-id 必须包含非负整数，实际为: {gpu_id}")
    if max(gpu_ids) >= torch.cuda.device_count():
        raise ValueError(
            f"请求 GPU {gpu_id}，但当前只检测到 {torch.cuda.device_count()} 张可见 GPU。"
        )
    return torch.cuda.get_device_name(gpu_ids[0])


def to_jsonable(value: Any) -> Any:
    """转换 RecBole/PyTorch 返回值，确保可以写入 JSON。"""
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


def run_experiment(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    """执行 SASRec，并返回归档记录和结果文件路径。"""
    config_file = args.config.resolve()
    data_path = args.data_path.resolve()
    output_dir = args.output_dir.resolve()
    dataset_dir = data_path / args.dataset
    if not config_file.is_file():
        raise FileNotFoundError(f"找不到配置文件: {config_file}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"找不到数据目录: {dataset_dir}\n"
            "请先执行 prepare_ml100k.py，并将 --data-path 指向相同目录。"
        )

    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    result_dir = output_dir / "results"
    for directory in (checkpoint_dir, log_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)

    gpu_name = validate_cuda(args.gpu_id)
    run_name = args.run_name or f"sasrec_{args.dataset}"
    log_file = log_dir / f"{run_name}.log"
    result_file = result_dir / f"{run_name}.json"
    config_dict: dict[str, Any] = {
        "data_path": str(data_path),
        "checkpoint_dir": str(checkpoint_dir),
        "gpu_id": str(args.gpu_id),
    }
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.seed is not None:
        config_dict["seed"] = args.seed

    try:
        from recbole.quick_start import run_recbole
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "当前环境未安装 RecBole，请在 d2l 环境中安装 RecBole 后再运行。"
        ) from exc

    # RecBole 会读取 sys.argv；清空本脚本参数，避免 --run-name 等参数被误解析。
    original_argv = sys.argv[:]
    sys.argv[:] = [sys.argv[0]]
    try:
        with log_file.open("w", encoding="utf-8") as log_stream:
            stdout_tee = Tee(sys.stdout, log_stream)
            stderr_tee = Tee(sys.stderr, log_stream)
            with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                print(f"NVIDIA GPU: {args.gpu_id} ({gpu_name})")
                result = run_recbole(
                    model="SASRec",
                    dataset=args.dataset,
                    config_file_list=[str(config_file)],
                    config_dict=config_dict,
                    saved=not args.no_save,
                )
    finally:
        sys.argv[:] = original_argv

    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "SASRec",
        "dataset": args.dataset,
        "config_file": str(config_file),
        "config_overrides": config_dict,
        "gpu": {"id": args.gpu_id, "name": gpu_name},
        "saved": not args.no_save,
        "result": to_jsonable(result),
    }
    result_file.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record, result_file


def main() -> int:
    """脚本入口。"""
    record, result_file = run_experiment(parse_args())
    print(f"实验完成，结果已写入: {result_file}")
    print(json.dumps(record["result"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
