"""统一运行 LightGCN 或 SASRec 实验。

模型、数据集划分和超参数放在 config/ 中；本脚本只处理两种模型共用的实验流程：
输入检查、CUDA 检查、调用 RecBole、保存日志和归档结果。
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = EXPERIMENTS_DIR / "data"
DEFAULT_OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"
CONFIG_DIR = EXPERIMENTS_DIR / "config"


class Tee:
    """同时将 RecBole 输出写到终端和日志文件。"""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self.streams)


def parse_args() -> argparse.Namespace:
    """解析统一实验参数。"""
    parser = argparse.ArgumentParser(description="Run RecBole recommendation experiments")
    parser.add_argument("--model", choices=["LightGCN", "SASRec"], required=True)
    parser.add_argument("--dataset", default="ml-100k")
    parser.add_argument("--config", type=Path, help="可选的自定义 YAML 配置")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, help="覆盖配置文件中的 epoch")
    parser.add_argument("--seed", type=int, help="覆盖配置文件中的随机种子")
    parser.add_argument("--gpu-id", default="0", help="NVIDIA GPU 编号，例如 0 或 0,1")
    parser.add_argument("--run-name", help="日志和结果文件名，不含扩展名")
    parser.add_argument("--no-save", action="store_true", help="不保存 checkpoint")
    return parser.parse_args()


def validate_cuda(gpu_id: str) -> str:
    """检查 CUDA 和 GPU 编号，防止实验意外退回 CPU。"""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前环境未安装 PyTorch。") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA/NVIDIA GPU，请在 AutoDL 选择 GPU 实例。")
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
    """将 RecBole/PyTorch 标量转换为 JSON 可序列化对象。"""
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


def main() -> int:
    """统一实验入口。"""
    args = parse_args()
    config_file = (args.config or CONFIG_DIR / f"{args.model.lower()}_{args.dataset}.yaml").resolve()
    data_path = args.data_path.resolve()
    output_dir = args.output_dir.resolve()
    dataset_dir = data_path / args.dataset
    if not config_file.is_file():
        raise FileNotFoundError(f"找不到配置文件: {config_file}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"找不到数据目录: {dataset_dir}，请先准备数据。")

    gpu_name = validate_cuda(args.gpu_id)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    result_dir = output_dir / "results"
    for directory in (checkpoint_dir, log_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"{args.model.lower()}_{args.dataset}"
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
        raise RuntimeError("当前环境未安装 RecBole，请使用 d2l 环境。") from exc

    original_argv = sys.argv[:]
    sys.argv[:] = [sys.argv[0]]
    try:
        with log_file.open("w", encoding="utf-8") as log_stream:
            with redirect_stdout(Tee(sys.stdout, log_stream)), redirect_stderr(
                Tee(sys.stderr, log_stream)
            ):
                print(f"model={args.model}, dataset={args.dataset}")
                print(f"NVIDIA GPU={args.gpu_id}, name={gpu_name}")
                result = run_recbole(
                    model=args.model,
                    dataset=args.dataset,
                    config_file_list=[str(config_file)],
                    config_dict=config_dict,
                    saved=not args.no_save,
                )
    finally:
        sys.argv[:] = original_argv

    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
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
    print(f"结果已写入: {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
