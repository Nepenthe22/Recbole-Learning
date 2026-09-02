"""下载 MovieLens 100K 并转换为 RecBole atomic file。

生成文件：experiments/data/ml-100k/ml-100k.inter
该脚本只准备数据，不启动任何模型训练。
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = EXPERIMENTS_DIR / "data"
HEADER = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"


def convert_u_data(raw_data: bytes) -> str:
    """把原始 u.data 转为带类型表头的 .inter 文件。"""
    rows = [HEADER]
    for line_number, line in enumerate(raw_data.decode("utf-8").splitlines(), 2):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"u.data 第 {line_number} 行不是 4 列")
        rows.append("\t".join(fields) + "\n")
    return "".join(rows)


def main() -> int:
    """下载并生成数据文件。"""
    parser = argparse.ArgumentParser(description="Prepare MovieLens 100K for RecBole")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--force", action="store_true", help="覆盖已有 .inter 文件")
    args = parser.parse_args()

    output_file = args.data_path.resolve() / "ml-100k" / "ml-100k.inter"
    if output_file.exists() and not args.force:
        print(f"数据已存在，跳过下载: {output_file}")
        return 0

    print(f"正在下载: {args.url}")
    with urlopen(args.url, timeout=60) as response:  # noqa: S310 - 用户可显式指定 URL
        archive = response.read()
    with ZipFile(BytesIO(archive)) as zip_file:
        raw_data = zip_file.read("ml-100k/u.data")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(convert_u_data(raw_data), encoding="utf-8", newline="")
    print(f"已生成: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
