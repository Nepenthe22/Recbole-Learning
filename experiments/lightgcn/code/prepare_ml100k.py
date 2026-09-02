"""下载并转换 MovieLens 100K，使其可以直接被 RecBole 读取。

MovieLens 原始数据中的 ``u.data`` 是空白分隔、没有字段类型声明的文本。
RecBole 的 atomic file 需要在表头中写出 ``字段名:字段类型``，因此本脚本只做
这一层格式转换，不改变用户、物品、评分和时间戳的值。

用法（在仓库根目录执行）：
    python experiments/lightgcn/code/prepare_ml100k.py
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_PATH = EXPERIMENT_DIR / "data"
ATOMIC_HEADER = "user_id:token\titem_id:token\trating:float\ttimestamp:float\n"


def parse_args() -> argparse.Namespace:
    """读取数据准备参数。"""
    parser = argparse.ArgumentParser(description="Prepare MovieLens 100K for RecBole")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="RecBole data_path；数据会写入该目录下的 ml-100k/",
    )
    parser.add_argument(
        "--url",
        default=DATASET_URL,
        help="MovieLens 压缩包地址，默认使用 GroupLens 官方地址",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="目标文件已存在时重新下载并覆盖",
    )
    return parser.parse_args()


def convert_u_data(raw_data: bytes) -> str:
    """把 ``u.data`` 转换为 RecBole 的 ``.inter`` 文本。"""
    rows = [ATOMIC_HEADER]
    for line_number, raw_line in enumerate(raw_data.decode("utf-8").splitlines(), 2):
        fields = raw_line.split()
        if len(fields) != 4:
            raise ValueError(
                f"u.data 第 {line_number} 行应有 4 列，实际得到 {len(fields)} 列"
            )
        user_id, item_id, rating, timestamp = fields
        rows.append(f"{user_id}\t{item_id}\t{rating}\t{timestamp}\n")
    return "".join(rows)


def prepare_dataset(data_path: Path, url: str, force: bool = False) -> Path:
    """下载数据并生成 ``<data_path>/ml-100k/ml-100k.inter``。"""
    dataset_dir = data_path.resolve() / "ml-100k"
    output_file = dataset_dir / "ml-100k.inter"

    if output_file.exists() and not force:
        print(f"数据已存在，跳过下载: {output_file}")
        return output_file

    print(f"正在下载 MovieLens 100K: {url}")
    with urlopen(url, timeout=60) as response:  # noqa: S310 - URL 可由用户显式指定
        archive = response.read()

    with ZipFile(BytesIO(archive)) as zip_file:
        member_name = "ml-100k/u.data"
        try:
            raw_data = zip_file.read(member_name)
        except KeyError as exc:
            raise ValueError(f"压缩包中找不到预期文件: {member_name}") from exc

    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_file.write_text(convert_u_data(raw_data), encoding="utf-8", newline="")
    print(f"已生成 RecBole atomic file: {output_file}")
    return output_file


def main() -> int:
    """脚本入口。"""
    args = parse_args()
    prepare_dataset(args.data_path, args.url, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
