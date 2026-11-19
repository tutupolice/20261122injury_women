"""步骤 01：检查 SoccerMon 与本地数据的文件结构和字段。

默认只输出文件元信息、列名和数据类型，不输出原始数据行。
如果确认数据中没有姓名、联系方式或其他敏感信息，可使用：

    python src/step01_inspect_data.py --show-preview
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOCCERMON_DIR = PROJECT_ROOT / "data" / "raw" / "soccermon"
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "local_24"
SUPPORTED_EXTS = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".feather"}


def read_sample(file_path: Path) -> tuple[Any, Any, str | None]:
    """读取字段信息和少量样本，避免把完整数据加载到控制台。"""
    ext = file_path.suffix.lower()
    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(file_path, nrows=5)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, nrows=5, encoding="gb18030")
            with file_path.open("r", encoding="utf-8", errors="ignore") as file_obj:
                total_rows = max(sum(1 for _ in file_obj) - 1, 0)
            return df, total_rows, None

        if ext in {".xlsx", ".xls"}:
            excel_file = pd.ExcelFile(file_path)
            sheet_info: dict[str, dict[str, Any]] = {}
            for sheet_name in excel_file.sheet_names[:5]:
                df_sheet = pd.read_excel(file_path, sheet_name=sheet_name, nrows=5)
                sheet_info[sheet_name] = {
                    "columns": list(df_sheet.columns),
                    "dtypes": {str(k): str(v) for k, v in df_sheet.dtypes.items()},
                }
            return sheet_info, "Excel 多工作表", None

        if ext == ".parquet":
            df = pd.read_parquet(file_path)
            return df.head(5), len(df), None

        if ext == ".json":
            df = pd.read_json(file_path)
            return df.head(5), len(df), None

        if ext == ".feather":
            df = pd.read_feather(file_path)
            return df.head(5), len(df), None

    except Exception as exc:  # noqa: BLE001 - 检查脚本需要把单文件错误打印出来
        return None, 0, f"{type(exc).__name__}: {exc}"

    return None, 0, "不支持的格式"


def inspect_directory(title: str, dir_path: Path, show_preview: bool) -> None:
    print("=" * 70)
    print(f"正在扫描 [{title}] 目录: {dir_path}")
    print("=" * 70)

    if not dir_path.exists():
        print(f"路径不存在：{dir_path}\n")
        return

    files = sorted(
        path
        for path in dir_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    )

    if not files:
        existing = [path.name for path in dir_path.iterdir()][:15]
        print("该目录下没有找到常见数据文件。")
        print(f"目录下现有内容：{existing}\n")
        return

    print(f"共找到 {len(files)} 个数据文件（最多详细展示前 15 个）：\n")
    for index, file_path in enumerate(files[:15], start=1):
        relative_path = file_path.relative_to(dir_path)
        if title == "本地 24 人女足小样本数据集":
            display_path = f"文件_{index:04d}{file_path.suffix.lower()}"
        else:
            display_path = relative_path
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"--- [{index}] 文件：{display_path}（{size_mb:.2f} MB）---")

        sample, total_rows, error = read_sample(file_path)
        if error:
            print(f"读取出错：{error}")
        elif isinstance(sample, dict):
            print(f"类型：Excel 文件；{total_rows}")
            for sheet_name, info in sample.items():
                print(f"  Sheet [{sheet_name}]：")
                print(f"    列名：{info['columns'][:50]}")
                print(f"    数据类型：{json.dumps(info['dtypes'], ensure_ascii=False)}")
        elif isinstance(sample, pd.DataFrame):
            print(f"总行数（约）：{total_rows} | 总列数：{len(sample.columns)}")
            print(f"列名列表：{list(sample.columns[:50])}")
            print(
                "数据类型："
                + json.dumps(
                    {str(k): str(v) for k, v in sample.dtypes.items()},
                    ensure_ascii=False,
                )
            )
            if show_preview:
                print("前 2 行数据预览：")
                print(sample.head(2).to_string(index=False))
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查项目原始数据文件结构")
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="打印每个文件前两行；确认不含敏感信息后再使用",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for relative_dir in [
        "data/raw/soccermon",
        "data/raw/local_24",
        "data/processed",
        "outputs",
    ]:
        (PROJECT_ROOT / relative_dir).mkdir(parents=True, exist_ok=True)

    inspect_directory("SoccerMon 数据集", SOCCERMON_DIR, args.show_preview)
    inspect_directory("本地 24 人女足小样本数据集", LOCAL_DATA_DIR, args.show_preview)
    print("步骤 01 探查完成。")
