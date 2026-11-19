# src/step01b_find_labels_and_units.py
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOCCERMON_DIR = PROJECT_ROOT / "data" / "raw" / "soccermon"
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "local_24"


def scan_folder_structure(title: str, base_dir: Path):
    print("=" * 75)
    print(f"🔍 1. 深度排查 [{title}] 的目录结构与特殊文件（寻找损伤标签与球员映射）")
    print("=" * 75)

    # 统计一级、二级子目录
    sub_dirs = sorted({p.parent.relative_to(base_dir) for p in base_dir.rglob("*") if p.is_file()})
    print(f"📁 包含子目录数量: {len(sub_dirs)} (前10个): {[str(d) for d in sub_dirs[:10]]}")

    # 按后缀统计文件
    all_files = [p for p in base_dir.rglob("*") if p.is_file() and not p.name.startswith(".")]
    ext_counts = {}
    for f in all_files:
        ext_counts[f.suffix.lower()] = ext_counts.get(f.suffix.lower(), 0) + 1
    print(f"📊 文件类型统计: {ext_counts}")

    # 找出非标准传感器命名或特殊后缀的文件（如 csv, json, txt，或名字不含 '文件_' / 'Team' 的文件）
    special_files = []
    for f in all_files:
        name_lower = f.name.lower()
        if f.suffix.lower() in [".csv", ".json", ".txt", ".tsv"]:
            special_files.append(f)
        elif not (name_lower.startswith("文件_") or "-teama-" in name_lower or "-teamb-" in name_lower):
            special_files.append(f)

    print(f"\n📌 发现疑似【标签/汇总/映射】特殊文件共 {len(special_files)} 个（最多展示前 30 个）：")
    for idx, sf in enumerate(sorted(special_files)[:30], 1):
        rel = sf.relative_to(base_dir)
        size_kb = sf.stat().st_size / 1024
        print(f"  [{idx}] {rel} ({size_kb:.1f} KB)")
        # 如果是表格文件且不太大，打印列名和前2行
        try:
            if sf.suffix.lower() == ".csv":
                df_tmp = pd.read_csv(sf, nrows=3)
                print(f"      列名: {list(df_tmp.columns)}")
                print(f"      预览: {df_tmp.head(2).to_dict(orient='records')}")
            elif sf.suffix.lower() in [".xlsx", ".xls"]:
                xl = pd.ExcelFile(sf)
                print(f"      工作表: {xl.sheet_names}")
                for s in xl.sheet_names[:3]:
                    df_tmp = xl.parse(s, nrows=3)
                    print(f"      Sheet [{s}] 列名: {list(df_tmp.columns)}")
        except Exception as e:
            print(f"      读取预览失败: {e}")
    print()


def compare_sensor_units():
    print("=" * 75)
    print("⚖️ 2. 抽样对比两边传感器数据的真实数值、采样频率与时间格式")
    print("=" * 75)

    # 1. 抽样 1 个 SoccerMon parquet 文件
    sm_files = sorted(SOCCERMON_DIR.rglob("*.parquet"))
    if sm_files:
        f_sm = sm_files[0]
        print(f"👉 [SoccerMon 样本]: {f_sm.relative_to(SOCCERMON_DIR)}")
        df_sm = pd.read_parquet(f_sm)
        print("   前 3 行数据:")
        print(df_sm.head(3).to_string())
        num_cols = ['speed', 'heart_rate', 'inst_acc_impulse', 'accl_x', 'accl_y', 'accl_z', 'gyro_x', 'gyro_y', 'gyro_z']
        num_cols = [c for c in num_cols if c in df_sm.columns]
        print("\n   数值分布统计 (min / 50% / 95% / max):")
        print(df_sm[num_cols].quantile([0.0, 0.5, 0.95, 1.0]).to_string())
    print("-" * 75)

    # 2. 抽样 1 个 本地 24 人 Excel 文件
    loc_files = sorted(LOCAL_DATA_DIR.rglob("*.xlsx"))
    if loc_files:
        f_loc = loc_files[0]
        print(f"👉 [本地 24人 样本]: {f_loc.relative_to(LOCAL_DATA_DIR)}")
        xl = pd.ExcelFile(f_loc)
        print(f"   包含的所有 Sheet 名称: {xl.sheet_names}")
        df_loc = xl.parse(xl.sheet_names[0])
        print(f"   总行数: {len(df_loc)}")
        print("   前 3 行数据:")
        print(df_loc.head(3).to_string())
        loc_num_cols = ['速度 m/s', '心率', '加速度计X', '加速度计Y', '加速度计Z', '陀螺仪X', '陀螺仪Y', '陀螺仪Z']
        loc_num_cols = [c for c in loc_num_cols if c in df_loc.columns]
        print("\n   数值分布统计 (min / 50% / 95% / max):")
        print(df_loc[loc_num_cols].quantile([0.0, 0.5, 0.95, 1.0]).to_string())


if __name__ == "__main__":
    scan_folder_structure("SoccerMon 数据集", SOCCERMON_DIR)
    scan_folder_structure("本地 24人女足小样本数据集", LOCAL_DATA_DIR)
    compare_sensor_units()
    print("\n🎉 步骤 01b 探查完成！请把控制台输出发给我。")