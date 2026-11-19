# src/step01c_check_local_labels.py
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOCCERMON_DIR = PROJECT_ROOT / "data" / "raw" / "soccermon"
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "local_24"


def check_local_structure():
    print("=" * 75)
    print("📂 1. 查看 [本地 24人数据集] 的所有文件夹名称")
    print("=" * 75)
    all_dirs = sorted({str(p.parent.relative_to(LOCAL_DATA_DIR)) for p in LOCAL_DATA_DIR.rglob("*") if p.is_file()})
    for d in all_dirs:
        print(f"  - {d}")

    print("\n" + "=" * 75)
    print("📌 2. 查找 [本地 24人数据集] 中不属于 '00_原始客观数据' 的其他文件（如损伤表/主观表）")
    print("=" * 75)
    other_files = [
        p for p in LOCAL_DATA_DIR.rglob("*")
        if p.is_file() and not p.name.startswith(".") and "00_原始客观数据" not in str(p)
    ]
    if not other_files:
        print("⚠️ 在 local_24 下除了 '00_原始客观数据' 外，没有发现其他文件夹或文件！")
        print("   👉 请确认：本地 24 名球员的【运动损伤记录表 / 主观问卷表】存放在电脑的哪个位置？")
    else:
        for idx, f in enumerate(sorted(other_files), 1):
            rel = f.relative_to(LOCAL_DATA_DIR)
            print(f"  [{idx}] {rel} ({f.stat().st_size / 1024:.1f} KB)")
            if f.suffix.lower() in [".xlsx", ".xls"]:
                xl = pd.ExcelFile(f)
                for s in xl.sheet_names[:5]:
                    df_tmp = pd.read_excel(f, sheet_name=s, nrows=3)
                    print(f"      Sheet [{s}] 列名: {list(df_tmp.columns)}")
                    print(f"      预览: {df_tmp.head(2).to_dict(orient='records')}")
            elif f.suffix.lower() == ".csv":
                df_tmp = pd.read_csv(f, nrows=3)
                print(f"      列名: {list(df_tmp.columns)}")
                print(f"      预览: {df_tmp.head(2).to_dict(orient='records')}")


def check_soccermon_injury():
    print("\n" + "=" * 75)
    print("🏥 3. 查看 SoccerMon 的所有主观文件清单与 injury.csv 详情")
    print("=" * 75)
    subj_dir = SOCCERMON_DIR / "subjective"
    for f in sorted(subj_dir.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            print(f"  - {f.relative_to(SOCCERMON_DIR)}")

    inj_file = subj_dir / "injury" / "injury.csv"
    if inj_file.exists():
        df_inj = pd.read_csv(inj_file)
        print(f"\n✅ SoccerMon injury.csv 总行数: {len(df_inj)}，涉及受伤球员数: {df_inj['player_name'].nunique()}")
        print("前 10 条伤病记录:")
        print(df_inj.head(10).to_string(index=False))
        print("\n伤病类型 (type) 出现频次前 10 名:")
        print(df_inj["type"].value_counts().head(10).to_string())


if __name__ == "__main__":
    check_local_structure()
    check_soccermon_injury()
    print("\n🎉 步骤 01c 完成！请把输出发给我。")