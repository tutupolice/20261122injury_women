# src/step02d_export_local_adjudication.py
"""
Step 02d: 导出本地 24 人【64条原始伤病人工复核表】与【基线伤病史小数核查表】
（输出仅保存在本地 outputs/private_review/，不提交到 GitHub）
"""
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "local_24"
REVIEW_DIR = PROJECT_ROOT / "outputs" / "private_review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 95)
    print("🔎 Step 02d: 扫描本地原始伤病记录表与基线问卷，生成人工复核清单")
    print("=" * 95)

    # 1. 查找伤病记录表
    inj_path = LOCAL_RAW_DIR / "02_基线数据" / "女足损伤记录表.xlsx"
    if inj_path.exists():
        inj_df = pd.read_excel(inj_path)
        required = {"球员姓名", "受伤日期", "严重程度（天）"}
        if not required.issubset(inj_df.columns):
            raise ValueError(f"损伤记录表缺少字段: {sorted(required - set(inj_df.columns))}")
        inj_df["人工复核_是否非接触(1是/0否)"] = ""
        inj_df["人工复核_是否致停训>=1天(1是/0否)"] = ""
        inj_df["人工复核_是否为同一伤病连续复发(1是/0否)"] = ""
        inj_df["复核备注"] = ""
        out_inj = REVIEW_DIR / "local_injuries_for_manual_review.csv"
        inj_df.to_csv(out_inj, index=False, encoding="utf-8-sig")
        print(f"✅ 已导出原始伤病人工复核表 ({len(inj_df)} 条记录): {out_inj}")
    else:
        print(f"⚠️ 未找到损伤记录表: {inj_path}")

    # 2. 查找基线问卷并筛出非整数伤病次数填报
    base_dir = LOCAL_RAW_DIR / "02_基线数据"
    base_files = list(base_dir.rglob("*.xlsx")) if base_dir.exists() else []
    if base_files:
        for bf in base_files:
            bdf = pd.read_excel(bf)
            num_cols = [c for c in bdf.columns if any(k in str(c) for k in ["次数", "伤", "count"])]
            if num_cols:
                out_base = REVIEW_DIR / f"baseline_check_{bf.stem}.csv"
                bdf.to_csv(out_base, index=False, encoding="utf-8-sig")
                print(f"✅ 已导出基线问卷核查表 ({bf.name}, {len(bdf)} 行): {out_base}")
                for col in num_cols:
                    vals = pd.to_numeric(bdf[col], errors="coerce")
                    non_int = vals.dropna()[vals.dropna() % 1 != 0]
                    if len(non_int) > 0:
                        print(f"   ⚠️ 发现非整数填报列 [{col}]: 共 {len(non_int)} 条 -> {non_int.tolist()}")
    else:
        print("⚠️ 未在 02_基线数据/ 下找到 xlsx 文件。")
    print("=" * 95)


if __name__ == "__main__":
    main()
