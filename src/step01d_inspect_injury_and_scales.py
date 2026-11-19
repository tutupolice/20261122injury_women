# src/step01d_inspect_injury_and_scales.py
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOCCERMON_DIR = PROJECT_ROOT / "data" / "raw" / "soccermon"
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "local_24"


def inspect_local_baseline_and_injury():
    print("=" * 75)
    print("📋 1. 本地 [女足损伤记录表.xlsx] 全部内容与统计")
    print("=" * 75)
    inj_path = LOCAL_DATA_DIR / "02_基线数据" / "女足损伤记录表.xlsx"
    df_inj = pd.read_excel(inj_path)
    print(f"总伤病记录条数: {len(df_inj)} | 涉及球员人数: {df_inj['球员姓名'].nunique()}")
    print("\n完整伤病记录表:")
    print(df_inj.to_string(index=True))

    print("\n" + "=" * 75)
    print("📋 2. 本地 [基线健康与既往损伤史] & [匿名基准信息] 概览")
    print("=" * 75)
    hist_path = LOCAL_DATA_DIR / "02_基线数据" / "东北师范大学女足基线健康与既往损伤史调查问卷xlsx.xlsx"
    df_hist = pd.read_excel(hist_path)
    print(f"既往损伤史问卷人数: {len(df_hist)}")
    cols_keep = [c for c in df_hist.columns if any(k in c for k in ["姓名", "健康状态", "12 个月", "次数", "ACL", "重大伤病"])]
    print(df_hist[cols_keep].to_string(index=False))

    info_path = LOCAL_DATA_DIR / "02_基线数据" / "匿名女足基准信息.xlsx"
    df_info = pd.read_excel(info_path)
    print(f"\n匿名基准信息人数: {len(df_info)}")
    print(df_info.head(10).to_string(index=False))


def inspect_soccermon_overlap_and_scales():
    print("\n" + "=" * 75)
    print("🔗 3. 检查 SoccerMon 客观 GPS (Parquet) 与 主观/伤病数据的球员重合度")
    print("=" * 75)
    # 从文件名快速提取拥有 GPS 数据的球员 ID 和日期（无需打开 parquet，极快）
    parquet_files = list(SOCCERMON_DIR.rglob("*.parquet"))
    gps_players = set()
    gps_dates = []
    for p in parquet_files:
        # 文件名形如: 2020-06-01-TeamA-1846d424-c17c-6279-23c6-612f48268673.parquet
        stem = p.stem
        if len(stem) > 11 and stem[4] == "-" and stem[7] == "-":
            gps_dates.append(stem[:10])
            gps_players.add(stem[11:])

    print(f"SoccerMon GPS 文件总数: {len(parquet_files)}")
    print(f"拥有 GPS 数据的球员数: {len(gps_players)} (例如: {sorted(list(gps_players))[:3]})")
    if gps_dates:
        print(f"GPS 数据日期范围: {min(gps_dates)} ~ {max(gps_dates)}")

    # 检查主观 wellness 表的球员和数值范围
    wellness_dir = SOCCERMON_DIR / "subjective" / "wellness"
    print("\n📊 SoccerMon Wellness 各量表数值分布 (min / median / max / 非空样本数):")
    for csv_f in sorted(wellness_dir.glob("*.csv")):
        df_w = pd.read_csv(csv_f)
        date_col = df_w.columns[0]
        player_cols = [c for c in df_w.columns[1:]]
        vals = df_w[player_cols].values.flatten()
        vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
        print(f"  - {csv_f.name:20s}: min={vals.min():.1f}, median={vals.median():.1f}, max={vals.max():.1f}, 非空记录数={len(vals)}, 球员列数={len(player_cols)}")

    # 检查 injury.csv 中有多少人在 gps_players 里
    df_sm_inj = pd.read_csv(SOCCERMON_DIR / "subjective" / "injury" / "injury.csv")
    inj_players = set(df_sm_inj["player_name"].unique())
    overlap_inj = inj_players.intersection(gps_players)
    print(f"\n🏥 SoccerMon injury.csv 受伤球员数: {len(inj_players)} | 其中同时拥有 GPS 数据的受伤球员数: {len(overlap_inj)}")
    print("injury.csv 按 TeamA / TeamB 统计记录数:")
    df_sm_inj["team"] = df_sm_inj["player_name"].apply(lambda x: x.split("-")[0] if isinstance(x, str) else "Unknown")
    print(df_sm_inj.groupby("team")["player_name"].agg(记录条数="count", 受伤人数="nunique").to_string())


if __name__ == "__main__":
    inspect_local_baseline_and_injury()
    inspect_soccermon_overlap_and_scales()
    print("\n🎉 步骤 01d 完成！请把控制台结果发给我。")