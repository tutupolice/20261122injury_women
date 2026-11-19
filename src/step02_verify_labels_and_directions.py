# src/step02_verify_labels_and_directions.py
import re
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOCCERMON_DIR = PROJECT_ROOT / "data" / "raw" / "soccermon"
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "local_24"


def verify_soccermon():
    print("=" * 75)
    print("🔍 1. 验证 SoccerMon 伤病日期分布、新发损伤(Onset)数量与量表正反向")
    print("=" * 75)

    # 读取 injury.csv
    df_inj = pd.read_csv(SOCCERMON_DIR / "subjective" / "injury" / "injury.csv")
    df_inj["date"] = pd.to_datetime(df_inj["timestamp"], format="%d.%m.%Y", errors="coerce")
    df_inj = df_inj.dropna(subset=["date"]).sort_values(["player_name", "date"]).drop_duplicates(["player_name", "date"])

    # 计算与上一次上报伤病的间隔天数，间隔 > 3 天视为一次“新发损伤事件 (Injury Onset)”
    df_inj["prev_date"] = df_inj.groupby("player_name")["date"].shift(1)
    df_inj["days_since_prev"] = (df_inj["date"] - df_inj["prev_date"]).dt.days
    df_inj["is_onset"] = df_inj["days_since_prev"].isna() | (df_inj["days_since_prev"] > 3)

    print(f"SoccerMon 去重后伤病人日总数: {len(df_inj)} | 日期范围: {df_inj['date'].min().date()} ~ {df_inj['date'].max().date()}")
    print(f"其中【新发损伤首日 (间隔>3天)】次数: {df_inj['is_onset'].sum()} 次")
    print(f"发生在 2020-06-01 (有GPS数据起始日) 之后的伤病总人日: {(df_inj['date'] >= '2020-06-01').sum()} | 新发次数: {(df_inj['is_onset'] & (df_inj['date'] >= '2020-06-01')).sum()}")

    # 检查 GPS 实际覆盖的 (player_name, date)
    gps_pairs = set()
    for p in SOCCERMON_DIR.rglob("*.parquet"):
        stem = p.stem
        if len(stem) > 11 and stem[4] == "-" and stem[7] == "-":
            gps_pairs.add((stem[11:], pd.to_datetime(stem[:10])))

    # 检查伤病前 7 天内是否有 GPS 记录
    has_gps_prev7 = 0
    for _, row in df_inj.iterrows():
        p_name, d = row["player_name"], row["date"]
        if any((p_name, d - pd.Timedelta(days=k)) in gps_pairs for k in range(0, 8)):
            has_gps_prev7 += 1
    print(f"伤病发生前 7 天内拥有至少 1 次 GPS 训练记录的伤病数: {has_gps_prev7} / {len(df_inj)}")

    # 验证 SoccerMon 主观量表方向：对比伤病当天 vs 正常天的均值
    wellness_dir = SOCCERMON_DIR / "subjective" / "wellness"
    inj_set = set(zip(df_inj["player_name"], df_inj["date"]))
    print("\n📐 SoccerMon 量表方向验证 (对比【正常日】vs【伤病日】的平均值):")
    for metric in ["fatigue", "soreness", "sleep_quality", "stress", "mood", "readiness"]:
        f_csv = wellness_dir / f"{metric}.csv"
        df_m = pd.read_csv(f_csv)
        date_col = df_m.columns[0]
        df_m["date"] = pd.to_datetime(df_m[date_col], format="%d.%m.%Y", errors="coerce")
        df_long = df_m.melt(id_vars=["date"], value_vars=[c for c in df_m.columns[1:] if c != "date"],
                            var_name="player_name", value_name="val").dropna(subset=["val", "date"])
        df_long["val"] = pd.to_numeric(df_long["val"], errors="coerce")
        df_long["is_inj"] = df_long.apply(lambda r: (r["player_name"], r["date"]) in inj_set, axis=1)
        m_norm = df_long.loc[~df_long["is_inj"], "val"].mean()
        m_inj = df_long.loc[df_long["is_inj"], "val"].mean()
        print(f"  - {metric:15s}: 正常日均值 = {m_norm:.3f} | 伤病日均值 = {m_inj:.3f} ({'伤病日更低⬇️' if m_inj < m_norm else '伤病日更高⬆️'})")


def clean_name(raw_name: str) -> str:
    """提取规范化内部标识，去除球衣号、空格和重复导出后缀。"""
    if pd.isna(raw_name):
        return ""
    s = str(raw_name).strip()
    s = re.sub(r"^\d+号[-—_\s]*", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def verify_local_24():
    print("\n" + "=" * 75)
    print("🔍 2. 验证本地 24 人数据集：球员姓名对齐、伤病类型分布与量表正反向")
    print("=" * 75)

    # 1) 从 00_原始客观数据 提取姓名
    obj_files = list((LOCAL_DATA_DIR / "00_原始客观数据").rglob("*.xlsx"))
    obj_players = {clean_name(f.stem) for f in obj_files}

    # 2) 从 01_原始主观数据/晨间 提取姓名并汇总所有晨间记录
    morning_files = sorted((LOCAL_DATA_DIR / "01_原始主观数据" / "晨间").rglob("*.xlsx"))
    morning_dfs = []
    for mf in morning_files:
        df_tmp = pd.read_excel(mf)
        # 从文件名提取真实日期 (如 3.25.xlsx -> 2026-03-25)
        parts = mf.stem.split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            file_date = pd.Timestamp(year=2026, month=int(parts[0]), day=int(parts[1]))
        else:
            file_date = pd.NaT
        df_tmp["file_date"] = file_date
        morning_dfs.append(df_tmp)
    df_morning = pd.concat(morning_dfs, ignore_index=True)
    name_col = [c for c in df_morning.columns if "姓名" in c][0]
    df_morning["player_clean"] = df_morning[name_col].apply(clean_name)
    subj_players = set(df_morning["player_clean"].unique())

    # 3) 从 02_基线数据/女足损伤记录表.xlsx 提取姓名
    df_loc_inj = pd.read_excel(LOCAL_DATA_DIR / "02_基线数据" / "女足损伤记录表.xlsx")
    df_loc_inj["player_clean"] = df_loc_inj["球员姓名"].apply(clean_name)
    df_loc_inj["date"] = pd.to_datetime(df_loc_inj["受伤日期"])
    inj_players = set(df_loc_inj["player_clean"].unique())

    print(f"客观传感器球员数: {len(obj_players)} -> {sorted(obj_players)}")
    print(f"晨间主观问卷球员数: {len(subj_players)} -> {sorted(subj_players)}")
    print(f"损伤记录表涉及球员数: {len(inj_players)}")
    print(f"客观 vs 主观 姓名差异: {obj_players ^ subj_players}")
    print(f"损伤表姓名是否全部在全队名单中: {inj_players.issubset(obj_players | subj_players)} (未匹配: {inj_players - (obj_players | subj_players)})")

    # 验证本地晨间量表方向
    inj_loc_set = set(zip(df_loc_inj["player_clean"], df_loc_inj["date"]))
    df_morning["is_inj"] = df_morning.apply(lambda r: (r["player_clean"], r["file_date"]) in inj_loc_set, axis=1)

    print("\n📐 本地晨间问卷 1~7 分量表方向验证 (对比【正常日】vs【伤病日】的平均值):")
    for key in ["睡眠质量", "疲劳感", "肌肉酸痛", "心理压力"]:
        col = [c for c in df_morning.columns if key in c][0]
        vals_norm = pd.to_numeric(df_morning.loc[~df_morning["is_inj"], col], errors="coerce")
        vals_inj = pd.to_numeric(df_morning.loc[df_morning["is_inj"], col], errors="coerce")
        print(f"  - {key:10s}: 正常日均值 = {vals_norm.mean():.3f} | 伤病日均值 = {vals_inj.mean():.3f} ({'伤病日更高⬆️' if vals_inj.mean() > vals_norm.mean() else '伤病日更低⬇️'})")


if __name__ == "__main__":
    verify_soccermon()
    verify_local_24()
    print("\n🎉 步骤 02 验证完成！请把控制台结果发给我。")
