# src/step02b_build_soccermon_dataset.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOCCERMON_DIR = PROJECT_ROOT / "data" / "raw" / "soccermon"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_wide_csv(csv_path: Path, value_name: str) -> pd.DataFrame:
    """读取 SoccerMon 宽表 (第一列为日期 dd.mm.yyyy，其余列为球员 ID) 并转为长表"""
    df = pd.read_csv(csv_path)
    date_col = df.columns[0]
    df["date"] = pd.to_datetime(df[date_col], format="%d.%m.%Y", errors="coerce")
    player_cols = [c for c in df.columns[1:] if c != "date"]
    df_long = df.melt(id_vars=["date"], value_vars=player_cols, var_name="player_id", value_name=value_name)
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")
    return df_long.dropna(subset=["date"]).drop_duplicates(subset=["player_id", "date"])


def process_soccermon_subjective() -> pd.DataFrame:
    print("=" * 75)
    print("📝 1. 正在读取并翻转对齐 SoccerMon 主观 Wellness 与 sRPE 负荷数据...")
    print("=" * 75)
    w_dir = SOCCERMON_DIR / "subjective" / "wellness"
    tl_dir = SOCCERMON_DIR / "subjective" / "training-load"

    df_fatigue = load_wide_csv(w_dir / "fatigue.csv", "sm_fatigue")
    df_sore = load_wide_csv(w_dir / "soreness.csv", "sm_soreness")
    df_sleep_q = load_wide_csv(w_dir / "sleep_quality.csv", "sm_sleep_quality")
    df_sleep_d = load_wide_csv(w_dir / "sleep_duration.csv", "sm_sleep_duration")
    df_stress = load_wide_csv(w_dir / "stress.csv", "sm_stress")
    df_mood = load_wide_csv(w_dir / "mood.csv", "sm_mood")
    df_readiness = load_wide_csv(w_dir / "readiness.csv", "sm_readiness")
    df_dload = load_wide_csv(tl_dir / "daily_load.csv", "srpe_daily_load")

    # 合并所有主观与负荷表
    panel = df_fatigue
    for sub_df in [df_sore, df_sleep_q, df_sleep_d, df_stress, df_mood, df_readiness, df_dload]:
        panel = panel.merge(sub_df, on=["player_id", "date"], how="outer")

    # 在 SoccerMon 的 daily_load.csv 中，未训练/未填报常被记为 0.0；将 <=0 设为 NaN 以便与本地一致区分有效记录
    panel.loc[panel["srpe_daily_load"] <= 0, "srpe_daily_load"] = np.nan

    # 核心对齐步骤：将 SoccerMon 1(最差)~5(最好) 翻转为 0(最好)~1(最差)，与本地 1(最好)~7(最差) 完全同向！
    panel["sleep_poor_01"] = (5.0 - panel["sm_sleep_quality"].clip(1, 5)) / 4.0
    panel["fatigue_01"] = (5.0 - panel["sm_fatigue"].clip(1, 5)) / 4.0
    panel["soreness_01"] = (5.0 - panel["sm_soreness"].clip(1, 5)) / 4.0
    panel["stress_01"] = (5.0 - panel["sm_stress"].clip(1, 5)) / 4.0
    # 融合 mood (1~5) 与 readiness (1~10) 为综合情绪/准备度不良指数
    mood_poor = (5.0 - panel["sm_mood"].clip(1, 5)) / 4.0
    unready = (10.0 - panel["sm_readiness"].clip(1, 10)) / 9.0
    panel["mood_poor_01"] = np.where(mood_poor.notna() & unready.notna(), 0.5 * mood_poor + 0.5 * unready,
                                     mood_poor.fillna(unready))

    print(f"主观面板行数: {len(panel)} | 涉及球员数: {panel['player_id'].nunique()}")
    return panel


def parse_single_parquet(fpath: Path) -> dict | None:
    """读取单个 100Hz Parquet 文件，降采样至 1Hz 并提取与本地完全同构的物理特征"""
    stem = fpath.stem
    if len(stem) <= 11 or stem[4] != "-" or stem[7] != "-":
        return None
    date_str = stem[:10]
    player_id = stem[11:]

    cols_needed = ["speed", "heart_rate", "accl_x", "accl_y", "accl_z", "gyro_x", "gyro_y", "gyro_z"]
    try:
        df = pd.read_parquet(fpath, columns=cols_needed)
    except Exception:
        return None

    # 100Hz -> 1Hz 降采样 (每隔 100 行取 1 个秒级观测点，与本地 1Hz 设备完全对齐)
    df_1hz = df.iloc[::100].reset_index(drop=True)
    n_seconds = len(df_1hz)
    if n_seconds < 60:
        return None

    # 1) 速度 (m/s)
    spd = pd.to_numeric(df_1hz["speed"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=10.5)
    total_distance_m = float(spd.sum())
    hsr_distance_m = float(spd[spd >= 4.0].sum())
    sprint_distance_m = float(spd[spd >= 5.5].sum())
    max_speed_ms = float(spd.quantile(0.995)) if n_seconds > 10 else float(spd.max())
    mean_speed_ms = float(spd.mean())

    acc_1hz = spd.diff().fillna(0.0)
    high_accel_sec = int((acc_1hz >= 1.5).sum())
    high_decel_sec = int((acc_1hz <= -1.5).sum())

    # 2) 心率 (bpm)
    hr = pd.to_numeric(df_1hz["heart_rate"], errors="coerce")
    hr_valid = hr[(hr >= 45) & (hr <= 220)]
    hr_mean = float(hr_valid.mean()) if len(hr_valid) > 30 else np.nan
    hr_max = float(hr_valid.quantile(0.99)) if len(hr_valid) > 30 else np.nan
    hr_high_sec = int((hr_valid >= 170).sum()) if len(hr_valid) > 30 else 0
    hr_high_ratio = float(hr_high_sec / len(hr_valid)) if len(hr_valid) > 30 else np.nan

    # 3) IMU 加速度计 (重力基准归一化为 g)
    ax = pd.to_numeric(df_1hz["accl_x"], errors="coerce").fillna(0.0)
    ay = pd.to_numeric(df_1hz["accl_y"], errors="coerce").fillna(0.0)
    az = pd.to_numeric(df_1hz["accl_z"], errors="coerce").fillna(0.0)
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    g_base = float(acc_mag.median())
    if g_base < 0.2:
        g_base = 1.0
    acc_mag_g = acc_mag / g_base
    dyn_acc_g = np.abs(acc_mag_g - 1.0)

    imu_dyn_mean_g = float(dyn_acc_g.mean())
    imu_dyn_p95_g = float(dyn_acc_g.quantile(0.95))
    imu_high_impact_sec = int((dyn_acc_g >= 0.5).sum())

    gx = pd.to_numeric(df_1hz["gyro_x"], errors="coerce").fillna(0.0)
    gy = pd.to_numeric(df_1hz["gyro_y"], errors="coerce").fillna(0.0)
    gz = pd.to_numeric(df_1hz["gyro_z"], errors="coerce").fillna(0.0)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    gyro_mean = float(gyro_mag.mean())
    gyro_p95 = float(gyro_mag.quantile(0.95))

    return {
        "player_id": player_id,
        "date": pd.to_datetime(date_str),
        "gps_duration_min": round(n_seconds / 60.0, 2),
        "total_distance_m": round(total_distance_m, 2),
        "hsr_distance_m": round(hsr_distance_m, 2),
        "sprint_distance_m": round(sprint_distance_m, 2),
        "max_speed_ms": round(max_speed_ms, 3),
        "mean_speed_ms": round(mean_speed_ms, 3),
        "high_accel_sec": high_accel_sec,
        "high_decel_sec": high_decel_sec,
        "hr_mean": round(hr_mean, 2) if pd.notna(hr_mean) else np.nan,
        "hr_max": round(hr_max, 2) if pd.notna(hr_max) else np.nan,
        "hr_high_sec": hr_high_sec,
        "hr_high_ratio": round(hr_high_ratio, 4) if pd.notna(hr_high_ratio) else np.nan,
        "imu_dyn_mean_g": round(imu_dyn_mean_g, 4),
        "imu_dyn_p95_g": round(imu_dyn_p95_g, 4),
        "imu_high_impact_sec": imu_high_impact_sec,
        "gyro_mean": round(gyro_mean, 4),
        "gyro_p95": round(gyro_p95, 4),
    }


def process_soccermon_gps(valid_players: set) -> pd.DataFrame:
    print("\n" + "=" * 75)
    print("🏃‍♀️ 2. 正在并发提取 SoccerMon GPS+IMU Parquet 文件 (100Hz -> 1Hz 物理对齐)...")
    print("=" * 75)
    all_parquets = sorted(SOCCERMON_DIR.rglob("*.parquet"))
    # 只处理在主观/伤病名单中存在的球员文件，省去无标签的冗余读取
    target_parquets = [
        p for p in all_parquets
        if len(p.stem) > 11 and p.stem[11:] in valid_players
    ]
    print(f"总 Parquet 文件: {len(all_parquets)} | 属于主观/伤病监测名单球员的文件: {len(target_parquets)}")

    gps_records = []
    total = len(target_parquets)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(parse_single_parquet, p): p for p in target_parquets}
        for idx, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is not None:
                gps_records.append(res)
            if idx % 500 == 0 or idx == total:
                print(f"  进度: {idx} / {total} 个 Parquet 文件已处理...")

    df_gps = pd.DataFrame(gps_records)
    if not df_gps.empty:
        # 若同一球员同一天有多个训练文件，按天聚合（距离/时长求和，极值取最大，均值取平均）
        df_gps = df_gps.groupby(["player_id", "date"], as_index=False).agg({
            "gps_duration_min": "sum",
            "total_distance_m": "sum",
            "hsr_distance_m": "sum",
            "sprint_distance_m": "sum",
            "max_speed_ms": "max",
            "mean_speed_ms": "mean",
            "high_accel_sec": "sum",
            "high_decel_sec": "sum",
            "hr_mean": "mean",
            "hr_max": "max",
            "hr_high_sec": "sum",
            "hr_high_ratio": "mean",
            "imu_dyn_mean_g": "mean",
            "imu_dyn_p95_g": "max",
            "imu_high_impact_sec": "sum",
            "gyro_mean": "mean",
            "gyro_p95": "max",
        })
    return df_gps


def build_soccermon_panel():
    df_subj = process_soccermon_subjective()

    # 读取伤病表
    df_inj = pd.read_csv(SOCCERMON_DIR / "subjective" / "injury" / "injury.csv")
    df_inj["date"] = pd.to_datetime(df_inj["timestamp"], format="%d.%m.%Y", errors="coerce")
    df_inj = df_inj.dropna(subset=["date"]).sort_values(["player_id" if "player_id" in df_inj.columns else "player_name", "date"])
    df_inj = df_inj.rename(columns={"player_name": "player_id"}).drop_duplicates(["player_id", "date"])

    # 识别新发损伤首日 (距离该球员上一次上报伤病 > 3 天)
    df_inj["prev_date"] = df_inj.groupby("player_id")["date"].shift(1)
    df_inj["days_since_prev"] = (df_inj["date"] - df_inj["prev_date"]).dt.days
    df_inj["is_onset"] = df_inj["days_since_prev"].isna() | (df_inj["days_since_prev"] > 3)
    df_inj["is_major"] = df_inj["type"].astype(str).str.contains("major", case=False)

    valid_players = set(df_subj["player_id"].unique()) | set(df_inj["player_id"].unique())
    df_gps = process_soccermon_gps(valid_players)

    # 合并主观与客观数据
    panel = df_subj.merge(df_gps, on=["player_id", "date"], how="outer")

    # 过滤掉既无主观记录、又无sRPE、又无GPS的完全空白行
    panel["is_active_day"] = (
        panel["fatigue_01"].notna() | panel["srpe_daily_load"].notna() | panel["total_distance_m"].notna()
    ).astype(int)

    # 限定在每位球员首个活跃日至最后一个活跃日之间，确保滚动窗口连续
    active_bounds = panel[panel["is_active_day"] == 1].groupby("player_id")["date"].agg(["min", "max"]).reset_index()
    panel = panel.merge(active_bounds, on="player_id", how="inner")
    panel = panel[(panel["date"] >= panel["min"]) & (panel["date"] <= panel["max"])].drop(columns=["min", "max"])

    # 标注伤病标签
    all_inj_days = set(zip(df_inj["player_id"], df_inj["date"]))
    onset_days = set(zip(df_inj.loc[df_inj["is_onset"], "player_id"], df_inj.loc[df_inj["is_onset"], "date"]))
    major_days = set(zip(df_inj.loc[df_inj["is_major"], "player_id"], df_inj.loc[df_inj["is_major"], "date"]))

    panel["inj_prevalence"] = panel.apply(lambda r: int((r["player_id"], r["date"]) in all_inj_days), axis=1)
    panel["inj_non_contact_onset"] = panel.apply(lambda r: int((r["player_id"], r["date"]) in onset_days), axis=1)
    panel["inj_all_onset"] = panel["inj_non_contact_onset"]
    panel["inj_timeloss_onset"] = panel.apply(lambda r: int((r["player_id"], r["date"]) in major_days), axis=1)

    panel = panel.sort_values(["player_id", "date"]).reset_index(drop=True)
    for horizon in [1, 3, 7]:
        panel[f"target_non_contact_next_{horizon}d"] = (
            panel.groupby("player_id")["inj_non_contact_onset"]
            .transform(lambda s: s.rolling(window=horizon + 1, min_periods=1).max().shift(-horizon).fillna(s))
            .astype(int)
        )
        panel[f"target_prevalence_next_{horizon}d"] = (
            panel.groupby("player_id")["inj_prevalence"]
            .transform(lambda s: s.rolling(window=horizon + 1, min_periods=1).max().shift(-horizon).fillna(s))
            .astype(int)
        )

    out_csv = PROCESSED_DIR / "soccermon_daily_panel.csv"
    panel.to_csv(out_csv, index=False, encoding="utf-8-sig")

    active_panel = panel[panel["is_active_day"] == 1]
    print("\n" + "=" * 75)
    print(f"✅ SoccerMon 每日面板数据已生成并保存至: {out_csv}")
    print("=" * 75)
    print(f"总网格行数: {len(panel)} | 活跃监测人日: {len(active_panel)} | 涉及球员数: {active_panel['player_id'].nunique()}")
    print(f"其中包含客观 GPS 数据的训练人日: {active_panel['total_distance_m'].notna().sum()}")
    print(f"其中包含晨间 Wellness 的人日: {active_panel['fatigue_01'].notna().sum()}")
    print(f"其中包含训练后 sRPE 的人日: {active_panel['srpe_daily_load'].notna().sum()}")
    print("\n📊 SoccerMon 损伤标签统计 (活跃监测日内):")
    for col in ["inj_prevalence", "inj_non_contact_onset", "target_non_contact_next_3d",
                "target_non_contact_next_7d", "target_prevalence_next_3d", "target_prevalence_next_7d"]:
        pos = active_panel[col].sum()
        print(f"  - {col:30s}: 正样本数 = {pos:4d} / {len(active_panel)} (占比 {pos/len(active_panel)*100:.2f}%)")


if __name__ == "__main__":
    build_soccermon_panel()