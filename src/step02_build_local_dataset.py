# src/step02_build_local_dataset.py
import re
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "local_24"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_player_name(raw_name: str) -> str:
    """移除球衣号、空格、重复文件后缀和导出时间戳，得到规范化内部标识。"""
    if pd.isna(raw_name):
        return ""
    s = str(raw_name).strip()
    # 去掉前缀球衣号，如 '16号-'
    s = re.sub(r"^\d+号[-—_\s]*", "", s)
    # 去掉文件名末尾的括号或时间戳数字，如 '(1)'、'09.52.05'
    s = re.sub(r"[\(\（]\d+[\)\）]", "", s)
    s = re.sub(r"[\d\._-]+$", "", s)
    # 去掉所有空格
    s = re.sub(r"\s+", "", s)
    return s


def extract_date_from_folder_or_name(fpath: Path) -> pd.Timestamp:
    """从月/日目录或文件名提取标准日期 2026-MM-DD。"""
    for candidate in [fpath.stem, fpath.parent.name]:
        m = re.match(r"^(\d{1,2})\.(\d{1,2})$", candidate.strip())
        if m:
            return pd.Timestamp(year=2026, month=int(m.group(1)), day=int(m.group(2)))
    return pd.NaT


def process_objective_files() -> pd.DataFrame:
    print("=" * 75)
    print("🏃‍♀️ 1. 正在汇总本地 865 个客观穿戴设备文件 (1Hz 物理量纲对齐与同日去重)...")
    print("=" * 75)
    obj_dir = LOCAL_DATA_DIR / "00_原始客观数据"
    all_files = sorted([f for f in obj_dir.rglob("*.xlsx") if not f.name.startswith(".")])

    # 按 (player_name, date) 分组，处理同一天存在原文件和“(1)”重复导出文件的情况
    grouped_files = {}
    for idx, f in enumerate(all_files, 1):
        p_name = clean_player_name(f.stem)
        f_date = extract_date_from_folder_or_name(f)
        if p_name and pd.notna(f_date):
            grouped_files.setdefault((p_name, f_date), []).append(f)

    print(f"共扫描到 {len(all_files)} 个 Excel 文件，归并为 {len(grouped_files)} 个【球员-训练日】。")

    records = []
    total_groups = len(grouped_files)
    for idx, ((p_name, f_date), flist) in enumerate(sorted(grouped_files.items()), 1):
        if idx % 100 == 0 or idx == total_groups:
            print(f"  进度: {idx} / {total_groups} 个球员训练日已处理...")

        dfs = []
        for f in flist:
            try:
                df_tmp = pd.read_excel(f)
                dfs.append(df_tmp)
            except Exception as e:
                print(f"  ⚠️ 读取失败 {f}: {e}")
        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)
        # 按时间戳去重（防止重名导出文件 (1) / 09.52.05 造成负荷翻倍）
        if "时间" in df.columns:
            df = df.drop_duplicates(subset=["时间"]).sort_values("时间")

        n_seconds = len(df)
        if n_seconds < 60:  # 少于1分钟视为无效记录
            continue

        # 1) 速度 (m/s)，过滤异常 GPS 漂移 (>10.5 m/s)
        spd = pd.to_numeric(df.get("速度 m/s", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        spd = spd.clip(lower=0.0, upper=10.5)

        # 因为采样率是 1Hz (每秒1行)，每秒的速度(m/s)之和即为总距离(m)
        total_distance_m = float(spd.sum())
        hsr_distance_m = float(spd[spd >= 4.0].sum())       # 高速跑距离 (>=4.0 m/s, 即 14.4 km/h)
        sprint_distance_m = float(spd[spd >= 5.5].sum())    # 冲刺距离 (>=5.5 m/s, 即 19.8 km/h)
        max_speed_ms = float(spd.quantile(0.995)) if len(spd) > 10 else float(spd.max())
        mean_speed_ms = float(spd.mean())

        # 加减速特征 (1Hz 差分 m/s^2)
        acc_1hz = spd.diff().fillna(0.0)
        high_accel_sec = int((acc_1hz >= 1.5).sum())        # 急加速秒数 (>=1.5 m/s^2)
        high_decel_sec = int((acc_1hz <= -1.5).sum())       # 急减速秒数 (<=-1.5 m/s^2)

        # 2) 心率 (bpm)，过滤未佩戴好的 0 值或 <40 异常值
        hr = pd.to_numeric(df.get("心率", pd.Series(dtype=float)), errors="coerce")
        hr_valid = hr[(hr >= 45) & (hr <= 220)]
        hr_mean = float(hr_valid.mean()) if len(hr_valid) > 30 else np.nan
        hr_max = float(hr_valid.quantile(0.99)) if len(hr_valid) > 30 else np.nan
        hr_high_sec = int((hr_valid >= 170).sum()) if len(hr_valid) > 30 else 0
        hr_high_ratio = float(hr_high_sec / len(hr_valid)) if len(hr_valid) > 30 else np.nan

        # 3) IMU 加速度计与陀螺仪：消除 4096 寄存器量纲，转换为标准 g 单位动态负荷
        ax = pd.to_numeric(df.get("加速度计X", 0), errors="coerce").fillna(0.0)
        ay = pd.to_numeric(df.get("加速度计Y", 0), errors="coerce").fillna(0.0)
        az = pd.to_numeric(df.get("加速度计Z", 0), errors="coerce").fillna(0.0)
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        # 用中位数模长作为 1g 重力基准 (正常约 4096)
        g_base = float(acc_mag.median())
        if g_base < 100:
            g_base = 4096.0
        acc_mag_g = acc_mag / g_base
        dyn_acc_g = np.abs(acc_mag_g - 1.0)  # 去除 1g 重力后的动态加速度 (单位: g)

        imu_dyn_mean_g = float(dyn_acc_g.mean())
        imu_dyn_p95_g = float(dyn_acc_g.quantile(0.95))
        imu_high_impact_sec = int((dyn_acc_g >= 0.5).sum())  # 动态冲击 >= 0.5g 的秒数

        # 陀螺仪模长 (除以 1000 归一化量级)
        gx = pd.to_numeric(df.get("陀螺仪X", 0), errors="coerce").fillna(0.0)
        gy = pd.to_numeric(df.get("陀螺仪Y", 0), errors="coerce").fillna(0.0)
        gz = pd.to_numeric(df.get("陀螺仪Z", 0), errors="coerce").fillna(0.0)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2) / 1000.0
        gyro_mean = float(gyro_mag.mean())
        gyro_p95 = float(gyro_mag.quantile(0.95))

        records.append({
            "player_id": p_name,
            "date": f_date,
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
        })

    return pd.DataFrame(records)


def process_subjective_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 75)
    print("📝 2. 正在汇总本地晨间 Wellness 问卷与训练后 sRPE 问卷...")
    print("=" * 75)

    # --- A. 晨间问卷 ---
    morning_files = sorted((LOCAL_DATA_DIR / "01_原始主观数据" / "晨间").rglob("*.xlsx"))
    m_rows = []
    likert_map = {"完全没有": 0, "有一点": 1, "中等程度": 2, "比较强烈": 3, "极其强烈": 4}
    pos_mood_keys = ["精力充沛", "充满活力", "活跃", "清醒"]
    neg_mood_keys = ["焦虑", "迷茫", "疲倦", "沮丧", "恼怒", "疲惫", "昏昏欲睡", "愤愤不平", "不开心", "惊恐", "忧虑", "苦闷", "困惑", "紧张", "暴躁", "筋疲力尽", "脾气坏", "意志消沉", "不知所措"]

    for mf in morning_files:
        if mf.name.startswith("."):
            continue
        f_date = extract_date_from_folder_or_name(mf)
        df_m = pd.read_excel(mf)
        name_col = [c for c in df_m.columns if "姓名" in c][0]
        c_sleep = [c for c in df_m.columns if "睡眠质量" in c][0]
        c_fatigue = [c for c in df_m.columns if "疲劳感" in c][0]
        c_sore = [c for c in df_m.columns if "肌肉酸痛" in c][0]
        c_stress = [c for c in df_m.columns if "心理压力" in c][0]
        c_mens = [c for c in df_m.columns if "生理期" in c][0]
        c_dys = [c for c in df_m.columns if "痛经" in c][0]

        for _, row in df_m.iterrows():
            p_name = clean_player_name(row[name_col])
            if not p_name:
                continue
            sleep_raw = pd.to_numeric(row[c_sleep], errors="coerce")     # 1(好)~7(差)
            fatigue_raw = pd.to_numeric(row[c_fatigue], errors="coerce") # 1(好)~7(差)
            sore_raw = pd.to_numeric(row[c_sore], errors="coerce")       # 1(好)~7(差)
            stress_raw = pd.to_numeric(row[c_stress], errors="coerce")   # 1(好)~7(差)

            is_mens = 1 if str(row[c_mens]).strip() == "是" else 0
            dys_score = pd.to_numeric(row[c_dys], errors="coerce") if is_mens else 0.0
            if pd.isna(dys_score):
                dys_score = 0.0

            # 计算 BRUMS 正向活力得分与负向情绪得分 (0~4分)
            pos_vals = [likert_map.get(str(row[c]).strip(), np.nan) for c in df_m.columns if any(k in c for k in pos_mood_keys)]
            neg_vals = [likert_map.get(str(row[c]).strip(), np.nan) for c in df_m.columns if any(k in c for k in neg_mood_keys)]
            pos_mean = np.nanmean(pos_vals) if pos_vals else 2.0
            neg_mean = np.nanmean(neg_vals) if neg_vals else 0.0

            # 合成与 [0, 1] 对齐的不良状态指数 (越高代表状态越差/风险越高):
            # 将 1~7 分线性映射到 0~1: (x - 1) / 6.0
            m_rows.append({
                "player_id": p_name,
                "date": f_date,
                "sleep_poor_raw": sleep_raw,
                "fatigue_raw": fatigue_raw,
                "soreness_raw": sore_raw,
                "stress_raw": stress_raw,
                "sleep_poor_01": (sleep_raw - 1.0) / 6.0 if pd.notna(sleep_raw) else np.nan,
                "fatigue_01": (fatigue_raw - 1.0) / 6.0 if pd.notna(fatigue_raw) else np.nan,
                "soreness_01": (sore_raw - 1.0) / 6.0 if pd.notna(sore_raw) else np.nan,
                "stress_01": (stress_raw - 1.0) / 6.0 if pd.notna(stress_raw) else np.nan,
                "mood_poor_01": np.clip((neg_mean / 4.0) * 0.7 + (1.0 - pos_mean / 4.0) * 0.3, 0.0, 1.0),
                "is_menstruation": is_mens,
                "dysmenorrhea_score": float(dys_score),
            })

    df_morning_all = pd.DataFrame(m_rows).drop_duplicates(subset=["player_id", "date"], keep="last")

    # --- B. 训练后问卷 (提取时长、RPE、sRPE Daily Load) ---
    post_files = sorted((LOCAL_DATA_DIR / "01_原始主观数据" / "训练后").rglob("*.xlsx"))
    p_rows = []
    for pf in post_files:
        if pf.name.startswith("."):
            continue
        f_date = extract_date_from_folder_or_name(pf)
        df_p = pd.read_excel(pf)
        name_col = [c for c in df_p.columns if "姓名" in c][0]
        c_type = [c for c in df_p.columns if "活动类型" in c][0]
        c_dur = [c for c in df_p.columns if "总时长" in c][0]
        c_rpe = [c for c in df_p.columns if "多累" in c][0]

        for _, row in df_p.iterrows():
            p_name = clean_player_name(row[name_col])
            if not p_name:
                continue
            # 提取分钟数 (如 '90分钟' -> 90.0)
            dur_str = str(row[c_dur])
            dur_match = re.search(r"(\d+(\.\d+)?)", dur_str)
            dur_min = float(dur_match.group(1)) if dur_match else np.nan

            # 提取 RPE 分值 (如 '3 (中等)' -> 3.0, '0.5 (极轻微)' -> 0.5)
            rpe_str = str(row[c_rpe])
            rpe_match = re.search(r"(\d+(\.\d+)?)", rpe_str)
            rpe_val = float(rpe_match.group(1)) if rpe_match else np.nan

            is_match = 1 if "比赛" in str(row[c_type]) else 0
            srpe_load = dur_min * rpe_val if (pd.notna(dur_min) and pd.notna(rpe_val)) else np.nan

            p_rows.append({
                "player_id": p_name,
                "date": f_date,
                "post_duration_min": dur_min,
                "rpe": rpe_val,
                "srpe_daily_load": srpe_load,
                "is_match_day": is_match,
            })

    df_post_all = pd.DataFrame(p_rows).drop_duplicates(subset=["player_id", "date"], keep="last")
    print(f"晨间问卷有效人日: {len(df_morning_all)} | 训练后问卷有效人日: {len(df_post_all)}")
    return df_morning_all, df_post_all


def build_local_panel():
    df_obj = process_objective_files()
    df_morn, df_post = process_subjective_files()

    # 读取基线信息与既往伤病史
    df_hist = pd.read_excel(LOCAL_DATA_DIR / "02_基线数据" / "东北师范大学女足基线健康与既往损伤史调查问卷xlsx.xlsx")
    name_col_h = [c for c in df_hist.columns if "姓名" in c][0]
    c_prev12 = [c for c in df_hist.columns if "12 个月" in c and "是否" in c][0]
    c_acl = [c for c in df_hist.columns if "ACL" in c and "是否" in c][0]
    c_major = [c for c in df_hist.columns if "重大伤病" in c and "是否" in c][0]

    baseline_rows = []
    for _, r in df_hist.iterrows():
        raw_n = str(r[name_col_h]).strip()
        num_m = re.match(r"^(\d+号)", raw_n)
        jersey_id = num_m.group(1) if num_m else ""
        p_name = clean_player_name(raw_n)
        has_prev_inj = 0 if "没有" in str(r[c_prev12]) else 1
        has_acl_or_major = 0 if ("否" in str(r[c_acl]) and "没有" in str(r[c_major])) else 1
        baseline_rows.append({
            "player_id": p_name,
            "jersey_no": jersey_id,
            "hist_inj_12m": has_prev_inj,
            "hist_major_inj": has_acl_or_major,
        })
    df_base = pd.DataFrame(baseline_rows)

    # 合并身高体重
    df_info = pd.read_excel(LOCAL_DATA_DIR / "02_基线数据" / "匿名女足基准信息.xlsx")
    df_info["jersey_no"] = df_info["姓名"].astype(str).str.strip()
    df_base = df_base.merge(df_info[["jersey_no", "身高(cm)", "体重（kg)", "主要位置"]], on="jersey_no", how="left")
    df_base["bmi"] = df_base["体重（kg)"] / ((df_base["身高(cm)"] / 100.0) ** 2)

    # 构建完整日期网格 (24名球员 × 每日日期，以便准确计算 7天/28天滚动负荷 ATL/CTL/ACWR)
    all_players = sorted(df_base["player_id"].unique())
    min_date = min(df_obj["date"].min(), df_morn["date"].min(), df_post["date"].min())
    max_date = max(df_obj["date"].max(), df_morn["date"].max(), df_post["date"].max())
    full_grid = pd.MultiIndex.from_product(
        [all_players, pd.date_range(min_date, max_date, freq="D")],
        names=["player_id", "date"]
    ).to_frame().reset_index(drop=True)

    panel = (
        full_grid
        .merge(df_base, on="player_id", how="left")
        .merge(df_morn, on=["player_id", "date"], how="left")
        .merge(df_post, on=["player_id", "date"], how="left")
        .merge(df_obj, on=["player_id", "date"], how="left")
    )

    # 标记当天是否有监测数据（晨间、训练后或GPS任意一项存在即为活跃监测日）
    panel["is_active_day"] = (
        panel["fatigue_01"].notna() | panel["srpe_daily_load"].notna() | panel["total_distance_m"].notna()
    ).astype(int)

    # 匹配损伤记录表
    df_inj = pd.read_excel(LOCAL_DATA_DIR / "02_基线数据" / "女足损伤记录表.xlsx")
    df_inj["player_id"] = df_inj["球员姓名"].apply(clean_player_name)
    df_inj["date"] = pd.to_datetime(df_inj["受伤日期"])

    # 区分【纯外力急性碰撞/踢伤】与【非接触/过度使用/扭拉伤】
    contact_keywords = ["踢伤", "接触性", "碰撞", "摔倒", "摔伤", "杵到", "磕碰", "撞击", "岔气"]
    def check_is_contact(row):
        txt = f"{row.get('受伤部位', '')}_{row.get('损伤类型', '')}"
        # 如果包含过度使用，仍归为可预测的负荷相关损伤
        if "过度使用" in txt:
            return False
        return any(k in txt for k in contact_keywords)

    df_inj["is_pure_contact"] = df_inj.apply(check_is_contact, axis=1)
    df_inj["is_time_loss"] = ~df_inj["严重程度"].astype(str).str.contains("0|轻", regex=True)

    all_inj_set = set(zip(df_inj["player_id"], df_inj["date"]))
    non_contact_set = set(zip(df_inj.loc[~df_inj["is_pure_contact"], "player_id"], df_inj.loc[~df_inj["is_pure_contact"], "date"]))
    timeloss_set = set(zip(df_inj.loc[df_inj["is_time_loss"], "player_id"], df_inj.loc[df_inj["is_time_loss"], "date"]))

    panel["inj_all_onset"] = panel.apply(lambda r: int((r["player_id"], r["date"]) in all_inj_set), axis=1)
    panel["inj_non_contact_onset"] = panel.apply(lambda r: int((r["player_id"], r["date"]) in non_contact_set), axis=1)
    panel["inj_timeloss_onset"] = panel.apply(lambda r: int((r["player_id"], r["date"]) in timeloss_set), axis=1)

    # 生成未来 1天、3天、7天 损伤预警标签 (用于风险预测模型验证)
    panel = panel.sort_values(["player_id", "date"]).reset_index(drop=True)
    for horizon in [1, 3, 7]:
        # 未来 [0, horizon] 天内是否发生非接触性损伤
        panel[f"target_non_contact_next_{horizon}d"] = (
            panel.groupby("player_id")["inj_non_contact_onset"]
            .transform(lambda s: s.rolling(window=horizon + 1, min_periods=1).max().shift(-horizon).fillna(s))
            .astype(int)
        )
        panel[f"target_all_next_{horizon}d"] = (
            panel.groupby("player_id")["inj_all_onset"]
            .transform(lambda s: s.rolling(window=horizon + 1, min_periods=1).max().shift(-horizon).fillna(s))
            .astype(int)
        )

    out_csv = PROCESSED_DIR / "local_daily_panel.csv"
    panel.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 75)
    print(f"✅ 本地 24 人每日面板数据已生成并保存至: {out_csv}")
    print("=" * 75)
    active_panel = panel[panel["is_active_day"] == 1]
    print(f"日期跨度: {min_date.date()} ~ {max_date.date()} | 总网格行数: {len(panel)} | 活跃监测人日: {len(active_panel)}")
    print(f"其中包含客观 GPS 数据的训练人日: {panel['total_distance_m'].notna().sum()}")
    print(f"其中包含晨间 Wellness 的人日: {panel['fatigue_01'].notna().sum()}")
    print(f"其中包含训练后 sRPE 的人日: {panel['srpe_daily_load'].notna().sum()}")
    print("\n📊 本地损伤标签统计 (活跃监测日内):")
    for col in ["inj_all_onset", "inj_non_contact_onset", "inj_timeloss_onset",
                "target_non_contact_next_3d", "target_non_contact_next_7d"]:
        pos = active_panel[col].sum()
        print(f"  - {col:30s}: 正样本数 = {pos:4d} / {len(active_panel)} (占比 {pos/len(active_panel)*100:.2f}%)")


if __name__ == "__main__":
    build_local_panel()
