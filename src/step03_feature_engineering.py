# src/step03_feature_engineering.py
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def compute_player_zscore(series: pd.Series, window: int = 21, min_periods: int = 3) -> pd.Series:
    """计算每位球员相对于自身历史基线的滚动 Z-Score (使用 shift(1) 避免同日稀释，纯历史基线)"""
    history = series.shift(1)
    hist_mean = history.rolling(window=window, min_periods=min_periods).mean()
    hist_std = history.rolling(window=window, min_periods=min_periods).std()
    # 若早期样本不足，用“截至前一日”的 expanding 结果兜底，避免当前日泄露
    exp_mean = history.expanding(min_periods=1).mean()
    exp_std = history.expanding(min_periods=2).std().fillna(1.0)
    mean_used = hist_mean.fillna(exp_mean)
    std_used = hist_std.fillna(exp_std).replace(0, 1.0).clip(lower=0.05)
    z = (series - mean_used) / std_used
    return z.clip(-4.0, 4.0)


def compute_acwr(series: pd.Series, acute_win: int = 7, chronic_win: int = 28) -> tuple[pd.Series, pd.Series, pd.Series]:
    """计算急性负荷(ATL)、慢性负荷(CTL)与急慢性负荷比(ACWR)"""
    s_filled = series.fillna(0.0)
    atl = s_filled.rolling(window=acute_win, min_periods=1).mean()
    ctl = s_filled.rolling(window=chronic_win, min_periods=3).mean()
    acwr = atl / (ctl + 1e-3)
    return atl, ctl, acwr.clip(0.0, 4.0)


def engineer_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    player_key = df["player_id"]
    features: dict[str, pd.Series] = {}

    def by_player_transform(series: pd.Series, func) -> pd.Series:
        """按运动员分组计算，并保持与当前排序后的 df 相同索引。"""
        return series.groupby(
            player_key, sort=False, group_keys=False
        ).transform(func)

    # ================= 1. 主观状态合成与个体 Z-Score =================
    subj_cols = ["sleep_poor_01", "fatigue_01", "soreness_01", "stress_01", "mood_poor_01"]
    features["wellness_poor_mean"] = df[subj_cols].mean(axis=1)
    features["wellness_poor_max"] = df[subj_cols].max(axis=1)
    # 疼痛-疲劳交互项 (既酸痛又疲劳时风险倍增)
    features["sore_x_fatigue"] = df["soreness_01"] * df["fatigue_01"]

    subject_inputs = {col: df[col] for col in subj_cols}
    subject_inputs["wellness_poor_mean"] = features["wellness_poor_mean"]

    for col in ["sleep_poor_01", "fatigue_01", "soreness_01", "stress_01", "mood_poor_01", "wellness_poor_mean"]:
        # 对主观缺失值先用球员前向填充最多3天，再算滚动Z-score和3日突增
        source = subject_inputs[col]
        s_ffill = by_player_transform(source, lambda s: s.ffill(limit=3))
        features[f"{col}_z"] = by_player_transform(
            source, lambda s: compute_player_zscore(s.ffill(limit=3))
        )
        features[f"{col}_delta3d"] = s_ffill - by_player_transform(
            source, lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        features[f"{col}_roll3d"] = by_player_transform(
            source, lambda s: s.rolling(3, min_periods=1).mean()
        )
        features[f"{col}_roll7d"] = by_player_transform(
            source, lambda s: s.rolling(7, min_periods=1).mean()
        )

    # ================= 2. 客观 GPS+IMU 无量纲比例与生物力学特征 =================
    features["hsr_ratio"] = df["hsr_distance_m"] / (df["total_distance_m"] + 1.0)
    features["sprint_ratio"] = df["sprint_distance_m"] / (df["total_distance_m"] + 1.0)
    features["decel_accel_ratio"] = df["high_decel_sec"] / (df["high_accel_sec"] + 1.0)
    features["decel_per_km"] = df["high_decel_sec"] / (df["total_distance_m"] / 1000.0 + 0.1)
    features["imu_impact_per_min"] = df["imu_high_impact_sec"] / (df["gps_duration_min"] + 1.0)
    # 心率-速度比 (内外负荷耦合效率：相同配速下心率越高代表生理负担越重)
    features["hr_speed_ratio"] = df["hr_mean"] / (df["mean_speed_ms"] + 0.5)

    for col in ["total_distance_m", "hsr_distance_m", "sprint_distance_m", "max_speed_ms",
                "high_accel_sec", "high_decel_sec", "hr_mean", "hr_high_ratio",
                "imu_dyn_mean_g", "imu_dyn_p95_g", "gyro_p95", "hr_speed_ratio"]:
        source = features[col] if col in features else df[col]
        features[f"{col}_z"] = by_player_transform(
            source, lambda s: compute_player_zscore(s)
        )

    # ================= 3. 急慢性负荷比 (ACWR)、单调性 (Monotony) 与应变 (Strain) =================
    # 构造统一日负荷指数 unified_load_01：分母只使用当日之前的个人历史 median。
    # 这样不会把未来训练量带入当前日特征，正式预测时可安全放入时间切分/CV。
    srpe_hist_median = by_player_transform(
        df["srpe_daily_load"],
        lambda s: s.shift(1).expanding(min_periods=1).median(),
    )
    gps_hist_median = by_player_transform(
        df["total_distance_m"],
        lambda s: s.shift(1).expanding(min_periods=1).median(),
    )
    srpe_norm = df["srpe_daily_load"] / (srpe_hist_median + 1.0)
    gps_norm = df["total_distance_m"] / (gps_hist_median + 1.0)
    features["unified_daily_load"] = pd.Series(np.where(
        srpe_norm.notna() & gps_norm.notna(), 0.5 * srpe_norm + 0.5 * gps_norm,
        srpe_norm.fillna(gps_norm).fillna(0.0)
    ), index=df.index)

    for load_col, prefix in [
        ("unified_daily_load", "uload"),
        ("srpe_daily_load", "srpe"),
        ("total_distance_m", "dist"),
        ("hsr_distance_m", "hsr"),
        ("high_decel_sec", "decel"),
        ("imu_high_impact_sec", "impact"),
    ]:
        source = features[load_col] if load_col in features else df[load_col]
        atl = by_player_transform(
            source, lambda s: s.fillna(0.0).rolling(7, min_periods=1).mean()
        )
        ctl = by_player_transform(
            source, lambda s: s.fillna(0.0).rolling(28, min_periods=3).mean()
        )
        std7 = by_player_transform(
            source, lambda s: s.fillna(0.0).rolling(7, min_periods=2).std().fillna(0.0)
        )
        features[f"{prefix}_atl7"] = atl
        features[f"{prefix}_ctl28"] = ctl
        features[f"{prefix}_acwr"] = (atl / (ctl + 1e-3)).clip(0.0, 4.0)
        features[f"{prefix}_monotony7"] = (atl / (std7 + 0.1)).clip(0.0, 10.0)
        features[f"{prefix}_strain7"] = (
            features[f"{prefix}_atl7"] * features[f"{prefix}_monotony7"]
        )

    # 过去 7 天训练天数密度 (拥挤度)
    features["active_train_days_7d"] = by_player_transform(
        features["unified_daily_load"],
        lambda s: (s > 0.05).astype(int).rolling(7, min_periods=1).sum(),
    )

    # ================= 4. 严格的既往伤病史特征 (仅用 t-1 及以前，绝不泄露当日标签) =================
    injury_source = df["inj_non_contact_onset"]
    features["prev_inj_7d"] = by_player_transform(
        injury_source, lambda s: s.shift(1).fillna(0).rolling(7, min_periods=1).sum()
    )
    features["prev_inj_14d"] = by_player_transform(
        injury_source, lambda s: s.shift(1).fillna(0).rolling(14, min_periods=1).sum()
    )
    features["prev_inj_30d"] = by_player_transform(
        injury_source, lambda s: s.shift(1).fillna(0).rolling(30, min_periods=1).sum()
    )

    # 计算距离上一次受伤的天数 (无伤或超过60天记为60)
    def calc_days_since_inj(s: pd.Series) -> pd.Series:
        out = []
        last_idx = None
        for i, val in enumerate(s.shift(1).fillna(0).values):
            if val >= 1:
                last_idx = i
            if last_idx is None:
                out.append(60.0)
            else:
                out.append(float(min(i - last_idx, 60)))
        return pd.Series(out, index=s.index)

    features["days_since_last_inj"] = by_player_transform(injury_source, calc_days_since_inj)
    features["recent_inj_decay"] = np.exp(-features["days_since_last_inj"] / 14.0)  # 指数衰减复发风险因子 (0~1)

    # 所有新特征一次性拼接，避免逐列插入导致 DataFrame 高度碎片化。
    feature_frame = pd.DataFrame(features, index=df.index)
    df = pd.concat([df, feature_frame], axis=1).copy()

    print(f"✅ [{dataset_name}] 特征工程完成！当前总列数: {len(df.columns)}")
    return df


def check_cross_domain_consistency(df_sm: pd.DataFrame, df_loc: pd.DataFrame):
    print("\n" + "=" * 80)
    print("🔬 跨数据集特征与损伤相关性一致性检验 (检验 SoccerMon 与 本地24人 是否同向！)")
    print("=" * 80)

    # 在 SoccerMon 中，只取至少有过 1 次伤病记录的活跃监测球员子集 (排除从未使用伤病上报模块的 35 人)
    sm_active = df_sm[df_sm["is_active_day"] == 1].copy()
    inj_players_sm = sm_active.groupby("player_id")["inj_prevalence"].sum()
    valid_inj_Reporting_players = set(inj_players_sm[inj_players_sm > 0].index)
    sm_cohort = sm_active[sm_active["player_id"].isin(valid_inj_Reporting_players)].copy()

    loc_cohort = df_loc[df_loc["is_active_day"] == 1].copy()

    print(f"SoccerMon 有效上报伤病队列人日: {len(sm_cohort)} (15人) | 本地活跃监测人日: {len(loc_cohort)} (24人)\n")

    test_features = [
        "soreness_01", "fatigue_01", "sleep_poor_01", "stress_01", "mood_poor_01",
        "wellness_poor_mean", "wellness_poor_mean_z", "soreness_01_z", "fatigue_01_z",
        "soreness_01_roll7d", "uload_acwr", "dist_acwr", "hsr_acwr", "decel_acwr",
        "decel_accel_ratio", "imu_dyn_p95_g_z", "hr_speed_ratio_z",
        "prev_inj_14d", "prev_inj_30d", "recent_inj_decay"
    ]

    target_col = "target_non_contact_next_7d"
    print(f"对比目标标签: {target_col} (未来 7 天非接触损伤风险)")
    print(f"{'特征名称':24s} | {'SoccerMon 相关性(r)':>18s} | {'本地24人 相关性(r)':>18s} | {'两域方向是否一致'}")
    print("-" * 80)
    consistent_count = 0
    for feat in test_features:
        r_sm = sm_cohort[feat].corr(sm_cohort[target_col])
        r_loc = loc_cohort[feat].corr(loc_cohort[target_col])
        same_sign = (r_sm * r_loc) > 0
        if same_sign:
            consistent_count += 1
        flag = "✅ 同向" if same_sign else "⚠️ 反向"
        print(f"{feat:24s} | {r_sm:+18.4f} | {r_loc:+18.4f} | {flag}")

    print("-" * 80)
    print(f"🎯 关键特征两域同向率: {consistent_count} / {len(test_features)}")


if __name__ == "__main__":
    df_sm_raw = pd.read_csv(PROCESSED_DIR / "soccermon_daily_panel.csv", parse_dates=["date"])
    df_loc_raw = pd.read_csv(PROCESSED_DIR / "local_daily_panel.csv", parse_dates=["date"])

    df_sm_feat = engineer_features(df_sm_raw, "SoccerMon 训练集")
    df_loc_feat = engineer_features(df_loc_raw, "本地 24人验证集")

    df_sm_feat.to_csv(PROCESSED_DIR / "soccermon_features.csv", index=False, encoding="utf-8-sig")
    df_loc_feat.to_csv(PROCESSED_DIR / "local_features.csv", index=False, encoding="utf-8-sig")

    check_cross_domain_consistency(df_sm_feat, df_loc_feat)
    print("\n🎉 步骤 03 完成！请把控制台输出发给我。")
