# src/step04_train_and_transfer.py
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def fix_gps_missing_and_rehab(df: pd.DataFrame, is_soccermon: bool = False) -> pd.DataFrame:
    """
    修复关键问题：
    1) 当某天未佩戴 GPS (total_distance_m 为 NaN) 但有 sRPE 负荷时，不能把 GPS 负荷填为 0.0 去算 ACWR，
       而是用个体标准化后的 unified_daily_load 估算当日外部负荷，消除 2020年1-5月 无GPS期的假零值。
    2) 构建多维度复合风险指数，提升小样本信噪比。
    """
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    player_key = df["player_id"]

    def by_p(s: pd.Series, fn):
        return s.groupby(player_key, sort=False, group_keys=False).transform(fn)

    # 用个体历史中位数对齐 GPS 距离与 unified_daily_load，填补未佩戴 GPS 日的负荷空缺
    for col, prefix in [
        ("total_distance_m", "dist_adj"),
        ("hsr_distance_m", "hsr_adj"),
        ("high_decel_sec", "decel_adj"),
    ]:
        p_med = by_p(df[col], lambda s: s.shift(1).expanding(min_periods=1).median()).fillna(df[col].median())
        proxy = df["unified_daily_load"] * p_med
        filled_col = df[col].where(df[col].notna(), proxy)
        atl7 = by_p(filled_col, lambda s: s.fillna(0.0).rolling(7, min_periods=1).mean())
        ctl28 = by_p(filled_col, lambda s: s.fillna(0.0).rolling(28, min_periods=3).mean())
        df[f"{prefix}_acwr"] = (atl7 / (ctl28 + 1e-3)).clip(0.0, 4.0)

    # 构造三大生理机制复合特征 (均满足：数值越大 -> 损伤风险越高)
    # 1) 综合急慢性外负荷比 (External & Unified ACWR Composite)
    df["mech_overload_score"] = df[["uload_acwr", "dist_adj_acwr", "hsr_adj_acwr", "decel_adj_acwr"]].mean(axis=1)
    # 2) 综合主观疲劳与酸痛恶化指数 (Subjective Wellness & Z-score Composite)
    df["subj_strain_score"] = (
        0.35 * df["wellness_poor_mean"].fillna(0.3)
        + 0.35 * (df["wellness_poor_mean_z"].fillna(0.0).clip(-3, 3) / 3.0)
        + 0.30 * df["soreness_01_roll7d"].fillna(0.3)
    )
    # 3) 负荷×疲劳耦合超载项 (高负荷同时伴随高主观酸痛/疲劳)
    df["overload_x_fatigue"] = df["mech_overload_score"] * (1.0 + df["wellness_poor_mean"].fillna(0.0))

    return df


# 精选跨数据集物理含义 100% 同构、且生理方向全部为正（值越大风险越高）的核心特征集
CORE_FEATURES = [
    # A. 急慢性负荷比与训练密度 (Overload Dynamics)
    "uload_acwr",
    "dist_adj_acwr",
    "hsr_adj_acwr",
    "decel_adj_acwr",
    "uload_monotony7",
    "active_train_days_7d",
    # B. 主观状态与个体基线偏离 (Subjective Wellness & Individual Z-Scores)
    "soreness_01",
    "fatigue_01",
    "sleep_poor_01",
    "stress_01",
    "mood_poor_01",
    "wellness_poor_mean",
    "wellness_poor_mean_z",
    "soreness_01_z",
    "fatigue_01_z",
    "soreness_01_roll7d",
    "fatigue_01_roll7d",
    # C. 客观生物力学与离心制动 (Biomechanics)
    "decel_accel_ratio",
    "hsr_ratio",
    "imu_dyn_p95_g_z",
    # D. 生理复合机制与既往伤病复发风险 (Composite & Prior History)
    "mech_overload_score",
    "subj_strain_score",
    "overload_x_fatigue",
    "prev_inj_14d",
    "prev_inj_30d",
    "recent_inj_decay",
]


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, use_default_05: bool = False) -> dict:
    """计算全面评估指标：ROC-AUC, PR-AUC, 以及校准阈值下的 F1 / Recall / Precision"""
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    if use_default_05:
        thresh = 0.50
    else:
        # 针对跨数据集先验发病率偏移 (Prior Shift)，采用分位数风险预警阈值 (Top 20% 高危预警)
        thresh = float(np.quantile(y_prob, 0.80))

    y_pred = (y_prob >= thresh).astype(int)
    return {
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Recall(召回率)": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Precision(精确率)": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Brier": round(brier, 4),
        "Threshold": round(thresh, 4),
    }


def run_experiments(df_sm: pd.DataFrame, df_loc: pd.DataFrame, target_col: str, horizon_label: str):
    print("\n" + "=" * 88)
    print(f"🏆 跨数据集损伤风险建模与本地验证结果 —— 目标: [{horizon_label}] ({target_col})")
    print("=" * 88)

    # 1. 准备本地 24 人测试集 (全部活跃监测日)
    loc_active = df_loc[df_loc["is_active_day"] == 1].copy().reset_index(drop=True)
    X_loc = loc_active[CORE_FEATURES].copy()
    y_loc = loc_active[target_col].values
    players_loc = loc_active["player_id"].values
    base_rate_loc = y_loc.mean()
    print(f"📌 本地 24 人测试集样本数: {len(loc_active)} 人日 | 正样本数: {y_loc.sum()} ({base_rate_loc*100:.2f}%)")

    # 2. 准备 SoccerMon 源域训练集
    sm_active = df_sm[df_sm["is_active_day"] == 1].copy()

    # --- [Model 1] Baseline: 传统直接迁移 (全量50人未清洗 + 无单调约束 + 默认0.5阈值) ---
    X_sm_raw = sm_active[CORE_FEATURES].fillna(0.0)
    y_sm_raw = sm_active[target_col].values
    clf_base = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    clf_base.fit(X_sm_raw, y_sm_raw)
    prob_base = clf_base.predict_proba(X_loc.fillna(0.0))[:, 1]
    res_m1 = evaluate_predictions(y_loc, prob_base, use_default_05=True)
    res_m1_cal = evaluate_predictions(y_loc, prob_base, use_default_05=False)

    # --- 清洗 SoccerMon 源域训练队列 (仅保留有效上报伤病队列 + 剔除伤病康复期存续日污染) ---
    inj_counts = sm_active.groupby("player_id")["inj_prevalence"].sum()
    valid_sm_players = set(inj_counts[inj_counts > 0].index)
    sm_clean = sm_active[sm_active["player_id"].isin(valid_sm_players)].copy()
    # 剔除“已经处于伤病存续期但非新发首日”的康复休息日，防止低负荷休息日污染负样本
    sm_clean = sm_clean[~((sm_clean["inj_prevalence"] == 1) & (sm_clean["inj_non_contact_onset"] == 0))].copy()

    X_sm_clean = sm_clean[CORE_FEATURES].copy()
    y_sm_clean = sm_clean[target_col].values

    # --- [Model 2] Zero-Shot 物理对齐 + 运动生理单调性约束 GBDT (100% 仅用 SoccerMon 训练) ---
    # monotonic_cst 全设为 +1：强制模型遵循“过载/疲劳/酸痛/近期伤病越高，风险单调不减”的生理学定律
    mono_constraints = [1] * len(CORE_FEATURES)
    clf_mono_gbdt = HistGradientBoostingClassifier(
        max_iter=150,
        max_depth=3,
        learning_rate=0.03,
        min_samples_leaf=30,
        l2_regularization=5.0,
        monotonic_cst=mono_constraints,
        class_weight="balanced",
        random_state=42,
    )
    clf_mono_gbdt.fit(X_sm_clean, y_sm_clean)
    prob_m2 = clf_mono_gbdt.predict_proba(X_loc)[:, 1]
    res_m2 = evaluate_predictions(y_loc, prob_m2, use_default_05=False)

    # --- [Model 3] Zero-Shot 跨人群稳健集成模型 (生理单调 GBDT + 正则化逻辑回归 + 运动科学先验打分) ---
    # 构造强正则化非负权重特征标准化管线 (对源域和目标域分别做分布标准化 CorAl-lite，消除残余均值漂移)
    imp = SimpleImputer(strategy="median")
    X_sm_imp = imp.fit_transform(X_sm_clean)
    X_loc_imp = imp.transform(X_loc)

    X_sm_z = StandardScaler().fit_transform(X_sm_imp)
    X_loc_z = StandardScaler().fit_transform(X_loc_imp)

    clf_lr = LogisticRegression(C=0.05, class_weight="balanced", random_state=42)
    clf_lr.fit(X_sm_z, y_sm_clean)
    # 将源域中学成负数的噪声特征权重截断为 >=0 (生理非负先验)
    clf_lr.coef_ = np.maximum(clf_lr.coef_, 0.0)
    prob_lr = clf_lr.predict_proba(X_loc_z)[:, 1]

    # 结合源域训练的单调 GBDT 与生理正则 LR (100% Zero-Shot，未使用任何本地标签)
    prob_m3 = 0.55 * prob_m2 + 0.45 * prob_lr
    res_m3 = evaluate_predictions(y_loc, prob_m3, use_default_05=False)

    # --- [Model 4] Few-Shot 预训练 + 本地小样本域适应微调 (留一球员交叉验证 LOPO-CV) ---
    # 每次留出本地 1 名球员作为盲测集，利用其余 23 人 + SoccerMon 预训练概率做域适应校准
    oof_probs_m4 = np.zeros(len(loc_active))
    unique_loc_players = np.unique(players_loc)
    for test_p in unique_loc_players:
        train_mask = players_loc != test_p
        test_mask = players_loc == test_p

        # 把 SoccerMon 零样本模型输出的先验风险分 prob_m3 作为预训练底座特征，在本地训练集上微调
        X_tr_da = np.column_stack([X_loc_z[train_mask], prob_m3[train_mask] * 3.0])
        y_tr_da = y_loc[train_mask]
        X_te_da = np.column_stack([X_loc_z[test_mask], prob_m3[test_mask] * 3.0])

        da_gbdt = HistGradientBoostingClassifier(
            max_iter=80,
            max_depth=3,
            learning_rate=0.04,
            min_samples_leaf=20,
            l2_regularization=3.0,
            monotonic_cst=[1] * X_tr_da.shape[1],
            class_weight="balanced",
            random_state=42,
        )
        da_gbdt.fit(X_tr_da, y_tr_da)
        p_da_gbdt = da_gbdt.predict_proba(X_te_da)[:, 1]

        # 与 SoccerMon 零样本底座融合 (防止小样本过拟合)
        oof_probs_m4[test_mask] = 0.40 * prob_m3[test_mask] + 0.60 * p_da_gbdt

    res_m4 = evaluate_predictions(y_loc, oof_probs_m4, use_default_05=False)

    # 汇总打印对比表
    summary_df = pd.DataFrame([
        {"模型方案": "1a. Baseline 直接迁移 (未清洗+默认0.5阈值)", **res_m1},
        {"模型方案": "1b. Baseline 直接迁移 (仅做分位数阈值校准)", **res_m1_cal},
        {"模型方案": "2. Zero-Shot 物理对齐 + 生理单调GBDT (纯SoccerMon训练)", **res_m2},
        {"模型方案": "3. Zero-Shot 域对齐 + 生理单调集成 (纯SoccerMon训练)", **res_m3},
        {"模型方案": "4. Few-Shot 预训练 + 本地微调 (留一球员验证 LOPO-CV)", **res_m4},
    ])
    print("\n📊 消融实验与迁移效果对比表:")
    print(summary_df.to_string(index=False))

    # 分析在最佳模型 (Model 4) 下，预测为高风险 Top 20% 组 vs 低风险 80% 组的实际受伤率倍数 (Risk Ratio)
    high_risk_mask = oof_probs_m4 >= np.quantile(oof_probs_m4, 0.80)
    rate_high = y_loc[high_risk_mask].mean()
    rate_low = y_loc[~high_risk_mask].mean()
    rr = rate_high / (rate_low + 1e-6)
    print(f"\n💡 临床/运动队实用价值验证 (Model 4):")
    print(f"   - 模型预警【高风险组 (Top 20% 人日)】实际损伤发生率: {rate_high*100:.2f}%")
    print(f"   - 模型判定【低风险组 (其余 80% 人日)】实际损伤发生率: {rate_low*100:.2f}%")
    print(f"   - 高危组相对低危组的真实发病风险倍数 (Relative Risk, RR): {rr:.2f} 倍！")

    return summary_df, loc_active, prob_m3, oof_probs_m4


if __name__ == "__main__":
    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv", parse_dates=["date"])
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv", parse_dates=["date"])

    df_sm = fix_gps_missing_and_rehab(df_sm, is_soccermon=True)
    df_loc = fix_gps_missing_and_rehab(df_loc, is_soccermon=False)

    # 分别验证【未来 7 天非接触损伤预警】与【未来 3 天非接触损伤预警】
    sum_7d, loc_df, p_zs_7d, p_fs_7d = run_experiments(
        df_sm, df_loc, target_col="target_non_contact_next_7d", horizon_label="未来 7 天非接触性损伤预警"
    )
    sum_3d, _, p_zs_3d, p_fs_3d = run_experiments(
        df_sm, df_loc, target_col="target_non_contact_next_3d", horizon_label="未来 3 天非接触性损伤预警"
    )

    # 保存本地预测结果与评估汇总表
    loc_df["pred_prob_zeroshot_7d"] = np.round(p_zs_7d, 4)
    loc_df["pred_prob_fewshot_7d"] = np.round(p_fs_7d, 4)
    loc_df["pred_prob_zeroshot_3d"] = np.round(p_zs_3d, 4)
    loc_df["pred_prob_fewshot_3d"] = np.round(p_fs_3d, 4)

    loc_df.to_csv(OUTPUTS_DIR / "local_24_predictions.csv", index=False, encoding="utf-8-sig")
    all_summary = pd.concat([
        sum_7d.assign(预测窗口="未来7天非接触损伤"),
        sum_3d.assign(预测窗口="未来3天非接触损伤")
    ], ignore_index=True)
    all_summary.to_csv(OUTPUTS_DIR / "ablation_model_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n🎉 步骤 04 完成！评估表已保存至 {OUTPUTS_DIR / 'ablation_model_comparison.csv'}。请把控制台结果发给我！")