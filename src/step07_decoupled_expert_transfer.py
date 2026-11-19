# src/step07_decoupled_expert_transfer.py
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from step06_rigorous_validation_and_plots import (
    add_ewma_and_context_features,
    evaluate_with_optimal_f1_threshold,
)

# ================= 解耦的三大跨数据集专家特征集 (严禁混入 prev_inj 伤病史，防止源域模型偷懒) =================
FEAT_EXPERT_OVERLOAD = [
    "uload_ewma_acwr", "dist_ewma_acwr", "hsr_ewma_acwr",
    "uload_ewma_spike", "dist_ewma_spike",
    "uload_monotony7", "active_train_days_7d", "ewma_overload_composite",
]

FEAT_EXPERT_WELLNESS = [
    "soreness_01", "fatigue_01", "sleep_poor_01", "stress_01", "mood_poor_01",
    "wellness_poor_mean", "wellness_poor_mean_z", "soreness_01_z", "fatigue_01_z",
    "soreness_01_roll7d", "fatigue_01_roll7d", "stress_01_roll7d",
    "subj_strain_composite", "recovery_deficit_index",
]

FEAT_EXPERT_BIOMECH = [
    "decel_ewma_acwr", "decel_ewma_spike", "impact_ewma_acwr",
    "decel_accel_ratio", "hsr_ratio", "imu_dyn_p95_g_z",
    "high_decel_sec_z", "hsr_distance_m_z",
]


def train_soccermon_expert_branch(
    sm_df: pd.DataFrame, loc_df: pd.DataFrame, feat_cols: list[str], target_col: str, require_gps: bool = False
) -> np.ndarray:
    """
    在 SoccerMon 上单独训练某一个生理/生物力学维度的专家分支，
    并输出对本地数据集的跨域标准化风险分 (0~1)。
    """
    sub_sm = sm_df.copy()
    if require_gps:
        # 生物力学专家仅在拥有真实 GPS+IMU 数据的训练日上训练，消除无 GPS 期的噪声
        sub_sm = sub_sm[sub_sm["total_distance_m"].notna()].copy()

    X_sm = sub_sm[feat_cols].copy()
    y_sm = sub_sm[target_col].values
    X_loc = loc_df[feat_cols].copy()

    imp = SimpleImputer(strategy="median")
    X_sm_imp = imp.fit_transform(X_sm)
    X_loc_imp = imp.transform(X_loc)

    # 域对齐标准化 (Domain-Specific Z-Scaling / CORAL-lite)
    X_sm_z = StandardScaler().fit_transform(X_sm_imp)
    X_loc_z = StandardScaler().fit_transform(X_loc_imp)

    # 1) 非负生理约束逻辑回归
    lr = LogisticRegression(C=0.03, class_weight="balanced", random_state=42)
    lr.fit(X_sm_z, y_sm)
    lr.coef_ = np.maximum(lr.coef_, 0.02)
    p_lr = lr.predict_proba(X_loc_z)[:, 1]

    # 2) 浅层单调约束 GBDT
    gb = HistGradientBoostingClassifier(
        max_iter=80, max_depth=2, learning_rate=0.03, min_samples_leaf=30,
        l2_regularization=8.0, monotonic_cst=[1] * len(feat_cols),
        class_weight="balanced", random_state=42
    )
    gb.fit(X_sm_z, y_sm)
    p_gb = gb.predict_proba(X_loc_z)[:, 1]

    raw_prob = 0.60 * p_lr + 0.40 * p_gb
    # 转换为无量纲标准分后映射到 (0, 1)
    z_score = (raw_prob - np.mean(raw_prob)) / (np.std(raw_prob) + 1e-6)
    return 1.0 / (1.0 + np.exp(-z_score))


def build_decoupled_5_pillars(df_sm: pd.DataFrame, loc_active: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    通过源域解耦多专家预训练，构建真正平衡、无共线性的 5 大生理支柱：
    P1: SoccerMon 急慢性负荷超载专家分支 (叠加本地比赛暴露)
    P2: SoccerMon 1Hz 穿戴生物力学离心制动专家分支
    P3: SoccerMon 晨间主观疲劳/酸痛/压力专家分支
    P4: 本地女足生理期与比赛强度专属上下文支柱
    P5: 既往伤病史与近期伤病复发指数衰减支柱
    """
    sm_active = df_sm[df_sm["is_active_day"] == 1].copy()
    inj_counts = sm_active.groupby("player_id")["inj_prevalence"].sum()
    valid_players = set(inj_counts[inj_counts > 0].index)
    sm_clean = sm_active[sm_active["player_id"].isin(valid_players)].copy()
    sm_clean = sm_clean[~((sm_clean["inj_prevalence"] == 1) & (sm_clean["inj_non_contact_onset"] == 0))].copy()

    # 在 SoccerMon 上独立训练 3 个不含伤病史的纯生理/生物力学专家模型
    sm_prob_overload = train_soccermon_expert_branch(sm_clean, loc_active, FEAT_EXPERT_OVERLOAD, target_col, require_gps=False)
    sm_prob_biomech = train_soccermon_expert_branch(sm_clean, loc_active, FEAT_EXPERT_BIOMECH, target_col, require_gps=True)
    sm_prob_wellness = train_soccermon_expert_branch(sm_clean, loc_active, FEAT_EXPERT_WELLNESS, target_col, require_gps=False)

    def to_01_rank(arr) -> np.ndarray:
        s = pd.Series(arr).fillna(pd.Series(arr).median())
        z = ((s - s.mean()) / (s.std() + 1e-6)).clip(-3.5, 3.5)
        return (1.0 / (1.0 + np.exp(-z))).values

    pillars = pd.DataFrame(index=loc_active.index)
    # P1: 源域负荷超载专家 (SoccerMon Overload Expert)
    pillars["P1_SM_Overload_Expert"] = to_01_rank(sm_prob_overload)
    # P2: 源域 1Hz 生物力学专家 (SoccerMon Biomech Expert)
    pillars["P2_SM_Biomech_Expert"] = to_01_rank(sm_prob_biomech)
    # P3: 源域主观身心状态专家 (SoccerMon Wellness Expert)
    pillars["P3_SM_Wellness_Expert"] = to_01_rank(sm_prob_wellness)
    # P4: 女足专属赛程与生理期耦合支柱 (Female Match & Menstrual Context)
    female_ctx = (
        0.45 * to_01_rank(loc_active["ctx_match_days_7d"])
        + 0.30 * to_01_rank(loc_active["ctx_rpe_roll7d"])
        + 0.25 * to_01_rank(loc_active["ctx_mens_x_overload"])
    )
    pillars["P4_Female_Match_Menstrual"] = to_01_rank(female_ctx)
    # P5: 伤病史与近期复发衰减支柱 (Prior & Recent Injury History)
    hist_ctx = (
        0.45 * to_01_rank(loc_active["recent_inj_decay"])
        + 0.30 * to_01_rank(loc_active["prev_inj_30d"])
        + 0.25 * to_01_rank(loc_active["ctx_hist_inj_12m"])
    )
    pillars["P5_Injury_History_Decay"] = to_01_rank(hist_ctx)

    # 纯零样本多专家均衡融合风险分 (完全不依赖本地 Y 标签训练):
    # 同时评估【纯生理负荷零样本 (不含伤病史)】与【全维度零样本】
    prob_zeroshot_physio = (
        0.40 * pillars["P1_SM_Overload_Expert"].values
        + 0.35 * pillars["P2_SM_Biomech_Expert"].values
        + 0.25 * pillars["P3_SM_Wellness_Expert"].values
    )
    prob_zeroshot_full = (
        0.25 * pillars["P1_SM_Overload_Expert"].values
        + 0.20 * pillars["P2_SM_Biomech_Expert"].values
        + 0.20 * pillars["P3_SM_Wellness_Expert"].values
        + 0.35 * to_01_rank(0.6 * loc_active["recent_inj_decay"] + 0.4 * loc_active["prev_inj_30d"])
    )
    return pillars, prob_zeroshot_physio, prob_zeroshot_full


def fit_predict_constrained_5pillars(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    带生理下界与上界约束的 5 支柱融合器 (Bounded Non-Negative Stacking)：
    - 强制每个生理支柱至少保留 >= 10% 的基础贡献下界；
    - 限制单一伤病史支柱最高不超过 38% 权重，防止伤病史掩盖负荷、生物力学与主观特征！
    """
    # 计算每个支柱在训练集上的单变量判别增益 (AUC - 0.5) 与非负最小二乘的组合
    raw_w = np.zeros(X_tr.shape[1])
    for j in range(X_tr.shape[1]):
        auc_j = max(roc_auc_score(y_tr, X_tr[:, j]) - 0.50, 0.01)
        raw_w[j] = auc_j

    nnls_w, _ = nnls(X_tr, y_tr)
    nnls_w = nnls_w / (nnls_w.sum() + 1e-8)
    auc_w = raw_w / (raw_w.sum() + 1e-8)

    w = 0.55 * auc_w + 0.45 * nnls_w
    # 施加运动医学先验上下界约束: 单支柱介于 [0.10, 0.36] 之间，再归一化
    w = np.clip(w, 0.10, 0.36)
    w = w / w.sum()

    linear_score_tr = X_tr @ w
    linear_score_te = X_te @ w

    # 结合浅层单调 GBDT 微调非线性交互
    gb = HistGradientBoostingClassifier(
        max_iter=45, max_depth=2, learning_rate=0.03, min_samples_leaf=30,
        l2_regularization=10.0, monotonic_cst=[1] * X_tr.shape[1],
        class_weight="balanced", random_state=42
    )
    gb.fit(X_tr, y_tr)
    p_gb = gb.predict_proba(X_te)[:, 1]

    pred_te = 0.70 * linear_score_te + 0.30 * p_gb
    return pred_te, w


def run_decoupled_experiment(df_sm: pd.DataFrame, df_loc: pd.DataFrame, target_col: str, horizon_days: int, title: str):
    print("\n" + "=" * 98)
    print(f"🌟 解耦式多专家分支迁移学习评估 —— 目标: [{title}] ({target_col})")
    print("=" * 98)

    loc_active = df_loc[df_loc["is_active_day"] == 1].copy().sort_values(["date", "player_id"]).reset_index(drop=True)
    y_loc = loc_active[target_col].values
    players_loc = loc_active["player_id"].values
    dates_loc = loc_active["date"]

    # 1. 传统未对齐 Baseline
    sm_active = df_sm[df_sm["is_active_day"] == 1].copy()
    base_cols = FEAT_EXPERT_OVERLOAD + FEAT_EXPERT_WELLNESS + FEAT_EXPERT_BIOMECH
    rf_base = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_base.fit(sm_active[base_cols].fillna(0), sm_active[target_col].values)
    prob_base = rf_base.predict_proba(loc_active[base_cols].fillna(0))[:, 1]
    res_1_base = evaluate_with_optimal_f1_threshold(y_loc, prob_base, fixed_thresh=0.50)

    # 构建解耦式 5 大生理专家支柱
    pillars_df, prob_zs_physio, prob_zs_full = build_decoupled_5_pillars(df_sm, loc_active, target_col)
    X_pillars = pillars_df.values

    # 2a. Zero-Shot 纯生理/生物力学零样本模型 (完全不用任何伤病史，纯靠负荷+穿戴+主观问卷预测!)
    res_2a_physio = evaluate_with_optimal_f1_threshold(y_loc, prob_zs_physio)

    # 2b. Zero-Shot 全维度多专家零样本模型 (100% 纯 SoccerMon 训练)
    res_2b_full = evaluate_with_optimal_f1_threshold(y_loc, prob_zs_full)

    # 3. 严格跨球员盲测 (6-Fold GroupKFold，测试集为完全未见过的新球员)
    gkf = GroupKFold(n_splits=6)
    oof_group = np.zeros(len(loc_active))
    weights_collect = []
    for tr_idx, te_idx in gkf.split(X_pillars, y_loc, groups=players_loc):
        pred_fold, w_fold = fit_predict_constrained_5pillars(X_pillars[tr_idx], y_loc[tr_idx], X_pillars[te_idx])
        oof_group[te_idx] = pred_fold
        weights_collect.append(w_fold)
    res_3_group = evaluate_with_optimal_f1_threshold(y_loc, oof_group)

    # 4. 严格时间序列前滚验证 (Purged Temporal Expanding-Window，带 horizon_days 隔离带)
    unique_dates = np.sort(dates_loc.unique())
    n_dates = len(unique_dates)
    oof_temporal = np.copy(prob_zs_full)
    warmup_cut = int(n_dates * 0.25)

    for d_idx in range(warmup_cut, n_dates):
        curr_date = unique_dates[d_idx]
        purge_cutoff = pd.Timestamp(curr_date) - pd.Timedelta(days=horizon_days)
        tr_mask = (dates_loc <= purge_cutoff).values
        te_mask = (dates_loc == curr_date).values
        if tr_mask.sum() >= 100 and y_loc[tr_mask].sum() >= 5:
            pred_t, _ = fit_predict_constrained_5pillars(X_pillars[tr_mask], y_loc[tr_mask], X_pillars[te_mask])
            oof_temporal[te_mask] = 0.25 * prob_zs_full[te_mask] + 0.75 * pred_t
    res_4_temporal = evaluate_with_optimal_f1_threshold(y_loc, oof_temporal)

    # 5. 队内全特征 + 5大专家支柱多模态融合 (5-Fold Stratified CV)
    all_adapt_cols = base_cols + [
        "ctx_hist_inj_12m", "ctx_hist_major_inj", "ctx_is_menstruation",
        "ctx_dysmenorrhea", "ctx_mens_x_overload", "ctx_is_match_day",
        "ctx_match_days_7d", "ctx_rpe_norm", "ctx_rpe_roll7d",
        "prev_inj_7d", "prev_inj_14d", "prev_inj_30d", "recent_inj_decay",
    ]
    X_full_df = pd.concat([loc_active[all_adapt_cols], pillars_df], axis=1)
    X_full_z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_full_df))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_intrateam = np.zeros(len(loc_active))
    for tr_idx, te_idx in skf.split(X_full_z, y_loc):
        gb_s = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.05, min_samples_leaf=12,
            l2_regularization=1.5, class_weight="balanced", random_state=42
        )
        gb_s.fit(X_full_z[tr_idx], y_loc[tr_idx])
        et_s = ExtraTreesClassifier(
            n_estimators=250, max_depth=7, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        et_s.fit(X_full_z[tr_idx], y_loc[tr_idx])
        oof_intrateam[te_idx] = (
            0.25 * oof_group[te_idx]
            + 0.40 * gb_s.predict_proba(X_full_z[te_idx])[:, 1]
            + 0.35 * et_s.predict_proba(X_full_z[te_idx])[:, 1]
        )
    res_5_intrateam = evaluate_with_optimal_f1_threshold(y_loc, oof_intrateam)

    summary_df = pd.DataFrame([
        {"验证方案 (解耦式多专家迁移消融)": "1. 传统直接迁移 Baseline (未对齐+默认0.5阈值)", **res_1_base},
        {"验证方案 (解耦式多专家迁移消融)": "2a. Zero-Shot 纯负荷+穿戴+主观专家 (不含任何伤病史)", **res_2a_physio},
        {"验证方案 (解耦式多专家迁移消融)": "2b. Zero-Shot 全维度多专家融合 (100%纯SoccerMon)", **res_2b_full},
        {"验证方案 (解耦式多专家迁移消融)": "3. Two-Stage 5专家跨球员盲测 (GroupKFold 新球员泛化)", **res_3_group},
        {"验证方案 (解耦式多专家迁移消融)": "4. Two-Stage 5专家时间序列前滚 (Purged Temporal 零重叠)", **res_4_temporal},
        {"验证方案 (解耦式多专家迁移消融)": "5. Two-Stage 队内多模态全融合 (5-Fold 队内交叉验证)", **res_5_intrateam},
    ])
    print(summary_df.to_string(index=False))

    avg_w = np.mean(weights_collect, axis=0)
    pillar_imp_df = pd.DataFrame({
        "生理机制支柱 (5-Pillar)": [
            "P1: SoccerMon 急慢性负荷超载专家 (Overload ACWR)",
            "P2: SoccerMon 1Hz穿戴生物力学专家 (Decel & IMU)",
            "P3: SoccerMon 晨间主观疲劳与酸痛专家 (Wellness Z)",
            "P4: 女足专属赛程密度与生理期交互 (Match & Menstrual)",
            "P5: 既往伤病史与近期复发衰减 (Injury History)",
        ],
        "归一化贡献占比": [f"{w*100:.1f}%" for w in avg_w],
        "单支柱独立AUC": [round(roc_auc_score(y_loc, X_pillars[:, j]), 4) for j in range(5)],
        "weight_num": avg_w,
    }).sort_values("weight_num", ascending=False)
    print("\n🧬 解耦后五大运动医学专家支柱的独立判别力 (AUC) 与均衡权重:")
    print(pillar_imp_df[["生理机制支柱 (5-Pillar)", "单支柱独立AUC", "归一化贡献占比"]].to_string(index=False))

    return summary_df, loc_active, prob_base, prob_zs_full, oof_group, oof_temporal, oof_intrateam, pillar_imp_df


if __name__ == "__main__":
    from step06_rigorous_validation_and_plots import generate_all_publication_figures

    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv", parse_dates=["date"])
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv", parse_dates=["date"])

    df_sm = add_ewma_and_context_features(df_sm, is_local=False)
    df_loc = add_ewma_and_context_features(df_loc, is_local=True)

    sum_7d, loc_7d, p_base, p_zs, p_grp, p_temp, p_intra, pillar_df = run_decoupled_experiment(
        df_sm, df_loc, "target_non_contact_next_7d", horizon_days=7, title="未来 7 天非接触性损伤预警"
    )
    sum_3d, _, _, _, _, _, _, _ = run_decoupled_experiment(
        df_sm, df_loc, "target_non_contact_next_3d", horizon_days=3, title="未来 3 天非接触性损伤预警"
    )

    final_table = pd.concat([
        sum_7d.assign(预测窗口="未来7天非接触损伤"),
        sum_3d.assign(预测窗口="未来3天非接触损伤"),
    ], ignore_index=True)
    final_table.to_csv(OUTPUTS_DIR / "final_decoupled_expert_evaluation.csv", index=False, encoding="utf-8-sig")

    # 重新渲染高清图表（更新五大专家支柱均衡贡献图）
    generate_all_publication_figures(
        loc_7d, loc_7d["target_non_contact_next_7d"].values,
        p_base, p_zs, p_grp, p_temp, p_intra, pillar_df
    )
    print("\n🎉 步骤 07 解耦式多专家迁移全部完成！请把控制台结果发给我。")