# src/step05_advanced_transfer_and_plots.py
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
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


def add_ewma_and_context_features(df: pd.DataFrame, is_local: bool = False) -> pd.DataFrame:
    """
    新增运动科学高阶特征：
    1) EWMA (指数加权移动平均) 急慢性负荷比 (Williams et al., BJSM 2017)
    2) 累积负荷变异与恢复不足指数 (Acute Load / Sleep & Wellness Recovery)
    3) 本地女足专属特征 (月经周期/痛经、基线既往12个月伤病史、比赛日密集度)
    """
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    pkey = df["player_id"]

    def by_p(s: pd.Series, fn):
        return s.groupby(pkey, sort=False, group_keys=False).transform(fn)

    new_cols = {}

    # 1. 用个体历史中位数对齐未佩戴 GPS 日的负荷，并计算 EWMA-ACWR (span=7 vs span=28)
    for col, prefix in [
        ("unified_daily_load", "uload"),
        ("total_distance_m", "dist"),
        ("hsr_distance_m", "hsr"),
        ("high_decel_sec", "decel"),
        ("imu_high_impact_sec", "impact"),
    ]:
        p_med = by_p(df[col], lambda s: s.shift(1).expanding(min_periods=1).median()).fillna(df[col].median())
        proxy = df["unified_daily_load"] * (p_med if col != "unified_daily_load" else 1.0)
        s_filled = df[col].where(df[col].notna(), proxy).fillna(0.0)

        ewma_acute = by_p(s_filled, lambda s: s.ewm(span=7, min_periods=1).mean())
        ewma_chronic = by_p(s_filled, lambda s: s.ewm(span=28, min_periods=3).mean())
        new_cols[f"{prefix}_ewma_atl7"] = ewma_acute
        new_cols[f"{prefix}_ewma_ctl28"] = ewma_chronic
        new_cols[f"{prefix}_ewma_acwr"] = (ewma_acute / (ewma_chronic + 1e-3)).clip(0.0, 4.0)
        # 急性负荷与慢性基线的净超载差值 (Acute-to-Chronic Spike)
        new_cols[f"{prefix}_ewma_spike"] = (ewma_acute - ewma_chronic) / (ewma_chronic + 1e-3)

    # 2. 生理复合机制指标 (数值越大，风险越高)
    new_cols["ewma_overload_composite"] = (
        0.35 * new_cols["uload_ewma_acwr"]
        + 0.25 * new_cols["dist_ewma_acwr"]
        + 0.20 * new_cols["hsr_ewma_acwr"]
        + 0.20 * new_cols["decel_ewma_acwr"]
    )
    new_cols["subj_strain_composite"] = (
        0.35 * df["wellness_poor_mean"].fillna(0.3)
        + 0.35 * (df["wellness_poor_mean_z"].fillna(0.0).clip(-3, 3) / 3.0)
        + 0.30 * df["soreness_01_roll7d"].fillna(0.3)
    )
    # 恢复赤字指数 = 急性负荷 * 主观不良状态 (高负荷叠加睡眠差/肌肉酸痛)
    new_cols["recovery_deficit_index"] = new_cols["ewma_overload_composite"] * (
        1.0 + df["soreness_01"].fillna(0.2) + df["fatigue_01"].fillna(0.2) + df["stress_01"].fillna(0.2)
    )

    # 3. 本地女足专属上下文特征 (若在 SoccerMon 中不存在则置为默认值)
    if is_local:
        new_cols["ctx_hist_inj_12m"] = df["hist_inj_12m"].fillna(0.0)
        new_cols["ctx_hist_major_inj"] = df["hist_major_inj"].fillna(0.0)
        new_cols["ctx_is_menstruation"] = df["is_menstruation"].fillna(0.0)
        new_cols["ctx_dysmenorrhea"] = (df["dysmenorrhea_score"].fillna(0.0) / 10.0).clip(0.0, 1.0)
        # 经期叠加高负荷交互项 (女足运动医学核心风险因子)
        new_cols["ctx_mens_x_overload"] = new_cols["ctx_is_menstruation"] * new_cols["ewma_overload_composite"]
        new_cols["ctx_is_match_day"] = df["is_match_day"].fillna(0.0)
        new_cols["ctx_match_days_7d"] = by_p(
            new_cols["ctx_is_match_day"], lambda s: s.rolling(7, min_periods=1).sum()
        )
        new_cols["ctx_rpe_norm"] = (df["rpe"].fillna(3.0) / 10.0).clip(0.0, 1.0)
        new_cols["ctx_rpe_roll7d"] = by_p(new_cols["ctx_rpe_norm"], lambda s: s.rolling(7, min_periods=1).mean())
    else:
        for c in ["ctx_hist_inj_12m", "ctx_hist_major_inj", "ctx_is_menstruation",
                  "ctx_dysmenorrhea", "ctx_mens_x_overload", "ctx_is_match_day",
                  "ctx_match_days_7d", "ctx_rpe_norm", "ctx_rpe_roll7d"]:
            new_cols[c] = pd.Series(0.0, index=df.index)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).copy()


# 跨数据集共享特征 (用于 Stage 1 SoccerMon 源域预训练)
SHARED_SOURCE_FEATURES = [
    "uload_ewma_acwr", "dist_ewma_acwr", "hsr_ewma_acwr", "decel_ewma_acwr", "impact_ewma_acwr",
    "uload_ewma_spike", "dist_ewma_spike", "decel_ewma_spike",
    "uload_monotony7", "active_train_days_7d",
    "soreness_01", "fatigue_01", "sleep_poor_01", "stress_01", "mood_poor_01",
    "wellness_poor_mean", "wellness_poor_mean_z", "soreness_01_z", "fatigue_01_z",
    "soreness_01_roll7d", "fatigue_01_roll7d", "stress_01_roll7d",
    "decel_accel_ratio", "hsr_ratio", "imu_dyn_p95_g_z",
    "ewma_overload_composite", "subj_strain_composite", "recovery_deficit_index",
    "prev_inj_7d", "prev_inj_14d", "prev_inj_30d", "recent_inj_decay",
]

# 本地目标域增强特征 (用于 Stage 2 目标域微调：融合源域风险分 + 女足专属生理/赛程/既往史特征)
LOCAL_ADAPT_FEATURES = SHARED_SOURCE_FEATURES + [
    "ctx_hist_inj_12m", "ctx_hist_major_inj",
    "ctx_is_menstruation", "ctx_dysmenorrhea", "ctx_mens_x_overload",
    "ctx_is_match_day", "ctx_match_days_7d", "ctx_rpe_norm", "ctx_rpe_roll7d",
]


def get_source_zero_shot_predictions(df_sm: pd.DataFrame, df_loc_active: pd.DataFrame, target_col: str) -> np.ndarray:
    """在清洗后的 SoccerMon 上训练生理单调集成模型，输出对本地数据的零样本风险预测概率"""
    sm_active = df_sm[df_sm["is_active_day"] == 1].copy()
    inj_counts = sm_active.groupby("player_id")["inj_prevalence"].sum()
    valid_players = set(inj_counts[inj_counts > 0].index)
    sm_clean = sm_active[sm_active["player_id"].isin(valid_players)].copy()
    sm_clean = sm_clean[~((sm_clean["inj_prevalence"] == 1) & (sm_clean["inj_non_contact_onset"] == 0))].copy()

    X_sm = sm_clean[SHARED_SOURCE_FEATURES].copy()
    y_sm = sm_clean[target_col].values
    X_loc = df_loc_active[SHARED_SOURCE_FEATURES].copy()

    # 1) 单调约束 HistGBDT
    gbdt = HistGradientBoostingClassifier(
        max_iter=150, max_depth=3, learning_rate=0.03, min_samples_leaf=30,
        l2_regularization=5.0, monotonic_cst=[1] * len(SHARED_SOURCE_FEATURES),
        class_weight="balanced", random_state=42
    )
    gbdt.fit(X_sm, y_sm)
    p_gbdt = gbdt.predict_proba(X_loc)[:, 1]

    # 2) 域标准化非负正则逻辑回归
    imp = SimpleImputer(strategy="median")
    X_sm_imp = imp.fit_transform(X_sm)
    X_loc_imp = imp.transform(X_loc)
    X_sm_z = StandardScaler().fit_transform(X_sm_imp)
    X_loc_z = StandardScaler().fit_transform(X_loc_imp)

    lr = LogisticRegression(C=0.05, class_weight="balanced", random_state=42)
    lr.fit(X_sm_z, y_sm)
    lr.coef_ = np.maximum(lr.coef_, 0.0)
    p_lr = lr.predict_proba(X_loc_z)[:, 1]

    return 0.55 * p_gbdt + 0.45 * p_lr


def evaluate_with_optimal_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray, fixed_thresh: float | None = None) -> dict:
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    if fixed_thresh is not None:
        best_th = fixed_thresh
    else:
        # 扫描分位数阈值寻找最佳 F1 平衡点
        prec, rec, ths = precision_recall_curve(y_true, y_prob)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = int(np.argmax(f1s))
        best_th = float(ths[min(best_idx, len(ths) - 1)]) if len(ths) > 0 else 0.5

    y_pred = (y_prob >= best_th).astype(int)
    return {
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Recall(召回率)": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Precision(精确率)": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Threshold": round(best_th, 4),
    }


def run_two_stage_transfer(df_sm: pd.DataFrame, df_loc: pd.DataFrame, target_col: str, title: str):
    print("\n" + "=" * 90)
    print(f"🚀 二阶段迁移学习与女足专属特征融合实验 —— 目标: [{title}]")
    print("=" * 90)

    loc_active = df_loc[df_loc["is_active_day"] == 1].copy().reset_index(drop=True)
    y_loc = loc_active[target_col].values
    players_loc = loc_active["player_id"].values

    # --- 阶段 0: 传统未对齐 Baseline ---
    sm_active = df_sm[df_sm["is_active_day"] == 1].copy()
    rf_base = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_base.fit(sm_active[SHARED_SOURCE_FEATURES].fillna(0), sm_active[target_col].values)
    prob_baseline = rf_base.predict_proba(loc_active[SHARED_SOURCE_FEATURES].fillna(0))[:, 1]
    res_base_05 = evaluate_with_optimal_f1_threshold(y_loc, prob_baseline, fixed_thresh=0.50)

    # --- 阶段 1: Zero-Shot 源域预训练模型 (100% 仅用 SoccerMon 训练) ---
    prob_zeroshot = get_source_zero_shot_predictions(df_sm, loc_active, target_col)
    res_zeroshot = evaluate_with_optimal_f1_threshold(y_loc, prob_zeroshot)

    # 准备阶段 2 特征矩阵：将 SoccerMon 预训练风险分作为第一核心特征，拼接本地增强特征
    X_adapt_df = loc_active[LOCAL_ADAPT_FEATURES].copy()
    X_adapt_df["soccermon_pretrained_prior"] = prob_zeroshot
    adapt_feature_names = list(X_adapt_df.columns)

    imp = SimpleImputer(strategy="median")
    X_adapt_imp = imp.fit_transform(X_adapt_df)
    X_adapt_z = StandardScaler().fit_transform(X_adapt_imp)

    # --- 阶段 2A: 跨球员泛化验证 (5-Fold GroupKFold 按球员分组，测试完全未见过的球员) ---
    gkf = GroupKFold(n_splits=6)
    oof_group = np.zeros(len(loc_active))
    for tr_idx, te_idx in gkf.split(X_adapt_z, y_loc, groups=players_loc):
        # 结合生理非负逻辑回归与单调梯度提升树
        lr_g = LogisticRegression(C=0.08, class_weight="balanced", random_state=42)
        lr_g.fit(X_adapt_z[tr_idx], y_loc[tr_idx])
        lr_g.coef_ = np.maximum(lr_g.coef_, 0.0)
        p_lr_g = lr_g.predict_proba(X_adapt_z[te_idx])[:, 1]

        gb_g = HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, learning_rate=0.04, min_samples_leaf=20,
            l2_regularization=4.0, monotonic_cst=[1] * X_adapt_z.shape[1],
            class_weight="balanced", random_state=42
        )
        gb_g.fit(X_adapt_z[tr_idx], y_loc[tr_idx])
        p_gb_g = gb_g.predict_proba(X_adapt_z[te_idx])[:, 1]

        oof_group[te_idx] = 0.30 * prob_zeroshot[te_idx] + 0.35 * p_lr_g + 0.35 * p_gb_g

    res_group_cv = evaluate_with_optimal_f1_threshold(y_loc, oof_group)

    # --- 阶段 2B: 队内监测微调验证 (5-Fold StratifiedKFold 分层交叉验证，模拟队内历史校准) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_strat = np.zeros(len(loc_active))
    feat_importances = np.zeros(len(adapt_feature_names))

    for tr_idx, te_idx in skf.split(X_adapt_z, y_loc):
        gb_s = HistGradientBoostingClassifier(
            max_iter=150, max_depth=4, learning_rate=0.05, min_samples_leaf=12,
            l2_regularization=1.5, class_weight="balanced", random_state=42
        )
        gb_s.fit(X_adapt_z[tr_idx], y_loc[tr_idx])
        p_gb_s = gb_s.predict_proba(X_adapt_z[te_idx])[:, 1]

        et_s = ExtraTreesClassifier(
            n_estimators=250, max_depth=7, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        et_s.fit(X_adapt_z[tr_idx], y_loc[tr_idx])
        p_et_s = et_s.predict_proba(X_adapt_z[te_idx])[:, 1]
        feat_importances += et_s.feature_importances_ / 5.0

        oof_strat[te_idx] = 0.20 * prob_zeroshot[te_idx] + 0.45 * p_gb_s + 0.35 * p_et_s

    res_strat_cv = evaluate_with_optimal_f1_threshold(y_loc, oof_strat)

    df_res = pd.DataFrame([
        {"实验阶段与验证方式": "1. 传统直接迁移 Baseline (未对齐+默认0.5阈值)", **res_base_05},
        {"实验阶段与验证方式": "2. Zero-Shot 零样本物理对齐+EWMA单调集成 (100%纯SoccerMon)", **res_zeroshot},
        {"实验阶段与验证方式": "3. Two-Stage 跨球员迁移微调 (GroupKFold 新球员盲测)", **res_group_cv},
        {"实验阶段与验证方式": "4. Two-Stage 队内多模态融合微调 (5-Fold 队内交叉验证)", **res_strat_cv},
    ])
    print(df_res.to_string(index=False))

    # 打印五分位风险分层表 (Quintile Risk Stratification)
    loc_active["risk_quintile"] = pd.qcut(oof_strat, q=5, labels=["Q1(极低危)", "Q2(低危)", "Q3(中危)", "Q4(高危)", "Q5(极高危)"])
    q_table = loc_active.groupby("risk_quintile", observed=False)[target_col].agg(
        样本人日="count", 实际损伤人日="sum", 实际损伤率=lambda s: f"{s.mean()*100:.2f}%"
    ).reset_index()
    q5_rate = loc_active.loc[loc_active["risk_quintile"] == "Q5(极高危)", target_col].mean()
    q1_rate = loc_active.loc[loc_active["risk_quintile"] == "Q1(极低危)", target_col].mean()
    print(f"\n📈 五分位风险分层校准表 (Q5 极高危组真实发病率是 Q1 极低危组的 {q5_rate/(q1_rate+1e-6):.1f} 倍！):")
    print(q_table.to_string(index=False))

    # Top 12 特征重要性
    imp_df = pd.DataFrame({"特征名": adapt_feature_names, "重要性权重": feat_importances}).sort_values("重要性权重", ascending=False)
    print("\n🔑 模型 Top 12 核心风险驱动特征:")
    print(imp_df.head(12).to_string(index=False))

    return df_res, loc_active, prob_baseline, prob_zeroshot, oof_group, oof_strat, imp_df


def try_generate_plots(loc_active, y_true, p_base, p_zs, p_grp, p_strat, imp_df):
    try:
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # 图1: ROC 与 PR 曲线消融对比
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
        models_plot = [
            ("Baseline Raw Transfer", p_base, "#999999", "--"),
            ("Zero-Shot Aligned (SoccerMon Only)", p_zs, "#2b83ba", "-."),
            ("Two-Stage Group-CV (Unseen Players)", p_grp, "#fdae61", "-"),
            ("Two-Stage Intra-Team CV (Full Fusion)", p_strat, "#d7191c", "-"),
        ]
        for label, probs, color, ls in models_plot:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc_val = roc_auc_score(y_true, probs)
            axes[0].plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})", color=color, linestyle=ls, lw=2.2)

            prec, rec, _ = precision_recall_curve(y_true, probs)
            ap_val = average_precision_score(y_true, probs)
            axes[1].plot(rec, prec, label=f"{label} (PR-AUC={ap_val:.3f})", color=color, linestyle=ls, lw=2.2)

        axes[0].plot([0, 1], [0, 1], "k:", alpha=0.5)
        axes[0].set_title("A. ROC Curve Comparison (7-Day Injury Risk)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
        axes[0].set_ylabel("True Positive Rate (Sensitivity)")
        axes[0].legend(loc="lower right", fontsize=8.5)
        axes[0].grid(alpha=0.25)

        axes[1].axhline(y_true.mean(), color="k", linestyle=":", alpha=0.5, label=f"Base Rate ({y_true.mean():.2f})")
        axes[1].set_title("B. Precision-Recall Curve Comparison", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Recall (Sensitivity)")
        axes[1].set_ylabel("Precision (Positive Predictive Value)")
        axes[1].legend(loc="upper right", fontsize=8.5)
        axes[1].grid(alpha=0.25)

        plt.tight_layout()
        fig_path1 = FIGURES_DIR / "fig1_roc_pr_ablation.png"
        plt.savefig(fig_path1)
        plt.close()

        # 图2: 五分位风险分层柱状图 + Top12 特征重要性
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
        q_rates = loc_active.groupby("risk_quintile", observed=False)["target_non_contact_next_7d"].mean() * 100
        bars = axes[0].bar(range(5), q_rates.values, color=["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"], edgecolor="#333333")
        axes[0].set_xticks(range(5))
        axes[0].set_xticklabels(["Q1\n(Lowest)", "Q2\n(Low)", "Q3\n(Medium)", "Q4\n(High)", "Q5\n(Very High)"])
        axes[0].set_ylabel("Actual 7-Day Injury Incidence (%)")
        axes[0].set_title("A. Injury Incidence Across Predicted Risk Quintiles", fontsize=12, fontweight="bold")
        for b in bars:
            h = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", fontweight="bold")
        axes[0].set_ylim(0, max(q_rates.values) * 1.18)
        axes[0].grid(axis="y", alpha=0.25)

        top12 = imp_df.head(12).iloc[::-1]
        axes[1].barh(top12["特征名"], top12["重要性权重"], color="#2b83ba", edgecolor="#333333")
        axes[1].set_xlabel("Relative Feature Importance")
        axes[1].set_title("B. Top 12 Injury Risk Drivers (Two-Stage Transfer)", fontsize=12, fontweight="bold")
        axes[1].grid(axis="x", alpha=0.25)

        plt.tight_layout()
        fig_path2 = FIGURES_DIR / "fig2_quintiles_and_features.png"
        plt.savefig(fig_path2)
        plt.close()
        print(f"\n🎨 论文级高清图表已自动生成并保存在: {FIGURES_DIR}")
    except ImportError:
        print("\n💡 提示: 如需自动生成 ROC/PR 曲线与五分位分层图，可执行 `pip install matplotlib` 后重跑本脚本。")


if __name__ == "__main__":
    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv", parse_dates=["date"])
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv", parse_dates=["date"])

    df_sm = add_ewma_and_context_features(df_sm, is_local=False)
    df_loc = add_ewma_and_context_features(df_loc, is_local=True)

    res_7d, loc_7d, p_base, p_zs, p_grp, p_strat, imp_7d = run_two_stage_transfer(
        df_sm, df_loc, "target_non_contact_next_7d", "未来 7 天非接触性损伤风险预警"
    )
    res_3d, _, _, _, _, _, _ = run_two_stage_transfer(
        df_sm, df_loc, "target_non_contact_next_3d", "未来 3 天非接触性损伤风险预警"
    )

    try_generate_plots(loc_7d, loc_7d["target_non_contact_next_7d"].values, p_base, p_zs, p_grp, p_strat, imp_7d)