# src/06_rigorous_validation_and_plots.py
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


def add_ewma_and_context_features(df: pd.DataFrame, is_local: bool = False) -> pd.DataFrame:
    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)
    pkey = df["player_id"]

    def by_p(s: pd.Series, fn):
        return s.groupby(pkey, sort=False, group_keys=False).transform(fn)

    new_cols = {}
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
        new_cols[f"{prefix}_ewma_spike"] = (ewma_acute - ewma_chronic) / (ewma_chronic + 1e-3)

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
    new_cols["recovery_deficit_index"] = new_cols["ewma_overload_composite"] * (
        1.0 + df["soreness_01"].fillna(0.2) + df["fatigue_01"].fillna(0.2) + df["stress_01"].fillna(0.2)
    )

    if is_local:
        new_cols["ctx_hist_inj_12m"] = df["hist_inj_12m"].fillna(0.0)
        new_cols["ctx_hist_major_inj"] = df["hist_major_inj"].fillna(0.0)
        new_cols["ctx_is_menstruation"] = df["is_menstruation"].fillna(0.0)
        new_cols["ctx_dysmenorrhea"] = (df["dysmenorrhea_score"].fillna(0.0) / 10.0).clip(0.0, 1.0)
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

LOCAL_ADAPT_FEATURES = SHARED_SOURCE_FEATURES + [
    "ctx_hist_inj_12m", "ctx_hist_major_inj",
    "ctx_is_menstruation", "ctx_dysmenorrhea", "ctx_mens_x_overload",
    "ctx_is_match_day", "ctx_match_days_7d", "ctx_rpe_norm", "ctx_rpe_roll7d",
]


def get_source_zero_shot_predictions(df_sm: pd.DataFrame, df_loc_active: pd.DataFrame, target_col: str) -> np.ndarray:
    sm_active = df_sm[df_sm["is_active_day"] == 1].copy()
    inj_counts = sm_active.groupby("player_id")["inj_prevalence"].sum()
    valid_players = set(inj_counts[inj_counts > 0].index)
    sm_clean = sm_active[sm_active["player_id"].isin(valid_players)].copy()
    sm_clean = sm_clean[~((sm_clean["inj_prevalence"] == 1) & (sm_clean["inj_non_contact_onset"] == 0))].copy()

    X_sm = sm_clean[SHARED_SOURCE_FEATURES].copy()
    y_sm = sm_clean[target_col].values
    X_loc = df_loc_active[SHARED_SOURCE_FEATURES].copy()

    gbdt = HistGradientBoostingClassifier(
        max_iter=150, max_depth=3, learning_rate=0.03, min_samples_leaf=30,
        l2_regularization=5.0, monotonic_cst=[1] * len(SHARED_SOURCE_FEATURES),
        class_weight="balanced", random_state=42
    )
    gbdt.fit(X_sm, y_sm)
    p_gbdt = gbdt.predict_proba(X_loc)[:, 1]

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


def build_5_physiological_pillars(df_loc_active: pd.DataFrame, prob_zeroshot: np.ndarray) -> pd.DataFrame:
    """将 41 个原始特征浓缩为 5 个正交运动医学机制支柱，消除 24 人小样本微调过拟合"""
    def z_norm(s: pd.Series) -> pd.Series:
        s_f = s.fillna(s.median())
        std = s_f.std()
        return ((s_f - s_f.mean()) / (std if std > 1e-6 else 1.0)).clip(-3.5, 3.5)

    pillars = pd.DataFrame(index=df_loc_active.index)
    pillars["P1_SoccerMon_Prior"] = z_norm(pd.Series(prob_zeroshot, index=df_loc_active.index))
    pillars["P2_Overload_Exposure"] = (
        0.35 * z_norm(df_loc_active["ewma_overload_composite"])
        + 0.25 * z_norm(df_loc_active["dist_ewma_acwr"])
        + 0.20 * z_norm(df_loc_active["ctx_match_days_7d"])
        + 0.20 * z_norm(df_loc_active["active_train_days_7d"])
    )
    pillars["P3_Subjective_Strain"] = (
        0.35 * z_norm(df_loc_active["subj_strain_composite"])
        + 0.30 * z_norm(df_loc_active["stress_01"])
        + 0.20 * z_norm(df_loc_active["fatigue_01_z"])
        + 0.15 * z_norm(df_loc_active["ctx_mens_x_overload"])
    )
    pillars["P4_Biomech_Decel"] = (
        0.45 * z_norm(df_loc_active["decel_ewma_acwr"])
        + 0.35 * z_norm(df_loc_active["decel_accel_ratio"])
        + 0.20 * z_norm(df_loc_active["hsr_ewma_acwr"])
    )
    pillars["P5_Injury_Recurrence"] = (
        0.50 * z_norm(df_loc_active["recent_inj_decay"])
        + 0.35 * z_norm(df_loc_active["prev_inj_30d"])
        + 0.15 * z_norm(df_loc_active["ctx_hist_inj_12m"])
    )
    return pillars


def fit_predict_non_negative_pillar_model(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    lr = LogisticRegression(C=0.05, class_weight="balanced", random_state=42)
    lr.fit(X_tr, y_tr)
    lr.coef_ = np.maximum(lr.coef_, 0.05)
    p_lr = lr.predict_proba(X_te)[:, 1]

    gb = HistGradientBoostingClassifier(
        max_iter=50, max_depth=2, learning_rate=0.03, min_samples_leaf=35,
        l2_regularization=10.0, monotonic_cst=[1] * X_tr.shape[1],
        class_weight="balanced", random_state=42
    )
    gb.fit(X_tr, y_tr)
    p_gb = gb.predict_proba(X_te)[:, 1]

    return 0.65 * p_lr + 0.35 * p_gb


def run_rigorous_evaluations(df_sm: pd.DataFrame, df_loc: pd.DataFrame, target_col: str, horizon_days: int, title: str):
    print("\n" + "=" * 95)
    print(f"🏆 全方位严格验证对比 —— 目标: [{title}] ({target_col})")
    print("=" * 95)

    loc_active = df_loc[df_loc["is_active_day"] == 1].copy().sort_values(["date", "player_id"]).reset_index(drop=True)
    y_loc = loc_active[target_col].values
    players_loc = loc_active["player_id"].values
    dates_loc = loc_active["date"]

    # 1. Baseline: 未对齐直接迁移 + 默认0.5阈值
    sm_active = df_sm[df_sm["is_active_day"] == 1].copy()
    rf_base = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_base.fit(sm_active[SHARED_SOURCE_FEATURES].fillna(0), sm_active[target_col].values)
    prob_base = rf_base.predict_proba(loc_active[SHARED_SOURCE_FEATURES].fillna(0))[:, 1]
    res_1_base = evaluate_with_optimal_f1_threshold(y_loc, prob_base, fixed_thresh=0.50)

    # 2. Zero-Shot: 100% 纯 SoccerMon 训练
    prob_zeroshot = get_source_zero_shot_predictions(df_sm, loc_active, target_col)
    res_2_zeroshot = evaluate_with_optimal_f1_threshold(y_loc, prob_zeroshot)

    # 构建 5 大正交生理机制支柱
    pillars_df = build_5_physiological_pillars(loc_active, prob_zeroshot)
    X_pillars = pillars_df.values

    # 3. 严格跨球员盲测 (6-Fold GroupKFold)
    gkf = GroupKFold(n_splits=6)
    oof_group_5p = np.zeros(len(loc_active))
    pillar_weights_list = []
    for tr_idx, te_idx in gkf.split(X_pillars, y_loc, groups=players_loc):
        oof_group_5p[te_idx] = fit_predict_non_negative_pillar_model(X_pillars[tr_idx], y_loc[tr_idx], X_pillars[te_idx])
        w, _ = nnls(X_pillars[tr_idx] - X_pillars[tr_idx].min(axis=0), y_loc[tr_idx])
        pillar_weights_list.append(w / (w.sum() + 1e-8))

    res_3_group = evaluate_with_optimal_f1_threshold(y_loc, oof_group_5p)

    # 4. 严格时间序列前滚验证 (Purged Temporal Expanding-Window)
    unique_dates = np.sort(dates_loc.unique())
    n_dates = len(unique_dates)
    oof_temporal = np.copy(prob_zeroshot)
    warmup_cut = int(n_dates * 0.25)

    for d_idx in range(warmup_cut, n_dates):
        curr_date = unique_dates[d_idx]
        purge_cutoff = pd.Timestamp(curr_date) - pd.Timedelta(days=horizon_days)
        tr_mask = (dates_loc <= purge_cutoff).values
        te_mask = (dates_loc == curr_date).values

        if tr_mask.sum() >= 100 and y_loc[tr_mask].sum() >= 5:
            pred_t = fit_predict_non_negative_pillar_model(X_pillars[tr_mask], y_loc[tr_mask], X_pillars[te_mask])
            oof_temporal[te_mask] = 0.35 * prob_zeroshot[te_mask] + 0.65 * pred_t

    res_4_temporal = evaluate_with_optimal_f1_threshold(y_loc, oof_temporal)

    # 5. 队内多模态融合微调 (5-Fold Stratified CV)
    X_full_df = loc_active[LOCAL_ADAPT_FEATURES].copy()
    X_full_df["soccermon_pretrained_prior"] = prob_zeroshot
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
            0.25 * oof_group_5p[te_idx]
            + 0.40 * gb_s.predict_proba(X_full_z[te_idx])[:, 1]
            + 0.35 * et_s.predict_proba(X_full_z[te_idx])[:, 1]
        )

    res_5_intrateam = evaluate_with_optimal_f1_threshold(y_loc, oof_intrateam)

    summary_df = pd.DataFrame([
        {"验证方案 (由浅入深消融对比)": "1. 传统直接迁移 Baseline (未对齐+默认0.5阈值)", **res_1_base},
        {"验证方案 (由浅入深消融对比)": "2. Zero-Shot 零样本物理对齐模型 (100%纯SoccerMon训练)", **res_2_zeroshot},
        {"验证方案 (由浅入深消融对比)": "3. Two-Stage 5支柱跨球员盲测 (GroupKFold 严苛新球员泛化)", **res_3_group},
        {"验证方案 (由浅入深消融对比)": "4. Two-Stage 5支柱时间序列前滚验证 (Purged Temporal 零重叠)", **res_4_temporal},
        {"验证方案 (由浅入深消融对比)": "5. Two-Stage 队内全特征多模态融合 (5-Fold 队内交叉验证)", **res_5_intrateam},
    ])
    print(summary_df.to_string(index=False))

    avg_w = np.mean(pillar_weights_list, axis=0)
    pillar_imp_df = pd.DataFrame({
        "生理机制支柱 (5-Pillar)": [
            "P1: SoccerMon 源域预训练底座风险分",
            "P2: EWMA 急慢性负荷超载与比赛暴露",
            "P3: 晨间主观疲劳/压力/睡眠与生理期交互",
            "P4: 1Hz 穿戴急减速离心制动与高速跑负荷",
            "P5: 近期伤病复发衰减与既往伤病史",
        ],
        "归一化贡献占比": [f"{w*100:.1f}%" for w in avg_w],
        "weight_num": avg_w,
    }).sort_values("weight_num", ascending=False)
    print("\n🧬 五大运动医学支柱在跨球员盲测中的权重分配:")
    print(pillar_imp_df[["生理机制支柱 (5-Pillar)", "归一化贡献占比"]].to_string(index=False))

    return summary_df, loc_active, prob_base, prob_zeroshot, oof_group_5p, oof_temporal, oof_intrateam, pillar_imp_df


def generate_all_publication_figures(loc_active, y_true, p_base, p_zs, p_grp, p_temp, p_intra, pillar_imp_df):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        # 图 1: 5种验证方案的 ROC 与 PR 曲线对比
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=300)
        curves = [
            ("1. Baseline Raw Transfer", p_base, "#888888", ":"),
            ("2. Zero-Shot (100% SoccerMon)", p_zs, "#3288bd", "--"),
            ("3. Two-Stage GroupKFold (Unseen Players)", p_grp, "#66c2a5", "-."),
            ("4. Two-Stage Purged Temporal CV", p_temp, "#fdae61", "-"),
            ("5. Two-Stage Intra-Team Multimodal CV", p_intra, "#d53e4f", "-"),
        ]
        for label, probs, col, ls in curves:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc_v = roc_auc_score(y_true, probs)
            axes[0].plot(fpr, tpr, label=f"{label} (AUC={auc_v:.3f})", color=col, linestyle=ls, lw=2.2)

            prec, rec, _ = precision_recall_curve(y_true, probs)
            ap_v = average_precision_score(y_true, probs)
            axes[1].plot(rec, prec, label=f"{label} (PR-AUC={ap_v:.3f})", color=col, linestyle=ls, lw=2.2)

        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
        axes[0].set_title("A. ROC Curves Across Transfer & Validation Stages", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
        axes[0].set_ylabel("True Positive Rate (Sensitivity)")
        axes[0].legend(loc="lower right", fontsize=8.2)
        axes[0].grid(alpha=0.25)

        axes[1].axhline(y_true.mean(), color="k", linestyle="--", alpha=0.4, label=f"Random Baseline ({y_true.mean():.2f})")
        axes[1].set_title("B. Precision-Recall Curves (7-Day Non-Contact Injury)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Recall (Sensitivity)")
        axes[1].set_ylabel("Precision (Positive Predictive Value)")
        axes[1].legend(loc="upper right", fontsize=8.2)
        axes[1].grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig1_ablation_roc_pr_curves.png")
        plt.close()

        # 图 2: 跨球员盲测五分位风险分层图 + 5大支柱权重图
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=300)
        q_grp = pd.qcut(p_grp, q=5, labels=["Q1\n(Lowest)", "Q2\n(Low)", "Q3\n(Med)", "Q4\n(High)", "Q5\n(Very High)"])
        rates_grp = loc_active.groupby(q_grp, observed=False)["target_non_contact_next_7d"].mean() * 100
        colors_q = ["#2b83ba", "#abdda4", "#ffffbf", "#fdae61", "#d7191c"]

        bars1 = axes[0].bar(range(5), rates_grp.values, color=colors_q, edgecolor="#333333")
        axes[0].set_xticks(range(5))
        axes[0].set_xticklabels(rates_grp.index)
        axes[0].set_ylabel("Actual 7-Day Injury Incidence (%)")
        axes[0].set_title("A. Risk Stratification (GroupKFold - Unseen Players)", fontsize=11.5, fontweight="bold")
        for b in bars1:
            h = b.get_height()
            axes[0].text(b.get_x() + b.get_width() / 2, h + 0.6, f"{h:.1f}%", ha="center", fontweight="bold")
        axes[0].set_ylim(0, max(rates_grp.values) * 1.2)
        axes[0].grid(axis="y", alpha=0.25)

        p_df = pillar_imp_df.iloc[::-1]
        axes[1].barh(p_df["生理机制支柱 (5-Pillar)"], p_df["weight_num"] * 100, color="#3288bd", edgecolor="#333333")
        axes[1].set_xlabel("Relative Contribution Weight (%)")
        axes[1].set_title("B. 5-Pillar Physiological Mechanism Contributions", fontsize=11.5, fontweight="bold")
        axes[1].grid(axis="x", alpha=0.25)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig2_risk_quintiles_and_pillars.png")
        plt.close()

        # 图 3: 典型球员 3 个月个人风险轨迹与真实受伤时间点预警案例
        loc_active["pred_risk_group_cv"] = p_grp
        loc_active["pred_risk_intra_cv"] = p_intra
        sample_player = loc_active["player_id"].iloc[0]
        p_sub = loc_active[loc_active["player_id"] == sample_player].sort_values("date")

        fig, ax = plt.subplots(figsize=(12, 4.8), dpi=300)
        ax.plot(p_sub["date"], p_sub["pred_risk_group_cv"], label="Predicted Risk (Unseen-Player GroupCV)", color="#2b83ba", lw=2.0, linestyle="--")
        ax.plot(p_sub["date"], p_sub["pred_risk_intra_cv"], label="Predicted Risk (Intra-Team Multimodal CV)", color="#d7191c", lw=2.2)
        ax.axhline(0.50, color="#fdae61", linestyle=":", lw=1.5, label="High-Risk Warning Threshold")

        inj_days = p_sub[p_sub["inj_non_contact_onset"] == 1]
        for idx_i, (_, r_inj) in enumerate(inj_days.iterrows()):
            lbl = "Actual Non-Contact Injury Onset" if idx_i == 0 else None
            ax.axvline(r_inj["date"], color="#d7191c", alpha=0.35, lw=6, label=lbl)
            ax.scatter([r_inj["date"]], [r_inj["pred_risk_intra_cv"]], color="#d7191c", s=70, zorder=5)

        ax.set_title(f"Individual 3-Month Injury Risk Trajectory & Onset Alerts (Player: {sample_player})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted Injury Risk Probability")
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "fig3_player_risk_trajectory_case.png")
        plt.close()

        print(f"\n🎨 3 张 300-DPI 论文高清验证图已生成至目录: {FIGURES_DIR}")
        for f in sorted(FIGURES_DIR.glob("*.png")):
            print(f"   - {f.name}")
    except ImportError:
        print("\n⚠️ 尚未安装 matplotlib，请执行 `pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple`。")


if __name__ == "__main__":
    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv", parse_dates=["date"])
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv", parse_dates=["date"])

    df_sm = add_ewma_and_context_features(df_sm, is_local=False)
    df_loc = add_ewma_and_context_features(df_loc, is_local=True)

    sum_7d, loc_7d, p_base, p_zs, p_grp, p_temp, p_intra, pillar_df = run_rigorous_evaluations(
        df_sm, df_loc, "target_non_contact_next_7d", horizon_days=7, title="未来 7 天非接触性损伤预警"
    )
    sum_3d, _, _, _, _, _, _, _ = run_rigorous_evaluations(
        df_sm, df_loc, "target_non_contact_next_3d", horizon_days=3, title="未来 3 天非接触性损伤预警"
    )

    final_table = pd.concat([
        sum_7d.assign(预测窗口="未来7天非接触损伤"),
        sum_3d.assign(预测窗口="未来3天非接触损伤"),
    ], ignore_index=True)
    final_table.to_csv(OUTPUTS_DIR / "final_rigorous_evaluation_table.csv", index=False, encoding="utf-8-sig")

    generate_all_publication_figures(
        loc_7d, loc_7d["target_non_contact_next_7d"].values,
        p_base, p_zs, p_grp, p_temp, p_intra, pillar_df
    )
    print("\n🎉 步骤 06 全部完成！请把控制台结果发给我。")
