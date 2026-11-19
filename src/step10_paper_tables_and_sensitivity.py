# src/step10_paper_tables_and_sensitivity.py
"""
女足运动损伤风险跨数据集迁移与小样本外部验证 —— 标准学术报表生成脚本 (Step 10)
包含：
1. 源域严格双层嵌套 OOF (Nested Source OOF Stacking) 训练源域融合器与源域阈值；
2. 表 1：主验证结果表（同时报告整体 Pooled 指标 [Cluster 95% CI] 与外层各折 Mean ± SD）；
3. 表 2：不同预警比例预算（Top 10% / 15% / 20% / 25% / 30% Alert Budget）下的敏感性分析表；
4. 表 3：外层交叉验证各折独立性能明细表（Fold-by-Fold Breakdown）。
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

from step09_final_clean_pipeline import (
    add_causal_ewma_and_context_features,
    FEAT_EXPERT_OVERLOAD,
    FEAT_EXPERT_WELLNESS,
    FEAT_EXPERT_BIOMECH,
    FEAT_LOCAL_FEMALE,
    FEAT_INJ_HISTORY,
    OOFSourceExpertBranch,
    NestedOOFAdapter,
    select_threshold_from_oof,
    cluster_bootstrap_ci,
)


def fit_nested_source_2b_and_threshold(sm_clean: pd.DataFrame, loc_active: pd.DataFrame, target_col: str):
    """
    严格双层嵌套源域 Stacking (Strictly Nested Source OOF Stacking)：
    在外层第 k 折训练集中，独立重新训练三位源域专家并生成该外层训练折的内层 OOF 分数，
    再训练二层逻辑回归融合器预测第 k 折源域验证集，彻底消除源域不同折之间的信息交叉！
    """
    y_sm = sm_clean[target_col].values
    g_sm = sm_clean["player_id"].values
    outer_gkf = GroupKFold(n_splits=5)

    nested_sm_oof_2a = np.zeros(len(sm_clean))
    nested_sm_oof_2b = np.zeros(len(sm_clean))

    for out_tr, out_val in outer_gkf.split(sm_clean, y_sm, groups=g_sm):
        sm_tr_df = sm_clean.iloc[out_tr].reset_index(drop=True)
        sm_val_df = sm_clean.iloc[out_val].reset_index(drop=True)
        y_out_tr = y_sm[out_tr]

        # 在外层训练折 sm_tr_df 内部独立训练三位专家并获取内层 OOF
        e_ov = OOFSourceExpertBranch(FEAT_EXPERT_OVERLOAD, require_gps=False)
        e_bi = OOFSourceExpertBranch(FEAT_EXPERT_BIOMECH, require_gps=True)
        e_we = OOFSourceExpertBranch(FEAT_EXPERT_WELLNESS, require_gps=False)

        sm_tr_df["sm_expert_overload"] = e_ov.fit_and_get_source_oof(sm_tr_df, target_col)
        sm_tr_df["sm_expert_biomech"] = e_bi.fit_and_get_source_oof(sm_tr_df, target_col)
        sm_tr_df["sm_expert_wellness"] = e_we.fit_and_get_source_oof(sm_tr_df, target_col)

        # 对外层验证折 sm_val_df 仅做纯归纳预测
        sm_val_df["sm_expert_overload"] = e_ov.predict_inductive_score(sm_val_df)
        sm_val_df["sm_expert_biomech"] = e_bi.predict_inductive_score(sm_val_df)
        sm_val_df["sm_expert_wellness"] = e_we.predict_inductive_score(sm_val_df)

        nested_sm_oof_2a[out_val] = (
            0.40 * sm_val_df["sm_expert_overload"].values
            + 0.30 * sm_val_df["sm_expert_biomech"].values
            + 0.30 * sm_val_df["sm_expert_wellness"].values
        )

        zs_cols = ["sm_expert_overload", "sm_expert_biomech", "sm_expert_wellness", "recent_inj_decay", "prev_inj_30d"]
        imp_s = SimpleImputer(strategy="median").fit(sm_tr_df[zs_cols].values)
        sc_s = StandardScaler().fit(imp_s.transform(sm_tr_df[zs_cols].values))
        X_tr_s = sc_s.transform(imp_s.transform(sm_tr_df[zs_cols].values))
        X_val_s = sc_s.transform(imp_s.transform(sm_val_df[zs_cols].values))

        lr_s = LogisticRegression(C=0.05, class_weight="balanced", random_state=42).fit(X_tr_s, y_out_tr)
        lr_s.coef_ = np.maximum(lr_s.coef_, 0.05)
        nested_sm_oof_2b[out_val] = lr_s.predict_proba(X_val_s)[:, 1]

    # 基于严格双层嵌套源域 OOF 确定源域先验阈值
    th_2a = select_threshold_from_oof(y_sm, nested_sm_oof_2a)
    th_2b = select_threshold_from_oof(y_sm, nested_sm_oof_2b)

    # 在完整源域上拟合最终模型用于目标域盲测
    exp_ov_full = OOFSourceExpertBranch(FEAT_EXPERT_OVERLOAD, require_gps=False)
    exp_bi_full = OOFSourceExpertBranch(FEAT_EXPERT_BIOMECH, require_gps=True)
    exp_we_full = OOFSourceExpertBranch(FEAT_EXPERT_WELLNESS, require_gps=False)

    sm_clean_copy = sm_clean.copy()
    sm_clean_copy["sm_expert_overload"] = exp_ov_full.fit_and_get_source_oof(sm_clean_copy, target_col)
    sm_clean_copy["sm_expert_biomech"] = exp_bi_full.fit_and_get_source_oof(sm_clean_copy, target_col)
    sm_clean_copy["sm_expert_wellness"] = exp_we_full.fit_and_get_source_oof(sm_clean_copy, target_col)

    loc_out = loc_active.copy()
    loc_out["sm_expert_overload"] = exp_ov_full.predict_inductive_score(loc_out)
    loc_out["sm_expert_biomech"] = exp_bi_full.predict_inductive_score(loc_out)
    loc_out["sm_expert_wellness"] = exp_we_full.predict_inductive_score(loc_out)

    prob_2a_loc = (
        0.40 * loc_out["sm_expert_overload"].values
        + 0.30 * loc_out["sm_expert_biomech"].values
        + 0.30 * loc_out["sm_expert_wellness"].values
    )

    zs_cols = ["sm_expert_overload", "sm_expert_biomech", "sm_expert_wellness", "recent_inj_decay", "prev_inj_30d"]
    imp_full = SimpleImputer(strategy="median").fit(sm_clean_copy[zs_cols].values)
    sc_full = StandardScaler().fit(imp_full.transform(sm_clean_copy[zs_cols].values))
    lr_full = LogisticRegression(C=0.05, class_weight="balanced", random_state=42).fit(
        sc_full.transform(imp_full.transform(sm_clean_copy[zs_cols].values)), y_sm
    )
    lr_full.coef_ = np.maximum(lr_full.coef_, 0.05)
    prob_2b_loc = lr_full.predict_proba(sc_full.transform(imp_full.transform(loc_out[zs_cols].values)))[:, 1]

    return loc_out, prob_2a_loc, th_2a, prob_2b_loc, th_2b


def compute_fold_mean_sd(y_true: np.ndarray, y_prob: np.ndarray, fold_ids: np.ndarray) -> tuple[str, str]:
    """计算各个外层折或时间分段的 Mean ± SD"""
    aucs, aps = [], []
    for fid in np.unique(fold_ids):
        m = fold_ids == fid
        if len(np.unique(y_true[m])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[m], y_prob[m]))
        aps.append(average_precision_score(y_true[m], y_prob[m]))
    return f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}", f"{np.mean(aps):.4f} ± {np.std(aps):.4f}"


def evaluate_alert_budget_sensitivity(
    y_true: np.ndarray, prob_group: np.ndarray, prob_temp: np.ndarray, fold_ids_grp: np.ndarray, dates_loc: pd.Series, horizon_days: int
) -> pd.DataFrame:
    """
    表 2：不同预警比例预算 (Alert Budget: Top 10% / 15% / 20% / 25% / 30%) 敏感性分析表。
    注意：每个预警预算的分位数阈值同样严格在【训练折内部 (GroupKFold) 或纯历史窗口 (Temporal)】计算，不看测试集分位数！
    """
    budgets = [0.10, 0.15, 0.20, 0.25, 0.30]
    sens_rows = []
    base_rate = y_true.mean()

    for b in budgets:
        # 1) 跨运动员 GroupKFold 在训练折内部确定 Top-b% 分位阈值
        pred_grp_b = np.zeros(len(y_true), dtype=int)
        for fid in np.unique(fold_ids_grp):
            tr_m = fold_ids_grp != fid
            te_m = fold_ids_grp == fid
            th_b = float(np.quantile(prob_group[tr_m], 1.0 - b))
            pred_grp_b[te_m] = (prob_group[te_m] >= th_b).astype(int)

        actual_alert_grp = pred_grp_b.mean()
        rec_g = recall_score(y_true, pred_grp_b, zero_division=0)
        prec_g = precision_score(y_true, pred_grp_b, zero_division=0)
        f1_g = f1_score(y_true, pred_grp_b, zero_division=0)
        lift_g = prec_g / (base_rate + 1e-8)

        # 2) 时间前滚 Purged Temporal 在历史窗口确定 Top-b% 分位阈值
        pred_tmp_b = np.zeros(len(y_true), dtype=int)
        u_dates = np.sort(dates_loc.unique())
        w_cut = int(len(u_dates) * 0.25)
        # 预热期用前序 expanding 分位数
        for d_i, cd in enumerate(u_dates):
            te_m = (dates_loc == cd).values
            purge_c = pd.Timestamp(cd) - pd.Timedelta(days=horizon_days)
            tr_m = (dates_loc <= purge_c).values
            if tr_m.sum() >= 30:
                th_tb = float(np.quantile(prob_temp[tr_m], 1.0 - b))
            else:
                th_tb = float(np.quantile(prob_temp[: max(1, np.where(te_m)[0][0])], 1.0 - b))
            pred_tmp_b[te_m] = (prob_temp[te_m] >= th_tb).astype(int)

        actual_alert_tmp = pred_tmp_b.mean()
        rec_t = recall_score(y_true, pred_tmp_b, zero_division=0)
        prec_t = precision_score(y_true, pred_tmp_b, zero_division=0)
        f1_t = f1_score(y_true, pred_tmp_b, zero_division=0)
        lift_t = prec_t / (base_rate + 1e-8)

        sens_rows.append({
            "目标预警预算": f"Top {int(b*100)}%",
            "GroupKFold_实际报警率": f"{actual_alert_grp*100:.1f}%",
            "GroupKFold_Recall": round(rec_g, 4),
            "GroupKFold_Precision": round(prec_g, 4),
            "GroupKFold_F1": round(f1_g, 4),
            "GroupKFold_Lift(风险倍数)": f"{lift_g:.2f}x",
            "Temporal_实际报警率": f"{actual_alert_tmp*100:.1f}%",
            "Temporal_Recall": round(rec_t, 4),
            "Temporal_Precision": round(prec_t, 4),
            "Temporal_F1": round(f1_t, 4),
            "Temporal_Lift(风险倍数)": f"{lift_t:.2f}x",
        })

    return pd.DataFrame(sens_rows)


def run_step10_suite(df_sm: pd.DataFrame, df_loc: pd.DataFrame, target_col: str, horizon_days: int, title: str):
    print("\n" + "=" * 112)
    print(f"📊 论文标准评估报表 —— 任务: [{title}] ({target_col})")
    print("=" * 112)

    sm_active = df_sm[df_sm["is_active_day"] == 1].copy()
    inj_counts = sm_active.groupby("player_id")["inj_prevalence"].sum()
    valid_sm_players = set(inj_counts[inj_counts > 0].index)
    sm_clean = sm_active[sm_active["player_id"].isin(valid_sm_players)].copy()
    sm_clean = sm_clean[~((sm_clean["inj_prevalence"] == 1) & (sm_clean["inj_non_contact_onset"] == 0))].reset_index(drop=True)

    loc_active = df_loc[df_loc["is_active_day"] == 1].copy().sort_values(["date", "player_id"]).reset_index(drop=True)
    y_loc = loc_active[target_col].values
    players_loc = loc_active["player_id"].values
    dates_loc = loc_active["date"]
    date_blocks = ((dates_loc - dates_loc.min()).dt.days // 7).values

    # 1. Baseline
    base_cols = FEAT_EXPERT_OVERLOAD + FEAT_EXPERT_WELLNESS + FEAT_EXPERT_BIOMECH
    rf_base = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_base.fit(sm_active[base_cols].fillna(0).values, sm_active[target_col].values)
    prob_1_base = rf_base.predict_proba(loc_active[base_cols].fillna(0).values)[:, 1]
    pred_1_base = (prob_1_base >= 0.50).astype(int)

    # 2. 严格双层嵌套源域 Zero-Shot (2a & 2b)
    loc_active, prob_2a, th_2a, prob_2b, th_2b = fit_nested_source_2b_and_threshold(sm_clean, loc_active, target_col)
    pred_2a = (prob_2a >= th_2a).astype(int)
    pred_2b = (prob_2b >= th_2b).astype(int)

    # 3. 跨运动员 6-Fold GroupKFold (记录每一折编号与各折独立表现)
    gkf = GroupKFold(n_splits=6)
    oof_prob_grp_nh = np.zeros(len(loc_active))
    oof_pred_grp_nh = np.zeros(len(loc_active), dtype=int)
    oof_prob_grp_full = np.zeros(len(loc_active))
    oof_pred_grp_full = np.zeros(len(loc_active), dtype=int)
    fold_ids_grp = np.zeros(len(loc_active), dtype=int)
    fold_breakdown = []

    for f_idx, (tr_idx, te_idx) in enumerate(gkf.split(loc_active, y_loc, groups=players_loc), 1):
        fold_ids_grp[te_idx] = f_idx
        df_tr, y_tr = loc_active.iloc[tr_idx], y_loc[tr_idx]
        df_te, y_te = loc_active.iloc[te_idx], y_loc[te_idx]

        ad_nh = NestedOOFAdapter(include_history=False).fit_with_inner_group_oof_threshold(df_tr, y_tr)
        oof_prob_grp_nh[te_idx], oof_pred_grp_nh[te_idx] = ad_nh.predict(df_te)

        ad_full = NestedOOFAdapter(include_history=True).fit_with_inner_group_oof_threshold(df_tr, y_tr)
        oof_prob_grp_full[te_idx], oof_pred_grp_full[te_idx] = ad_full.predict(df_te)

        test_p_names = sorted(df_te["player_id"].unique())
        fold_breakdown.append({
            "外层折号": f"Fold {f_idx}",
            "测试集运动员(4人)": ",".join(test_p_names),
            "样本人日": len(y_te),
            "阳性人日": int(y_te.sum()),
            "阳性率": f"{y_te.mean()*100:.1f}%",
            "ZeroShot_AUC": round(roc_auc_score(y_te, prob_2b[te_idx]), 4) if len(np.unique(y_te)) > 1 else np.nan,
            "GroupKFold_3b_AUC": round(roc_auc_score(y_te, oof_prob_grp_full[te_idx]), 4) if len(np.unique(y_te)) > 1 else np.nan,
            "GroupKFold_3b_AP": round(average_precision_score(y_te, oof_prob_grp_full[te_idx]), 4) if len(np.unique(y_te)) > 1 else np.nan,
        })

    # 4. 严格时间序列前滚 Purged Temporal CV
    unique_dates = np.sort(dates_loc.unique())
    n_dates = len(unique_dates)
    oof_prob_temp = np.copy(prob_2b)
    oof_pred_temp = np.copy(pred_2b)
    warmup_cut = int(n_dates * 0.25)

    for d_idx in range(warmup_cut, n_dates):
        curr_date = unique_dates[d_idx]
        purge_cutoff = pd.Timestamp(curr_date) - pd.Timedelta(days=horizon_days)
        tr_mask = (dates_loc <= purge_cutoff).values
        te_mask = (dates_loc == curr_date).values

        n_hist = int(tr_mask.sum())
        if n_hist >= 120 and y_loc[tr_mask].sum() >= 8:
            ad_temp = NestedOOFAdapter(include_history=True)
            ad_temp._fit_core(loc_active.iloc[tr_mask], y_loc[tr_mask])
            p_local_te = ad_temp._predict_prob_only(loc_active.iloc[te_mask])

            alpha = float(np.clip((n_hist - 100) / 1000.0, 0.15, 0.50))
            oof_prob_temp[te_mask] = (1.0 - alpha) * prob_2b[te_mask] + alpha * p_local_te

            th_hist_oof = select_threshold_from_oof(y_loc[tr_mask], oof_prob_temp[tr_mask])
            oof_pred_temp[te_mask] = (oof_prob_temp[te_mask] >= th_hist_oof).astype(int)

    # 将时间序列按月度/双周分为 5 个时间段用于计算时间分段 Mean ± SD
    temp_period_ids = pd.qcut(dates_loc.rank(method="first"), q=5, labels=False).values

    def build_full_row(name, y_t, p_prob, p_pred, boot_clusters, split_ids):
        f_auc_msd, f_ap_msd = compute_fold_mean_sd(y_t, p_prob, split_ids)
        return {
            "验证方案": name,
            "Pooled_ROC_AUC (Cluster 95% CI)": cluster_bootstrap_ci(y_t, p_prob, boot_clusters, roc_auc_score),
            "Fold_ROC_AUC (Mean±SD)": f_auc_msd,
            "Pooled_AP (Cluster 95% CI)": cluster_bootstrap_ci(y_t, p_prob, boot_clusters, average_precision_score),
            "Fold_AP (Mean±SD)": f_ap_msd,
            "报警率": f"{p_pred.mean()*100:.1f}%",
            "F1": round(f1_score(y_t, p_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_t, p_pred, zero_division=0), 4),
            "Precision": round(precision_score(y_t, p_pred, zero_division=0), 4),
        }

    main_table = pd.DataFrame([
        build_full_row("1. 未对齐直接迁移 Baseline (默认0.5阈值)", y_loc, prob_1_base, pred_1_base, players_loc, fold_ids_grp),
        build_full_row("2a. Zero-Shot 纯生理+穿戴+主观 (无伤病史/源域嵌套定阈值)", y_loc, prob_2a, pred_2a, players_loc, fold_ids_grp),
        build_full_row("2b. Zero-Shot 全维度源域迁移 (双层嵌套Stacking+源域定阈值)", y_loc, prob_2b, pred_2b, players_loc, fold_ids_grp),
        build_full_row("3a. 跨运动员 GroupKFold (无伤病史+内层OOF定阈值)", y_loc, oof_prob_grp_nh, oof_pred_grp_nh, players_loc, fold_ids_grp),
        build_full_row("3b. 跨运动员 GroupKFold (全维度5支柱+内层OOF定阈值)", y_loc, oof_prob_grp_full, oof_pred_grp_full, players_loc, fold_ids_grp),
        build_full_row("4. 时间前滚 Purged Temporal (历史OOF定阈值+时间块CI)", y_loc, oof_prob_temp, oof_pred_temp, date_blocks, temp_period_ids),
    ])

    print("\n[表 1] 主验证性能对比表 (Pooled 总体指标 + 聚类95%CI + 外层各折 Mean±SD):")
    print(main_table.to_string(index=False))

    sens_table = evaluate_alert_budget_sensitivity(y_loc, oof_prob_grp_full, oof_prob_temp, fold_ids_grp, dates_loc, horizon_days)
    print(f"\n[表 2] 预警预算敏感性分析表 (不同每日预警比例 Top 10%~30% 下的折内阈值性能):")
    print(sens_table.to_string(index=False))

    fold_df = pd.DataFrame(fold_breakdown)
    print(f"\n[表 3] 外层 6-Fold GroupKFold 各折独立明细表 (展示 24 名运动员分折波动情况):")
    print(fold_df.to_string(index=False))

    return main_table, sens_table, fold_df


if __name__ == "__main__":
    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv", parse_dates=["date"])
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv", parse_dates=["date"])

    df_sm = add_causal_ewma_and_context_features(df_sm, is_local=False)
    df_loc = add_causal_ewma_and_context_features(df_loc, is_local=True)

    m7, s7, f7 = run_step10_suite(
        df_sm, df_loc, "target_non_contact_next_7d", horizon_days=7, title="未来 7 天非接触性损伤预警 (阳性率 18.30%)"
    )
    m3, s3, f3 = run_step10_suite(
        df_sm, df_loc, "target_non_contact_next_3d", horizon_days=3, title="未来 3 天非接触性损伤预警 (阳性率 10.27%)"
    )

    pd.concat([m7.assign(窗口="7天"), m3.assign(窗口="3天")], ignore_index=True).to_csv(
        OUTPUTS_DIR / "table1_main_validation_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([s7.assign(窗口="7天"), s3.assign(窗口="3天")], ignore_index=True).to_csv(
        OUTPUTS_DIR / "table2_alert_budget_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([f7.assign(窗口="7天"), f3.assign(窗口="3天")], ignore_index=True).to_csv(
        OUTPUTS_DIR / "table3_outer_folds_breakdown.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n✅ 三张论文标准表格已全部保存至 {OUTPUTS_DIR} 目录！")