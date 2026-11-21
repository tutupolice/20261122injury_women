# src/step12_final_figures.py
"""
女足运动损伤风险跨数据集迁移与目标队伍更新 —— 论文定稿级高清配图生成脚本 (Step 12 审查整改版)
严格基于 Step 11 锁定管线输出 4 张 300-DPI 学术出版级图表 (全匿名化 P01~P24，无任何泄露)：
1. fig1_roc_pr_validation.png          : 7天 & 3天预警 ROC 与 Precision-Recall 曲线对比
2. fig2_calibration_and_quintiles.png  : 概率校准可靠性曲线 (含 Brier Score) 与五分位风险分层
3. fig3_alert_budget_and_folds.png     : Top 10%~30% 预警预算权衡曲线与 StratifiedGroup-5折稳定性
4. fig4_anonymized_trajectories.png    : 匿名化典型运动员 3 个月动态风险轨迹与真实损伤事件标记
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures_final"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from step09_final_clean_pipeline import (
    add_causal_ewma_and_context_features,
    FEAT_EXPERT_OVERLOAD,
    FEAT_EXPERT_WELLNESS,
    FEAT_EXPERT_BIOMECH,
)
from step11_locked_paper_suite import (
    anonymize_player_ids,
    fit_nested_crossfitted_source,
    CalibratedNestedAdapter,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#dddddd",
})


def compute_step11_predictions(df_sm: pd.DataFrame, df_loc: pd.DataFrame, target_col: str, horizon_days: int):
    """复现 Step 11 审查整改版的严格零泄露预测概率，供绘图使用"""
    df_sm = df_sm.copy()
    sm_alt = f"target_all_inj_next_{horizon_days}d"
    if target_col not in df_sm.columns and sm_alt in df_sm.columns:
        df_sm[target_col] = df_sm[sm_alt]

    sm_act_col = "active_day" if "active_day" in df_sm.columns else "is_active_day"
    sm_active = df_sm[(df_sm[sm_act_col] == 1) & (df_sm["is_rehab"] == 0) & (df_sm["is_injured"] == 0)].copy()
    sm_onset_col = "inj_all_onset" if "inj_all_onset" in sm_active.columns else "inj_all_onset"
    if "inj_prevalence" in sm_active.columns and sm_onset_col in sm_active.columns:
        sm_clean = sm_active[~((sm_active["inj_prevalence"] == 1) & (sm_active[sm_onset_col] == 0))].reset_index(drop=True)
    else:
        sm_clean = sm_active.reset_index(drop=True)

    loc_act_col = "active_day" if "active_day" in df_loc.columns else "is_active_day"
    loc_active = df_loc[(df_loc[loc_act_col] == 1) & (df_loc["is_rehab"] == 0) & (df_loc["is_injured"] == 0)].copy().sort_values(["date", "player_id"]).reset_index(drop=True)
    loc_active = anonymize_player_ids(loc_active)
    y_loc = loc_active[target_col].values
    players_loc = loc_active["anon_id"].values
    dates_loc = loc_active["date"]
    budgets = (0.10, 0.15, 0.20, 0.25, 0.30)

    # 1. Raw Baseline
    base_cols = FEAT_EXPERT_OVERLOAD + FEAT_EXPERT_WELLNESS + FEAT_EXPERT_BIOMECH
    rf_base = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_base.fit(sm_active[base_cols].fillna(0).values, sm_active[target_col].values)
    prob_base = rf_base.predict_proba(loc_active[base_cols].fillna(0).values)[:, 1]

    # 2. Nested Cross-Fitted Zero-Shot (2b)
    loc_active, _, _, prob_zs, _ = fit_nested_crossfitted_source(sm_clean, loc_active, target_col)

    # 3. StratifiedGroupKFold-5 (3c)
    sgkf5 = StratifiedGroupKFold(n_splits=5)
    oof_sgkf = np.zeros(len(loc_active))
    fold_aucs, fold_aps, fold_labels = [], [], []
    for f_idx, (tr_idx, te_idx) in enumerate(sgkf5.split(loc_active, y_loc, groups=players_loc), 1):
        ad = CalibratedNestedAdapter(include_history=True).fit_nested(loc_active.iloc[tr_idx], y_loc[tr_idx], budgets)
        p_cal, _, _ = ad.predict_calibrated(loc_active.iloc[te_idx])
        oof_sgkf[te_idx] = p_cal
        fold_aucs.append(roc_auc_score(y_loc[te_idx], p_cal))
        fold_aps.append(average_precision_score(y_loc[te_idx], p_cal))
        fold_labels.append(f"Fold {f_idx}\n(n={len(te_idx)})")

    # 4. Purged Temporal CV (4)
    unique_dates = np.sort(dates_loc.unique())
    oof_temp = np.copy(prob_zs)
    warmup_cut = int(len(unique_dates) * 0.25)
    for d_idx, curr_date in enumerate(unique_dates):
        te_mask = (dates_loc == curr_date).values
        purge_cutoff = pd.Timestamp(curr_date) - pd.Timedelta(days=horizon_days)
        tr_mask = (dates_loc <= purge_cutoff).values
        n_hist = int(tr_mask.sum())
        if d_idx >= warmup_cut and n_hist >= 120 and y_loc[tr_mask].sum() >= 8:
            ad_t = CalibratedNestedAdapter(include_history=True).fit_nested(loc_active.iloc[tr_mask], y_loc[tr_mask], budgets)
            p_loc_te, _, _ = ad_t.predict_calibrated(loc_active.iloc[te_mask])
            alpha = float(np.clip((n_hist - 100) / 1000.0, 0.15, 0.50))
            oof_temp[te_mask] = (1.0 - alpha) * prob_zs[te_mask] + alpha * p_loc_te

    return {
        "df": loc_active,
        "y": y_loc,
        "prob_base": prob_base,
        "prob_zs": prob_zs,
        "prob_sgkf": oof_sgkf,
        "prob_temp": oof_temp,
        "fold_aucs": fold_aucs,
        "fold_aps": fold_aps,
        "fold_labels": fold_labels,
    }


def plot_fig1_roc_pr(res7: dict, res3: dict):
    """图 1: 7天 & 3天预警 ROC 与 Precision-Recall 对比曲线 (2x2)"""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0), dpi=300)
    configs = [
        ("Raw Baseline Transfer", "prob_base", "#8c8c8c", ":"),
        ("Zero-Shot Cross-Fitted (SoccerMon)", "prob_zs", "#2b83ba", "--"),
        ("StratifiedGroup 5-Fold (Unseen Players)", "prob_sgkf", "#d7191c", "-"),
        ("Purged Temporal CV (Prospective)", "prob_temp", "#e66101", "-."),
    ]

    p7_pct = res7["y"].mean() * 100
    p3_pct = res3["y"].mean() * 100

    for row_idx, (res, title_prefix) in enumerate([
        (res7, f"7-Day Horizon (Prevalence = {p7_pct:.1f}%)"),
        (res3, f"3-Day Horizon (Prevalence = {p3_pct:.1f}%)"),
    ]):
        y = res["y"]
        ax_roc, ax_pr = axes[row_idx, 0], axes[row_idx, 1]

        for label, key, col, ls in configs:
            p = res[key]
            fpr, tpr, _ = roc_curve(y, p)
            auc_v = roc_auc_score(y, p)
            ax_roc.plot(fpr, tpr, label=f"{label} (AUC={auc_v:.3f})", color=col, linestyle=ls, lw=2.1)

            prec, rec, _ = precision_recall_curve(y, p)
            ap_v = average_precision_score(y, p)
            ax_pr.plot(rec, prec, label=f"{label} (AP={ap_v:.3f})", color=col, linestyle=ls, lw=2.1)

        ax_roc.plot([0, 1], [0, 1], "k:", alpha=0.4, lw=1.0)
        ax_roc.set_title(f"{'A' if row_idx==0 else 'C'}. ROC Curve — {title_prefix}", fontsize=11, fontweight="bold")
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
        ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
        ax_roc.legend(loc="lower right", fontsize=8.2)
        ax_roc.grid(alpha=0.22)

        ax_pr.axhline(y.mean(), color="#444444", linestyle=":", alpha=0.55, label=f"Chance Baseline ({y.mean():.3f})")
        ax_pr.set_title(f"{'B' if row_idx==0 else 'D'}. Precision-Recall Curve — {title_prefix}", fontsize=11, fontweight="bold")
        ax_pr.set_xlabel("Recall (Sensitivity)")
        ax_pr.set_ylabel("Precision (Positive Predictive Value)")
        ax_pr.set_ylim(0, 0.75 if row_idx == 0 else 0.55)
        ax_pr.legend(loc="upper right", fontsize=8.2)
        ax_pr.grid(alpha=0.22)

    plt.tight_layout(pad=1.6)
    plt.savefig(FIG_DIR / "fig1_roc_pr_validation.png")
    plt.close()


def plot_fig2_calibration_and_quintiles(res7: dict):
    """图 2: 概率校准可靠性曲线 (含 Brier Score) + 五分位真实损伤率分层"""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=300)
    y = res7["y"]

    cal_models = [
        (f"Zero-Shot (Brier={brier_score_loss(y, res7['prob_zs']):.4f})", res7["prob_zs"], "#2b83ba", "s--"),
        (f"StratifiedGroup-5F (Brier={brier_score_loss(y, res7['prob_sgkf']):.4f})", res7["prob_sgkf"], "#d7191c", "o-"),
        (f"Purged Temporal (Brier={brier_score_loss(y, res7['prob_temp']):.4f})", res7["prob_temp"], "#e66101", "^-."),
    ]
    for label, p, col, fmt in cal_models:
        prob_true, prob_pred = calibration_curve(y, p, n_bins=6, strategy="quantile")
        axes[0].plot(prob_pred, prob_true, fmt, label=label, color=col, lw=2.0, markersize=6)

    axes[0].plot([0, 0.6], [0, 0.6], "k:", alpha=0.5, label="Perfect Calibration")
    axes[0].set_title("A. Probability Calibration Reliability (7-Day Risk)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Mean Predicted Probability (Platt-Calibrated)")
    axes[0].set_ylabel("Observed 7-Day Injury Frequency")
    axes[0].set_xlim(0, 0.55)
    axes[0].set_ylim(0, 0.55)
    axes[0].legend(loc="upper left", fontsize=8.8)
    axes[0].grid(alpha=0.25)

    q_labels = ["Q1\n(Lowest)", "Q2\n(Low)", "Q3\n(Moderate)", "Q4\n(High)", "Q5\n(Very High)"]
    q_bins = pd.qcut(res7["prob_sgkf"], q=5, labels=q_labels)
    q_rates = pd.Series(y).groupby(q_bins, observed=False).mean() * 100
    colors_q = ["#2b83ba", "#abdda4", "#fee08b", "#fdae61", "#d7191c"]

    bars = axes[1].bar(range(5), q_rates.values, color=colors_q, edgecolor="#333333", width=0.62)
    axes[1].axhline(y.mean() * 100, color="#555555", linestyle="--", lw=1.2, label=f"Cohort Base Rate ({y.mean()*100:.1f}%)")
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels(q_labels)
    axes[1].set_ylabel("Observed 7-Day Injury Incidence (%)")
    axes[1].set_title("B. Injury Incidence by Predicted Risk Quintile (Out-of-Fold)", fontsize=11.5, fontweight="bold")
    for b in bars:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", fontweight="bold", fontsize=9.5)
    axes[1].set_ylim(0, max(q_rates.values) * 1.22)
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout(pad=1.5)
    plt.savefig(FIG_DIR / "fig2_calibration_and_quintiles.png")
    plt.close()


def plot_fig3_budget_and_folds(res7: dict):
    """图 3: 表 2 预警预算权衡曲线 + StratifiedGroup-5折各折稳定性"""
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=300)

    t2 = pd.read_csv(OUTPUTS_DIR / "step11_table2_budget_sensitivity.csv")
    t2_7d = t2[t2["窗口"] == "7天"].copy()
    x_pos = np.arange(len(t2_7d))

    axes[0].plot(x_pos, t2_7d["Temporal_Recall"], "o-", color="#d7191c", lw=2.2, label="Temporal Recall (Sensitivity)")
    axes[0].plot(x_pos, t2_7d["Temporal_Precision"], "s--", color="#2b83ba", lw=2.2, label="Temporal Precision (PPV)")
    axes[0].plot(x_pos, t2_7d["Group6折_F1"], "^-.", color="#1a9641", lw=2.0, label="GroupKFold F1-Score")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(t2_7d["目标预警预算"])
    axes[0].set_ylabel("Score (0 - 1)")
    axes[0].set_xlabel("Target Daily Alert Budget (In-Fold Quantile Threshold)")
    axes[0].set_title("A. Alert Budget Sensitivity Trade-off (7-Day Risk)", fontsize=11.5, fontweight="bold")
    axes[0].set_ylim(0, 0.85)
    axes[0].legend(loc="upper left", fontsize=8.8)
    axes[0].grid(alpha=0.25)

    x_f = np.arange(len(res7["fold_aucs"]))
    w = 0.36
    b1 = axes[1].bar(x_f - w/2, res7["fold_aucs"], width=w, color="#2b83ba", edgecolor="#333333", label="Fold ROC-AUC")
    b2 = axes[1].bar(x_f + w/2, res7["fold_aps"], width=w, color="#fdae61", edgecolor="#333333", label="Fold AP")
    axes[1].axhline(np.mean(res7["fold_aucs"]), color="#2b83ba", linestyle="--", lw=1.3,
                    label=f"Mean AUC = {np.mean(res7['fold_aucs']):.3f}")
    axes[1].axhline(0.50, color="#888888", linestyle=":", lw=1.1, label="Random Chance (AUC=0.50)")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.2f}", ha="center", fontsize=8.2)

    axes[1].set_xticks(x_f)
    axes[1].set_xticklabels(res7["fold_labels"], fontsize=8.8)
    axes[1].set_ylabel("Metric Value")
    axes[1].set_ylim(0, 0.95)
    axes[1].set_title("B. Fold-by-Fold Stability (StratifiedGroup 5-Fold)", fontsize=11.5, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8.2)
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout(pad=1.5)
    plt.savefig(FIG_DIR / "fig3_alert_budget_and_folds.png")
    plt.close()


def plot_fig4_anonymized_trajectories(res7: dict):
    """图 4: 匿名化高危/低危典型运动员 3 个月动态风险轨迹"""
    df = res7["df"].copy()
    df["pred_sgkf"] = res7["prob_sgkf"]
    df["pred_temp"] = res7["prob_temp"]

    inj_counts = df.groupby("anon_id")["inj_all_onset"].sum()
    high_pid = inj_counts.idxmax()
    zero_pids = inj_counts[inj_counts == 0].index.tolist()
    low_pid = zero_pids[0] if zero_pids else inj_counts.idxmin()

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.2), dpi=300, sharex=True)

    for ax, pid, subtitle in [
        (axes[0], high_pid, f"High-Risk Case Athlete ({high_pid}: Multiple Self-Reported Injury Onsets)"),
        (axes[1], low_pid, f"Low-Risk Control Athlete ({low_pid}: Injury-Free Throughout 3 Months)"),
    ]:
        sub = df[df["anon_id"] == pid].sort_values("date")
        ax.plot(sub["date"], sub["pred_sgkf"], color="#2b83ba", lw=1.9, linestyle="--", label="Cross-Athlete OOF Risk (StratifiedGroup)")
        ax.plot(sub["date"], sub["pred_temp"], color="#d7191c", lw=2.2, label="Prospective Temporal Risk (Purged)")
        ax.axhline(0.25, color="#fdae61", linestyle=":", lw=1.4, label="Top-20% Alert Reference (0.25)")

        inj_rows = sub[sub["inj_all_onset"] == 1]
        for idx_i, (_, r) in enumerate(inj_rows.iterrows()):
            lbl = "Actual Self-Reported Injury Onset" if idx_i == 0 else None
            ax.axvline(r["date"], color="#d7191c", alpha=0.25, lw=6, label=lbl)
            ax.scatter([r["date"]], [r["pred_temp"]], color="#d7191c", s=60, zorder=5)

        ax.set_title(subtitle, fontsize=11.2, fontweight="bold")
        ax.set_ylabel("Calibrated 7-Day Injury Risk")
        ax.set_ylim(0, 0.72)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(alpha=0.25)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[1].set_xlabel("Monitoring Date")
    plt.tight_layout(pad=1.5)
    plt.savefig(FIG_DIR / "fig4_anonymized_trajectories.png")
    plt.close()


if __name__ == "__main__":
    print("🎨 正在基于 Step 11 锁定管线生成 4 张论文高清配图...")
    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv", parse_dates=["date"])
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv", parse_dates=["date"])

    df_sm = add_causal_ewma_and_context_features(df_sm, is_local=False)
    df_loc = add_causal_ewma_and_context_features(df_loc, is_local=True)

    res7 = compute_step11_predictions(df_sm, df_loc, "target_all_inj_next_7d", horizon_days=7)
    res3 = compute_step11_predictions(df_sm, df_loc, "target_all_inj_next_3d", horizon_days=3)

    plot_fig1_roc_pr(res7, res3)
    plot_fig2_calibration_and_quintiles(res7)
    plot_fig3_budget_and_folds(res7)
    plot_fig4_anonymized_trajectories(res7)

    print(f"✅ 全部 4 张 300-DPI 论文配图已生成并保存至: {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"   - {f.name}")