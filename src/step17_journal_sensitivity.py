# src/step17_journal_sensitivity.py
"""
Step 17 (英文期刊投稿补充敏感性分析):
完全基于已锁定的 data/processed/* 面板，不修改任何现有主结果，补充两项审稿人核心检验：
1. 分析 S1 [纯伤病史简单基线对照 (Injury-History-Only Baseline)]:
   在同一批本地 1,315 风险人日上，分别评估：
   - (A) 源域冻结【纯伤病史 LR 基线】(仅使用 days_since_last_inj, prev_inj_7d, prev_inj_14d, prev_inj_30d)
   - (B) 本地 5 折 StratifiedGroupKFold【纯伤病史 LR 基线】
   并与多源模型 (Step 05 冻结全维 LR、Step 11 2b 生理单调迁移、Step 11 3c 本地 5 折更新) 对比，
   采用运动员聚类 Bootstrap (B=1000) 计算 95% CI 与配对差值，量化多源监测数据的真实增量价值。
2. 分析 S2 [连续上报 >3 天合并规则敏感性分析 (Aligned >3-Day Cluster Onset Rule)]:
   将本地 24 人的 61 个自报告伤病日按 SoccerMon 同款规则（同队员间隔 <=3 天合并为同一伤病簇，
   仅间隔 >3 天计为新发）重新构造严格未来 (t, t+7] 结局，重估 7 天表现以检验对上报合并规则的稳健性。
"""
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from step09_final_clean_pipeline import add_causal_ewma_and_context_features
from step11_locked_paper_suite import (
    anonymize_player_ids,
    fit_nested_crossfitted_source,
    CalibratedNestedAdapter,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROTOCOL_PATH = PROJECT_ROOT / "config" / "study_protocol.json"


def cluster_boot_metrics(y_true, y_prob, groups, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_map = {g: np.where(groups == g)[0] for g in uniq}
    aucs, aps = [], []
    for _ in range(n_boot):
        sg = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_map[g] for g in sg])
        yb, pb = y_true[idx], y_prob[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        aucs.append(roc_auc_score(yb, pb))
        aps.append(average_precision_score(yb, pb))
    return (
        roc_auc_score(y_true, y_prob),
        np.percentile(aucs, 2.5),
        np.percentile(aucs, 97.5),
        average_precision_score(y_true, y_prob),
        np.percentile(aps, 2.5),
        np.percentile(aps, 97.5),
        brier_score_loss(y_true, y_prob),
    )


def paired_delta_ci(y_true, p_model, p_ref, groups, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_map = {g: np.where(groups == g)[0] for g in uniq}
    diffs = []
    for _ in range(n_boot):
        sg = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_map[g] for g in sg])
        yb = y_true[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        diffs.append(metric_fn(yb, p_model[idx]) - metric_fn(yb, p_ref[idx]))
    pt = metric_fn(y_true, p_model) - metric_fn(y_true, p_ref)
    return f"{pt:+.4f} ({np.percentile(diffs, 2.5):+.3f} to {np.percentile(diffs, 97.5):+.3f})"


def apply_3day_cluster_rule(onset_series: pd.Series) -> pd.Series:
    """按 SoccerMon 同款规则：同一队员间隔 <= 3 天的连续上报合并为同一次事件，仅间隔 > 3 天计为新发"""
    dates = onset_series.index[onset_series == 1]
    kept_dates = set()
    last_dt = None
    for dt in dates:
        if last_dt is None or (dt - last_dt).days > 3:
            kept_dates.add(dt)
        last_dt = dt
    return pd.Series(onset_series.index.isin(kept_dates).astype(int), index=onset_series.index)


def strict_future_7d(onset_series: pd.Series, followup_end: pd.Timestamp) -> pd.Series:
    ext_dates = pd.date_range(onset_series.index.min(), max(onset_series.index.max(), followup_end), freq="D")
    ext_s = onset_series.reindex(ext_dates).fillna(0).astype(int)
    s_next = ext_s.shift(-1).fillna(0).astype(int)
    rev_max = s_next.iloc[::-1].rolling(window=7, min_periods=1).max().iloc[::-1]
    return rev_max.reindex(onset_series.index).astype(int)


def run_all_models_on_target(df_sm_risk, df_loc_risk, target_col, full_38_feats, fixed_splits=None):
    hist_feats = [c for c in ["days_since_last_inj", "prev_inj_7d", "prev_inj_14d", "prev_inj_30d", "recent_inj_decay"] if c in df_sm_risk.columns and c in df_loc_risk.columns]

    y_sm = df_sm_risk["target_all_inj_next_7d"].astype(int).values
    y_loc = df_loc_risk[target_col].astype(int).values
    g_loc = df_loc_risk["anon_id"].values

    # 1. 源域冻结【纯伤病史 LR 基线】
    imp_h = SimpleImputer(strategy="median").fit(df_sm_risk[hist_feats])
    sc_h = StandardScaler().fit(imp_h.transform(df_sm_risk[hist_feats]))
    lr_h_src = LogisticRegression(C=0.01, penalty="l2", solver="saga", class_weight="balanced", max_iter=2000, random_state=42)
    lr_h_src.fit(sc_h.transform(imp_h.transform(df_sm_risk[hist_feats])), y_sm)
    prob_hist_frozen = lr_h_src.predict_proba(sc_h.transform(imp_h.transform(df_loc_risk[hist_feats])))[:, 1]

    # 2. 本地 5 折 StratifiedGroupKFold【纯伤病史 LR 基线】(使用 5 个历史变量的标准 L2 逻辑回归，无额外 Platt 校准)
    splits = fixed_splits if fixed_splits is not None else list(StratifiedGroupKFold(n_splits=5).split(df_loc_risk, y_loc, groups=g_loc))
    prob_hist_local5 = np.zeros(len(y_loc))
    for tr_i, te_i in splits:
        imp_l = SimpleImputer(strategy="median").fit(df_loc_risk.iloc[tr_i][hist_feats])
        sc_l = StandardScaler().fit(imp_l.transform(df_loc_risk.iloc[tr_i][hist_feats]))
        m_l = LogisticRegression(C=0.05, penalty="l2", max_iter=1000, random_state=42)
        m_l.fit(sc_l.transform(imp_l.transform(df_loc_risk.iloc[tr_i][hist_feats])), y_loc[tr_i])
        prob_hist_local5[te_i] = m_l.predict_proba(sc_l.transform(imp_l.transform(df_loc_risk.iloc[te_i][hist_feats])))[:, 1]

    # 3. 源域冻结【全维度 38 特征 LR】(Step 05 5a 直接映射)
    imp_f = SimpleImputer(strategy="median").fit(df_sm_risk[full_38_feats])
    sc_f = StandardScaler().fit(imp_f.transform(df_sm_risk[full_38_feats]))
    lr_full = LogisticRegression(C=0.01, penalty="l2", solver="saga", class_weight="balanced", max_iter=2000, random_state=42)
    lr_full.fit(sc_f.transform(imp_f.transform(df_sm_risk[full_38_feats])), y_sm)
    prob_full_frozen = lr_full.predict_proba(sc_f.transform(imp_f.transform(df_loc_risk[full_38_feats])))[:, 1]

    # 4. 生理单调源域迁移 (2b) 与本地 StratifiedGroup-5折更新 (3c)
    loc_tmp, _, _, prob_2b, _, _ = fit_nested_crossfitted_source(df_sm_risk, df_loc_risk, "target_all_inj_next_7d")
    prob_3c = np.zeros(len(y_loc))
    budgets = (0.10, 0.15, 0.20, 0.25, 0.30)
    for tr_i, te_i in splits:
        ad = CalibratedNestedAdapter(include_history=True).fit_nested(loc_tmp.iloc[tr_i], y_loc[tr_i], budgets)
        p_cal, _, _ = ad.predict_calibrated(loc_tmp.iloc[te_i])
        prob_3c[te_i] = p_cal

    models = [
        ("1. Frozen History-Only LR (Source Baseline)", prob_hist_frozen),
        ("2. Local 5-Fold History-Only LR (Local Baseline)", prob_hist_local5),
        ("3. Frozen Full 38-Feature LR (Stage 2 Transfer)", prob_full_frozen),
        ("4. Monotonic Cross-Fitted Transfer (2b)", prob_2b),
        ("5. Target-Cohort Updated Model (3c, StratifiedGroup-5F)", prob_3c),
    ]

    rows = []
    for name, p in models:
        auc, auc_lo, auc_hi, ap, ap_lo, ap_hi, brier = cluster_boot_metrics(y_loc, p, g_loc)
        d_auc_f = "Ref" if name.startswith("1.") else paired_delta_ci(y_loc, p, prob_hist_frozen, g_loc, roc_auc_score)
        d_ap_f = "Ref" if name.startswith("1.") else paired_delta_ci(y_loc, p, prob_hist_frozen, g_loc, average_precision_score)
        d_auc_l = "Ref" if name.startswith("2.") else (paired_delta_ci(y_loc, p, prob_hist_local5, g_loc, roc_auc_score) if name.startswith("5.") else "—")
        d_ap_l = "Ref" if name.startswith("2.") else (paired_delta_ci(y_loc, p, prob_hist_local5, g_loc, average_precision_score) if name.startswith("5.") else "—")
        rows.append({
            "Model": name,
            "ROC_AUC (95% CI)": f"{auc:.4f} ({auc_lo:.3f}-{auc_hi:.3f})",
            "AP (95% CI)": f"{ap:.4f} ({ap_lo:.3f}-{ap_hi:.3f})",
            "Brier": round(brier, 4),
            "Delta_AUC vs Frozen History-Only LR": d_auc_f,
            "Delta_AP vs Frozen History-Only LR": d_ap_f,
            "Delta_AUC vs Local History-Only LR": d_auc_l,
            "Delta_AP vs Local History-Only LR": d_ap_l,
        })
    return pd.DataFrame(rows), splits


def main():
    proto = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    local_followup_end = pd.Timestamp(proto["local_cohort_protocol"]["injury_surveillance_end"])

    with open(OUTPUTS_DIR / "step14_best_source_model_params.json", "r", encoding="utf-8") as f:
        spec = json.load(f)
    full_38_feats = spec["feature_audit"]["lasso_selected_features"]

    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv", parse_dates=["date"])
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv", parse_dates=["date"])
    df_sm = add_causal_ewma_and_context_features(df_sm, is_local=False)
    df_loc = add_causal_ewma_and_context_features(df_loc, is_local=True)

    # 先在本地完整连续日历上构造【>3 天合并簇规则】的新发标签与严格未来 (t, t+7] 标签
    df_loc = df_loc.sort_values(["player_id", "date"]).reset_index(drop=True)
    cluster_onsets, cluster_t7 = [], []
    for pid, grp in df_loc.groupby("player_id", sort=False):
        s_onset = pd.Series(grp["inj_all_onset"].fillna(0).astype(int).values, index=grp["date"])
        c_onset = apply_3day_cluster_rule(s_onset)
        c_t7 = strict_future_7d(c_onset, local_followup_end)
        cluster_onsets.append(c_onset.values)
        cluster_t7.append(c_t7.values)
    df_loc["inj_cluster3d_onset"] = np.concatenate(cluster_onsets)
    df_loc["target_cluster3d_next_7d"] = np.concatenate(cluster_t7)

    # 严格统一风险集过滤
    sm_risk = df_sm[
        (df_sm["active_day"] == 1) & (df_sm["is_rehab"] == 0) & (df_sm["is_injured"] == 0)
        & (df_sm["target_all_inj_next_7d"].notna())
    ].copy().reset_index(drop=True)
    sm_risk["target_all_inj_next_7d"] = sm_risk["target_all_inj_next_7d"].astype(int)

    loc_risk = df_loc[
        (df_loc["active_day"] == 1) & (df_loc["is_rehab"] == 0) & (df_loc["is_injured"] == 0)
    ].copy().sort_values(["date", "player_id"]).reset_index(drop=True)
    loc_risk = anonymize_player_ids(loc_risk)

    print("=" * 115)
    print("📊 [补充分析 S1] 纯伤病史简单基线 vs 多源监测模型增量价值检验 (本地 1,315 风险人日, 7天阳性=225 / 17.11%)")
    print("=" * 115)
    tab_s1, s1_splits = run_all_models_on_target(sm_risk, loc_risk, "target_all_inj_next_7d", full_38_feats)
    print(tab_s1.to_string(index=False))
    s1_path = OUTPUTS_DIR / "step17_table_s1_history_only_baseline.csv"
    tab_s1.to_csv(s1_path, index=False, encoding="utf-8-sig")

    n_raw_events = int(df_loc["inj_all_onset"].sum())
    n_clu_events = int(df_loc["inj_cluster3d_onset"].sum())
    n_clu_pos7 = int(loc_risk["target_cluster3d_next_7d"].sum())
    print("\n" + "=" * 115)
    print(f"📊 [补充分析 S2] 连续上报 >3 天合并规则敏感性分析 (与 SoccerMon 完全同款簇合并规则)")
    print(f"   本地上报日: 61 -> 按间隔 >3 天合并后46 report clusters (上报簇: {n_clu_events} 个) | 1,315 风险人日内 7 天阳性 = {n_clu_pos7} ({n_clu_pos7/len(loc_risk)*100:.2f}%)")
    print("=" * 115)
    tab_s2, _ = run_all_models_on_target(sm_risk, loc_risk, "target_cluster3d_next_7d", full_38_feats, fixed_splits=s1_splits)
    print(tab_s2.to_string(index=False))
    s2_path = OUTPUTS_DIR / "step17_table_s2_cluster3d_sensitivity.csv"
    tab_s2.to_csv(s2_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 115)
    print(f"✅ 期刊补充敏感性分析完成！已保存至:\n   - {s1_path}\n   - {s2_path}")
    print("=" * 115)


if __name__ == "__main__":
    main()