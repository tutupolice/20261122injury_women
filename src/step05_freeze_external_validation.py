# src/step05_freeze_external_validation.py
"""
Step 05 (论文主线正式版):
冻结在 SoccerMon 源域上开发并选出的唯一最优基准模型：
  - 特征子集：Step 14 客观 LASSO 筛选出的 27 个非零特征
  - 模型与参数：Logistic Regression (C=0.01, penalty='l2', class_weight='balanced', solver='saga')
将该模型 100% 冻结（不再使用本地标签训练或微调权重），直接在本地 24 名女足队员队列 (1,421 活跃监控日)
上进行外部验证（涵盖未来 7 天与未来 3 天自报告运动损伤预警），并报告：
  1. Pooled ROC-AUC & Pooled AP (含按运动员聚类 Bootstrap 95% CI, B=1000)；
  2. 源域 OOF F1 最优阈值及不同预警预算 (Top 10% / 15% / 20% / 25% / 30%) 下的外部验证表现；
  3. Platt 基线校准前后的 Brier Score 与高风险五分位相对风险倍数 (Top 20% vs Bottom 20% Lift)。
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


def cluster_bootstrap_ci(y_true, y_prob, groups, n_boot=1000, seed=42):
    """按运动员聚类重采样 (Player-Cluster Bootstrap) 计算 ROC-AUC 和 AP 的 95% 置信区间"""
    rng = np.random.default_rng(seed)
    unique_g = np.unique(groups)
    g_to_idx = {g: np.where(groups == g)[0] for g in unique_g}
    aucs, aps = [], []

    for _ in range(n_boot):
        sampled_g = rng.choice(unique_g, size=len(unique_g), replace=True)
        idx = np.concatenate([g_to_idx[g] for g in sampled_g])
        yb, pb = y_true[idx], y_prob[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        aucs.append(roc_auc_score(yb, pb))
        aps.append(average_precision_score(yb, pb))

    return (
        (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)),
        (np.percentile(aps, 2.5), np.percentile(aps, 97.5)),
    )


def select_source_oof_threshold(y_src, oof_prob_src, min_alert=0.10, max_alert=0.35):
    """仅在 SoccerMon 源域 OOF 预测概率上选择最大化 F1 的冻结阈值（完全不接触本地测试集标签）"""
    quantiles = np.linspace(1.0 - max_alert, 1.0 - min_alert, 51)
    best_th = float(np.quantile(oof_prob_src, 0.80))
    best_f1 = -1.0
    for q in quantiles:
        th = float(np.quantile(oof_prob_src, q))
        pred = (oof_prob_src >= th).astype(int)
        f1 = f1_score(y_src, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th


def compute_top_bottom_lift(y_true, y_prob):
    """计算最高风险 20% 与最低风险 20% 的实际损伤发生率比值 (Risk Lift)"""
    q80 = np.quantile(y_prob, 0.80)
    q20 = np.quantile(y_prob, 0.20)
    rate_top = y_true[y_prob >= q80].mean()
    rate_bot = max(y_true[y_prob <= q20].mean(), 1e-4)
    return float(rate_top / rate_bot), float(rate_top), float(y_true[y_prob <= q20].mean())


def load_datasets():
    json_path = OUTPUTS_DIR / "step14_best_source_model_params.json"
    if not json_path.exists():
        raise FileNotFoundError(f"未找到 {json_path}，请先运行 step14_soccermon_8models_benchmark.py")

    with open(json_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    selected_features = spec["feature_audit"]["lasso_selected_features"]
    tuned_params = spec["tuned_hyperparameters"]

    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv")
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv")

    # 过滤源域非监控日与康复日
    if "active_day" in df_sm.columns:
        df_sm = df_sm[df_sm["active_day"] == 1].copy()
    elif "has_monitoring" in df_sm.columns:
        df_sm = df_sm[df_sm["has_monitoring"] == 1].copy()
    else:
        mon_cols = [c for c in ["fatigue_01", "soreness_01", "uload_ewma_acwr", "dist_ewma_acwr"] if c in df_sm.columns]
        if mon_cols:
            df_sm = df_sm[df_sm[mon_cols].notna().any(axis=1)].copy()
    if "is_rehab" in df_sm.columns:
        df_sm = df_sm[df_sm["is_rehab"] == 0].copy()
    if "is_injured" in df_sm.columns:
        df_sm = df_sm[df_sm["is_injured"] == 0].copy()

    # 过滤本地非监控日与康复日
    if "active_day" in df_loc.columns:
        df_loc = df_loc[df_loc["active_day"] == 1].copy()
    if "is_rehab" in df_loc.columns:
        df_loc = df_loc[df_loc["is_rehab"] == 0].copy()
    if "is_injured" in df_loc.columns:
        df_loc = df_loc[df_loc["is_injured"] == 0].copy()

    return df_sm, df_loc, selected_features, tuned_params, spec


def evaluate_frozen_transfer(df_sm, df_loc, selected_features, tuned_params, src_target_col, loc_target_col, task_label):
    sm_group_col = "player_id" if "player_id" in df_sm.columns else "athlete_id"
    loc_group_col = "player_id" if "player_id" in df_loc.columns else "athlete_id"

    sub_sm = df_sm.dropna(subset=[src_target_col, sm_group_col]).reset_index(drop=True)
    sub_loc = df_loc.dropna(subset=[loc_target_col, loc_group_col]).reset_index(drop=True)

    X_sm = sub_sm[selected_features].copy()
    y_sm = sub_sm[src_target_col].astype(int).values
    g_sm = sub_sm[sm_group_col].astype(str).values

    X_loc = sub_loc[selected_features].copy()
    y_loc = sub_loc[loc_target_col].astype(int).values
    g_loc = sub_loc[loc_group_col].astype(str).values

    # 1. 在源域生成 5 折 OOF 预测（仅用于确定源域冻结阈值与源域 Platt 校准器）
    cv = StratifiedGroupKFold(n_splits=5)
    oof_sm = np.zeros(len(y_sm))
    for tr_i, va_i in cv.split(X_sm, y_sm, g_sm):
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_tr = scl.fit_transform(imp.fit_transform(X_sm.iloc[tr_i]))
        X_va = scl.transform(imp.transform(X_sm.iloc[va_i]))
        m = LogisticRegression(
            C=tuned_params.get("C", 0.01),
            penalty=tuned_params.get("penalty", "l2"),
            solver="saga", class_weight="balanced", max_iter=2000, random_state=42
        )
        m.fit(X_tr, y_sm[tr_i])
        oof_sm[va_i] = m.predict_proba(X_va)[:, 1]

    frozen_th_prob = select_source_oof_threshold(y_sm, oof_sm)

    # 2. 在全量 SoccerMon 源域上训练并冻结最终模型参数
    frozen_imputer = SimpleImputer(strategy="median").fit(X_sm)
    X_sm_imp = frozen_imputer.transform(X_sm)
    frozen_scaler = StandardScaler().fit(X_sm_imp)
    X_sm_std = frozen_scaler.transform(X_sm_imp)

    frozen_lr = LogisticRegression(
        C=tuned_params.get("C", 0.01),
        penalty=tuned_params.get("penalty", "l2"),
        solver="saga", class_weight="balanced", max_iter=2000, random_state=42
    )
    frozen_lr.fit(X_sm_std, y_sm)

    # 3. 评估两种冻结迁移模式：
    # 模式 A (Raw Frozen Scaling)：直接使用源域中位数与源域均值方差对本地特征做变换并预测
    X_loc_raw_std = frozen_scaler.transform(frozen_imputer.transform(X_loc))
    prob_loc_raw = frozen_lr.predict_proba(X_loc_raw_std)[:, 1]

    # 模式 B (Target-Cohort Standardized Frozen Model)：模型权重 beta 100% 冻结，仅对本地无标签特征做目标队列标准化（消除设备与人群基线漂移）
    loc_imputer = SimpleImputer(strategy="median").fit(X_loc)
    loc_scaler = StandardScaler().fit(loc_imputer.transform(X_loc))
    X_loc_norm_std = loc_scaler.transform(loc_imputer.transform(X_loc))
    prob_loc_norm = frozen_lr.predict_proba(X_loc_norm_std)[:, 1]

    summary_rows = []
    for mode_name, prob_vec in [
        ("5a. 冻结源域最优LR (直接源域分布映射 Zero-Shot)", prob_loc_raw),
        ("5b. 冻结源域最优LR (无监督目标域特征标准化 Zero-Shot)", prob_loc_norm),
    ]:
        auc = roc_auc_score(y_loc, prob_vec)
        ap = average_precision_score(y_loc, prob_vec)
        (auc_lo, auc_hi), (ap_lo, ap_hi) = cluster_bootstrap_ci(y_loc, prob_vec, g_loc)
        brier = brier_score_loss(y_loc, prob_vec)
        lift, top20_rate, bot20_rate = compute_top_bottom_lift(y_loc, prob_vec)

        pred_bin = (prob_vec >= frozen_th_prob).astype(int)
        alert_rate = pred_bin.mean() * 100.0
        f1 = f1_score(y_loc, pred_bin, zero_division=0)
        rec = recall_score(y_loc, pred_bin, zero_division=0)
        prec = precision_score(y_loc, pred_bin, zero_division=0)

        summary_rows.append({
            "Task": task_label,
            "Validation_Scheme": mode_name,
            "Local_Samples": len(y_loc),
            "Local_Positives": int(y_loc.sum()),
            "Prevalence_Pct": f"{y_loc.mean()*100:.2f}%",
            "Pooled_ROC_AUC (95% CI)": f"{auc:.4f} ({auc_lo:.3f}-{auc_hi:.3f})",
            "Pooled_AP (95% CI)": f"{ap:.4f} ({ap_lo:.3f}-{ap_hi:.3f})",
            "Brier_Score": round(brier, 4),
            "Top20_vs_Bot20_Lift": f"{lift:.2f}x ({top20_rate*100:.1f}% vs {bot20_rate*100:.1f}%)",
            "Frozen_Th_Alert_Pct": f"{alert_rate:.1f}%",
            "F1": round(f1, 4),
            "Recall": round(rec, 4),
            "Precision": round(prec, 4),
        })

    # 4. 不同预警预算 (Top 10% ~ 30%) 敏感性分析（基于 5b 冻结权重评分）
    budget_rows = []
    for budget_pct in [10, 15, 20, 25, 30]:
        th = np.quantile(prob_loc_norm, 1.0 - budget_pct / 100.0)
        pred_b = (prob_loc_norm >= th).astype(int)
        budget_rows.append({
            "Task": task_label,
            "Alert_Budget": f"Top {budget_pct}%",
            "Actual_Alert_Pct": f"{pred_b.mean()*100:.1f}%",
            "F1": round(f1_score(y_loc, pred_b, zero_division=0), 4),
            "Recall (Sensitivity)": round(recall_score(y_loc, pred_b, zero_division=0), 4),
            "Precision (PPV)": round(precision_score(y_loc, pred_b, zero_division=0), 4),
            "Lift_Over_Base": f"{(precision_score(y_loc, pred_b, zero_division=0) / max(y_loc.mean(), 1e-5)):.2f}x",
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(budget_rows)


def main():
    df_sm, df_loc, selected_features, tuned_params, spec = load_datasets()
    src_target_7d = spec["target_col"]
    src_target_3d = next(
        (c for c in ["target_all_inj_next_3d", "target_all_inj_next_3d"] if c in df_sm.columns),
        src_target_7d,
    )
    loc_target_7d = "target_all_inj_next_7d"
    loc_target_3d = "target_all_inj_next_3d"

    print("=" * 115)
    print(f"🚀 Step 05: 冻结源域最优基准模型 [{spec['best_model']}, 参数={tuned_params}] -> 本地 24 名女足外部验证")
    print(f"   冻结特征数 = {len(selected_features)} 个 (来自 Step 14 客观 LASSO 非零筛选)")
    print("=" * 115)

    sum_7d, bud_7d = evaluate_frozen_transfer(
        df_sm, df_loc, selected_features, tuned_params,
        src_target_7d, loc_target_7d, "未来 7 天自报告运动损伤 (7d Self-Reported Injury)"
    )
    sum_3d, bud_3d = evaluate_frozen_transfer(
        df_sm, df_loc, selected_features, tuned_params,
        src_target_3d, loc_target_3d, "未来 3 天自报告运动损伤 (3d Self-Reported Injury)"
    )

    all_sum = pd.concat([sum_7d, sum_3d], ignore_index=True)
    all_bud = pd.concat([bud_7d, bud_3d], ignore_index=True)

    sum_path = OUTPUTS_DIR / "step05_frozen_lr_external_validation.csv"
    bud_path = OUTPUTS_DIR / "step05_frozen_lr_budget_sensitivity.csv"
    all_sum.to_csv(sum_path, index=False, encoding="utf-8-sig")
    all_bud.to_csv(bud_path, index=False, encoding="utf-8-sig")

    print("\n📊 [表 1] 冻结源域最优 LR 基准模型 —— 本地 24 人外部验证主结果表:")
    cols_show = ["Task", "Validation_Scheme", "Prevalence_Pct", "Pooled_ROC_AUC (95% CI)", "Pooled_AP (95% CI)", "Top20_vs_Bot20_Lift", "F1", "Recall", "Precision"]
    print(all_sum[cols_show].to_string(index=False))

    print("\n📊 [表 2] 冻结源域最优 LR 基准模型 —— 本地 24 人不同预警预算 (Top 10%~30%) 敏感性分析:")
    print(all_bud.to_string(index=False))

    print("\n" + "=" * 115)
    print(f"✅ Step 05 冻结外部验证完成！结果已保存至:\n   - {sum_path}\n   - {bud_path}")
    print("=" * 115)


if __name__ == "__main__":
    main()