"""Step 19: Local target-cohort LinearSHAP ranking & Source-vs-Local consistency.

Evaluates the exact 38 locked features from step13 on the locked local risk set
(24 collegiate players, 1,315 player-days, 225 7-day positives) using 5-fold
StratifiedGroupKFold out-of-fold LinearSHAP, and compares feature contribution
rankings and domain shares against the SoccerMon source model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
TARGET = "target_all_inj_next_7d"


def main() -> None:
    # 1. 读取源域锁定的 38 特征 SHAP 排名表
    src_path = OUTPUTS / "step13_shap_soccermon_best_lr_ranking.csv"
    src = pd.read_csv(src_path)
    if len(src) != 38:
        raise AssertionError(f"step13 特征数应为 38，实际为 {len(src)}")
    features = src["Feature"].tolist()

    # 2. 读取本地锁定风险集
    loc = pd.read_csv(PROCESSED / "local_features.csv")
    risk = (
        loc.loc[
            (loc["active_day"] == 1)
            & (loc["is_rehab"] == 0)
            & (loc["is_injured"] == 0)
        ]
        .dropna(subset=[TARGET, "player_id"])
        .reset_index(drop=True)
    )
    y = risk[TARGET].astype(int).to_numpy()
    groups = risk["player_id"].astype(str).to_numpy()
    if len(risk) != 1315 or int(y.sum()) != 225 or len(np.unique(groups)) != 24:
        raise AssertionError(
            f"本地风险集与锁定回执不符: n={len(risk)}, pos={y.sum()}, players={len(np.unique(groups))}"
        )

    missing_feats = [f for f in features if f not in risk.columns]
    if missing_feats:
        raise ValueError(f"本地特征表缺少源域特征: {missing_feats}")

    X = risk[features].copy()

    # 3. 5 折按队员分组折外 LinearSHAP（与源域 LR 保持一致的正则化逻辑回归）
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_shap = np.zeros((len(risk), len(features)), dtype=float)
    fold_betas = []

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups), 1):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X.iloc[tr_idx])
        X_te = imp.transform(X.iloc[te_idx])

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        lr = LogisticRegression(
            C=0.01,
            penalty="l2",
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
        lr.fit(X_tr_s, y[tr_idx])
        beta = lr.coef_[0]
        fold_betas.append(beta)

        # 线性模型解析 SHAP：SHAP_ij = beta_j * (z_ij - E_train[z_j])，标准化后 E_train[z_j] = 0
        oof_shap[te_idx, :] = X_te_s * beta[np.newaxis, :]

    local_mean_abs = np.mean(np.abs(oof_shap), axis=0)
    local_pct = 100.0 * local_mean_abs / local_mean_abs.sum()
    local_beta_mean = np.mean(fold_betas, axis=0)

    # 4. 合并源域与本地 38 项特征对比表
    comp = src[
        [
            "Rank",
            "Feature",
            "Domain_Category",
            "Mean_Abs_SHAP",
            "Contribution_Pct",
            "Standardized_LR_Beta",
        ]
    ].rename(
        columns={
            "Rank": "Source_Rank",
            "Mean_Abs_SHAP": "Source_Mean_Abs_SHAP",
            "Contribution_Pct": "Source_Pct",
            "Standardized_LR_Beta": "Source_Beta",
        }
    )
    comp["Local_Mean_Abs_SHAP"] = local_mean_abs
    comp["Local_Pct"] = local_pct
    comp["Local_Beta_Mean"] = local_beta_mean
    comp["Local_Rank"] = comp["Local_Pct"].rank(ascending=False, method="min").astype(int)
    comp["Sign_Consistent"] = np.sign(comp["Source_Beta"]) == np.sign(comp["Local_Beta_Mean"])

    comp_sorted_local = comp.sort_values("Local_Rank").reset_index(drop=True)
    comp_sorted_local.to_csv(
        OUTPUTS / "step19_local_vs_source_shap_38features.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 5. 两队特征一致性统计（Spearman 秩相关 + Top10 重合 + 四大域对比）
    rho, p_val = spearmanr(comp["Source_Pct"], comp["Local_Pct"])
    top10_src = set(comp.nsmallest(10, "Source_Rank")["Feature"])
    top10_loc = set(comp.nsmallest(10, "Local_Rank")["Feature"])
    overlap10 = sorted(top10_src & top10_loc)

    domain_summary = (
        comp.groupby("Domain_Category", as_index=False)
        .agg(
            Features=("Feature", "size"),
            Source_Pct=("Source_Pct", "sum"),
            Local_Pct=("Local_Pct", "sum"),
        )
        .sort_values("Local_Pct", ascending=False)
    )
    domain_summary["Source_Pct"] = domain_summary["Source_Pct"].round(2)
    domain_summary["Local_Pct"] = domain_summary["Local_Pct"].round(2)
    domain_summary.to_csv(
        OUTPUTS / "step19_domain_consistency_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("本地高校女足微调模型（24人, 1,315风险人日）折外 LinearSHAP 排名前 10 特征：")
    print("=" * 80)
    cols_show = [
        "Local_Rank",
        "Feature",
        "Domain_Category",
        "Local_Pct",
        "Source_Rank",
        "Source_Pct",
        "Local_Beta_Mean",
        "Source_Beta",
        "Sign_Consistent",
    ]
    print(comp_sorted_local[cols_show].head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("两队 38 项共有特征一致性统计：")
    print("=" * 80)
    print(f"Spearman 秩相关系数 ρ = {rho:.4f} (p = {p_val:.4g})")
    print(f"两队 Top-10 共同入选特征 ({len(overlap10)} 项): {overlap10}")
    print(
        f"38 项特征系数方向一致数: {int(comp['Sign_Consistent'].sum())} / 38 "
        f"({100 * comp['Sign_Consistent'].mean():.1f}%)"
    )

    print("\n" + "=" * 80)
    print("四大特征域贡献占比对照（源域 SoccerMon vs 本队高校女足）：")
    print("=" * 80)
    print(domain_summary.to_string(index=False))
    print(
        "\n结果已保存至 outputs/step19_local_vs_source_shap_38features.csv "
        "与 outputs/step19_domain_consistency_summary.csv"
    )


if __name__ == "__main__":
    main()