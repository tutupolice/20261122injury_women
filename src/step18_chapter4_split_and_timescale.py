"""Chapter 4 supplement for the locked strict-future 7-day SoccerMon pipeline.

4.7 compares record-level and athlete-level five-fold OOF validation on the
same risk days using the same fixed LR specification. Imputation, feature
selection and scaling are fitted independently within each training fold.
This is a descriptive comparison of two validation targets, not a paired
significance test or an estimate of deployment performance.

4.8 groups the already saved full-source LR SHAP contributions by their
feature time scale. It does not compute lag-specific or interaction SHAP:
the selected linear model has no such decomposition.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from step14_soccermon_8models_benchmark import (
    ObjectiveLassoFeatureSelector,
    is_leakage_or_meta_col,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
TARGET = "target_all_inj_next_7d"


def load_locked_risk_set() -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    sm = pd.read_csv(PROCESSED / "soccermon_features.csv")
    loc = pd.read_csv(PROCESSED / "local_features.csv")
    required = ["player_id", "date", "active_day", "is_rehab", "is_injured", TARGET]
    missing = [c for c in required if c not in sm.columns]
    if missing:
        raise ValueError(f"SoccerMon 缺少正式风险集列: {missing}")

    def candidate_columns(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if not is_leakage_or_meta_col(c)
                and pd.api.types.is_numeric_dtype(df[c])]

    sm_cols = candidate_columns(sm)
    loc_cols = set(candidate_columns(loc))
    risk = sm.loc[(sm["active_day"] == 1) & (sm["is_rehab"] == 0)
                  & (sm["is_injured"] == 0)].dropna(subset=[TARGET, "player_id"]).reset_index(drop=True)
    features = [c for c in sm_cols if c in loc_cols
                and risk[c].notna().mean() >= .40
                and loc[c].notna().mean() >= .40
                and risk[c].nunique(dropna=True) > 1]
    x = risk[features]
    y = risk[TARGET].astype(int).to_numpy()
    groups = risk["player_id"].astype(str).to_numpy()
    if len(risk) != 19266 or int(y.sum()) != 264 or len(np.unique(groups)) != 50 or len(features) != 70:
        raise AssertionError("输入与锁定回执不符；请先重跑 step02→step16 并重新核对。")
    return risk, x, y, groups


def nearby_training_fraction(
    dates: np.ndarray, groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
    horizon_days: int,
) -> float:
    """Fraction of test days with the same athlete in training within ±h days."""
    train_by_group: dict[str, np.ndarray] = {}
    for group in np.unique(groups[train_idx]):
        ix = train_idx[groups[train_idx] == group]
        train_by_group[group] = np.sort(dates[ix])
    window = np.timedelta64(horizon_days, "D")
    nearby = 0
    for ix in test_idx:
        arr = train_by_group.get(groups[ix])
        if arr is None:
            continue
        left = np.searchsorted(arr, dates[ix] - window, side="left")
        if left < len(arr) and arr[left] <= dates[ix] + window:
            nearby += 1
    return nearby / len(test_idx)


def evaluate_split(
    name: str, splitter, risk: pd.DataFrame, x: pd.DataFrame,
    y: np.ndarray, groups: np.ndarray,
) -> dict:
    dates = risk["date"].to_numpy(dtype="datetime64[D]")
    oof = np.full(len(y), np.nan, dtype=float)
    fold_ap, fold_auc, fold_brier, fold_features = [], [], [], []
    overlap7, overlap28 = [], []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), 1):
        if y[train_idx].sum() == 0 or y[test_idx].sum() == 0:
            raise AssertionError(f"{name} 第{fold}折没有阳性；停止而不跳折。")
        imputer = SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(x.iloc[train_idx])
        x_test = imputer.transform(x.iloc[test_idx])
        selector = ObjectiveLassoFeatureSelector(corr_threshold=.80, l1_C=.05)
        x_train = selector.fit_transform(x_train, y[train_idx])
        x_test = selector.transform(x_test)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = LogisticRegression(C=.01, penalty="l2", solver="saga",
                                   class_weight="balanced", max_iter=2000,
                                   random_state=42)
        model.fit(x_train, y[train_idx])
        pred = model.predict_proba(x_test)[:, 1]
        oof[test_idx] = pred
        fold_ap.append(average_precision_score(y[test_idx], pred))
        fold_auc.append(roc_auc_score(y[test_idx], pred))
        fold_brier.append(brier_score_loss(y[test_idx], pred))
        fold_features.append(x_train.shape[1])
        overlap7.append(nearby_training_fraction(dates, groups, train_idx, test_idx, 7) * len(test_idx))
        overlap28.append(nearby_training_fraction(dates, groups, train_idx, test_idx, 28) * len(test_idx))
        print(f"{name} fold {fold}: n={len(test_idx)}, positives={y[test_idx].sum()}, "
              f"AP={fold_ap[-1]:.4f}, AUC={fold_auc[-1]:.4f}, selected={fold_features[-1]}")

    if np.isnan(oof).any():
        raise AssertionError(f"{name}: OOF 预测未覆盖全部风险日。")
    return {
        "Validation_Scheme": name,
        "Risk_Days": len(y), "Players": len(np.unique(groups)),
        "Positive_Windows": int(y.sum()), "Prevalence": round(float(y.mean()), 6),
        "Pooled_AP": round(float(average_precision_score(y, oof)), 6),
        "Pooled_ROC_AUC": round(float(roc_auc_score(y, oof)), 6),
        "Pooled_Brier": round(float(brier_score_loss(y, oof)), 6),
        "Fold_AP_Mean": round(float(np.mean(fold_ap)), 6),
        "Fold_AP_SD": round(float(np.std(fold_ap)), 6),
        "Fold_AUC_Mean": round(float(np.mean(fold_auc)), 6),
        "Fold_AUC_SD": round(float(np.std(fold_auc)), 6),
        "Within_Player_Training_Day_7d_Pct": round(100 * sum(overlap7) / len(y), 2),
        "Within_Player_Training_Day_28d_Pct": round(100 * sum(overlap28) / len(y), 2),
        "Fold_Selected_Features": json.dumps(fold_features),
    }


def time_scale(name: str) -> str:
    if name.startswith("days_since_last_inj") or name.startswith("recent_inj_decay"):
        return "既往伤病时间（截至t-1）"
    if name.startswith("prev_inj_"):
        return "既往伤病计数（截至t-1）"
    if name.endswith("_delta3d"):
        return "当日相对过去3日"
    if name.endswith("_z"):
        return "当日相对过去21日"
    if name.endswith("_roll3d"):
        return "近3日（含t）"
    if name.endswith("_roll7d") or name.endswith("_atl7") or name.endswith("_monotony7") or name.endswith("_strain7"):
        return "近7日（含t）"
    if name.endswith("_ctl28"):
        return "近28日（含t）"
    if name.endswith("_acwr"):
        return "7日与28日比（含t）"
    return "当日指标"


def export_current_shap_time_scales() -> None:
    ranking = pd.read_csv(OUTPUTS / "step13_shap_soccermon_best_lr_ranking.csv")
    if len(ranking) != 38 or not np.isclose(ranking["Contribution_Pct"].sum(), 100, atol=.01):
        raise AssertionError("Step13 SHAP 排名与当前38特征模型不符。")
    ranking = ranking[["Rank", "Feature", "Domain_Category", "Mean_Abs_SHAP",
                       "Contribution_Pct", "Standardized_LR_Beta"]].copy()
    ranking["Time_Scale"] = ranking["Feature"].map(time_scale)
    ranking.to_csv(OUTPUTS / "step18_table_4_8_feature_time_scales.csv", index=False, encoding="utf-8-sig")
    summary = (ranking.groupby("Time_Scale", as_index=False)
               .agg(Features=("Feature", "size"), SHAP_Share_Pct=("Contribution_Pct", "sum"))
               .sort_values("SHAP_Share_Pct", ascending=False))
    summary["SHAP_Share_Pct"] = summary["SHAP_Share_Pct"].round(4)
    summary.to_csv(OUTPUTS / "step18_table_4_8_time_scale_summary.csv", index=False, encoding="utf-8-sig")
    print("\n4.8 当前LR的特征时间尺度归因（全样本模型，描述性）：")
    print(summary.to_string(index=False))
    print("说明：线性LR不提供旧版TreeSHAP逐时点及特征两两交互结果；这些旧结论不能迁用。")


def main() -> None:
    risk, x, y, groups = load_locked_risk_set()
    print(f"锁定输入：{len(y)}风险人日，{len(np.unique(groups))}人，{y.sum()}个7天阳性窗口，{x.shape[1]}个共有候选特征。")
    rows = [
        evaluate_split("Record-level 5-fold (within-athlete)",
                       StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                       risk, x, y, groups),
        evaluate_split("Athlete-disjoint 5-fold (new-athlete)",
                       StratifiedGroupKFold(n_splits=5), risk, x, y, groups),
    ]
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUTS / "step18_table_4_7_split_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n4.7 相同风险人日、相同固定LR设置的两种验证目标：")
    print(result[["Validation_Scheme", "Pooled_AP", "Pooled_ROC_AUC", "Pooled_Brier",
                  "Within_Player_Training_Day_7d_Pct", "Within_Player_Training_Day_28d_Pct"]].to_string(index=False))
    export_current_shap_time_scales()
    print("\n结果已保存至 outputs/step18_table_4_7_split_comparison.csv、"
          "step18_table_4_8_feature_time_scales.csv、step18_table_4_8_time_scale_summary.csv。")


if __name__ == "__main__":
    main()
