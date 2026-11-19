# src/step14b_next1d_benchmark.py
"""
Step 14b: 检验【次日/目标日新发伤病 (Next-Day Onset, 1天窗口)】在 SoccerMon 源域 8 大模型上的真实表现
注：soccermon_features.csv 中的特征均已严格 shift(1)（仅含目标日之前历史信息），
因此以 inj_all_onset 为因变量即等价于“用截至昨日的历史数据，预测今日（次日）是否新发伤病通知”。
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from step14_soccermon_8models_benchmark import (
    ObjectiveLassoFeatureSelector,
    is_leakage_or_meta_col,
    get_model_spaces,
    make_clf_pipeline,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def main():
    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv")
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv")

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
    if "inj_all_onset" in df_sm.columns and "is_injured" in df_sm.columns:
        df_sm = df_sm[(df_sm["is_injured"] == 0) | (df_sm["inj_all_onset"] == 1)].copy()

    target_col = "inj_all_onset"
    if target_col not in df_sm.columns:
        raise ValueError("未找到 inj_all_onset 列，请检查 soccermon_features.csv 列名。")

    group_col = "player_id" if "player_id" in df_sm.columns else "athlete_id"
    df_sm = df_sm.dropna(subset=[target_col, group_col]).reset_index(drop=True)

    eng_sm_cols = [c for c in df_sm.columns if not is_leakage_or_meta_col(c) and pd.api.types.is_numeric_dtype(df_sm[c])]
    eng_loc_cols = [c for c in df_loc.columns if not is_leakage_or_meta_col(c) and pd.api.types.is_numeric_dtype(df_loc[c])]

    shared_cols = [
        c for c in eng_sm_cols
        if c in eng_loc_cols
        and df_sm[c].notna().mean() >= 0.40
        and df_loc[c].notna().mean() >= 0.40
        and df_sm[c].nunique(dropna=True) > 1
    ]

    X = df_sm[shared_cols].copy()
    y = df_sm[target_col].astype(int).values
    groups = df_sm[group_col].astype(str).values
    pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1))

    print("=" * 105)
    print(f"🔍 SoccerMon 源域【次日/目标日新发伤病 (1天窗口: {target_col})】8 大模型对比")
    print(f"   有效样本数 = {len(y)}, 新发阳性数 = {y.sum()}, 阳性率 = {y.mean()*100:.3f}%, 候选特征 = {len(shared_cols)}")
    print("=" * 105)

    outer_cv = StratifiedGroupKFold(n_splits=5)
    model_spaces = get_model_spaces(pos_weight)

    fold_data = []
    for tr_idx, va_idx in outer_cv.split(X, y, groups):
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X.iloc[tr_idx])
        X_va_imp = imp.transform(X.iloc[va_idx])

        selector = ObjectiveLassoFeatureSelector(corr_threshold=0.80, l1_C=0.05)
        X_tr_sel = selector.fit_transform(X_tr_imp, y[tr_idx])
        X_va_sel = selector.transform(X_va_imp)

        fold_data.append({
            "va_idx": va_idx,
            "X_tr": X_tr_sel, "y_tr": y[tr_idx], "g_tr": groups[tr_idx],
            "X_va": X_va_sel, "y_va": y[va_idx],
        })

    results = []
    for name, spec in model_spaces.items():
        oof_probs = np.full(len(y), np.nan)
        for fd in fold_data:
            X_tr, y_tr, g_tr = fd["X_tr"], fd["y_tr"], fd["g_tr"]
            X_va = fd["X_va"]

            inner_cv = StratifiedGroupKFold(n_splits=3)
            best_key = (-1.0, -1.0)
            best_p = spec["grid"][0]
            for params in spec["grid"]:
                oof_in = np.full(len(y_tr), np.nan)
                for in_tr, in_va in inner_cv.split(X_tr, y_tr, g_tr):
                    if y_tr[in_tr].sum() == 0 or y_tr[in_va].sum() == 0:
                        continue
                    pipe = make_clf_pipeline(spec["builder"](params), spec["needs_scaler"])
                    pipe.fit(X_tr[in_tr], y_tr[in_tr])
                    oof_in[in_va] = pipe.predict_proba(X_tr[in_va])[:, 1]
                valid_in = ~np.isnan(oof_in)
                if valid_in.sum() > 0 and y_tr[valid_in].sum() > 0:
                    ap_in = average_precision_score(y_tr[valid_in], oof_in[valid_in])
                    auc_in = roc_auc_score(y_tr[valid_in], oof_in[valid_in])
                    if (ap_in, auc_in) > best_key:
                        best_key = (ap_in, auc_in)
                        best_p = params

            pipe = make_clf_pipeline(spec["builder"](best_p), spec["needs_scaler"])
            pipe.fit(X_tr, y_tr)
            oof_probs[fd["va_idx"]] = pipe.predict_proba(X_va)[:, 1]

        valid = ~np.isnan(oof_probs)
        ap = average_precision_score(y[valid], oof_probs[valid])
        auc = roc_auc_score(y[valid], oof_probs[valid])
        brier = brier_score_loss(y[valid], oof_probs[valid])
        results.append({
            "Model": name,
            "Pooled_AP": round(ap, 4),
            "AP_Lift": f"{ap / max(y.mean(), 1e-6):.1f}x",
            "Pooled_ROC_AUC": round(auc, 4),
            "Brier": round(brier, 4),
        })
        print(f"完成 -> {name:30s} | AP={ap:.4f} ({ap/y.mean():.1f}x) | AUC={auc:.4f}")

    res_df = pd.DataFrame(results).sort_values(by=["Pooled_AP", "Pooled_ROC_AUC"], ascending=False)
    out_path = OUTPUTS_DIR / "step14b_next1d_8models_cv.csv"
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n🏆 次日新发伤病 (1天窗口) 8 模型汇总表:")
    print(res_df.to_string(index=False))
    print(f"\n💾 已保存至: {out_path}")


if __name__ == "__main__":
    main()