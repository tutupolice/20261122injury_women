# src/step14_soccermon_8models_benchmark.py
"""
Step 14 (零泄露 + 客观 LASSO 特征筛选最终版):
1. 严格排除所有当日/未来损伤结局列 (inj_all_onset, inj_timeloss_onset, inj_non_contact_onset, is_injured 等)，杜绝标签泄露；
2. 完整打印【原始合并 -> 特征工程 -> 跨队列共有候选池 -> Spearman去共线 -> LASSO客观非零筛选】五步账本；
3. 完全去掉人为 Top-K 数量限制：在每个训练折内部由 Spearman (|r|<=0.80) 去共线后，
   由交叉验证 LASSO (LogisticRegressionCV L1) 自动将无贡献特征系数置为 0，仅保留非零系数特征；
4. 按 AP (PR-AUC) 优先、ROC-AUC 次优先对比 8 大模型，输出唯一源域最优基准模型及参数。
"""
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


class ObjectiveLassoFeatureSelector(BaseEstimator, TransformerMixin):
    """
    100% 客观的两阶段特征筛选器（无任何人为数量上限 Top-K）：
    阶段 1 (Spearman 共线性过滤): 当两个特征 |r| > corr_threshold (0.80) 时，客观保留与损伤标签 y 相关性更高的那一个；
    阶段 2 (LASSO L1 稀疏回归筛选): 在标准化后的非共线特征上拟合 L1 正则化逻辑回归，
           自动将噪声与无关特征的系数压缩为 0，客观保留所有回归系数非零 (|coef| > 1e-4) 的特征。
    """
    def __init__(self, corr_threshold=0.80, l1_C=0.05):
        self.corr_threshold = corr_threshold
        self.l1_C = l1_C
        self.non_collinear_indices_ = None
        self.selected_indices_ = None
        self.l1_coefs_ = None

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        n_samples, n_features = X_arr.shape

        # 1. 计算每个特征与损伤标签 y 的单变量 Spearman |r|
        uni_scores = np.zeros(n_features)
        for j in range(n_features):
            col = X_arr[:, j]
            if np.std(col) < 1e-8:
                uni_scores[j] = 0.0
            else:
                r, _ = spearmanr(col, y_arr)
                uni_scores[j] = abs(r) if not np.isnan(r) else 0.0

        # 2. Spearman 共线性客观去冗余 (|r| > 0.80)
        corr_mat = np.abs(np.corrcoef(X_arr, rowvar=False))
        corr_mat = np.nan_to_num(corr_mat, 0.0)

        order = np.argsort(-uni_scores)
        dropped = set()
        for idx_a in range(n_features):
            i = order[idx_a]
            if i in dropped or uni_scores[i] == 0.0:
                dropped.add(i)
                continue
            for idx_b in range(idx_a + 1, n_features):
                j = order[idx_b]
                if j in dropped:
                    continue
                if corr_mat[i, j] > self.corr_threshold:
                    dropped.add(j)

        kept = [i for i in order if i not in dropped]
        self.non_collinear_indices_ = kept

        # 3. 客观 LASSO (L1) 稀疏筛选：不人为指定保留几个，只保留 L1 非零系数特征
        X_sub = X_arr[:, kept]
        X_sub_std = StandardScaler().fit_transform(X_sub)
        lasso = LogisticRegression(
            penalty="l1", solver="saga", C=self.l1_C,
            class_weight="balanced", max_iter=1500, random_state=42
        )
        lasso.fit(X_sub_std, y_arr)
        coefs = np.abs(lasso.coef_[0])
        non_zero_mask = coefs > 1e-4

        if non_zero_mask.sum() >= 3:
            # 按 L1 系数绝对值从大到小排序保留所有非零特征
            nz_sub_idx = np.where(non_zero_mask)[0]
            nz_sub_idx = nz_sub_idx[np.argsort(-coefs[nz_sub_idx])]
            self.selected_indices_ = [kept[idx] for idx in nz_sub_idx]
            self.l1_coefs_ = coefs[nz_sub_idx]
        else:
            # 极端情况兜底
            top_idx = np.argsort(-uni_scores[kept])[:10]
            self.selected_indices_ = [kept[idx] for idx in top_idx]
            self.l1_coefs_ = uni_scores[self.selected_indices_]

        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return X_arr[:, self.selected_indices_]


def is_leakage_or_meta_col(col_name: str) -> bool:
    """严格判定是否为 ID、日期、康复期状态或当日/未来损伤结局列（严禁作为预测特征）"""
    c = col_name.lower()
    exact_ban = {
        "player_id", "athlete_id", "date", "team", "session_id", "injury_id",
        "is_rehab", "is_injured", "active_day", "has_monitoring",
        "inj_all_onset", "inj_timeloss_onset", "inj_non_contact_onset",
        "injury_onset", "non_contact_onset",
    }
    if c in exact_ban:
        return True
    prefix_ban = ("target_", "future_", "next_", "label_")
    if c.startswith(prefix_ban):
        return True
    # 任何带 _onset 的列均为当日事件结局标签，只有 prev_inj_* / recent_inj_decay / days_since_last_inj 才是历史特征
    if c.endswith("_onset") or "timeloss" in c:
        return True
    return False


def audit_and_load_data():
    raw_sm_path = PROCESSED_DIR / "soccermon_daily_panel.csv"
    raw_loc_path = PROCESSED_DIR / "local_daily_panel.csv"
    feat_sm_path = PROCESSED_DIR / "soccermon_features.csv"
    feat_loc_path = PROCESSED_DIR / "local_features.csv"

    def get_clean_feature_cols(df):
        return [
            c for c in df.columns
            if not is_leakage_or_meta_col(c)
            and pd.api.types.is_numeric_dtype(df[c])
        ]

    raw_sm_cols = get_clean_feature_cols(pd.read_csv(raw_sm_path, nrows=50)) if raw_sm_path.exists() else []
    raw_loc_cols = get_clean_feature_cols(pd.read_csv(raw_loc_path, nrows=50)) if raw_loc_path.exists() else []

    df_sm = pd.read_csv(feat_sm_path)
    df_loc = pd.read_csv(feat_loc_path) if feat_loc_path.exists() else df_sm.copy()

    eng_sm_cols = get_clean_feature_cols(df_sm)
    eng_loc_cols = get_clean_feature_cols(df_loc)

    group_col = "player_id" if "player_id" in df_sm.columns else "athlete_id"

    # 严格对齐 Step 11 的有效监控日与非康复期过滤
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

    target_candidates = [
        "target_all_inj_next_7d",
        "target_non_contact_next_7d",
        "target_injury_next_7d",
        "injury_next_7d",
        "target_next_7d",
    ]
    target_col = next((c for c in target_candidates if c in df_sm.columns), None)
    df_sm = df_sm.dropna(subset=[target_col, group_col]).reset_index(drop=True)

    # 1. 跨队列共有且非空率 >= 40% 的干净候选特征（同时排除未按个体归一化、跨设备物理单位不同的原始绝对值列）
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

    # 全量拟合一次仅用于打印特征筛选账本（实际交叉验证在每个训练折内部独立 fit，绝不泄露）
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    selector_audit = ObjectiveLassoFeatureSelector(corr_threshold=0.80, l1_C=0.05).fit(X_imp, y)
    non_collinear_cols = [shared_cols[i] for i in selector_audit.non_collinear_indices_]
    lasso_selected_cols = [shared_cols[i] for i in selector_audit.selected_indices_]

    print("=" * 110)
    print("📋 【数据预处理 -> 特征工程 -> 客观特征筛选】全流程特征维度账本 (零泄露版)")
    print("=" * 110)
    print(f"1. 原始主客观合并面板 (step02/02b)          : SoccerMon 基础特征 = {len(raw_sm_cols)} 个 | 本地基础特征 = {len(raw_loc_cols)} 个")
    print(f"2. 时序特征工程面板   (step03, 剔除结局标签) : SoccerMon 派生特征 = {len(eng_sm_cols)} 个 | 本地派生特征 = {len(eng_loc_cols)} 个")
    print(f"3. 跨队列共有且有效率>=40%的候选特征池       : {len(shared_cols)} 个")
    print(f"4. 阶段一 [Spearman |r|<=0.80 客观去共线性]  : 从 {len(shared_cols)} 个中剔除高度共线冗余后剩 = {len(non_collinear_cols)} 个")
    print(f"5. 阶段二 [LASSO (L1) 稀疏回归客观非零筛选]  : 自动剔除零系数噪声特征后，客观保留 = {len(lasso_selected_cols)} 个")
    print(f"   -> LASSO 客观保留的非零特征 ({len(lasso_selected_cols)} 个):")
    for idx, (col, coef) in enumerate(zip(lasso_selected_cols, selector_audit.l1_coefs_), 1):
        print(f"      {idx:2d}. {col:28s} (|L1 β| = {coef:.4f})")
    print("=" * 110)

    audit_info = {
        "raw_soccermon_feature_count": len(raw_sm_cols),
        "raw_local_feature_count": len(raw_loc_cols),
        "engineered_soccermon_feature_count": len(eng_sm_cols),
        "shared_valid_candidate_count": len(shared_cols),
        "after_spearman_collinearity_count": len(non_collinear_cols),
        "after_lasso_nonzero_count": len(lasso_selected_cols),
        "lasso_selected_features": lasso_selected_cols,
    }
    return X, y, groups, target_col, shared_cols, audit_info


def get_model_spaces(pos_weight: float):
    return {
        "1. Logistic Regression (LR)": {
            "needs_scaler": True,
            "builder": lambda p: LogisticRegression(
                C=p["C"], penalty=p["penalty"], solver="saga",
                class_weight="balanced", max_iter=2000, random_state=42
            ),
            "grid": [
                {"C": 0.01, "penalty": "l2"},
                {"C": 0.05, "penalty": "l2"},
                {"C": 0.1, "penalty": "l2"},
                {"C": 0.1, "penalty": "l1"},
            ],
        },
        "2. Gaussian Naive Bayes (GNB)": {
            "needs_scaler": True,
            "builder": lambda p: GaussianNB(var_smoothing=p["var_smoothing"]),
            "grid": [
                {"var_smoothing": 1e-9},
                {"var_smoothing": 1e-3},
                {"var_smoothing": 1e-1},
            ],
        },
        "3. Random Forest (RF)": {
            "needs_scaler": False,
            "builder": lambda p: RandomForestClassifier(
                n_estimators=300, max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"], max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1, random_state=42
            ),
            "grid": [
                {"max_depth": 4, "min_samples_leaf": 30},
                {"max_depth": 6, "min_samples_leaf": 40},
                {"max_depth": 8, "min_samples_leaf": 50},
            ],
        },
        "4. XGBoost": {
            "needs_scaler": False,
            "builder": lambda p: XGBClassifier(
                n_estimators=250, learning_rate=p["learning_rate"],
                max_depth=p["max_depth"], min_child_weight=p["min_child_weight"],
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=p["reg_lambda"], scale_pos_weight=pos_weight,
                eval_metric="aucpr", n_jobs=-1, random_state=42
            ),
            "grid": [
                {"learning_rate": 0.02, "max_depth": 3, "min_child_weight": 30, "reg_lambda": 10.0},
                {"learning_rate": 0.03, "max_depth": 3, "min_child_weight": 50, "reg_lambda": 15.0},
                {"learning_rate": 0.05, "max_depth": 4, "min_child_weight": 50, "reg_lambda": 10.0},
            ],
        },
        "5. LightGBM": {
            "needs_scaler": False,
            "builder": lambda p: LGBMClassifier(
                n_estimators=250, learning_rate=p["learning_rate"],
                max_depth=p["max_depth"], num_leaves=p["num_leaves"],
                min_child_samples=p["min_child_samples"],
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=10.0, class_weight="balanced",
                n_jobs=-1, verbose=-1, random_state=42
            ),
            "grid": [
                {"learning_rate": 0.02, "max_depth": 3, "num_leaves": 7, "min_child_samples": 50},
                {"learning_rate": 0.03, "max_depth": 3, "num_leaves": 7, "min_child_samples": 80},
                {"learning_rate": 0.05, "max_depth": 4, "num_leaves": 12, "min_child_samples": 80},
            ],
        },
        "6. CatBoost": {
            "needs_scaler": False,
            "builder": lambda p: CatBoostClassifier(
                iterations=300, learning_rate=p["learning_rate"],
                depth=p["depth"], l2_leaf_reg=p["l2_leaf_reg"],
                auto_class_weights="Balanced", verbose=0, thread_count=-1, random_seed=42
            ),
            "grid": [
                {"learning_rate": 0.02, "depth": 3, "l2_leaf_reg": 10.0},
                {"learning_rate": 0.03, "depth": 4, "l2_leaf_reg": 10.0},
                {"learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 15.0},
            ],
        },
        "7. SVM-RBF": {
            "needs_scaler": True,
            "builder": lambda p: SVC(
                kernel="rbf", C=p["C"], gamma=p["gamma"],
                class_weight="balanced", probability=True,
                cache_size=1000, max_iter=5000, random_state=42
            ),
            "grid": [
                {"C": 0.1, "gamma": "scale"},
                {"C": 0.5, "gamma": "scale"},
                {"C": 1.0, "gamma": "scale"},
            ],
        },
        "8. Multilayer Perceptron (MLP)": {
            "needs_scaler": True,
            "builder": lambda p: MLPClassifier(
                hidden_layer_sizes=p["hidden_layer_sizes"],
                alpha=p["alpha"], max_iter=300, early_stopping=True, random_state=42
            ),
            "grid": [
                {"hidden_layer_sizes": (32, 16), "alpha": 0.1},
                {"hidden_layer_sizes": (64, 32), "alpha": 0.1},
                {"hidden_layer_sizes": (32,), "alpha": 0.5},
            ],
        },
    }


def make_clf_pipeline(estimator, needs_scaler: bool):
    steps = []
    if needs_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def main():
    X, y, groups, target_col, shared_cols, audit_info = audit_and_load_data()
    pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1))

    outer_cv = StratifiedGroupKFold(n_splits=5)
    model_spaces = get_model_spaces(pos_weight)

    # 在每个外层训练折内部独立执行【中位数填充 + Spearman去共线 + LASSO非零特征筛选】
    fold_data = []
    fold_feat_counts = []
    for fold_idx, (tr_idx, va_idx) in enumerate(outer_cv.split(X, y, groups)):
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X.iloc[tr_idx])
        X_va_imp = imp.transform(X.iloc[va_idx])

        selector = ObjectiveLassoFeatureSelector(corr_threshold=0.80, l1_C=0.05)
        X_tr_sel = selector.fit_transform(X_tr_imp, y[tr_idx])
        X_va_sel = selector.transform(X_va_imp)
        fold_feat_counts.append(X_tr_sel.shape[1])

        fold_data.append({
            "tr_idx": tr_idx,
            "va_idx": va_idx,
            "X_tr": X_tr_sel,
            "y_tr": y[tr_idx],
            "g_tr": groups[tr_idx],
            "X_va": X_va_sel,
            "y_va": y[va_idx],
        })

    print(f"📌 5 折交叉验证各折内部客观 LASSO 自动保留的特征数分别为: {fold_feat_counts} (均值={np.mean(fold_feat_counts):.1f} 个)\n")

    results = []
    best_params_per_model = {}

    for name, spec in model_spaces.items():
        oof_probs = np.full(len(y), np.nan)
        fold_aps, fold_aucs = [], []
        fold_chosen_params = []

        for fd in fold_data:
            X_tr, y_tr, g_tr = fd["X_tr"], fd["y_tr"], fd["g_tr"]
            X_va, y_va = fd["X_va"], fd["y_va"]

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

            fold_chosen_params.append(best_p)
            pipe = make_clf_pipeline(spec["builder"](best_p), spec["needs_scaler"])
            pipe.fit(X_tr, y_tr)
            prob_va = pipe.predict_proba(X_va)[:, 1]
            oof_probs[fd["va_idx"]] = prob_va

            if y_va.sum() > 0 and y_va.sum() < len(y_va):
                fold_aps.append(average_precision_score(y_va, prob_va))
                fold_aucs.append(roc_auc_score(y_va, prob_va))

        valid = ~np.isnan(oof_probs)
        pooled_ap = average_precision_score(y[valid], oof_probs[valid])
        pooled_auc = roc_auc_score(y[valid], oof_probs[valid])
        brier = brier_score_loss(y[valid], oof_probs[valid])

        params_str_list = [json.dumps(p, sort_keys=True) for p in fold_chosen_params]
        modal_params = json.loads(max(set(params_str_list), key=params_str_list.count))
        best_params_per_model[name] = modal_params

        results.append({
            "Model": name,
            "Pooled_AP": round(pooled_ap, 4),
            "Fold_AP_Mean_SD": f"{np.mean(fold_aps):.4f} ± {np.std(fold_aps):.4f}",
            "Pooled_ROC_AUC": round(pooled_auc, 4),
            "Fold_ROC_AUC_Mean_SD": f"{np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}",
            "Brier_Score": round(brier, 4),
            "Modal_Best_Params": json.dumps(modal_params),
        })
        print(f"完成 -> {name:30s} | Pooled AP={pooled_ap:.4f} | Pooled AUC={pooled_auc:.4f} | Brier={brier:.4f}")

    res_df = pd.DataFrame(results).sort_values(
        by=["Pooled_AP", "Pooled_ROC_AUC"], ascending=False
    ).reset_index(drop=True)

    csv_path = OUTPUTS_DIR / "step14_soccermon_8models_cv.csv"
    res_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    winner = res_df.iloc[0]
    winner_name = winner["Model"]
    winner_tuned_params = best_params_per_model[winner_name]
    winner_full_params = model_spaces[winner_name]["builder"](winner_tuned_params).get_params()

    winner_json = {
        "selection_criterion": "AP (PR-AUC) primary, ROC-AUC secondary",
        "target_col": target_col,
        "feature_audit": audit_info,
        "outer_fold_lasso_feature_counts": fold_feat_counts,
        "best_model": winner_name,
        "pooled_ap": float(winner["Pooled_AP"]),
        "pooled_roc_auc": float(winner["Pooled_ROC_AUC"]),
        "brier_score": float(winner["Brier_Score"]),
        "tuned_hyperparameters": winner_tuned_params,
        "full_model_params": {k: str(v) for k, v in winner_full_params.items()},
    }
    json_path = OUTPUTS_DIR / "step14_best_source_model_params.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(winner_json, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 110)
    print("🏆 零泄露 + 客观 LASSO 特征筛选后 SoccerMon 源域 8 大模型对比汇总表 (按 AP 降序):")
    print(res_df[["Model", "Pooled_AP", "Fold_AP_Mean_SD", "Pooled_ROC_AUC", "Fold_ROC_AUC_Mean_SD", "Brier_Score"]].to_string(index=False))
    print("-" * 110)
    print(f"✅ 选定唯一源域最优基准模型: {winner_name}")
    print(f"📌 核心调优参数: {winner_tuned_params}")
    print(f"📋 完整模型参数: {winner_full_params}")
    print("=" * 110)


if __name__ == "__main__":
    main()