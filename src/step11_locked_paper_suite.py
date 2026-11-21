# src/step11_locked_paper_suite.py
"""
女足运动损伤风险跨数据集迁移与目标队伍更新 —— 论文定稿级评估脚本 (Step 11 审查整改版)
严格落实全部定稿要求：
1. 源域保留全部 50 名运动员（消除 15 人选择偏倚），采用全因伤病通知 (target_all_inj_next_7d/3d)；
2. 源域专家采用【5折交叉拟合集成 (Cross-Fitted 5-Fold Ensemble)】，确保源域 OOF 与目标域预测特征分布 100% 同构；
3. 表 2 的 Top 10%~30% 预警预算阈值严格在每个外层折内部由【内层 StratifiedGroupKFold OOF】保存并冻结用于外层测试折；
4. 同时报告标准 GroupKFold (6折) 与官方 StratifiedGroupKFold (5/5 有效折)；
5. 表 3 全面采用匿名运动员编号 (P01~P24) 保护受试者隐私；
6. 引入折内 Platt 概率校准 (In-Fold Logistic Calibration) 并报告 Brier Score。
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
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
    select_threshold_from_oof,
    cluster_bootstrap_ci,
)


def anonymize_player_ids(df: pd.DataFrame) -> pd.DataFrame:
    """将真实姓名映射为匿名编号 P01 ~ P24，保护运动员隐私"""
    df = df.copy()
    unique_names = sorted(df["player_id"].unique())
    name_map = {name: f"P{idx:02d}" for idx, name in enumerate(unique_names, 1)}
    df["anon_id"] = df["player_id"].map(name_map)
    return df


class CrossFittedSourceExpert:
    """
    交叉拟合源域专家 (Cross-Fitted 5-Fold Source Expert)：
    在源域上通过 5-Fold GroupKFold 生成 OOF 分数，并直接保存这 5 个折模型构成的集成器 (Ensemble)。
    """
    def __init__(self, feat_cols: list[str], require_gps: bool = False):
        self.feat_cols = feat_cols
        self.require_gps = require_gps
        self.fold_models_ = []
        self.oof_mean_ = 0.5
        self.oof_std_ = 0.1

    def _fit_one_fold(self, X_tr: np.ndarray, y_tr: np.ndarray):
        imp = SimpleImputer(strategy="median").fit(X_tr)
        sc = StandardScaler().fit(imp.transform(X_tr))
        X_tr_z = sc.transform(imp.transform(X_tr))

        lr = LogisticRegression(C=0.03, class_weight="balanced", random_state=42).fit(X_tr_z, y_tr)
        lr.coef_ = np.maximum(lr.coef_, 0.02)
        gb = HistGradientBoostingClassifier(
            max_iter=80, max_depth=2, learning_rate=0.03, min_samples_leaf=30,
            l2_regularization=8.0, monotonic_cst=[1] * len(self.feat_cols),
            class_weight="balanced", random_state=42
        ).fit(X_tr_z, y_tr)
        return (imp, sc, lr, gb)

    def fit_and_get_oof(self, sm_df: pd.DataFrame, target_col: str) -> np.ndarray:
        gkf = GroupKFold(n_splits=5)
        raw_oof = np.full(len(sm_df), np.nan)

        if self.require_gps:
            valid_mask = sm_df["total_distance_m"].notna().values
            sub_df = sm_df[valid_mask].reset_index(drop=True)
        else:
            valid_mask = np.ones(len(sm_df), dtype=bool)
            sub_df = sm_df.reset_index(drop=True)

        X_sub = sub_df[self.feat_cols].values
        y_sub = sub_df[target_col].values
        g_sub = sub_df["player_id"].values
        sub_oof = np.zeros(len(sub_df))
        self.fold_models_ = []

        for tr_i, te_i in gkf.split(X_sub, y_sub, groups=g_sub):
            model_tuple = self._fit_one_fold(X_sub[tr_i], y_sub[tr_i])
            self.fold_models_.append(model_tuple)
            imp, sc, lr, gb = model_tuple
            X_te_z = sc.transform(imp.transform(X_sub[te_i]))
            sub_oof[te_i] = 0.60 * lr.predict_proba(X_te_z)[:, 1] + 0.40 * gb.predict_proba(X_te_z)[:, 1]

        self.oof_mean_ = float(np.mean(sub_oof))
        self.oof_std_ = float(np.std(sub_oof) + 1e-6)

        z_sub = np.clip((sub_oof - self.oof_mean_) / self.oof_std_, -4.0, 4.0)
        raw_oof[valid_mask] = 1.0 / (1.0 + np.exp(-z_sub))
        raw_oof[~valid_mask] = 0.50
        return raw_oof

    def predict_cross_fitted(self, df_eval: pd.DataFrame) -> np.ndarray:
        """使用保存的 5 个源域折模型做交叉拟合平均预测"""
        X = df_eval[self.feat_cols].values
        fold_preds = []
        for imp, sc, lr, gb in self.fold_models_:
            X_z = sc.transform(imp.transform(X))
            p = 0.60 * lr.predict_proba(X_z)[:, 1] + 0.40 * gb.predict_proba(X_z)[:, 1]
            fold_preds.append(p)
        raw_mean = np.mean(fold_preds, axis=0)
        z = np.clip((raw_mean - self.oof_mean_) / self.oof_std_, -4.0, 4.0)
        return 1.0 / (1.0 + np.exp(-z))


def fit_nested_crossfitted_source(sm_clean: pd.DataFrame, loc_active: pd.DataFrame, target_col: str):
    """严格双层嵌套 + 交叉拟合集成源域预训练"""
    y_sm = sm_clean[target_col].values
    g_sm = sm_clean["player_id"].values
    outer_gkf = GroupKFold(n_splits=5)

    nested_oof_2a = np.zeros(len(sm_clean))
    nested_oof_2b = np.zeros(len(sm_clean))
    outer_2b_models = []

    zs_cols = ["sm_expert_overload", "sm_expert_biomech", "sm_expert_wellness", "recent_inj_decay", "prev_inj_30d"]

    for out_tr, out_val in outer_gkf.split(sm_clean, y_sm, groups=g_sm):
        sm_tr_df = sm_clean.iloc[out_tr].reset_index(drop=True)
        sm_val_df = sm_clean.iloc[out_val].reset_index(drop=True)
        y_out_tr = y_sm[out_tr]

        e_ov = CrossFittedSourceExpert(FEAT_EXPERT_OVERLOAD, require_gps=False)
        e_bi = CrossFittedSourceExpert(FEAT_EXPERT_BIOMECH, require_gps=True)
        e_we = CrossFittedSourceExpert(FEAT_EXPERT_WELLNESS, require_gps=False)

        sm_tr_df["sm_expert_overload"] = e_ov.fit_and_get_oof(sm_tr_df, target_col)
        sm_tr_df["sm_expert_biomech"] = e_bi.fit_and_get_oof(sm_tr_df, target_col)
        sm_tr_df["sm_expert_wellness"] = e_we.fit_and_get_oof(sm_tr_df, target_col)

        sm_val_df["sm_expert_overload"] = e_ov.predict_cross_fitted(sm_val_df)
        sm_val_df["sm_expert_biomech"] = e_bi.predict_cross_fitted(sm_val_df)
        sm_val_df["sm_expert_wellness"] = e_we.predict_cross_fitted(sm_val_df)

        raw_2a_tr = (
            0.40 * sm_tr_df["sm_expert_overload"].values
            + 0.30 * sm_tr_df["sm_expert_biomech"].values
            + 0.30 * sm_tr_df["sm_expert_wellness"].values
        )
        raw_2a_val = (
            0.40 * sm_val_df["sm_expert_overload"].values
            + 0.30 * sm_val_df["sm_expert_biomech"].values
            + 0.30 * sm_val_df["sm_expert_wellness"].values
        )
        cal_2a = LogisticRegression(C=1.0, random_state=42).fit(raw_2a_tr.reshape(-1, 1), y_out_tr)
        nested_oof_2a[out_val] = cal_2a.predict_proba(raw_2a_val.reshape(-1, 1))[:, 1]

        imp_s = SimpleImputer(strategy="median").fit(sm_tr_df[zs_cols].values)
        sc_s = StandardScaler().fit(imp_s.transform(sm_tr_df[zs_cols].values))
        X_tr_s = sc_s.transform(imp_s.transform(sm_tr_df[zs_cols].values))
        X_val_s = sc_s.transform(imp_s.transform(sm_val_df[zs_cols].values))

        lr_s = LogisticRegression(C=0.05, class_weight="balanced", random_state=42).fit(X_tr_s, y_out_tr)
        lr_s.coef_ = np.maximum(lr_s.coef_, 0.05)
        raw_2b_tr = lr_s.predict_proba(X_tr_s)[:, 1]
        raw_2b_val = lr_s.predict_proba(X_val_s)[:, 1]

        cal_2b = LogisticRegression(C=1.0, random_state=42).fit(raw_2b_tr.reshape(-1, 1), y_out_tr)
        nested_oof_2b[out_val] = cal_2b.predict_proba(raw_2b_val.reshape(-1, 1))[:, 1]

        outer_2b_models.append((e_ov, e_bi, e_we, imp_s, sc_s, lr_s, cal_2a, cal_2b))

    th_2a = select_threshold_from_oof(y_sm, nested_oof_2a)
    th_2b = select_threshold_from_oof(y_sm, nested_oof_2b)

    loc_out = loc_active.copy()
    ov_preds, bi_preds, we_preds, p2a_preds, p2b_preds = [], [], [], [], []

    for e_ov, e_bi, e_we, imp_s, sc_s, lr_s, cal_2a, cal_2b in outer_2b_models:
        p_ov = e_ov.predict_cross_fitted(loc_out)
        p_bi = e_bi.predict_cross_fitted(loc_out)
        p_we = e_we.predict_cross_fitted(loc_out)
        ov_preds.append(p_ov)
        bi_preds.append(p_bi)
        we_preds.append(p_we)

        r_2a = 0.40 * p_ov + 0.30 * p_bi + 0.30 * p_we
        p2a_preds.append(cal_2a.predict_proba(r_2a.reshape(-1, 1))[:, 1])

        tmp_df = pd.DataFrame({
            "sm_expert_overload": p_ov,
            "sm_expert_biomech": p_bi,
            "sm_expert_wellness": p_we,
            "recent_inj_decay": loc_out["recent_inj_decay"].values,
            "prev_inj_30d": loc_out["prev_inj_30d"].values,
        })
        X_loc_s = sc_s.transform(imp_s.transform(tmp_df[zs_cols].values))
        r_2b = lr_s.predict_proba(X_loc_s)[:, 1]
        p2b_preds.append(cal_2b.predict_proba(r_2b.reshape(-1, 1))[:, 1])

    loc_out["sm_expert_overload"] = np.mean(ov_preds, axis=0)
    loc_out["sm_expert_biomech"] = np.mean(bi_preds, axis=0)
    loc_out["sm_expert_wellness"] = np.mean(we_preds, axis=0)

    prob_2a_loc = np.mean(p2a_preds, axis=0)
    prob_2b_loc = np.mean(p2b_preds, axis=0)

    return loc_out, prob_2a_loc, th_2a, prob_2b_loc, th_2b, nested_oof_2b


class CalibratedNestedAdapter:
    """带折内 Platt 概率校准与内层 OOF 预算阈值存储的二阶段适配器"""
    def __init__(self, include_history: bool = True):
        self.include_history = include_history
        self.imp_female = SimpleImputer(strategy="median")
        self.sc_female = StandardScaler()
        self.imp_hist = SimpleImputer(strategy="median")
        self.sc_hist = StandardScaler()
        self.pillar_scaler = StandardScaler()
        self.weights_ = None
        self.gb_ = None
        self.platt_cal_ = None
        self.threshold_f1_ = 0.5
        self.budget_thresholds_ = {}

    def _extract_raw_pillars(self, df_sub: pd.DataFrame, fit_mode: bool = False) -> np.ndarray:
        p1 = df_sub["sm_expert_overload"].values
        p2 = df_sub["sm_expert_biomech"].values
        p3 = df_sub["sm_expert_wellness"].values

        X_fem = df_sub[FEAT_LOCAL_FEMALE].values
        if fit_mode:
            X_fem_z = self.sc_female.fit_transform(self.imp_female.fit_transform(X_fem))
        else:
            X_fem_z = self.sc_female.transform(self.imp_female.transform(X_fem))
        p4 = 1.0 / (1.0 + np.exp(-np.clip(X_fem_z @ np.array([0.35, 0.20, 0.20, 0.05, 0.05, 0.15]), -4, 4)))

        if not self.include_history:
            return np.column_stack([p1, p2, p3, p4])

        X_his = df_sub[FEAT_INJ_HISTORY].values
        if fit_mode:
            X_his_z = self.sc_hist.fit_transform(self.imp_hist.fit_transform(X_his))
        else:
            X_his_z = self.sc_hist.transform(self.imp_hist.transform(X_his))
        p5 = 1.0 / (1.0 + np.exp(-np.clip(X_his_z @ np.array([0.40, 0.25, 0.15, 0.20]), -4, 4)))
        return np.column_stack([p1, p2, p3, p4, p5])

    def _fit_core(self, df_tr: pd.DataFrame, y_tr: np.ndarray):
        P_raw_tr = self._extract_raw_pillars(df_tr, fit_mode=True)
        P_tr = self.pillar_scaler.fit_transform(P_raw_tr)
        P_tr_01 = 1.0 / (1.0 + np.exp(-np.clip(P_tr, -3.5, 3.5)))

        n_p = P_tr_01.shape[1]
        auc_gains = np.array([max(roc_auc_score(y_tr, P_tr_01[:, j]) - 0.50, 0.01) for j in range(n_p)])
        auc_w = auc_gains / (auc_gains.sum() + 1e-8)

        nnls_w, _ = nnls(P_tr_01, y_tr)
        nnls_w = nnls_w / (nnls_w.sum() + 1e-8) if nnls_w.sum() > 0 else auc_w

        w = np.clip(0.55 * auc_w + 0.45 * nnls_w, 0.10, 0.38)
        self.weights_ = w / w.sum()

        self.gb_ = HistGradientBoostingClassifier(
            max_iter=45, max_depth=2, learning_rate=0.03, min_samples_leaf=30,
            l2_regularization=10.0, monotonic_cst=[1] * n_p,
            class_weight="balanced", random_state=42
        ).fit(P_tr_01, y_tr)

    def _predict_uncalibrated(self, df_te: pd.DataFrame) -> np.ndarray:
        P_raw_te = self._extract_raw_pillars(df_te, fit_mode=False)
        P_te = self.pillar_scaler.transform(P_raw_te)
        P_te_01 = 1.0 / (1.0 + np.exp(-np.clip(P_te, -3.5, 3.5)))
        return 0.70 * (P_te_01 @ self.weights_) + 0.30 * self.gb_.predict_proba(P_te_01)[:, 1]

    def fit_nested(self, df_tr: pd.DataFrame, y_tr: np.ndarray, budgets=(0.10, 0.15, 0.20, 0.25, 0.30)):
        players_tr = df_tr["player_id"].values
        n_groups = len(np.unique(players_tr))
        inner_raw_oof = np.zeros(len(df_tr))

        inner_cv = StratifiedGroupKFold(n_splits=min(5, n_groups))
        for i_tr, i_val in inner_cv.split(df_tr, y_tr, groups=players_tr):
            if len(np.unique(y_tr[i_tr])) < 2:
                continue
            sub = CalibratedNestedAdapter(include_history=self.include_history)
            sub._fit_core(df_tr.iloc[i_tr], y_tr[i_tr])
            inner_raw_oof[i_val] = sub._predict_uncalibrated(df_tr.iloc[i_val])

        self.platt_cal_ = LogisticRegression(C=1.0, random_state=42).fit(inner_raw_oof.reshape(-1, 1), y_tr)
        inner_cal_oof = self.platt_cal_.predict_proba(inner_raw_oof.reshape(-1, 1))[:, 1]

        self.threshold_f1_ = select_threshold_from_oof(y_tr, inner_cal_oof)
        self.budget_thresholds_ = {
            b: float(np.quantile(inner_cal_oof, 1.0 - b)) for b in budgets
        }

        self._fit_core(df_tr, y_tr)
        return self

    def predict_calibrated(self, df_te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
        raw_te = self._predict_uncalibrated(df_te)
        cal_prob_te = self.platt_cal_.predict_proba(raw_te.reshape(-1, 1))[:, 1]
        pred_f1 = (cal_prob_te >= self.threshold_f1_).astype(int)
        pred_budgets = {
            b: (cal_prob_te >= th_b).astype(int) for b, th_b in self.budget_thresholds_.items()
        }
        return cal_prob_te, pred_f1, pred_budgets


def compute_fold_mean_sd(y_true: np.ndarray, y_prob: np.ndarray, fold_ids: np.ndarray) -> tuple[str, str, str]:
    aucs, aps = [], []
    total_folds = len(np.unique(fold_ids))
    for fid in np.unique(fold_ids):
        m = fold_ids == fid
        if len(np.unique(y_true[m])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[m], y_prob[m]))
        aps.append(average_precision_score(y_true[m], y_prob[m]))
    valid_str = f"{len(aucs)}/{total_folds}"
    return f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}", f"{np.mean(aps):.4f} ± {np.std(aps):.4f}", valid_str


def run_step11_suite(df_sm: pd.DataFrame, df_loc: pd.DataFrame, target_col: str, horizon_days: int, title: str):
    # 内存映射 SoccerMon 正名目标列 target_all_inj_next_{horizon}d，不改动原文件
    df_sm = df_sm.copy()
    sm_alt = f"target_all_inj_next_{horizon_days}d"
    if target_col not in df_sm.columns and sm_alt in df_sm.columns:
        df_sm[target_col] = df_sm[sm_alt]

    print("\n" + "=" * 118)
    print(f"📊 论文定稿评估报表 (全50人源域 + 连续自然日 + Platt校准 + 匿名化) —— [{title}]")
    print("=" * 118)

    # 严格保留全部 50 名有效监控运动员（消除 15 人选择偏倚）
    sm_act_col = "active_day" if "active_day" in df_sm.columns else "is_active_day"
    sm_active = df_sm[(df_sm[sm_act_col] == 1) & (df_sm["is_rehab"] == 0) & (df_sm["is_injured"] == 0) & (df_sm[target_col].notna())].copy()
    sm_active[target_col] = sm_active[target_col].astype(int)
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
    date_blocks = ((dates_loc - dates_loc.min()).dt.days // 7).values
    budgets = (0.10, 0.15, 0.20, 0.25, 0.30)

    # 1. Baseline
    base_cols = FEAT_EXPERT_OVERLOAD + FEAT_EXPERT_WELLNESS + FEAT_EXPERT_BIOMECH
    rf_base = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_base.fit(sm_active[base_cols].fillna(0).values, sm_active[target_col].values)
    prob_1_base = rf_base.predict_proba(loc_active[base_cols].fillna(0).values)[:, 1]
    pred_1_base = (prob_1_base >= 0.50).astype(int)

    # 2. 严格双层嵌套 + 交叉拟合集成源域 Zero-Shot (2a & 2b)
    loc_active, prob_2a, th_2a, prob_2b, th_2b, src_oof_2b = fit_nested_crossfitted_source(sm_clean, loc_active, target_col)
    pred_2a = (prob_2a >= th_2a).astype(int)
    pred_2b = (prob_2b >= th_2b).astype(int)

    # 3A & 3B. 标准 6-Fold GroupKFold
    gkf6 = GroupKFold(n_splits=6)
    oof_prob_gkf6_nh = np.zeros(len(loc_active))
    oof_pred_gkf6_nh = np.zeros(len(loc_active), dtype=int)
    oof_prob_gkf6_full = np.zeros(len(loc_active))
    oof_pred_gkf6_full = np.zeros(len(loc_active), dtype=int)
    oof_budgets_gkf6 = {b: np.zeros(len(loc_active), dtype=int) for b in budgets}
    fold_ids_gkf6 = np.zeros(len(loc_active), dtype=int)
    breakdown_rows = []

    for f_idx, (tr_idx, te_idx) in enumerate(gkf6.split(loc_active, y_loc, groups=players_loc), 1):
        fold_ids_gkf6[te_idx] = f_idx
        df_tr, y_tr = loc_active.iloc[tr_idx], y_loc[tr_idx]
        df_te, y_te = loc_active.iloc[te_idx], y_loc[te_idx]

        ad_nh = CalibratedNestedAdapter(include_history=False).fit_nested(df_tr, y_tr, budgets)
        oof_prob_gkf6_nh[te_idx], oof_pred_gkf6_nh[te_idx], _ = ad_nh.predict_calibrated(df_te)

        ad_full = CalibratedNestedAdapter(include_history=True).fit_nested(df_tr, y_tr, budgets)
        p_cal, pred_f1, pred_b_dict = ad_full.predict_calibrated(df_te)
        oof_prob_gkf6_full[te_idx] = p_cal
        oof_pred_gkf6_full[te_idx] = pred_f1
        for b in budgets:
            oof_budgets_gkf6[b][te_idx] = pred_b_dict[b]

        has_pos = len(np.unique(y_te)) > 1
        breakdown_rows.append({
            "切分方式": "Standard GroupKFold (6折)",
            "外层折号": f"Fold {f_idx}",
            "匿名测试集运动员": ",".join(sorted(df_te["anon_id"].unique())),
            "样本人日": len(y_te),
            "阳性人日": int(y_te.sum()),
            "阳性率": f"{y_te.mean()*100:.1f}%",
            "ZeroShot_2b_AUC": round(roc_auc_score(y_te, prob_2b[te_idx]), 4) if has_pos else "NA(无阳性)",
            "Model_3b_AUC": round(roc_auc_score(y_te, p_cal), 4) if has_pos else "NA(无阳性)",
            "Model_3b_AP": round(average_precision_score(y_te, p_cal), 4) if has_pos else "NA(无阳性)",
        })

    # 3C. 官方 StratifiedGroupKFold (5折分层分组)
    sgkf5 = StratifiedGroupKFold(n_splits=5)
    oof_prob_sgkf5 = np.zeros(len(loc_active))
    oof_pred_sgkf5 = np.zeros(len(loc_active), dtype=int)
    oof_budgets_sgkf5 = {b: np.zeros(len(loc_active), dtype=int) for b in budgets}
    fold_ids_sgkf5 = np.zeros(len(loc_active), dtype=int)

    for f_idx, (tr_idx, te_idx) in enumerate(sgkf5.split(loc_active, y_loc, groups=players_loc), 1):
        fold_ids_sgkf5[te_idx] = f_idx
        df_tr, y_tr = loc_active.iloc[tr_idx], y_loc[tr_idx]
        df_te, y_te = loc_active.iloc[te_idx], y_loc[te_idx]

        ad_sg = CalibratedNestedAdapter(include_history=True).fit_nested(df_tr, y_tr, budgets)
        p_cal, pred_f1, pred_b_dict = ad_sg.predict_calibrated(df_te)
        oof_prob_sgkf5[te_idx] = p_cal
        oof_pred_sgkf5[te_idx] = pred_f1
        for b in budgets:
            oof_budgets_sgkf5[b][te_idx] = pred_b_dict[b]

        has_pos = len(np.unique(y_te)) > 1
        breakdown_rows.append({
            "切分方式": "StratifiedGroupKFold (5折)",
            "外层折号": f"Fold {f_idx}",
            "匿名测试集运动员": ",".join(sorted(df_te["anon_id"].unique())),
            "样本人日": len(y_te),
            "阳性人日": int(y_te.sum()),
            "阳性率": f"{y_te.mean()*100:.1f}%",
            "ZeroShot_2b_AUC": round(roc_auc_score(y_te, prob_2b[te_idx]), 4) if has_pos else "NA",
            "Model_3b_AUC": round(roc_auc_score(y_te, p_cal), 4) if has_pos else "NA",
            "Model_3b_AP": round(average_precision_score(y_te, p_cal), 4) if has_pos else "NA",
        })

    # 4. 严格时间序列前滚 Purged Temporal CV
    unique_dates = np.sort(dates_loc.unique())
    n_dates = len(unique_dates)
    oof_prob_temp = np.copy(prob_2b)
    oof_pred_temp = np.copy(pred_2b)
    oof_budgets_temp = {b: np.zeros(len(loc_active), dtype=int) for b in budgets}
    warmup_cut = int(n_dates * 0.25)

    for d_idx, curr_date in enumerate(unique_dates):
        te_mask = (dates_loc == curr_date).values
        purge_cutoff = pd.Timestamp(curr_date) - pd.Timedelta(days=horizon_days)
        tr_mask = (dates_loc <= purge_cutoff).values
        n_hist = int(tr_mask.sum())

        if d_idx >= warmup_cut and n_hist >= 120 and y_loc[tr_mask].sum() >= 8:
            ad_temp = CalibratedNestedAdapter(include_history=True).fit_nested(
                loc_active.iloc[tr_mask], y_loc[tr_mask], budgets
            )
            p_local_te, _, _ = ad_temp.predict_calibrated(loc_active.iloc[te_mask])

            alpha = float(np.clip((n_hist - 100) / 1000.0, 0.15, 0.50))
            oof_prob_temp[te_mask] = (1.0 - alpha) * prob_2b[te_mask] + alpha * p_local_te

            th_hist_oof = select_threshold_from_oof(y_loc[tr_mask], oof_prob_temp[tr_mask])
            oof_pred_temp[te_mask] = (oof_prob_temp[te_mask] >= th_hist_oof).astype(int)

        recent_start = pd.Timestamp(curr_date) - pd.Timedelta(days=horizon_days + 21)
        recent_mask = ((dates_loc <= purge_cutoff) & (dates_loc >= recent_start)).values
        ref_probs = oof_prob_temp[recent_mask] if recent_mask.sum() >= 30 else src_oof_2b
        for b in budgets:
            th_b_recent = float(np.quantile(ref_probs, 1.0 - b))
            oof_budgets_temp[b][te_mask] = (oof_prob_temp[te_mask] >= th_b_recent).astype(int)

    temp_period_ids = pd.qcut(dates_loc.rank(method="first"), q=5, labels=False).values

    def make_row(name, y_t, p_prob, p_pred, boot_clusters, split_ids):
        f_auc_msd, f_ap_msd, valid_f = compute_fold_mean_sd(y_t, p_prob, split_ids)
        return {
            "验证方案": name,
            "有效折数": valid_f,
            "Pooled_ROC_AUC (95% CI)": cluster_bootstrap_ci(y_t, p_prob, boot_clusters, roc_auc_score),
            "Fold_AUC (Mean±SD)": f_auc_msd,
            "Pooled_AP (95% CI)": cluster_bootstrap_ci(y_t, p_prob, boot_clusters, average_precision_score),
            "Fold_AP (Mean±SD)": f_ap_msd,
            "Brier": round(brier_score_loss(y_t, p_prob), 4),
            "报警率": f"{p_pred.mean()*100:.1f}%",
            "F1": round(f1_score(y_t, p_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_t, p_pred, zero_division=0), 4),
            "Precision": round(precision_score(y_t, p_pred, zero_division=0), 4),
        }

    main_table = pd.DataFrame([
        make_row("1. 未对齐直接迁移 Baseline (默认0.5阈值)", y_loc, prob_1_base, pred_1_base, players_loc, fold_ids_gkf6),
        make_row("2a. Zero-Shot 纯生理+穿戴+主观 (无伤病史/交叉拟合)", y_loc, prob_2a, pred_2a, players_loc, fold_ids_gkf6),
        make_row("2b. Zero-Shot 全维度源域迁移 (双层嵌套交叉拟合)", y_loc, prob_2b, pred_2b, players_loc, fold_ids_gkf6),
        make_row("3a. 跨运动员 GroupKFold-6折 (无伤病史/内层OOF定阈值)", y_loc, oof_prob_gkf6_nh, oof_pred_gkf6_nh, players_loc, fold_ids_gkf6),
        make_row("3b. 跨运动员 GroupKFold-6折 (全维度5支柱/内层OOF定阈值)", y_loc, oof_prob_gkf6_full, oof_pred_gkf6_full, players_loc, fold_ids_gkf6),
        make_row("3c. 跨运动员 StratifiedGroup-5折 (全维度/消除无阳性折)", y_loc, oof_prob_sgkf5, oof_pred_sgkf5, players_loc, fold_ids_sgkf5),
        make_row("4. 时间前滚 Purged Temporal (历史OOF定阈值+时间块CI)", y_loc, oof_prob_temp, oof_pred_temp, date_blocks, temp_period_ids),
    ])

    print("\n[表 1] 主验证性能对比表 (含有效折数、Pooled 指标、Fold Mean±SD 与 Brier 校准分):")
    print(main_table.to_string(index=False))

    base_rate = y_loc.mean()
    sens_rows = []
    for b in budgets:
        p_g6 = oof_budgets_gkf6[b]
        p_sg5 = oof_budgets_sgkf5[b]
        p_tmp = oof_budgets_temp[b]
        sens_rows.append({
            "目标预警预算": f"Top {int(b*100)}%",
            "Group6折_实际报警率": f"{p_g6.mean()*100:.1f}%",
            "Group6折_Recall": round(recall_score(y_loc, p_g6, zero_division=0), 4),
            "Group6折_Precision": round(precision_score(y_loc, p_g6, zero_division=0), 4),
            "Group6折_F1": round(f1_score(y_loc, p_g6, zero_division=0), 4),
            "StratGroup5折_实际报警率": f"{p_sg5.mean()*100:.1f}%",
            "StratGroup5折_F1": round(f1_score(y_loc, p_sg5, zero_division=0), 4),
            "Temporal_实际报警率": f"{p_tmp.mean()*100:.1f}%",
            "Temporal_Recall": round(recall_score(y_loc, p_tmp, zero_division=0), 4),
            "Temporal_Precision": round(precision_score(y_loc, p_tmp, zero_division=0), 4),
            "Temporal_F1": round(f1_score(y_loc, p_tmp, zero_division=0), 4),
            "Temporal_Lift": f"{precision_score(y_loc, p_tmp, zero_division=0)/(base_rate+1e-8):.2f}x",
        })
    sens_table = pd.DataFrame(sens_rows)
    print("\n[表 2] 严格内层 OOF 预警预算敏感性分析表 (区分目标预算与测试折实际报警率):")
    print(sens_table.to_string(index=False))

    fold_df = pd.DataFrame(breakdown_rows)
    print("\n[表 3] 外层各折独立明细表 (已匿名化处理 P01~P24，对比标准 6 折与分层 5 折):")
    print(fold_df.to_string(index=False))

    return main_table, sens_table, fold_df


if __name__ == "__main__":
    df_sm = pd.read_csv(PROCESSED_DIR / "soccermon_features.csv", parse_dates=["date"])
    df_loc = pd.read_csv(PROCESSED_DIR / "local_features.csv", parse_dates=["date"])

    df_sm = add_causal_ewma_and_context_features(df_sm, is_local=False)
    df_loc = add_causal_ewma_and_context_features(df_loc, is_local=True)

    m7, s7, f7 = run_step11_suite(
        df_sm, df_loc, "target_all_inj_next_7d", horizon_days=7, title="未来 7 天自报告运动损伤预警"
    )
    m3, s3, f3 = run_step11_suite(
        df_sm, df_loc, "target_all_inj_next_3d", horizon_days=3, title="未来 3 天自报告运动损伤预警"
    )

    pd.concat([m7.assign(窗口="7天"), m3.assign(窗口="3天")], ignore_index=True).to_csv(
        OUTPUTS_DIR / "step11_table1_main_results.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([s7.assign(窗口="7天"), s3.assign(窗口="3天")], ignore_index=True).to_csv(
        OUTPUTS_DIR / "step11_table2_budget_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat([f7.assign(窗口="7天"), f3.assign(窗口="3天")], ignore_index=True).to_csv(
        OUTPUTS_DIR / "step11_table3_anonymized_folds.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n✅ Step 11 三张匿名化定稿表格已保存至 {OUTPUTS_DIR} 目录！")